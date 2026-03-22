"""
Evaluate trained YOLOv8 checkpoints against the validation set using the
competition scoring formula:

    score = 0.7 * detection_mAP@0.5 + 0.3 * classification_mAP@0.5

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_stage_1.pt
    python scripts/evaluate.py --all
    python scripts/evaluate.py --watch
"""

import argparse
import copy
import io
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# PyTorch 2.6 compat: ultralytics 8.1.0 needs weights_only=False
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# ---------------------------------------------------------------------------
# Baselines for context
# ---------------------------------------------------------------------------

BASELINE_RANDOM = 0.01
BASELINE_PRETRAINED_COCO = 0.30
BASELINE_FINETUNED = 0.60

EVAL_RESULTS_PATH = config.CHECKPOINT_ROOT / "eval_results.json"

# ---------------------------------------------------------------------------
# Image helpers (mirrors run.py)
# ---------------------------------------------------------------------------


def extract_image_id(stem: str) -> int:
    digits = re.findall(r"\d+", stem)
    if digits:
        return int(digits[-1])
    return abs(hash(stem)) % (10 ** 9)


def collect_val_images() -> list[Path]:
    val_dir = config.YOLO_DIR / "val" / "images"
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Validation images directory not found: {val_dir}")
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
        images.extend(val_dir.glob(ext))
    images.sort(key=lambda p: extract_image_id(p.stem))
    return images


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _load_classifier():
    """Load classifier ONNX if it exists."""
    import json as _json
    cfg_path = config.SUBMISSION_DIR / "config.json"
    if not cfg_path.exists():
        return None, 0, 0.0

    cfg = _json.load(open(cfg_path))
    cls_file = cfg.get("classifier_file")
    cls_input_size = cfg.get("classifier_input_size", 128)
    cls_conf_thresh = cfg.get("classifier_confidence_threshold", 0.3)

    if not cls_file:
        return None, cls_input_size, cls_conf_thresh

    cls_path = config.SUBMISSION_DIR / cls_file
    if not cls_path.exists():
        cls_path = config.CHECKPOINT_ROOT / cls_file
    if not cls_path.exists():
        print(f"  Classifier not found, skipping reclassification")
        return None, cls_input_size, cls_conf_thresh

    import onnxruntime as ort
    print(f"  Loading classifier: {cls_path.name}")
    sess = ort.InferenceSession(str(cls_path), providers=["CPUExecutionProvider"])
    return sess, cls_input_size, cls_conf_thresh


def _classify_crop(crop_img, classifier, input_size):
    """Classify a single crop image, return (category_id, confidence)."""
    import numpy as np
    from PIL import Image

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    resized = crop_img.resize((input_size, input_size), Image.BILINEAR)
    arr = np.array(resized, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = (arr - mean) / std
    arr = np.expand_dims(arr, 0)

    input_name = classifier.get_inputs()[0].name
    logits = classifier.run(None, {input_name: arr})[0][0]
    e = np.exp(logits - np.max(logits))
    probs = e / e.sum()
    cls_id = int(np.argmax(probs))
    return cls_id, float(probs[cls_id])


def run_inference(checkpoint: Path, image_paths: list[Path]) -> list[dict]:
    """Load checkpoint and run inference on image_paths, returning COCO predictions."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("ultralytics is required. Install with: pip install ultralytics")
    from PIL import Image

    print(f"  Loading model: {checkpoint}")
    model = YOLO(str(checkpoint))

    classifier, cls_input_size, cls_conf_thresh = _load_classifier()

    predictions: list[dict] = []
    total = len(image_paths)

    for idx, image_path in enumerate(image_paths, 1):
        if idx % 20 == 0 or idx == total:
            print(f"  Inference: {idx}/{total}", end="\r", flush=True)

        image_id = extract_image_id(image_path.stem)

        results = model(str(image_path), device="cpu", verbose=False)

        img_pil = None
        if classifier:
            img_pil = Image.open(image_path).convert("RGB")

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xywh = boxes.xywh.cpu()
            xyxy = boxes.xyxy.cpu()
            scores = boxes.conf.cpu()
            class_ids = boxes.cls.cpu().int()

            for i in range(len(boxes)):
                x_center, y_center, w, h = xywh[i].tolist()
                x = x_center - w / 2.0
                y = y_center - h / 2.0
                score = float(scores[i].item())
                cat_id = int(class_ids[i].item())

                # Reclassify with classifier if score is low
                if classifier and img_pil and score < cls_conf_thresh:
                    x1, y1, x2, y2 = xyxy[i].tolist()
                    if (x2 - x1) > 2 and (y2 - y1) > 2:
                        crop = img_pil.crop((int(x1), int(y1), int(x2), int(y2)))
                        new_id, conf = _classify_crop(crop, classifier, cls_input_size)
                        if conf > 0.5:
                            cat_id = new_id

                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": cat_id,
                        "bbox": [
                            round(x, 4),
                            round(y, 4),
                            round(w, 4),
                            round(h, 4),
                        ],
                        "score": round(score, 6),
                    }
                )

    print()  # newline after \r progress
    predictions.sort(key=lambda p: p["image_id"])
    return predictions


# ---------------------------------------------------------------------------
# COCO scoring helpers (adapted from scorer.py)
# ---------------------------------------------------------------------------


def _load_coco_gt() -> dict:
    annotations_path = config.COCO_EXTRACT_DIR / "train" / "annotations.json"
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations not found: {annotations_path}")
    with open(annotations_path) as f:
        return json.load(f)


def _filter_to_val_images(coco_gt: dict, val_image_names: set[str]) -> dict:
    """Restrict ground truth to the validation image filenames."""
    kept_images = [
        img for img in coco_gt["images"]
        if Path(img["file_name"]).name in val_image_names
    ]
    kept_ids = {img["id"] for img in kept_images}
    kept_annotations = [a for a in coco_gt["annotations"] if a["image_id"] in kept_ids]

    filtered = copy.deepcopy(coco_gt)
    filtered["images"] = kept_images
    filtered["annotations"] = kept_annotations
    return filtered


def _make_detection_gt(coco_gt: dict) -> dict:
    """Collapse all categories to a single 'object' category for detection eval."""
    det_gt = copy.deepcopy(coco_gt)
    det_gt["categories"] = [{"id": 1, "name": "object", "supercategory": "object"}]
    for ann in det_gt["annotations"]:
        ann["category_id"] = 1
    return det_gt


def _run_coco_eval(gt_dict: dict, dt_list: list[dict]) -> float:
    """Run COCOeval and return mAP@0.5 (suppresses stdout noise)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not dt_list:
        return 0.0

    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dt_list)
    finally:
        sys.stdout = old_stdout

    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.iouThrs = [0.5]
    evaluator.evaluate()
    evaluator.accumulate()

    sys.stdout = io.StringIO()
    try:
        evaluator.summarize()
    finally:
        sys.stdout = old_stdout

    return max(0.0, float(evaluator.stats[0]))


def _per_category_ap(coco_gt_dict: dict, dt_list: list[dict]) -> dict[int, float]:
    """
    Compute per-category AP@0.5 using COCOeval's per-category precision array.
    Returns {category_id: ap} for all categories that appear in gt.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import numpy as np

    if not dt_list:
        return {}

    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dt_list)
    finally:
        sys.stdout = old_stdout

    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.iouThrs = [0.5]
    evaluator.evaluate()
    evaluator.accumulate()

    # precision shape: [T, R, K, A, M]
    #   T = iou thresholds, R = recall thresholds, K = categories,
    #   A = area ranges, M = max dets
    precision = evaluator.eval["precision"]  # shape (T, R, K, A, M)
    cat_ids = evaluator.params.catIds

    per_cat: dict[int, float] = {}
    for k, cat_id in enumerate(cat_ids):
        # IoU=0.5 only (index 0), area=all (index 0), maxDets=100 (index -1)
        prec = precision[0, :, k, 0, -1]
        valid = prec[prec >= 0]
        per_cat[cat_id] = float(np.mean(valid)) if len(valid) > 0 else 0.0

    return per_cat


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------


def evaluate_checkpoint(checkpoint: Path, val_images: list[Path], coco_gt_full: dict) -> dict:
    """
    Run full evaluation for one checkpoint.

    Returns a result dict with scores, per-category breakdown, and metadata.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {checkpoint.name}")
    print(f"{'='*60}")

    val_image_names = {p.name for p in val_images}
    coco_gt = _filter_to_val_images(coco_gt_full, val_image_names)

    if not coco_gt["images"]:
        raise RuntimeError(
            "No ground-truth images matched the validation set filenames. "
            "Check that annotations.json and val/images/ are aligned."
        )

    gt_image_ids = {img["id"] for img in coco_gt["images"]}
    print(f"  Val images matched in GT: {len(coco_gt['images'])}")

    # Run inference
    t0 = time.time()
    raw_predictions = run_inference(checkpoint, val_images)
    inference_time = time.time() - t0
    print(f"  Inference time: {inference_time:.1f}s  |  Predictions: {len(raw_predictions)}")

    # Remap image_ids from filename-based IDs to GT annotation IDs
    # The val images are derived from the training split, so we need to match
    # filenames precisely. Build a stem->gt_id mapping.
    stem_to_gt_id: dict[str, int] = {}
    for img in coco_gt["images"]:
        name = Path(img["file_name"]).name
        stem = Path(name).stem
        stem_to_gt_id[name] = img["id"]
        stem_to_gt_id[stem] = img["id"]

    # Also map extracted numeric IDs from the val filenames back to GT ids
    val_extract_to_gt: dict[int, int] = {}
    for img_path in val_images:
        extracted = extract_image_id(img_path.stem)
        gt_id = stem_to_gt_id.get(img_path.name) or stem_to_gt_id.get(img_path.stem)
        if gt_id is not None:
            val_extract_to_gt[extracted] = gt_id

    # Remap predictions
    predictions: list[dict] = []
    for p in raw_predictions:
        gt_id = val_extract_to_gt.get(p["image_id"])
        if gt_id is None:
            continue
        predictions.append(dict(p, image_id=gt_id))

    print(f"  Predictions after ID remapping: {len(predictions)}")

    if not predictions:
        print("  WARNING: No predictions could be mapped to GT image IDs.")
        return {
            "checkpoint": str(checkpoint),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "score": 0.0,
            "detection_map": 0.0,
            "classification_map": 0.0,
            "num_val_images": len(val_images),
            "num_predictions": 0,
            "inference_time_s": round(inference_time, 2),
            "per_category_ap": {},
        }

    # Detection mAP (categories collapsed)
    print("  Computing detection mAP...")
    det_gt = _make_detection_gt(coco_gt)
    det_preds = [dict(p, category_id=1) for p in predictions]
    detection_map = _run_coco_eval(det_gt, det_preds)

    # Classification mAP (correct category required)
    print("  Computing classification mAP...")
    classification_map = _run_coco_eval(coco_gt, predictions)

    # Per-category AP
    print("  Computing per-category AP...")
    per_cat_ap = _per_category_ap(coco_gt, predictions)

    # Combined score
    score = round(0.7 * detection_map + 0.3 * classification_map, 6)

    result = {
        "checkpoint": str(checkpoint),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "score": score,
        "detection_map": round(detection_map, 6),
        "classification_map": round(classification_map, 6),
        "num_val_images": len(val_images),
        "num_predictions": len(predictions),
        "inference_time_s": round(inference_time, 2),
        "per_category_ap": {str(k): round(v, 6) for k, v in per_cat_ap.items()},
    }
    return result


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------


def print_report(result: dict, coco_gt_full: dict) -> None:
    det = result["detection_map"]
    cls = result["classification_map"]
    score = result["score"]

    print()
    print(f"  Checkpoint : {Path(result['checkpoint']).name}")
    print(f"  Evaluated  : {result['timestamp']}")
    print(f"  Val images : {result['num_val_images']}")
    print(f"  Predictions: {result['num_predictions']}")
    print(f"  Inference  : {result['inference_time_s']}s")
    print()
    print(f"  {'Metric':<28} {'Score':>8}  {'Target':>8}")
    print(f"  {'-'*48}")
    print(f"  {'Detection mAP@0.5 (70%)':<28} {det:>8.4f}  {'1.0':>8}")
    print(f"  {'Classification mAP@0.5 (30%)':<28} {cls:>8.4f}  {'1.0':>8}")
    print(f"  {'Combined score':<28} {score:>8.4f}  {'1.0':>8}")
    print()
    print(f"  Detection:      {det:.4f} / 1.0")
    print(f"  Classification: {cls:.4f} / 1.0")
    print(f"  Combined:       {score:.4f} / 1.0")
    print()

    # Baseline comparison
    def _bracket(s: float) -> str:
        if s < BASELINE_RANDOM + 0.02:
            return "below random baseline"
        if s < BASELINE_PRETRAINED_COCO - 0.05:
            return "below pretrained-COCO baseline"
        if s < BASELINE_FINETUNED - 0.05:
            return "below fine-tuned baseline"
        if s < BASELINE_FINETUNED:
            return "approaching fine-tuned baseline"
        return "at or above fine-tuned baseline"

    print(f"  Baselines: random (~{BASELINE_RANDOM}), pretrained-COCO (~{BASELINE_PRETRAINED_COCO}), fine-tuned (~{BASELINE_FINETUNED}+)")
    print(f"  Combined score assessment: {_bracket(score)}")

    # Per-category breakdown
    per_cat = result.get("per_category_ap", {})
    if per_cat:
        # Build category id -> name map
        cat_id_to_name: dict[int, str] = {}
        for cat in coco_gt_full.get("categories", []):
            cat_id_to_name[cat["id"]] = cat.get("name", str(cat["id"]))

        sorted_cats = sorted(per_cat.items(), key=lambda kv: kv[1], reverse=True)
        int_sorted = [(int(k), v) for k, v in sorted_cats]

        top10 = int_sorted[:10]
        worst10 = int_sorted[-10:]

        print()
        print(f"  {'Top 10 categories (best AP)'}")
        print(f"  {'ID':<6} {'Name':<35} {'AP@0.5':>8}")
        print(f"  {'-'*52}")
        for cat_id, ap in top10:
            name = cat_id_to_name.get(cat_id, str(cat_id))[:34]
            print(f"  {cat_id:<6} {name:<35} {ap:>8.4f}")

        print()
        print(f"  {'Bottom 10 categories (worst AP)'}")
        print(f"  {'ID':<6} {'Name':<35} {'AP@0.5':>8}")
        print(f"  {'-'*52}")
        for cat_id, ap in reversed(worst10):
            name = cat_id_to_name.get(cat_id, str(cat_id))[:34]
            print(f"  {cat_id:<6} {name:<35} {ap:>8.4f}")

    print()


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------


def load_eval_results() -> list[dict]:
    if EVAL_RESULTS_PATH.exists():
        with open(EVAL_RESULTS_PATH) as f:
            return json.load(f)
    return []


def save_eval_results(results: list[dict]) -> None:
    EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


def already_evaluated(checkpoint: Path, history: list[dict]) -> bool:
    checkpoint_str = str(checkpoint)
    return any(r["checkpoint"] == checkpoint_str for r in history)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def mode_one_shot(checkpoint: Path, max_images: int = 0) -> None:
    checkpoint = checkpoint.resolve()
    if not checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint}", file=sys.stderr)
        sys.exit(1)

    val_images = collect_val_images()
    if max_images > 0:
        val_images = val_images[:max_images]
    print(f"Validation images: {len(val_images)}")

    coco_gt_full = _load_coco_gt()
    print(f"Ground-truth loaded: {len(coco_gt_full['images'])} images, "
          f"{len(coco_gt_full['annotations'])} annotations")

    result = evaluate_checkpoint(checkpoint, val_images, coco_gt_full)
    print_report(result, coco_gt_full)

    history = load_eval_results()
    # Replace existing entry for same checkpoint or append
    history = [r for r in history if r["checkpoint"] != result["checkpoint"]]
    history.append(result)
    history.sort(key=lambda r: r["timestamp"])
    save_eval_results(history)
    print(f"Results saved to: {EVAL_RESULTS_PATH}")


def mode_all() -> None:
    checkpoint_dir = config.CHECKPOINT_ROOT
    if not checkpoint_dir.is_dir():
        print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}", file=sys.stderr)
        sys.exit(1)

    checkpoints = sorted(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        print(f"No .pt files found in {checkpoint_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoint(s) in {checkpoint_dir}")

    val_images = collect_val_images()
    print(f"Validation images: {len(val_images)}")

    coco_gt_full = _load_coco_gt()
    print(f"Ground-truth loaded: {len(coco_gt_full['images'])} images, "
          f"{len(coco_gt_full['annotations'])} annotations")

    history = load_eval_results()
    evaluated_count = 0

    for checkpoint in checkpoints:
        result = evaluate_checkpoint(checkpoint, val_images, coco_gt_full)
        print_report(result, coco_gt_full)

        history = [r for r in history if r["checkpoint"] != result["checkpoint"]]
        history.append(result)
        history.sort(key=lambda r: r["timestamp"])
        save_eval_results(history)
        evaluated_count += 1

    print(f"\nEvaluated {evaluated_count} checkpoint(s). Results saved to: {EVAL_RESULTS_PATH}")

    # Summary table
    if len(history) > 1:
        print("\nSummary (all evaluated checkpoints, sorted by combined score):")
        ranked = sorted(history, key=lambda r: r["score"], reverse=True)
        print(f"  {'Checkpoint':<40} {'Det':>7} {'Cls':>7} {'Score':>7}")
        print(f"  {'-'*64}")
        for r in ranked:
            name = Path(r["checkpoint"]).name[:39]
            print(f"  {name:<40} {r['detection_map']:>7.4f} {r['classification_map']:>7.4f} {r['score']:>7.4f}")


def _update_best_final(checkpoint_dir: Path) -> None:
    """Copy the most recently modified best.pt from any stage to best_final.pt."""
    import shutil
    best_candidates = sorted(
        checkpoint_dir.glob("*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not best_candidates:
        return

    latest = best_candidates[0]
    dest = checkpoint_dir / "best_final.pt"

    # Only copy if source is newer
    if dest.exists() and dest.stat().st_mtime >= latest.stat().st_mtime:
        return

    shutil.copy2(latest, dest)
    stage_name = latest.parent.parent.name
    print(f"  [{_now()}] Updated best_final.pt from {stage_name} ({latest.stat().st_size / 1024 / 1024:.1f} MB)")


def mode_watch(poll_interval: int = 60) -> None:
    checkpoint_dir = config.CHECKPOINT_ROOT
    print(f"Watching {checkpoint_dir} for new .pt files (polling every {poll_interval}s)")
    print("Press Ctrl+C to stop.\n")

    val_images: list[Path] | None = None
    coco_gt_full: dict | None = None

    try:
        while True:
            history = load_eval_results()
            evaluated_paths = {r["checkpoint"] for r in history}

            if not checkpoint_dir.is_dir():
                print(f"  [{_now()}] Checkpoint directory not found yet: {checkpoint_dir}")
                time.sleep(poll_interval)
                continue

            # Find all best.pt and epochN.pt files in stage subdirs
            found = sorted(p for p in checkpoint_dir.glob("*.pt")
                           if not p.name.startswith("classifier"))
            found += sorted(checkpoint_dir.glob("*/weights/best.pt"))
            new_checkpoints = [c for c in found if str(c.resolve()) not in evaluated_paths]

            # Always update best_final.pt from the latest stage's best.pt
            _update_best_final(checkpoint_dir)

            if new_checkpoints:
                # Lazy-load val data once
                if val_images is None:
                    val_images = collect_val_images()
                    print(f"  Validation images: {len(val_images)}")
                if coco_gt_full is None:
                    coco_gt_full = _load_coco_gt()
                    print(f"  Ground-truth: {len(coco_gt_full['images'])} images loaded")

                for checkpoint in new_checkpoints:
                    checkpoint = checkpoint.resolve()
                    result = evaluate_checkpoint(checkpoint, val_images, coco_gt_full)
                    print_report(result, coco_gt_full)

                    # Reload history to avoid overwrites if multiple evals run
                    history = load_eval_results()
                    history = [r for r in history if r["checkpoint"] != result["checkpoint"]]
                    history.append(result)
                    history.sort(key=lambda r: r["timestamp"])
                    save_eval_results(history)
                    print(f"  Results appended to: {EVAL_RESULTS_PATH}")
            else:
                print(f"  [{_now()}] No new checkpoints. Next check in {poll_interval}s...")

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\nWatch mode stopped.")


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv8 checkpoints using the competition scoring formula.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate.py --checkpoint checkpoints/best_stage_1.pt
  python scripts/evaluate.py --all
  python scripts/evaluate.py --watch
  python scripts/evaluate.py --watch --interval 30
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--checkpoint",
        type=Path,
        metavar="PATH",
        help="Path to a specific .pt checkpoint to evaluate.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all .pt files in the checkpoints directory.",
    )
    group.add_argument(
        "--watch",
        action="store_true",
        help="Continuously watch for new checkpoints and evaluate them.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        metavar="SECONDS",
        help="Polling interval for --watch mode (default: 60).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        metavar="N",
        help="Limit evaluation to N val images (0 = all). For quick scoring.",
    )

    args = parser.parse_args()

    if args.checkpoint:
        mode_one_shot(args.checkpoint, max_images=args.max_images)
    elif args.all:
        mode_all()
    elif args.watch:
        mode_watch(poll_interval=args.interval)


if __name__ == "__main__":
    main()
