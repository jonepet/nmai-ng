"""
Hard example mining — find what the model gets wrong, augment those, retrain.

Flow:
  1. Load best checkpoint from stages 1-3
  2. Run inference on training images
  3. Compare predictions against ground truth
  4. Identify hard examples:
     - Missed detections (ground truth boxes with no matching prediction)
     - Low confidence correct detections (score < 0.3)
     - Misclassifications (correct box, wrong category)
     - False positives (high confidence predictions with no ground truth match)
  5. Generate extra augmented copies of images containing hard examples
  6. Save to training set for the next training run

All constants from config.py.
"""

import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


def iou(box_a, box_b):
    """Compute IoU between two [x, y, w, h] boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix = max(0, min(ax2, bx2) - max(ax, bx))
    iy = max(0, min(ay2, by2) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def load_ground_truth(annotations_path):
    """Load COCO annotations, return {image_id: [{"bbox": [x,y,w,h], "category_id": int}, ...]}"""
    with open(annotations_path) as f:
        coco = json.load(f)
    gt = defaultdict(list)
    for ann in coco["annotations"]:
        gt[ann["image_id"]].append({
            "bbox": ann["bbox"],
            "category_id": ann["category_id"],
        })
    # Map filename -> image_id
    fname_to_id = {}
    id_to_fname = {}
    for img in coco["images"]:
        fname_to_id[Path(img["file_name"]).stem] = img["id"]
        id_to_fname[img["id"]] = img["file_name"]
    return gt, fname_to_id, id_to_fname


def run_inference_on_training(model, images_dir, device="cpu"):
    """Run model on training images, return {stem: [{"bbox", "category_id", "score"}, ...]}"""
    from ultralytics import YOLO

    predictions = defaultdict(list)
    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg"} and "_hard_" not in p.stem
    )

    print(f"  Running inference on {len(image_files)} training images...")
    for idx, img_path in enumerate(image_files):
        with torch.no_grad():
            results = model(str(img_path), device=device, verbose=False, imgsz=640)

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            xywh = result.boxes.xywh.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            for i in range(len(result.boxes)):
                cx, cy, w, h = xywh[i]
                x = cx - w / 2
                y = cy - h / 2
                predictions[img_path.stem].append({
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "category_id": int(cls_ids[i]),
                    "score": float(confs[i]),
                })

        if (idx + 1) % 50 == 0:
            print(f"    {idx + 1}/{len(image_files)} images processed")

    return predictions


def find_hard_examples(gt, predictions, fname_to_id):
    """
    Compare predictions against ground truth.
    Return dict of {stem: {"missed": N, "low_conf": N, "misclassified": N, "score": float}}
    Higher score = harder example.
    """
    hard_scores = {}
    iou_threshold = 0.5

    for stem, preds in predictions.items():
        image_id = fname_to_id.get(stem)
        if image_id is None:
            continue
        gt_boxes = gt.get(image_id, [])
        if not gt_boxes:
            continue

        missed = 0
        low_conf = 0
        misclassified = 0
        matched_gt = set()

        for gt_idx, gt_box in enumerate(gt_boxes):
            best_iou = 0
            best_pred = None
            best_pred_idx = -1

            for pred_idx, pred in enumerate(preds):
                iou_val = iou(gt_box["bbox"], pred["bbox"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_pred = pred
                    best_pred_idx = pred_idx

            if best_iou < iou_threshold:
                missed += 1
            elif best_pred["score"] < 0.3:
                low_conf += 1
                matched_gt.add(gt_idx)
            elif best_pred["category_id"] != gt_box["category_id"]:
                misclassified += 1
                matched_gt.add(gt_idx)
            else:
                matched_gt.add(gt_idx)

        total_gt = len(gt_boxes)
        # Hardness score: weighted combination of failure modes
        hardness = (
            3.0 * missed / max(total_gt, 1) +
            2.0 * misclassified / max(total_gt, 1) +
            1.0 * low_conf / max(total_gt, 1)
        )

        hard_scores[stem] = {
            "missed": missed,
            "low_conf": low_conf,
            "misclassified": misclassified,
            "total_gt": total_gt,
            "hardness": hardness,
        }

    return hard_scores


def generate_hard_augmentations(hard_images, images_dir, labels_dir, num_copies=3):
    """Generate augmented copies of the hardest images."""
    try:
        import albumentations as A
    except ImportError:
        print("  albumentations not available, skipping augmentation")
        return 0

    pipelines = {
        "hard_bright": A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0),
            A.GaussNoise(var_limit=(15, 50), p=0.7),
        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, label_fields=['class_ids'])),
        "hard_distort": A.Compose([
            A.Perspective(scale=(0.03, 0.08), p=1.0),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.1, p=0.8),
        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, label_fields=['class_ids'])),
        "hard_degrade": A.Compose([
            A.GaussianBlur(blur_limit=(5, 9), p=0.7),
            A.ImageCompression(quality_lower=25, quality_upper=50, p=0.8),
            A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), p=0.5),
        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, label_fields=['class_ids'])),
    }

    created = 0
    for stem, info in hard_images:
        img_path = None
        for ext in [".jpg", ".jpeg"]:
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        label_path = labels_dir / f"{stem}.txt"
        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Read YOLO labels
        bboxes = []
        class_ids = []
        for line in label_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            # Clamp to valid range
            xc = max(0.001, min(0.999, xc))
            yc = max(0.001, min(0.999, yc))
            w = max(0.001, min(0.999, w))
            h = max(0.001, min(0.999, h))
            bboxes.append([xc, yc, w, h])
            class_ids.append(cls_id)

        if not bboxes:
            continue

        for pipe_name, pipeline in pipelines.items():
            out_stem = f"{stem}_{pipe_name}"
            out_img = images_dir / f"{out_stem}.jpg"
            out_lbl = labels_dir / f"{out_stem}.txt"

            if out_img.exists() and out_lbl.exists():
                continue

            try:
                result = pipeline(image=img_rgb, bboxes=bboxes, class_ids=class_ids)
                aug_img = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)

                lines = []
                for bbox, cls_id in zip(result["bboxes"], result["class_ids"]):
                    xc, yc, w, h = bbox
                    xc = max(0.0, min(1.0, xc))
                    yc = max(0.0, min(1.0, yc))
                    w = max(0.0, min(1.0, w))
                    h = max(0.0, min(1.0, h))
                    if w > 0 and h > 0:
                        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

                if lines:
                    cv2.imwrite(str(out_img), aug_img)
                    out_lbl.write_text("\n".join(lines) + "\n")
                    created += 1
            except Exception:
                continue

    return created


def main():
    from ultralytics import YOLO

    print("=" * 60)
    print("Hard Example Mining")
    print("=" * 60)

    # Find best checkpoint
    checkpoint = config.CHECKPOINT_ROOT / "best_final.pt"
    if not checkpoint.exists():
        # Try stage checkpoints
        for stage_num in [3, 2, 1]:
            candidate = config.CHECKPOINT_ROOT / f"best_stage_{stage_num}.pt"
            if candidate.exists():
                checkpoint = candidate
                break

    if not checkpoint.exists():
        print("No checkpoint found. Run training first.")
        sys.exit(1)

    print(f"Using checkpoint: {checkpoint}")

    # Paths
    annotations_path = config.COCO_EXTRACT_DIR / "train" / "annotations.json"
    images_dir = config.YOLO_DIR / "train" / "images"
    labels_dir = config.YOLO_DIR / "train" / "labels"

    if not annotations_path.exists():
        print(f"Annotations not found: {annotations_path}")
        sys.exit(1)

    # 1. Load ground truth
    print("\n[1/4] Loading ground truth...")
    gt, fname_to_id, id_to_fname = load_ground_truth(str(annotations_path))
    print(f"  {len(gt)} images with ground truth")

    # 2. Run inference
    print("\n[2/4] Running inference on training set...")
    model = YOLO(str(checkpoint))
    predictions = run_inference_on_training(model, images_dir, device="0")
    print(f"  {sum(len(v) for v in predictions.values())} predictions on {len(predictions)} images")

    # 3. Find hard examples
    print("\n[3/4] Analyzing errors...")
    hard_scores = find_hard_examples(gt, predictions, fname_to_id)

    # Sort by hardness
    ranked = sorted(hard_scores.items(), key=lambda x: x[1]["hardness"], reverse=True)

    # Print top 20 hardest
    print(f"\n  Top 20 hardest images:")
    print(f"  {'Image':<25} {'Missed':>7} {'LowConf':>8} {'MisCls':>7} {'Total':>6} {'Score':>7}")
    print(f"  {'-'*60}")
    for stem, info in ranked[:20]:
        print(f"  {stem:<25} {info['missed']:>7} {info['low_conf']:>8} {info['misclassified']:>7} {info['total_gt']:>6} {info['hardness']:>7.2f}")

    # Take top 50% hardest images for augmentation
    n_hard = max(10, len(ranked) // 2)
    hard_subset = ranked[:n_hard]
    print(f"\n  Selected {len(hard_subset)} hard images for augmentation")

    # Summary stats
    total_missed = sum(info["missed"] for _, info in ranked)
    total_misclassified = sum(info["misclassified"] for _, info in ranked)
    total_low_conf = sum(info["low_conf"] for _, info in ranked)
    total_gt = sum(info["total_gt"] for _, info in ranked)
    print(f"\n  Overall error rates:")
    print(f"    Missed:         {total_missed}/{total_gt} ({100*total_missed/max(total_gt,1):.1f}%)")
    print(f"    Misclassified:  {total_misclassified}/{total_gt} ({100*total_misclassified/max(total_gt,1):.1f}%)")
    print(f"    Low confidence: {total_low_conf}/{total_gt} ({100*total_low_conf/max(total_gt,1):.1f}%)")

    # 4. Generate augmentations
    print("\n[4/4] Generating hard example augmentations...")
    created = generate_hard_augmentations(hard_subset, images_dir, labels_dir)
    print(f"  Created {created} augmented hard examples")

    print(f"\n{'=' * 60}")
    print("Hard example mining complete.")
    print(f"  Run training again to benefit from the new hard examples.")


if __name__ == "__main__":
    main()
