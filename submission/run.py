# NOTE: This file is self-contained for sandbox deployment.
# Uses ONNX format (recommended by competition docs) — no pickle, no version issues.
"""
NorgesGruppen grocery product detection — ONNX inference.
Two-stage: YOLO detects boxes, classifier identifies products.

Sandbox: Python 3.11, onnxruntime-gpu 1.20.0, numpy 1.26.4, Pillow 10.2.0,
         ensemble-boxes 1.0.9, NVIDIA L4 GPU (24GB VRAM), 300s timeout.

Usage:
    python run.py --input /path/to/images --output /path/to/predictions.json
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort
from ensemble_boxes import weighted_boxes_fusion, nms
from PIL import Image


# ---------------------------------------------------------------------------
# Configuration — loaded from config.json (single source of truth)
# ---------------------------------------------------------------------------

_cfg_path = Path(__file__).parent / "config.json"
with open(_cfg_path, encoding="utf-8") as _f:
    _cfg = json.load(_f)

MODEL_FILES = _cfg["model_files"]
IMGSZ = _cfg["imgsz"]
CONF_THRESHOLD = _cfg["conf_threshold"]
WBF_IOU_THRESHOLD = _cfg["wbf_iou_threshold"]
WBF_SCORE_THRESHOLD = _cfg["wbf_score_threshold"]
CLASSIFIER_FILE = _cfg.get("classifier_file")
CLASSIFIER_INPUT_SIZE = _cfg.get("classifier_input_size", 128)
CLASSIFIER_CONF_THRESHOLD = _cfg.get("classifier_confidence_threshold", 0.3)

# ImageNet normalization for classifier
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_image_id(stem: str) -> int:
    digits = re.findall(r"\d+", stem)
    if digits:
        return int(digits[-1])
    return abs(hash(stem)) % (10 ** 9)


def collect_images(input_dir: Path) -> list[Path]:
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png"):
        images.extend(input_dir.glob(ext))
    images.sort(key=lambda p: extract_image_id(p.stem))
    return images


# ---------------------------------------------------------------------------
# ONNX model loading
# ---------------------------------------------------------------------------

def load_onnx_sessions(script_dir: Path) -> list[ort.InferenceSession]:
    available = ort.get_available_providers()
    providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available]
    print(f"ONNX providers: {providers}")
    sessions = []
    for name in MODEL_FILES:
        path = script_dir / name
        if path.exists():
            print(f"Loading ONNX model: {path.name}")
            sess = ort.InferenceSession(str(path), providers=providers)
            sessions.append(sess)
    if not sessions:
        raise FileNotFoundError(
            f"No ONNX models found in {script_dir}. Expected: {MODEL_FILES}"
        )
    print(f"Loaded {len(sessions)} detector(s), providers: {sessions[0].get_providers()}")
    return sessions


def load_classifier(script_dir: Path) -> ort.InferenceSession | None:
    if not CLASSIFIER_FILE:
        return None
    path = script_dir / CLASSIFIER_FILE
    if not path.exists():
        print(f"Classifier not found: {path.name}, using YOLO classes only")
        return None
    available = ort.get_available_providers()
    cls_providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available]
    print(f"Loading classifier: {path.name}")
    sess = ort.InferenceSession(str(path), providers=cls_providers)
    print(f"Classifier loaded, providers: {sess.get_providers()}")
    return sess


# ---------------------------------------------------------------------------
# Preprocessing — letterbox resize matching ultralytics
# ---------------------------------------------------------------------------

def preprocess(img: Image.Image, imgsz: int) -> tuple[np.ndarray, float, int, int, int, int]:
    orig_w, orig_h = img.size
    ratio = min(imgsz / orig_w, imgsz / orig_h)
    new_w = int(round(orig_w * ratio))
    new_h = int(round(orig_h * ratio))

    resized = img.resize((new_w, new_h), Image.BILINEAR)

    pad_w = (imgsz - new_w) // 2
    pad_h = (imgsz - new_h) // 2

    padded = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
    padded.paste(resized, (pad_w, pad_h))

    arr = np.array(padded, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    arr = np.expand_dims(arr, 0)  # add batch dim

    return arr, ratio, pad_w, pad_h, orig_w, orig_h


def preprocess_crop(crop: Image.Image, size: int) -> np.ndarray:
    """Preprocess a crop for the classifier (ImageNet normalization)."""
    resized = crop.resize((size, size), Image.BILINEAR)
    arr = np.array(resized, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr


# ---------------------------------------------------------------------------
# Postprocessing — YOLOv8 ONNX output
# ---------------------------------------------------------------------------

def postprocess(output: np.ndarray, ratio: float, pad_w: int, pad_h: int,
                orig_w: int, orig_h: int, conf_threshold: float) -> list[dict]:
    preds = output[0].T  # (num_boxes, 4+nc)

    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(len(class_ids)), class_ids]

    mask = confidences > conf_threshold
    boxes_xywh = boxes_xywh[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    results = []
    for i in range(len(boxes_xywh)):
        cx, cy, w, h = boxes_xywh[i]

        x1 = (cx - w / 2 - pad_w) / ratio
        y1 = (cy - h / 2 - pad_h) / ratio
        bw = w / ratio
        bh = h / ratio

        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        bw = min(bw, orig_w - x1)
        bh = min(bh, orig_h - y1)

        if bw > 0 and bh > 0:
            results.append({
                "bbox": [round(float(x1), 1), round(float(y1), 1),
                         round(float(bw), 1), round(float(bh), 1)],
                "category_id": int(class_ids[i]),
                "score": round(float(confidences[i]), 3),
            })

    return results


# ---------------------------------------------------------------------------
# Classifier reclassification
# ---------------------------------------------------------------------------

def reclassify_detections(
    detections: list[dict],
    img: Image.Image,
    classifier: ort.InferenceSession,
) -> list[dict]:
    """Reclassify ALL detections using the product classifier for better category accuracy."""
    if not detections:
        return detections

    indices_to_classify = []
    crops = []
    for i, det in enumerate(detections):
        x, y, w, h = det["bbox"]
        if w > 2 and h > 2:  # skip tiny crops
            crop = img.crop((int(x), int(y), int(x + w), int(y + h)))
            crops.append(preprocess_crop(crop, CLASSIFIER_INPUT_SIZE))
            indices_to_classify.append(i)

    if not crops:
        return detections

    # Batch classify in chunks to limit memory
    input_name = classifier.get_inputs()[0].name
    BATCH = 64
    all_logits = []
    for start in range(0, len(crops), BATCH):
        batch = np.stack(crops[start:start + BATCH], axis=0)
        logits = classifier.run(None, {input_name: batch})[0]
        all_logits.append(logits)
    all_logits = np.concatenate(all_logits, axis=0)

    for j, idx in enumerate(indices_to_classify):
        probs = _softmax(all_logits[j])
        cls_id = int(np.argmax(probs))
        cls_conf = float(probs[cls_id])

        if cls_conf > 0.5:  # only override if classifier is confident
            detections[idx]["category_id"] = cls_id

    return detections


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ---------------------------------------------------------------------------
# Single-pass inference
# ---------------------------------------------------------------------------

def _detections_to_normalized(detections: list[dict], img_w: int, img_h: int
                              ) -> tuple[list[list[float]], list[float], list[int]]:
    boxes_norm = []
    scores = []
    labels = []
    for det in detections:
        x, y, w, h = det["bbox"]
        boxes_norm.append([
            max(0.0, x / img_w),
            max(0.0, y / img_h),
            min(1.0, (x + w) / img_w),
            min(1.0, (y + h) / img_h),
        ])
        scores.append(det["score"])
        labels.append(det["category_id"])
    return boxes_norm, scores, labels


def run_single_pass(session: ort.InferenceSession, img: Image.Image,
                    img_w: int, img_h: int
                    ) -> tuple[list[list[float]], list[float], list[int]]:
    arr, ratio, pad_w, pad_h, _, _ = preprocess(img, IMGSZ)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: arr})

    detections = postprocess(outputs[0], ratio, pad_w, pad_h, img_w, img_h, CONF_THRESHOLD)
    boxes_norm, scores, labels = _detections_to_normalized(detections, img_w, img_h)

    if not boxes_norm:
        return [], [], []

    # Apply NMS to remove overlapping boxes from same model
    nms_boxes, nms_scores, nms_labels = nms(
        [boxes_norm], [scores], [labels],
        iou_thr=0.5,
    )
    return nms_boxes.tolist(), nms_scores.tolist(), nms_labels.tolist()


def run_tta_pass(session: ort.InferenceSession, img: Image.Image,
                 img_w: int, img_h: int
                 ) -> tuple[list[list[float]], list[float], list[int]]:
    """Test-Time Augmentation: original + horizontal flip, merged with WBF."""
    # Original
    boxes1, scores1, labels1 = run_single_pass(session, img, img_w, img_h)

    # Horizontal flip
    img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
    boxes_f, scores_f, labels_f = run_single_pass(session, img_flip, img_w, img_h)

    # Mirror flip boxes back: x1_new = 1 - x2_old, x2_new = 1 - x1_old
    boxes_flip = []
    for b in boxes_f:
        x1, y1, x2, y2 = b
        boxes_flip.append([
            max(0.0, min(1.0, 1.0 - x2)),
            max(0.0, min(1.0, y1)),
            max(0.0, min(1.0, 1.0 - x1)),
            max(0.0, min(1.0, y2)),
        ])

    # Merge with WBF
    all_boxes = []
    all_scores = []
    all_labels = []
    if boxes1:
        all_boxes.append(boxes1)
        all_scores.append(scores1)
        all_labels.append(labels1)
    if boxes_flip:
        all_boxes.append(boxes_flip)
        all_scores.append(scores_f)
        all_labels.append(labels_f)

    if not all_boxes:
        return [], [], []

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels,
        weights=[1.0] * len(all_boxes),
        iou_thr=WBF_IOU_THRESHOLD,
        skip_box_thr=WBF_SCORE_THRESHOLD,
    )
    return fused_boxes.tolist(), fused_scores.tolist(), fused_labels.tolist()


# ---------------------------------------------------------------------------
# Ensemble inference per image
# ---------------------------------------------------------------------------

def run_ensemble_for_image(
    sessions: list[ort.InferenceSession],
    classifier: ort.InferenceSession | None,
    image_path: Path,
) -> list[dict]:
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    image_id = extract_image_id(image_path.stem)

    all_boxes = []
    all_scores = []
    all_labels = []

    for session in sessions:
        boxes, scores, labels = run_tta_pass(session, img, img_w, img_h)
        if boxes:
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

    if not all_boxes:
        return []

    # Always run WBF — acts as NMS for single model, merges overlapping boxes
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels,
        weights=[1.0] * len(all_boxes),
        iou_thr=WBF_IOU_THRESHOLD,
        skip_box_thr=WBF_SCORE_THRESHOLD,
    )

    predictions = []
    for i in range(len(fused_boxes)):
        x1 = fused_boxes[i][0] * img_w
        y1 = fused_boxes[i][1] * img_h
        x2 = fused_boxes[i][2] * img_w
        y2 = fused_boxes[i][3] * img_h
        w = x2 - x1
        h = y2 - y1

        predictions.append({
            "image_id": image_id,
            "category_id": int(round(fused_labels[i])),
            "bbox": [round(float(x1), 1), round(float(y1), 1),
                     round(float(w), 1), round(float(h), 1)],
            "score": round(float(fused_scores[i]), 3),
        })

    # Limit detections per image (ultralytics default: 300)
    MAX_DET = 300
    if len(predictions) > MAX_DET:
        predictions.sort(key=lambda p: p["score"], reverse=True)
        predictions = predictions[:MAX_DET]

    if classifier and predictions:
        predictions = reclassify_detections(predictions, img, classifier)

    return predictions


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    if not input_dir.is_dir():
        raise ValueError(f"--input is not a directory: {input_dir}")

    script_dir = Path(__file__).parent
    sessions = load_onnx_sessions(script_dir)
    classifier = load_classifier(script_dir)

    image_paths = collect_images(input_dir)
    print(f"Found {len(image_paths)} image(s)")

    if not image_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump([], f)
        return

    all_predictions = []
    for idx, image_path in enumerate(image_paths):
        preds = run_ensemble_for_image(sessions, classifier, image_path)
        all_predictions.extend(preds)
        print(f"  {idx + 1}/{len(image_paths)} images, {len(all_predictions)} predictions", flush=True)

        # Write partial results every 10 images for incremental scoring
        if (idx + 1) % 10 == 0 or idx == len(image_paths) - 1:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(all_predictions, f)

    # Hard cap at 49000 predictions to stay under competition limit of 50000
    MAX_PREDICTIONS = 49000
    if len(all_predictions) > MAX_PREDICTIONS:
        all_predictions.sort(key=lambda p: p["score"], reverse=True)
        all_predictions = all_predictions[:MAX_PREDICTIONS]
        print(f"Capped to {MAX_PREDICTIONS} predictions (sorted by score)")

    print(f"Total: {len(all_predictions)} predictions")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_predictions, f)
    print(f"Written to: {output_path}")


if __name__ == "__main__":
    main()
