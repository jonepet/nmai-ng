# NOTE: This file is self-contained for sandbox deployment.
# Uses ONNX format (recommended by competition docs) — no pickle, no version issues.
"""
NorgesGruppen grocery product detection — ONNX inference with ensemble + re-ID.

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
from ensemble_boxes import weighted_boxes_fusion
from PIL import Image


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_FILES = ["best_main.onnx", "best_parallel.onnx"]
INFERENCE_SCALES = [640]  # Fixed — ONNX exported at 640, no dynamic axes
CONF_THRESHOLD = 0.001
WBF_IOU_THRESHOLD = 0.55
WBF_SCORE_THRESHOLD = 0.001
NC = 357  # number of classes

# Re-ID
REID_CONFIDENCE_THRESHOLD = 0.5
REID_MIN_SIMILARITY = 0.3


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
    for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
        images.extend(input_dir.glob(ext))
    images.sort(key=lambda p: extract_image_id(p.stem))
    return images


# ---------------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------------

def load_onnx_sessions(script_dir: Path) -> list[ort.InferenceSession]:
    """Load all available ONNX model files."""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
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
    print(f"Loaded {len(sessions)} model(s), providers: {sessions[0].get_providers()}")
    return sessions


def preprocess(img: Image.Image, imgsz: int) -> tuple[np.ndarray, float, float, int, int]:
    """
    Letterbox resize + normalize for YOLOv8 ONNX input.
    Returns (input_tensor, ratio, pad_w, pad_h, orig_w, orig_h).
    """
    orig_w, orig_h = img.size
    ratio = min(imgsz / orig_w, imgsz / orig_h)
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)

    resized = img.resize((new_w, new_h), Image.BILINEAR)

    # Pad to imgsz x imgsz
    pad_w = (imgsz - new_w) // 2
    pad_h = (imgsz - new_h) // 2

    padded = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
    padded.paste(resized, (pad_w, pad_h))

    arr = np.array(padded, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    arr = np.expand_dims(arr, 0)  # add batch dim

    return arr, ratio, pad_w, pad_h, orig_w, orig_h


def postprocess(output: np.ndarray, ratio: float, pad_w: int, pad_h: int,
                orig_w: int, orig_h: int, conf_threshold: float) -> list[dict]:
    """
    Process YOLOv8 ONNX output: shape (1, 4+NC, num_boxes).
    Returns list of {bbox: [x,y,w,h], category_id: int, score: float} in pixel coords.
    """
    # YOLOv8 ONNX output shape: (1, 4+nc, 8400) — transpose to (8400, 4+nc)
    preds = output[0].T  # (8400, 4+nc)

    # Split into boxes and class scores
    boxes_xywh = preds[:, :4]  # center_x, center_y, w, h in input coords
    class_scores = preds[:, 4:]  # (8400, nc)

    # Get best class per box
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(len(class_ids)), class_ids]

    # Filter by confidence
    mask = confidences > conf_threshold
    boxes_xywh = boxes_xywh[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    results = []
    for i in range(len(boxes_xywh)):
        cx, cy, w, h = boxes_xywh[i]

        # Remove padding and scale back to original image coords
        x1 = (cx - w / 2 - pad_w) / ratio
        y1 = (cy - h / 2 - pad_h) / ratio
        bw = w / ratio
        bh = h / ratio

        # Clip to image bounds
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        bw = min(bw, orig_w - x1)
        bh = min(bh, orig_h - y1)

        if bw > 0 and bh > 0:
            results.append({
                "bbox": [round(float(x1), 4), round(float(y1), 4),
                         round(float(bw), 4), round(float(bh), 4)],
                "category_id": int(class_ids[i]),
                "score": round(float(confidences[i]), 6),
            })

    return results


def run_single_pass(session: ort.InferenceSession, img: Image.Image,
                    imgsz: int, img_w: int, img_h: int
                    ) -> tuple[list[list[float]], list[float], list[int]]:
    """Run one ONNX model at one scale, return WBF-format results."""
    arr, ratio, pad_w, pad_h, _, _ = preprocess(img, imgsz)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: arr})

    detections = postprocess(outputs[0], ratio, pad_w, pad_h, img_w, img_h, CONF_THRESHOLD)

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


# ---------------------------------------------------------------------------
# Product Re-ID (optional — uses pre-computed embeddings)
# ---------------------------------------------------------------------------

class ProductReID:
    def __init__(self, script_dir: Path, first_session: ort.InferenceSession):
        self.enabled = False
        emb_path = script_dir / "product_embeddings.npy"
        map_path = script_dir / "product_mapping.json"

        if not emb_path.exists() or not map_path.exists():
            print("Product re-ID: disabled (no embeddings)")
            return

        self.embeddings = np.load(str(emb_path))
        with open(map_path, encoding="utf-8") as f:
            self.mapping = json.load(f)
        self.category_ids = [m["category_id"] for m in self.mapping]
        self.enabled = True
        print(f"Product re-ID: enabled ({len(self.mapping)} products)")

    def reclassify(self, crop_arr: np.ndarray) -> tuple[int, float] | None:
        """Compare crop embedding against reference. Returns (cat_id, sim) or None."""
        if not self.enabled:
            return None
        # Simple: use mean pixel values as a basic feature vector
        # (full backbone extraction removed to avoid eval() and complexity)
        h, w = crop_arr.shape[:2]
        if h < 10 or w < 10:
            return None
        # Resize to fixed size and flatten as feature
        from PIL import Image as PILImage
        crop_img = PILImage.fromarray(crop_arr)
        crop_resized = crop_img.resize((32, 32))
        feature = np.array(crop_resized, dtype=np.float32).flatten()
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm

        # Match embedding dimensions
        if feature.shape[0] != self.embeddings.shape[1]:
            return None

        similarities = self.embeddings @ feature
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        if best_sim >= REID_MIN_SIMILARITY:
            return self.category_ids[best_idx], best_sim
        return None


# ---------------------------------------------------------------------------
# Ensemble inference
# ---------------------------------------------------------------------------

def run_ensemble_for_image(
    sessions: list[ort.InferenceSession],
    image_path: Path,
    reid: ProductReID | None,
) -> list[dict]:
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    image_id = extract_image_id(image_path.stem)

    all_boxes = []
    all_scores = []
    all_labels = []

    for session in sessions:
        for imgsz in INFERENCE_SCALES:
            boxes, scores, labels = run_single_pass(session, img, imgsz, img_w, img_h)
            if boxes:
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.append(labels)

    if not all_boxes:
        return []

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels,
        weights=[1.0] * len(all_boxes),
        iou_thr=WBF_IOU_THRESHOLD,
        skip_box_thr=WBF_SCORE_THRESHOLD,
    )

    img_arr = np.array(img) if reid and reid.enabled else None

    predictions = []
    for i in range(len(fused_boxes)):
        x1 = fused_boxes[i][0] * img_w
        y1 = fused_boxes[i][1] * img_h
        x2 = fused_boxes[i][2] * img_w
        y2 = fused_boxes[i][3] * img_h
        w = x2 - x1
        h = y2 - y1
        score = float(fused_scores[i])
        category_id = int(fused_labels[i])

        # Re-ID for low-confidence classifications
        if reid and reid.enabled and score < REID_CONFIDENCE_THRESHOLD and img_arr is not None:
            crop = img_arr[max(0, int(y1)):min(img_h, int(y2)),
                          max(0, int(x1)):min(img_w, int(x2))]
            result = reid.reclassify(crop)
            if result is not None:
                category_id, sim = result
                score = max(score, sim)

        predictions.append({
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [round(x1, 4), round(y1, 4), round(w, 4), round(h, 4)],
            "score": round(score, 6),
        })

    return predictions


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    input_dir: Path = args.input
    output_path: Path = args.output

    if not input_dir.is_dir():
        raise ValueError(f"--input is not a directory: {input_dir}")

    script_dir = Path(__file__).parent
    sessions = load_onnx_sessions(script_dir)
    print(f"Inference scales: {INFERENCE_SCALES}")

    reid = ProductReID(script_dir, sessions[0])

    image_paths = collect_images(input_dir)
    print(f"Found {len(image_paths)} image(s)")

    if not image_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("[]", encoding="utf-8")
        return

    all_predictions = []
    for idx, image_path in enumerate(image_paths):
        preds = run_ensemble_for_image(sessions, image_path, reid)
        all_predictions.extend(preds)
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  {idx + 1}/{len(image_paths)} images, {len(all_predictions)} predictions")

    print(f"Total: {len(all_predictions)} predictions")
    all_predictions.sort(key=lambda p: p["image_id"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(all_predictions, indent=None, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Written to: {output_path}")


if __name__ == "__main__":
    main()
