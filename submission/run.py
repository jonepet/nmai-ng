# NOTE: This file is self-contained for sandbox deployment. Config values are duplicated here intentionally.
"""
NorgesGruppen grocery product detection — ensemble inference script.

Strategy:
    1. Load multiple trained YOLOv8 models (diverse training)
    2. Run each model at multiple scales (640, 1280) for better small-object detection
    3. Merge all predictions using Weighted Box Fusion (WBF)
    4. Maximize the 24GB L4 GPU and 300s timeout

Sandbox environment:
    Python 3.11, PyTorch 2.6.0+cu124, ultralytics 8.1.0
    NVIDIA L4 GPU (24GB VRAM), 8GB RAM, 4 vCPU
    300 s timeout, no network access
    Pre-installed: ensemble-boxes 1.0.9

Usage:
    python run.py --input /path/to/images --output /path/to/predictions.json
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from ensemble_boxes import weighted_boxes_fusion
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Configuration (duplicated from config.py — sandbox can't import it)
# ---------------------------------------------------------------------------

MODEL_FILES = ["best_main.pt", "best_parallel.pt"]
INFERENCE_SCALES = [640, 1280]
WBF_IOU_THRESHOLD = 0.55
WBF_SCORE_THRESHOLD = 0.001
WBF_SKIP_BOX_THRESHOLD = 0.0001


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


def load_models(script_dir: Path, device: str) -> list[YOLO]:
    """Load all available model weight files."""
    models = []
    for name in MODEL_FILES:
        path = script_dir / name
        if path.exists():
            print(f"Loading model: {path.name}")
            m = YOLO(str(path))
            m.to(device)
            models.append(m)
    if not models:
        # Fallback: try best.pt for backward compatibility
        fallback = script_dir / "best.pt"
        if fallback.exists():
            print(f"Loading fallback model: {fallback.name}")
            m = YOLO(str(fallback))
            m.to(device)
            models.append(m)
    if not models:
        raise FileNotFoundError(
            f"No model weights found in {script_dir}. "
            f"Expected: {MODEL_FILES} or best.pt"
        )
    print(f"Loaded {len(models)} model(s)")
    return models


# ---------------------------------------------------------------------------
# Single-pass inference
# ---------------------------------------------------------------------------

def run_single_pass(
    model: YOLO,
    image_path: Path,
    imgsz: int,
    device: str,
    img_w: int,
    img_h: int,
) -> tuple[list[list[float]], list[float], list[int]]:
    """
    Run one model at one scale on one image.

    Returns (boxes_norm, scores, labels) where boxes_norm is [[x1,y1,x2,y2], ...]
    normalized to [0,1] as required by WBF.
    """
    boxes_norm = []
    scores = []
    labels = []

    with torch.no_grad():
        results = model(str(image_path), device=device, verbose=False, imgsz=imgsz)

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)

        for i in range(len(result.boxes)):
            score = float(confs[i])
            if score < WBF_SKIP_BOX_THRESHOLD:
                continue

            x1, y1, x2, y2 = xyxy[i]
            # Normalize to [0, 1] for WBF
            boxes_norm.append([
                max(0.0, x1 / img_w),
                max(0.0, y1 / img_h),
                min(1.0, x2 / img_w),
                min(1.0, y2 / img_h),
            ])
            scores.append(score)
            labels.append(int(cls_ids[i]))

    return boxes_norm, scores, labels


# ---------------------------------------------------------------------------
# Image dimensions
# ---------------------------------------------------------------------------

def get_image_dimensions(image_path: Path) -> tuple[int, int]:
    """Get image width and height without importing PIL (uses torch/cv2)."""
    from PIL import Image
    with Image.open(image_path) as img:
        return img.size  # (width, height)


# ---------------------------------------------------------------------------
# Ensemble inference per image
# ---------------------------------------------------------------------------

def run_ensemble_for_image(
    models: list[YOLO],
    image_path: Path,
    device: str,
) -> list[dict]:
    """
    Run all models at all scales on one image, fuse with WBF,
    return COCO-format predictions.
    """
    image_id = extract_image_id(image_path.stem)
    img_w, img_h = get_image_dimensions(image_path)

    # Collect predictions from all model × scale combinations
    all_boxes = []
    all_scores = []
    all_labels = []

    for model in models:
        for imgsz in INFERENCE_SCALES:
            boxes, scores, labels = run_single_pass(
                model, image_path, imgsz, device, img_w, img_h
            )
            if boxes:
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.append(labels)

    if not all_boxes:
        return []

    # Weighted Box Fusion
    # weights: all passes contribute equally
    weights = [1.0] * len(all_boxes)

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        all_boxes,
        all_scores,
        all_labels,
        weights=weights,
        iou_thr=WBF_IOU_THRESHOLD,
        skip_box_thr=WBF_SCORE_THRESHOLD,
    )

    # Convert back to COCO format [x, y, w, h] in pixels
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
            "category_id": int(fused_labels[i]),
            "bbox": [round(x1, 4), round(y1, 4), round(w, 4), round(h, 4)],
            "score": round(float(fused_scores[i]), 6),
        })

    return predictions


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="NorgesGruppen product detection — ensemble inference")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    input_dir: Path = args.input
    output_path: Path = args.output

    if not input_dir.is_dir():
        raise ValueError(f"--input is not a directory: {input_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load all models
    script_dir = Path(__file__).parent
    models = load_models(script_dir, device)
    print(f"Inference scales: {INFERENCE_SCALES}")
    print(f"Total passes per image: {len(models)} model(s) x {len(INFERENCE_SCALES)} scale(s) = {len(models) * len(INFERENCE_SCALES)}")

    # Collect images
    image_paths = collect_images(input_dir)
    print(f"Found {len(image_paths)} image(s)")

    if not image_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("[]", encoding="utf-8")
        print("No images found — wrote empty predictions.")
        return

    # Run ensemble inference
    all_predictions = []
    for idx, image_path in enumerate(image_paths):
        preds = run_ensemble_for_image(models, image_path, device)
        all_predictions.extend(preds)
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  Processed {idx + 1}/{len(image_paths)} images, {len(all_predictions)} predictions so far")

    print(f"Total: {len(all_predictions)} predictions from {len(image_paths)} images")

    # Sort by image_id
    all_predictions.sort(key=lambda p: p["image_id"])

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(all_predictions, indent=None, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Predictions written to: {output_path}")


if __name__ == "__main__":
    main()
