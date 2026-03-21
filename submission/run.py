# NOTE: This file is self-contained for sandbox deployment. Config values are duplicated here intentionally.
"""
NorgesGruppen grocery product detection — ensemble inference with product re-ID.

Strategy:
    1. Load YOLOv8 models and run multi-scale detection
    2. Merge predictions with Weighted Box Fusion (WBF)
    3. For low-confidence classifications, re-identify products by comparing
       cropped detections against pre-computed product reference embeddings
    4. Maximize the 24GB L4 GPU and 300s timeout

Sandbox: Python 3.11, PyTorch 2.6.0+cu124, ultralytics 8.1.0,
         ensemble-boxes 1.0.9, numpy 1.26.4, Pillow 10.2.0
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from ensemble_boxes import weighted_boxes_fusion
from PIL import Image
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Configuration (duplicated — sandbox can't import config.py)
# ---------------------------------------------------------------------------

MODEL_FILES = ["best_main.pt", "best_parallel.pt"]
INFERENCE_SCALES = [640, 1280]
WBF_IOU_THRESHOLD = 0.55
WBF_SCORE_THRESHOLD = 0.001
WBF_SKIP_BOX_THRESHOLD = 0.0001

# Re-ID: reclassify detections below this confidence threshold
REID_CONFIDENCE_THRESHOLD = 0.5
# Re-ID: minimum similarity to accept reclassification
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


def load_models(script_dir: Path, device: str) -> list[YOLO]:
    models = []
    for name in MODEL_FILES:
        path = script_dir / name
        if path.exists():
            print(f"Loading model: {path.name}")
            m = YOLO(str(path))
            m.to(device)
            models.append(m)
    if not models:
        fallback = script_dir / "best.pt"
        if fallback.exists():
            print(f"Loading fallback model: {fallback.name}")
            m = YOLO(str(fallback))
            m.to(device)
            models.append(m)
    if not models:
        raise FileNotFoundError(f"No model weights found in {script_dir}")
    print(f"Loaded {len(models)} model(s)")
    return models


# ---------------------------------------------------------------------------
# Product Re-ID
# ---------------------------------------------------------------------------

class ProductReID:
    """Re-identify products by comparing crops against reference embeddings."""

    def __init__(self, script_dir: Path, yolo_model: YOLO, device: str):
        self.enabled = False
        self.device = device

        emb_path = script_dir / "product_embeddings.npy"
        map_path = script_dir / "product_mapping.json"

        if not emb_path.exists() or not map_path.exists():
            print("Product re-ID: disabled (no embeddings found)")
            return

        self.embeddings = np.load(str(emb_path))  # (N, embed_dim)
        with open(map_path, encoding="utf-8") as f:
            self.mapping = json.load(f)

        self.category_ids = [m["category_id"] for m in self.mapping]

        # Extract backbone from the YOLO model for feature extraction
        self.backbone = yolo_model.model.model[:10]
        self.backbone.eval()

        # Pre-build transform pipeline (reused for every crop)
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.enabled = True
        print(f"Product re-ID: enabled ({len(self.mapping)} products, dim={self.embeddings.shape[1]})")

    def _extract_embedding(self, crop_img: Image.Image) -> np.ndarray:
        """Extract normalized feature vector from a cropped product image."""
        tensor = self.transform(crop_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = tensor
            for layer in self.backbone:
                features = layer(features)
            pooled = torch.nn.functional.adaptive_avg_pool2d(features, 1)
            embedding = pooled.flatten().cpu().numpy()

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def reclassify(self, crop_img: Image.Image, current_score: float) -> tuple[int, float] | None:
        """
        Try to reclassify a crop using reference embeddings.

        Returns (category_id, similarity) if a good match is found, else None.
        Only called when current_score < REID_CONFIDENCE_THRESHOLD.
        """
        if not self.enabled:
            return None

        embedding = self._extract_embedding(crop_img)

        # Cosine similarity (embeddings are L2-normalized)
        similarities = self.embeddings @ embedding
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        if best_sim >= REID_MIN_SIMILARITY:
            return self.category_ids[best_idx], best_sim

        return None


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
    with Image.open(image_path) as img:
        return img.size


# ---------------------------------------------------------------------------
# Ensemble inference per image
# ---------------------------------------------------------------------------

def run_ensemble_for_image(
    models: list[YOLO],
    image_path: Path,
    device: str,
    reid: ProductReID | None,
) -> list[dict]:
    image_id = extract_image_id(image_path.stem)
    img_w, img_h = get_image_dimensions(image_path)

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

    weights = [1.0] * len(all_boxes)

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels,
        weights=weights,
        iou_thr=WBF_IOU_THRESHOLD,
        skip_box_thr=WBF_SCORE_THRESHOLD,
    )

    # Re-ID pass: reclassify low-confidence detections
    pil_img = None
    if reid and reid.enabled:
        pil_img = Image.open(image_path).convert("RGB")

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

        # Try re-ID for low-confidence classifications
        if reid and reid.enabled and score < REID_CONFIDENCE_THRESHOLD and pil_img is not None:
            # Crop the detection region
            crop = pil_img.crop((
                max(0, int(x1)),
                max(0, int(y1)),
                min(img_w, int(x2)),
                min(img_h, int(y2)),
            ))
            if crop.size[0] > 10 and crop.size[1] > 10:
                result = reid.reclassify(crop, score)
                if result is not None:
                    category_id, sim = result
                    # Use the higher of original score and similarity
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
    parser = argparse.ArgumentParser(description="NorgesGruppen product detection — ensemble + re-ID")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    input_dir: Path = args.input
    output_path: Path = args.output

    if not input_dir.is_dir():
        raise ValueError(f"--input is not a directory: {input_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    script_dir = Path(__file__).parent
    models = load_models(script_dir, device)
    print(f"Inference scales: {INFERENCE_SCALES}")

    # Initialize product re-ID (uses first model's backbone)
    reid = ProductReID(script_dir, models[0], device)

    image_paths = collect_images(input_dir)
    print(f"Found {len(image_paths)} image(s)")

    if not image_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("[]", encoding="utf-8")
        return

    all_predictions = []
    for idx, image_path in enumerate(image_paths):
        preds = run_ensemble_for_image(models, image_path, device, reid)
        all_predictions.extend(preds)
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  Processed {idx + 1}/{len(image_paths)} images, {len(all_predictions)} predictions")

    print(f"Total: {len(all_predictions)} predictions")

    all_predictions.sort(key=lambda p: p["image_id"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(all_predictions, indent=None, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Predictions written to: {output_path}")


if __name__ == "__main__":
    main()
