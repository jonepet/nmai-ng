"""
Pre-compute product reference embeddings for classification re-ID.

Uses the trained YOLO backbone to extract feature vectors from the
327 product reference images (multi-angle photos organized by barcode).

Outputs:
  - submission/product_embeddings.npy — (N, embed_dim) float32 array
  - submission/product_mapping.json — [{product_code, category_id, name}, ...]

The embeddings are used by run.py at inference time to reclassify
low-confidence detections by comparing crops against known products.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from PIL import Image
from ultralytics import YOLO


def load_category_mapping(annotations_path: Path, product_images_dir: Path) -> dict:
    """Build product_code → category_id mapping using metadata.json + category names."""
    with open(annotations_path) as f:
        coco = json.load(f)

    # category name → id
    name_to_cat_id = {c["name"]: c["id"] for c in coco["categories"]}
    # category id → name
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}

    # Use metadata.json to map product_code → product_name → category_id
    metadata_path = product_images_dir / "metadata.json"
    code_to_cat = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        for p in meta.get("products", []) + meta.get("missing", []):
            name = p.get("product_name", "")
            code = p.get("product_code", "")
            if name in name_to_cat_id and code:
                code_to_cat[code] = name_to_cat_id[name]

    return code_to_cat, cat_names


def extract_backbone_features(model: YOLO, img_path: Path, device: str) -> np.ndarray:
    """
    Extract feature vector from YOLO backbone for a single image.
    Returns a 1D numpy array (the global average pooled backbone output).
    """
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    # Get the backbone from the YOLO model
    backbone = model.model.model[:10]  # layers 0-9 are the backbone

    with torch.no_grad():
        features = tensor
        for layer in backbone:
            features = layer(features)

        # Global average pooling
        pooled = torch.nn.functional.adaptive_avg_pool2d(features, 1)
        embedding = pooled.flatten().cpu().numpy()

    return embedding


def main():
    print("=" * 60)
    print("Computing product reference embeddings")
    print("=" * 60)

    annotations_path = config.COCO_EXTRACT_DIR / "train" / "annotations.json"
    product_images_dir = config.PRODUCT_EXTRACT_DIR

    if not annotations_path.exists():
        print(f"Annotations not found: {annotations_path}")
        sys.exit(1)
    if not product_images_dir.exists():
        print(f"Product images not found: {product_images_dir}")
        sys.exit(1)

    # Load best model
    checkpoint = config.CHECKPOINT_ROOT / "best_final.pt"
    if not checkpoint.exists():
        for stage_num in [3, 2, 1]:
            candidate = config.CHECKPOINT_ROOT / f"best_stage_{stage_num}.pt"
            if candidate.exists():
                checkpoint = candidate
                break

    if not checkpoint.exists():
        print("No checkpoint found. Run training first.")
        sys.exit(1)

    print(f"Using checkpoint: {checkpoint}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(checkpoint))
    model.to(device)

    # Load mappings
    print("\nLoading category mappings...")
    code_to_cat, cat_names = load_category_mapping(annotations_path, product_images_dir)
    print(f"  {len(code_to_cat)} product codes mapped to categories")

    # Find all product reference images
    product_dirs = sorted(
        d for d in product_images_dir.iterdir()
        if d.is_dir()
    )
    print(f"  {len(product_dirs)} product directories found")

    # Compute embeddings for each product
    print("\nComputing embeddings...")
    embeddings = []
    mapping = []
    skipped = 0

    for idx, product_dir in enumerate(product_dirs):
        product_code = product_dir.name
        category_id = code_to_cat.get(product_code)

        if category_id is None:
            skipped += 1
            continue

        # Get all image files for this product
        image_files = sorted(
            p for p in product_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        if not image_files:
            skipped += 1
            continue

        # Compute embedding for each angle, then average
        product_embeddings = []
        for img_path in image_files:
            try:
                emb = extract_backbone_features(model, img_path, device)
                product_embeddings.append(emb)
            except Exception:
                continue

        if not product_embeddings:
            skipped += 1
            continue

        # Average all angles into one embedding per product
        avg_embedding = np.mean(product_embeddings, axis=0)
        # L2 normalize
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm

        embeddings.append(avg_embedding)
        mapping.append({
            "product_code": product_code,
            "category_id": category_id,
            "name": cat_names.get(category_id, "unknown"),
            "num_images": len(product_embeddings),
        })

        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(product_dirs)} products processed")

    print(f"\n  Computed embeddings for {len(embeddings)} products")
    print(f"  Skipped {skipped} products (no category mapping or images)")

    if not embeddings:
        print("No embeddings computed. Check data.")
        sys.exit(1)

    # Save
    embeddings_array = np.array(embeddings, dtype=np.float32)
    output_dir = config.SUBMISSION_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_path = output_dir / "product_embeddings.npy"
    json_path = output_dir / "product_mapping.json"

    np.save(str(npy_path), embeddings_array)
    with open(json_path, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"\n  Saved: {npy_path} ({embeddings_array.shape}, {npy_path.stat().st_size / 1024:.0f} KB)")
    print(f"  Saved: {json_path} ({len(mapping)} products)")
    print(f"  Embedding dimension: {embeddings_array.shape[1]}")
    print("\nDone.")


if __name__ == "__main__":
    main()
