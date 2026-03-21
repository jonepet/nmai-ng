"""
Train a product classifier on reference product images.

Uses the product_images/ directory (345 products, ~7 images each) to train
a lightweight classifier that maps cropped product images to category_ids.

Runs on a separate GPU or CPU in parallel with YOLO detection training.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# PyTorch 2.6 compat
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


PRODUCT_IMAGES_DIR = config.PRODUCT_EXTRACT_DIR
CLASSIFIER_CHECKPOINT = config.CHECKPOINT_ROOT / "classifier_best.pt"
CLASSIFIER_ONNX = config.SUBMISSION_DIR / "classifier.onnx"
INPUT_SIZE = 224  # EfficientNet-B0 native input size
BATCH_SIZE = 8
EPOCHS = 50
LR = 0.001


class ProductDataset(Dataset):
    """Dataset of product reference images mapped to category_ids."""

    def __init__(self, product_dir: Path, category_map: dict[str, int],
                 transform=None, augment_factor: int = 1):
        self.samples = []  # (image_path, category_id)
        self.transform = transform

        for product_code_dir in sorted(product_dir.iterdir()):
            if not product_code_dir.is_dir():
                continue
            code = product_code_dir.name
            if code not in category_map:
                continue
            cat_id = category_map[code]
            for img_path in product_code_dir.iterdir():
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    for _ in range(augment_factor):
                        self.samples.append((img_path, cat_id))

        print(f"  {len(self.samples)} samples, {len(set(s[1] for s in self.samples))} categories")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cat_id = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, cat_id


def build_category_map(annotations_path: Path) -> dict[str, int]:
    """Map product directory names (barcodes/codes) to YOLO category_ids.

    The product_images directories are named by product codes.
    The COCO annotations map category names to IDs.
    We need to link product codes to category IDs.
    """
    with open(annotations_path) as f:
        coco = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    # Build mapping from annotations: look at which product codes appear
    # in the image filenames or in a separate mapping file
    # For now, use the compute_embeddings approach: map directory names
    # to category IDs via the category name matching

    # The product directories might be named by barcode/EAN
    # We need a mapping file or heuristic to connect them to category IDs
    # Check if there's a mapping in the annotations

    # Try direct: see if product codes appear in category names or annotations
    product_dir = PRODUCT_IMAGES_DIR
    code_to_catid = {}

    if not product_dir.exists():
        print(f"  WARNING: Product images directory not found: {product_dir}")
        return {}

    product_codes = [d.name for d in product_dir.iterdir() if d.is_dir()]

    # Check if there's a product_mapping.json from compute_embeddings
    mapping_file = config.SUBMISSION_DIR / "product_mapping.json"
    if mapping_file.exists():
        with open(mapping_file) as f:
            mapping = json.load(f)
        # mapping format: {code: category_id} or similar
        if isinstance(mapping, dict):
            code_to_catid = {str(k): int(v) for k, v in mapping.items()}
            print(f"  Loaded mapping from {mapping_file}: {len(code_to_catid)} products")
            return code_to_catid

    # Fallback: try to match product codes to annotations
    # Look for product_code in image metadata or annotations
    for ann in coco.get("annotations", []):
        # Some datasets include product_code in annotation metadata
        if "product_code" in ann:
            code = str(ann["product_code"])
            if code in product_codes:
                code_to_catid[code] = ann["category_id"]

    if not code_to_catid:
        # Last resort: sequential mapping based on sorted product codes
        # matching sorted category names
        print("  WARNING: No direct mapping found. Using sequential mapping.")
        for i, code in enumerate(sorted(product_codes)):
            if i < len(categories):
                code_to_catid[code] = i

    print(f"  Mapped {len(code_to_catid)} product codes to category IDs")
    return code_to_catid


def train_classifier(device: str = "cpu"):
    print("=" * 60)
    print("Product Classifier Training")
    print("=" * 60)

    # Find annotations
    annotations_path = config.COCO_EXTRACT_DIR / "train" / "annotations.json"
    if not annotations_path.exists():
        print(f"ERROR: Annotations not found: {annotations_path}")
        sys.exit(1)

    print("\nBuilding category mapping...")
    category_map = build_category_map(annotations_path)
    if not category_map:
        print("ERROR: No category mapping found")
        sys.exit(1)

    num_classes = config.NC

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Dataset with augmentation (10x for training since we have few images)
    print("\nLoading training data...")
    dataset = ProductDataset(PRODUCT_IMAGES_DIR, category_map,
                             transform=train_transform, augment_factor=10)

    if len(dataset) == 0:
        print("ERROR: No training samples found")
        sys.exit(1)

    # Split 90/10
    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=(device != "cpu"))
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=(device != "cpu"))

    # Model: EfficientNet-B0 — good accuracy/size tradeoff
    print(f"\nBuilding EfficientNet-B0 classifier ({num_classes} classes)...")
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    config.CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining on {device} for {EPOCHS} epochs...")
    print(f"  Train: {n_train} samples, Val: {n_val} samples")

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        train_acc = train_correct / train_total if train_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        avg_loss = train_loss / train_total if train_total > 0 else 0

        print(f"  Epoch {epoch}/{EPOCHS}: loss={avg_loss:.4f} "
              f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} "
              f"lr={scheduler.get_last_lr()[0]:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "input_size": INPUT_SIZE,
                "category_map": category_map,
                "val_acc": val_acc,
                "epoch": epoch,
            }, CLASSIFIER_CHECKPOINT)
            print(f"    -> Saved best (val_acc={val_acc:.3f})")

            # Export ONNX on every new best
            model.eval()
            dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)
            config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
            torch.onnx.export(
                model, dummy, str(CLASSIFIER_ONNX),
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                opset_version=config.ONNX_OPSET,
            )
            print(f"    -> Exported ONNX ({CLASSIFIER_ONNX.stat().st_size / 1e6:.1f} MB)")
            model.train()

    print(f"\nTraining complete. Best val_acc: {best_val_acc:.3f}")

    # Export to ONNX
    print(f"\nExporting to ONNX: {CLASSIFIER_ONNX}")
    model.eval()
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)
    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model, dummy, str(CLASSIFIER_ONNX),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=config.ONNX_OPSET,
    )
    size_mb = CLASSIFIER_ONNX.stat().st_size / 1e6
    print(f"  Exported: {size_mb:.1f} MB")
    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu",
                        help="Training device: 'cpu', '0' (GPU 0), '1' (GPU 1)")
    args = parser.parse_args()

    device = args.device
    if device not in ("cpu",):
        device = f"cuda:{device}"

    train_classifier(device=device)
