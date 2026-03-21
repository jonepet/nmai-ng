#!/usr/bin/env python3
"""
Prepare NorgesGruppen COCO dataset for YOLOv8 training.

Input:
  - /workspace/project/NM_NGD_coco_dataset.zip
  - /workspace/project/NM_NGD_product_images.zip

Output:
  - /workspace/project/data/yolo/{train,val}/{images,labels}/
  - /workspace/project/data/dataset.yaml
"""

import json
import os
import random
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract a zip file to dest_dir if not already extracted."""
    if dest_dir.exists():
        print(f"  Already extracted: {dest_dir}")
        return
    if not zip_path.exists():
        print(f"ERROR: zip file not found: {zip_path}", file=sys.stderr)
        sys.exit(1)
    print(f"  Extracting {zip_path.name} -> {dest_dir} ...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print(f"  Done extracting {zip_path.name}")


def find_images_dir(base: Path) -> Path:
    """Locate the images directory inside an extracted COCO archive."""
    # Try common layouts
    candidates = [
        base / "train" / "images",
        base / "images",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    # Fall back: first directory named 'images' found recursively
    for p in sorted(base.rglob("images")):
        if p.is_dir():
            return p
    print(f"ERROR: could not locate images directory under {base}", file=sys.stderr)
    sys.exit(1)


def find_annotations_file(base: Path) -> Path:
    """Locate annotations.json inside an extracted COCO archive."""
    candidates = [
        base / "train" / "annotations.json",
        base / "annotations" / "instances_train.json",
        base / "annotations.json",
    ]
    for c in candidates:
        if c.is_file():
            return c
    # Fall back: first annotations.json found recursively
    for p in sorted(base.rglob("annotations.json")):
        return p
    print(f"ERROR: could not locate annotations.json under {base}", file=sys.stderr)
    sys.exit(1)


def coco_bbox_to_yolo(bbox, img_width: int, img_height: int):
    """
    Convert COCO bbox [x, y, w, h] (pixels) to YOLO format
    [x_center, y_center, width, height] normalised to [0, 1].
    """
    x, y, w, h = bbox
    x_center = (x + w / 2.0) / img_width
    y_center = (y + h / 2.0) / img_height
    norm_w = w / img_width
    norm_h = h / img_height
    return x_center, y_center, norm_w, norm_h


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Step 1 – Extract
# ---------------------------------------------------------------------------


def step_extract():
    print("\n[1/7] Extracting zip files ...")
    extract_zip(config.COCO_ZIP, config.COCO_EXTRACT_DIR)
    extract_zip(config.PRODUCT_ZIP, config.PRODUCT_EXTRACT_DIR)


# ---------------------------------------------------------------------------
# Step 2 – Load COCO annotations
# ---------------------------------------------------------------------------


def step_load_annotations():
    print("\n[2/7] Loading COCO annotations ...")
    ann_file = find_annotations_file(config.COCO_EXTRACT_DIR)
    print(f"  Annotations file: {ann_file}")
    with open(ann_file, "r") as f:
        coco = json.load(f)
    return coco, ann_file


# ---------------------------------------------------------------------------
# Step 3 – Print statistics
# ---------------------------------------------------------------------------


def step_statistics(coco: dict) -> None:
    print("\n[3/7] Dataset statistics ...")

    num_images = len(coco.get("images", []))
    num_annotations = len(coco.get("annotations", []))
    num_categories = len(coco.get("categories", []))

    # Annotations per image
    ann_per_image = defaultdict(int)
    for ann in coco.get("annotations", []):
        ann_per_image[ann["image_id"]] += 1

    counts = list(ann_per_image.values())
    if counts:
        min_ann = min(counts)
        max_ann = max(counts)
        avg_ann = sum(counts) / len(counts)
    else:
        min_ann = max_ann = avg_ann = 0

    # Images with zero annotations
    zero_ann = num_images - len(ann_per_image)

    print(f"  Images      : {num_images}")
    print(f"  Annotations : {num_annotations}")
    print(f"  Categories  : {num_categories}")
    print(f"  Ann/image   : min={min_ann}  max={max_ann}  avg={avg_ann:.1f}")
    print(f"  Images with 0 annotations: {zero_ann}")


# ---------------------------------------------------------------------------
# Step 4 – Convert to YOLO labels
# ---------------------------------------------------------------------------


def step_convert(coco: dict, images_dir: Path) -> tuple[dict, dict, dict]:
    """
    Write YOLO label files to a temporary staging directory.

    Returns:
        image_id_to_info  – {image_id: {file_name, width, height}}
        image_id_to_anns  – {image_id: [ann, ...]}
        cat_id_to_yolo_id – {coco_cat_id: yolo_class_id}  (0-based)
    """
    print("\n[4/7] Converting COCO annotations to YOLO format ...")

    # Build category mapping: COCO category id -> YOLO class index
    categories = sorted(coco.get("categories", []), key=lambda c: c["id"])
    cat_id_to_yolo_id = {cat["id"]: idx for idx, cat in enumerate(categories)}

    # Build image info lookup
    image_id_to_info = {
        img["id"]: img for img in coco.get("images", [])
    }

    # Group annotations by image
    image_id_to_anns = defaultdict(list)
    for ann in coco.get("annotations", []):
        image_id_to_anns[ann["image_id"]].append(ann)

    # Staging directory for label files (keyed by image file stem)
    staging_dir = config.DATA_DIR / "_yolo_labels_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    skipped = 0
    written = 0

    for img_id, img_info in image_id_to_info.items():
        img_w = img_info.get("width")
        img_h = img_info.get("height")

        # Derive label filename from image filename
        img_stem = Path(img_info["file_name"]).stem
        label_path = staging_dir / f"{img_stem}.txt"

        anns = image_id_to_anns.get(img_id, [])
        lines = []
        for ann in anns:
            cat_id = ann.get("category_id")
            if cat_id not in cat_id_to_yolo_id:
                skipped += 1
                continue

            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                skipped += 1
                continue

            # If image dimensions are missing, try to infer from bbox (fallback)
            if not img_w or not img_h:
                skipped += 1
                continue

            yolo_id = cat_id_to_yolo_id[cat_id]
            xc, yc, nw, nh = coco_bbox_to_yolo(bbox, img_w, img_h)

            # Clamp to [0, 1] in case of minor floating-point overshoot
            xc = clamp(xc)
            yc = clamp(yc)
            nw = clamp(nw)
            nh = clamp(nh)

            lines.append(f"{yolo_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        written += 1

    print(f"  Written label files : {written}")
    print(f"  Skipped annotations : {skipped}")

    return image_id_to_info, dict(image_id_to_anns), cat_id_to_yolo_id, staging_dir


# ---------------------------------------------------------------------------
# Step 5 – Train/val split (stratified by dominant category)
# ---------------------------------------------------------------------------


def step_split(
    image_id_to_info: dict,
    image_id_to_anns: dict,
    cat_id_to_yolo_id: dict,
) -> tuple[list, list]:
    """Return (train_image_ids, val_image_ids) using stratified 80/20 split."""
    print("\n[5/7] Splitting into train/val (stratified 80/20) ...")

    rng = random.Random(config.RANDOM_SEED)

    # Determine dominant category for each image (by annotation count)
    image_dominant_cat = {}
    for img_id in image_id_to_info:
        anns = image_id_to_anns.get(img_id, [])
        if not anns:
            image_dominant_cat[img_id] = -1  # no annotations
            continue
        cat_counts = defaultdict(int)
        for ann in anns:
            cat_counts[ann.get("category_id", -1)] += 1
        dominant = max(cat_counts, key=lambda k: cat_counts[k])
        image_dominant_cat[img_id] = dominant

    # Group images by dominant category
    cat_to_images = defaultdict(list)
    for img_id, cat_id in image_dominant_cat.items():
        cat_to_images[cat_id].append(img_id)

    train_ids = []
    val_ids = []

    for cat_id, img_ids in cat_to_images.items():
        shuffled = list(img_ids)
        rng.shuffle(shuffled)
        n_train = max(1, round(len(shuffled) * config.TRAIN_RATIO))
        train_ids.extend(shuffled[:n_train])
        val_ids.extend(shuffled[n_train:])

    # Shuffle final lists for good measure
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)

    print(f"  Train images: {len(train_ids)}")
    print(f"  Val   images: {len(val_ids)}")

    return train_ids, val_ids


# ---------------------------------------------------------------------------
# Step 6 – Build YOLO directory structure
# ---------------------------------------------------------------------------


def _find_image_file(images_dir: Path, file_name: str) -> Path | None:
    """Locate image file handling both .jpg and .jpeg extensions."""
    candidate = images_dir / file_name
    if candidate.exists():
        return candidate

    stem = Path(file_name).stem
    for ext in (".jpg", ".jpeg", ".JPG", ".JPEG"):
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def step_build_yolo(
    train_ids: list,
    val_ids: list,
    image_id_to_info: dict,
    images_dir: Path,
    staging_dir: Path,
) -> None:
    print("\n[6/7] Building YOLO directory structure ...")

    splits = {"train": train_ids, "val": val_ids}

    for split_name, img_ids in splits.items():
        images_out = config.YOLO_DIR / split_name / "images"
        labels_out = config.YOLO_DIR / split_name / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        missing_images = 0
        missing_labels = 0
        copied = 0

        for img_id in img_ids:
            img_info = image_id_to_info[img_id]
            file_name = img_info["file_name"]
            stem = Path(file_name).stem

            # Locate source image
            src_img = _find_image_file(images_dir, file_name)
            if src_img is None:
                missing_images += 1
                continue

            # Locate staging label
            src_label = staging_dir / f"{stem}.txt"
            if not src_label.exists():
                missing_labels += 1
                continue

            # Copy (or symlink) image
            dst_img = images_out / src_img.name
            if not dst_img.exists():
                try:
                    dst_img.symlink_to(src_img.resolve())
                except (OSError, NotImplementedError):
                    shutil.copy2(src_img, dst_img)

            # Copy label
            dst_label = labels_out / f"{stem}.txt"
            if not dst_label.exists():
                shutil.copy2(src_label, dst_label)

            copied += 1

        print(
            f"  {split_name:5s}: {copied} pairs"
            f"  (missing_img={missing_images}, missing_label={missing_labels})"
        )


# ---------------------------------------------------------------------------
# Step 7 – Validate
# ---------------------------------------------------------------------------


def step_validate() -> bool:
    print("\n[7/7] Validating YOLO dataset ...")

    errors = []

    for split_name in ("train", "val"):
        images_dir = config.YOLO_DIR / split_name / "images"
        labels_dir = config.YOLO_DIR / split_name / "labels"

        if not images_dir.exists():
            errors.append(f"Missing directory: {images_dir}")
            continue
        if not labels_dir.exists():
            errors.append(f"Missing directory: {labels_dir}")
            continue

        image_files = sorted(
            p for p in images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        empty_labels = 0
        missing_labels = 0
        bbox_errors = 0

        for img_path in image_files:
            stem = img_path.stem
            label_path = labels_dir / f"{stem}.txt"

            if not label_path.exists():
                missing_labels += 1
                continue

            text = label_path.read_text().strip()
            if not text:
                empty_labels += 1
                continue

            for line_no, line in enumerate(text.splitlines(), start=1):
                parts = line.strip().split()
                if len(parts) != 5:
                    bbox_errors += 1
                    continue
                try:
                    _, xc, yc, w, h = (float(p) for p in parts)
                    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0
                            and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                        bbox_errors += 1
                except ValueError:
                    bbox_errors += 1

        status_parts = []
        if missing_labels:
            errors.append(f"[{split_name}] {missing_labels} images without label files")
            status_parts.append(f"missing_labels={missing_labels}")
        if bbox_errors:
            errors.append(f"[{split_name}] {bbox_errors} malformed bbox lines")
            status_parts.append(f"bbox_errors={bbox_errors}")

        ok = "PASS" if not status_parts else "FAIL"
        extra = f"  ({', '.join(status_parts)})" if status_parts else ""
        print(
            f"  {split_name:5s}: {ok} — "
            f"{len(image_files)} images, {empty_labels} empty labels{extra}"
        )

    if errors:
        print("\nValidation FAILED:")
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return False

    print("  Validation PASSED.")
    return True


# ---------------------------------------------------------------------------
# dataset.yaml
# ---------------------------------------------------------------------------


def write_dataset_yaml(coco: dict) -> None:
    print("\n  Writing dataset.yaml ...")

    categories = sorted(coco.get("categories", []), key=lambda c: c["id"])
    names = [cat["name"] for cat in categories]
    # Docs specify nc=357 (ids 0-356 with 356=unknown_product)
    # If data only has 356 categories (0-355), add the missing unknown_product
    if len(names) == 356:
        names.append("unknown_product")
    nc = config.NC  # from submission/config.json

    yaml_lines = [
        f"# NorgesGruppen COCO -> YOLOv8 dataset",
        f"path: {config.YOLO_DIR}",
        f"train: train/images",
        f"val: val/images",
        f"",
        f"nc: {nc}",
        f"names:",
    ]
    for name in names:
        # In YAML single-quoted strings, apostrophes must be doubled (not backslash-escaped)
        safe = name.replace("'", "''")
        yaml_lines.append(f"  - '{safe}'")

    config.DATASET_YAML_PATH.write_text("\n".join(yaml_lines) + "\n")
    print(f"  Written: {config.DATASET_YAML_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("NorgesGruppen COCO -> YOLOv8 data preparation")
    print("=" * 60)

    # 1. Extract
    step_extract()

    # 2. Load annotations
    coco, ann_file = step_load_annotations()

    # 3. Statistics
    step_statistics(coco)

    # 4. Convert to YOLO labels (staging)
    images_dir = find_images_dir(config.COCO_EXTRACT_DIR)
    print(f"  Images directory: {images_dir}")

    image_id_to_info, image_id_to_anns, cat_id_to_yolo_id, staging_dir = step_convert(
        coco, images_dir
    )

    # 5. Split
    train_ids, val_ids = step_split(image_id_to_info, image_id_to_anns, cat_id_to_yolo_id)

    # 6. Build YOLO dirs
    step_build_yolo(train_ids, val_ids, image_id_to_info, images_dir, staging_dir)

    # Write dataset.yaml
    write_dataset_yaml(coco)

    # 7. Validate
    ok = step_validate()

    print("\n" + "=" * 60)
    if ok:
        print("Data preparation complete.")
        print(f"  YOLO dataset : {config.YOLO_DIR}")
        print(f"  dataset.yaml : {config.DATASET_YAML_PATH}")
    else:
        print("Data preparation finished with validation errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
