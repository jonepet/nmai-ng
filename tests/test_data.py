"""
Training data integrity tests for the NorgesGruppen competition dataset.

Skips gracefully when data hasn't been extracted yet.
DATA_DIR defaults to config.DATA_DIR but can be overridden via the DATA_DIR
environment variable.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.environ.get("DATA_DIR", config.DATA_DIR))

# COCO dataset paths (after extraction)
COCO_EXTRACT_DIR = config.COCO_EXTRACT_DIR
ANNOTATIONS_FILE = config.COCO_EXTRACT_DIR / "train" / "annotations.json"
IMAGES_DIR = config.COCO_EXTRACT_DIR / "train" / "images"

# Product images
PRODUCT_IMAGES_DIR = config.PRODUCT_EXTRACT_DIR

# YOLO output paths (after prepare_data runs)
YOLO_DIR = config.YOLO_DIR
YOLO_TRAIN_IMAGES = YOLO_DIR / "train" / "images"
YOLO_TRAIN_LABELS = YOLO_DIR / "train" / "labels"
YOLO_VAL_IMAGES = YOLO_DIR / "val" / "images"
YOLO_VAL_LABELS = YOLO_DIR / "val" / "labels"

# Expected dataset statistics
EXPECTED_IMAGE_COUNT = config.EXPECTED_IMAGE_COUNT
EXPECTED_ANNOTATION_COUNT_MIN = config.EXPECTED_ANNOTATION_COUNT_MIN
EXPECTED_NUM_CATEGORIES = config.EXPECTED_NUM_CATEGORIES
EXPECTED_MIN_CATEGORY_ID = 0
EXPECTED_MAX_CATEGORY_ID = 356

# Skip markers
_data_not_extracted = pytest.mark.skipif(
    not ANNOTATIONS_FILE.exists(),
    reason=f"COCO data not yet extracted — annotations.json not found at {ANNOTATIONS_FILE}",
)
_yolo_not_prepared = pytest.mark.skipif(
    not YOLO_TRAIN_LABELS.exists(),
    reason=f"YOLO data not yet prepared — {YOLO_TRAIN_LABELS} does not exist",
)

# ---------------------------------------------------------------------------
# Shared fixture — load COCO annotations once per session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def coco() -> dict:
    """Load and return the parsed COCO annotations file."""
    with ANNOTATIONS_FILE.open("r") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def image_id_map(coco: dict) -> dict[int, dict]:
    """Return a dict mapping image_id -> image info dict."""
    return {img["id"]: img for img in coco.get("images", [])}


@pytest.fixture(scope="session")
def category_id_set(coco: dict) -> set[int]:
    """Return a set of all category IDs."""
    return {cat["id"] for cat in coco.get("categories", [])}


# ---------------------------------------------------------------------------
# 1. File existence and structure
# ---------------------------------------------------------------------------


def test_annotations_file_exists():
    """annotations.json must exist on disk after extraction."""
    assert ANNOTATIONS_FILE.exists(), (
        f"annotations.json not found at {ANNOTATIONS_FILE}. "
        "Run extraction first."
    )


@_data_not_extracted
def test_annotations_valid_json():
    """annotations.json must be valid JSON with required top-level keys."""
    with ANNOTATIONS_FILE.open("r") as fh:
        data = json.load(fh)

    assert isinstance(data, dict), "Annotations root must be a JSON object"
    for key in ("images", "categories", "annotations"):
        assert key in data, f"Missing required key '{key}' in annotations.json"
    assert isinstance(data["images"], list), "'images' must be a list"
    assert isinstance(data["categories"], list), "'categories' must be a list"
    assert isinstance(data["annotations"], list), "'annotations' must be a list"


# ---------------------------------------------------------------------------
# 2. Counts
# ---------------------------------------------------------------------------


@_data_not_extracted
def test_image_count(coco: dict):
    """Dataset must contain approximately 248 images."""
    count = len(coco.get("images", []))
    # Allow ±10 to tolerate minor dataset updates
    assert abs(count - EXPECTED_IMAGE_COUNT) <= 10, (
        f"Expected ~{EXPECTED_IMAGE_COUNT} images, got {count}"
    )


@_data_not_extracted
def test_annotation_count(coco: dict):
    """Dataset must have more than 20 000 annotations."""
    count = len(coco.get("annotations", []))
    assert count > EXPECTED_ANNOTATION_COUNT_MIN, (
        f"Expected >{EXPECTED_ANNOTATION_COUNT_MIN} annotations, got {count}"
    )


@_data_not_extracted
def test_category_count(coco: dict):
    """Dataset must contain exactly 357 categories."""
    count = len(coco.get("categories", []))
    assert count == EXPECTED_NUM_CATEGORIES, (
        f"Expected {EXPECTED_NUM_CATEGORIES} categories, got {count}"
    )


# ---------------------------------------------------------------------------
# 3. Category integrity
# ---------------------------------------------------------------------------


@_data_not_extracted
def test_category_ids_continuous(coco: dict):
    """Category IDs must form an unbroken range from 0 to 356."""
    ids = sorted(cat["id"] for cat in coco.get("categories", []))
    expected = list(range(EXPECTED_MIN_CATEGORY_ID, EXPECTED_MAX_CATEGORY_ID + 1))
    assert ids == expected, (
        f"Category IDs are not continuous [0..{EXPECTED_MAX_CATEGORY_ID}]. "
        f"Min={min(ids)}, max={max(ids)}, count={len(ids)}"
    )


# ---------------------------------------------------------------------------
# 4. Image file existence
# ---------------------------------------------------------------------------


@_data_not_extracted
def test_all_images_exist(coco: dict):
    """Every image referenced in the annotations file must exist on disk."""
    missing = []
    for img in coco.get("images", []):
        img_path = IMAGES_DIR / img["file_name"]
        if not img_path.exists():
            # Also try resolving symlinks / alternate extension
            alt = img_path.with_suffix(".jpeg")
            if not alt.exists():
                missing.append(img["file_name"])

    assert not missing, (
        f"{len(missing)} image file(s) missing from {IMAGES_DIR}. "
        f"First few: {missing[:5]}"
    )


# ---------------------------------------------------------------------------
# 5. Bounding box integrity
# ---------------------------------------------------------------------------


@_data_not_extracted
def test_bbox_format(coco: dict):
    """Every bbox must be a list of 4 numeric values."""
    bad = []
    for ann in coco.get("annotations", []):
        bbox = ann.get("bbox")
        if bbox is None:
            bad.append(ann["id"])
            continue
        if len(bbox) != 4:
            bad.append(ann["id"])
            continue
        if not all(isinstance(v, (int, float)) for v in bbox):
            bad.append(ann["id"])

    assert not bad, (
        f"{len(bad)} annotation(s) have malformed bbox (not 4 numeric values). "
        f"First few IDs: {bad[:5]}"
    )


@_data_not_extracted
def test_bbox_valid(coco: dict, image_id_map: dict[int, dict]):
    """All bboxes must have positive width/height and stay within image dimensions."""
    bad_size = []
    bad_bounds = []

    for ann in coco.get("annotations", []):
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue  # covered by test_bbox_format

        x, y, w, h = bbox

        if w <= 0 or h <= 0:
            bad_size.append(ann["id"])
            continue

        img_info = image_id_map.get(ann.get("image_id"))
        if img_info is None:
            continue  # covered by test_annotation_references_valid_image

        img_w = img_info.get("width")
        img_h = img_info.get("height")
        if img_w is None or img_h is None:
            continue

        if x < 0 or y < 0 or (x + w) > img_w or (y + h) > img_h:
            bad_bounds.append(ann["id"])

    assert not bad_size, (
        f"{len(bad_size)} annotation(s) have non-positive bbox dimensions. "
        f"First few IDs: {bad_size[:5]}"
    )
    assert not bad_bounds, (
        f"{len(bad_bounds)} annotation(s) have bbox extending outside image boundaries. "
        f"First few IDs: {bad_bounds[:5]}"
    )


# ---------------------------------------------------------------------------
# 6. Cross-reference integrity
# ---------------------------------------------------------------------------


@_data_not_extracted
def test_annotation_references_valid_image(coco: dict, image_id_map: dict[int, dict]):
    """Every annotation.image_id must reference an image in the images list."""
    bad = [
        ann["id"]
        for ann in coco.get("annotations", [])
        if ann.get("image_id") not in image_id_map
    ]
    assert not bad, (
        f"{len(bad)} annotation(s) reference unknown image_id. "
        f"First few annotation IDs: {bad[:5]}"
    )


@_data_not_extracted
def test_annotation_references_valid_category(coco: dict, category_id_set: set[int]):
    """Every annotation.category_id must reference a category in the categories list."""
    bad = [
        ann["id"]
        for ann in coco.get("annotations", [])
        if ann.get("category_id") not in category_id_set
    ]
    assert not bad, (
        f"{len(bad)} annotation(s) reference unknown category_id. "
        f"First few annotation IDs: {bad[:5]}"
    )


# ---------------------------------------------------------------------------
# 7. Per-image annotation coverage
# ---------------------------------------------------------------------------


@_data_not_extracted
def test_no_empty_images(coco: dict, image_id_map: dict[int, dict]):
    """Warn (not fail) if any image has zero annotations."""
    ann_image_ids = {ann["image_id"] for ann in coco.get("annotations", [])}
    empty = [
        img["file_name"]
        for img in coco.get("images", [])
        if img["id"] not in ann_image_ids
    ]
    if empty:
        import warnings
        warnings.warn(
            f"{len(empty)} image(s) have no annotations: {empty[:5]}",
            stacklevel=2,
        )
    # Not a hard failure — warn only
    assert True


# ---------------------------------------------------------------------------
# 8. Area consistency
# ---------------------------------------------------------------------------


@_data_not_extracted
def test_area_matches_bbox(coco: dict):
    """Annotation area must approximately equal bbox_width * bbox_height."""
    tolerance = 0.01  # 1 % relative tolerance
    bad = []

    for ann in coco.get("annotations", []):
        area = ann.get("area")
        bbox = ann.get("bbox")
        if area is None or not bbox or len(bbox) != 4:
            continue

        _, _, w, h = bbox
        expected_area = w * h
        if expected_area == 0:
            continue

        rel_diff = abs(area - expected_area) / expected_area
        if rel_diff > tolerance:
            bad.append((ann["id"], area, expected_area, rel_diff))

    assert not bad, (
        f"{len(bad)} annotation(s) have area inconsistent with bbox (>1% diff). "
        f"First few (id, stored_area, bbox_area, rel_diff): {bad[:5]}"
    )


# ---------------------------------------------------------------------------
# 9. YOLO conversion accuracy (requires prepare_data to have run)
# ---------------------------------------------------------------------------


def _yolo_bbox_to_coco(xc: float, yc: float, nw: float, nh: float, img_w: int, img_h: int):
    """Convert normalised YOLO bbox back to COCO [x, y, w, h] pixel coords."""
    w = nw * img_w
    h = nh * img_h
    x = (xc * img_w) - w / 2.0
    y = (yc * img_h) - h / 2.0
    return x, y, w, h


def _iou(a: tuple, b: tuple) -> float:
    """Compute IoU between two [x, y, w, h] bboxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix = max(0.0, min(ax2, bx2) - max(ax, bx))
    iy = max(0.0, min(ay2, by2) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@_data_not_extracted
@_yolo_not_prepared
def test_yolo_conversion_accuracy(coco: dict, image_id_map: dict[int, dict]):
    """
    Sample 10 images from the YOLO train split, reconstruct COCO bboxes from
    the YOLO label files, and verify IoU >= 0.99 against the original COCO
    annotations for every box.
    """
    import random as _random

    # Build category id -> yolo class index mapping (same logic as prepare_data)
    categories = sorted(coco.get("categories", []), key=lambda c: c["id"])
    cat_id_to_yolo_id: dict[int, int] = {cat["id"]: idx for idx, cat in enumerate(categories)}
    yolo_id_to_cat_id: dict[int, int] = {v: k for k, v in cat_id_to_yolo_id.items()}

    # Build image_id -> list of annotations
    image_id_to_anns: dict[int, list] = defaultdict(list)
    for ann in coco.get("annotations", []):
        image_id_to_anns[ann["image_id"]].append(ann)

    # Build stem -> image_id from annotations file
    stem_to_image_id: dict[str, int] = {
        Path(img["file_name"]).stem: img["id"] for img in coco.get("images", [])
    }

    # Collect YOLO label files from the train split
    label_files = sorted(YOLO_TRAIN_LABELS.glob("*.txt"))
    assert label_files, f"No label files found in {YOLO_TRAIN_LABELS}"

    rng = _random.Random(42)
    sample = rng.sample(label_files, min(10, len(label_files)))

    low_iou_cases: list[tuple] = []

    for label_path in sample:
        stem = label_path.stem
        img_id = stem_to_image_id.get(stem)
        if img_id is None:
            continue  # image not in COCO — skip

        img_info = image_id_map[img_id]
        img_w = img_info.get("width")
        img_h = img_info.get("height")
        if not img_w or not img_h:
            continue

        coco_anns = image_id_to_anns.get(img_id, [])
        # Build lookup: yolo_class_id -> list of COCO bboxes
        coco_by_class: dict[int, list] = defaultdict(list)
        for ann in coco_anns:
            yolo_id = cat_id_to_yolo_id.get(ann["category_id"])
            if yolo_id is not None:
                coco_by_class[yolo_id].append(ann["bbox"])

        lines = label_path.read_text().strip().splitlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            yolo_id, xc, yc, nw, nh = int(parts[0]), *map(float, parts[1:])
            reconstructed = _yolo_bbox_to_coco(xc, yc, nw, nh, img_w, img_h)

            # Find best matching COCO bbox for this class
            candidates = coco_by_class.get(yolo_id, [])
            if not candidates:
                continue

            best_iou = max(_iou(reconstructed, tuple(c)) for c in candidates)
            if best_iou < 0.99:
                low_iou_cases.append((stem, yolo_id, best_iou, reconstructed))

    assert not low_iou_cases, (
        f"{len(low_iou_cases)} YOLO box(es) had IoU < 0.99 vs COCO originals. "
        f"First few (image_stem, yolo_class, iou, reconstructed_bbox): {low_iou_cases[:5]}"
    )


# ---------------------------------------------------------------------------
# 10. Train/val split existence and ratio
# ---------------------------------------------------------------------------


@_yolo_not_prepared
def test_train_val_split():
    """YOLO train and val directories must exist and be roughly 80/20."""
    for d in (YOLO_TRAIN_IMAGES, YOLO_TRAIN_LABELS, YOLO_VAL_IMAGES, YOLO_VAL_LABELS):
        assert d.exists(), f"Expected YOLO directory missing: {d}"

    _img_exts = {".jpg", ".jpeg", ".png"}
    n_train = sum(1 for p in YOLO_TRAIN_IMAGES.iterdir() if p.suffix.lower() in _img_exts)
    n_val = sum(1 for p in YOLO_VAL_IMAGES.iterdir() if p.suffix.lower() in _img_exts)

    assert n_train > 0, "Train split is empty"
    assert n_val > 0, "Val split is empty"

    total = n_train + n_val
    train_ratio = n_train / total

    # Expect somewhere between 75% and 85%
    assert 0.75 <= train_ratio <= 0.85, (
        f"Train/val split is {train_ratio:.1%} / {1 - train_ratio:.1%} "
        f"({n_train}/{n_val}), expected ~80/20"
    )
