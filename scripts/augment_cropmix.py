"""
augment_cropmix.py — Offline crop and mosaic augmentation for training data.

Generates two types of synthetic images with 100% accurate YOLO annotations:
  Type 1: Random crops from individual training images
  Type 2: 2x2 mosaic grids from 4 randomly combined training images

Output goes directly into the training set (same images/ and labels/ dirs).
Idempotent: skips files that already exist.
"""

import sys
import re
import random
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COUNTER_START = 90000
TARGET_CROPS = 200
TARGET_MOSAICS = 100

# Crop parameters
CROP_SCALE_MIN = 0.50
CROP_SCALE_MAX = 0.80
CROPS_PER_IMAGE_MIN = 2
CROPS_PER_IMAGE_MAX = 4
MIN_BOXES_IN_CROP = 3
MIN_BOX_VISIBILITY = 0.30   # fraction of original box area that must survive crop
BOX_IOU_KEEP_THRESHOLD = 0.70  # IoU with crop region to count as "complete"

# Mosaic parameters
MOSAIC_TARGET_W = 1000
MOSAIC_TARGET_H = 750

# Augmentation suffix pattern — exclude files produced by other augmenters
_AUG_SUFFIX_RE = re.compile(
    r"_(bright|dark|blur|flip|rot|crop|mosaic|aug)\d*$", re.IGNORECASE
)

IMAGES_DIR = config.YOLO_DIR / "train" / "images"
LABELS_DIR = config.YOLO_DIR / "train" / "labels"

# ---------------------------------------------------------------------------
# Helpers — YOLO label I/O
# ---------------------------------------------------------------------------


def read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Return list of (class_id, cx, cy, w, h) normalized tuples."""
    boxes = []
    if not label_path.exists():
        return boxes
    with label_path.open() as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            boxes.append((cls, cx, cy, bw, bh))
    return boxes


def write_yolo_labels(label_path: Path, boxes: list[tuple[int, float, float, float, float]]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w") as fh:
        for cls, cx, cy, bw, bh in boxes:
            fh.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


# ---------------------------------------------------------------------------
# Helpers — bounding box math
# ---------------------------------------------------------------------------


def yolo_to_pixel(cx: float, cy: float, bw: float, bh: float,
                  img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert normalized YOLO box to pixel (x1, y1, x2, y2)."""
    pw = bw * img_w
    ph = bh * img_h
    px1 = cx * img_w - pw / 2
    py1 = cy * img_h - ph / 2
    px2 = px1 + pw
    py2 = py1 + ph
    return px1, py1, px2, py2


def pixel_to_yolo(px1: float, py1: float, px2: float, py2: float,
                  img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert pixel (x1,y1,x2,y2) to normalized YOLO (cx, cy, w, h)."""
    cx = (px1 + px2) / 2 / img_w
    cy = (py1 + py2) / 2 / img_h
    bw = (px2 - px1) / img_w
    bh = (py2 - py1) / img_h
    return cx, cy, bw, bh


def clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def validate_box(cx: float, cy: float, bw: float, bh: float) -> bool:
    """Return True if the normalized box is geometrically valid."""
    if bw <= 0 or bh <= 0:
        return False
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        return False
    return True


# ---------------------------------------------------------------------------
# Type 1 — Random crops
# ---------------------------------------------------------------------------


def crop_boxes(
    src_boxes: list[tuple[int, float, float, float, float]],
    crop_x1: int, crop_y1: int, crop_x2: int, crop_y2: int,
    img_w: int, img_h: int,
) -> list[tuple[int, float, float, float, float]]:
    """
    Given a crop region (pixel coords) and the original image dimensions,
    return YOLO boxes re-anchored to the crop, filtering by visibility.
    """
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1
    result = []
    for cls, cx, cy, bw, bh in src_boxes:
        bpx1, bpy1, bpx2, bpy2 = yolo_to_pixel(cx, cy, bw, bh, img_w, img_h)

        # Original box area
        orig_area = (bpx2 - bpx1) * (bpy2 - bpy1)
        if orig_area <= 0:
            continue

        # Intersect with crop
        ix1 = max(bpx1, crop_x1)
        iy1 = max(bpy1, crop_y1)
        ix2 = min(bpx2, crop_x2)
        iy2 = min(bpy2, crop_y2)

        if ix2 <= ix1 or iy2 <= iy1:
            continue  # no overlap

        visible_area = (ix2 - ix1) * (iy2 - iy1)
        visibility = visible_area / orig_area

        if visibility < MIN_BOX_VISIBILITY:
            continue

        # Re-anchor to crop coordinate system
        rx1 = ix1 - crop_x1
        ry1 = iy1 - crop_y1
        rx2 = ix2 - crop_x1
        ry2 = iy2 - crop_y1

        # Clamp strictly within crop
        rx1 = max(0.0, rx1)
        ry1 = max(0.0, ry1)
        rx2 = min(float(crop_w), rx2)
        ry2 = min(float(crop_h), ry2)

        if rx2 <= rx1 or ry2 <= ry1:
            continue

        ncx, ncy, nbw, nbh = pixel_to_yolo(rx1, ry1, rx2, ry2, crop_w, crop_h)
        ncx = clamp(ncx)
        ncy = clamp(ncy)
        nbw = clamp(nbw)
        nbh = clamp(nbh)

        if not validate_box(ncx, ncy, nbw, nbh):
            continue

        result.append((cls, ncx, ncy, nbw, nbh))
    return result


def count_complete_boxes(
    src_boxes: list[tuple[int, float, float, float, float]],
    crop_x1: int, crop_y1: int, crop_x2: int, crop_y2: int,
    img_w: int, img_h: int,
) -> int:
    """Count boxes with IoU(box, crop_region) >= threshold."""
    crop_area = (crop_x2 - crop_x1) * (crop_y2 - crop_y1)
    count = 0
    for _, cx, cy, bw, bh in src_boxes:
        bpx1, bpy1, bpx2, bpy2 = yolo_to_pixel(cx, cy, bw, bh, img_w, img_h)
        ix1 = max(bpx1, crop_x1)
        iy1 = max(bpy1, crop_y1)
        ix2 = min(bpx2, crop_x2)
        iy2 = min(bpy2, crop_y2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        inter = (ix2 - ix1) * (iy2 - iy1)
        box_area = (bpx2 - bpx1) * (bpy2 - bpy1)
        union = box_area + crop_area - inter
        if union <= 0:
            continue
        iou = inter / union
        if iou >= BOX_IOU_KEEP_THRESHOLD:
            count += 1
    return count


def generate_crops(
    image_paths: list[Path],
    counter_start: int,
    target: int,
) -> int:
    """Generate random crop augmentations. Returns final counter value."""
    counter = counter_start
    generated = 0
    attempts_per_image = CROPS_PER_IMAGE_MAX * 3  # allow retries

    # Shuffle so we spread across all images
    shuffled = list(image_paths)
    random.shuffle(shuffled)

    for img_path in shuffled:
        if generated >= target:
            break

        label_path = LABELS_DIR / (img_path.stem + ".txt")
        src_boxes = read_yolo_labels(label_path)
        if len(src_boxes) < MIN_BOXES_IN_CROP:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] Cannot read {img_path.name}")
            continue
        img_h, img_w = img.shape[:2]

        n_crops = random.randint(CROPS_PER_IMAGE_MIN, CROPS_PER_IMAGE_MAX)
        crops_done = 0

        for _ in range(attempts_per_image):
            if crops_done >= n_crops or generated >= target:
                break

            scale = random.uniform(CROP_SCALE_MIN, CROP_SCALE_MAX)
            crop_w = int(img_w * scale)
            crop_h = int(img_h * scale)

            max_x = img_w - crop_w
            max_y = img_h - crop_h
            if max_x < 1 or max_y < 1:
                continue

            crop_x1 = random.randint(0, max_x)
            crop_y1 = random.randint(0, max_y)
            crop_x2 = crop_x1 + crop_w
            crop_y2 = crop_y1 + crop_h

            complete = count_complete_boxes(
                src_boxes, crop_x1, crop_y1, crop_x2, crop_y2, img_w, img_h
            )
            if complete < MIN_BOXES_IN_CROP:
                continue

            new_boxes = crop_boxes(
                src_boxes, crop_x1, crop_y1, crop_x2, crop_y2, img_w, img_h
            )
            if len(new_boxes) < MIN_BOXES_IN_CROP:
                continue

            stem = f"img_{counter:05d}_crop_{crops_done}"
            out_img_path = IMAGES_DIR / f"{stem}.jpg"
            out_lbl_path = LABELS_DIR / f"{stem}.txt"

            if out_img_path.exists() and out_lbl_path.exists():
                counter += 1
                crops_done += 1
                generated += 1
                continue

            cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
            cv2.imwrite(str(out_img_path), cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
            write_yolo_labels(out_lbl_path, new_boxes)

            counter += 1
            crops_done += 1
            generated += 1

        if crops_done > 0:
            print(f"  Crop: {img_path.name} -> {crops_done} crops, boxes kept range")

    print(f"  Total crops generated: {generated}")
    return counter


# ---------------------------------------------------------------------------
# Type 2 — 2x2 mosaic
# ---------------------------------------------------------------------------


def build_mosaic(
    quad_paths: list[Path],
    tile_w: int, tile_h: int,
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]] | None:
    """
    Build a 2x2 mosaic image and its merged YOLO annotations.
    Returns (mosaic_image, boxes) or None on failure.
    """
    mosaic_w = tile_w * 2
    mosaic_h = tile_h * 2
    mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)
    all_boxes: list[tuple[int, float, float, float, float]] = []

    positions = [
        (0, 0),            # top-left
        (tile_w, 0),       # top-right
        (0, tile_h),       # bottom-left
        (tile_w, tile_h),  # bottom-right
    ]

    for idx, img_path in enumerate(quad_paths):
        label_path = LABELS_DIR / (img_path.stem + ".txt")
        src_boxes = read_yolo_labels(label_path)

        img = cv2.imread(str(img_path))
        if img is None:
            return None

        tile = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
        x_off, y_off = positions[idx]
        mosaic[y_off:y_off + tile_h, x_off:x_off + tile_w] = tile

        # Transform boxes from original image -> tile (scaled) -> mosaic (offset)
        orig_h, orig_w = img.shape[:2]
        for cls, cx, cy, bw, bh in src_boxes:
            # Convert to pixel coords in original image
            bpx1, bpy1, bpx2, bpy2 = yolo_to_pixel(cx, cy, bw, bh, orig_w, orig_h)

            # Scale to tile size (pixel-perfect)
            sx = tile_w / orig_w
            sy = tile_h / orig_h
            tpx1 = bpx1 * sx
            tpy1 = bpy1 * sy
            tpx2 = bpx2 * sx
            tpy2 = bpy2 * sy

            # Add mosaic offset (pixels)
            mpx1 = tpx1 + x_off
            mpy1 = tpy1 + y_off
            mpx2 = tpx2 + x_off
            mpy2 = tpy2 + y_off

            # Clamp to mosaic boundaries (pixel)
            mpx1 = max(0.0, min(mpx1, float(mosaic_w)))
            mpy1 = max(0.0, min(mpy1, float(mosaic_h)))
            mpx2 = max(0.0, min(mpx2, float(mosaic_w)))
            mpy2 = max(0.0, min(mpy2, float(mosaic_h)))

            if mpx2 <= mpx1 or mpy2 <= mpy1:
                continue

            # Normalize to mosaic dimensions
            ncx, ncy, nbw, nbh = pixel_to_yolo(mpx1, mpy1, mpx2, mpy2, mosaic_w, mosaic_h)
            ncx = clamp(ncx)
            ncy = clamp(ncy)
            nbw = clamp(nbw)
            nbh = clamp(nbh)

            if not validate_box(ncx, ncy, nbw, nbh):
                continue

            all_boxes.append((cls, ncx, ncy, nbw, nbh))

    return mosaic, all_boxes


def generate_mosaics(
    image_paths: list[Path],
    counter_start: int,
    target: int,
) -> int:
    """Generate mosaic augmentations. Returns final counter value."""
    counter = counter_start
    generated = 0

    tile_w = MOSAIC_TARGET_W // 2
    tile_h = MOSAIC_TARGET_H // 2

    # Need at least 4 images
    if len(image_paths) < 4:
        print("  [WARN] Not enough images for mosaic generation (need >= 4)")
        return counter

    for i in range(target * 3):  # allow extra attempts for failures
        if generated >= target:
            break

        stem = f"img_{counter:05d}_mosaic_0"
        out_img_path = IMAGES_DIR / f"{stem}.jpg"
        out_lbl_path = LABELS_DIR / f"{stem}.txt"

        if out_img_path.exists() and out_lbl_path.exists():
            counter += 1
            generated += 1
            continue

        quad = random.sample(image_paths, 4)
        result = build_mosaic(quad, tile_w, tile_h)
        if result is None:
            continue

        mosaic, boxes = result
        if len(boxes) == 0:
            continue

        cv2.imwrite(str(out_img_path), mosaic, [cv2.IMWRITE_JPEG_QUALITY, 95])
        write_yolo_labels(out_lbl_path, boxes)

        sources = ", ".join(p.name for p in quad)
        print(f"  Mosaic {generated + 1}/{target}: {sources} -> {len(boxes)} boxes")

        counter += 1
        generated += 1

    print(f"  Total mosaics generated: {generated}")
    return counter


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def collect_original_images() -> list[Path]:
    """Return training images that are not augmentation outputs."""
    all_images = sorted(IMAGES_DIR.glob("*.jpg")) + sorted(IMAGES_DIR.glob("*.png"))
    originals = []
    for p in all_images:
        if _AUG_SUFFIX_RE.search(p.stem):
            continue
        originals.append(p)
    return originals


def main() -> None:
    random.seed(config.RANDOM_SEED + 100)

    print("=== augment_cropmix.py ===")
    print(f"Images dir : {IMAGES_DIR}")
    print(f"Labels dir : {LABELS_DIR}")

    if not IMAGES_DIR.exists():
        print(f"[ERROR] Images directory not found: {IMAGES_DIR}")
        sys.exit(1)
    if not LABELS_DIR.exists():
        print(f"[ERROR] Labels directory not found: {LABELS_DIR}")
        sys.exit(1)

    originals = collect_original_images()
    print(f"Original training images found: {len(originals)}")
    if not originals:
        print("[ERROR] No original training images found.")
        sys.exit(1)

    counter = COUNTER_START

    # --- Type 1: Random crops ---
    print(f"\n--- Type 1: Random Crops (target={TARGET_CROPS}) ---")
    counter = generate_crops(originals, counter, TARGET_CROPS)

    # --- Type 2: Mosaics ---
    print(f"\n--- Type 2: Mosaic 2x2 (target={TARGET_MOSAICS}) ---")
    counter = generate_mosaics(originals, counter, TARGET_MOSAICS)

    final_count = counter - COUNTER_START
    print(f"\n=== Done. {final_count} files written (counter={COUNTER_START}..{counter - 1}) ===")


if __name__ == "__main__":
    main()
