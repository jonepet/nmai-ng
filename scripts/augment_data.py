"""
Offline data augmentation for YOLO training set.

Loads training images and their YOLO labels, applies multiple augmentation
pipelines (one copy per pipeline per image), and saves the results alongside
the originals. The validation set is never touched.

Re-runnable: skips augmented files that already exist.
"""

import json
import sys
from pathlib import Path

import cv2
import albumentations as A
from albumentations.core.composition import BboxParams

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

IMAGES_DIR = config.YOLO_DIR / "train" / "images"
LABELS_DIR = config.YOLO_DIR / "train" / "labels"
ANNOTATIONS_FILE = config.COCO_EXTRACT_DIR / "train" / "annotations.json"

# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------

BBOX_PARAMS = BboxParams(
    format="yolo",
    min_visibility=0.3,
    min_area=100,
    label_fields=["class_ids"],
)


def build_pipelines() -> dict[str, A.BasicTransform]:
    """Return a dict of named augmentation pipelines."""
    return {
        "bright": A.Compose(
            [A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0)],
            bbox_params=BBOX_PARAMS,
        ),
        "dark": A.Compose(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.4, -0.1),
                    contrast_limit=(-0.3, 0.0),
                    p=1.0,
                )
            ],
            bbox_params=BBOX_PARAMS,
        ),
        "color": A.Compose(
            [A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.1, p=1.0)],
            bbox_params=BBOX_PARAMS,
        ),
        "perspective": A.Compose(
            [
                A.Perspective(scale=(0.02, 0.06), p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            ],
            bbox_params=BBOX_PARAMS,
        ),
        "combined": A.Compose(
            [
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.GaussNoise(var_limit=(10, 40), p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.2, hue=0.05, p=0.5),
                A.ImageCompression(quality_lower=40, quality_upper=70, p=0.3),
            ],
            bbox_params=BBOX_PARAMS,
        ),
    }


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------


def read_yolo_label(label_path: Path) -> list[list[float]]:
    """Return list of [class_id, x_center, y_center, width, height]."""
    if not label_path.exists():
        return []
    rows = []
    with label_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            rows.append([float(p) for p in parts])
    return rows


def write_yolo_label(label_path: Path, rows: list[list[float]]) -> None:
    """Write YOLO-format label rows to file."""
    with label_path.open("w") as fh:
        for row in rows:
            cls = int(row[0])
            coords = row[1:]
            fh.write(f"{cls} {' '.join(f'{v:.6f}' for v in coords)}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Verify directories exist
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Training images directory not found: {IMAGES_DIR}")
    if not LABELS_DIR.exists():
        raise FileNotFoundError(f"Training labels directory not found: {LABELS_DIR}")

    # Collect original training images (exclude already-augmented files)
    pipelines = build_pipelines()
    pipeline_names = set(pipelines.keys())

    image_paths: list[Path] = []
    for p in sorted(IMAGES_DIR.glob("*.jpg")):
        # Exclude augmented copies (stem ends with _<pipeline_name>)
        stem_parts = p.stem.rsplit("_", 1)
        if len(stem_parts) == 2 and stem_parts[1] in pipeline_names:
            continue
        image_paths.append(p)

    # Also check .png originals
    for p in sorted(IMAGES_DIR.glob("*.png")):
        stem_parts = p.stem.rsplit("_", 1)
        if len(stem_parts) == 2 and stem_parts[1] in pipeline_names:
            continue
        image_paths.append(p)

    n_originals = len(image_paths)
    n_pipelines = len(pipelines)
    n_created = 0

    print(f"Found {n_originals} original training images, {n_pipelines} augmentation pipelines.")
    print(f"Output directories:\n  images → {IMAGES_DIR}\n  labels → {LABELS_DIR}")

    global_idx = 0
    for img_path in image_paths:
        stem = img_path.stem
        suffix = img_path.suffix  # preserve original extension

        label_path = LABELS_DIR / (stem + ".txt")
        label_rows = read_yolo_label(label_path)

        if not label_rows:
            # No annotations — skip augmentation for this image
            global_idx += 1
            continue

        # Load image once; convert BGR→RGB for albumentations
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  [WARN] Could not read image: {img_path}")
            global_idx += 1
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Separate class ids from bbox coords for albumentations API
        class_ids = [int(r[0]) for r in label_rows]
        bboxes = [r[1:] for r in label_rows]  # [x_center, y_center, w, h]

        for pipeline_name, transform in pipelines.items():
            global_idx += 1
            out_img_path = IMAGES_DIR / f"{stem}_{pipeline_name}{suffix}"
            out_lbl_path = LABELS_DIR / f"{stem}_{pipeline_name}.txt"

            print(
                f"Augmenting image {global_idx}/{n_originals * n_pipelines}"
                f" (pipeline: {pipeline_name})..."
            )

            # Idempotent — skip if both files already exist
            if out_img_path.exists() and out_lbl_path.exists():
                continue

            try:
                result = transform(image=rgb, bboxes=bboxes, class_ids=class_ids)
            except Exception as exc:  # noqa: BLE001
                print(f"  [WARN] Transform '{pipeline_name}' failed for {stem}: {exc}")
                continue

            aug_rgb = result["image"]
            aug_bboxes = result["bboxes"]        # list of (x_center, y_center, w, h)
            aug_class_ids = result["class_ids"]  # list of int

            # Convert RGB→BGR for cv2 saving
            aug_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_img_path), aug_bgr)

            # Reconstruct label rows
            aug_rows = [
                [float(cls), *bbox]
                for cls, bbox in zip(aug_class_ids, aug_bboxes)
            ]
            write_yolo_label(out_lbl_path, aug_rows)
            n_created += 1

    print(
        f"\nCreated {n_created} augmented images from {n_originals} originals"
        f" ({n_pipelines} pipelines)"
    )


if __name__ == "__main__":
    main()
