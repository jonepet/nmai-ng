"""
Visualize predictions on images. Draws predicted bounding boxes
and compares with ground truth if available.

Usage:
    python scripts/visualize_predictions.py \
        --predictions /tmp/sandbox_output/predictions.json \
        --images /path/to/images \
        --output /tmp/viz \
        [--annotations data/coco_dataset/train/annotations.json] \
        [--max-images 5]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def draw_predictions(img, preds, color="red", label_prefix=""):
    draw = ImageDraw.Draw(img)
    for p in preds:
        x, y, w, h = p["bbox"]
        cat = p.get("category_id", "?")
        score = p.get("score", 0)
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        draw.text((x, y - 12), f"{label_prefix}{cat}:{score:.2f}", fill=color)
    return img


def draw_gt(img, annotations, color="green"):
    draw = ImageDraw.Draw(img)
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        cat = ann["category_id"]
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        draw.text((x, y - 12), f"GT:{cat}", fill=color)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--max-images", type=int, default=5)
    args = parser.parse_args()

    with open(args.predictions) as f:
        predictions = json.load(f)

    # Group predictions by image_id
    preds_by_id = defaultdict(list)
    for p in predictions:
        preds_by_id[p["image_id"]].append(p)

    # Load ground truth if available
    gt_by_id = defaultdict(list)
    img_id_to_filename = {}
    if args.annotations:
        with open(args.annotations) as f:
            coco = json.load(f)
        for img in coco["images"]:
            img_id_to_filename[img["id"]] = img["file_name"]
        for ann in coco["annotations"]:
            gt_by_id[ann["image_id"]].append(ann)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = Path(args.images)
    image_files = sorted(images_dir.glob("*.jpg"))[:args.max_images]

    for img_path in image_files:
        # Extract image_id from filename
        import re
        digits = re.findall(r"\d+", img_path.stem)
        image_id = int(digits[-1]) if digits else 0

        img = Image.open(img_path).convert("RGB")

        preds = preds_by_id.get(image_id, [])
        gt = gt_by_id.get(image_id, [])

        # Draw GT first (green), then predictions (red)
        if gt:
            img = draw_gt(img, gt, color="green")
        img = draw_predictions(img, preds, color="red", label_prefix="P:")

        out_path = output_dir / f"viz_{img_path.name}"
        img.save(out_path)

        print(f"  {img_path.name}: {len(preds)} preds, {len(gt)} gt → {out_path.name}")
        print(f"    Pred scores: {[p['score'] for p in preds[:5]]}...")
        if preds:
            print(f"    Pred bbox example: {preds[0]['bbox']}")
        if gt:
            print(f"    GT bbox example: {gt[0]['bbox']}")

    print(f"\nSaved {len(image_files)} visualizations to {output_dir}")


if __name__ == "__main__":
    main()
