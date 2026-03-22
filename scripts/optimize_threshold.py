"""
Quick parameter sweep to find optimal conf_threshold.
Uses ultralytics .pt model on CPU (fast, ~30s per threshold on val set).
"""
import json
import sys
import copy
import io
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import re


def extract_image_id(stem):
    digits = re.findall(r"\d+", stem)
    return int(digits[-1]) if digits else 0


def run_eval(model, val_images, coco_gt, conf_threshold):
    predictions = []
    for img_path in val_images:
        image_id = extract_image_id(img_path.stem)
        results = model(str(img_path), device="cpu", verbose=False,
                       conf=conf_threshold, iou=0.5, max_det=300)
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xywh = boxes.xywh.cpu()
            scores = boxes.conf.cpu()
            class_ids = boxes.cls.cpu().int()
            for i in range(len(boxes)):
                xc, yc, w, h = xywh[i].tolist()
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(class_ids[i].item()),
                    "bbox": [xc - w/2, yc - h/2, w, h],
                    "score": float(scores[i].item()),
                })

    if not predictions:
        return 0, 0, 0, 0

    # Detection mAP
    det_gt = copy.deepcopy(coco_gt)
    det_gt["categories"] = [{"id": 1, "name": "object"}]
    for ann in det_gt["annotations"]:
        ann["category_id"] = 1
    det_preds = [dict(p, category_id=1) for p in predictions]

    det_map = _coco_eval(det_gt, det_preds)
    cls_map = _coco_eval(coco_gt, predictions)
    score = 0.7 * det_map + 0.3 * cls_map
    return score, det_map, cls_map, len(predictions)


def _coco_eval(gt_dict, dt_list):
    if not dt_list:
        return 0.0
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dt_list)
    finally:
        sys.stdout = old
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.iouThrs = [0.5]
    ev.evaluate()
    ev.accumulate()
    sys.stdout = io.StringIO()
    try:
        ev.summarize()
    finally:
        sys.stdout = old
    return max(0.0, float(ev.stats[0]))


def main():
    checkpoint = config.CHECKPOINT_ROOT / "best_final.pt"
    if not checkpoint.exists():
        print("No best_final.pt")
        sys.exit(1)

    model = YOLO(str(checkpoint))

    # Val images
    val_dir = config.YOLO_DIR / "val" / "images"
    val_images = sorted(val_dir.glob("*.jpg"))
    if not val_images:
        print("No val images")
        sys.exit(1)

    # Load GT
    ann_path = config.COCO_EXTRACT_DIR / "train" / "annotations.json"
    with open(ann_path) as f:
        coco_full = json.load(f)

    # Filter GT to val images
    val_names = {p.name for p in val_images}
    kept_imgs = [img for img in coco_full["images"] if Path(img["file_name"]).name in val_names]
    kept_ids = {img["id"] for img in kept_imgs}
    kept_anns = [a for a in coco_full["annotations"] if a["image_id"] in kept_ids]
    coco_gt = {
        "images": kept_imgs,
        "annotations": kept_anns,
        "categories": coco_full["categories"],
    }

    print(f"Checkpoint: {checkpoint.name}")
    print(f"Val images: {len(val_images)}")
    print(f"GT annotations: {len(kept_anns)}")
    print()

    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    best_score = 0
    best_conf = 0.25

    for conf in thresholds:
        t0 = time.time()
        score, det, cls, n_preds = run_eval(model, val_images, coco_gt, conf)
        elapsed = time.time() - t0
        marker = " <<<" if score > best_score else ""
        print(f"  conf={conf:.2f}: det={det:.3f} cls={cls:.3f} score={score:.3f} "
              f"preds={n_preds} ({elapsed:.0f}s){marker}")
        if score > best_score:
            best_score = score
            best_conf = conf

    print(f"\nBest: conf={best_conf:.2f} score={best_score:.3f}")


if __name__ == "__main__":
    main()
