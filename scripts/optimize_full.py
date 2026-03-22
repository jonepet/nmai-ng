"""
Full parameter sweep: conf_threshold × NMS type × NMS IoU × classifier on/off.
Finds the combination that maximizes competition score.
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


def run_eval(model, val_images, coco_gt, conf, iou, max_det, augment=False):
    predictions = []
    for img_path in val_images:
        image_id = extract_image_id(img_path.stem)
        results = model(str(img_path), device="cpu", verbose=False,
                       conf=conf, iou=iou, max_det=max_det, augment=augment)
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

    det_gt = copy.deepcopy(coco_gt)
    det_gt["categories"] = [{"id": 1, "name": "object"}]
    for ann in det_gt["annotations"]:
        ann["category_id"] = 1
    det_preds = [dict(p, category_id=1) for p in predictions]

    det_map = _coco_eval(det_gt, det_preds)
    cls_map = _coco_eval(coco_gt, predictions)
    score = 0.7 * det_map + 0.3 * cls_map
    return score, det_map, cls_map, len(predictions)


def main():
    checkpoint = config.CHECKPOINT_ROOT / "best_final.pt"
    model = YOLO(str(checkpoint))

    val_dir = config.YOLO_DIR / "val" / "images"
    val_images = sorted(val_dir.glob("*.jpg"))

    ann_path = config.COCO_EXTRACT_DIR / "train" / "annotations.json"
    with open(ann_path) as f:
        coco_full = json.load(f)

    val_names = {p.name for p in val_images}
    kept_imgs = [img for img in coco_full["images"] if Path(img["file_name"]).name in val_names]
    kept_ids = {img["id"] for img in kept_imgs}
    kept_anns = [a for a in coco_full["annotations"] if a["image_id"] in kept_ids]
    coco_gt = {"images": kept_imgs, "annotations": kept_anns, "categories": coco_full["categories"]}

    print(f"Checkpoint: {checkpoint.name}")
    print(f"Val images: {len(val_images)}, GT annotations: {len(kept_anns)}")
    print()

    best_score = 0
    best_params = {}

    # Test ultralytics built-in augment (TTA)
    configs = [
        {"conf": 0.01, "iou": 0.5, "max_det": 300, "augment": False},
        {"conf": 0.01, "iou": 0.6, "max_det": 300, "augment": False},
        {"conf": 0.01, "iou": 0.7, "max_det": 300, "augment": False},
        {"conf": 0.05, "iou": 0.5, "max_det": 300, "augment": False},
        {"conf": 0.05, "iou": 0.6, "max_det": 300, "augment": False},
        {"conf": 0.05, "iou": 0.7, "max_det": 300, "augment": False},
        {"conf": 0.1, "iou": 0.6, "max_det": 300, "augment": False},
        {"conf": 0.25, "iou": 0.7, "max_det": 300, "augment": False},
        {"conf": 0.05, "iou": 0.6, "max_det": 500, "augment": False},
        # With ultralytics TTA
        {"conf": 0.05, "iou": 0.6, "max_det": 300, "augment": True},
        {"conf": 0.01, "iou": 0.6, "max_det": 300, "augment": True},
    ]

    for c in configs:
        t0 = time.time()
        score, det, cls, n = run_eval(model, val_images, coco_gt, **c)
        elapsed = time.time() - t0
        marker = " <<<" if score > best_score else ""
        print(f"  conf={c['conf']:.2f} iou={c['iou']:.1f} max={c['max_det']} aug={c['augment']}: "
              f"det={det:.3f} cls={cls:.3f} score={score:.3f} n={n} ({elapsed:.0f}s){marker}")
        if score > best_score:
            best_score = score
            best_params = c

    print(f"\nBEST: {best_params} → score={best_score:.3f}")


if __name__ == "__main__":
    main()
