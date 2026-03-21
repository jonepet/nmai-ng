"""
COCO mAP scoring for the NorgesGruppen competition.

Scoring formula:
  - 70%  detection mAP  : IoU >= 0.5, category ignored (any category_id counts)
  - 30%  classification mAP : IoU >= 0.5 AND correct category_id
  - Final score = 0.7 * detection_mAP + 0.3 * classification_mAP
"""

import copy
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

def _load_annotations(annotations_path: str) -> dict:
    with open(annotations_path) as f:
        return json.load(f)


def _filter_to_images(coco_gt: dict, image_name_filter: set[str]) -> dict:
    """
    Return a copy of coco_gt restricted to the images whose file_name appears
    in image_name_filter.  If image_name_filter is empty, return the full dataset.
    """
    if not image_name_filter:
        return coco_gt

    kept_images = [
        img for img in coco_gt["images"]
        if Path(img["file_name"]).name in image_name_filter
    ]
    kept_ids = {img["id"] for img in kept_images}
    kept_annotations = [a for a in coco_gt["annotations"] if a["image_id"] in kept_ids]

    filtered = copy.deepcopy(coco_gt)
    filtered["images"] = kept_images
    filtered["annotations"] = kept_annotations
    return filtered


def _make_detection_gt(coco_gt: dict) -> dict:
    """
    Collapse all categories into a single 'object' category so that any
    correct bounding-box localisation counts as a true positive regardless
    of the predicted category.
    """
    det_gt = copy.deepcopy(coco_gt)
    det_gt["categories"] = [{"id": 1, "name": "object", "supercategory": "object"}]
    for ann in det_gt["annotations"]:
        ann["category_id"] = 1
    return det_gt


def _make_detection_preds(predictions: list[dict]) -> list[dict]:
    """Remap every prediction's category_id to 1 for detection evaluation."""
    return [dict(p, category_id=1) for p in predictions]


def _run_coco_eval(gt_dict: dict, dt_list: list[dict], iou_type: str = "bbox") -> float:
    """
    Run COCOeval and return mAP@0.5 (area=all, maxDets=100).

    Returns 0.0 if pycocotools is unavailable or there are no predictions.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        # pycocotools not installed — return sentinel
        raise RuntimeError(
            "pycocotools is required for scoring. "
            "Install it with: pip install pycocotools"
        )

    import io
    import sys

    if not dt_list:
        return 0.0

    # Build COCO ground-truth object from dict (avoids writing to disk)
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()

    # Suppress stdout noise from loadRes
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        coco_dt = coco_gt.loadRes(dt_list)
    finally:
        sys.stdout = old_stdout

    evaluator = COCOeval(coco_gt, coco_dt, iou_type)
    # Restrict to IoU threshold 0.5 only
    evaluator.params.iouThrs = [0.5]
    evaluator.evaluate()
    evaluator.accumulate()

    # Suppress summary print
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        evaluator.summarize()
    finally:
        sys.stdout = old_stdout

    # stats[0] is AP @ IoU=0.50:0.95, stats[1] is AP @ IoU=0.50
    # With iouThrs=[0.5], stats[0] == stats[1] (both are mAP@0.5)
    map_at_50 = float(evaluator.stats[0])
    return max(0.0, map_at_50)


def score_predictions(
    annotations_path: str,
    predictions: Any,
    image_name_filter: set[str] | None = None,
) -> dict:
    """
    Score a list of COCO-format predictions against ground truth.

    Parameters
    ----------
    annotations_path:
        Path to the COCO annotations JSON file.
    predictions:
        List of dicts with keys: image_id, category_id, bbox, score.
        (Standard COCO results format.)
    image_name_filter:
        If provided, restrict evaluation to images whose file_name basename
        appears in this set.  Pass an empty set or None to evaluate all images.

    Returns
    -------
    dict with keys:
        score              — final weighted score (0–1)
        detection_map      — mAP@0.5 ignoring category (0–1)
        classification_map — mAP@0.5 with correct category required (0–1)
        errors             — list of error strings
        warnings           — list of warning strings
    """
    errors: list[str] = []
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # Validate predictions format
    # ------------------------------------------------------------------
    if not isinstance(predictions, list):
        errors.append(
            "Predictions must be a JSON array of COCO result objects"
        )
        return {
            "score": 0.0,
            "detection_map": 0.0,
            "classification_map": 0.0,
            "errors": errors,
            "warnings": warnings,
        }

    required_keys = {"image_id", "category_id", "bbox", "score"}
    invalid = [
        i for i, p in enumerate(predictions[:20])
        if not required_keys.issubset(p.keys())
    ]
    if invalid:
        errors.append(
            f"Predictions at indices {invalid[:5]} are missing required keys "
            f"({required_keys})"
        )
        return {
            "score": 0.0,
            "detection_map": 0.0,
            "classification_map": 0.0,
            "errors": errors,
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Load ground truth
    # ------------------------------------------------------------------
    try:
        coco_gt_full = _load_annotations(annotations_path)
    except FileNotFoundError:
        errors.append(f"Annotations file not found: {annotations_path}")
        return {
            "score": 0.0,
            "detection_map": 0.0,
            "classification_map": 0.0,
            "errors": errors,
            "warnings": warnings,
        }
    except json.JSONDecodeError as exc:
        errors.append(f"Annotations file is not valid JSON: {exc}")
        return {
            "score": 0.0,
            "detection_map": 0.0,
            "classification_map": 0.0,
            "errors": errors,
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Filter to subset
    # ------------------------------------------------------------------
    coco_gt = _filter_to_images(coco_gt_full, image_name_filter or set())

    if not coco_gt["images"]:
        warnings.append(
            "No ground-truth images matched the subset filter; score will be 0"
        )
        return {
            "score": 0.0,
            "detection_map": 0.0,
            "classification_map": 0.0,
            "errors": errors,
            "warnings": warnings,
        }

    gt_image_ids = {img["id"] for img in coco_gt["images"]}
    predictions_for_subset = [p for p in predictions if p["image_id"] in gt_image_ids]

    if not predictions_for_subset:
        warnings.append(
            "No predictions matched the evaluation image IDs; score is 0"
        )
        return {
            "score": 0.0,
            "detection_map": 0.0,
            "classification_map": 0.0,
            "errors": errors,
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Detection mAP (categories collapsed to 1)
    # ------------------------------------------------------------------
    try:
        det_gt = _make_detection_gt(coco_gt)
        det_preds = _make_detection_preds(predictions_for_subset)
        detection_map = _run_coco_eval(det_gt, det_preds)
    except RuntimeError as exc:
        errors.append(str(exc))
        return {
            "score": 0.0,
            "detection_map": 0.0,
            "classification_map": 0.0,
            "errors": errors,
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Classification mAP (correct category required)
    # ------------------------------------------------------------------
    try:
        classification_map = _run_coco_eval(coco_gt, predictions_for_subset)
    except RuntimeError as exc:
        errors.append(str(exc))
        return {
            "score": 0.0,
            "detection_map": detection_map,
            "classification_map": 0.0,
            "errors": errors,
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Final score
    # ------------------------------------------------------------------
    score = round(0.7 * detection_map + 0.3 * classification_map, 6)

    return {
        "score": score,
        "detection_map": round(detection_map, 6),
        "classification_map": round(classification_map, 6),
        "errors": errors,
        "warnings": warnings,
    }
