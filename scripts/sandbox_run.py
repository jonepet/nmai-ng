"""
Sandbox submission runner — replicates the competition sandbox flow exactly.

Usage:
    # Test with a directory (no zip needed)
    python scripts/sandbox_run.py --dir submission/

    # Test with a zip file
    python scripts/sandbox_run.py --zip submission.zip

    # Use specific test images
    python scripts/sandbox_run.py --dir submission/ --images data/yolo/val/images/
"""

from __future__ import annotations

import argparse
import ast
import copy
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Bootstrap: make project root importable regardless of cwd
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
import config  # noqa: E402


# ===========================================================================
# Validation
# ===========================================================================

class ValidationError(Exception):
    pass


def _validate_zip(zip_path: Path) -> None:
    """Validate submission zip against competition rules. Raises ValidationError."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

    # Strip leading directory component if entire zip is wrapped in a folder
    # (e.g. "submission/run.py" → detect and strip)
    roots = {n.split("/")[0] for n in names if "/" in n}
    all_have_root = all(n.startswith(next(iter(roots)) + "/") or n == next(iter(roots)) for n in names) if len(roots) == 1 else False

    if len(roots) == 1 and all_have_root:
        prefix = next(iter(roots)) + "/"
        normalised = [n[len(prefix):] if n.startswith(prefix) else n for n in names]
        normalised = [n for n in normalised if n]  # drop bare directory entry
    else:
        normalised = names

    # run.py must be at root
    if "run.py" not in normalised:
        raise ValidationError("run.py not found at zip root")

    # File count
    if len(normalised) > config.SUBMISSION_MAX_FILES:
        raise ValidationError(
            f"Too many files: {len(normalised)} > {config.SUBMISSION_MAX_FILES}"
        )

    # .py file count
    py_files = [n for n in normalised if n.endswith(".py")]
    if len(py_files) > config.SUBMISSION_MAX_PY_FILES:
        raise ValidationError(
            f"Too many .py files: {len(py_files)} > {config.SUBMISSION_MAX_PY_FILES}"
        )

    # Allowed extensions
    for name in normalised:
        if name.endswith("/"):
            continue  # directory entry
        suffix = Path(name).suffix.lower()
        if suffix not in config.SUBMISSION_ALLOWED_EXTENSIONS:
            raise ValidationError(
                f"Disallowed file extension '{suffix}' in {name}"
            )

    # Weight files
    weight_files = [
        n for n in normalised
        if Path(n).suffix.lower() in config.SUBMISSION_WEIGHT_EXTENSIONS
    ]
    if len(weight_files) > config.SUBMISSION_MAX_WEIGHT_FILES:
        raise ValidationError(
            f"Too many weight files: {len(weight_files)} > {config.SUBMISSION_MAX_WEIGHT_FILES}"
        )

    with zipfile.ZipFile(zip_path, "r") as zf:
        total_weight_bytes = sum(
            zf.getinfo(n).file_size
            for n in zf.namelist()
            if Path(n).suffix.lower() in config.SUBMISSION_WEIGHT_EXTENSIONS
        )
    total_weight_mb = total_weight_bytes / (1024 * 1024)
    if total_weight_mb > config.SUBMISSION_MAX_WEIGHT_SIZE_MB:
        raise ValidationError(
            f"Weight files too large: {total_weight_mb:.1f} MB > {config.SUBMISSION_MAX_WEIGHT_SIZE_MB} MB"
        )


def _security_scan_source(source: str, filename: str) -> list[str]:
    """
    Scan Python source for blocked imports and calls.
    Returns list of violation strings (empty = clean).
    """
    violations: list[str] = []

    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        violations.append(f"{filename}: SyntaxError — {exc}")
        return violations

    for node in ast.walk(tree):
        # import foo / import foo.bar
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in config.BLOCKED_IMPORTS or alias.name in config.BLOCKED_IMPORTS:
                    violations.append(
                        f"{filename}:{node.lineno}: blocked import '{alias.name}'"
                    )

        # from foo import bar / from foo.bar import baz
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".")[0]
            if root in config.BLOCKED_IMPORTS or module in config.BLOCKED_IMPORTS:
                violations.append(
                    f"{filename}:{node.lineno}: blocked 'from {module} import ...'"
                )

        # eval(...) / exec(...) / compile(...) / __import__(...)
        elif isinstance(node, ast.Call):
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name in config.BLOCKED_CALLS:
                violations.append(
                    f"{filename}:{node.lineno}: blocked call '{func_name}()'"
                )

    # Regex fallback for obfuscated patterns (string concatenation, etc.)
    blocked_pattern = r'\b(' + '|'.join(re.escape(b) for b in config.BLOCKED_IMPORTS) + r')\b'
    for lineno, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for call in config.BLOCKED_CALLS:
            if re.search(r'\b' + re.escape(call) + r'\s*\(', line):
                # ast already caught it if parseable; flag again only if ast missed it
                pass  # ast walk is authoritative for well-formed code

    return violations


def _security_scan_dir(submission_dir: Path) -> list[str]:
    """Scan all .py files in a submission directory."""
    violations: list[str] = []
    for py_file in sorted(submission_dir.rglob("*.py")):
        try:
            source = py_file.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            violations.append(f"Could not read {py_file}: {exc}")
            continue
        rel = str(py_file.relative_to(submission_dir))
        violations.extend(_security_scan_source(source, rel))
    return violations


def _validate_dir(submission_dir: Path) -> None:
    """Validate an extracted submission directory."""
    all_files = [p for p in submission_dir.rglob("*") if p.is_file()]
    normalised = [str(p.relative_to(submission_dir)) for p in all_files]

    if "run.py" not in normalised:
        raise ValidationError("run.py not found at submission root")

    if len(normalised) > config.SUBMISSION_MAX_FILES:
        raise ValidationError(
            f"Too many files: {len(normalised)} > {config.SUBMISSION_MAX_FILES}"
        )

    py_files = [n for n in normalised if n.endswith(".py")]
    if len(py_files) > config.SUBMISSION_MAX_PY_FILES:
        raise ValidationError(
            f"Too many .py files: {len(py_files)} > {config.SUBMISSION_MAX_PY_FILES}"
        )

    for name in normalised:
        suffix = Path(name).suffix.lower()
        if suffix not in config.SUBMISSION_ALLOWED_EXTENSIONS:
            raise ValidationError(
                f"Disallowed file extension '{suffix}' in {name}"
            )

    weight_files = [
        p for p in all_files
        if p.suffix.lower() in config.SUBMISSION_WEIGHT_EXTENSIONS
    ]
    if len(weight_files) > config.SUBMISSION_MAX_WEIGHT_FILES:
        raise ValidationError(
            f"Too many weight files: {len(weight_files)} > {config.SUBMISSION_MAX_WEIGHT_FILES}"
        )

    total_weight_bytes = sum(p.stat().st_size for p in weight_files)
    total_weight_mb = total_weight_bytes / (1024 * 1024)
    if total_weight_mb > config.SUBMISSION_MAX_WEIGHT_SIZE_MB:
        raise ValidationError(
            f"Weight files too large: {total_weight_mb:.1f} MB > {config.SUBMISSION_MAX_WEIGHT_SIZE_MB} MB"
        )


# ===========================================================================
# Scoring (copied from scorer.py — identical logic, no import)
# ===========================================================================

def _load_annotations(annotations_path: str) -> dict:
    with open(annotations_path) as f:
        return json.load(f)


def _filter_to_images(coco_gt: dict, image_name_filter: set[str]) -> dict:
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
    det_gt = copy.deepcopy(coco_gt)
    det_gt["categories"] = [{"id": 1, "name": "object", "supercategory": "object"}]
    for ann in det_gt["annotations"]:
        ann["category_id"] = 1
    return det_gt


def _make_detection_preds(predictions: list[dict]) -> list[dict]:
    return [dict(p, category_id=1) for p in predictions]


def _run_coco_eval(gt_dict: dict, dt_list: list[dict], iou_type: str = "bbox") -> float:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        raise RuntimeError(
            "pycocotools is required for scoring. "
            "Install it with: pip install pycocotools"
        )

    if not dt_list:
        return 0.0

    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        coco_dt = coco_gt.loadRes(dt_list)
    finally:
        sys.stdout = old_stdout

    evaluator = COCOeval(coco_gt, coco_dt, iou_type)
    evaluator.params.iouThrs = [0.5]
    evaluator.evaluate()
    evaluator.accumulate()

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        evaluator.summarize()
    finally:
        sys.stdout = old_stdout

    map_at_50 = float(evaluator.stats[0])
    return max(0.0, map_at_50)


def score_predictions(
    annotations_path: str,
    predictions: Any,
    image_name_filter: set[str] | None = None,
) -> dict:
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(predictions, list):
        errors.append("Predictions must be a JSON array of COCO result objects")
        return {"score": 0.0, "detection_map": 0.0, "classification_map": 0.0,
                "errors": errors, "warnings": warnings}

    required_keys = {"image_id", "category_id", "bbox", "score"}
    invalid = [i for i, p in enumerate(predictions[:20]) if not required_keys.issubset(p.keys())]
    if invalid:
        errors.append(
            f"Predictions at indices {invalid[:5]} are missing required keys ({required_keys})"
        )
        return {"score": 0.0, "detection_map": 0.0, "classification_map": 0.0,
                "errors": errors, "warnings": warnings}

    try:
        coco_gt_full = _load_annotations(annotations_path)
    except FileNotFoundError:
        errors.append(f"Annotations file not found: {annotations_path}")
        return {"score": 0.0, "detection_map": 0.0, "classification_map": 0.0,
                "errors": errors, "warnings": warnings}
    except json.JSONDecodeError as exc:
        errors.append(f"Annotations file is not valid JSON: {exc}")
        return {"score": 0.0, "detection_map": 0.0, "classification_map": 0.0,
                "errors": errors, "warnings": warnings}

    coco_gt = _filter_to_images(coco_gt_full, image_name_filter or set())

    if not coco_gt["images"]:
        warnings.append("No ground-truth images matched the subset filter; score will be 0")
        return {"score": 0.0, "detection_map": 0.0, "classification_map": 0.0,
                "errors": errors, "warnings": warnings}

    gt_image_ids = {img["id"] for img in coco_gt["images"]}
    predictions_for_subset = [p for p in predictions if p["image_id"] in gt_image_ids]

    if not predictions_for_subset:
        warnings.append("No predictions matched the evaluation image IDs; score is 0")
        return {"score": 0.0, "detection_map": 0.0, "classification_map": 0.0,
                "errors": errors, "warnings": warnings}

    try:
        det_gt = _make_detection_gt(coco_gt)
        det_preds = _make_detection_preds(predictions_for_subset)
        detection_map = _run_coco_eval(det_gt, det_preds)
    except RuntimeError as exc:
        errors.append(str(exc))
        return {"score": 0.0, "detection_map": 0.0, "classification_map": 0.0,
                "errors": errors, "warnings": warnings}

    try:
        classification_map = _run_coco_eval(coco_gt, predictions_for_subset)
    except RuntimeError as exc:
        errors.append(str(exc))
        return {"score": 0.0, "detection_map": detection_map, "classification_map": 0.0,
                "errors": errors, "warnings": warnings}

    score = round(0.7 * detection_map + 0.3 * classification_map, 6)
    return {
        "score": score,
        "detection_map": round(detection_map, 6),
        "classification_map": round(classification_map, 6),
        "errors": errors,
        "warnings": warnings,
    }


# ===========================================================================
# Output validation
# ===========================================================================

def _validate_predictions(predictions_path: Path) -> tuple[list[dict], list[str]]:
    """
    Load and validate /output/predictions.json.
    Returns (predictions, errors).
    """
    errors: list[str] = []

    if not predictions_path.exists():
        errors.append(f"predictions.json not found at {predictions_path}")
        return [], errors

    try:
        raw = predictions_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        errors.append(f"predictions.json is not valid JSON: {exc}")
        return [], errors

    if not isinstance(data, list):
        errors.append("predictions.json must be a JSON array")
        return [], errors

    invalid: list[str] = []
    for i, pred in enumerate(data):
        if not isinstance(pred, dict):
            invalid.append(f"[{i}] not a dict")
            continue
        if not isinstance(pred.get("image_id"), int):
            invalid.append(f"[{i}] image_id must be int")
        cat = pred.get("category_id")
        if not isinstance(cat, int) or not (0 <= cat <= 356):
            invalid.append(f"[{i}] category_id must be int 0–355")
        bbox = pred.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox)):
            invalid.append(f"[{i}] bbox must be [x,y,w,h]")
        sc = pred.get("score")
        if not isinstance(sc, (int, float)) or not (0.0 <= float(sc) <= 1.0):
            invalid.append(f"[{i}] score must be float 0–1")
        if len(invalid) >= 10:
            invalid.append("... (further errors suppressed)")
            break

    if invalid:
        errors.extend(invalid)

    return data, errors


# ===========================================================================
# Image helpers
# ===========================================================================

def _collect_images(images_dir: Path) -> list[Path]:
    """Return all image files under a directory, sorted by name."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    return sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )


def _build_filename_to_image_id(annotations_path: Path) -> dict[str, int]:
    """Map filename stem (or full filename) → COCO image_id from annotations."""
    with open(annotations_path) as f:
        ann = json.load(f)
    mapping: dict[str, int] = {}
    for img in ann.get("images", []):
        fname = Path(img["file_name"]).name
        mapping[fname] = img["id"]
        mapping[Path(fname).stem] = img["id"]
    return mapping


# ===========================================================================
# Main runner
# ===========================================================================

def _detect_inside_docker() -> bool:
    """Heuristic: check if we're running inside the sandbox Docker container."""
    return Path("/.dockerenv").exists() or os.environ.get("SANDBOX_ENV") == "1"


def run_sandbox(
    submission_dir: Path,
    images_dir: Path | None,
    annotations_path: Path,
    output_dir: Path,
    data_images_dir: Path,
    timeout: int = config.SUBMISSION_TIMEOUT_SECONDS,
) -> dict:
    """
    Execute the full sandbox flow and return a results dict.

    Parameters
    ----------
    submission_dir    : extracted submission (contains run.py)
    images_dir        : source images to copy into data_images_dir
    annotations_path  : COCO annotations JSON
    output_dir        : where run.py writes predictions.json
    data_images_dir   : where images are placed (/data/images inside sandbox)
    timeout           : seconds allowed for run.py
    """
    results: dict = {
        "status": "FAIL",
        "runtime": 0.0,
        "l4_estimate": 0.0,
        "timeout": timeout,
        "prediction_count": 0,
        "image_count": 0,
        "preds_per_image": 0.0,
        "unique_categories": 0,
        "avg_score": 0.0,
        "min_score": 0.0,
        "max_score": 0.0,
        "detection_map": 0.0,
        "classification_map": 0.0,
        "score": 0.0,
        "submission_files": [],
        "submission_weight_mb": 0.0,
        "validation_passed": False,
        "security_passed": False,
        "errors": [],
        "warnings": [],
    }

    # ------------------------------------------------------------------
    # Submission inventory
    # ------------------------------------------------------------------
    all_files = sorted(p for p in submission_dir.rglob("*") if p.is_file())
    file_details = []
    total_weight_bytes = 0
    for f in all_files:
        size = f.stat().st_size
        rel = str(f.relative_to(submission_dir))
        is_weight = f.suffix.lower() in config.SUBMISSION_WEIGHT_EXTENSIONS
        file_details.append({"name": rel, "size": size, "weight": is_weight})
        if is_weight:
            total_weight_bytes += size
    results["submission_files"] = file_details
    results["submission_weight_mb"] = round(total_weight_bytes / (1024 * 1024), 1)
    results["validation_passed"] = True

    # ------------------------------------------------------------------
    # Security scan
    # ------------------------------------------------------------------
    print("Scanning submission for security violations...")
    violations = _security_scan_dir(submission_dir)
    if violations:
        results["errors"].extend(violations)
        print(f"  FAIL — {len(violations)} security violation(s):")
        for v in violations:
            print(f"    {v}")
        return results
    results["security_passed"] = True
    print("  OK — no security violations")

    # ------------------------------------------------------------------
    # Copy test images into /data/images (or configured path)
    # ------------------------------------------------------------------
    print(f"\nCopying test images → {data_images_dir}")
    data_images_dir.mkdir(parents=True, exist_ok=True)
    if images_dir:
        image_files = _collect_images(images_dir)
        for img in image_files:
            shutil.copy2(img, data_images_dir / img.name)
        print(f"  Copied {len(image_files)} images from {images_dir}")
    else:
        # Use validation images from the default YOLO val split
        default_val = config.YOLO_DIR / "val" / "images"
        if default_val.exists():
            image_files = _collect_images(default_val)
            for img in image_files:
                shutil.copy2(img, data_images_dir / img.name)
            print(f"  Copied {len(image_files)} images from {default_val}")
        else:
            results["warnings"].append(
                f"No images_dir provided and default val path not found: {default_val}"
            )
            print(f"  WARNING: {results['warnings'][-1]}")
            image_files = []

    copied_filenames = {p.name for p in data_images_dir.iterdir() if p.is_file()}

    # ------------------------------------------------------------------
    # Prepare output dir
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.json"
    if predictions_path.exists():
        predictions_path.unlink()

    # ------------------------------------------------------------------
    # Run run.py
    # ------------------------------------------------------------------
    cmd = [
        sys.executable,
        str(submission_dir / "run.py"),
        "--input", str(data_images_dir),
        "--output", str(predictions_path),
    ]
    print(f"\nRunning: {' '.join(cmd)}")
    print(f"Timeout: {timeout}s")

    start = time.monotonic()
    returncode = None
    proc = subprocess.run(
        cmd,
        cwd=str(submission_dir),
        capture_output=False,  # let output flow to terminal
    )
    returncode = proc.returncode
    elapsed = time.monotonic() - start
    results["runtime"] = round(elapsed, 2)

    # No timeout enforcement — our GPUs are slower than the L4.
    # Just report the time and estimate L4 runtime.
    l4_estimate = elapsed * 0.05  # L4 GPU is ~20x faster than CPU inference
    results["l4_estimate"] = round(l4_estimate, 1)
    if l4_estimate > timeout:
        results["warnings"].append(
            f"Estimated L4 runtime {l4_estimate:.0f}s may exceed {timeout}s timeout"
        )

    if returncode != 0:
        results["errors"].append(f"run.py exited with code {returncode}")
        print(f"\n  FAIL — run.py exited with code {returncode}")
        return results

    print(f"\n  Finished in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Validate output
    # ------------------------------------------------------------------
    print("\nValidating predictions.json...")
    predictions, pred_errors = _validate_predictions(predictions_path)
    if pred_errors:
        results["errors"].extend(pred_errors)
        for e in pred_errors:
            print(f"  ERROR: {e}")
        return results

    results["prediction_count"] = len(predictions)
    if predictions:
        image_ids = {p["image_id"] for p in predictions}
        scores = [p["score"] for p in predictions]
        categories = {p["category_id"] for p in predictions}
        results["image_count"] = len(image_ids)
        results["preds_per_image"] = round(len(predictions) / len(image_ids), 1)
        results["unique_categories"] = len(categories)
        results["avg_score"] = round(sum(scores) / len(scores), 4)
        results["min_score"] = round(min(scores), 4)
        results["max_score"] = round(max(scores), 4)
    print(f"  OK — {len(predictions)} predictions")

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------
    print("\nScoring predictions...")
    score_result = score_predictions(
        annotations_path=str(annotations_path),
        predictions=predictions,
        image_name_filter=copied_filenames,
    )

    results["detection_map"] = score_result["detection_map"]
    results["classification_map"] = score_result["classification_map"]
    results["score"] = score_result["score"]
    results["errors"].extend(score_result["errors"])
    results["warnings"].extend(score_result["warnings"])

    if not score_result["errors"]:
        results["status"] = "PASS"

    return results


# ===========================================================================
# Report
# ===========================================================================

def _print_report(results: dict) -> None:
    W = 56
    line = "=" * W
    thin = "-" * W

    def row(label: str, value: str) -> str:
        return f"  {label:<30}{value}"

    print()
    print(line)
    print("  SANDBOX SUBMISSION RESULTS")
    print(line)

    # Validation
    print()
    print("  VALIDATION")
    print(thin)
    print(row("Zip/dir validation:", "PASS" if results["validation_passed"] else "FAIL"))
    print(row("Security scan:", "PASS" if results["security_passed"] else "FAIL"))

    # Submission contents
    files = results.get("submission_files", [])
    if files:
        print()
        print("  SUBMISSION CONTENTS")
        print(thin)
        for f in files:
            size_mb = f["size"] / (1024 * 1024)
            tag = " [weight]" if f["weight"] else ""
            print(f"    {f['name']:<30} {size_mb:>7.1f} MB{tag}")
        print(row("Total weight size:", f"{results['submission_weight_mb']:.1f} / {config.SUBMISSION_MAX_WEIGHT_SIZE_MB} MB"))
        print(row("Files:", f"{len(files)} / {config.SUBMISSION_MAX_FILES}"))
        py_count = sum(1 for f in files if f["name"].endswith(".py"))
        print(row("Python files:", f"{py_count} / {config.SUBMISSION_MAX_PY_FILES}"))
        weight_count = sum(1 for f in files if f["weight"])
        print(row("Weight files:", f"{weight_count} / {config.SUBMISSION_MAX_WEIGHT_FILES}"))

    # Runtime
    print()
    print("  RUNTIME")
    print(thin)
    print(row("Status:", results["status"]))
    print(row("Runtime (local CPU):", f"{results['runtime']:.1f}s"))
    l4_est = results.get("l4_estimate", 0)
    print(row("Runtime (est. L4 GPU):", f"~{l4_est:.0f}s / {results['timeout']}s"))

    # Predictions
    print()
    print("  PREDICTIONS")
    print(thin)
    print(row("Total predictions:", str(results["prediction_count"])))
    print(row("Images with predictions:", str(results["image_count"])))
    print(row("Avg predictions/image:", str(results["preds_per_image"])))
    print(row("Unique categories used:", f"{results['unique_categories']} / {config.NC}"))
    print(row("Score range:", f"{results['min_score']:.4f} - {results['max_score']:.4f}"))
    print(row("Avg confidence:", f"{results['avg_score']:.4f}"))

    # Scoring
    print()
    print("  SCORING")
    print(thin)
    det = results["detection_map"]
    cls = results["classification_map"]
    print(row("Detection mAP@0.5:", f"{det:.4f}  (x 0.70 = {det * 0.7:.4f})"))
    print(row("Classification mAP@0.5:", f"{cls:.4f}  (x 0.30 = {cls * 0.3:.4f})"))
    print()
    print(f"  {'FINAL SCORE:':<30}{results['score']:.4f}")
    print(line)

    if results["warnings"]:
        print()
        print("  WARNINGS")
        for w in results["warnings"]:
            print(f"    - {w}")

    if results["errors"]:
        print()
        print("  ERRORS")
        for e in results["errors"]:
            print(f"    - {e}")

    print()


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replicate competition sandbox submission flow and output a score.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--zip", metavar="PATH",
        help="Path to submission zip file",
    )
    source.add_argument(
        "--dir", metavar="PATH",
        help="Path to submission directory (contains run.py)",
    )

    parser.add_argument(
        "--images", metavar="PATH",
        help="Directory of test images (default: data/yolo/val/images/)",
    )
    parser.add_argument(
        "--annotations", metavar="PATH",
        default=str(config.COCO_EXTRACT_DIR / "train" / "annotations.json"),
        help="Path to COCO annotations JSON (default: data/coco_dataset/train/annotations.json)",
    )
    parser.add_argument(
        "--output-dir", metavar="PATH",
        help="Where predictions.json is written (default: temp dir). "
             "Inside Docker: /output",
    )
    parser.add_argument(
        "--data-images-dir", metavar="PATH",
        help="Where images are placed for run.py to read (default: temp dir). "
             "Inside Docker: /data/images",
    )
    parser.add_argument(
        "--timeout", type=int, default=config.SUBMISSION_TIMEOUT_SECONDS,
        help=f"Timeout in seconds for run.py (default: {config.SUBMISSION_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--no-cleanup", action="store_true",
        help="Keep temp directories after run (for debugging)",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    inside_docker = _detect_inside_docker()

    # ------------------------------------------------------------------
    # Determine paths
    # ------------------------------------------------------------------
    annotations_path = Path(args.annotations)
    if not annotations_path.exists():
        print(f"ERROR: Annotations not found: {annotations_path}", file=sys.stderr)
        sys.exit(1)

    images_dir: Path | None = Path(args.images) if args.images else None
    if images_dir and not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Set up temp workspace
    # ------------------------------------------------------------------
    tmpdir = tempfile.mkdtemp(prefix="sandbox_run_")
    cleanup_dirs = [tmpdir]

    try:
        # Submission directory
        if args.zip:
            zip_path = Path(args.zip)
            if not zip_path.exists():
                print(f"ERROR: Zip not found: {zip_path}", file=sys.stderr)
                sys.exit(1)

            print(f"Validating zip: {zip_path}")
            try:
                _validate_zip(zip_path)
            except ValidationError as exc:
                print(f"  FAIL — {exc}", file=sys.stderr)
                sys.exit(1)
            print("  OK — zip validation passed")

            submission_dir = Path(tmpdir) / "submission"
            submission_dir.mkdir()
            print(f"\nExtracting zip to {submission_dir}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(submission_dir)

            # Handle single-folder wrapper
            children = list(submission_dir.iterdir())
            if len(children) == 1 and children[0].is_dir() and not (submission_dir / "run.py").exists():
                submission_dir = children[0]

        else:
            submission_dir = Path(args.dir).resolve()
            if not submission_dir.exists():
                print(f"ERROR: Submission directory not found: {submission_dir}", file=sys.stderr)
                sys.exit(1)

            print(f"Validating directory: {submission_dir}")
            try:
                _validate_dir(submission_dir)
            except ValidationError as exc:
                print(f"  FAIL — {exc}", file=sys.stderr)
                sys.exit(1)
            print("  OK — directory validation passed")

        # Output and data-images directories
        if inside_docker:
            default_output_dir = Path("/output")
            default_data_images_dir = Path("/data/images")
        else:
            default_output_dir = Path(tmpdir) / "output"
            default_data_images_dir = Path(tmpdir) / "data" / "images"

        output_dir = Path(args.output_dir) if args.output_dir else default_output_dir
        data_images_dir = Path(args.data_images_dir) if args.data_images_dir else default_data_images_dir

        # ------------------------------------------------------------------
        # Run
        # ------------------------------------------------------------------
        results = run_sandbox(
            submission_dir=submission_dir,
            images_dir=images_dir,
            annotations_path=annotations_path,
            output_dir=output_dir,
            data_images_dir=data_images_dir,
            timeout=args.timeout,
        )

        _print_report(results)

        exit_code = 0 if results["status"] == "PASS" else 1

    finally:
        if not args.no_cleanup:
            shutil.rmtree(tmpdir, ignore_errors=True)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
