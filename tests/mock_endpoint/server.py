"""
Mock competition submission endpoint for local testing.

Mimics the NorgesGruppen competition submission endpoint.
Accepts a zip file, validates structure, runs run.py, and scores predictions.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config

from flask import Flask, jsonify, request
from scorer import score_predictions

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAIN_IMAGES_DIR = config.COCO_EXTRACT_DIR / "train" / "images"
ANNOTATIONS_FILE = config.COCO_EXTRACT_DIR / "train" / "annotations.json"
SUBSET_IMAGES_DIR = config.DATA_DIR / "train" / "images_subset"

# In-memory leaderboard (resets when server restarts)
leaderboard: list[dict] = []
submission_counter = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_image_subset() -> Path:
    """
    Create a subset of the first N training images in a deterministic temp dir.
    Recreates the subset dir on each server start so images are always available.
    """
    subset_dir = SUBSET_IMAGES_DIR
    subset_dir.mkdir(parents=True, exist_ok=True)

    if TRAIN_IMAGES_DIR.exists():
        image_files = sorted(
            p for p in TRAIN_IMAGES_DIR.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )[:config.MOCK_ENDPOINT_IMAGE_SUBSET_COUNT]
        for img in image_files:
            dest = subset_dir / img.name
            if not dest.exists():
                shutil.copy2(img, dest)

    return subset_dir


def _validate_zip(zip_path: Path) -> list[str]:
    """
    Validate the uploaded zip file.

    Returns a list of error strings. Empty list means the zip is valid.
    """
    errors = []

    # Size check
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    if size_mb > config.SUBMISSION_MAX_ZIP_SIZE_MB:
        errors.append(f"Zip exceeds {config.SUBMISSION_MAX_ZIP_SIZE_MB} MB limit ({size_mb:.1f} MB)")

    if not zipfile.is_zipfile(zip_path):
        errors.append("Uploaded file is not a valid zip archive")
        return errors

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

        # File count
        if len(names) > config.SUBMISSION_MAX_FILES:
            errors.append(
                f"Zip contains {len(names)} files; limit is {config.SUBMISSION_MAX_FILES}"
            )

        # run.py at root
        root_files = [n for n in names if "/" not in n.rstrip("/")]
        if "run.py" not in root_files:
            errors.append("run.py must be present at the root of the zip")

        # Allowed extensions
        disallowed = [
            n for n in names
            if not n.endswith("/") and Path(n).suffix.lower() not in config.SUBMISSION_ALLOWED_EXTENSIONS
        ]
        if disallowed:
            errors.append(
                f"Disallowed file types: {', '.join(disallowed[:5])}"
                + (" ..." if len(disallowed) > 5 else "")
            )

        # Total uncompressed size (model weights check)
        total_uncompressed_mb = sum(info.file_size for info in zf.infolist()) / (1024 * 1024)
        if total_uncompressed_mb > config.SUBMISSION_MAX_ZIP_SIZE_MB * 2:
            errors.append(
                f"Uncompressed content exceeds {config.SUBMISSION_MAX_ZIP_SIZE_MB * 2} MB "
                f"({total_uncompressed_mb:.1f} MB)"
            )

    return errors


def _run_submission(extract_dir: Path, output_json: Path) -> tuple[bool, str, list[str]]:
    """
    Run run.py from the extracted submission directory.

    Returns (success, stdout+stderr combined, warnings).
    """
    subset_dir = _prepare_image_subset()
    warnings = []

    if not list(subset_dir.iterdir()):
        warnings.append(
            "No training images found in subset directory; "
            "predictions will be empty and score will be 0"
        )

    cmd = [
        "python",
        "run.py",
        "--input", str(subset_dir),
        "--output", str(output_json),
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(extract_dir),
            capture_output=True,
            text=True,
            timeout=config.SUBMISSION_TIMEOUT_SECONDS,
        )
        combined_output = result.stdout + ("\n" + result.stderr if result.stderr else "")
        if result.returncode != 0:
            return False, combined_output, warnings
        return True, combined_output, warnings
    except subprocess.TimeoutExpired:
        return False, f"Execution timed out after {config.SUBMISSION_TIMEOUT_SECONDS} seconds", warnings
    except FileNotFoundError as exc:
        return False, f"Failed to start process: {exc}", warnings


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/leaderboard", methods=["GET"])
def get_leaderboard():
    sorted_board = sorted(leaderboard, key=lambda x: x["score"], reverse=True)
    return jsonify(sorted_board)


@app.route("/submit", methods=["POST"])
def submit():
    global submission_counter

    # ------------------------------------------------------------------
    # 1. Receive file
    # ------------------------------------------------------------------
    if "file" not in request.files:
        return jsonify({
            "status": "error",
            "errors": ["No file uploaded; send a zip as form field 'file'"],
        }), 400

    uploaded = request.files["file"]
    if not uploaded.filename or not uploaded.filename.endswith(".zip"):
        return jsonify({
            "status": "error",
            "errors": ["Uploaded file must be a .zip archive"],
        }), 400

    team_name = request.form.get("team_name", "anonymous")

    tmp_root = Path(tempfile.mkdtemp(prefix="ngcomp_"))
    try:
        zip_path = tmp_root / "submission.zip"
        uploaded.save(str(zip_path))

        # ------------------------------------------------------------------
        # 2. Validate zip
        # ------------------------------------------------------------------
        errors = _validate_zip(zip_path)
        if errors:
            return jsonify({
                "status": "error",
                "score": None,
                "detection_map": None,
                "classification_map": None,
                "errors": errors,
                "warnings": [],
            }), 422

        # ------------------------------------------------------------------
        # 3. Extract
        # ------------------------------------------------------------------
        extract_dir = tmp_root / "extracted"
        extract_dir.mkdir()
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(extract_dir))

        # ------------------------------------------------------------------
        # 4. Run submission
        # ------------------------------------------------------------------
        output_json = tmp_root / "predictions.json"
        success, run_output, warnings = _run_submission(extract_dir, output_json)

        if not success:
            return jsonify({
                "status": "error",
                "score": None,
                "detection_map": None,
                "classification_map": None,
                "errors": [f"run.py failed: {run_output[:2000]}"],
                "warnings": warnings,
            }), 422

        # ------------------------------------------------------------------
        # 5. Load predictions
        # ------------------------------------------------------------------
        if not output_json.exists():
            return jsonify({
                "status": "error",
                "score": None,
                "detection_map": None,
                "classification_map": None,
                "errors": [
                    f"run.py did not produce output at {output_json}; "
                    "ensure --output path is honoured"
                ],
                "warnings": warnings,
            }), 422

        with open(output_json) as f:
            try:
                predictions = json.load(f)
            except json.JSONDecodeError as exc:
                return jsonify({
                    "status": "error",
                    "errors": [f"Output is not valid JSON: {exc}"],
                    "warnings": warnings,
                }), 422

        # ------------------------------------------------------------------
        # 6. Score
        # ------------------------------------------------------------------
        subset_image_names = {p.name for p in SUBSET_IMAGES_DIR.iterdir()} if SUBSET_IMAGES_DIR.exists() else set()

        score_result = score_predictions(
            annotations_path=str(ANNOTATIONS_FILE),
            predictions=predictions,
            image_name_filter=subset_image_names,
        )

        # ------------------------------------------------------------------
        # 7. Record on leaderboard
        # ------------------------------------------------------------------
        submission_counter += 1
        entry = {
            "rank": None,  # filled after sort on GET /leaderboard
            "submission_id": submission_counter,
            "team_name": team_name,
            "score": score_result["score"],
            "detection_map": score_result["detection_map"],
            "classification_map": score_result["classification_map"],
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
        leaderboard.append(entry)

        return jsonify({
            "status": "ok",
            "score": score_result["score"],
            "detection_map": score_result["detection_map"],
            "classification_map": score_result["classification_map"],
            "errors": score_result.get("errors", []),
            "warnings": warnings + score_result.get("warnings", []),
            "run_output": run_output[:1000] if run_output else "",
        })

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Prepare image subset eagerly so the first submission is fast
    _prepare_image_subset()
    app.run(host="0.0.0.0", port=config.MOCK_ENDPOINT_PORT, debug=False)
