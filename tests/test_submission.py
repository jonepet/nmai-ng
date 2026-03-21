"""
Comprehensive pytest tests for the NorgesGruppen competition submission.

Validates:
- Zip structure and file limits
- run.py contract (CLI args, output format)
- Security restrictions (blocked imports, dangerous calls)
- Prediction format (types, ranges, bbox, scores)
"""

import ast
import io
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = config.PROJECT_ROOT
SUBMISSION_DIR = config.SUBMISSION_DIR
RUN_PY = config.SUBMISSION_DIR / "run.py"

# ---------------------------------------------------------------------------
# Constants (mirrors competition rules)
# ---------------------------------------------------------------------------

MAX_FILES = config.SUBMISSION_MAX_FILES
MAX_PY_FILES = config.SUBMISSION_MAX_PY_FILES
MAX_WEIGHT_FILES = config.SUBMISSION_MAX_WEIGHT_FILES
MAX_WEIGHT_SIZE_MB = config.SUBMISSION_MAX_WEIGHT_SIZE_MB
MAX_WEIGHT_SIZE_BYTES = config.SUBMISSION_MAX_WEIGHT_SIZE_MB * 1024 * 1024

ALLOWED_EXTENSIONS = config.SUBMISSION_ALLOWED_EXTENSIONS

WEIGHT_EXTENSIONS = config.SUBMISSION_WEIGHT_EXTENSIONS

BLOCKED_MODULES = config.BLOCKED_IMPORTS

BLOCKED_CALLS = config.BLOCKED_CALLS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def run_py_source() -> str:
    """Read and return the source text of submission/run.py."""
    if not RUN_PY.exists():
        pytest.skip("submission/run.py does not exist — skipping AST tests")
    return RUN_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def run_py_ast(run_py_source: str) -> ast.Module:
    """Parse submission/run.py into an AST."""
    return ast.parse(run_py_source, filename=str(RUN_PY))


@pytest.fixture(scope="session")
def submission_zip_bytes() -> bytes:
    """
    Build an in-memory zip from the submission/ directory and return the raw bytes.
    Files not present in submission/ will simply not appear in the zip.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(SUBMISSION_DIR.rglob("*")):
            if path.is_file():
                arcname = path.relative_to(SUBMISSION_DIR)
                zf.write(path, arcname=arcname)
    return buf.getvalue()


@pytest.fixture(scope="session")
def submission_zip(submission_zip_bytes: bytes) -> zipfile.ZipFile:
    """Return an open ZipFile backed by the in-memory submission zip."""
    return zipfile.ZipFile(io.BytesIO(submission_zip_bytes))


@pytest.fixture(scope="session")
def dummy_image_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Create a temporary directory with 3 solid-color 640x480 JPEG images
    named img_00001.jpg, img_00002.jpg, img_00003.jpg.
    """
    d = tmp_path_factory.mktemp("dummy_images")
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i, color in enumerate(colors, start=1):
        img = Image.new("RGB", (640, 480), color=color)
        img.save(d / f"img_{i:05d}.jpg", format="JPEG")
    return d


@pytest.fixture(scope="session")
def run_py_output(dummy_image_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Any:
    """
    Invoke submission/run.py via subprocess with the dummy image dir and
    return the parsed JSON output.  Skips if run.py is missing.
    """
    if not RUN_PY.exists():
        pytest.skip("submission/run.py does not exist — skipping runtime tests")

    out_dir = tmp_path_factory.mktemp("run_output")
    out_file = out_dir / "predictions.json"

    result = subprocess.run(
        [sys.executable, str(RUN_PY), "--input", str(dummy_image_dir), "--output", str(out_file)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"run.py exited with code {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert out_file.exists(), "run.py did not produce an output file"
    return json.loads(out_file.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Existence
# ---------------------------------------------------------------------------


def test_runpy_exists() -> None:
    """submission/run.py must exist at the root of the submission directory."""
    assert RUN_PY.exists(), f"run.py not found at {RUN_PY}"
    assert RUN_PY.is_file(), f"{RUN_PY} is not a regular file"


# ---------------------------------------------------------------------------
# AST / security
# ---------------------------------------------------------------------------


def _collect_imports(tree: ast.Module) -> list[str]:
    """Return a flat list of all imported top-level module names."""
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Take the root module (e.g. "os.path" → "os")
                names.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module.split(".")[0])
    return names


def test_runpy_no_blocked_imports(run_py_ast: ast.Module) -> None:
    """run.py must not import any blocked module."""
    imported = _collect_imports(run_py_ast)
    violations = [m for m in imported if m in BLOCKED_MODULES]
    assert not violations, (
        f"run.py imports blocked module(s): {violations}"
    )


def _collect_dangerous_calls(tree: ast.Module) -> list[tuple[str, int]]:
    """
    Return (call_name, line_no) for each call to eval/exec/compile/__import__
    and dangerous getattr patterns (getattr with a string literal attribute).
    """
    hits: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Direct name calls: eval(...), exec(...), etc.
        if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
            hits.append((node.func.id, node.lineno))
        # Attribute calls: builtins.eval(...) etc. — check attribute name only
        if isinstance(node.func, ast.Attribute) and node.func.attr in BLOCKED_CALLS:
            hits.append((node.func.attr, node.lineno))
        # getattr with dynamic string: getattr(obj, "eval")
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
            and node.args[1].value in BLOCKED_CALLS
        ):
            hits.append((f"getattr(..., '{node.args[1].value}')", node.lineno))
    return hits


def test_runpy_no_blocked_calls(run_py_ast: ast.Module) -> None:
    """run.py must not call eval(), exec(), compile(), or __import__()."""
    hits = _collect_dangerous_calls(run_py_ast)
    assert not hits, (
        "run.py contains dangerous call(s): "
        + ", ".join(f"{name}() at line {line}" for name, line in hits)
    )


# ---------------------------------------------------------------------------
# Zip structure
# ---------------------------------------------------------------------------


def test_submission_zip_structure(submission_zip: zipfile.ZipFile) -> None:
    """Zip must contain run.py at the archive root."""
    names = submission_zip.namelist()
    assert "run.py" in names, (
        f"run.py not found at zip root. Files present: {names[:20]}"
    )


def test_submission_zip_file_limits(submission_zip: zipfile.ZipFile) -> None:
    """Total file count ≤ 1000, .py files ≤ 10, all extensions must be allowed."""
    names = submission_zip.namelist()

    # Total file count
    assert len(names) <= MAX_FILES, (
        f"Zip contains {len(names)} files; limit is {MAX_FILES}"
    )

    # .py file count
    py_files = [n for n in names if n.endswith(".py")]
    assert len(py_files) <= MAX_PY_FILES, (
        f"Zip contains {len(py_files)} .py files; limit is {MAX_PY_FILES}. "
        f"Files: {py_files}"
    )

    # Only allowed extensions
    disallowed = [
        n for n in names
        if Path(n).suffix.lower() not in ALLOWED_EXTENSIONS and Path(n).suffix != ""
    ]
    assert not disallowed, (
        f"Zip contains files with disallowed extensions: {disallowed}"
    )


def test_submission_zip_weight_size(submission_zip: zipfile.ZipFile) -> None:
    """Weight files ≤ 3 and their total uncompressed size ≤ 420 MB."""
    infos = submission_zip.infolist()
    weight_infos = [i for i in infos if Path(i.filename).suffix.lower() in WEIGHT_EXTENSIONS]

    assert len(weight_infos) <= MAX_WEIGHT_FILES, (
        f"Zip contains {len(weight_infos)} weight files; limit is {MAX_WEIGHT_FILES}. "
        f"Files: {[i.filename for i in weight_infos]}"
    )

    total_bytes = sum(i.file_size for i in weight_infos)
    total_mb = total_bytes / (1024 * 1024)
    assert total_bytes <= MAX_WEIGHT_SIZE_BYTES, (
        f"Total weight file size is {total_mb:.1f} MB; limit is {MAX_WEIGHT_SIZE_MB} MB"
    )


# ---------------------------------------------------------------------------
# Runtime output
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_runpy_produces_valid_output(run_py_output: Any) -> None:
    """run.py must produce a JSON array (list) at the output path."""
    assert isinstance(run_py_output, list), (
        f"Expected a JSON array, got {type(run_py_output).__name__}"
    )


@pytest.mark.slow
def test_prediction_format(run_py_output: Any) -> None:
    """Each prediction must have image_id, category_id, bbox, and score with correct types."""
    assert isinstance(run_py_output, list), "Output must be a JSON array"
    required_keys = {"image_id", "category_id", "bbox", "score"}
    for idx, pred in enumerate(run_py_output):
        assert isinstance(pred, dict), f"Prediction {idx} is not a dict: {pred!r}"
        missing = required_keys - pred.keys()
        assert not missing, f"Prediction {idx} missing keys: {missing}"

        assert isinstance(pred["image_id"], int), (
            f"Prediction {idx}: image_id must be int, got {type(pred['image_id']).__name__}"
        )
        assert isinstance(pred["category_id"], int), (
            f"Prediction {idx}: category_id must be int, got {type(pred['category_id']).__name__}"
        )
        assert isinstance(pred["bbox"], list) and len(pred["bbox"]) == 4, (
            f"Prediction {idx}: bbox must be a list of 4 elements, got {pred['bbox']!r}"
        )
        assert isinstance(pred["score"], (int, float)), (
            f"Prediction {idx}: score must be numeric, got {type(pred['score']).__name__}"
        )


@pytest.mark.slow
def test_image_id_extraction(run_py_output: Any, dummy_image_dir: Path) -> None:
    """
    image_id values in predictions must correspond to filenames in the input dir.
    Filenames are img_NNNNN.jpg, so image_id should equal the parsed integer N.
    """
    assert isinstance(run_py_output, list), "Output must be a JSON array"
    if not run_py_output:
        pytest.skip("run.py produced no predictions — cannot verify image_id extraction")

    expected_ids = set()
    for p in dummy_image_dir.iterdir():
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            stem = p.stem  # e.g. "img_00001"
            digits = "".join(filter(str.isdigit, stem))
            if digits:
                expected_ids.add(int(digits))

    returned_ids = {pred["image_id"] for pred in run_py_output if isinstance(pred, dict)}
    unknown_ids = returned_ids - expected_ids
    assert not unknown_ids, (
        f"Predictions reference image_id(s) not derived from input filenames: {unknown_ids}. "
        f"Expected ids from filenames: {expected_ids}"
    )


@pytest.mark.slow
def test_bbox_format(run_py_output: Any) -> None:
    """Each bbox must be [x, y, w, h] with all values being non-negative floats/ints."""
    assert isinstance(run_py_output, list), "Output must be a JSON array"
    for idx, pred in enumerate(run_py_output):
        if not isinstance(pred, dict) or "bbox" not in pred:
            continue
        bbox = pred["bbox"]
        assert isinstance(bbox, list) and len(bbox) == 4, (
            f"Prediction {idx}: bbox must have exactly 4 elements, got {bbox!r}"
        )
        for j, val in enumerate(bbox):
            assert isinstance(val, (int, float)), (
                f"Prediction {idx}: bbox[{j}] must be numeric, got {type(val).__name__}"
            )
            assert val >= 0, (
                f"Prediction {idx}: bbox[{j}] must be non-negative, got {val}"
            )
        # w and h (indices 2 and 3) must be strictly positive for a valid box
        assert bbox[2] > 0, (
            f"Prediction {idx}: bbox width (bbox[2]) must be positive, got {bbox[2]}"
        )
        assert bbox[3] > 0, (
            f"Prediction {idx}: bbox height (bbox[3]) must be positive, got {bbox[3]}"
        )


@pytest.mark.slow
def test_score_range(run_py_output: Any) -> None:
    """All scores must be in [0.0, 1.0]."""
    assert isinstance(run_py_output, list), "Output must be a JSON array"
    out_of_range = [
        (idx, pred["score"])
        for idx, pred in enumerate(run_py_output)
        if isinstance(pred, dict)
        and "score" in pred
        and isinstance(pred["score"], (int, float))
        and not (0.0 <= pred["score"] <= 1.0)
    ]
    assert not out_of_range, (
        "Predictions have scores outside [0, 1]: "
        + ", ".join(f"prediction[{i}]={s}" for i, s in out_of_range)
    )


@pytest.mark.slow
def test_category_id_range(run_py_output: Any) -> None:
    """category_id must be an integer in [0, 356]."""
    assert isinstance(run_py_output, list), "Output must be a JSON array"
    violations = [
        (idx, pred["category_id"])
        for idx, pred in enumerate(run_py_output)
        if isinstance(pred, dict)
        and "category_id" in pred
        and isinstance(pred["category_id"], int)
        and not (0 <= pred["category_id"] <= 356)
    ]
    assert not violations, (
        "Predictions have category_id outside [0, 356]: "
        + ", ".join(f"prediction[{i}]={c}" for i, c in violations)
    )
