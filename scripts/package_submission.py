#!/usr/bin/env python3
"""Package the submission zip for the NorgesGruppen competition.

Supports ensemble submission: multiple model weight files from config.SUBMISSION_MODEL_FILES.
"""

import argparse
import ast
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

PROJECT_ROOT = config.PROJECT_ROOT
SUBMISSION_DIR = config.SUBMISSION_DIR
CHECKPOINT_ROOT = config.CHECKPOINT_ROOT

MAX_SIZE_BYTES = config.SUBMISSION_MAX_WEIGHT_SIZE_MB * 1024 * 1024
MAX_FILES = config.SUBMISSION_MAX_FILES
MAX_PY_FILES = config.SUBMISSION_MAX_PY_FILES
MAX_WEIGHT_FILES = config.SUBMISSION_MAX_WEIGHT_FILES

BLOCKED_IMPORTS = config.BLOCKED_IMPORTS
WEIGHT_EXTENSIONS = config.SUBMISSION_WEIGHT_EXTENSIONS

EXCLUDE_PATTERNS = {"__pycache__", ".pyc", "__MACOSX", ".DS_Store"}

# Weight sources derived from submission/config.json model_files list
WEIGHT_SOURCES = {name: SUBMISSION_DIR / name for name in config.SUBMISSION_MODEL_FILES}


def should_exclude(path: Path) -> bool:
    for part in path.parts:
        for pattern in EXCLUDE_PATTERNS:
            if pattern in part:
                return True
    return False


def collect_imports(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])
                imported.add(node.module)
    return imported


def scan_blocked_imports(py_file: Path) -> list[str]:
    source = py_file.read_text(encoding="utf-8", errors="replace")
    found_imports = collect_imports(source)
    blocked = []
    for imp in found_imports:
        for blocked_name in BLOCKED_IMPORTS:
            if imp == blocked_name or imp.startswith(blocked_name + "."):
                blocked.append(imp)
    return blocked


def scan_blocked_calls(py_file: Path) -> list[str]:
    """Check for eval(), exec(), compile(), __import__() calls."""
    source = py_file.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [f"SyntaxError in {py_file.name}"]
    blocked = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name in config.BLOCKED_CALLS:
            blocked.append(f"{name}() at line {node.lineno}")
    return blocked


def format_size(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 ** 3):.2f} GB"
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 ** 2):.2f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    return f"{size_bytes} B"


def validate_and_collect() -> list[tuple[Path, str]]:
    """Validate all requirements and return list of (source_path, archive_name)."""
    errors = []
    entries: list[tuple[Path, str]] = []

    # --- run.py ---
    run_py = SUBMISSION_DIR / "run.py"
    if not run_py.exists():
        errors.append(f"run.py not found at {run_py}")
    else:
        entries.append((run_py, "run.py"))
        blocked = scan_blocked_imports(run_py)
        if blocked:
            errors.append(f"Blocked import(s) {blocked} in run.py")
        blocked_calls = scan_blocked_calls(run_py)
        if blocked_calls:
            errors.append(f"Blocked call(s) {blocked_calls} in run.py")

    # --- Additional .py files ---
    extra_py = sorted(
        p for p in SUBMISSION_DIR.rglob("*.py")
        if p.name != "run.py" and not should_exclude(p)
    )
    for py_file in extra_py:
        entries.append((py_file, py_file.name))
        blocked = scan_blocked_imports(py_file)
        if blocked:
            errors.append(f"Blocked import(s) {blocked} in {py_file.name}")
        blocked_calls = scan_blocked_calls(py_file)
        if blocked_calls:
            errors.append(f"Blocked call(s) {blocked_calls} in {py_file.name}")

    py_count = 1 + len(extra_py)
    if py_count > MAX_PY_FILES:
        errors.append(f"Too many .py files: {py_count} (max {MAX_PY_FILES})")

    # --- Weight files (.pt, .npy, etc.) ---
    weight_entries = []
    for zip_name, source_path in WEIGHT_SOURCES.items():
        if source_path.exists():
            weight_entries.append((source_path, zip_name))
            print(f"  Found weight: {source_path} -> {zip_name} ({format_size(source_path.stat().st_size)})")
        else:
            print(f"  Missing weight: {source_path} (skipping {zip_name})")

    # Classifier ONNX (optional)
    import json as _json
    _sub_cfg_path = SUBMISSION_DIR / "config.json"
    _sub_cfg = _json.load(open(_sub_cfg_path)) if _sub_cfg_path.exists() else {}
    classifier_name = _sub_cfg.get("classifier_file")
    if classifier_name:
        cls_path = SUBMISSION_DIR / classifier_name
        if cls_path.exists():
            weight_entries.append((cls_path, classifier_name))
            print(f"  Found weight: {cls_path} -> {classifier_name} ({format_size(cls_path.stat().st_size)})")
        else:
            print(f"  Classifier not found: {cls_path} (optional, skipping)")

    if not weight_entries:
        errors.append("No weight files found. Need at least one of: " +
                       ", ".join(str(v) for v in WEIGHT_SOURCES.values()))

    if len(weight_entries) > MAX_WEIGHT_FILES:
        errors.append(f"Too many weight files: {len(weight_entries)} (max {MAX_WEIGHT_FILES})")

    total_weight_bytes = sum(src.stat().st_size for src, _ in weight_entries)
    if total_weight_bytes > MAX_SIZE_BYTES:
        errors.append(
            f"Total weight size {format_size(total_weight_bytes)} exceeds {format_size(MAX_SIZE_BYTES)}")

    entries.extend(weight_entries)

    # --- config.json — required by run.py ---
    cfg_path = SUBMISSION_DIR / "config.json"
    if cfg_path.exists():
        entries.append((cfg_path, "config.json"))
        print(f"  Found config: {cfg_path} -> config.json ({format_size(cfg_path.stat().st_size)})")
    else:
        errors.append("config.json not found in submission directory")

    # --- Total size check ---
    total_size = sum(src.stat().st_size for src, _ in entries)
    if total_size > MAX_SIZE_BYTES:
        errors.append(f"Total uncompressed size {format_size(total_size)} exceeds {format_size(MAX_SIZE_BYTES)}")

    if len(entries) > MAX_FILES:
        errors.append(f"Too many files: {len(entries)} (max {MAX_FILES})")

    if errors:
        print("\nValidation FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    return entries


def build_zip(entries: list[tuple[Path, str]], output_path: Path) -> None:
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, archive_name in entries:
            zf.write(src, archive_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Package submission zip")
    parser.add_argument("--output", default=None, help="Output zip path")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else PROJECT_ROOT / "submission.zip"

    print(f"Project root   : {PROJECT_ROOT}")
    print(f"Submission dir : {SUBMISSION_DIR}")
    print(f"Checkpoint root: {CHECKPOINT_ROOT}")
    print(f"Expected models: {list(WEIGHT_SOURCES.keys())}")
    print(f"Output path    : {output_path}")
    print()

    entries = validate_and_collect()
    print("\nValidation passed.")

    print("\nBuilding zip...")
    build_zip(entries, output_path)

    # Summary
    print(f"\nSubmission package: {output_path}")
    print(f"{'='*55}")
    total_size = 0
    for src, name in entries:
        size = src.stat().st_size
        total_size += size
        tag = " [weight]" if src.suffix in WEIGHT_EXTENSIONS else ""
        print(f"  {name:<30} {format_size(size):>10}{tag}")
    print(f"{'='*55}")
    print(f"  Total uncompressed: {format_size(total_size)}")
    print(f"  Zip file size:      {format_size(output_path.stat().st_size)}")

    # Verify
    print("\nVerification:")
    with zipfile.ZipFile(output_path, "r") as zf:
        for info in zf.infolist():
            print(f"  {info.filename:<30} {format_size(info.file_size):>10}")

    print("\nDone.")


if __name__ == "__main__":
    main()
