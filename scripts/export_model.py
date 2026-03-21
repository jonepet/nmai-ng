"""
Export trained models for NorgesGruppen competition submission.

Copies both main and parallel model weights to submission/ directory
with the filenames expected by run.py's ensemble inference.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

import torch
# PyTorch 2.6 compat: ultralytics 8.1.0 needs weights_only=False
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from ultralytics import YOLO


def get_file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def format_size(size_bytes: int) -> str:
    mb = size_bytes / (1024 * 1024)
    return f"{mb:.1f} MB"


# Weight source mapping: zip name -> checkpoint path
WEIGHT_MAP = {
    "best_main.pt": config.CHECKPOINT_ROOT / "best_final.pt",
    "best_parallel.pt": config.CHECKPOINT_ROOT / "best_parallel_gpu0.pt",
}

MAX_TOTAL_MB = config.SUBMISSION_MAX_WEIGHT_SIZE_MB  # 420 MB


def _find_sample_image() -> Path | None:
    for root in [config.DATA_DIR, Path("/workspace/data")]:
        if not root.exists():
            continue
        for ext in ("*.jpg", "*.jpeg"):
            candidates = sorted(root.rglob(ext))
            if candidates:
                return candidates[0]
    return None


def main() -> None:
    submission_dir = config.SUBMISSION_DIR
    submission_dir.mkdir(parents=True, exist_ok=True)

    print("NorgesGruppen model export for ensemble submission")
    print(f"Submission dir: {submission_dir}")
    print(f"Max total weight size: {MAX_TOTAL_MB} MB")
    print()

    # Copy each available weight file
    copied = []
    total_bytes = 0

    for zip_name, source_path in WEIGHT_MAP.items():
        dest = submission_dir / zip_name
        if source_path.exists():
            size = source_path.stat().st_size
            size_mb = size / (1024 * 1024)
            print(f"  {source_path.name} -> {zip_name}  ({size_mb:.1f} MB)")

            # Verify model loads correctly
            try:
                model = YOLO(str(source_path))
                print(f"    Model loaded OK: {model.model.__class__.__name__}")
            except Exception as exc:
                print(f"    WARNING: Failed to load {source_path}: {exc}")
                continue

            shutil.copy2(source_path, dest)
            copied.append((zip_name, size))
            total_bytes += size
        else:
            print(f"  {source_path} not found — skipping {zip_name}")

    if not copied:
        print("\nERROR: No weight files found to export.", file=sys.stderr)
        sys.exit(1)

    total_mb = total_bytes / (1024 * 1024)
    print(f"\nExported {len(copied)} model(s), total: {total_mb:.1f} MB")

    if total_mb > MAX_TOTAL_MB:
        print(f"WARNING: Total {total_mb:.1f} MB exceeds {MAX_TOTAL_MB} MB limit!")
    else:
        print(f"Size check passed: {total_mb:.1f} / {MAX_TOTAL_MB} MB ({total_mb/MAX_TOTAL_MB*100:.0f}% used)")

    # Verify inference with a sample image
    sample = _find_sample_image()
    if sample and copied:
        print(f"\nVerifying inference with {sample.name}...")
        test_model = YOLO(str(submission_dir / copied[0][0]))
        results = test_model(str(sample), verbose=False)
        n_det = sum(len(r.boxes) for r in results if r.boxes is not None)
        print(f"  {n_det} detection(s) — inference OK")

    print("\nExport complete.")


if __name__ == "__main__":
    main()
