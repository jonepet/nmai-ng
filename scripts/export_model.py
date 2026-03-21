"""
Export trained models to ONNX format for submission.

ONNX is the recommended format per competition docs:
- Universal, no pickle issues
- Works with onnxruntime-gpu (pre-installed in sandbox)
- Use CUDAExecutionProvider for GPU acceleration
- opset ≤ 20 (we use 17)
"""

import shutil
import sys
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


def get_file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


WEIGHT_MAP = {
    "best_main": config.CHECKPOINT_ROOT / "best_final.pt",
    "best_parallel": config.CHECKPOINT_ROOT / "best_parallel_gpu0.pt",
}


def export_one(name: str, source_pt: Path, submission_dir: Path) -> bool:
    """Export a single .pt checkpoint to .onnx in submission dir."""
    if not source_pt.exists():
        print(f"  {source_pt} not found — skipping {name}")
        return False

    print(f"\n  Exporting {name}: {source_pt} ({get_file_size_mb(source_pt):.1f} MB)")

    model = YOLO(str(source_pt))
    onnx_path = Path(model.export(
        format="onnx",
        imgsz=config.IMGSZ,
        opset=config.ONNX_OPSET,
        simplify=True,
        dynamic=False,
    ))

    dest = submission_dir / f"{name}.onnx"
    shutil.copy2(onnx_path, dest)

    size_mb = get_file_size_mb(dest)
    print(f"  -> {dest.name} ({size_mb:.1f} MB)")

    if size_mb > config.SUBMISSION_MAX_WEIGHT_SIZE_MB:
        print(f"  WARNING: {size_mb:.1f} MB exceeds {config.SUBMISSION_MAX_WEIGHT_SIZE_MB} MB limit")
        # Try FP16
        print(f"  Retrying with FP16...")
        onnx_path = Path(model.export(
            format="onnx",
            imgsz=config.IMGSZ,
            opset=config.ONNX_OPSET,
            simplify=True,
            dynamic=False,
            half=True,
        ))
        shutil.copy2(onnx_path, dest)
        size_mb = get_file_size_mb(dest)
        print(f"  -> {dest.name} FP16 ({size_mb:.1f} MB)")

    return True


def main() -> None:
    submission_dir = config.SUBMISSION_DIR
    submission_dir.mkdir(parents=True, exist_ok=True)

    print("NorgesGruppen model export to ONNX")
    print(f"Submission dir: {submission_dir}")
    print(f"ONNX opset: {config.ONNX_OPSET}")
    print(f"Image size: {config.IMGSZ}")

    exported = 0
    total_bytes = 0

    for name, source_pt in WEIGHT_MAP.items():
        if export_one(name, source_pt, submission_dir):
            exported += 1
            total_bytes += (submission_dir / f"{name}.onnx").stat().st_size

    if exported == 0:
        print("\nERROR: No models exported.", file=sys.stderr)
        sys.exit(1)

    total_mb = total_bytes / (1024 * 1024)
    print(f"\nExported {exported} model(s), total: {total_mb:.1f} MB")
    print(f"Size check: {total_mb:.1f} / {config.SUBMISSION_MAX_WEIGHT_SIZE_MB} MB ({total_mb/config.SUBMISSION_MAX_WEIGHT_SIZE_MB*100:.0f}% used)")

    # Verify with onnxruntime
    import onnxruntime as ort
    import numpy as np
    first_onnx = submission_dir / f"{list(WEIGHT_MAP.keys())[0]}.onnx"
    if first_onnx.exists():
        print(f"\nVerifying {first_onnx.name} with onnxruntime...")
        sess = ort.InferenceSession(str(first_onnx), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        print(f"  Input: {inp.name} shape={inp.shape} type={inp.type}")
        for out in sess.get_outputs():
            print(f"  Output: {out.name} shape={out.shape} type={out.type}")
        dummy = np.random.randn(*[d if isinstance(d, int) else 1 for d in inp.shape]).astype(np.float32)
        results = sess.run(None, {inp.name: dummy})
        print(f"  Inference OK — output shape: {results[0].shape}")

    print("\nExport complete.")


if __name__ == "__main__":
    main()
