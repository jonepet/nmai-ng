"""
Full submission pipeline: export → package → sandbox test → score.
Runs in a single container. All output streams directly.
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# PyTorch 2.6 compat
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


def step_export():
    """Export best checkpoint to ONNX."""
    print("\n" + "=" * 60)
    print("STEP 1/3: Export to ONNX")
    print("=" * 60)

    from ultralytics import YOLO

    checkpoint = config.CHECKPOINT_ROOT / "best_final.pt"
    if not checkpoint.exists():
        # Find latest stage best
        candidates = sorted(
            config.CHECKPOINT_ROOT.glob("*/weights/best.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            shutil.copy2(candidates[0], checkpoint)
            print(f"Created best_final.pt from {candidates[0].parent.parent.name}")
        else:
            print("ERROR: No checkpoints found")
            return False

    print(f"Checkpoint: {checkpoint} ({checkpoint.stat().st_size / 1e6:.1f} MB)")

    model = YOLO(str(checkpoint))
    onnx_path = Path(model.export(
        format="onnx",
        imgsz=config.IMGSZ,
        opset=config.ONNX_OPSET,
        simplify=True,
        dynamic=False,
    ))

    dest = config.SUBMISSION_DIR / "best_main.onnx"
    shutil.copy2(onnx_path, dest)
    print(f"Exported: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")

    # Export classifier if checkpoint exists but ONNX doesn't
    cls_onnx = config.SUBMISSION_DIR / "classifier.onnx"
    cls_checkpoint = config.CHECKPOINT_ROOT / "classifier_best.pt"
    if cls_checkpoint.exists() and not cls_onnx.exists():
        print(f"\nExporting classifier to ONNX...")
        from torchvision import models as tv_models
        import torch.nn as nn

        ckpt = torch.load(str(cls_checkpoint), map_location="cpu")
        num_classes = ckpt["num_classes"]
        input_size = ckpt["input_size"]

        clf_model = tv_models.mobilenet_v3_small()
        clf_model.classifier[-1] = nn.Linear(clf_model.classifier[-1].in_features, num_classes)
        clf_model.load_state_dict(ckpt["model_state_dict"])
        clf_model.eval()

        dummy = torch.randn(1, 3, input_size, input_size)
        torch.onnx.export(
            clf_model, dummy, str(cls_onnx),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=config.ONNX_OPSET,
        )
        print(f"Exported: {cls_onnx.name} ({cls_onnx.stat().st_size / 1e6:.1f} MB)")

    return True


def step_package():
    """Package submission zip."""
    print("\n" + "=" * 60)
    print("STEP 2/3: Package submission.zip")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "package_submission.py")],
        capture_output=False,
    )
    return result.returncode == 0


def step_test():
    """Quick score using .pt model on CPU (subset of val images for speed)."""
    print("\n" + "=" * 60)
    print("STEP 3/3: Quick score (CPU, subset)")
    print("=" * 60)

    eval_script = Path(__file__).parent / "evaluate.py"
    checkpoint = config.CHECKPOINT_ROOT / "best_final.pt"

    if not checkpoint.exists():
        print("ERROR: No best_final.pt for evaluation")
        return False

    result = subprocess.run(
        [sys.executable, str(eval_script),
         "--checkpoint", str(checkpoint),
         "--max-images", "10"],
        capture_output=False,
    )
    return result.returncode == 0


def main():
    print("=" * 60)
    print("  SUBMISSION PIPELINE")
    print("=" * 60)

    start = time.time()

    if not step_export():
        print("\nExport failed.")
        sys.exit(1)

    if not step_package():
        print("\nPackage failed.")
        sys.exit(1)

    if not step_test():
        print("\nTest failed.")
        sys.exit(1)

    elapsed = time.time() - start
    print(f"\nPipeline complete in {elapsed:.0f}s")
    print("Upload submission.zip at the competition website.")


if __name__ == "__main__":
    main()
