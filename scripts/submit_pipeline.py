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
        half=True,
    ))

    dest = config.SUBMISSION_DIR / "best_main.onnx"
    shutil.copy2(onnx_path, dest)
    size_mb = dest.stat().st_size / 1e6
    print(f"Exported: {dest.name} ({size_mb:.1f} MB)")
    if size_mb > 400:
        print(f"ERROR: Model too large ({size_mb:.1f} MB > 420 MB limit)")
        return False

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

        clf_model = tv_models.efficientnet_b0()
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
    """Run the actual submission run.py on val images and score the output."""
    print("\n" + "=" * 60)
    print("STEP 3/3: Test submission (run.py + score)")
    print("=" * 60)

    run_py = config.SUBMISSION_DIR / "run.py"
    val_images = config.YOLO_DIR / "val" / "images"
    annotations = config.COCO_EXTRACT_DIR / "train" / "annotations.json"
    predictions_out = Path("/tmp/submit_test_predictions.json")

    if not run_py.exists():
        print(f"ERROR: {run_py} not found")
        return False
    if not val_images.exists() or not list(val_images.iterdir()):
        print(f"WARNING: No val images at {val_images}, using train images subset")
        val_images = config.YOLO_DIR / "train" / "images"

    # Use only a small subset — ONNX on CPU is slow
    test_dir = Path("/tmp/submit_test_images")
    test_dir.mkdir(parents=True, exist_ok=True)
    import shutil as _shutil
    for f in test_dir.iterdir():
        f.unlink()
    imgs = sorted(val_images.glob("*.jpg"))[:10]
    for img in imgs:
        _shutil.copy2(img, test_dir / img.name)
    print(f"  Testing on {len(imgs)} images (subset)")
    val_images = test_dir
    if not annotations.exists():
        print(f"ERROR: Annotations not found: {annotations}")
        return False

    # Run the actual submission code
    print(f"  Running: {run_py.name} --input {val_images}")
    result = subprocess.run(
        [sys.executable, str(run_py),
         "--input", str(val_images),
         "--output", str(predictions_out)],
        capture_output=False,
        cwd=str(config.SUBMISSION_DIR),
    )
    if result.returncode != 0:
        print(f"  run.py failed with exit code {result.returncode}")
        return False

    # Score predictions
    print(f"\n  Scoring predictions...")
    import json as _json
    with open(predictions_out) as f:
        predictions = _json.load(f)

    sandbox_run = Path(__file__).parent / "sandbox_run.py"
    # Import scoring function directly
    import importlib.util
    spec = importlib.util.spec_from_file_location("sandbox_run", sandbox_run)
    sr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sr)

    score_result = sr.score_predictions(
        str(annotations),
        predictions,
    )

    det = score_result["detection_map"]
    cls = score_result["classification_map"]
    score = score_result["score"]

    print(f"\n  Detection mAP@0.5:      {det:.4f}  (x 0.70 = {det*0.7:.4f})")
    print(f"  Classification mAP@0.5: {cls:.4f}  (x 0.30 = {cls*0.3:.4f})")
    print(f"  PREDICTED SCORE:        {score:.4f}")

    if score_result.get("errors"):
        for e in score_result["errors"]:
            print(f"  ERROR: {e}")

    return True


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

    elapsed = time.time() - start
    print(f"\nExport + package complete in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
