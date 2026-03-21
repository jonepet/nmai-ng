"""
Parallel diversity training on GPU 1 (GTX 960, 2GB VRAM).

Runs in its own Docker container with CUDA_VISIBLE_DEVICES=0 mapped to the
physical GPU 1. The container only sees one GPU as CUDA:0.

Trains YOLOv8n with different augmentation for ensemble diversity.
Saves best checkpoint to checkpoints/best_parallel_gpu0.pt.

All constants are sourced from config.py.
"""

import shutil
import sys
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


def find_dataset_yaml() -> str:
    for candidate in [str(config.DATASET_YAML_PATH), "/data/dataset.yaml"]:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("dataset.yaml not found")


def main() -> None:
    from ultralytics import YOLO

    print("[parallel] NorgesGruppen YOLOv8n diversity training")
    print(f"[parallel] Device: CUDA:0 (container sees single GPU)")

    # Wait for label cache from main training container
    cache_path = config.YOLO_DIR / "train" / "labels.cache"
    print(f"[parallel] Waiting for label cache: {cache_path}")
    for i in range(180):  # up to 3 minutes
        if cache_path.exists():
            print(f"[parallel] Cache found after {i}s")
            break
        time.sleep(1)
    else:
        print("[parallel] Cache not found after 3 min — proceeding anyway")

    data_yaml = find_dataset_yaml()
    config.CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    cfg = config.PARALLEL_GPU0_CONFIG
    name = cfg["name"]
    stage_num = cfg["stage_num"]

    # Force device to "0" — this container only sees one GPU
    device = "0"

    print(f"\n{'='*60}")
    print(f"[parallel] {name.upper()}")
    print(f"  model   : {cfg['model']}")
    print(f"  epochs  : {cfg['epochs']}")
    print(f"  lr0     : {cfg['lr0']}")
    print(f"  batch   : {cfg['batch']}")
    print(f"  imgsz   : {cfg['imgsz']}")
    print(f"  device  : {device} (physical: GPU {config.GPU_SECONDARY})")
    print(f"{'='*60}")

    model = YOLO(cfg["model"])

    kwargs = {
        "data": data_yaml,
        "epochs": cfg["epochs"],
        "lr0": cfg["lr0"],
        "batch": cfg["batch"],
        "imgsz": cfg["imgsz"],
        "device": device,
        "workers": cfg.get("workers", 4),
        "project": str(config.CHECKPOINT_ROOT),
        "name": f"stage_{stage_num}_{name}",
        "exist_ok": True,
        "pretrained": cfg.get("pretrained", True),
        "plots": True,
        "val": True,
        "patience": config.PATIENCE,
        "save_period": config.SAVE_PERIOD,
        "cos_lr": config.COS_LR,
        "amp": config.AMP,
        "hsv_h": cfg["hsv_h"],
        "hsv_s": cfg["hsv_s"],
        "hsv_v": cfg["hsv_v"],
        "degrees": cfg["degrees"],
        "translate": cfg["translate"],
        "scale": cfg["scale"],
        "flipud": cfg["flipud"],
        "fliplr": cfg["fliplr"],
        "mosaic": cfg["mosaic"],
        "mixup": cfg["mixup"],
    }

    t0 = time.time()
    try:
        results = model.train(**kwargs)
        elapsed = time.time() - t0
        print(f"\n[parallel] Completed in {elapsed/60:.1f} min.")

        best_src = Path(results.save_dir) / "weights" / "best.pt"
        best_dst = config.CHECKPOINT_ROOT / "best_parallel_gpu0.pt"
        if best_src.exists():
            shutil.copy2(best_src, best_dst)
            print(f"[parallel] Best checkpoint -> {best_dst}")
            print(f"[parallel] Size: {best_dst.stat().st_size / 1024 / 1024:.1f} MB")
        else:
            print("[parallel] WARNING: No best.pt found")

    except Exception as exc:
        elapsed = time.time() - t0
        print(f"\n[parallel] Failed after {elapsed/60:.1f} min: {exc}")
        sys.exit(1)

    print("[parallel] Done.")


if __name__ == "__main__":
    main()
