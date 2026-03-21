"""
Multi-stage YOLOv8 training script for NorgesGruppen grocery product detection.

Runs in its own Docker container with a single GPU (CUDA:0).
The parallel diversity job runs in a separate container (train-parallel service).

Stages (from config.TRAINING_STAGES):
  1. Warmup    — frozen backbone, YOLOv8s pretrained
  2. Fine-tune — full model, loaded from Stage 1 best
  3. Polish    — full model, loaded from Stage 2 best

All constants are sourced from config.py.
"""

import shutil
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Config import — project root is one level above this script
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# ---------------------------------------------------------------------------
# PyTorch 2.6 compatibility: ultralytics 8.1.0 predates the weights_only=True
# default change. Patch torch.load so pretrained weights can be loaded.
# ---------------------------------------------------------------------------
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_dataset_yaml() -> str:
    """
    Resolve the dataset YAML path.

    Priority:
      1. config.DATASET_YAML_PATH (canonical project location)
      2. /data/dataset.yaml (Docker bind-mount shortcut)
    """
    candidates = [
        str(config.DATASET_YAML_PATH),
        "/data/dataset.yaml",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            print(f"[train] Using dataset YAML: {candidate}")
            return candidate
    raise FileNotFoundError(
        f"dataset.yaml not found. Tried: {candidates}"
    )


def clear_cuda_cache() -> None:
    try:
        import torch
        torch.cuda.empty_cache()
        print("[train] CUDA cache cleared.")
    except Exception as exc:
        print(f"[train] Could not clear CUDA cache: {exc}")


def adjust_lr_between_stages(current_lr: float, val_loss: float | None) -> float:
    """
    Adjust the learning rate for the next stage based on the val loss of the
    current stage.

      val_loss > config.ADAPTIVE_LR_HIGH_LOSS  -> keep lr unchanged
      val_loss < config.ADAPTIVE_LR_LOW_LOSS   -> reduce by 10x
      otherwise                                 -> keep lr unchanged
    """
    if val_loss is None:
        print("[train] Val loss unavailable; keeping lr unchanged.")
        return current_lr

    if val_loss < config.ADAPTIVE_LR_LOW_LOSS:
        new_lr = current_lr / 10.0
        print(
            f"[train] Val loss {val_loss:.4f} < {config.ADAPTIVE_LR_LOW_LOSS} "
            f"— reducing lr {current_lr} -> {new_lr}"
        )
        return new_lr

    print(
        f"[train] Val loss {val_loss:.4f} — keeping lr = {current_lr}"
    )
    return current_lr


def extract_best_val_loss(results) -> float | None:
    """
    Extract the best val box loss from a completed YOLO training run.
    Ultralytics stores results in the results.results_dict attribute.
    """
    try:
        rd = results.results_dict
        for key in ("val/box_loss", "metrics/box_loss", "box_loss"):
            if key in rd:
                return float(rd[key])
    except Exception:
        pass

    # Fallback: read last line of results.csv if available
    try:
        csv_path = Path(results.save_dir) / "results.csv"
        if csv_path.exists():
            lines = csv_path.read_text().strip().splitlines()
            if len(lines) >= 2:
                headers = [h.strip() for h in lines[0].split(",")]
                values = [v.strip() for v in lines[-1].split(",")]
                row = dict(zip(headers, values))
                for key in ("val/box_loss", "metrics/box_loss", "box_loss"):
                    if key in row:
                        return float(row[key])
    except Exception:
        pass

    return None


def build_train_kwargs(cfg: dict, data_yaml: str, stage_dir: Path) -> dict:
    """Build the keyword-argument dict for YOLO.train()."""
    workers = cfg.get("workers", config.WORKERS)
    kwargs = {
        "data": data_yaml,
        "epochs": cfg["epochs"],
        "lr0": cfg["lr0"],
        "batch": cfg["batch"],
        "imgsz": cfg["imgsz"],
        "device": "0",  # Each container sees only its own GPU as CUDA:0
        "workers": workers,
        "project": str(config.CHECKPOINT_ROOT),
        "name": f"stage_{cfg['stage_num']}_{cfg['name']}",
        "exist_ok": True,
        "pretrained": cfg.get("pretrained", False),
        "plots": True,
        "val": True,
        "patience": config.PATIENCE,
        "save_period": config.SAVE_PERIOD,
        "cos_lr": config.COS_LR,
        "amp": config.AMP,
        # augmentation
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

    if cfg.get("freeze", 0) > 0:
        kwargs["freeze"] = cfg["freeze"]

    return kwargs


def evaluate_checkpoint(checkpoint_path, stage_num):
    """Run competition-style evaluation after a stage completes (on CPU to avoid GPU contention)."""
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        print(f"[stage {stage_num}] No checkpoint to evaluate.")
        return

    try:
        import subprocess as sp
        eval_script = Path(__file__).resolve().parent / "evaluate.py"
        if not eval_script.exists():
            print(f"[stage {stage_num}] evaluate.py not found, skipping evaluation.")
            return

        print(f"\n[stage {stage_num}] Running competition evaluation on {Path(checkpoint_path).name}...")
        result = sp.run(
            [sys.executable, str(eval_script), "--checkpoint", str(checkpoint_path)],
            capture_output=True, text=True, timeout=600,
        )
        if result.stdout:
            print(result.stdout)
        if result.returncode != 0 and result.stderr:
            print(f"[stage {stage_num}] Evaluation warning: {result.stderr[:500]}")
    except Exception as exc:
        print(f"[stage {stage_num}] Evaluation failed (non-fatal): {exc}")


def run_stage(cfg: dict, data_yaml: str) -> tuple[bool, float | None]:
    """
    Execute a single training stage.

    Returns (success, best_val_loss).
    On OOM, retries once with batch size halved.
    """
    from ultralytics import YOLO

    stage_dir = config.CHECKPOINT_ROOT / f"stage_{cfg['stage_num']}_{cfg['name']}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    model_path = cfg["model"]
    print(
        f"\n{'='*60}\n"
        f"[stage {cfg['stage_num']}] {cfg['name'].upper()}\n"
        f"  model   : {model_path}\n"
        f"  epochs  : {cfg['epochs']}\n"
        f"  lr0     : {cfg['lr0']}\n"
        f"  batch   : {cfg['batch']}\n"
        f"  imgsz   : {cfg['imgsz']}\n"
        f"  device  : {cfg['device']}\n"
        f"  freeze  : {cfg.get('freeze', 0)}\n"
        f"{'='*60}"
    )

    for attempt in range(1, 3):  # up to 2 attempts
        batch = cfg["batch"] if attempt == 1 else max(1, cfg["batch"] // 2)
        if attempt == 2:
            print(
                f"[stage {cfg['stage_num']}] Retry attempt {attempt} "
                f"with halved batch size = {batch}"
            )

        try:
            model = YOLO(model_path)
            kwargs = build_train_kwargs(cfg, data_yaml, stage_dir)
            kwargs["batch"] = batch

            t0 = time.time()
            results = model.train(**kwargs)
            elapsed = time.time() - t0

            print(
                f"[stage {cfg['stage_num']}] Completed in "
                f"{elapsed/60:.1f} min."
            )

            # Copy best checkpoint
            best_src = Path(results.save_dir) / "weights" / "best.pt"
            if best_src.exists():
                best_dst = config.CHECKPOINT_ROOT / f"best_stage_{cfg['stage_num']}.pt"
                shutil.copy2(best_src, best_dst)
                print(
                    f"[stage {cfg['stage_num']}] Best checkpoint -> {best_dst}"
                )

            val_loss = extract_best_val_loss(results)
            if val_loss is not None:
                print(
                    f"[stage {cfg['stage_num']}] Best val box_loss: "
                    f"{val_loss:.4f}"
                )

            # Run competition-style evaluation on the best checkpoint
            evaluate_checkpoint(best_dst if best_src.exists() else None, cfg['stage_num'])

            return True, val_loss

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() or "CUDA out of memory" in str(exc):
                print(
                    f"[stage {cfg['stage_num']}] OOM on attempt {attempt}: {exc}"
                )
                clear_cuda_cache()
                if attempt == 2:
                    print(
                        f"[stage {cfg['stage_num']}] OOM persists after "
                        "batch-size reduction. Skipping stage."
                    )
                    return False, None
            else:
                print(
                    f"[stage {cfg['stage_num']}] RuntimeError: {exc}"
                )
                return False, None

        except Exception as exc:
            print(f"[stage {cfg['stage_num']}] Unexpected error: {exc}")
            return False, None

    return False, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("[train] NorgesGruppen YOLOv8 multi-stage training (main GPU)")
    print("[train] ultralytics version: ", end="")
    try:
        import ultralytics
        print(ultralytics.__version__)
    except ImportError:
        print("NOT INSTALLED — aborting.")
        sys.exit(1)

    data_yaml = find_dataset_yaml()
    config.CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Run augmentation if not already done
    # -----------------------------------------------------------------------
    augment_scripts = [
        Path(__file__).resolve().parent / "augment_data.py",
        Path(__file__).resolve().parent / "augment_cropmix.py",
    ]
    yolo_train_dir = config.YOLO_DIR / "train" / "images"
    n_images = len(list(yolo_train_dir.glob("*"))) if yolo_train_dir.exists() else 0
    if n_images <= config.EXPECTED_IMAGE_COUNT:
        print(f"[train] Only {n_images} training images — running augmentation...")
        for script in augment_scripts:
            if script.exists():
                import subprocess as sp
                result = sp.run([sys.executable, str(script)], capture_output=False)
                if result.returncode != 0:
                    print(f"[train] WARNING: {script.name} failed (exit {result.returncode})")
        n_after = len(list(yolo_train_dir.glob("*")))
        print(f"[train] Augmentation done: {n_images} → {n_after} images")
    else:
        print(f"[train] {n_images} training images (augmented)")

    # -----------------------------------------------------------------------
    # Continuous training loop: stages → evaluate → update best → repeat
    # -----------------------------------------------------------------------
    round_num = 0

    while True:
        round_num += 1
        print(f"\n{'='*60}")
        print(f"[train] ROUND {round_num}")
        print(f"{'='*60}")

        stage_configs = [dict(c) for c in config.TRAINING_STAGES]
        best_val_loss = float("inf")
        best_checkpoint = None

        for i, cfg in enumerate(stage_configs):
            stage_num = cfg["stage_num"]
            stage_dir = config.CHECKPOINT_ROOT / cfg["name"]

            # Skip if this stage already completed all its epochs in round 1
            if round_num == 1:
                best_stage_pt = config.CHECKPOINT_ROOT / f"best_stage_{stage_num}.pt"
                if best_stage_pt.exists():
                    print(f"[train] Stage {stage_num} ({cfg['name']}) already complete, skipping")
                    continue

            # Load from best available checkpoint
            best_final = config.CHECKPOINT_ROOT / "best_final.pt"
            prev_best = config.CHECKPOINT_ROOT / f"best_stage_{stage_num - 1}.pt"
            if prev_best.exists():
                cfg["model"] = str(prev_best)
            elif best_final.exists():
                cfg["model"] = str(best_final)
            # else: use MODEL_PRIMARY from config (COCO pretrained, first ever run)

            print(f"[train] Stage {stage_num} loading from {cfg['model']}")

            # Scale LR down for later rounds
            if round_num > 1:
                lr_scale = max(0.1, 1.0 / round_num)
                cfg["lr0"] = cfg["lr0"] * lr_scale
                print(f"[train] Round {round_num} LR scaled: {cfg['lr0']:.6f}")

            clear_cuda_cache()
            success, val_loss = run_stage(cfg, data_yaml)

            if not success:
                print(f"[train] Stage {stage_num} failed, continuing")
                continue

            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint = config.CHECKPOINT_ROOT / f"best_stage_{stage_num}.pt"

        # Update best_final.pt
        if best_checkpoint and best_checkpoint.exists():
            shutil.copy2(best_checkpoint, config.CHECKPOINT_ROOT / "best_final.pt")
            print(f"\n[train] Round {round_num} best: val_loss={best_val_loss:.4f} from {best_checkpoint.name}")
        else:
            print(f"\n[train] Round {round_num}: no improvement")

        # -------------------------------------------------------------------
        # Feedback loop: evaluate → find failures → augment failures → retrain
        # -------------------------------------------------------------------
        best_final = config.CHECKPOINT_ROOT / "best_final.pt"
        if best_final.exists():
            print(f"\n[train] Running feedback analysis...")

            # Run competition evaluation
            evaluate_checkpoint(str(best_final), f"round_{round_num}")

            # Hard mining: find what the model gets wrong, augment those
            import subprocess as sp
            hm_script = Path(__file__).resolve().parent / "hard_mining.py"
            if hm_script.exists():
                print(f"[train] Mining hard examples...")
                hm_result = sp.run(
                    [sys.executable, str(hm_script)],
                    capture_output=False,
                )
                if hm_result.returncode == 0:
                    n_images = len(list(yolo_train_dir.glob("*")))
                    print(f"[train] Hard mining done. Training set: {n_images} images")
                else:
                    print(f"[train] Hard mining failed (exit {hm_result.returncode})")

        print(f"\n[train] Round {round_num} complete. Starting next round...")


if __name__ == "__main__":
    main()
