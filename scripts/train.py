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
    # Multi-stage training — single GPU per container (CUDA:0)
    # The parallel job runs in a separate container (train-parallel service)
    # -----------------------------------------------------------------------
    stage_configs = [dict(c) for c in config.TRAINING_STAGES]
    current_lr_override: float | None = None
    # Track val losses per stage for final model selection
    stage_val_losses: dict[int, float] = {}

    for i, cfg in enumerate(stage_configs):
        stage_num = cfg["stage_num"]

        # Wire up model path:
        #   Stage 1     — uses its own pretrained model (from config, e.g. yolov8s.pt)
        #   Stage 4     — "upgrade" stage, uses its own pretrained model (e.g. yolov8m.pt)
        #   Other stages — load from previous stage's best checkpoint
        if stage_num == 1:
            # Stage 1 always starts from COCO pretrained weights defined in config
            # cfg["model"] is already set to MODEL_PRIMARY from TRAINING_STAGES
            pass
        elif cfg.get("model") is not None:
            # Stage has an explicit model defined (e.g. Stage 4 upgrade with yolov8m.pt)
            # Keep it — this stage trains from COCO pretrained weights of its own model
            pass
        else:
            # Load from the previous stage's best checkpoint
            prev_best = config.CHECKPOINT_ROOT / f"best_stage_{stage_num - 1}.pt"
            if prev_best.exists():
                cfg["model"] = str(prev_best)
                print(
                    f"[train] Stage {stage_num} loading weights "
                    f"from {prev_best}"
                )
            else:
                print(
                    f"[train] WARNING: best_stage_{stage_num - 1}.pt not "
                    f"found — using default {config.MODEL_PRIMARY} weights."
                )
                cfg["model"] = config.MODEL_PRIMARY

        # Apply adaptive lr from previous stage's outcome
        if current_lr_override is not None:
            print(
                f"[train] Overriding lr0 for stage {stage_num}: "
                f"{cfg['lr0']} -> {current_lr_override}"
            )
            cfg["lr0"] = current_lr_override
            current_lr_override = None

        clear_cuda_cache()

        success, val_loss = run_stage(cfg, data_yaml)

        if not success:
            print(
                f"[train] Stage {stage_num} failed. "
                "Continuing to next stage if possible."
            )
            continue

        if val_loss is not None:
            stage_val_losses[stage_num] = val_loss

        # Compute adaptive lr for next stage
        if i + 1 < len(stage_configs):
            next_cfg = stage_configs[i + 1]
            adjusted = adjust_lr_between_stages(next_cfg["lr0"], val_loss)
            if adjusted != next_cfg["lr0"]:
                current_lr_override = adjusted

    # -----------------------------------------------------------------------
    # Hard mining rounds: mine failures → augment → retrain
    # -----------------------------------------------------------------------
    for hm_round in range(1, config.HARD_MINING_ROUNDS + 1):
        print(f"\n{'='*60}")
        print(f"[train] HARD MINING ROUND {hm_round}/{config.HARD_MINING_ROUNDS}")
        print(f"{'='*60}")

        # Run hard_mining.py as subprocess (separate import scope)
        import subprocess as sp
        hm_script = Path(__file__).resolve().parent / "hard_mining.py"
        if not hm_script.exists():
            print(f"[train] hard_mining.py not found — skipping")
            break

        hm_result = sp.run(
            [sys.executable, str(hm_script)],
            capture_output=False,
        )
        if hm_result.returncode != 0:
            print(f"[train] Hard mining failed (exit {hm_result.returncode}) — skipping retrain")
            break

        # Retrain: stages 2+3 only (fine-tune + polish on enriched data)
        # Start from best checkpoint so far
        print(f"\n[train] Retraining with hard examples (round {hm_round})...")
        retrain_stages = [
            {
                "name": f"retrain_r{hm_round}_finetune",
                "stage_num": 100 + hm_round * 10 + 1,
                "model": None,  # will be set to best_final.pt below
                "epochs": 30,
                "lr0": 0.0005,
                "batch": 4,
                "imgsz": config.IMGSZ,
                "device": config.GPU_PRIMARY,
                "freeze": 0,
                "pretrained": False,
                "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
                "degrees": 5.0, "translate": 0.1, "scale": 0.5,
                "flipud": 0.0, "fliplr": 0.5,
                "mosaic": 1.0, "mixup": 0.1,
            },
            {
                "name": f"retrain_r{hm_round}_polish",
                "stage_num": 100 + hm_round * 10 + 2,
                "model": None,  # loaded from previous retrain stage
                "epochs": 15,
                "lr0": 0.0001,
                "batch": 4,
                "imgsz": config.IMGSZ,
                "device": config.GPU_PRIMARY,
                "freeze": 0,
                "pretrained": False,
                "hsv_h": 0.005, "hsv_s": 0.3, "hsv_v": 0.2,
                "degrees": 0.0, "translate": 0.05, "scale": 0.2,
                "flipud": 0.0, "fliplr": 0.5,
                "mosaic": 0.5, "mixup": 0.0,
            },
        ]

        # Set first retrain stage to load from current best
        best_so_far = config.CHECKPOINT_ROOT / "best_final.pt"
        if best_so_far.exists():
            retrain_stages[0]["model"] = str(best_so_far)
        else:
            print(f"[train] No best_final.pt — skipping retrain round {hm_round}")
            break

        for j, rt_cfg in enumerate(retrain_stages):
            if j > 0:
                # Load from previous retrain stage
                prev_sn = retrain_stages[j - 1]["stage_num"]
                prev_best = config.CHECKPOINT_ROOT / f"best_stage_{prev_sn}.pt"
                if prev_best.exists():
                    rt_cfg["model"] = str(prev_best)
                else:
                    rt_cfg["model"] = str(best_so_far)

            clear_cuda_cache()
            success, val_loss = run_stage(rt_cfg, data_yaml)

            if success and val_loss is not None:
                stage_val_losses[rt_cfg["stage_num"]] = val_loss

        # Update best_final.pt if retrain improved
        retrain_candidates = []
        for rt_cfg in retrain_stages:
            pt = config.CHECKPOINT_ROOT / f"best_stage_{rt_cfg['stage_num']}.pt"
            if pt.exists():
                loss = stage_val_losses.get(rt_cfg["stage_num"])
                if loss is not None:
                    retrain_candidates.append((loss, pt, rt_cfg["name"]))

        if retrain_candidates:
            retrain_candidates.sort()
            best_loss, best_pt, best_name = retrain_candidates[0]
            # Compare with current best
            current_best_loss = min(
                (v for k, v in stage_val_losses.items() if k < 100),
                default=float("inf")
            )
            if best_loss < current_best_loss:
                shutil.copy2(best_pt, config.CHECKPOINT_ROOT / "best_final.pt")
                print(f"[train] Round {hm_round} improved! {best_name} val_loss={best_loss:.4f} (was {current_best_loss:.4f})")
            else:
                print(f"[train] Round {hm_round} did not improve ({best_loss:.4f} >= {current_best_loss:.4f})")

    # -----------------------------------------------------------------------
    # Determine final best model from all stages
    # (parallel GPU0 model is selected separately by train_parallel.py)
    # -----------------------------------------------------------------------
    print("\n[train] Selecting final best model...")

    # Gather all stage candidates (Stages 1–4 + parallel GPU0)
    stage_nums_in_config = [c["stage_num"] for c in config.TRAINING_STAGES]
    candidates: list[tuple[float, Path, str]] = []  # (val_loss, path, label)

    for sn in stage_nums_in_config:
        pt = config.CHECKPOINT_ROOT / f"best_stage_{sn}.pt"
        if pt.exists():
            loss = stage_val_losses.get(sn)
            if loss is not None:
                candidates.append((loss, pt, f"stage_{sn}"))
                print(f"[train]   stage_{sn}: val_loss={loss:.4f}  {pt}")
            else:
                # Stage completed but val loss not captured — still a candidate
                # Use a sentinel high loss so it only wins if nothing else exists
                candidates.append((float("inf"), pt, f"stage_{sn} (loss unknown)"))
                print(f"[train]   stage_{sn}: val_loss=unknown  {pt}")

    # Parallel GPU0
    parallel_pt = config.CHECKPOINT_ROOT / "best_parallel_gpu0.pt"
    if parallel_pt.exists():
        # Val loss for parallel job is not tracked in stage_val_losses;
        # use infinity so it only wins if no main-stage checkpoints exist
        candidates.append((float("inf"), parallel_pt, "parallel_gpu0 (loss unknown)"))
        print(f"[train]   parallel_gpu0: val_loss=unknown  {parallel_pt}")

    best_final_dst = config.CHECKPOINT_ROOT / "best_final.pt"

    if candidates:
        # Sort by val loss ascending; lowest loss wins
        candidates.sort(key=lambda t: t[0])
        best_loss, best_src, best_label = candidates[0]
        shutil.copy2(best_src, best_final_dst)
        loss_str = f"{best_loss:.4f}" if best_loss != float("inf") else "unknown"
        print(
            f"[train] Final best model: {best_final_dst}\n"
            f"        source: {best_src.name}  ({best_label}, val_loss={loss_str})"
        )
    else:
        print("[train] WARNING: No stage checkpoint found — best_final.pt not created.")

    print("\n[train] All stages complete.")
    print(f"[train] Checkpoints saved under: {config.CHECKPOINT_ROOT}")


if __name__ == "__main__":
    main()
