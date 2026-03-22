"""
Centralized configuration for NorgesGruppen object detection project.

ALL parameters and settings live here. No magic numbers in other files.
Hardware-specific settings (GPU, model, batch sizes) are read from
environment variables, set via .env (not in git).
"""

import os
from pathlib import Path


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

# Data
DATA_DIR = PROJECT_ROOT / "data"
COCO_ZIP = PROJECT_ROOT / "NM_NGD_coco_dataset.zip"
PRODUCT_ZIP = PROJECT_ROOT / "NM_NGD_product_images.zip"
COCO_EXTRACT_DIR = DATA_DIR / "coco_dataset"
PRODUCT_EXTRACT_DIR = DATA_DIR / "product_images"
YOLO_DIR = DATA_DIR / "yolo"
DATASET_YAML_PATH = DATA_DIR / "dataset.yaml"

# Training output
CHECKPOINT_ROOT = Path("/workspace/checkpoints")

# Submission
SUBMISSION_DIR = PROJECT_ROOT / "submission"

# ---------------------------------------------------------------------------
# Hardware — read from .env via environment variables
# ---------------------------------------------------------------------------

GPU_PRIMARY = _env("GPU_PRIMARY", "0")
GPU_SECONDARY = _env("GPU_SECONDARY", "") or None
GPU_PRIMARY_VRAM_GB = _env_int("GPU_PRIMARY_VRAM_GB", 4)
CPU_CORES = _env_int("CPU_CORES", 16)
RAM_GB = _env_int("RAM_GB", 30)
SHM_SIZE = _env("SHM_SIZE", "4g")

# ---------------------------------------------------------------------------
# Model — read from .env
# ---------------------------------------------------------------------------

MODEL_PRIMARY = _env("MODEL_PRIMARY", "yolov8s.pt")
MODEL_PARALLEL = _env("MODEL_PARALLEL", "yolov8n.pt")

# Load submission config (single source of truth for inference params)
import json as _json
_submission_cfg_path = PROJECT_ROOT / "submission" / "config.json"
with open(_submission_cfg_path, encoding="utf-8") as _f:
    _submission_cfg = _json.load(_f)

NC = _submission_cfg["nc"]
IMGSZ = _env_int("IMGSZ", _submission_cfg["imgsz"])

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

WORKERS = _env_int("WORKERS", 8)
PATIENCE = 10              # Early stopping patience (epochs without improvement)
SAVE_PERIOD = 5            # Save checkpoint every N epochs
COS_LR = True              # Cosine annealing learning rate
AMP = True                 # Automatic Mixed Precision (FP16) — essential for VRAM savings

# Adaptive LR thresholds
ADAPTIVE_LR_LOW_LOSS = 1.0   # If val loss < this, reduce lr by 10x for next stage
ADAPTIVE_LR_HIGH_LOSS = 2.0  # If val loss > this, keep lr unchanged

# Hard mining rounds — after stages 1-3, mine hard examples and retrain
HARD_MINING_ROUNDS = 0       # Disabled — using active test loop instead

# Stage configurations
_BATCH_S1 = _env_int("BATCH_STAGE1", 8)
_BATCH_S2 = _env_int("BATCH_STAGE2", 4)
_BATCH_S3 = _env_int("BATCH_STAGE3", 4)
_BATCH_PAR = _env_int("BATCH_PARALLEL", 4)

TRAINING_STAGES = [
    {
        "name": "warmup",
        "stage_num": 1,
        "model": MODEL_PRIMARY,
        "epochs": 50,
        "lr0": 0.01,
        "batch": _BATCH_S1,
        "imgsz": IMGSZ,
        "device": GPU_PRIMARY,
        "freeze": 10,
        "pretrained": True,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 5.0, "translate": 0.1, "scale": 0.5,
        "flipud": 0.0, "fliplr": 0.5,
        "mosaic": 1.0, "mixup": 0.0,
    },
    {
        "name": "finetune",
        "stage_num": 2,
        "model": None,
        "epochs": 60,
        "lr0": 0.001,
        "batch": _BATCH_S2,
        "imgsz": IMGSZ,
        "device": GPU_PRIMARY,
        "freeze": 0,
        "pretrained": False,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 5.0, "translate": 0.1, "scale": 0.5,
        "flipud": 0.0, "fliplr": 0.5,
        "mosaic": 1.0, "mixup": 0.1,
    },
    {
        "name": "polish",
        "stage_num": 3,
        "model": None,
        "epochs": 30,
        "lr0": 0.0001,
        "batch": _BATCH_S3,
        "imgsz": IMGSZ,
        "device": GPU_PRIMARY,
        "freeze": 0,
        "pretrained": False,
        "hsv_h": 0.005, "hsv_s": 0.3, "hsv_v": 0.2,
        "degrees": 0.0, "translate": 0.05, "scale": 0.2,
        "flipud": 0.0, "fliplr": 0.5,
        "mosaic": 0.5, "mixup": 0.0,
    },
]

# Parallel training — diversity model, runs in separate container
PARALLEL_GPU0_CONFIG = {
    "name": "parallel_gpu0",
    "stage_num": 0,
    "model": MODEL_PARALLEL,
    "epochs": 80,
    "lr0": 0.005,
    "batch": _BATCH_PAR,
    "imgsz": IMGSZ,
    "device": "0",  # Always CUDA:0 — container sees only its own GPU
    "freeze": 0,
    "pretrained": True,
    "workers": max(4, WORKERS // 2),
    "hsv_h": 0.02, "hsv_s": 0.8, "hsv_v": 0.5,
    "degrees": 10.0, "translate": 0.15, "scale": 0.6,
    "flipud": 0.01, "fliplr": 0.5,
    "mosaic": 1.0, "mixup": 0.1,
}

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

TRAIN_RATIO = 0.95
RANDOM_SEED = 42
EXPECTED_IMAGE_COUNT = 248
EXPECTED_ANNOTATION_COUNT_MIN = 20_000
EXPECTED_NUM_CATEGORIES = NC

# ---------------------------------------------------------------------------
# Submission constraints (from competition docs)
# ---------------------------------------------------------------------------

SUBMISSION_MAX_ZIP_SIZE_MB = 420       # Max uncompressed zip size
SUBMISSION_MAX_FILES = 1000
SUBMISSION_MAX_PY_FILES = 10
SUBMISSION_MAX_WEIGHT_FILES = 3
SUBMISSION_MAX_WEIGHT_SIZE_MB = 420    # Max total weight file size
SUBMISSION_TIMEOUT_SECONDS = 300

SUBMISSION_ALLOWED_EXTENSIONS = {
    ".py", ".json", ".yaml", ".yml", ".cfg",
    ".pt", ".pth", ".onnx", ".safetensors", ".npy",
}

SUBMISSION_WEIGHT_EXTENSIONS = {
    ".pt", ".pth", ".onnx", ".safetensors", ".npy",
}

# Sandbox environment
SANDBOX_GPU = "NVIDIA L4"
SANDBOX_VRAM_GB = 24
SANDBOX_RAM_GB = 8
SANDBOX_PYTHON = "3.11"
SANDBOX_CUDA = "12.4"

# ---------------------------------------------------------------------------
# Security — blocked imports for submission run.py
# ---------------------------------------------------------------------------

BLOCKED_IMPORTS = {
    "os", "sys", "subprocess", "socket", "ctypes", "builtins",
    "importlib", "pickle", "marshal", "shelve", "shutil", "yaml",
    "requests", "urllib", "http.client", "multiprocessing", "threading",
    "signal", "gc", "code", "codeop", "pty",
}

BLOCKED_CALLS = {"eval", "exec", "compile", "__import__"}

# ---------------------------------------------------------------------------
# Submission inference — read from submission/config.json
# ---------------------------------------------------------------------------

SUBMISSION_MODEL_FILES = _submission_cfg["model_files"]
SUBMISSION_CHECKPOINT_MAPPING = _submission_cfg["checkpoint_mapping"]
CONF_THRESHOLD = _submission_cfg["conf_threshold"]
WBF_IOU_THRESHOLD = _submission_cfg["wbf_iou_threshold"]
WBF_SCORE_THRESHOLD = _submission_cfg["wbf_score_threshold"]
ONNX_OPSET = _submission_cfg["onnx_opset"]


# ---------------------------------------------------------------------------
# Mock endpoint
# ---------------------------------------------------------------------------

MOCK_ENDPOINT_PORT = 8080
MOCK_ENDPOINT_IMAGE_SUBSET_COUNT = 10
