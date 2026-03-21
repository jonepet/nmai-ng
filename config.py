"""
Centralized configuration for NorgesGruppen object detection project.

ALL parameters and settings live here. No magic numbers in other files.
"""

from pathlib import Path

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
# Remote machine
# ---------------------------------------------------------------------------

REMOTE_HOST = "192.168.10.118"
REMOTE_DEPLOY_DIR = "~/nmai-ng"

# ---------------------------------------------------------------------------
# Hardware (remote training machine)
# ---------------------------------------------------------------------------

GPU_PRIMARY = "0"          # GTX 1050 Ti (4GB VRAM, dedicated) — maps to CUDA:0 inside Docker
GPU_SECONDARY = "1"        # GTX 960 (2GB VRAM, runs desktop) — maps to CUDA:1 inside Docker
GPU_PRIMARY_VRAM_GB = 4
GPU_SECONDARY_VRAM_GB = 2
CPU_CORES = 16
RAM_GB = 30
SHM_SIZE = "4g"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

# Primary model: YOLOv8s — best balance for 4GB GPU training
# Submission runs on L4 24GB so inference is unconstrained
MODEL_PRIMARY = "yolov8s.pt"
# Parallel GPU0 model: YOLOv8n — fits in 2GB VRAM
MODEL_PARALLEL = "yolov8n.pt"

NC = 357                   # Number of categories (0-356 including unknown_product)
IMGSZ = 640                # Training and inference image size
IMGSZ_PARALLEL = 640       # Parallel GPU0 image size (same to maximize quality)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

WORKERS = 8                # DataLoader workers per training job
PATIENCE = 10              # Early stopping patience (epochs without improvement)
SAVE_PERIOD = 5            # Save checkpoint every N epochs
COS_LR = True              # Cosine annealing learning rate
AMP = True                 # Automatic Mixed Precision (FP16) — essential for VRAM savings

# Adaptive LR thresholds
ADAPTIVE_LR_LOW_LOSS = 1.0   # If val loss < this, reduce lr by 10x for next stage
ADAPTIVE_LR_HIGH_LOSS = 2.0  # If val loss > this, keep lr unchanged

# Stage configurations
TRAINING_STAGES = [
    {
        "name": "warmup",
        "stage_num": 1,
        "model": MODEL_PRIMARY,       # YOLOv8s pretrained
        "epochs": 30,
        "lr0": 0.01,
        "batch": 8,                   # Fits in 4GB with AMP
        "imgsz": IMGSZ,
        "device": GPU_PRIMARY,
        "freeze": 10,                 # Freeze backbone layers
        "pretrained": True,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 5.0, "translate": 0.1, "scale": 0.5,
        "flipud": 0.0, "fliplr": 0.5,
        "mosaic": 1.0, "mixup": 0.0,
    },
    {
        "name": "finetune",
        "stage_num": 2,
        "model": None,                # Loaded from Stage 1 best
        "epochs": 50,
        "lr0": 0.001,
        "batch": 4,                   # Reduced for unfrozen model
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
        "model": None,                # Loaded from Stage 2 best
        "epochs": 20,
        "lr0": 0.0001,
        "batch": 4,
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

# Parallel GPU0 config (diversity training on GTX 960 2GB, runs concurrently)
# YOLOv8n at batch=1 imgsz=480 uses ~1.2GB VRAM — fits in 2GB with room to spare
PARALLEL_GPU0_CONFIG = {
    "name": "parallel_gpu0",
    "stage_num": 0,
    "model": MODEL_PARALLEL,          # YOLOv8n — 3.4M params, ~6MB
    "epochs": 80,
    "lr0": 0.005,
    "batch": 1,                       # Minimal batch for 2GB VRAM
    "imgsz": 480,                     # Reduced from 640 to fit in 2GB
    "device": GPU_SECONDARY,          # GTX 960 (2GB)
    "freeze": 0,
    "pretrained": True,
    "workers": 4,                     # Share CPU cores with main job
    "hsv_h": 0.02, "hsv_s": 0.8, "hsv_v": 0.5,
    "degrees": 10.0, "translate": 0.15, "scale": 0.6,
    "flipud": 0.01, "fliplr": 0.5,
    "mosaic": 1.0, "mixup": 0.1,
}

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

TRAIN_RATIO = 0.80
RANDOM_SEED = 42
EXPECTED_IMAGE_COUNT = 248
EXPECTED_ANNOTATION_COUNT_MIN = 20_000
EXPECTED_NUM_CATEGORIES = NC  # 357

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
# Ensemble inference (submission)
# ---------------------------------------------------------------------------

# Model weight filenames included in submission zip
SUBMISSION_MODEL_FILES = ["best_main.pt", "best_parallel.pt"]

# Inference scales — run each model at these image sizes, merge with WBF
INFERENCE_SCALES = [640, 1280]

# WBF (Weighted Box Fusion) parameters
WBF_IOU_THRESHOLD = 0.55      # IoU threshold for merging overlapping boxes
WBF_SCORE_THRESHOLD = 0.001   # Minimum score to keep a prediction
WBF_SKIP_BOX_THRESHOLD = 0.0001  # Skip boxes below this score before WBF

# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

ONNX_OPSET = 17

# ---------------------------------------------------------------------------
# Mock endpoint
# ---------------------------------------------------------------------------

MOCK_ENDPOINT_PORT = 8080
MOCK_ENDPOINT_IMAGE_SUBSET_COUNT = 10
