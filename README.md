# NorgesGruppen Object Detection (NM i AI 2026)

Two-stage grocery product detection pipeline for the NorgesGruppen competition track of NM i AI 2026. YOLOv8l detects bounding boxes; EfficientNet-B0 reclassifies low-confidence detections to improve category accuracy. Inference runs entirely in ONNX format via `onnxruntime-gpu`.

## Architecture

```
Image
  └─> YOLOv8l (ONNX) ──── detect boxes + initial class
        └─> WBF ensemble (multi-model or TTA)
              └─> EfficientNet-B0 (ONNX) ── reclassify uncertain boxes
                    └─> predictions.json (COCO format)
```

- **Detection**: YOLOv8l trained in three stages (warmup → finetune → polish) with frozen backbone in stage 1, full fine-tuning thereafter.
- **Classification**: EfficientNet-B0 trained on the reference product images. Applied only to detections below the confidence threshold.
- **Ensemble**: Weighted Boxes Fusion (WBF) merges multi-model or TTA (horizontal flip) outputs.
- **Augmentation**: Offline albumentations pipeline + crop/mosaic synthesis; hard example mining refines the training set across rounds.

## Setup

Prerequisites: Docker, Docker Compose, NVIDIA GPU with drivers installed.

```bash
# 1. Copy and configure environment
cp .env.example .env   # edit GPU IDs, batch sizes, remote host

# 2. Prepare data (extracts zips, converts to YOLO format, splits train/val)
docker compose run --rm prepare-data

# 3. Augment training data (offline augmentation + crop-mosaic)
docker compose run --rm augment
```

## Usage

### Training

```bash
bin/train.sh
```

Syncs the project to the remote training host and launches `docker compose up train evaluate`. Runs a continuous multi-round loop: stages → evaluation → hard mining → repeat.

Train the product classifier in parallel (separate GPU):

```bash
docker compose up train-classifier
```

### Building the submission

```bash
bin/submit.sh
```

Full pipeline: export models to ONNX → package `submission.zip` → test in sandbox → copy locally.

For a manual step-by-step flow with local verification:

```bash
bin/prepare-submission.sh
```

Runs export + package on the remote host, copies `submission.zip` locally, then verifies file counts, sizes, and that `run.py` is at the zip root.

### Other services

| Command | What it does |
|---|---|
| `docker compose run --rm export` | Export best checkpoint to ONNX |
| `docker compose run --rm package` | Package `submission.zip` |
| `docker compose run --rm sandbox` | Test inference in the competition sandbox environment |
| `docker compose run --rm test` | Run test suite |
| `docker compose run --rm hard-mining` | Mine hard examples from the training set |
| `bin/status.sh` | Show training logs from remote host |

## Configuration

All inference parameters (confidence thresholds, WBF settings, model file names, class count) live in `submission/config.json` — the single source of truth used by both `submission/run.py` and `config.py`.

Hardware-specific settings (GPU IDs, batch sizes, remote host) are set in `.env` (not committed).

## Competition constraints

- Submission: `submission.zip` ≤ 420 MB, ≤ 1000 files, ≤ 10 `.py` files, ≤ 3 weight files
- Sandbox: Python 3.11, `onnxruntime-gpu` 1.20.0, NVIDIA L4 (24 GB VRAM), 300 s timeout
- Output: COCO-format JSON, ≤ 50 000 predictions total
