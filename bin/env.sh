#!/usr/bin/env bash
# Single source of truth for shell script configuration.
# All bin/*.sh scripts source this file.

REMOTE_HOST="192.168.10.118"
REMOTE_DIR="~/nmai-ng"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RSYNC_EXCLUDES=(
  --exclude 'data/coco_dataset/'
  --exclude 'data/product_images/'
  --exclude 'data/yolo/'
  --exclude 'data/_yolo_labels_staging/'
  --exclude 'checkpoints/'
  --exclude '__pycache__/'
  --exclude '.git/'
  --exclude '*.zip'
)

ZIP_FILES=(
  NM_NGD_coco_dataset.zip
  NM_NGD_product_images.zip
)
