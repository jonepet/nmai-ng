#!/usr/bin/env bash
# Single source of truth for shell script configuration.
# All bin/*.sh scripts source this file.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load local config (IP addresses, not in git)
ENV_LOCAL="$PROJECT_DIR/.env"
if [ ! -f "$ENV_LOCAL" ]; then
  echo "ERROR: $ENV_LOCAL not found. Create it with:" >&2
  echo "  REMOTE_HOST=<ip>" >&2
  echo "  REMOTE_DIR=~/nmai-ng" >&2
  exit 1
fi
source "$ENV_LOCAL"

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
