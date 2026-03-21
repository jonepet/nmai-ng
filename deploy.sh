#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="192.168.10.118"
REMOTE_DIR="~/nmai-ng"
REMOTE="$REMOTE_HOST:$REMOTE_DIR"

BUILD=false
PREPARE=false
TRAIN=false
EXPORT=false
PACKAGE=false
MOCK_ENDPOINT=false
ALL=false

for arg in "$@"; do
  case $arg in
    --build)          BUILD=true ;;
    --prepare-data)   PREPARE=true ;;
    --train)          TRAIN=true ;;
    --export)         EXPORT=true ;;
    --package)        PACKAGE=true ;;
    --mock-endpoint)  MOCK_ENDPOINT=true ;;
    --all)            ALL=true ;;
    *) echo "Unknown flag: $arg"; exit 1 ;;
  esac
done

# Default (no flags): full pipeline
if ! $BUILD && ! $PREPARE && ! $TRAIN && ! $EXPORT && ! $PACKAGE && ! $MOCK_ENDPOINT && ! $ALL; then
  ALL=true
fi

echo "==> Syncing project files to $REMOTE_HOST..."
rsync -avz --progress \
  --exclude 'data/coco_dataset/' \
  --exclude 'data/product_images/' \
  --exclude 'data/yolo/' \
  --exclude 'checkpoints/' \
  --exclude '__pycache__/' \
  --exclude '.git/' \
  --exclude '*.zip' \
  ./ "$REMOTE/"

echo "==> Syncing zip files (only if missing on remote)..."
for zip in NM_NGD_coco_dataset.zip NM_NGD_product_images.zip; do
  if [ -f "$zip" ]; then
    ssh "$REMOTE_HOST" "test -f $REMOTE_DIR/$zip" 2>/dev/null \
      && echo "  Skipping $zip (already exists on remote)" \
      || { echo "  Uploading $zip..."; rsync -avz --progress "$zip" "$REMOTE/"; }
  fi
done

if $MOCK_ENDPOINT; then
  echo "==> Starting mock endpoint on remote..."
  ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose up mock-endpoint"
  exit 0
fi

if $BUILD; then
  echo "==> Building images on remote..."
  ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose build"
  exit 0
fi

if $PREPARE; then
  echo "==> Running data preparation on remote..."
  ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose run --rm prepare-data"
  exit 0
fi

if $TRAIN; then
  echo "==> Starting training + evaluation watcher + tests in parallel on remote..."
  ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose run --rm -d test && docker compose up train evaluate"
  exit 0
fi

if $EXPORT; then
  echo "==> Running model export on remote..."
  ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose run --rm export"
  exit 0
fi

if $PACKAGE; then
  echo "==> Running packaging on remote..."
  ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose run --rm package"
  exit 0
fi

# --all or default: full pipeline
echo "==> Building images on remote..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose build"

echo "==> Running data preparation on remote..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose run --rm prepare-data"

echo "==> Running tests in background on remote..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose run --rm -d test"

echo "==> Starting training + evaluation watcher in parallel on remote..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose up train evaluate"

echo "==> Running model export on remote..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose run --rm export"

echo "==> Running packaging on remote..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose run --rm package"
