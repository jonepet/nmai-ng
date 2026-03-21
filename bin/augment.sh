#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

"$(dirname "$0")/sync.sh"

echo "==> Running data augmentation on $REMOTE_HOST..."
echo "    (10 augmentation pipelines x 210 training images = ~2100 new images)"
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose run --rm augment"
echo "==> Augmentation complete."
