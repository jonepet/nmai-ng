#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

"$(dirname "$0")/sync.sh"

echo "==> Starting training on $REMOTE_HOST..."

# Check if dual-GPU setup (GPU_SECONDARY set in .env.local)
if [ -n "${GPU_SECONDARY:-}" ]; then
  echo "    Dual GPU: train + train-parallel + evaluate"
  ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose --profile dual-gpu run --rm -d test && docker compose --profile dual-gpu up train train-parallel evaluate"
else
  echo "    Single GPU: train + evaluate"
  ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose run --rm -d test && docker compose up train evaluate"
fi
