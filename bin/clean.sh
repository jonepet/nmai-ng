#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

echo "==> Cleaning data and checkpoints on $REMOTE_HOST..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose run --rm clean-data && docker compose run --rm clean-checkpoints"
echo "==> Clean complete."
