#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

"$(dirname "$0")/sync.sh"

echo "==> Running data preparation on $REMOTE_HOST..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose run --rm prepare-data"
echo "==> Data preparation complete."
