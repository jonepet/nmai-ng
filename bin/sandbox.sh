#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

"$(dirname "$0")/sync.sh"

echo "==> Running sandbox submission test on $REMOTE_HOST..."
echo ""
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose run --rm --build sandbox"
