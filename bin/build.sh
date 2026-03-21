#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

"$(dirname "$0")/sync.sh"

echo "==> Building Docker images on $REMOTE_HOST..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && BUILDKIT_PROGRESS=plain docker compose build $* 2>&1"
echo "==> Build complete."
