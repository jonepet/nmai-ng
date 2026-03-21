#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

"$(dirname "$0")/sync.sh"

SERVICE="${1:-train}"
echo "==> Restarting $SERVICE on $REMOTE_HOST..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose restart $SERVICE"
