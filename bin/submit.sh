#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

"$(dirname "$0")/sync.sh"

echo ""
echo "==> Running submission pipeline on $REMOTE_HOST..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose exec train python scripts/submit_pipeline.py"
