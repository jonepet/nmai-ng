#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

"$(dirname "$0")/sync.sh"

echo "==> Running sandbox submission test on $REMOTE_HOST..."
echo "    (exact replica: 4 vCPU, 8GB RAM, 1 GPU, no network, 300s timeout)"
echo ""
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose run --rm sandbox"
