#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

"$(dirname "$0")/sync.sh"

echo "==> Running hard example mining on $REMOTE_HOST..."
echo "    (finds model failures, generates targeted augmentations)"
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose run --rm hard-mining"
echo "==> Hard mining complete. Run bin/train.sh to retrain with hard examples."
