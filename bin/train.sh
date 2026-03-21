#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

"$(dirname "$0")/sync.sh"

echo "==> Starting on $REMOTE_HOST:"
echo "    - train:          YOLOv8s on GTX 1050 Ti (4GB)"
echo "    - train-parallel: YOLOv8n on GTX 960 (2GB)"
echo "    - evaluate:       checkpoint watcher"
echo "    - test:           test suite"
echo ""
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose run --rm -d test && docker compose up train train-parallel evaluate"
