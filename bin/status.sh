#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

echo "==> Container status on $REMOTE_HOST:"
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose ps -a"
