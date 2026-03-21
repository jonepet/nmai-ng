#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

SERVICE="${1:-train}"
LINES="${2:-50}"

echo "==> Tailing $SERVICE logs from $REMOTE_HOST (last $LINES lines)..."
# Filter out progress bar noise — only show completed epochs, metrics, and stage transitions
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose logs --tail=$LINES -f $SERVICE 2>&1" \
  | sed -u 's/.*\r//' \
  | grep -v --line-buffered '^\s*$'
