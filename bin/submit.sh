#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

"$(dirname "$0")/sync.sh"

echo ""
echo "============================================================"
echo "  FULL SUBMISSION PIPELINE"
echo "  export → package → sandbox test → score"
echo "============================================================"
echo ""

echo "==> Step 1: Exporting models to submission/ on $REMOTE_HOST..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose run --rm export"

echo ""
echo "==> Step 2: Packaging submission.zip on $REMOTE_HOST..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose run --rm package"

echo ""
echo "==> Step 3: Running sandbox test (CPU, locked down, no network)..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && mkdir -p /tmp/sandbox_output && docker compose run --rm sandbox"
