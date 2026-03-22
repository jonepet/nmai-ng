#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

"$(dirname "$0")/sync.sh"

echo ""
echo "============================================================"
echo "  FULL SUBMISSION PIPELINE"
echo "  export → package → test → score"
echo "============================================================"

echo ""
echo "==> Ensuring training is running..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose up -d train evaluate"

echo ""
echo "==> Step 1/3: Exporting + Packaging..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose exec train python scripts/submit_pipeline.py"

echo ""
echo "==> Step 2/3: Copying submission.zip locally..."
scp "$REMOTE_HOST:$REMOTE_DIR/submission.zip" "$PROJECT_DIR/submission.zip"

echo ""
echo "==> Step 3/3: Testing in sandbox environment..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose run --rm --build sandbox"
