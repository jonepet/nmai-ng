#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

"$(dirname "$0")/sync.sh"

echo ""
echo "============================================================"
echo "  SUBMISSION PIPELINE"
echo "  export → package → test → score"
echo "============================================================"

echo ""
echo "==> Ensuring training is running..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose up -d train"

echo ""
echo "==> Exporting + Packaging..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose exec train python scripts/submit_pipeline.py"

echo ""
echo "==> Building sandbox image..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && DOCKER_BUILDKIT=0 docker compose build sandbox"

echo ""
echo "==> Testing in sandbox..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose run --rm sandbox"

echo ""
echo "==> Copying submission.zip..."
scp "$REMOTE_HOST:$REMOTE_DIR/submission.zip" "$PROJECT_DIR/submission.zip"

SIZE=$(stat --printf="%s" "$PROJECT_DIR/submission.zip" 2>/dev/null || stat -f "%z" "$PROJECT_DIR/submission.zip")
SIZE_MB=$((SIZE / 1048576))

echo ""
echo "============================================================"
echo "  READY: $PROJECT_DIR/submission.zip (${SIZE_MB} MB)"
echo "  Upload at the competition website."
echo "============================================================"
