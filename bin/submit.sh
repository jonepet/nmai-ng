#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

"$(dirname "$0")/sync.sh"

echo ""
echo "============================================================"
echo "  FULL SUBMISSION PIPELINE"
echo "  export → embeddings → package → sandbox test → score"
echo "============================================================"
echo ""

echo "==> Step 1: Exporting models to submission/..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose run --rm export"

echo ""
echo "==> Step 2: Computing product reference embeddings..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose run --rm compute-embeddings"

echo ""
echo "==> Step 3: Packaging submission.zip..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose run --rm package"

echo ""
echo "==> Step 4: Running sandbox test (CPU, locked down, no network)..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && mkdir -p /tmp/sandbox_output && docker compose run --rm sandbox"
