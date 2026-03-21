#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

"$(dirname "$0")/sync.sh"

echo ""
echo "============================================================"
echo "  PREPARE SUBMISSION FOR UPLOAD"
echo "  export → package → verify → copy locally"
echo "============================================================"

# Run the pipeline (export + package + sandbox test)
echo ""
echo "==> Running submission pipeline..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose exec train python scripts/submit_pipeline.py"

# Copy submission.zip to local machine
echo ""
echo "==> Copying submission.zip to local machine..."
scp "$REMOTE_HOST:$REMOTE_DIR/submission.zip" "$PROJECT_DIR/submission.zip"

# Verify the zip locally
echo ""
echo "==> Verifying submission.zip..."
echo ""
echo "  Contents:"
unzip -l "$PROJECT_DIR/submission.zip" 2>&1 | grep -E '^\s+\d' | while read -r size date time name; do
  size_mb=$(echo "scale=2; $size / 1048576" | bc)
  echo "    ${name}  (${size_mb} MB)"
done

echo ""
total_size=$(stat --printf="%s" "$PROJECT_DIR/submission.zip" 2>/dev/null || stat -f "%z" "$PROJECT_DIR/submission.zip" 2>/dev/null)
total_mb=$(echo "scale=2; $total_size / 1048576" | bc)
echo "  Zip file size: ${total_mb} MB"

# Check limits
echo ""
n_files=$(unzip -l "$PROJECT_DIR/submission.zip" | grep -c '^\s\+[0-9]' || true)
n_py=$(unzip -l "$PROJECT_DIR/submission.zip" | grep -c '\.py$' || true)
n_weights=$(unzip -l "$PROJECT_DIR/submission.zip" | grep -cE '\.(pt|pth|onnx|safetensors|npy)$' || true)
echo "  Files: ${n_files} / 1000"
echo "  Python files: ${n_py} / 10"
echo "  Weight files: ${n_weights} / 3"

# Verify run.py is at root
if unzip -l "$PROJECT_DIR/submission.zip" | grep -q '^\s\+[0-9].*\s\+run\.py$'; then
  echo "  run.py at root: YES"
else
  echo "  ERROR: run.py NOT at zip root!"
  exit 1
fi

echo ""
echo "============================================================"
echo "  READY FOR UPLOAD"
echo "============================================================"
echo ""
echo "  File: $PROJECT_DIR/submission.zip"
echo "  Size: ${total_mb} MB"
echo ""
echo "  Upload at the competition submission page."
echo "============================================================"
