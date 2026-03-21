#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

echo "==> Syncing project files to $REMOTE_HOST..."
rsync -rltz --omit-dir-times --progress "${RSYNC_EXCLUDES[@]}" "$PROJECT_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"

echo "==> Syncing zip files (only if missing on remote)..."
for zip in "${ZIP_FILES[@]}"; do
  if [ -f "$PROJECT_DIR/$zip" ]; then
    ssh "$REMOTE_HOST" "test -f $REMOTE_DIR/$zip" 2>/dev/null \
      && echo "  Skipping $zip (exists on remote)" \
      || { echo "  Uploading $zip..."; rsync -avz --progress "$PROJECT_DIR/$zip" "$REMOTE_HOST:$REMOTE_DIR/"; }
  fi
done

echo "==> Sync complete."
