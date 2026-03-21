#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

MODE="${1:-tail}"

# Strip ANSI codes, carriage returns, docker compose prefixes, and blank lines
clean() {
  tr -d '\r' \
    | sed -u 's/\x1b\[[0-9;]*m//g; s/^train-1[[:space:]]*|[[:space:]]*//' \
    | grep -v --line-buffered '^\s*$'
}

# Extract only epoch result lines (the "all" summary) into a table
epoch_table() {
  clean | awk '
    /all .* [0-9]/ {
      n++
      printf "  Epoch %-3d | P=%-8s R=%-8s mAP50=%-10s mAP50-95=%-10s\n", n, $4, $5, $6, $7
    }
    /\[stage/ {
      gsub(/^[[:space:]]+/, "")
      printf "\n  >> %s\n", $0
    }
    /Completed in|val_loss|Selecting|Final best|OOM|ERROR|FAIL|checkpoint|saved/ {
      gsub(/^[[:space:]]+/, "")
      printf "  ** %s\n", $0
    }
  '
}

case "$MODE" in
  tail)
    echo "==> Live metrics from $REMOTE_HOST (Ctrl+C to stop)"
    echo ""
    printf "  %-9s | %-8s %-8s %-10s %-10s\n" "Epoch" "P" "R" "mAP50" "mAP50-95"
    printf "  %s\n" "--------- | -------- -------- ---------- ----------"
    ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose logs -f train 2>&1" | epoch_table
    ;;

  poll)
    INTERVAL="${2:-30}"
    echo "==> Polling every ${INTERVAL}s from $REMOTE_HOST (Ctrl+C to stop)"
    while true; do
      clear
      echo "  NorgesGruppen Training — $(date '+%Y-%m-%d %H:%M:%S')"
      echo ""

      echo "  CONTAINERS"
      ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose ps -a --format '{{.Service}}\t{{.Status}}' 2>&1" \
        | while IFS=$'\t' read -r svc status; do printf "    %-16s %s\n" "$svc" "$status"; done
      echo ""

      echo "  GPUs"
      ssh "$REMOTE_HOST" "python3 $REMOTE_DIR/bin/gpu_status.py" 2>/dev/null
      echo ""

      echo "  LAST 10 EPOCHS"
      printf "    %-9s | %-8s %-8s %-10s %-10s\n" "Epoch" "P" "R" "mAP50" "mAP50-95"
      printf "    %s\n" "--------- | -------- -------- ---------- ----------"
      ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose logs --tail=500 train 2>&1" \
        | tr -d '\r' | sed 's/\x1b\[[0-9;]*m//g; s/^train-1[[:space:]]*|[[:space:]]*//' \
        | awk '/all .* [0-9]/ { n++; printf "    Epoch %-3d | P=%-8s R=%-8s mAP50=%-10s mAP50-95=%-10s\n", n, $4, $5, $6, $7 }' \
        | tail -10
      echo ""

      echo "  EVENTS"
      ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose logs --tail=1000 train 2>&1" \
        | tr -d '\r' | sed 's/\x1b\[[0-9;]*m//g; s/^train-1[[:space:]]*|[[:space:]]*//' \
        | grep -E '(stage.*WARMUP|stage.*FINETUNE|stage.*POLISH|stage.*UPGRADE|parallel|Completed|val_loss|Final|OOM|ERROR|FAIL|checkpoint|saved)' \
        | tail -8 \
        | while IFS= read -r line; do printf "    %s\n" "$line"; done
      echo ""

      echo "  CHECKPOINTS"
      ssh "$REMOTE_HOST" "ls -lhS $REMOTE_DIR/checkpoints/*.pt 2>/dev/null" \
        | awk '{printf "    %-40s %s\n", $NF, $5}' \
        || echo "    (none yet)"
      echo ""

      sleep "$INTERVAL"
    done
    ;;

  snapshot)
    echo ""
    echo "  NorgesGruppen Training Snapshot — $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    echo "  CONTAINERS"
    ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose ps -a --format '{{.Service}}\t{{.Status}}' 2>&1" \
      | while IFS=$'\t' read -r svc status; do printf "    %-16s %s\n" "$svc" "$status"; done
    echo ""

    echo "  GPUs"
    ssh "$REMOTE_HOST" "python3 $REMOTE_DIR/bin/gpu_status.py" 2>/dev/null
    echo ""

    echo "  LAST 15 EPOCHS"
    printf "    %-9s | %-8s %-8s %-10s %-10s\n" "Epoch" "P" "R" "mAP50" "mAP50-95"
    printf "    %s\n" "--------- | -------- -------- ---------- ----------"
    ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose logs --tail=1000 train 2>&1" \
      | tr -d '\r' | sed 's/\x1b\[[0-9;]*m//g; s/^train-1[[:space:]]*|[[:space:]]*//' \
      | awk '/all .* [0-9]/ { n++; printf "    Epoch %-3d | P=%-8s R=%-8s mAP50=%-10s mAP50-95=%-10s\n", n, $4, $5, $6, $7 }' \
      | tail -15
    echo ""

    echo "  CHECKPOINTS"
    ssh "$REMOTE_HOST" "ls -lhS $REMOTE_DIR/checkpoints/*.pt 2>/dev/null" \
      | awk '{printf "    %-40s %s\n", $NF, $5}' \
      || echo "    (none yet)"
    echo ""
    ;;

  *)
    echo "Usage: bin/metrics.sh [tail|poll|snapshot]"
    echo ""
    echo "  tail           Live stream epoch results as table (continuous)"
    echo "  poll [sec]     Dashboard refreshing every N seconds (default: 30)"
    echo "  snapshot       One-shot status overview"
    ;;
esac
