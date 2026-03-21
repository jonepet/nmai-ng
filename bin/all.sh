#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Full pipeline: sync → build → clean → prepare → augment → train+evaluate+test"
./sync.sh
./build.sh
./clean.sh
./prepare.sh
./augment.sh
./train.sh
