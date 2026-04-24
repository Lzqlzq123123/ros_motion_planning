#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/src/rl_training/config/forklift_td3_noadv_nocurriculum.yaml}"
DEVICE="${DEVICE:-cuda}"

echo "[train] Using config: ${CONFIG_PATH}"
echo "[train] Device: ${DEVICE}"

cd "$ROOT_DIR"
python src/rl_training/train_velodyne_td3.py \
  --config "$CONFIG_PATH" \
  --device "$DEVICE"
