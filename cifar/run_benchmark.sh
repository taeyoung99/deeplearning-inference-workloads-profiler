#!/usr/bin/env bash
set -euo pipefail

MODELS="${MODELS:-ResNet18,ResNet50,EfficientNetB0,ShuffleNetV2_1.0}"
BATCH_SIZES="${BATCH_SIZES:-64,128}"
EPOCHS="${EPOCHS:-20}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
GPU_IDS="${GPU_IDS:-0}"
DATA_DIR="${DATA_DIR:-/home/taeyoung/data}"
AMP_FLAG="${AMP_FLAG:---amp}"

python Lucid/workloads/cifar/benchmark.py \
  --models "${MODELS}" \
  --batch-sizes "${BATCH_SIZES}" \
  --epochs "${EPOCHS}" \
  --warmup-epochs "${WARMUP_EPOCHS}" \
  --gpu-ids "${GPU_IDS}" \
  --data-dir "${DATA_DIR}" \
  ${AMP_FLAG}
