#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${RELEASE_ROOT}"

: "${DATA_FILE:?Set DATA_FILE to a prepared offline-preference JSON or JSONL file.}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-27B}"
MODEL_REVISION="${MODEL_REVISION:-6a9e13bd6fc8f0983b9b99948120bc37f49c13e9}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen36_27b_offline}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"

if [[ ! -f "${DATA_FILE}" ]]; then
  echo "DATA_FILE does not exist or is not a regular file: ${DATA_FILE}" >&2
  exit 2
fi
if ! command -v "${ACCELERATE_BIN}" >/dev/null 2>&1; then
  echo "accelerate was not found; activate the release environment first." >&2
  exit 127
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

exec "${ACCELERATE_BIN}" launch \
  --config_file configs/accelerate_zero3_8gpu.yaml \
  --module secpo.training_cli \
  --config configs/train_qwen36_27b_offline.json \
  --data-file "${DATA_FILE}" \
  --model-name-or-path "${MODEL_NAME}" \
  --model-revision "${MODEL_REVISION}" \
  --output-dir "${OUTPUT_DIR}"
