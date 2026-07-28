#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${RELEASE_ROOT}"

: "${DATA_FILE:?Set DATA_FILE to a prepared online-preference JSON or JSONL file.}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/llama31-online}"
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
  --module secpo.train \
  --config configs/llama31_online.json \
  --data-file "${DATA_FILE}" \
  --model-name-or-path "${MODEL_NAME}" \
  --output-dir "${OUTPUT_DIR}"
