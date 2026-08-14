#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${RELEASE_ROOT}"

: "${DATA_FILE:?Set DATA_FILE to a prepared online-preference JSON or JSONL file.}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_REVISION="${MODEL_REVISION:-0e9e39f249a16976918f6564b8830bc894c89659}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/llama31_8b_online}"
ONLINE_EXCHANGE_DIR="${ONLINE_EXCHANGE_DIR:-runtime/online_exchange}"
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
  --config configs/train_llama31_8b_online.json \
  --data-file "${DATA_FILE}" \
  --model-name-or-path "${MODEL_NAME}" \
  --model-revision "${MODEL_REVISION}" \
  --output-dir "${OUTPUT_DIR}" \
  --online-exchange-dir "${ONLINE_EXCHANGE_DIR}"
