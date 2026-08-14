#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${RELEASE_ROOT}"

: "${ATTACK_BACKEND:?Set ATTACK_BACKEND to an importable module:callable.}"
ONLINE_EXCHANGE_DIR="${ONLINE_EXCHANGE_DIR:-runtime/online_exchange}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

exec "${PYTHON_BIN}" scripts/serve_online_attack_requests.py \
  --exchange-dir "${ONLINE_EXCHANGE_DIR}" \
  --backend "${ATTACK_BACKEND}" \
  "$@"
