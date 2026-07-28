#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${RELEASE_ROOT}"

DATASET_DIRECTORY="data/qwen36_offline_dataset"
DATASET_MANIFEST="${DATASET_DIRECTORY}/manifest.json"
PYTHON_BIN="${PYTHON_BIN:-python3}"
status=0

report_matches() {
  local title="$1"
  local pattern="$2"
  local matches
  matches="$(
    rg --hidden --line-number --no-heading --color never \
      --glob '!.git/**' \
      --glob '!outputs/**' \
      --glob '!checkpoints/**' \
      --glob '!logs/**' \
      --glob '!runs/**' \
      --glob '!wandb/**' \
      --glob '!scripts/audit_anonymity.sh' \
      --regexp "${pattern}" . || true
  )"
  if [[ -n "${matches}" ]]; then
    echo "FAIL: ${title}" >&2
    echo "${matches}" >&2
    status=1
  fi
}

if ! command -v rg >/dev/null 2>&1; then
  echo "FAIL: audit requires ripgrep (rg)." >&2
  exit 2
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "FAIL: audit Python executable was not found: ${PYTHON_BIN}" >&2
  exit 2
fi

symlinks="$(find . -type l -print)"
if [[ -n "${symlinks}" ]]; then
  echo "FAIL: symbolic links are not allowed in the release:" >&2
  echo "${symlinks}" >&2
  status=1
fi

generated_directories="$(
  find . -type d \
    \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.venv' \
       -o -name 'build' -o -name 'dist' -o -name '*.egg-info' \
       -o -name 'outputs' -o -name 'checkpoints' -o -name 'logs' \
       -o -name 'runs' -o -name 'wandb' \) \
    -print
)"
if [[ -n "${generated_directories}" ]]; then
  echo "FAIL: generated, cached, or local-environment directories found:" >&2
  echo "${generated_directories}" >&2
  status=1
fi

generated_artifacts="$(
  find . -type f \
    \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' \
       -o -name '*.pth' -o -name '*.ckpt' -o -name '*.pem' \
       -o -name '*.key' -o -name '*.env' -o -name '*.log' \
       -o -name 'trainer_state.json' -o -name 'adapter_config.json' \) \
    -print
)"
if [[ -n "${generated_artifacts}" ]]; then
  echo "FAIL: generated model, log, environment, or key files found:" >&2
  echo "${generated_artifacts}" >&2
  status=1
fi

unexpected_data="$(
  find data -type f \
    ! -name 'README.md' \
    ! -name 'DATASET_CARD.md' \
    ! -name 'example_preferences.jsonl' \
    ! -path "${DATASET_MANIFEST}" \
    ! -path "${DATASET_DIRECTORY}/dataset.json.gz.part-*" \
    -print
)"
if [[ -n "${unexpected_data}" ]]; then
  echo "FAIL: undocumented data files found in the release:" >&2
  echo "${unexpected_data}" >&2
  status=1
fi

large_files="$(
  find . -type f -size +5M \
    ! -path "./${DATASET_DIRECTORY}/dataset.json.gz.part-*" \
    -print
)"
if [[ -n "${large_files}" ]]; then
  echo "FAIL: unexpectedly large files found:" >&2
  echo "${large_files}" >&2
  status=1
fi

if [[ ! -f "${DATASET_MANIFEST}" ]]; then
  echo "FAIL: release dataset manifest is missing: ${DATASET_MANIFEST}" >&2
  status=1
else
  oversized_parts="$(
    find "${DATASET_DIRECTORY}" -type f \
      -name 'dataset.json.gz.part-*' \
      -size +7000000c \
      -print
  )"
  if [[ -n "${oversized_parts}" ]]; then
    echo "FAIL: dataset parts exceed the 7,000,000-byte service budget:" >&2
    echo "${oversized_parts}" >&2
    status=1
  fi
  dataset_temporary="$(
    mktemp /tmp/secpo-release-dataset-audit.XXXXXXXX.json
  )"
  rm -f -- "${dataset_temporary}"
  if ! "${PYTHON_BIN}" scripts/reconstruct_dataset.py \
    --manifest "${DATASET_MANIFEST}" \
    --output "${dataset_temporary}"; then
    echo "FAIL: release dataset reconstruction failed." >&2
    status=1
  elif ! "${PYTHON_BIN}" scripts/audit_release_dataset.py \
    "${dataset_temporary}" \
    --schema raw \
    --expected-records 19157; then
    echo "FAIL: release dataset content audit failed." >&2
    status=1
  fi
  rm -f -- "${dataset_temporary}"
fi

out_of_scope_paths="$(
  find . -mindepth 1 \
    \( -iname '*ablation*' -o -iname '*baseline*' -o -iname '*result*' \
       -o -iname '*benchmark*' -o -iname '*evaluation*' \
       -o -iname '*slurm*' \) \
    -print
)"
if [[ -n "${out_of_scope_paths}" ]]; then
  echo "FAIL: out-of-scope experiment or scheduler paths found:" >&2
  echo "${out_of_scope_paths}" >&2
  status=1
fi

report_matches \
  "machine-local absolute storage path" \
  '/(home|Users|lustre|checkpoint|scratch|gpfs|nfs|fsx)/'
report_matches \
  "email address" \
  '[[:alnum:]._%+-]+@[[:alnum:].-]+\.[[:alpha:]]{2,}'
report_matches \
  "private-key material" \
  '-----BEGIN ([A-Z0-9 ]+ )?PRIVATE KEY-----'
report_matches \
  "credential-like token" \
  '(hf|sk|ghp|glpat)_[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}'
report_matches \
  "embedded credential assignment" \
  '(api[_-]?key|access[_-]?token|secret[_-]?key)[[:space:]]*[:=][[:space:]]*["'\''][^"'\'']+'
report_matches \
  "cluster scheduler directive" \
  '(^|[[:space:]])#SBATCH([[:space:]]|$)|(^|[[:space:]])srun([[:space:]]|$)'

if [[ "${status}" -ne 0 ]]; then
  echo "Anonymous-release audit failed." >&2
  exit "${status}"
fi

echo "Anonymous-release audit passed."
