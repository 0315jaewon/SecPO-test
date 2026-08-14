# Qwen3.6 DTap integration

DTap (DecodingTrust-Agent) is an external Apache-2.0 dependency pinned to
commit `e0323a521ba4ef88f8e14c1eccf68d0a3d19a458`. The release also pins the
reviewed DTap patch, Qwen input-role template, and `qwen-agent==0.0.34`.
The release patch is a security-cleaned derivative of the workspace patch,
not a byte-for-byte copy: `DTAP_QWEN_BASE_URL` is mandatory and there is no
implicit model-service address; it also freezes the released main-experiment
mode with thinking disabled. The audited source-patch SHA-256 was
`e80552a24cf3db6a905a0007c94b5295189cb4c45956a89fb6e648dbd0fca684`;
the cleaned release-patch SHA-256 is
`c63e22b067719827741e4847932ce6629010509a28d0f0fb4ea8833485b3185d`.
The clean Qwen-Agent adapter SHA-256 is
`4d9293fdf8f0ed7d03a0e55a46de7a81db22b30f56da2501cb8486d27d21ac9b`.

The runnable bridge is
`src/secpo/evaluation/dtap_qwen_agent_adapter.py`; the dependency-light task
callable is `src/secpo/evaluation/dtap_qwen_agent_backend.py:run_task`. For each
task, the callable invokes DTap's real `python -m eval.evaluation` lifecycle
with an exact one-row JSONL and `--max-parallel 1`. DTap therefore owns task
setup, Compose environments, MCP servers, the Qwen-Agent loop, both judges,
and shutdown. The wrapper accepts a nonzero DTap exit only when a fresh,
bounded regular `judge_result.json` contains explicit boolean utility and
attack outcomes without a judge error.

This is the final clean protocol extracted from the thinking-off SecPO Qwen
profile, not the older thinking-on runner:

- trusted instruction role: `user`;
- untrusted tool/environment content role: `input`;
- 1,838 indirect non-VM tasks, excluding Windows and macOS environments;
- task-list file SHA-256
  `ca0f2fc5728491561e4645ed25390b6311fc9831ecb9b72660d5f513fd6d8098`;
- ordered semantic task SHA-256
  `8ec56c88f064329fe3ee2fdf5472e09c64448be0f0e7d4ae62a33d436d56d1f5`;
- canonical ordering `(domain, type, threat_model, risk_category, task_id)`;
- 262,144-token context and 32,768-token output cap;
- 33 deterministic CPU shards;
- at most six concurrent task calls per worker, 198 aggregate; and
- fingerprinted selection, shard, run, resume, and summary state.

The exact task list is bundled at `data/dtap_qwen36_non_vm_tasks.jsonl`; it is
not a placeholder or download recipe. Every selected task payload remains
Git-tracked in the pinned DTap commit, and runtime preflight rejects modified
or untracked payload files.

The semantic digest is SHA-256 over compact, key-sorted JSON for the tasks in
source-file order, spelling the five identity fields as `domain`,
`task_type`, `task_id`, `threat_model`, and `risk_category`. Sharding then
sorts by the separate canonical tuple above and takes round-robin slices
`sorted_tasks[shard_id::33]`. Both the raw file and semantic digest must match;
an arbitrary 1,838-row substitute is rejected.

The historical profile used a benign-reference checkpoint trained at learning
rate `1.2e-4`. The main Qwen training configuration in this artifact uses
`1.6e-4`; the DTap protocol adapter must not be interpreted as evidence that
those checkpoints are identical.

## Workflow

Prerequisites are Linux, Python 3.11 or newer, Docker Engine with the Docker
Compose v2 plugin, and `uv`. Podman is not supported by this release. Before a
large run, pull the upstream task images and prewarm every selected MCP
project, Node helper, and browser bundle. Runtime sets `UV_NO_SYNC=1`, so task
workers will not resolve or install Python dependencies while 198 calls are
active.

Install the release and its Qwen-Agent extra first:

```bash
python -m pip install -e '.[dtap]'
```

Clone DTap, detach the exact reviewed revision, and apply the released patch:

```bash
git clone https://github.com/AI-secure/DecodingTrust-Agent.git \
  /path/to/DecodingTrust-Agent
git -C /path/to/DecodingTrust-Agent checkout --detach \
  e0323a521ba4ef88f8e14c1eccf68d0a3d19a458
git -C /path/to/DecodingTrust-Agent apply \
  /absolute/path/to/SecPO/third_party/patches/dtap.patch
```

Verify the patched external checkout:

```bash
python scripts/evaluate_dtap_qwen36_27b.py verify \
  --checkout /path/to/DecodingTrust-Agent \
  --patch third_party/patches/dtap.patch \
  --template templates/dtap_qwen36_input_role.jinja \
  --qwen-agent-license /path/to/qwen-agent/LICENSE
```

Verification intentionally precedes importing or editable-installing DTap:
those operations can create `__pycache__` or package metadata that the
full-tree verifier correctly rejects as extra state. After verification,
install the patched checkout into the same environment and suppress bytecode
inside the checkout during runs:

```bash
python -m pip install -e /path/to/DecodingTrust-Agent
export PYTHONDONTWRITEBYTECODE=1
```

Pre-create the `uv` environments after the exact-tree verification (because
`.venv` directories are intentionally rejected by that verifier). Projects
with a lock file use the lock; the two upstream projects without one resolve
once during this preparation step:

```bash
find /path/to/DecodingTrust-Agent/dt_arena -name pyproject.toml -print0 |
  while IFS= read -r -d '' project; do
    project_dir=$(dirname "$project")
    if test -f "$project_dir/uv.lock"; then
      (cd "$project_dir" && UV_NO_SYNC=0 uv sync --frozen)
    else
      (cd "$project_dir" && UV_NO_SYNC=0 uv sync)
    fi
  done
```

Prepare the exact reviewed task list into deterministic shards:

```bash
python scripts/evaluate_dtap_qwen36_27b.py prepare-shards \
  --shards-dir /tmp/dtap-shards
```

Omitting `--task-list` selects the bundled, checksum-pinned 1,838-row file.
Supplying another file is permitted only when its raw and semantic checksums
still match that exact manifest.

Supply deployment-specific values only at runtime. `SECPODTAP_MODEL` is the
served alias for the evaluated SecPO checkpoint. The model server must use
`templates/dtap_qwen36_input_role.jinja`, a 262,144-token context limit, and
the reviewed thinking-off mode. `SECPODTAP_RUNTIME_ROOT` and the Playwright
bundle must be direct absolute directories. The launcher derives per-shard
runtime roots, port ranges, and lock directories beneath that root.
`DTAP_QWEN_BASE_URL` must be an absolute HTTP(S) URL without a username,
password, query, or fragment.

The victim service is credential-free by default. If that deployment requires
authentication, set `DTAP_QWEN_API_KEY` and opt it in explicitly with
`--pass-env DTAP_QWEN_API_KEY` on `run-shard` or `run-all-shards`. Merely
having the variable in the parent shell does not expose it to DTap. No other
credential name is accepted by these DTap commands.

The selected research tasks also require an external OpenAI-compatible harm
judge. Its base URL, model alias, timeout, and credential are mandatory
provider-neutral runtime inputs. The backend strips a trailing slash, rejects
credentials embedded in the URL, maps the values to DTap's `OPENAI_*`
interface, and appends the chat-completions path with exactly one `/v1`.
Neither the endpoint, alias, nor credential is persisted. Resume identity
records only a SHA-256 digest over the normalized non-secret judge settings;
the credential is deliberately excluded.

```bash
export SECPODTAP_CHECKOUT=/path/to/DecodingTrust-Agent
export SECPODTAP_RESULTS_ROOT=/absolute/path/to/dtap-native-results
export SECPODTAP_MODEL=<served-secpo-checkpoint-alias>
export SECPODTAP_MODEL_IDENTITY_SHA256=<checkpoint-manifest-sha256>
export DTAP_QWEN_BASE_URL=<runtime-model-service-base-url>
export SECPODTAP_RUNTIME_ROOT=/absolute/path/to/dtap-runtime
export SECPODTAP_PLAYWRIGHT_BROWSERS_PATH=/absolute/path/to/playwright-browsers
export SECPODTAP_JUDGE_BASE_URL=<compatible-judge-base-url>
export SECPODTAP_JUDGE_MODEL=<compatible-judge-model-alias>
export SECPODTAP_JUDGE_API_KEY=<runtime-credential>
export SECPODTAP_JUDGE_TIMEOUT_SECONDS=60

install -d -m 700 "$SECPODTAP_RUNTIME_ROOT" \
  "$SECPODTAP_PLAYWRIGHT_BROWSERS_PATH"
PLAYWRIGHT_BROWSERS_PATH="$SECPODTAP_PLAYWRIGHT_BROWSERS_PATH" \
  python -m playwright install chromium

export SECPODTAP_JUDGE_IDENTITY_SHA256="$(python - <<'PY'
import hashlib
import json
import os

identity = {
    "base_url": os.environ["SECPODTAP_JUDGE_BASE_URL"].rstrip("/"),
    "model": os.environ["SECPODTAP_JUDGE_MODEL"],
    "timeout_seconds": int(os.environ["SECPODTAP_JUDGE_TIMEOUT_SECONDS"]),
}
payload = json.dumps(
    identity, sort_keys=True, separators=(",", ":")
).encode("utf-8")
print(hashlib.sha256(payload).hexdigest())
PY
)"
```

Run one shard with the released concrete backend:

```bash
python scripts/evaluate_dtap_qwen36_27b.py run-shard \
  --shards-dir /tmp/dtap-shards \
  --results-dir /tmp/dtap-results \
  --shard-id 0 \
  --backend secpo.evaluation.dtap_qwen_agent_backend:run_task
```

For a victim endpoint that requires the optional credential, append
`--pass-env DTAP_QWEN_API_KEY` to that command after exporting the named
variable. Omit the flag for a credential-free local endpoint.

Or launch the exact 33-process by six-thread topology without a cluster
scheduler:

```bash
python scripts/evaluate_dtap_qwen36_27b.py run-all-shards \
  --shards-dir /tmp/dtap-shards \
  --results-dir /tmp/dtap-results \
  --backend secpo.evaluation.dtap_qwen_agent_backend:run_task
```

Then summarize only a complete, fingerprint-compatible result set:

```bash
python scripts/evaluate_dtap_qwen36_27b.py summarize \
  --shards-dir /tmp/dtap-shards \
  --results-dir /tmp/dtap-results \
  --output /tmp/dtap-summary.json
```

The backend receives one decoded task plus the frozen protocol and returns
boolean `task_success` and `attack_success` fields from DTap's native judge.
Exceptions are stored only by type; backend error text and runtime connection
settings are never persisted.

Each of the 33 workers receives a disjoint configured-port offset and a
disjoint 1,000-port fallback range under a shard-local file lock. Six attempts
per shard share only immutable/prewarmed caches; HOME, temporary files, XDG
configuration/state/data, MCP logs, Compose prefixes, and lifecycle registries
are task-private. Normal shutdown writes a PID-bound cleanup marker only after
every Compose environment is verified stopped. Timeout handling then cleans
the evaluation session, the nested task-runner session, and, only when that
marker is absent or invalid, exact-label Docker resources under a run-wide
cleanup lock. Tool calls are bounded at 30 seconds and model requests at 1,800
seconds.

The clean release deliberately omits the research runtime-variant chain,
provider-specific judge routing, service leases, cluster node policy,
scheduler wrappers, renewal processes, logs, and generated results. DTap's
Git-tracked task payloads come from the pinned checkout; containers, MCP
dependencies, judge service, evaluated checkpoint, and model service remain
external requirements. Their absence is an infrastructure prerequisite, not
a missing task callable.
