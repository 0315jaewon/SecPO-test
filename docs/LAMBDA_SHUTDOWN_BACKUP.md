# Lambda shutdown backup — September 19, 2026

The authoritative bulk backup lives on REDS at:

```text
/data2/jaewonc/secpo_backups/lambda_20260919/
```

Check `status.json` before shutting down Lambda. A completed, verified backup
has `status: "complete"`, `safe_to_shutdown: true`, and an empty `failures` list.
The collection script does not terminate the instance or delete source files.

## Model and checkpoint locations

| Model | Full standalone HF weights | Tinker sampler archive |
|---|---|---|
| Qwen3.8-27B base | `models/base/` | Not applicable |
| SecPO lr=8e-5 | `models/secpo_lr8e5/` | `checkpoints/lr8e5/sampler.tar` |
| SecPO lr=6e-5 | `models/secpo_lr6e5/` | `checkpoints/lr6e5/sampler.tar` |
| SecPO lr=4e-5 | `models/secpo_lr4e5/` | `checkpoints/lr4e5/sampler.tar` |

The full models are copies of the exact BF16 model directories used on Lambda;
using them for inference does not require Tinker access or a fresh adapter merge.
The base revision is `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

`checkpoints/manifest.json` records all six final Tinker checkpoints and the
checksummed local sampler archives. Training-state checkpoints (including
optimizer state) remain on Tinker, private and with `expires_at: null`. Tinker
does not support downloading training-state archives through the endpoint used
here. Thus local copies preserve inference weights, while resuming optimizer
state still depends on access to the recorded Tinker training-state paths.

The three trained run identifiers are:

- lr=8e-5: `fcfc57ba-d2ab-568d-8f90-b3f60ab70560:train:0`
- lr=6e-5: `1b50b5c0-83c3-5a3c-a3e2-faeeded4aa00:train:0`
- lr=4e-5: `1dc6a174-d120-5ddc-884e-257ff14c7210:train:0`

For each run, the final paths end in `weights/secpo-final-state` and
`sampler_weights/secpo-final-sampler`.

## Results and reproduction data

- `workspace/outputs/qwen38_lambda_recommended_20260916/`: authoritative
  generation caches, raw token IDs, episode traces, commands, manifests,
  general/agentic scores, and original GPT-4o-mini utility judgments.
- `workspace/outputs/qwen38_lambda_luna_openai_20260918/`: authoritative **final**
  Luna judgments and scores for all four models, including every judge cache.
- `workspace/outputs/`: also preserves prior Tinker evaluations, training logs,
  older Lambda GPQA results and their existing reproduction snapshots.
- `workspace/{scripts,src,configs,docs,tests}/`: workspace implementation and
  documentation; frozen run-specific code remains inside result directories.
- `isolated_code/{lr4e5,lr6e5}/`: inventory-verified isolated evaluator sources.
- `workspace/data/` and `private_references/`: mentor subset indices, reference
  outputs, length-control assets and the exact approved SEP test/reference JSONs.
- `datasets/`: local cached GPQA, MMLU-Pro, MATH, and SecPO training datasets.
- `benchmark_sources/`: AgentDojo and AgentDyn source trees (without Git internals).
- `environments/`: Python/platform/package versions for the Lambda inference and
  merge environments and the three local evaluation environments.
- `lambda_project/`: the remote project's logs, merge validation reports,
  adapter exports, official converter sources, and original result mirrors.

Prefer `workspace/outputs/` for final scores. `lambda_project/results/` and
`lambda_project/luna_openai_judging_20260918/` are historical remote mirrors and
may contain stale statuses or fewer files; the last Luna mirror sync timed out.

This is a data/source backup, not a relocatable virtual environment or a turnkey
installer. Original commands retain historical absolute paths and the old SSH
endpoint; a new instance needs appropriately adjusted paths, dependencies, and
its own private SSH tunnel. Package inventories and exact source/command
snapshots document the original setup.

## Integrity and security

- `audit/*.remote_sha256.jsonl`: hashes computed on Lambda and verified against
  the copied files on REDS, including every model shard.
- `audit/*.local_source_sha256.jsonl`: source hashes for authoritative local
  result/data/code copies, checked against their backup copies.
- `audit/tasks/*.json`: per-stage completion and verification records.
- `logs/`: resumable transfer and verification logs.
- `identity.json`: source host, directory, model paths, and pinned base revision.
- `audit/shutdown_readiness.json`: all four models and all 28 full benchmark
  results were verified present before declaring the backup complete.
- `SHA256SUMS.models`: portable checksum list for every file in the four model
  directories. To recheck them later, run `sha256sum --check SHA256SUMS.models`
  from the backup root (this reads all model weights and can take some time).

API credential files, SSH private keys, Python virtual environments, caches
unrelated to this experiment, and Git internals are deliberately excluded.
The backup is private and includes model weights and gated/private benchmark
data; do not publish the whole directory or assume it is a shareable code bundle.

## Resume a backup

From the original SecPO workspace, while the Lambda host is still reachable:

```bash
runtime/eval_envs/lm/bin/python -u scripts/archive_lambda_shutdown.py --detach
```

Transfers reuse existing destination files; source files are never removed.
Do not start overlapping collectors. `--wait-for-lock --detach` can queue a
fresh verification pass after an existing collector exits.
