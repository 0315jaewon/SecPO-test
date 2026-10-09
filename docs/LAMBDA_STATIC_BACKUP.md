# Static-security evaluation backup — September 24, 2026

Backup location on REDS:
`/data2/jaewonc/secpo_backups/lambda_static_20260924/`

Check `status.json`: only `status: complete` and `safe_to_shutdown: true`
confirm successful completion. The archival script does not shut down Lambda.

## Contents

- `native_run/`: the completed lr=8e-5 and lr=4e-5 AgentDojo/AgentDyn
  static-security evaluations, including results, manifests, every task record,
  upstream episode trajectories, complete API request/response caches, smoke
  runs, source snapshots, and launch/worker logs.
- `superseded_run/`: the earlier custom-parser experiment and failure diagnostics,
  preserved separately and not mixed with final results.
- `lambda_project/native_server/` and `lambda_project/superseded_server/`:
  server scripts, logs, launch arguments, model validation records, and the exact
  configured Qwen3.8 chat template copied from `209.20.159.198`.
- `workspace/`: evaluation/launch/backup code, configs, docs, tests, and patches.
- `benchmark_sources/`: the pinned AgentDojo and AgentDyn source/data trees.
- `environments/`: installed-package inventories for Lambda inference and the
  REDS evaluation environments.
- `audit/`: SHA-256 inventories and evaluation/model preservation checks.

All newly copied run artifacts are SHA-256 compared with their source. Remote
server artifacts are also hashed before and after transfer.

## Models and earlier experiments

Large model files are not duplicated. Keep this existing backup alongside the
new one:
`/data2/jaewonc/secpo_backups/lambda_20260919/`

- `models/base/`, `models/secpo_lr8e5/`, `models/secpo_lr6e5/`, and
  `models/secpo_lr4e5/`: standalone model weights and tokenizers.
- `checkpoints/{lr8e5,lr6e5,lr4e5}/sampler.tar`: original Tinker adapter exports.
- `checkpoints/manifest.json`: checkpoint identities and earlier retention check.
- `lambda_project/`: the already archived September 16-19 Lambda project,
  including older results, exports, judging records, merge code, and server logs.

This backup freshly verifies the sampler archive SHA-256 hashes and checks every
standalone model file's inventory and size. For the two models evaluated here,
the instance's full-SHA-256 validation from its September 24 preparation is
matched against the earlier fully verified REDS model manifest. The large model
shards are not fully rehashed or recopied by this archival script.

Training/optimizer-state checkpoints live on Tinker, not Lambda; this backup
does not modify their retention. Earlier thinking/judge experiments also remain
in `/data2/jaewonc/secpo_backups/lambda_thinking_20260923/`.

Credentials, SSH keys, installed virtual environments, and rebuildable caches
are deliberately excluded. No source files are deleted.

## Command

From `/home/jaewonc/SecPO`, with SSH access and write permission to `/data2/jaewonc`:

```bash
python scripts/archive_lambda_static.py
```

For the evaluation launch commands, see
`workspace/docs/LAMBDA_NATIVE_STATIC_SECURITY.md` inside the backup.
