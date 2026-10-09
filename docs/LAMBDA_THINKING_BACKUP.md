# September 23 Lambda shutdown backup

Check `/data2/jaewonc/secpo_backups/lambda_thinking_20260923/status.json`.
Only `status: complete` with `safe_to_shutdown: true` indicates that collection
and verification succeeded. The collection script does not terminate Lambda.

## Locations on REDS

- Current experiment backup:
  `/data2/jaewonc/secpo_backups/lambda_thinking_20260923/`
- Original thinking-on artifacts:
  `/data2/jaewonc/secpo_evals/qwen38_thinking_alpaca_20260923/`
- Original GPT-6 Luna judgments of all five answer sets:
  `/data2/jaewonc/secpo_evals/gpt6_luna_alpaca_20260923/`
- Earlier full backup, including all four models and past benchmark runs:
  `/data2/jaewonc/secpo_backups/lambda_20260919/`

These are REDS local-disk directories, not Lambda filesystem mounts. Keep both
backup directories: the new backup intentionally references the old weights
and older runs instead of duplicating hundreds of GB.

## Current backup contents

- `thinking_run/`: all 805 generations, final answers, reasoning and raw token
  caches, judge prompts/configuration identities, mini and GPT-5.6 Luna judge
  caches and results, failure/recovery evidence, frozen source snapshots, logs,
  and exact generation/launch commands.
- `gpt6_judging/`: full GPT-6 Luna judgments and results for thinking-on base
  and thinking-off base/lr8e-5/lr6e-5/lr4e-5, including individual logprobs.
- `judge_shift_analysis/`: the earlier offline judge-comparison analysis.
- `lambda_project/`: remaining remote project files, exact server command,
  server-log snapshot, remote setup script, model validation, package list,
  and setup status. A running server can append idle log lines after capture;
  the archived log is explicitly a point-in-time snapshot.
- `workspace/{scripts,src,configs,docs,tests,data}/`: current source, configs,
  documentation, tests, subset indices, references, and LC metric assets.
- `audit/`: source/copy SHA-256 inventories, environment details, fresh remote
  and local base-weight verification, adapter-archive verification, and saved
  result validation with independent metric recomputation.

## Models and checkpoints

The exact standalone inference weights remain at:

```text
/data2/jaewonc/secpo_backups/lambda_20260919/models/base/
/data2/jaewonc/secpo_backups/lambda_20260919/models/secpo_lr8e5/
/data2/jaewonc/secpo_backups/lambda_20260919/models/secpo_lr6e5/
/data2/jaewonc/secpo_backups/lambda_20260919/models/secpo_lr4e5/
```

The three adapter archives are under that backup's `checkpoints/` directory.
Current collection rehashes the base weights on REDS and Lambda, rehashes the
adapter archives, and verifies the current file inventory/sizes of the three
previously SHA-256-verified merged models. It does not claim a fresh full hash
pass of all three merged models.

Training optimizer states remain on Tinker, not on Lambda; their recorded
paths and no-expiration settings are in the previous backup's
`checkpoints/manifest.json`. Local inference weights do not depend on Tinker
or Lambda. No new Tinker checkpoint mutation is needed for this shutdown.

## Misleading final supervisor failure

The thinking runner saved all results and complete progress, then its final
console `json.dumps` raised `TypeError` while printing metrics. Consequently,
the original supervisor says `failed`. The collection process preserves those
original files and separately checks all 805 raw generation caches and all
seven current full judge result sets, recomputing raw/discrete/LC metrics.
Use `audit/evaluation_readiness.json` for this independent completion check.

## Scope and security

This preserves experiment data, code, configurations, weights, and reproduction
metadata. Installed virtual environments, OS files, disposable compilation/
download caches, credentials, and SSH keys are excluded. Environment package
inventories and setup scripts are retained; this is not a relocatable VM image.
The backup contains private/gated research data and must not be published as a
public reproduction bundle. The original live artifacts are not deleted.

## Collect or retry

From `/home/jaewonc/SecPO`, while the current host is still available:

```bash
runtime/eval_envs/lm/bin/python -u scripts/archive_lambda_thinking.py
```

This collects from `ubuntu@209.20.159.162` using the already verified SSH host
identity. It performs no model inference or paid judge calls.
