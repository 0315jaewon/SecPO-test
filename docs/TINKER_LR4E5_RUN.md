# SecPO learning-rate 4e-5 run

Launched September 18, 2026 at 17:40 UTC (10:40 PDT), from the base
`Qwen/Qwen3.8-27B`, not from either previous SecPO adapter.

The configuration `configs/train_qwen38_27b_tinker_lr4e5.json` differs from
the lr6e5 configuration only in `learning_rate` and `output_dir`.
It uses the same pinned 18,951 training pairs, one epoch, batch size 64,
297 optimizer steps, seed 42, rank 64, beta 0.1, max length 6144,
linear schedule and 3% warmup. No periodic checkpoints; final training state
and inference weights are saved with `checkpoint_ttl_seconds: null`.

Exact fresh-run launch command (do not run again against the existing output):

```bash
env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  runtime/eval_envs/lm/bin/python scripts/launch_tinker_training.py \
  --config configs/train_qwen38_27b_tinker_lr4e5.json
```

Artifacts are under `outputs/qwen38_27b_tinker_lr4e5/`: full preflight,
launch provenance, worker status, log, per-step metrics and, upon completion,
`result.json` with final checkpoint paths. The durable supervisor validates
all 297 steps and both final paths before marking completion. It does not
automatically restart a failed run.

The preflight passed and the configuration/runner/runtime/launcher tests passed
(37 tests). The previous lr6e5 run took roughly 3 hours 11 minutes; runtime
for this run depends on Tinker capacity.

## Earlier models' retention

Both lr8e5 and lr6e5 final training-state and sampler checkpoints were verified
private and without expiration on September 18. Independent copies of both
inference adapters, with SHA-256 manifests, are in
`outputs/checkpoint_backups/lr8e5/` and `outputs/checkpoint_backups/lr6e5/`.
`scripts/preserve_tinker_checkpoints.py` checks retention and validates backups.

Tinker's archive endpoint rejects training-state exports: only sampler-weight
archives can be downloaded. Therefore optimizer/training state remains on
Tinker with no expiry; local backups contain inference adapters, not optimizer
state or base-model weights. No-expiry prevents automatic TTL deletion, not
manual deletion or account/provider loss. Local copies need normal disk backups.
