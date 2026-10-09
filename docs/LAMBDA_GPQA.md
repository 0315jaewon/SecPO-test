# Qwen3.8-27B base GPQA on Lambda

This is a cross-backend comparison with the completed Tinker base-model GPQA
run at a 16,384-output-token cap (159/198 correct, 80.3030%). It is not a new
training run and does not load a SecPO adapter.

## Protocol preservation

`scripts/lambda_gpqa.py` has three phases:

1. `prepare`, in the original local evaluation environment, validates the
   completed Tinker manifest, all 198 selected question identities, each saved
   rendered request, and its corresponding content-addressed Tinker cache entry.
   It tokenizes those exact prompts using the original pinned tokenizer. The
   exported bundle contains prompts and token IDs, not gold answers or previous
   model responses.
2. `generate`, on the Lambda GPU, checks that the inference environment's
   tokenizer produces exactly the same IDs for all 198 prompts. It feeds those
   IDs directly into vLLM: there is no second chat-template pass or reasoning
   parser. Two full-budget smoke requests precede the remaining requests. Valid
   smoke outputs are retained as part of the 198-question evaluation.
3. `score`, in the original local environment, decodes the saved output token
   IDs with the original tokenizer, applies the same stop-string trimming, and
   uses the installed lm-eval 0.4.11 extraction filters and exact-match metric.
   Before scoring new outputs, it requires an exact replay of the original
   per-question extracted answers and scores for BOTH scoring filters.

Fixed settings: `gpqa_diamond_cot_zeroshot`, 198 questions, zero-shot CoT,
thinking off, temperature 0.0, top-p 1.0, top-k -1, seed 42, output cap 16,384.
The primary score is `exact_match,flexible-extract`, not strict-match.
Stop strings are `</s>`, `<|im_end|>`, and `<|endoftext|>`.
The longest saved prompt is 2,774 tokens; the local engine's 32,768 total-context
limit therefore does not truncate any prompt or reduce the requested output cap.

The Lambda model and tokenizer use Hugging Face revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` of `Qwen/Qwen3.8-27B`.
That is the original Tinker run's pinned TOKENIZER revision. The original Tinker
manifest identifies base weights by model ID, not an independently verifiable
weight revision/dtype. Consequently this preserves the observed prompt/sampling/
scoring protocol, but does not establish bit-identical model weights or numerical
execution across Tinker and vLLM.

## Lambda environment

- Instance: `ubuntu@68.209.75.163`, one NVIDIA H100 80GB HBM3, x86_64.
- NVIDIA driver 580.105.08, Ubuntu 22.04.5 LTS.
- Dedicated directory: `/home/ubuntu/secpo_gpqa_lambda`.
- Isolated `.venv`: Python 3.10.12, vLLM 0.29.0, Transformers 5.10.4,
  PyTorch 2.13.0+cu130. The original system Python packages were not replaced.
- Inference: BF16 weights, no quantization or speculative decoding, tensor
  parallelism 1, text-only loading, batch size 4, eager execution, prefix caching
  disabled, GPU memory utilization 0.90, max batched prefill tokens 4,096.
- Full installed-package inventory is saved to the worker's `environment.json`.
- The model weights were downloaded publicly to this directory's `hf_cache`.
  No HF, Tinker, or judge API credential was transferred to Lambda.
- The dedicated SSH private key remains locally in ignored `runtime/`, mode 0600;
  the instance's ED25519 host key was verified against the user's fingerprint.

## Local commands

From `/home/jaewonc/SecPO`, prepare the frozen inputs:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  runtime/eval_envs/lm/bin/python scripts/lambda_gpqa.py prepare \
  --reference outputs/qwen38_27b_base_tinker_utility_max16384/gpqa_diamond \
  --tokenizer /data2/jaewonc/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --output outputs/qwen38_27b_lambda_gpqa_max16384/inputs.json
```

The resulting input file SHA-256 is
`b3b420aeb3f66705543e760968a64dbfe42389aadac734983f0ec2ceb85d0b6f`.
It was copied together with `scripts/lambda_gpqa.py` to the dedicated Lambda
directory. Copies were hash-verified before the first launch.

The scoring replay self-test uses the scoring command below with `--self-test`
in place of `--generations` and `--output`. It passed with 159/198 correct and
exact per-question extraction/score agreement. All 20 runner/collector tests passed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 HF_HUB_OFFLINE=1 \
  runtime/eval_envs/lm/bin/python -m pytest tests/test_lambda_gpqa.py tests/test_lambda_gpqa_collector.py -q
```

## Remote generation command

The worker runs in tmux session `secpo-gpqa-16k`, allowing it to survive SSH
disconnects. Inside `/home/ubuntu/secpo_gpqa_lambda`, its command is:

```bash
env PATH=/home/ubuntu/secpo_gpqa_lambda/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  HF_HOME=/home/ubuntu/secpo_gpqa_lambda/hf_cache \
  HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 \
  DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn \
  .venv/bin/python -u lambda_gpqa.py generate \
  --inputs inputs.json --output run_bf16_20260914 --batch-size 4
```

The actual launch uses absolute paths and appends stdout/stderr to
`run_bf16_20260914.log`. `run_bf16_20260914/progress.json` reports progress;
the manifest pins the source hash, package versions, input hash, and runtime
settings. A file lock prevents concurrent duplicate generation workers.
Each completed batch is cached. SIGTERM/SIGINT records an intentional pause.
There is no automatic failure retry or instance termination.

Initial startup at 2026-09-15 00:02:12 UTC failed before generation because the
virtualenv's `ninja` executable was not on PATH. The failure record is preserved
as `startup_failure_1.json` and the log retained. The retry added the virtualenv
bin directory to PATH without changing prompts, weights, or sampling settings.

The successful retry started at 2026-09-15 00:18:18 UTC (September 14, 17:18 PDT),
worker PID 3770. It loaded the model using 50.22 GiB for weights. Both smoke
questions completed at 00:22 UTC, producing 1,074 and 2,688 tokens without hitting
the cap, and generation continued automatically. These are launch observations,
not a current-status claim.

## Automatic collection and scoring

`scripts/collect_lambda_gpqa.py` was launched in a detached local process,
PID 3525819, using:

```bash
runtime/eval_envs/lm/bin/python scripts/collect_lambda_gpqa.py
```

It polls this exact instance/run every 45 seconds through the pinned SSH host
key and dedicated identity. On successful generation of all 198 answers it
downloads `generations.json`, the manifest, environment, progress, smoke record,
and run log into the local output directory's `remote/` subdirectory, then runs
the scoring command below. `generations.json` contains every per-question cache
record; individual remote cache files remain on Lambda.

Local `collector.json` records the latest remote progress and the collector's
phase. `collector.log` contains collection/scoring output. A local file lock
prevents duplicate collectors. It stops on remote failure/pause or ten consecutive
failed SSH responses; it never relaunches generation or terminates the instance.
Its process can be relaunched explicitly if collection alone fails.

## Retrieve and score

After `progress.json` says `generation_complete`, copy the remote
`run_bf16_20260914/` directory into the local output root as `remote/`, preserving
all manifests, caches, progress, and environment metadata. Then run locally:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  runtime/eval_envs/lm/bin/python scripts/lambda_gpqa.py score \
  --reference outputs/qwen38_27b_base_tinker_utility_max16384/gpqa_diamond \
  --inputs outputs/qwen38_27b_lambda_gpqa_max16384/inputs.json \
  --tokenizer /data2/jaewonc/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --generations outputs/qwen38_27b_lambda_gpqa_max16384/remote \
  --output outputs/qwen38_27b_lambda_gpqa_max16384
```

The final `results.json` includes scores, cap-hit count, all 198 new answers, and
the corresponding original Tinker answers/scores for question-level comparison.
`generation_complete` alone does not mean scoring has completed. The remote
worker exits after generation and releases GPU memory, but the Lambda instance
remains billable until its owner terminates it. Preserve artifacts elsewhere
before terminating the instance.
