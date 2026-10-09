# Qwen3.8-27B SecPO — learning rate 6e-5

The user requested an independent full SecPO training run at peak learning rate
`6e-5`, for later evaluation against the existing `8e-5` model. This run starts
from `Qwen/Qwen3.8-27B` using a fresh rank-64 LoRA policy and its own frozen initial
reference. It does not load or overwrite the previous trained adapter.

## Training setup

- Config: `configs/train_qwen38_27b_tinker_lr6e5.json`.
- Output: `outputs/qwen38_27b_tinker_lr6e5/`.
- Dataset: `Sizhe-Chen/Qwen3.8-27B-Instruct-SecPO-trainset`, train split,
  revision `7fd85bf8b670370ad396a049ad11f0dad36364af`.
- 18,951 preference pairs, one epoch, 297 optimizer updates, logical batch 64;
  the last batch contains 7 pairs.
- Peak learning rate `0.00006`, 3% warmup (9 steps), linear decay.
  The first warmup update has learning rate zero, as in the original run.
- Beta 0.1, rank 64, attention and MLP adapters enabled, unembedding disabled.
- AdamW betas 0.9/0.999, epsilon 1e-8, zero weight decay, gradient clipping 1.0.
- Seed 42, thinking-off pre-rendered dataset, prompt cap 512, sequence cap 6,144.
- Tokenizer revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Final-only saving (`save_every: 0`) and no expiration
  (`checkpoint_ttl_seconds: null`). Both final training state and sampler are
  exported. There are no saved intermediate recovery points.

Comparison with the original completed run's archived config found exactly four
changed fields: learning rate, output directory, save frequency, and checkpoint
TTL. Compared with the current final-only/no-expiration default config, only
learning rate and output directory differ. The core training implementation was
not changed for this launch.

## Validation and launch

The complete pinned dataset passed preflight: maximum attacked prompt 510,
clean prompt 441, and full sequence 6,001 tokens. All 36 focused config, runner,
runtime, and launch-guard tests passed. No extra paid smoke model was created.

Launch time: September 9, 2026, 21:19:29 UTC (14:19 America/Los_Angeles).
Supervisor PID 238263; worker PID 238338. These identify the launch, not current
status. Exact launch command from `/home/jaewonc/SecPO`:

```bash
env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  runtime/eval_envs/lm/bin/python scripts/launch_tinker_training.py \
  --config configs/train_qwen38_27b_tinker_lr6e5.json
```

The offline flags apply only to the cached Hugging Face dataset/tokenizer;
training and reference scoring still use paid Tinker API calls. Credentials are
loaded literally from the existing ignored `secpo_eval_keys.env`, never embedded
in the command or saved metadata.

The launcher records actual installed versions in `launch.json`: Tinker 0.27.0,
Transformers 4.57.1, Datasets 3.5.0, PyTorch 2.8.0. These are the existing runtime
environment versions, not a claim that every dependency matches `pyproject.toml`.
Configuration and source hashes are saved alongside the command for provenance.

## Monitoring and later evaluations

- `worker.json`: process IDs, command, start/end times, exit code, final status.
- `run.log`: preflight, training output, and sanitized error type on failure.
- `metrics.jsonl`: one row per completed optimizer update, including loss,
  preference accuracy, margin, and actual scheduled learning rate.
- `result.json`: final training-state and sampler paths after all 297 updates.
- `preflight.json` and `launch.json`: validation and launch provenance.

The background supervisor survives the initiating terminal and prevents duplicate
launches with a lock. It does not automatically restart failed training. Completion
requires exit code 0 and a validated full 297-step result with both final paths.

For later evaluation, use this run's `result.json` -> `final_sampler_path`.
Existing evaluation helpers currently resolve the older `8e-5` run by default;
they must be explicitly pointed at the new checkpoint when that comparison is
requested. Do not replace `outputs/qwen38_27b_tinker/result.json` or overwrite
the earlier evaluation outputs. No evaluations were launched with this training
request.

## Evaluation launch — September 10, 2026

After training completed all 297 updates, the user requested the same 16,384-token
evaluations on the new adapter. Five full evaluations were launched using:

`tinker://1b50b5c0-83c3-5a3c-a3e2-faeeded4aa00:train:0/sampler_weights/secpo-final-sampler`

| Benchmark | Examples | Worker PID | Supervisor PID | Start (UTC) |
| --- | ---: | ---: | ---: | --- |
| GPQA Diamond | 198 | 2092543 | 2092467 | 17:33:08 |
| MMLU-Pro | 1,400 | 2075289 | 2075159 | 17:21:06 |
| Minerva MATH | 700 | 2075288 | 2075160 | 17:21:06 |
| SEP utility | 9,160 | 2075715 | 2075520 | 17:21:10 |
| AlpacaEval2 | 805 | 2075910 | 2075759 | 17:21:12 |

PIDs and times identify this launch, not ongoing status. Results and live progress
are isolated under `outputs/qwen38_27b_tinker_lr6e5_utility_max16384/`.
AgentDojo and AgentDyn were not included: their earlier runs used an 8,192-token
per-response cap, not the general/reference utility 16k experiment.

Each run preserves the earlier 16k protocol: thinking off, greedy target
generation at temperature 0.0, the same dataset revisions and mentor indices,
and the same scoring configuration. Reference utilities retain the single
`openai/gpt-4o-mini` judge via OpenRouter (judge temperature 1.0 and one output
token, independently of the target's generation settings). SEP uses the approved
Llama-3-8B-Instruct reference below; Alpaca uses the existing GPT-4o-2024-05-13
reference, so this is not the standard AlpacaEval2 leaderboard configuration.

The explicit `secpo_lr6e5` variant routes to the new checkpoint and output folder.
The older default `secpo` variant remains unchanged. Each new run saves
`checkpoint_comparison.json`, checking its protocol against the earlier SecPO
16k manifest and recording source hashes. Reference-entrypoint changes only
route the new variant; judge parsing, retries, and scoring were not changed.
All 33 focused evaluation tests, all five full-dataset preflights, and all five
two-example smoke runs passed before launching the full workers. Full runs reuse
their own smoke responses, not responses from the previous adapter.

Exact launch commands from `/home/jaewonc/SecPO`:

```bash
env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  runtime/eval_envs/lm/bin/python scripts/launch_tinker_evals.py \
  --model-variant secpo_lr6e5 --max-output-tokens 16384 \
  --benchmarks mmlu_pro minerva_math

env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  runtime/eval_envs/lm/bin/python scripts/launch_tinker_evals.py \
  --model-variant secpo_lr6e5 --max-output-tokens 16384 \
  --benchmarks sep_utility \
  --reference /home/jaewonc/SecAlign_cont/data/SEP_dataset_test_Meta-Llama-3-8B-Instruct.json \
  --reference-sha256 63e965d2a25d2d561a7b6079048977314cfaacf5d286cf86ae168538c771dd34 \
  --sep-test-dataset /home/jaewonc/SecAlign_cont/data/SEP_dataset_test.json

env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  runtime/eval_envs/lm/bin/python scripts/launch_tinker_evals.py \
  --model-variant secpo_lr6e5 --max-output-tokens 16384 \
  --benchmarks alpacaeval2 \
  --reference data/eval_references/alpaca_eval2_gpt-4o-2024-05-13_official_reference_outputs.json

env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  runtime/eval_envs/lm/bin/python scripts/launch_tinker_evals.py \
  --model-variant secpo_lr6e5 --max-output-tokens 16384 \
  --benchmarks gpqa_diamond
```

Each benchmark folder contains `worker.json` (exact evaluator argv and process
status), `progress.json` (generation/judging progress), `run.log`, `manifest.json`,
and `checkpoint_comparison.json`. The offline flags only constrain Hugging Face;
Tinker generation and OpenRouter judging still use paid network APIs. These are
durable background workers, not an automatic failure-retry service.

## User-requested evaluation pause — September 10, 2026

At 18:02:28 UTC (11:02:28 PDT), the two unfinished evaluation workers were stopped
with SIGTERM after verifying each PID's complete command line against its saved
worker record. Both supervisors recorded exit code -15 and released their launch
locks. Worker/progress statuses were subsequently annotated `paused`, retaining
the original supervisor-reported `failed` status and exit code for provenance.
No training checkpoint, completion cache, judge cache, or completed result was
deleted. No automatic resumption is scheduled.

| Benchmark | Status at pause | Saved generation progress |
| --- | --- | ---: |
| GPQA Diamond | Already complete, untouched | 198 / 198 |
| MMLU-Pro | Paused | 1,261 / 1,400 |
| Minerva MATH | Already complete, untouched | 700 / 700 |
| SEP utility | Paused before judging | 2,065 / 9,160 |
| AlpacaEval2 | Already complete, including judging; untouched | 805 / 805 |

MMLU-Pro has 1,261 valid cached responses; SEP has 2,050 valid cached responses
covering 2,065 dataset rows because duplicate prompts reuse generations. All
saved completion JSON files in these two caches parsed successfully and matched
their request-hash filenames. Resumption uses the same launch commands above,
only when requested; completed workers are skipped and cached responses reused.
Requests already submitted to Tinker at termination may still finish remotely
and incur charges; locally uncached in-flight responses may need regeneration.
