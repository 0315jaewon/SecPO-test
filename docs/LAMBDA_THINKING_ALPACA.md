# Qwen3.8-27B thinking-on AlpacaEval2, September 23, 2026

Scope: **base Qwen3.8-27B only**, all 805 AlpacaEval2 instructions, generated
once and judged by both GPT-4o-mini and GPT-5.6 Luna. No Qwen3.6 run and no
SecPO-adapter run are included in this experiment.

## Protocol

- Exact base revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Lambda: `ubuntu@209.20.159.162`, one NVIDIA H100 PCIe, 81,559 MiB reported
  GPU memory; x86_64. There is no persistent Lambda filesystem mounted here.
- Verified BF16 weights come from the REDS September 19 backup. All 18 shards
  and accompanying files must match `SHA256SUMS.models` before serving.
- vLLM 0.29.0, Transformers 5.10.4, Torch 2.13.0+cu130. The full installed
  dependency inventory is saved as `packages.json` under the run directory.
- Official tokenizer template: `enable_thinking=True`,
  `preserve_thinking=True`, `reasoning_effort="xhigh"`. This template inserts
  Qwen's official xhigh system instruction; no custom evaluation system prompt
  or repeat-user-prompt defense is added.
- Target-model sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0,
  presence_penalty 0, repetition_penalty 1, frequency_penalty 0, seed 42.
  These are the [model card's thinking-mode sampling settings](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices).
- User-selected output cap: **32,768 tokens total, including reasoning and
  final answer**. This is not a claim of an official Alpaca-specific cap.
  Server context limit: 65,536. Longest rendered prompt: 546 tokens.
- Each judge sees only the final answer, extracted after the first actual
  closing-think special token. Raw reasoning, all output token IDs, final
  answers, finish reasons and token counts are retained separately.
- If generation exhausts its budget before producing a final answer, that
  example is recorded as an empty final answer, not dropped or replaced by
  reasoning text. Such cases and all length-limited outputs are counted.
- Fixed reference: saved **GPT-4o-2024-05-13** outputs, SHA-256
  `653c131d58e8d5802203c85e80ca3fb4b378f22e2a0280ad5cd910ff0bee9535`.
- Original AlpacaEval 0.6.6 comparison prompt, seeded answer-order
  randomization, weighted log-probability parser and LC fit are reused.
- Mini: `openai/gpt-4o-mini` through OpenRouter, temperature 1, max_tokens 1,
  logprobs true, top_logprobs 5, provider.require_parameters true.
- Luna: `gpt-5.6-luna` through direct OpenAI, temperature 1,
  reasoning_effort none, max_completion_tokens 16, logprobs true,
  top_logprobs 5. Exactly one completed verdict token is required.

The judge's reasoning setting is separate from the candidate model's thinking
setting. Both raw weighted and length-controlled win rates are reported for
both judges. The older non-thinking responses used different recommended
sampling and a 16,384-token output cap, so this is a comparison of the two
mode-specific recipes, not an experiment changing only one Boolean flag.

## Implementation and launch

- `scripts/lambda_thinking_remote.py`: isolated environment installation,
  remote model-file verification, and private loopback-only inference server.
- `scripts/launch_lambda_thinking.py`: durable preparation/transfer,
  verification, private SSH tunnel, worker launch and status tracking.
- `src/secpo/inference/thinking_eval.py`: official thinking prompt rendering,
  explicit sampling, raw-token generation cache and answer extraction.
- `scripts/evaluate_lambda_thinking.py`: local preflight, two-example smoke
  with both judges, 805-example generation, paired judgments and scoring.
- `tests/test_lambda_thinking.py`: parsing, token-boundary, context-budget,
  endpoint validation, explicit sampling and cache tests.

Preparation was launched separately while the new runner was being tested:

```bash
runtime/eval_envs/lm/bin/python -u scripts/launch_lambda_thinking.py \
  --prepare-only --detach
```

The approved evaluation is queued behind that preparation lock:

```bash
runtime/eval_envs/lm/bin/python -u scripts/launch_lambda_thinking.py \
  --from-prepared --wait-for-preparation --detach
```

It starts only if preparation finishes successfully. The first two generated
answers and their judge caches are reused in the full run. The full generation
stage begins only after these smoke tests pass. A failure stops progression;
completion is never inferred from a process merely having exited.

On a fresh compatible instance with paths/configuration adjusted, the combined
launch would be `scripts/launch_lambda_thinking.py --detach` in the pinned
environment. Do not run a second supervisor against the active experiment.
These scripts are workspace-specific, not a turnkey public installer.

## Artifacts and status

All generation and judgment outputs are written directly to REDS:

```text
/data2/jaewonc/secpo_evals/qwen38_thinking_alpaca_20260923/
```

- `supervisor.json`: authoritative stage/status once preparation or evaluation
  owns the run lock.
- `supervisor.log`, `continuation.log`, `installation.log`, `transfer.log`,
  `evaluation.log`: preparation and evaluation diagnostics.
- `continuation_launch.json`: queued worker PID and exact command. Its initial
  `queued` status is historical; use `supervisor.json` for current progress.
- `model_validation.json`, `packages.json`: transferred-weight verification
  and inference environment inventory.
- `base/alpacaeval2/manifest.json`, `code_snapshot/`: frozen full protocol and
  implementation snapshot.
- `base/alpacaeval2/progress.json`: generation stage, completed examples,
  total generated tokens, length-limit count and empty-answer count.
- `base/alpacaeval2/completion_cache/`: resumable raw-token generation records.
- `base/alpacaeval2/candidate_outputs.json`, `generations.json`: aligned full
  outputs, written after generation completes.
- `base/alpacaeval2/{mini,luna}/`: per-judge manifests, progress, annotations
  and results. The top-level `results.json` combines both judges' metrics.

No API credentials or SSH private keys are copied to Lambda. Judge calls and
scoring run on REDS. The inference server is reachable only through an SSH
tunnel. The Lambda root `/home/ubuntu/secpo_thinking_20260923` holds weights,
its environment and server logs, but should be treated as ephemeral.

The supervisor does not shut down the paid instance. It also leaves the
inference server running after evaluation; stop or terminate the instance
once results are complete and any desired server logs are collected.

## September 23 recovery

The initial worker stopped with a `ValueError` during generation. Its progress
file showed 510 accepted results; eight already-running requests subsequently
finished, leaving **518 valid cached generations**. All matched the expected
request identities and parsed identically on offline replay. The only missing
example among indices 0–518 was zero-based index 466 (question 467), requesting
a Shakespeare Programming Language multiplication program.

The original failing response was not saved before validation, and the outer
error handler recorded only the exception type. Therefore the exact failed
check could not be established retrospectively. A targeted retry of index 466
with unchanged sampling/cap/seed returned 17,461 tokens, a proper closing-think
boundary, matching token accounting, and passed the **original** validator.
This retry's response was reused by the resumed worker, not generated again.
The failure was not reproduced by this diagnostic retry.

Recovery changes are isolated from the frozen protocol:

- Save raw successful HTTP responses before content validation so failures
  retain evidence and can be reparsed without more inference.
- Serialize same-prompt requests with a per-request file lock, including the
  diagnostic retry, to prevent duplicate inference.
- Handle early EOS without the closing-think boundary as an empty final
  answer, just as unfinished reasoning at the length cap is handled. Flag this
  anomaly explicitly; never expose unfinished reasoning to the judge. This
  hardens a brittle case but is **not a confirmed cause of the original error**.
- Record error message, stack locations, and affected question indices instead
  of suppressing every diagnostic except exception type.
- Resume against the existing server and private tunnel without reloading
  weights or regenerating the 518 cached answers.

The original `manifest.json` and `code_snapshot/` remain unchanged.
`base/alpacaeval2/resume_history/<implementation-digest>/` contains the new
runtime implementation manifest/snapshot, original failure/progress records,
and checksums of all 518 preexisting cache files. All non-implementation
protocol fields are required to match exactly before a compatible resume.
The completed judge smoke results retain their original identities and are
reused. The legacy top-level `failure.json` is historical; use the latest
`supervisor.json` and `progress.json` for current status.

The exact resume command, after reopening the private tunnel, was:

```bash
runtime/eval_envs/lm/bin/python -u scripts/launch_lambda_thinking.py \
  --from-prepared --reuse-server --reuse-tunnel --resume-compatible --detach
```

Recovery tests: **33 passed**, including raw-response reuse without inference,
early EOS handling, and preservation of evidence for token-accounting errors.
Diagnostic details are in `base/alpacaeval2/failure_probe.json` and the raw
response under `completion_cache/raw_responses/`.
