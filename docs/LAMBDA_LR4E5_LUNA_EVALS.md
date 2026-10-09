# lr4e-5 Lambda evaluations and direct-OpenAI Luna judging

The completed training checkpoint is
`tinker://1dc6a174-d120-5ddc-884e-257ff14c7210:train:0/sampler_weights/secpo-final-sampler`
(297 steps). Both final training/sampler checkpoints were verified private with
`expires_at=null` on September 18, 2026 (Pacific time).

## Generation protocol

The new isolated implementation in `runtime/lambda_suite_20260916/lr4e5_code`
copies the frozen lr6e-5 implementation. Only model routing/labels change;
the provenance inventory records every source hash. Prior experiment sources,
responses and scores are not edited.

| Benchmark | Examples | Output cap |
|---|---:|---:|
| GPQA Diamond | 198 | 16,384 |
| MMLU-Pro mentor subset | 1,400 | 16,384 |
| Minerva MATH mentor subset | 700 | 16,384 |
| AgentDojo clean utility | 97 | 8,192 per call |
| AgentDyn clean utility | 60 | 8,192 per call |
| AlpacaEval2 utility | 805 | 16,384 |
| SEP utility, full set | 9,160 rows / 9,135 distinct generation prompts | 16,384 |

Non-thinking decoding: temperature 0.7, top_p 0.8, top_k 20, min_p 0,
presence_penalty 1.5, repetition_penalty 1, frequency_penalty 0. Seeds remain
0 for agentic tasks and 42 otherwise. Agentic tasks use `repeat_user_prompt`
defense and **no attack**. These are clean-utility, not ASR evaluations.

The CPU merge uses the same pinned official Tinker converter and validates
representative BF16 updates, tensor identities, and unchanged embeddings/head
before the sole GPU is switched from the idle lr6e-5 server.

## Judging protocol

`gpt-5.6-luna` is called directly at `https://api.openai.com/v1` using the ignored
`secpo_openai.env` file. No credential is copied to Lambda or an output archive.

The original AlpacaEval 0.6.6 comparison prompt, seed-0 answer-order randomization,
probability-weighted `logprob_parser`, and metrics are retained. Luna's requests
use temperature 1, reasoning_effort `none`, logprobs true, top_logprobs 5, and
max_completion_tokens 16. The API rejected a budget of 1. Each accepted response
must nevertheless be exactly one completed `m`/`M` verdict token. A compatibility
probe returned both alternatives and a genuinely fractional preference.

Alpaca references remain the saved GPT-4o-2024-05-13 outputs (not the standard
GPT-4-Turbo reference); SEP references remain the approved Llama-3-8B-Instruct
JSON. Alpaca reports length-controlled and raw win rates; SEP reports the
original probability-weighted win rate. These new judge results are separate
from the older GPT-4o-mini results and should be labeled accordingly.

Previously saved responses are reused for base, lr8e-5 and lr6e-5. The new lr4e-5
responses are judged once generation finishes. Its generation-only manifest
retains the legacy judge fields solely for frozen protocol comparisons: that
judge is never instantiated. `judging_deferred.json` explicitly documents this;
the new Luna run's manifest is the authoritative judge configuration.

## Launch commands and monitoring

```bash
runtime/eval_envs/lm/bin/python -u scripts/export_tinker_lr4e5_for_lambda.py
runtime/eval_envs/lm/bin/python -u scripts/deploy_lambda_lr4e5.py --detach
runtime/eval_envs/lm/bin/python -u scripts/launch_lambda_luna_judging.py \
  --variants base secpo secpo_lr6e5 secpo_lr4e5 --detach
```

Deployment waits for the export manifest, uploads only the adapter and conversion
scripts, validates the merge, then invokes `launch_lambda_lr4e5_evals.py --detach`.
That launcher performs seven preflights, seven smoke tests, then the full runs.
The independent Luna queue runs at most two benchmark jobs concurrently (four
API requests per job), with per-request caches and smoke tests before full jobs.

- Export: `runtime/lambda_suite_20260916/secpo_lr4e5_export/export_status.json`
- Deployment/generation: `outputs/qwen38_lambda_recommended_20260916/secpo_lr4e5/`
- Luna scores/progress: `outputs/qwen38_lambda_luna_openai_20260918/`
- API compatibility evidence: `outputs/judge_luna_openai_preflight_20260918/`
- Offline-only preflight artifacts (not real weights or generations):
  `outputs/qwen38_lambda_lr4e5_offline_preflight_20260918/`

The corresponding results are copied without deletion to the persistent Lambda
root `/lambda/nfs/jaewon/secpo_qwen38_eval_20260916`. Every worker saves its exact
command and status. The direct API reports token usage, not billed dollar costs;
zero-valued package per-example price placeholders are **not actual cost**.
