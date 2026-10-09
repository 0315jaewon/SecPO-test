# lr=8e-5: full utility and six SEP static attacks

**SEP superseded:** the user cancelled the single-Luna SEP variant and requested
deletion of its generated outputs. MMLU-Pro and Minerva Math continue unchanged.
The fresh SEP run and original three-judge method are documented in
[LAMBDA_SEP_ORIGINAL_20260927.md](LAMBDA_SEP_ORIGINAL_20260927.md). Do not relaunch
the old SEP generation or Luna judging commands below; they describe the retired
configuration for provenance only.

Instance: `ubuntu@209.20.159.207`, one H100 PCIe 80 GB.

Output root on REDS:
`/data2/jaewonc/secpo_evals/qwen38_full_sep_20260927/`

Victim: Qwen3.8-27B SecPO lr=8e-5, exact final sampler
`tinker://fcfc57ba-d2ab-568d-8f90-b3f60ab70560:train:0/sampler_weights/secpo-final-sampler`.
Standalone BF16 model shards are SHA-256 checked against the verified REDS backup.

## Scope and settings

- MMLU-Pro: all 12,032 test examples across 14 categories, 5-shot.
- Minerva Math: all 5,000 test examples across 7 categories, 4-shot.
- SEP: all 9,160 approved test examples for each of six static attacks:
  `straightforward`, `straightforward_before`, `ignore`, `ignore_before`,
  `completion`, and `completion_ignore` (54,960 generations).
- Thinking off, preserve thinking off, 16,384 output tokens, 65,536 context tokens.
- Sampling: temperature 0.7, top_p 0.8, top_k 20, min_p 0,
  presence_penalty 1.5, repetition_penalty 1, frequency_penalty 0.
- Generation seed 42. lm-eval random, NumPy, Torch, and few-shot seeds 42.
- SEP attack seed 20260630, original user/input trust-role separation, original
  attack construction, witness gate, semantic rubric, and ASR summarization.
- General utility retains lm-eval 0.4.11 tasks and scoring. The new full-split
  wrapper verifies the old subset's document hashes, dataset fingerprints,
  category counts, few-shot counts, and sampling settings before expanding.
- Original SecPO evaluator modules and older run artifacts are not modified.

The original SEP builder accepts only Llama3.1/Qwen3.6 family labels. Its
Qwen3.6 selector is used solely to call its unchanged six family-independent
static transforms. No Qwen3.6-specific fake-delimiter attack is selected; the
actual victim model and tokenizer are explicitly Qwen3.8 throughout inference.

## Judge policy

Requested judge: `openai/gpt-6-luna` via OpenRouter, not direct OpenAI.
Only witness-positive outputs are sent to the original YES/NO semantic rubric.
Two synthetic probes returned the expected YES and NO verdicts.

Single-model trial count requires the user's choice: one verdict or three
independent verdicts with unanimous YES. The launch queue deliberately starts
generation/utility only; it does not choose or launch this judging policy.
The separate `judge_manifest.json` freezes the selected choice when judging starts.
This is not the paper's three-distinct-judges/nine-vote protocol.

The OpenAI Docs skill informed using explicit `reasoning.effort=none` for the
binary judge. OpenRouter model availability and the supported request were
checked live; requests use `reasoning: {"effort":"none"}`, `max_tokens:16`,
`provider: {"require_parameters":true}`, and omit temperature/logprobs.
Model identity and complete YES/NO output are checked. Invalid/error judgments
remain unknown and resumable, never counted as NO.

Credentials are read locally from `secpo_openrouter.env`; no keys are sent to
Lambda or saved in manifests. Raw judge responses and reported usage/cost are cached.

## Launch and artifacts

```bash
runtime/eval_envs/lm/bin/python -u scripts/launch_lambda_full_sep.py --detach
```

The queue waits for isolated environment preparation and checkpoint verification,
starts a private vLLM server and SSH tunnel, smoke-tests one question per utility
category and one example per SEP attack, then launches all three full workers.
MMLU-Pro/Minerva each use concurrency 12; SEP uses 8. They share one model server.

Per-benchmark `manifest.json`, `progress.json`, worker launch records, logs, and
`completion_cache/` live under `secpo/{mmlu_pro,minerva_math,sep_static}/`.
Utility `results.json` contains per-question samples, also copied into `samples/`.
SEP per-example generation rows live in `rows/<attack>/`; witness counts are
not final ASR. Later semantic results are saved separately in `judgments/`,
`judge_cache/`, `judge_progress.json`, and final `results.json`.

After generations finish and the trial count is selected:

```bash
runtime/eval_envs/lm/bin/python -u scripts/evaluate_lambda_sep_static.py \
  --phase judge --judge-trials 3 --concurrency 8
```

Replace `3` with `1` only if that is the selected policy. A changed judge policy
requires a separate output/cache namespace; incompatible cached votes are refused.

All bulk artifacts stay on `/data2/jaewonc`. The queue does not terminate the
Lambda instance. Prior model backups and evaluation results remain intact.
