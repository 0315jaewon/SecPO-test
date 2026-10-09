# GPT-6 Luna judging of saved Qwen3.8 answers

Experiment started September 23, 2026. Scope is **AlpacaEval2 only**, with
805 examples for each of five answer sets. No SEP rejudging is launched here.

| Directory label | Saved candidate model | Original generation cap |
| --- | --- | --- |
| `base` | Qwen3.8-27B, thinking off | 16,384 |
| `secpo` | SecPO lr=8e-5, thinking off | 16,384 |
| `secpo_lr6e5` | SecPO lr=6e-5, thinking off | 16,384 |
| `secpo_lr4e5` | SecPO lr=4e-5, thinking off | 16,384 |
| `base_thinking` | Base Qwen3.8-27B, thinking on | 32,768, including reasoning |

Only existing final answers are judged. No model-generation requests are made.
The thinking-on job waits for all 805 saved final answers; it need not wait for
the other judges to finish. Empty final answers remain in the evaluation.

## Protocol

- Direct OpenAI Chat Completions, exact model ID `gpt-6-luna`.
- `reasoning_effort=none`, `temperature=1`, `logprobs=true`,
  `top_logprobs=5`, `max_completion_tokens=16`.
- Exactly one completed `m` or `M` verdict token is required. The 16-token API
  budget accommodates service overhead; it is not the candidate answer cap.
- Same original AlpacaEval 0.6.6 judge prompt, seeded randomized output ordering,
  probability-weighted `logprob_parser` (no explicit binarization), raw win rate,
  and length-controlled win rate as the preceding GPT-5.6 Luna comparison.
- The reference answers remain **GPT-4o-2024-05-13**, in
  `data/eval_references/alpaca_eval2_gpt-4o-2024-05-13_official_reference_outputs.json`.
  Changing the judge does not change the reference model.
- The smoke test returned the exact requested model with valid verdicts and
  logprobs. Both responses exposed only one alternative at logprob 0 despite
  requesting five. The unchanged parser therefore gave hard preferences for
  those two examples. We do not fabricate missing probabilities or interpret
  those two judgments as a benchmark score.

Official model and migration guidance checked before implementation:
[GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna),
[API parameter guidance](https://developers.openai.com/api/docs/guides/latest-model).
The OpenAI Docs skill informed retaining `reasoning_effort=none` for compatibility
with the existing logprob-scoring protocol. Candidate thinking mode is separate
from judge reasoning effort.

## Files and execution

- `src/secpo/tinker/gpt6_luna_judge.py`: isolated judge request/cache identity;
  reuses existing credential handling, direct-OpenAI transport, and parser.
- `scripts/judge_gpt6_luna.py`: verifies source answers, runs smoke/full scoring,
  and durably queues all five sets.
- `tests/test_gpt6_luna_judge.py`: model/parameter guards, cache isolation and
  corruption checks, fractional parser behavior, complete-set and row alignment.

From `/home/jaewonc/SecPO`, using the existing `secpo_openai.env` key file:

```bash
# Compatibility test on the first two saved base examples (cached for the full run)
runtime/eval_envs/lm/bin/python -u scripts/judge_gpt6_luna.py --variant base --smoke 2

# Launch/resume all five jobs; at most two jobs, four API requests per job
runtime/eval_envs/lm/bin/python -u scripts/judge_gpt6_luna.py --queue --detach

# Run/resume one full job
runtime/eval_envs/lm/bin/python -u scripts/judge_gpt6_luna.py --variant secpo_lr4e5

# Offline tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 runtime/eval_envs/lm/bin/python -m pytest -q tests/test_gpt6_luna_judge.py tests/test_luna_judge.py
```

The runner is configured for this machine's existing saved artifacts. For moving
to another machine, the input directories, reference/LC files, Python environment,
and local credentials must also be supplied; the script alone is not a portable
data bundle.

## Artifacts and preservation

New results are isolated under:

`/data2/jaewonc/secpo_evals/gpt6_luna_alpaca_20260923`

Root files: `launch.json`, `supervisor.json`, `supervisor.log`,
`source_identity.json`, and a credential-free `code_snapshot/`.

Each `<variant>/alpacaeval2/` contains:

- `manifest.json`: source paths, answer/generation hashes, original manifest hash,
  reference identity, judge settings, evaluator/implementation hashes.
- `smoke/results.json`: compatibility only, not full benchmark performance.
- `progress.json`, `full.log`, `full_worker.json`: ongoing full-job status.
- `judge_cache/*.json`: individual returned verdicts, logprobs, model identity,
  usage, and response ID; reused on resume.
- `annotations.json`, `results.json`: per-example judgments and aggregate metrics
  after successful full scoring.

Prior generations and prior judge results are read-only inputs. The four old
answer hashes are checked against completed GPT-5.6 Luna results. The thinking
answers are checked against saved final-answer fields and frozen as hashes in the
new manifest. All 805 rows must align with the same reference set before judging.
Per-job and supervisor file locks prevent duplicate launches. Source identity
checks refuse silent changes during a run. Invalid API verdicts are saved in a
separate `judge_cache/invalid_responses/` directory, never as valid cached scores.

API usage counts are recorded, but the API does not report a final dollar charge.
AlpacaEval package price placeholders should not be interpreted as the run cost.
A newer judge can change preferences; this experiment alone cannot establish
which judge best matches human assessment or repair candidate-generation errors.
