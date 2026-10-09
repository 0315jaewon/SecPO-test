# Qwen3.8-27B-SecPO utility evaluation

Launched September 7, 2026 (America/Los_Angeles), using the final sampler in
`outputs/qwen38_27b_tinker/result.json`. No training or base-model comparison is
part of this batch. Agent tasks run in the benchmarks' simulated environments;
they do not operate on real bank, Slack, GitHub, or shopping accounts.

| Benchmark | Selected items | Scoring |
| --- | ---: | --- |
| MMLU-Pro | 1,400 | `exact_match,custom-extract`; 5-shot |
| Minerva MATH | 700 | `math_verify,none`; 4-shot |
| GPQA Diamond | 198 | `exact_match,flexible-extract`; 0-shot CoT |
| AgentDojo | 97 | Clean task utility; no injection attack |
| AgentDyn | 60 | Clean task utility; no injection attack |

MMLU-Pro and MATH use exactly the supplied JSON indices in `data/eval_indices/`.
The runner verifies category cardinalities before inference, sorts indices to
avoid an upstream `doc_id` labeling issue, and checks scored document identities
afterwards. Original index files are unchanged. Each scoring filter may produce
another sample record for the same problem; this is not an additional generation.

## Outputs and monitoring

Everything is under `outputs/qwen38_27b_tinker_utility/<benchmark>/`:

- `manifest.json`: checkpoint, tokenizer revision, selection, seeds, and settings.
- `worker.json`: worker/supervisor PIDs, launch command, timestamps, exit status.
- `run.log`: persistent execution log.
- `progress.json`: generation/task progress and newly incurred token counts.
- `results.json`: final scores, plus per-sample question outputs for LM tasks.
- `episodes/` and `tasks/`: agent traces and per-task utility/error records.
- `completion_cache/`: resumable, checkpoint-and-prompt-specific sampled responses.
- `smoke/`: small end-to-end checks, excluded from the full-set score.

Background supervisors survive the launching terminal session. Completion means
`worker.json` has `status: complete`, `exit_code: 0`, and `results.json` exists.
An error is recorded separately from an incorrect model answer. These are local
background workers, not a notification service or automatic retry scheduler.

To launch/resume after checking any error, from the repository root:

```bash
runtime/eval_envs/lm/bin/python scripts/launch_tinker_evals.py --benchmarks agentdyn
```

The launcher requires a successful smoke, skips completed runs, and locks active
runs to prevent duplicate launches. Completed generations are cached. Token
counters show **new requests in that process**, not all historical cached tokens
or a dollar bill. Tinker billing remains authoritative for cost.

## Protocol and compatibility notes

- All five evaluations use **thinking off**, matching this checkpoint's training
  format. This is not a byte-identical reproduction of the Qwen3.6 evaluation
  configs, which enable thinking for the agent benchmarks.
- LM evaluation uses `lm-eval==0.4.11`, `math-verify==0.8.0`, deterministic decoding,
  and seeds 42. MMLU-Pro's task specifies a 2,048-output-token cap. The Tinker
  adapter explicitly also uses 2,048 for GPQA and MATH, whose task YAMLs do not
  specify a cap. This adapter default is **not** inherited from the original HF
  backend. Length-limited answers are counted during generation and their stop
  reasons remain in the completion cache. Early GPQA results include substantial
  truncation; interpret the resulting score as specific to this token budget.
- AgentDojo uses pinned v1.2.1 (97 tasks); AgentDyn uses pinned v1.2.2 (60 tasks)
  with the repository's reviewed patch. Installed source is verified against the
  pinned commits, excluding historical archives, Finder metadata and bytecode.
- Agents use `repeat_user_prompt`, trusted `user` / untrusted `input` roles,
  temperature 1, top-p .95, top-k 20, seed 0, and a 15-iteration tool loop. Their
  output cap is 8,192 per generation (an explicit run setting, not the original
  AgentDyn 32,768 cap). Concurrency is 3 per agent benchmark and 12 per LM benchmark.
- The agent adapter extends the pinned Qwen3.8 native chat template to support
  SecPO's `input` role and parses Qwen3.8's native XML tool calls. It does not
  substitute the older Qwen3.6 tool-call format.
- A JSON formatter compatibility fix serializes tool-returned dates as ISO
  strings. Upstream plain `json.dumps` otherwise fails on valid AgentDyn outputs.
  Normal JSON values, simulated state and utility predicates are unchanged.
  AgentDojo's first full worker started before this fallback was added; it was
  stopped and resumed with the fix after encountering the same error. Completed
  tasks were retained; date-serialization errors are retried, not model failures.
- Agent utility is computed by each benchmark's local task checks, without an
  external model judge. Only the Tinker and Hugging Face credentials are used.

## Implementation checks

The dedicated tests cover index validation, durable serialization, stop handling,
concurrent response ordering, typed XML tool arguments, date serialization and
offline rendering of a tool-call history with the pinned tokenizer:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 runtime/eval_envs/lm/bin/python -m pytest tests/test_tinker_evaluation.py -q
```

Plugin autoload is disabled because an unrelated inherited Hydra pytest plugin
requires a conflicting ANTLR version; the evaluation's math parser uses ANTLR 4.11.

## Matched pre-fine-tuning baseline

The same five evaluations also support `--model-variant base`. This creates a
Tinker sampler directly with `base_model="Qwen/Qwen3.8-27B"`, without creating a
training client or loading the SecPO adapter. The existing default remains
`--model-variant secpo`.

The base-model batch writes only to
`outputs/qwen38_27b_base_tinker_utility/<benchmark>/`. A baseline manifest must
match the corresponding SecPO manifest in every field except `checkpoint`;
the baseline identity is explicitly `base:Qwen/Qwen3.8-27B`. Completion cache
keys include this identity, preventing reuse of fine-tuned-model responses.

This is a weights-only comparison under the same interface: same selected
problems, few-shot examples, seeds, tokenizer revision, chat templates,
thinking-off setting and generation budgets. The agent benchmarks retain the
same SecPO trust-role template and repeat-user-prompt defense for both models.
It is not a separate evaluation of the base model with its stock agent prompt.
The question output cap stays at 2,048 and the agent cap at 8,192. A higher-cap
experiment would need to be labeled and run separately for both models.

After the model-specific smoke checks, launch or resume the base batch with:

```bash
runtime/eval_envs/lm/bin/python scripts/launch_tinker_evals.py --model-variant base
```

The tests also verify adapter-free sampler creation, disjoint base/SecPO caches,
and rejection of baseline protocol drift. The base run incurs separate Tinker
sampling usage; existing SecPO jobs and artifacts are unaffected.

## SecPO 4,096-token output-budget comparison

The requested higher-budget variant reruns **GPQA Diamond (198), MMLU-Pro (1,400),
and Minerva MATH (700)** on the final SecPO adapter. It does not launch additional
agent or base-model evaluations.

`--max-output-tokens 4096` overrides every leaf task's `max_gen_toks`, including
MMLU-Pro's explicit 2,048 default. Before sampling, each manifest is checked
against the original SecPO run: checkpoint, selected documents, tokenizer,
few-shot settings, seeds, thinking-off mode, stop sequences and decoding settings
must remain identical; only the output-token budget may differ.

Artifacts are separate at
`outputs/qwen38_27b_tinker_utility_max4096/<benchmark>/`. Existing 2,048-token
results and caches are not overwritten or imported. All selected questions are
regenerated, not just previously incorrect or truncated ones. Two-item smoke
responses at the new cap are cached for inclusion in the full run. New final
`progress.json` files retain the `length_limited` count for direct budget comparisons.

Launch/resume command, after successful smoke checks in that output directory:

```bash
runtime/eval_envs/lm/bin/python scripts/launch_tinker_evals.py \
  --model-variant secpo --max-output-tokens 4096 \
  --benchmarks gpqa_diamond mmlu_pro minerva_math
```

Interpret score differences as an output-budget sensitivity experiment, not a
new training run. A larger allowance may reduce truncation but does not guarantee
higher accuracy. A base-vs-SecPO comparison should still match budgets on both sides.

## Matched base-model 4,096-token rerun

The corresponding base-model experiment reruns the same 198 GPQA, 1,400 MMLU-Pro,
and 700 Minerva MATH questions using `Qwen/Qwen3.8-27B` without the SecPO adapter,
with a 4,096-token output cap and otherwise unchanged settings. No agent tasks
are relaunched. Artifacts are separate at
`outputs/qwen38_27b_base_tinker_utility_max4096/<benchmark>/`.

After successful two-item smoke checks for each benchmark:

```bash
runtime/eval_envs/lm/bin/python scripts/launch_tinker_evals.py \
  --model-variant base --max-output-tokens 4096 \
  --benchmarks gpqa_diamond mmlu_pro minerva_math
```

Before full launch, compare each new manifest with its SecPO 4,096-token
counterpart (only `checkpoint` may differ), and verify that smoke prompt hashes
and recorded output limits agree. Keep both models' original 2,048-token results
intact for the paired budget comparison.

## Matched 8,192-token question evaluations

The next output-budget experiment reruns all 198 GPQA Diamond, 1,400 MMLU-Pro,
and 700 Minerva MATH questions for **both** the base model and final SecPO adapter
at 8,192 output tokens: 2,298 questions per model, 4,596 total. The original
sample selections, prompts, few-shot settings, seeds, decoding and thinking-off
mode remain unchanged. Agent evaluations are not relaunched.

New artifacts are separate at:

- `outputs/qwen38_27b_base_tinker_utility_max8192/<benchmark>/`
- `outputs/qwen38_27b_tinker_utility_max8192/<benchmark>/`

The runner checks that only the output budget differs from each model's original
run. After each model/benchmark's two-item smoke, verify its prompt hashes against
the prior run and verify that base/SecPO manifests differ only in checkpoint.
Existing 2,048- and 4,096-token artifacts remain intact. Completion caches include
the model identity and output budget, so shorter-budget answers are not reused.

Launch/resume commands after those smoke checks:

```bash
runtime/eval_envs/lm/bin/python scripts/launch_tinker_evals.py \
  --model-variant base --max-output-tokens 8192 \
  --benchmarks gpqa_diamond mmlu_pro minerva_math

runtime/eval_envs/lm/bin/python scripts/launch_tinker_evals.py \
  --model-variant secpo --max-output-tokens 8192 \
  --benchmarks gpqa_diamond mmlu_pro minerva_math
```

As with previous batches, supervisors persist in the background, save exit status
in `worker.json`, and save final scores only after generation and scoring finish.
The final `progress.json` retains the number of length-limited responses.

## Matched 16,384-token question evaluations

Launched September 8, 2026, approximately 07:44–07:45 UTC (00:44–00:45 PDT),
after 11 regression tests and six successful two-item live smoke tests.
Initial worker PIDs: base MMLU-Pro 1078149, base MATH 1078148, base GPQA 1081184;
SecPO MMLU-Pro 1078150, SecPO MATH 1078163, SecPO GPQA 1081201. These are launch
records; use the per-run worker and progress files for current status.

The next requested comparison repeats **GPQA Diamond (198), MMLU-Pro (1,400),
and Minerva MATH (700) for both models**, with a maximum of 16,384 new output
tokens per response. This is 2,298 questions per model, 4,596 total. It is a
full rerun, not a selection of previously truncated or incorrect answers.

The unadapted base model and final SecPO sampler are unchanged. Both retain
thinking off, the same selected questions, pinned tokenizer, few-shot prompts,
seed 42, deterministic decoding, stop sequences, and scoring methods. The
output cap is overridden at both the adapter and individual task levels,
including MMLU-Pro. The existing runner supports this without a code change.

Artifacts are separate at:

- `outputs/qwen38_27b_base_tinker_utility_max16384/<benchmark>/`
- `outputs/qwen38_27b_tinker_utility_max16384/<benchmark>/`

The runner validates the complete dataset fingerprints and selected-document
hashes before sampling. The new manifests are additionally compared with the
8,192-token runs, allowing only the output budget to differ, and with each
other, allowing only model weights to differ. Each model/benchmark receives a
two-item smoke test; smoke responses at 16,384 tokens are reused in that full
run. Responses from shorter-cap experiments are not imported. Cached Hugging
Face datasets are loaded offline, with their identity checks still enforced.

Launch/resume after successful smoke checks:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 runtime/eval_envs/lm/bin/python scripts/launch_tinker_evals.py \
  --model-variant base --max-output-tokens 16384 \
  --benchmarks gpqa_diamond mmlu_pro minerva_math

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 runtime/eval_envs/lm/bin/python scripts/launch_tinker_evals.py \
  --model-variant secpo --max-output-tokens 16384 \
  --benchmarks gpqa_diamond mmlu_pro minerva_math
```

Each worker uses generation concurrency 12. Background supervision, duplicate
launch locks, resumable completion caches, `worker.json`, `run.log`, and
`progress.json` work as before. The final progress records count length-limited
answers at the new cap. Existing SEP, AlpacaEval, agent, and shorter-cap question
runs are not modified or relaunched by these commands.
