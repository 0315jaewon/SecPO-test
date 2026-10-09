# SEP and AlpacaEval reference utility on Tinker

This batch compares the unadapted `Qwen/Qwen3.8-27B` with the final SecPO sampler
recorded in `outputs/qwen38_27b_tinker/result.json`. It does not retrain either
model and does not run SEP attack-success evaluation.

The original four 8,192-token workers launched on September 8, 2026 at approximately 07:09 UTC
(00:09 America/Los_Angeles), after 29 focused offline tests and four successful
two-example live smoke runs. Initial worker PIDs: base Alpaca 1016827, SecPO
Alpaca 1016828, SecPO SEP 1016829, base SEP 1016830. These are launch records,
not a continuously updated status report; consult each `worker.json` and
`progress.json` for current state.

## Approved protocol

| Suite | Rows per model | Saved reference model | Primary metric |
| --- | ---: | --- | --- |
| SEP utility | 9,160 | `meta-llama/Meta-Llama-3-8B-Instruct` | Win rate |
| AlpacaEval2 utility | 805 | `gpt-4o-2024-05-13` | Length-controlled win rate |

Both models use the same pinned tokenizer, trust-role template, thinking off,
greedy decoding (temperature 0, top-p 1, top-k -1, seed 42), and an 8,192-token
answer cap in the original batch (16,384 in the rerun documented below).
SEP presents the clean instruction as `user` and clean data as
`input`; no injection is presented to the target model or judge. Alpaca prompts
are single user instructions. There is no added system or repeated-user prompt.
SEP contains 9,135 distinct clean prompts: identical prompts reuse generations,
while **all 9,160 rows** and their reference answers remain in the score.

The judge is the user-specified **`gpt-4o-mini` alias**, requested through
OpenRouter as `openai/gpt-4o-mini`. Official OpenAI documentation was consulted
to verify the model alias and Chat Completions interface:
https://developers.openai.com/api/docs/models/gpt-4o-mini . OpenRouter compatibility
was separately verified with a live synthetic comparison returning log probabilities.
This is not a claim that the alias is an immutable dated snapshot.

`alpaca-eval==0.6.6` supplies the existing
`weighted_alpaca_eval_gpt-4o-mini-2024-07-18` evaluator's classifier prompt,
output-order randomization (seed 0), log-probability parser, and metrics. Its
dated model name is explicitly replaced by the approved alias. The judge uses
temperature 1, **one output token**, `logprobs=true`, and `top_logprobs=5`.
The answer cap and judge classification cap serve different purposes.
There is one comparison per example, not a multi-judge panel or two-order vote.
The parser yields a continuous preference in [1, 2], restored to reference-first
orientation: 1 favors the reference, 2 favors the candidate. SEP's ordinary
win rate aggregates these probabilities. Alpaca additionally applies the
installed length-control fit; its 805-example auxiliary data is not used on SEP.

These are customized judge/reference runs, not standard leaderboard scores.
In particular, the repo's Alpaca reference is GPT-4o, not the GPT-4 Turbo reference
described in Meta-SecAlign. Prompting also uses the Qwen3.8/Tinker interface.

## Reference provenance

The user explicitly approved the available SEP files despite the reference
checksum difference. Source files under `SecAlign_cont` remain unchanged.

- Raw test: `/home/jaewonc/SecAlign_cont/data/SEP_dataset_test.json`
  SHA-256 `e659683ae5b10905064d28bca5fd786d1a4ec96ce016a6b93a7282fd977a95a3`.
- Approved SEP reference:
  `/home/jaewonc/SecAlign_cont/data/SEP_dataset_test_Meta-Llama-3-8B-Instruct.json`
  SHA-256 `63e965d2a25d2d561a7b6079048977314cfaacf5d286cf86ae168538c771dd34`.
  The release config instead freezes
  `6defcfdb3ac9da4a03eff21403a0ac58ce52c0c1dc45c23fc48b17a2bba0808c`.
  The runner requires the approved override explicitly, validates nonempty
  reference answers and generator labels, and checks row-by-row alignment to
  the exact raw test set. The release config itself is unchanged.
- Alpaca reference: `data/eval_references/alpaca_eval2_gpt-4o-2024-05-13_official_reference_outputs.json`
  SHA-256 `653c131d58e8d5802203c85e80ca3fb4b378f22e2a0280ad5cd910ff0bee9535`,
  exactly matching the release config. Download source:
  https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/results/gpt-4o-2024-05-13/model_outputs.json .
- Length-control asset: `data/eval_references/alpacaeval_df_gamed.csv`
  SHA-256 `97aeec3f1c7de6dee6fd31fe66f1702623f4614c3efe1c6fc5f4927cd5fd674d`.
  Download source:
  https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/df_gamed.csv .
  The metric's auxiliary-file resolver is bound to this verified local copy;
  the fitting and scoring implementation is unchanged.

## Implementation and tests

- `src/secpo/tinker/reference_eval.py`: reference and credential validation,
  clean prompts, cached OpenRouter transport, installed AlpacaEval integration,
  complete annotation checks, and suite-specific metrics.
- `scripts/evaluate_tinker_reference.py`: preflight, generation, judging,
  scoring, paired manifests, progress, and resumable caches.
- `scripts/launch_tinker_evals.py`: existing durable supervisor/lock workflow,
  extended with `alpacaeval2` and `sep_utility`. Original default benchmarks
  remain unchanged.
- `tests/test_tinker_reference_eval.py`: offline parser/cache tests, trust-role
  rendering, explicit reference override, real annotator order restoration for
  both suites, and a synthetic full 805-row length-control fit.

The installed annotator has a fresh-cache bug when missing-label storage is
disabled. The bridge initializes its empty in-memory cache with a numeric
preference column. Missing/invalid judgments still fail validation; they are
never counted as target-model losses or silently excluded.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 HF_HUB_OFFLINE=1 runtime/eval_envs/lm/bin/python -m pytest \
  tests/test_tinker_reference_eval.py tests/test_reference_utility.py tests/test_tinker_evaluation.py -q
```

## Launch and monitoring

Credentials stay in ignored `secpo_eval_keys.env` and `secpo_openrouter.env`.
Keys are never included in saved manifests or cache identities. OpenRouter
credentials may only be sent to the fixed OpenRouter endpoint; redirects are
disabled. Judge response caches store choices, log probabilities, provider/model
identity, usage, and request hashes, not credentials.

Output roots are `outputs/qwen38_27b_base_tinker_utility_max8192/` and
`outputs/qwen38_27b_tinker_utility_max8192/`, each with `sep_utility/` and
`alpacaeval2/`. Each run records `manifest.json`, `worker.json`, `run.log`,
`progress.json`, `candidate_outputs.json`, `generations.json`, `annotations.json`,
and final `results.json`. Separate `completion_cache/` and `judge_cache/` permit
resumption. Manifests pin source hashes and require paired protocols to match
except for model weights. Smoke results live under `smoke/`; their cached
responses are reused in full evaluation, not scored as extra examples.

After a successful `--smoke 2` with the corresponding reference runner:

```bash
runtime/eval_envs/lm/bin/python scripts/launch_tinker_evals.py \
  --model-variant base --max-output-tokens 8192 --benchmarks sep_utility \
  --reference /home/jaewonc/SecAlign_cont/data/SEP_dataset_test_Meta-Llama-3-8B-Instruct.json \
  --reference-sha256 63e965d2a25d2d561a7b6079048977314cfaacf5d286cf86ae168538c771dd34 \
  --sep-test-dataset /home/jaewonc/SecAlign_cont/data/SEP_dataset_test.json

runtime/eval_envs/lm/bin/python scripts/launch_tinker_evals.py \
  --model-variant base --max-output-tokens 8192 --benchmarks alpacaeval2 \
  --reference data/eval_references/alpaca_eval2_gpt-4o-2024-05-13_official_reference_outputs.json
```

Repeat with `--model-variant secpo` for the trained adapter. Generation concurrency
is 12 and judge concurrency is 8 per worker. A supervisor holds a launch lock,
records worker exit status, and survives the initiating terminal. It does not
automatically restart failed jobs. Completion requires worker exit 0 and a
complete `results.json` with the expected full row count and finite metric.

Counters distinguish new API requests from cache hits in each process. OpenRouter
`reported_cost_usd` sums costs returned for successful newly saved judgments;
it is not an end-to-end invoice and may omit charged failed requests. Tinker
counters are token usage, not dollar cost. The respective billing accounts are
authoritative. Cached responses persist even if a later judgment or metric fails.

## 16,384-token rerun — September 8, 2026

The user requested both reference-utility suites again with a 16,384-token
generated-answer cap. The model pair is the same unadapted base and final SecPO
adapter as before, not two separately fine-tuned adapters:

- Base: `Qwen/Qwen3.8-27B`.
- SecPO: `tinker://fcfc57ba-d2ab-568d-8f90-b3f60ab70560:train:0/sampler_weights/secpo-final-sampler`.

All 29 focused offline tests passed again. All four full-data preflights passed;
the maximum rendered prompt lengths were 506 tokens for AlpacaEval and 262 for
SEP, so every prompt fits alongside the requested answer budget. Each new
manifest was compared with its corresponding 8,192-token manifest: the only
changed field was `default_max_gen_toks`. This includes exact agreement of
reference hashes, prompt hashes, judge settings, evaluator assets, and
implementation hashes. Each base/SecPO manifest pair differs only in checkpoint.
No evaluator implementation was changed for this rerun.

Four two-example live smoke checks completed with exit code 0 and validated
generation/judgment artifacts. Full workers launched at 22:13:16–22:13:23 UTC
(15:13 America/Los_Angeles). These are launch records, not completion claims:

| Model | Benchmark | Full rows | Worker PID | Supervisor PID |
| --- | --- | ---: | ---: | ---: |
| Base | AlpacaEval2 | 805 | 2327424 | 2327246 |
| Base | SEP utility | 9,160 | 2327640 | 2327428 |
| SecPO | AlpacaEval2 | 805 | 2327954 | 2327720 |
| SecPO | SEP utility | 9,160 | 2328263 | 2328019 |

The new benchmark subdirectories are under
`outputs/qwen38_27b_base_tinker_utility_max16384/` and
`outputs/qwen38_27b_tinker_utility_max16384/`. Their `alpacaeval2/` and
`sep_utility/` directories contain independent caches and run artifacts.
Existing 8,192-token outputs and completed 16,384-token general benchmarks were
not overwritten. Only the new-cap smoke responses are reused in the full runs;
old-cap target answers are not reused.

Exact launcher commands, run from `/home/jaewonc/SecPO`:

```bash
env HF_HUB_OFFLINE=1 runtime/eval_envs/lm/bin/python scripts/launch_tinker_evals.py \
  --model-variant base --max-output-tokens 16384 --benchmarks alpacaeval2 \
  --reference data/eval_references/alpaca_eval2_gpt-4o-2024-05-13_official_reference_outputs.json

env HF_HUB_OFFLINE=1 runtime/eval_envs/lm/bin/python scripts/launch_tinker_evals.py \
  --model-variant base --max-output-tokens 16384 --benchmarks sep_utility \
  --reference /home/jaewonc/SecAlign_cont/data/SEP_dataset_test_Meta-Llama-3-8B-Instruct.json \
  --reference-sha256 63e965d2a25d2d561a7b6079048977314cfaacf5d286cf86ae168538c771dd34 \
  --sep-test-dataset /home/jaewonc/SecAlign_cont/data/SEP_dataset_test.json

env HF_HUB_OFFLINE=1 runtime/eval_envs/lm/bin/python scripts/launch_tinker_evals.py \
  --model-variant secpo --max-output-tokens 16384 --benchmarks alpacaeval2 \
  --reference data/eval_references/alpaca_eval2_gpt-4o-2024-05-13_official_reference_outputs.json

env HF_HUB_OFFLINE=1 runtime/eval_envs/lm/bin/python scripts/launch_tinker_evals.py \
  --model-variant secpo --max-output-tokens 16384 --benchmarks sep_utility \
  --reference /home/jaewonc/SecAlign_cont/data/SEP_dataset_test_Meta-Llama-3-8B-Instruct.json \
  --reference-sha256 63e965d2a25d2d561a7b6079048977314cfaacf5d286cf86ae168538c771dd34 \
  --sep-test-dataset /home/jaewonc/SecAlign_cont/data/SEP_dataset_test.json
```

`HF_HUB_OFFLINE=1` only uses the cached pinned Hugging Face tokenizer; Tinker
generation and OpenRouter judging still make paid network requests. The exact
spawned evaluator argument vector is also saved in each `worker.json.command`.
Generation concurrency remains 12 and judge concurrency 8 per worker. The judge
is still `openai/gpt-4o-mini`, with its separate one-token classification cap.
