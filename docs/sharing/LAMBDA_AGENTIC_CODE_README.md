# Lambda AgentDojo / AgentDyn evaluation code

This is a **source-code review package** for the clean-utility evaluations of
Qwen3.8-27B base, SecPO lr=8e-5, and SecPO lr=6e-5. It contains the scripts used
in the September 16, 2026 Lambda experiment, not a newly rewritten evaluator.

It is **not a standalone, unzip-and-run reproduction environment**. Historical
commands and remaining setup dependencies are documented below. No evaluation
is started by extracting this archive.

## Start here

- `scripts/evaluate_tinker_agents.py`: the shared AgentDojo / AgentDyn worker
  used for the base and lr8e-5 models.
- `src/secpo/tinker/agent_eval.py`: exact task counts and benchmark revisions,
  native Qwen tool-call rendering/parsing, trusted/untrusted role conversion,
  `repeat_user_prompt` defense, and date serialization for tool results.
- `src/secpo/inference/lambda_eval.py`: Lambda/vLLM completion client, model
  identity checks, protocol comparison, generation parameters and token caches.
- `configs/lambda_qwen38_*_recommended.json`: the three model configurations.
- `lr6e5_overrides/`: the two exact files that differed for the lr6e-5 run,
  at the same relative paths as the original files. They add third-model
  routing and consistency checks, not different prompting or scoring.
- `server/server_remote.py` and `server/server_lr6e5_remote.py`: exact server
  launch scripts. These contain historical absolute paths and are for inspection;
  do not run unchanged on an unrelated machine.
- `RUN_COMMANDS.json`: all six exact historical full-run command argument lists.
  They contain local paths but no credential values.
- `ARCHIVE_MANIFEST.json`: file hashes, source provenance and benchmark pins.

The supporting `src/secpo/` Python package is included because the evaluator
imports shared modules and package initializers. Some of these also implement
training or other benchmarks; they are not invoked as experiments by this runner.

Despite its historical name, `evaluate_tinker_agents.py` uses Lambda/vLLM when
`--inference-config` is supplied. Target-model generation does not use Tinker
in this mode. The harness ran on the local workstation and called the GPU
server on Lambda through a private SSH tunnel.

## Protocol used

| Setting | Value |
| --- | --- |
| Evaluation | Clean utility, not attack success |
| Defense | `repeat_user_prompt` |
| Attack | None (`null`); **not** `important_instructions` |
| AgentDojo | v1.2.1; banking 16, slack 21, travel 20, workspace 40: 97 tasks |
| AgentDyn | v1.2.2; shopping 20, github 20, dailylife 20: 60 tasks |
| Thinking | Off |
| Temperature / top-p / top-k | 0.7 / 0.8 / 20 |
| Min-p / presence penalty | 0.0 / 1.5 |
| Repetition / frequency penalty | 1.0 / 0.0 |
| Output cap | 8,192 tokens per model call |
| Agent sampling seed | 0 |
| Tool loop | 15 iterations |
| Concurrency | 3 tasks per benchmark |
| Scoring | Benchmark's exact local task-success checker; no LLM judge |
| Reported overall utility | Unweighted mean of suite utility rates (macro average) |

The initial Tinker defaults visible in the worker (temperature 1 and top-p
0.95) are checked against the earlier protocol and then overridden by the
Lambda configuration. The effective Lambda settings are those in this table.

## Historical evaluation commands

From the original repository root, with the base model served and the two
benchmark environments installed:

```bash
runtime/eval_envs/agentdojo/bin/python -u scripts/evaluate_tinker_agents.py \
  --benchmark agentdojo \
  --model-variant base \
  --inference-config configs/lambda_qwen38_base_recommended.json \
  --output-dir outputs/qwen38_lambda_recommended_20260916/base/agentdojo \
  --concurrency 3

runtime/eval_envs/agentdyn/bin/python -u scripts/evaluate_tinker_agents.py \
  --benchmark agentdyn \
  --model-variant base \
  --inference-config configs/lambda_qwen38_base_recommended.json \
  --output-dir outputs/qwen38_lambda_recommended_20260916/base/agentdyn \
  --concurrency 3
```

For lr8e-5, the variant is `secpo`, the configuration is
`configs/lambda_qwen38_secpo_recommended.json`, and the output branch is
`outputs/qwen38_lambda_recommended_20260916/secpo/`. The matching SecPO model
must be served; changing the evaluator flag alone does not load model weights.

The lr6e-5 commands used the isolated worker at
`runtime/lambda_suite_20260916/lr6e5_code/scripts/evaluate_tinker_agents.py`,
variant `secpo_lr6e5`, and configuration
`configs/lambda_qwen38_secpo_lr6e5_recommended.json`. The isolated workspace is
not included. Its two code changes are in `lr6e5_overrides/` for review; they
are not independently runnable from that directory. See `RUN_COMMANDS.json`.

## Requirements before reproducing on a different machine

1. Install separate benchmark environments. Both distributions import as
   `agentdojo`, so they must not replace each other in one environment.
   - AgentDojo: https://github.com/ethz-spylab/agentdojo.git at
     `d3640b5b03a88eb44dad96852e1de1ef437d836e` (package 0.1.30).
   - AgentDyn: https://github.com/SaFo-Lab/AgentDyn.git at
     `5353cf7615b135cace8d07c8f12dac53a16b6db3` (package 0.1.35), with the
     included `third_party/patches/agentdyn.patch`.
   - The worker also expects the pinned Git checkouts at
     `runtime/benchmarks/agentdojo` and `runtime/benchmarks/agentdyn` to verify
     installed source. This package does not include those repositories.
2. The original harness environments used Python 3.11.15, Transformers 4.57.1,
   tokenizers 0.22.1, Torch 2.8.0, Pydantic 2.12.3, OpenAI SDK 2.6.1,
   Requests 2.32.5, and Tinker 0.27.0. This list records key versions; it is
   not a complete dependency lock file.
3. Serve the matching base or merged model on Lambda. The original GPU server
   used vLLM 0.29.0, Transformers 5.10.4, Torch 2.13.0, BF16 weights, and a
   65,536-token context. Pin base/tokenizer revision
   `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Weights are not in this ZIP.
4. Provide the local endpoint `http://127.0.0.1:18080/v1` via a private tunnel
   or a local server. Served model names are `secpo-suite-base`,
   `secpo-suite-secpo`, and `secpo-suite-secpo_lr6e5` respectively.
5. Supply the original protocol-comparison manifests/checkpoint metadata, or
   make an explicit portability adaptation. The unmodified code reads:
   - `outputs/qwen38_27b_base_tinker_utility/{agentdojo,agentdyn}/manifest.json`
   - `outputs/qwen38_27b_tinker_utility/{agentdojo,agentdyn}/manifest.json`
   - `outputs/qwen38_27b_tinker/result.json` for the lr8e-5 checkpoint identity
   - `outputs/qwen38_27b_tinker_lr6e5/result.json` for the lr6e-5 identity
   - lr6e-5 additionally checks the lr8e-5 Lambda comparison manifests under
     `outputs/qwen38_lambda_recommended_20260916/secpo/`.
   These experiment artifacts are intentionally excluded from this code ZIP.
6. The shared legacy credential loader still requires `HF_TOKEN` and
   `TINKER_API_KEY` from a local credential file, even for Lambda inference.
   Use your own credentials or remove that redundant requirement in a clearly
   labeled portability copy. No OpenRouter/OpenAI judge key is needed for
   these clean agentic utility benchmarks. Never copy another person's keys.
7. Use fresh output directories for a genuinely new run. The worker reuses
   completed task records and generation caches when pointed to an existing run.

Changing model, template, benchmark revision or decoding settings constitutes
a different experiment. Identical stochastic settings do not guarantee
bit-identical results across inference hardware or software versions.

## Exclusions and licenses

No credentials, environment credential files, SSH keys, model weights,
generation caches, datasets, per-task outputs, transcripts, or virtual
environments are included. The archive contains no symlinks. The broad
seven-benchmark queue scripts are not needed to inspect these agentic runners
and are not included.

SecPO source license: `LICENSE`. AgentDyn patch license notice:
`third_party/licenses/AgentDyn-LICENSE`. External benchmark repositories
retain their own licenses. This package has not been tested as an independent
installation; source identity and ZIP integrity have been verified.
