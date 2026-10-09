# Qwen-recommended non-thinking utility evaluations

Run ID: `qwen38_lambda_recommended_20260916`.

This experiment evaluates the pinned Qwen3.8-27B base model and the **original
SecPO checkpoint trained with learning rate 8e-5**. It does not select the
later 6e-5 checkpoint. These are utility evaluations, not ASR experiments.

| Benchmark | Examples per model | Output-token cap |
|---|---:|---:|
| GPQA Diamond | 198 | 16,384 |
| MMLU-Pro | 1,400, original mentor-selected indices | 16,384 |
| Minerva MATH | 700, original mentor-selected indices | 16,384 |
| AgentDojo | 97 | 8,192 per model call |
| AgentDyn | 60 | 8,192 per model call |
| AlpacaEval2 utility | 805 | 16,384 |
| SEP utility | 9,160 rows / 9,135 distinct prompts | 16,384 |

Generation uses `enable_thinking=False`, temperature 0.7, top-p 0.8, top-k 20,
min-p 0, presence penalty 1.5, repetition penalty 1, and frequency penalty 0.
General/reference generations retain seed 42; agent generations retain seed 0.
The agent loop retains its 15-iteration limit and repeat-user-prompt defense.
Output limits are separate from the server's 65,536-token total-context limit.

AlpacaEval2 retains the existing GPT-4o-2024-05-13 reference outputs and SEP
retains the supplied Llama-3-8B-Instruct reference outputs. Both retain the
existing `openai/gpt-4o-mini` judge through OpenRouter, including its original
judging parameters. Target-model sampling settings do not change the judge.

## Code and execution

`scripts/launch_lambda_evals.py` constructs and records each exact command. It
uses the existing three harnesses with the optional `--inference-config` flag:

- `scripts/evaluate_tinker_lm.py`: GPQA, MMLU-Pro and Minerva MATH.
- `scripts/evaluate_tinker_agents.py`: AgentDojo and AgentDyn.
- `scripts/evaluate_tinker_reference.py`: AlpacaEval2 and SEP utility.
- `src/secpo/inference/lambda_eval.py`: private vLLM completion client,
  protocol verification and content-addressed raw-token completion caching.
- `configs/lambda_qwen38_{base,secpo}_recommended.json`: inference identities
  and all target-model sampling parameters.

The old script names are retained because their existing dataset, rendering,
scoring and judging logic is reused; **these runs perform inference on Lambda,
not through Tinker**. Tinker was used to download the saved training adapter.
The original standalone greedy Lambda GPQA reproduction script is unchanged.

On this workspace, with the pinned environments and private SSH tunnel ready:

```bash
runtime/eval_envs/lm/bin/python scripts/launch_lambda_evals.py --phase suite --detach
```

The supervisor verifies all preflights and small end-to-end smoke tests before
full evaluation. On the single H100 it smoke-tests base then SecPO, runs the
full SecPO suite, then switches to the full base suite. Seven benchmarks can
run concurrently within a model. A failed phase stops progression and records
the failure; it is not silently treated as a completed evaluation.

The launcher is specific to this workspace and Lambda instance. It is not a
portable one-command installer: another machine needs the same pinned harness
environments, dataset access, private reference/index files and SSH setup.
Never share the credential files or SSH private key.

## Model conversion and comparability

Base/tokenizer revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
SecPO source:

```text
tinker://fcfc57ba-d2ab-568d-8f90-b3f60ab70560:train:0/sampler_weights/secpo-final-sampler
```

The exported adapter has rank 64, alpha 32, dropout 0, attention/MLP updates,
and no unembedding updates. Official Tinker Cookbook commit
`dfe4d77e8e8c7df7e9cd9539220b54a01d1eaa40` merges it into the pinned base model
with the shard strategy, retaining BF16 weights. All 496 adapter updates were
applied to 400 target tensors. Independent checks verify the adapter checksum,
all 992 finite adapter tensors, representative standard and fused-hybrid
projections, and unchanged input embeddings/output head.

Serving uses vLLM 0.29.0, Transformers 5.10.4, Torch 2.13.0 and one H100.
The local prompt/scoring environment remains pinned to the prior evaluations.
Prompt token IDs are sent directly, and raw output token IDs are saved.
BF16 merging and a different inference engine are additional differences from
the historical Tinker runs; bit-identical logits across engines are not assumed.
Sampling is stochastic, so a single run should not be interpreted as a
noise-free estimate of the effect of changing decoding settings.

## Artifacts and monitoring

Local root: `outputs/qwen38_lambda_recommended_20260916/`.

- `supervisor.json` and `supervisor.log`: queue/process status.
- `<variant>/<benchmark>/{preflight,smoke,full}_worker.json`: exact commands,
  PIDs, exit codes and timestamps.
- `<variant>/<benchmark>/manifest.json`: complete evaluation protocol.
- `progress.json`, `results.json`, completion caches and per-example outputs:
  written as workers run. Smoke results are under `smoke/`, not full scores.
- `source_identity.json` and `code_snapshot/`: exact code/config snapshots.
- `merge_validation.json`: independent checkpoint-conversion checks.
- `backup_status.json`: most recent persistent-filesystem sync outcome.

Persistent Lambda root: `/lambda/nfs/jaewon/secpo_qwen38_eval_20260916/`.
It contains model weights, adapter, pinned converter code, server logs and
the mirrored local `results/` tree. During active evaluation the supervisor
syncs local artifacts about once per minute without deleting either copy.
Do not terminate the instance while generation is in progress. This launcher
does not create, resize or automatically shut down billable cloud instances.

## Additional lr=6e-5 checkpoint queued afterward

The lower-learning-rate checkpoint was subsequently added as a third model:

```text
tinker://1b50b5c0-83c3-5a3c-a3e2-faeeded4aa00:train:0/sampler_weights/secpo-final-sampler
```

It uses **all seven benchmarks, the same sample selections, caps, sampling
settings, seeds, references and judge** described above. The inference config
is `configs/lambda_qwen38_secpo_lr6e5_recommended.json`; only weight identity
and model labels differ from the lr=8e-5 inference configuration.

```bash
runtime/eval_envs/lm/bin/python scripts/queue_lambda_lr6e5_evals.py --mode queue --detach
```

This is a separate durable continuation of the original queue. The order is
**SecPO lr=8e-5 → base → SecPO lr=6e-5**. The continuation waits for the original
supervisor and all 14 full-evaluation workers to finish successfully, then
acquires the original GPU-ownership lock before switching servers. It does not
interrupt or restart the active evaluations. If the predecessor fails or is
paused, the third model remains queued instead of taking the GPU.

Because the active experiment pins its source hashes, its code was not edited
to add the third model. `scripts/prepare_lambda_lr6e5_eval.py` materializes an
isolated local snapshot at `runtime/lambda_suite_20260916/lr6e5_code`. Exactly
two files receive model-routing changes: the Lambda backend's variant/protocol
validation and the agent worker's accepted model variants. Prompt rendering,
generation transport, parsing, scoring and judging remain unchanged. Source
hashes, the two routing changes and all exact commands are saved with the run.
No credential files are copied into that snapshot or transferred to Lambda.

The lr=6e-5 results and queue status live under:

```text
outputs/qwen38_lambda_recommended_20260916/secpo_lr6e5/
```

Inspect that directory's `supervisor.json`, `supervisor.log`, per-benchmark
worker/progress files, `code_snapshot/provenance.json` and `merge_validation.json`.
Its artifacts are mirrored to the corresponding `results/secpo_lr6e5/`
directory on the same persistent Lambda filesystem. The checkpoint is merged
and numerically validated on CPU while the original GPU work continues; its
generation smoke tests wait until the GPU handoff.
