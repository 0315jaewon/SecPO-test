# Base Qwen3.8 static-security comparison

The base Qwen3.8-27B thinking-off run on `ubuntu@209.20.159.207` follows the
lr8e-5 full MMLU-Pro, full Minerva Math, and SEP generation jobs already using
that instance. Only one 27B checkpoint is resident on the H100 at a time.
SEP judging does not need the GPU and does not block the model handoff.

After the user cancelled/replaced SEP, the queue-only launcher is
`scripts/launch_native_static_base_after_sep.py`. It preserves the evaluation
settings below and waits for the fresh three-judge SEP run's **generation** plus
the two existing utility workers. See
[LAMBDA_SEP_ORIGINAL_20260927.md](LAMBDA_SEP_ORIGINAL_20260927.md).

Run from `/home/jaewonc/SecPO`:

```bash
runtime/eval_envs/lm/bin/python -u scripts/launch_native_static_base.py --detach
```

Results, per-episode records, native completions, exact requests, attack payloads,
manifests, source snapshots, logs, and supervisor state live under:
`/data2/jaewonc/secpo_evals/qwen38_native_base_static_20260927`.

## Protocol

| Setting | Value |
|---|---|
| Checkpoint | Qwen/Qwen3.8-27B, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| AgentDojo | 949 attacked episodes + 35 capability checks |
| AgentDyn | 560 attacked episodes + 28 capability checks |
| Attack / defense | `important_instructions` / `repeat_user_prompt` |
| Thinking | Disabled, including preserve_thinking |
| Generation cap / context | 8192 per call / 65536 |
| Temperature / top_p / top_k | 0.7 / 0.8 / 20 |
| min_p / presence / repetition / frequency | 0 / 1.5 / 1 / 0 |
| Request seed / tool iterations | 0 / 15 |
| Parsing | Native vLLM qwen3_coder tool parser, qwen3 reasoning parser |
| Scoring | Original benchmark checker booleans, no LLM judge |
| ASR denominator | All attacked pairs, no capability-based filtering |

`evaluate_native_static_base.py` configures only the checkpoint, endpoint, and
output directory in memory, then invokes the existing `evaluate_native_static.py`
worker. The original `src/secpo/evaluation/agentdojo_qwen36.py` and
`agentdyn_qwen36.py` remain unchanged. Preflight requires all shared source hashes,
task inventories, attack payloads, template, generation settings, and scoring
policy to match **both** prior SecPO static runs. Comparisons are saved in
`base/<benchmark>/protocol_comparison.json`.

The remote launcher verifies all 32 base snapshot files against the September 19
backup. It reuses the isolated vLLM 0.29.0 / torch 2.13.0 / transformers 5.10.4 /
FlashInfer 0.6.18 environment. It does not change the active lr8 scripts. It stops
only the recorded, ownership-checked lr8 server after all three GPU workers have
completed successfully, then runs smoke tests before the full static suites.
The new queue fails without switching models if the preceding queue fails.

The worker preserves cached base episodes for resumption after investigating a
failure. Check the owned server's state before relaunching a failed queue.
Do not change frozen sources within the same output namespace. The queue does
not terminate the Lambda instance; results are written directly on REDS.
