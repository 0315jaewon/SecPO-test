# AgentDojo / AgentDyn static-security runs

This is a separate experiment from the earlier **clean utility** evaluations.
It adds the upstream `important_instructions` attack to the existing
`repeat_user_prompt` defense, on the saved Qwen3.8-27B SecPO lr=4e-5 and lr=8e-5
checkpoints. It does not run a search-based or adaptive attacker.

## Protocol

| Setting | Value |
| --- | --- |
| Target models, in queue order | `secpo_lr4e5`, then `secpo` (lr=8e-5) |
| Defense | `repeat_user_prompt` |
| Attack | `important_instructions` |
| Thinking | Off |
| Per-call output cap | 8,192 tokens |
| Context limit | 65,536 tokens |
| Tool-loop limit | 15 iterations |
| Sampling | temperature 0.7, top_p 0.8, top_k 20, min_p 0, presence_penalty 1.5, repetition_penalty 1, frequency_penalty 0 |
| Request seed | 0 |
| Weights | Previously exported standalone BF16 merged checkpoints; SHA256-verified against the REDS archive |
| Inference runtime | vLLM 0.29.0, Transformers 5.10.4, Torch 2.13.0 |
| Judge | No external LLM judge; the pinned benchmark's environment-based checkers |

The benchmark source revisions, AgentDyn patch, tokenizer/template, checkpoints,
and sampling configuration are checked against the previous clean-evaluation
manifests. The pinned benchmarks predate Qwen3.8, so the attack's model-name
registry is extended in memory with the display name `Qwen 3`. The attack's
claimed user name is the upstream default `Emma Johnson`. These values and the
exact generated injection payloads are recorded in the new manifests.

| Benchmark / suite | User tasks | Injection objectives | Attacked episodes per model |
| --- | ---: | ---: | ---: |
| AgentDojo banking | 16 | 9 | 144 |
| AgentDojo slack | 21 | 5 | 105 |
| AgentDojo travel | 20 | 7 | 140 |
| AgentDojo workspace | 40 | 14 | 560 |
| **AgentDojo total** | **97** | **35** | **949** |
| AgentDyn shopping | 20 | 9 | 180 |
| AgentDyn github | 20 | 9 | 180 |
| AgentDyn dailylife | 20 | 10 | 200 |
| **AgentDyn total** | **60** | **28** | **560** |

Each injection objective also has a capability check, running the attack goal
as a legitimate user instruction without injecting a tool response: 35 for
AgentDojo and 28 for AgentDyn per model. These are not part of the ASR denominator
and are not used to filter attacked episodes. Across both models this is 3,018
attacked episodes plus 126 capability checks.

`security=True` in upstream traces means **the attacker succeeded**. The new
results report macro ASR (unweighted average over suites), micro ASR (all pairs
pooled), and per-suite utility under attack. Lower ASR is better. Infrastructure
errors block a final score, rather than being counted as either successes or
failures. Normal checker failures remain valid benchmark outcomes.

## Files and launch

- `scripts/launch_lambda_static.py`: detached queue, strict SSH tunnel, model
  lifecycle, progress tracking, and automatic model switch.
- `scripts/evaluate_lambda_static_agents.py`: source/config/payload preflight,
  smoke test, cached static attacked episodes, raw traces, and scoring.
- `src/secpo/inference/static_agent_eval.py`: model identity, static-attack
  registry adaptation, task inventory, and ASR aggregation.
- `scripts/lambda_static_remote.py`: isolated remote inference environment,
  checkpoint checksum verification, and private model server lifecycle.
- `src/secpo/tinker/agent_eval.py` and `src/secpo/inference/lambda_eval.py`:
  reused trust-role agent pipeline and raw-token vLLM sampling adapter.

From `/home/jaewonc/SecPO` on REDS:

```bash
runtime/eval_envs/lm/bin/python -u scripts/launch_lambda_static.py --detach
```

The queue first validates all four jobs, waits for the Lambda environment and
checkpoint verification, then starts lr=4e-5. For each model it runs a smoke test
(one capability check and one attacked episode per suite), followed by the full
benchmarks. Smoke episodes are reused in the full results. AgentDojo and AgentDyn
have separate Python environments and run concurrently against one model server;
the two model checkpoints run sequentially on the single GPU. The worker command
for an individual full job is:

```bash
runtime/eval_envs/agentdojo/bin/python -u scripts/evaluate_lambda_static_agents.py \
  --benchmark agentdojo --variant secpo_lr4e5 --phase full
```

Substitute both occurrences of `agentdojo` with `agentdyn` for AgentDyn, and use
`--variant secpo` for lr=8e-5. The supervisor should normally own these launches,
so the correct model is serving and the smoke test has passed. Do not start a
second worker while the first owns its lock.

## Storage and monitoring

All local artifacts are saved directly to:

```text
/data2/jaewonc/secpo_evals/qwen38_lambda_static_20260924/
  supervisor.json / supervisor.log
  source_identity.json / code_snapshot/
  remote/                     # copied setup, checkpoint-validation, server logs
  {secpo_lr4e5,secpo}/{agentdojo,agentdyn}/
    manifest.json             # exact settings, episode identities, code hashes
    attack_payloads.json      # exact injected tool-response text
    preflight.json
    smoke/results.json
    progress.json             # live counts, incomplete scores withheld
    results.json              # final metrics after all episodes succeed
    tasks/                    # per-episode checker outcomes
    episodes/                 # upstream conversation/tool traces
    completion_cache/         # raw inference requests and token outputs
```

Useful read-only status commands:

```bash
cat /data2/jaewonc/secpo_evals/qwen38_lambda_static_20260924/supervisor.json
tail -n 10 /data2/jaewonc/secpo_evals/qwen38_lambda_static_20260924/secpo_lr4e5/agentdojo/full.log
cat /data2/jaewonc/secpo_evals/qwen38_lambda_static_20260924/secpo_lr4e5/agentdojo/progress.json
```

Remote host: `ubuntu@209.20.159.198`. The server listens only on remote loopback
port 28780, tunneled to REDS loopback port 18780. No provider API keys are needed.
The remote project lives on `/lambda/nfs/jaewon/secpo_static_security_20260924`;
weights are reused from the prior persistent-NFS experiment, not re-exported.
The isolated runtime lives on the instance disk. The queue stops its model server
when done, but **does not shut down the billed Lambda instance**.

Completed episodes and raw completions are cached and reused after an interrupted
run. Manifests reject changed source code/settings rather than mixing protocols.
Before restarting an interrupted supervisor, inspect its worker/server state so
no orphan worker overlaps a restarted queue.
