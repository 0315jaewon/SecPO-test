# Native-adapter static-security evaluation

This run supersedes the custom-parser experiment in
`qwen38_lambda_static_20260924`. Its results and caches remain intact, but are not
mixed into this run. The user requested the original SecPO evaluation logic,
with separate launch/configuration glue for Qwen3.8 thinking-off.

## What is unchanged

The wrapper checks that these original modules are byte-identical to git HEAD:

- `src/secpo/evaluation/agentdojo_qwen36.py`: original `NativeToolLLM` request
  construction and native JSON tool-response parsing.
- `src/secpo/evaluation/agentdyn_qwen36.py`: original `install()` adapter,
  trusted-user/untrusted-input conversion, native tool parsing, XML fallback,
  and bounded request retry behavior.

The pinned AgentDojo/AgentDyn attack generators, defense pipeline, tool-loop
limits, environment/tool functions, utility predicates, security predicates,
and exception handling are not patched. Both benchmarks use their original
`run_task_with_injection_tasks` and `run_task_without_injection_tasks` functions.
`security=True` means attacker success, not successful defense.

The vLLM server uses its built-in `qwen3_coder` tool parser and `qwen3` reasoning
parser. The custom `secpo.tinker.agent_eval.parse_tool_response` is not called.
No repairs to incomplete tool calls or whitespace stripping are implemented in
the wrapper; whatever the original native parsing returns is evaluated.

## Launch-only configuration

- Checkpoints: Qwen3.8-27B SecPO lr=4e-5, then lr=8e-5; same previously verified
  standalone BF16 weights as the earlier Lambda runs.
- `important_instructions` attack and `repeat_user_prompt` defense.
- Thinking off, 8,192 output tokens per call, 65,536 context tokens, seed 0.
- Temperature 0.7, top_p 0.8, top_k 20, presence penalty 1.5. Server defaults
  explicitly set min_p=0, repetition penalty=1, frequency penalty=0.
- AgentDojo's existing configurable dataclass supplies the overrides. AgentDyn
  uses its original adapter's environment-variable interface; its paper-specific
  Qwen3.6 dataclass and CLI are neither edited nor falsified.
- A separate template is derived from the checksum-verified native Qwen3.8
  template. System text is trusted `user`; native `tool` messages and AgentDyn's
  already-converted `input` messages render as untrusted `input`. Native function
  syntax and argument rendering remain unchanged. Thinking-off is requested
  through the original adapters and the server defaults.
- An in-memory model-name registry entry lets the original attack address the
  actual Qwen model as `Qwen 3`, with the original `Emma Johnson` user name.
- The existing date/datetime-to-ISO JSON serializer is selected as the tool
  formatter's callback. It preserves ordinary JSON formatting; no simulated tool
  function or checker is replaced.
- An I/O-only cache records original chat requests and full structured server
  responses. It returns responses to the unchanged original adapter for parsing.

There are 949 attacked AgentDojo episodes plus 35 attacker-capability checks per
model, and 560 attacked AgentDyn episodes plus 28 capability checks per model.
Capability checks are not included in or used to filter the ASR denominator.
The summary reports per-suite metrics, macro ASR (mean across suites), and micro
ASR (all attacked pairs pooled). Unhandled errors prevent a final summary.

## Commands

From `/home/jaewonc/SecPO`, first offline-test both original adapters:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 runtime/eval_envs/agentdojo/bin/python -m pytest -q tests/test_native_static.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 runtime/eval_envs/agentdyn/bin/python -m pytest -q tests/test_native_static.py
```

Initial launch (replaces only the recorded, owned legacy run processes):

```bash
runtime/eval_envs/lm/bin/python -u scripts/launch_native_static.py --detach --replace-legacy
```

The supervisor validates all four manifests, deploys the separate server script
and template, then smoke-tests before each full run. AgentDojo's smoke includes
the three formerly failing Slack `user_task_7` injection pairs as well as one
capability check and attacked episode per suite (11 episodes total). AgentDyn's
smoke contains one of each per suite (6 total). Smoke outcomes need not be utility
wins; they must complete successfully through the original checkers. Full runs
reuse these smoke trajectories within this new, unchanged protocol.

An individual full worker command, normally launched by the supervisor:

```bash
runtime/eval_envs/agentdojo/bin/python -u scripts/evaluate_native_static.py \
  --benchmark agentdojo --variant secpo_lr4e5 --phase full
```

Use the AgentDyn environment and `--benchmark agentdyn` for AgentDyn, and
`--variant secpo` for lr=8e-5. Do not launch overlapping workers manually.

## Data and ownership

All local output lives in
`/data2/jaewonc/secpo_evals/qwen38_native_static_20260924/`:

- `supervisor.json`, `supervisor.log`, and `launch.json`: live queue state.
- `source_identity.json`, `code_snapshot/`: exact launch and original source.
- `remote/`: copied server records and logs, including checkpoint validation.
- `{secpo_lr4e5,secpo}/{agentdojo,agentdyn}/manifest.json`: exact protocol.
- Per job: `progress.json`, `results.json`, `smoke/`, `tasks/`, `episodes/`, and
  `completion_cache/` contain progress, metrics, checker outcomes, trajectories,
  and full original native chat requests/responses.

Remote project: `/lambda/nfs/jaewon/secpo_native_static_20260924` on
`ubuntu@209.20.159.198`. The existing isolated inference environment and verified
weights are reused. Remote loopback port 28781 is tunneled to REDS loopback 18781;
no provider keys are required. Both benchmarks share one model server; the two
models are sequential. The supervisor stops its model server after completion,
but does not shut down the billed Lambda instance.
