# AgentDojo OpenRouter component integration

## Preserved code and scope

The pre-experiment Qwen3.8 Tinker/Lambda/Brev training and evaluation code is
committed in the original checkout on `archive/qwen38-training-evals-20261009`,
commit `42280080f0f9224717f40a53928a761e554e1db0`. The snapshot contains 170
changed/added files and excludes ignored credentials, runtime caches, logs,
datasets outside the tracked release, and weights. Those runtime artifacts stay
in their existing locations, including `/data2/jaewonc`.

The personal fork is `https://github.com/0315jaewon/SecPO-test`. Publishing the
snapshot was attempted but is **not complete**: command-line Git lacks HTTPS
credentials and the connected GitHub integration rejects writes (403). From an
authenticated terminal on REDS, publish the already-created local branch with:

```bash
git -C /home/jaewonc/SecPO push -u personal archive/qwen38-training-evals-20261009
```

The independent working branch `agentdojo-openrouter-test` is based on mentor
commit `450e6a3f8bcad05277e6348118f34e13283408c3`. Its worktree is
`/home/jaewonc/SecPO/runtime/agentdojo-openrouter`. Existing scripts in the parent
checkout retain their contents and paths. The new integration does not alter
the mentor's search, prompts, victim adapter, scoring, or final-replay code.

## Existing API client and defaults

`src/secpo/evaluation/agentdojo_models.py` already contains:

- `_http_json` (line 63): HTTP POST of JSON, using `urllib.request`.
- `OpenAIChatModel.complete` (line 138): `model`, `messages`, `temperature`,
  `max_tokens`, optional `top_p`/`reasoning_effort`, and Bearer authentication.
- `AgentDojoModelBackend.from_environment` (line 205): independent attacker and
  critic selection and shared/per-role endpoint and key environment variables.

These are OpenAI-compatible message shapes; the code does not require a direct
OpenAI account or even the OpenAI SDK for the attacker/critic calls.

| Role | Released default identifier | Temperature | top_p | Reasoning | Output cap |
|---|---|---:|---|---|---:|
| Attacker | `gcp/google/gemini-2.5-pro` | 0.9 | 0.9 | high | 4096 |
| Critic | `gcp/google/gemini-3.1-flash-lite` | 0.7 | omitted | omitted | 4096 |

Those identifiers are NVIDIA-route defaults, not OpenRouter slugs. The default
victim is separately configured as Qwen3.6-27B thinking-on; do not accidentally
reuse that model configuration when a Qwen3.8 or another victim is intended.

For an actual adaptive run, `configs/agentdojo_openrouter.env.example` shows how
to select GLM as attacker and DeepSeek as critic using the existing environment
interface. This pairing is an example, not a claim about the mentor's intended
experimental design. Victim endpoint/configuration and the intended task/query
budget must be chosen separately. Full default search is 80 pairs with up to
800 search episodes each, plus final replay and setup; this is not a cheap API
connectivity test.

## Bounded compatibility probe

New code:

- `src/secpo/evaluation/agentdojo_openrouter.py`: safe credential-file parsing,
  endpoint/model/token/call guards, credential-free API receipts, and factory
  wiring to the **unchanged** original model classes.
- `scripts/probe_agentdojo_openrouter.py`: each requested model acts once as
  critic and once as attacker using the critic's actual feedback.
- `tests/test_agentdojo_openrouter.py`: mocked HTTP/parsing/feedback tests,
  call-budget and credential guards, and seed-batch/final-replay regression.

The probe uses a synthetic counter task and a fabricated failed victim trace.
It is **not an AgentDojo benchmark evaluation**, has no real victim API calls,
and produces no ASR. It checks both models in both roles before choosing a
model pairing. It keeps the released decoding defaults but deliberately uses
one worker and disables HTTP/critic-format retries to bound this smoke test to
at most four attempted API calls, each capped at 4096 output tokens by default.
No alternative model is configured; OpenRouter's default provider routing is
retained and the selected provider is recorded. The returned model must match the request,
and incomplete (`finish_reason != stop`) completions are saved but rejected.

From this worktree, show the plan without reading credentials or making calls:

```bash
PYTHONPATH=src python scripts/probe_agentdojo_openrouter.py
```

REDS live command (use a fresh output directory for another paid probe):

```bash
cd /home/jaewonc/SecPO/runtime/agentdojo-openrouter
PYTHONPATH=src /home/jaewonc/SecPO/runtime/eval_envs/agentdojo/bin/python \
  scripts/probe_agentdojo_openrouter.py --live \
  --key-file /home/jaewonc/SecPO/secpo_openrouter.env \
  --output-dir /data2/jaewonc/secpo_evals/agentdojo_openrouter_smoke_20261009
```

`manifest.json` records limits and source hashes. `http_cache/` holds complete
request/response receipts without auth headers. `results.json` contains parsed
feedback, proposed injections, checks, token counts, and provider-reported cost.
Failed HTTP calls can incur charges not captured by returned usage receipts.

The first live probe completed all four calls, costing **$0.00424114** in
reported usage. GLM passed both roles. DeepSeek's critic returned valid feedback,
but its attacker used all 4096 completion tokens for reasoning and returned no
final JSON (`finish_reason=length`, GMICloud provider). This is an output-budget
failure, not a key/endpoint authentication failure. Receipts are under the
output directory in the command above. A separate, explicit follow-up tests
DeepSeek with 8192 output tokens while preserving high reasoning and the same
released prompts:

```bash
PYTHONPATH=src /home/jaewonc/SecPO/runtime/eval_envs/agentdojo/bin/python \
  scripts/probe_agentdojo_openrouter.py --live \
  --model deepseek/deepseek-v4.1-flash --max-output-tokens 8192 \
  --key-file /home/jaewonc/SecPO/secpo_openrouter.env \
  --output-dir /data2/jaewonc/secpo_evals/agentdojo_openrouter_smoke_20261009_deepseek8k
```

This follow-up is bounded to **two** attempted API calls. A fresh critic response
is used for its attacker feedback; it is not a deterministic replay of the
first probe's prompt. A larger cap is an explicitly changed test condition,
not a modification of the released defaults.

The 8192-token follow-up **passed both DeepSeek roles**, costing **$0.0014374036**.
Its attacker returned complete JSON after 118 seconds, with 6110 reported
completion tokens (Decart provider). Combined across both probes: **six API
calls, $0.0056785436** in reported charges. These single examples demonstrate
connectivity and interface compatibility, not reliability across all tasks.
The retry used a different provider and fresh critic feedback, so it is not a
controlled comparison of token caps alone. Raw usage is preserved as returned;
the Decart receipt's reasoning-token subcount exceeds its completion-token
total, so those counters should not be treated as independently validated.

| Model | Critic | Attacker / refinement |
|---|---|---|
| `z-ai/glm-5.3-flash` | Passed at 4096 | Passed at 4096 |
| `deepseek/deepseek-v4.1-flash` | Passed at 4096 and 8192 | Truncated at 4096; passed at 8192 |

## Upstream test baseline

Before adding integration code, the four existing AgentDojo test modules gave
**44 passed, 2 failed, 2 skipped**. The failures are retained, not silently
patched or marked xfail:

1. `test_genetic_success_requires_exact_agentdojo_checker` supplies two seeds
   with a one-candidate batch and expects the second seed to run before mutation.
   The new implementation uses only one seed in this configuration.
2. `test_paper80_orchestrator_writes_all_pairs_and_skips_them_on_resume` expects
   80 episode calls, whereas 80 search episodes plus the new 80 final replays
   correctly produce 160 calls.

Two differential tests require an optional archived research source that is not
present. The pinned upstream AgentDojo checkout was verified pristine at
`d3640b5b03a88eb44dad96852e1de1ef437d836e`.

Run all relevant tests, including the inherited failures:

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_agentdojo_qwen36.py tests/test_agentdojo_adaptive.py \
  tests/test_agentdojo_research_search.py tests/test_agentdojo_transport.py \
  tests/test_agentdojo_openrouter.py
```

The full selection after adding integration tests gives **58 passed, 2 failed,
2 skipped**, with the same two inherited failures. Its JUnit report is saved at
`outputs/agentdojo-openrouter-offline-tests.xml`. The narrower
integration/research-search/transport selection gives **46 passed, 2 skipped**.
A successful component probe does not establish attack
effectiveness, end-to-end victim compatibility, or paper-reproduction parity.
