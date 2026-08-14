# Source-to-release refactoring map

This map records how the compact public artifact was derived from the
research workspace. A row means scientific lineage, not necessarily a file
copy. Unless marked **exact asset**, the released path is a clean refactor or
behavioral reimplementation with private paths, providers, scheduler state,
watchdogs, generated outputs, and one-time recovery logic removed.

## Public naming changes inside the release

| Earlier release name | Current public name | Reason |
| --- | --- | --- |
| `src/secpo/train.py` | `src/secpo/training_cli.py` | Distinguishes the shared CLI from the trainer implementation |
| `src/secpo/online.py` | `src/secpo/online_training.py` | Makes the Llama online-training role explicit |
| `scripts/train_llama31_online.sh` | `scripts/train_llama31_8b_online.sh` | Adds exact model size and preserves action/model/mode ordering |
| `scripts/train_qwen36_offline.sh` | `scripts/train_qwen36_27b_offline.sh` | Adds exact model size and preserves action/model/mode ordering |
| `scripts/run_llama31_online_worker.sh` | `scripts/serve_llama31_8b_online_attacks.sh` | Describes the long-running service action |
| `scripts/run_online_attack_worker.py` | `scripts/serve_online_attack_requests.py` | Describes the request-serving contract rather than cluster role |
| `evaluation/static_sep.py` | `evaluation/sep_static.py` | Uses benchmark-first evaluation naming |
| `evaluation/adaptive_sep.py` | `evaluation/sep_adaptive.py` | Uses benchmark-first evaluation naming |
| `evaluation/judge.py` | `evaluation/sep_judge.py` | Identifies which outputs are judged |
| `evaluation/utility.py` | `evaluation/lm_eval_utility.py` | Names the actual external harness |
| `evaluation/agentdojo.py` | `evaluation/agentdojo_qwen36.py` | Makes the model-specific adapter explicit |
| `scripts/evaluate_static_sep.py` | `scripts/evaluate_sep_static.py` | Aligns with `sep_static` |
| `scripts/evaluate_adaptive_sep.py` | `scripts/evaluate_sep_adaptive.py` | Aligns with `sep_adaptive` |
| `scripts/evaluate_judgments.py` | `scripts/judge_sep_outputs.py` | Uses a direct verb/object name |
| `scripts/evaluate_utility.py` | `scripts/evaluate_lm_eval_utility.py` | Names the external harness |
| `scripts/evaluate_agentdojo.py` | `scripts/evaluate_agentdojo_qwen36_27b.py` | Adds the exact evaluated model |

Configuration files mirror the command that consumes them, for example
`train_llama31_8b_online.json`, `train_qwen36_27b_offline.json`, and
`evaluate_lm_eval_qwen36_27b.json`.

## Research source lineage

| Research-workspace source | Clean release destination | Treatment |
| --- | --- | --- |
| `dpo_qwen36_27b_benign_ref.py` | `src/secpo/{data.py,loss.py,trainer.py,training_cli.py}` | Modular source-backed refactor of the four-forward benign-reference objective with stricter schema and truncation checks |
| `dpo_qwen36_27b_benign_ref.sh`, `accelerate_config_dpo_27b.yaml`, `deepspeed_config_zero3.json` | `scripts/train_qwen36_27b_offline.sh`, `configs/train_qwen36_27b_offline.json`, shared distributed configs | Portable launcher/config refactor; no scheduler directives |
| `lora_dpo_distributed.py`, `helpers/llama3.1_8B_lora.yaml`, `adv_secalign_llama3.1_8b.sh` | Shared training modules, `src/secpo/online_training.py`, Llama config and launchers | Clean Hugging Face implementation and atomic file-exchange replacement; not a bit-for-bit Torchtune port |
| Raw Llama preference fields `rejected_input` and `rejected_input_whole` | `scripts/prepare_preferences.py --online-pair` plus the public preference schema | Explicit online-only normalization maps the goal, derives a deterministic domain-separated content-hash ID when absent, and validates exact one-span injection removal. Default/offline preparation remains unchanged. |
| `attacker_interface_pair.py`, `pair_attack.py`, `attack_utils.py`, and the PAIR/final-judge helpers in `utils.py` | `src/secpo/backends/llama31_current_policy_pair.py`, `scripts/serve_llama31_current_policy_pair.py`, `scripts/run_llama31_current_policy_pair_exact_batch.py`, `configs/llama31_current_policy_pair.json`, and `docs/LLAMA31_CURRENT_POLICY_PAIR.md` | Source-backed clean port of the main current-policy Llama PAIR path: eight streams, 30 rounds, source-exact append-after-input search/final placement, original-slot replacement for returned training prompts, current-adapter refresh, 32 examples per rank for the 74 weight-affecting groups, and unanimous 3-by-3 final judging. Provider clients and the archived asynchronous accumulation-group producer remain external; manuscript routing labels resolve drift in retained scripts. The later selective-rerun `attacker_interface_pair_llama.py` is excluded. |
| `test.py`, `run_tests.py`, `run_tests_sep.py`, `run_tests_qwen.py` | `src/secpo/evaluation/sep_static.py`, `scripts/evaluate_sep_static.py` | Attack transforms extracted into an explicit registry and orchestration rewritten |
| `sep_witness_gate.py` | `src/secpo/evaluation/witness.py` | Logic-preserving contract cleanup |
| `genetic_attack.py`, `pair_attack.py`, `tap_attack.py`, `attack_utils.py`, `minimal_apo.py`, adaptive test runners | `src/secpo/attacks/`, `src/secpo/evaluation/sep_adaptive.py`, `src/secpo/evaluation/sep_model_policy.py`, and the two model-family adaptive configs | Clean behavioral reimplementations with fingerprinted atomic resume. Genetic/MAP-Elites is retained. Both families use the working Llama-derived PAIR/TAP orchestration; Qwen adds its own role rendering, stop tokens, and reasoning stripping instead of the stale Qwen TAP fork. Private provider clients, pickle state, and scheduler coupling are removed. |
| Multi-stage judge functions in `utils.py` and `test.py`, plus operational SEP judge scripts | `src/secpo/evaluation/sep_judge.py`, `scripts/judge_sep_outputs.py`, and `configs/evaluate_sep_judges.json` | Provider-neutral witness-first 3-by-3 unanimous protocol with the three manuscript routing labels and three trials per label |
| `test_lm_eval.py`, `test_lm_eval_qwen.py`, Qwen utility runner lineage | `src/secpo/evaluation/lm_eval_utility.py` and model-specific configs | Strict fingerprinted command/config replacement; external harness and datasets stay external |
| AlpacaEval2/SEP reference-output evaluation runners and frozen comparator metadata | `src/secpo/evaluation/reference_utility.py`, `scripts/evaluate_reference_utility.py`, the two reference-utility configs, and `docs/REFERENCE_UTILITY.md` | Strict row/order/generator/hash preflight and command construction. The exact 805-row GPT-4o AlpacaEval2 and 9,160-row Llama-3-8B SEP comparator files remain external. |
| `test_agentdojo.py`, `agentdojo_adaptive_genetic_attack.py`, reviewed compatibility patch | Qwen AgentDojo module, CLI, config, documentation, and patch | Pinned external integration plus exact paper-80 adaptive expansion (Slack 20, Travel 12, Workspace 48), fresh environment per candidate, exact checker success, and atomic whole-run resume; upstream tree is not vendored |
| `test_agentdyn.py`, `agentdyn_qwen36_adapter.py`, reviewed patch/template | Qwen AgentDyn module, CLI, config, documentation, patch, and template | Pinned external integration preserving trusted `user` and untrusted `input` roles |
| `scripts/run_qwen36_benignref12_qwen_agent_input_dtap.py`, `scripts/run_qwen36_benignref12_qwen_agent_input_cpu_worker.py`, `scripts/qwen36_benignref12_qwen_agent_input_dtap_cpu_watchdog.py`, the final runtime Qwen-Agent adapter/task-runner files, and `manifests/dtap1838_nonvm_v1/tasks.jsonl` | `src/secpo/evaluation/dtap_qwen_agent_adapter.py`, `src/secpo/evaluation/dtap_qwen_agent_backend.py`, `src/secpo/evaluation/dtap_qwen36.py`, `scripts/evaluate_dtap_qwen36_27b.py`, `configs/evaluate_dtap_qwen36_27b.json`, `data/dtap_qwen36_non_vm_tasks.jsonl`, `third_party/patches/dtap.patch`, and `templates/dtap_qwen36_input_role.jinja` | Final trust/function loop, complete `eval.evaluation` lifecycle, exact 1,838-task manifest, provider-neutral external judge contract, and 33-by-6 CPU execution refactored into portable, checksum-pinned code. Cluster watchdogs, leases, service control, logs, outputs, and recovery state remain excluded. |
| `merge.py` | `src/secpo/model_merge_qwen36.py`, `scripts/merge_qwen36_27b_lora.py` | Durable Qwen LoRA merge refactor preserving multimodal PEFT key remapping and adding a path-sanitized merge manifest |
| Qwen offline preference JSON | `data/qwen36_offline_dataset/` | **Exact asset** split into deterministic hosting-safe chunks |
| Reviewed AgentDojo, AgentDyn, and DTap patches/templates | `third_party/patches/`, `templates/` | Checksum-pinned release assets; the DTap patch removes the research workspace's implicit model-service address and freezes thinking off |

## Explicit exclusions

- Private model-specific delimiter-derivation utilities are not copied. Only
  the reviewed frozen delimiter values required by Basic Adaptive evaluation
  remain.
- Instruction-hierarchy system/user, system/input, mutual-processing,
  analysis, manual-review, watchdog, data, and scheduler experiments are not
  part of the artifact. This exclusion is separate from the intentional
  trusted-`user`/untrusted-`input` boundary in AgentDyn and DTap.
- Baselines, ablations, thinking variants, private provider clients,
  scheduler wrappers, recovery utilities, generated results, logs, model
  weights, environments, caches, and out-of-scope attack implementations have
  no released path.
- GCG has no released implementation, configuration, import, or result path.

The deny-first rules in `release_manifest.json` enforce these boundaries even
if the future `maintainer_source_backup` profile is expanded.
