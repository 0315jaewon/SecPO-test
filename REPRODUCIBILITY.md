# Reproducibility notes

## Main settings and implementation notes

| Setting | Llama-3.1-8B-Instruct | Qwen3.6-27B |
| --- | --- | --- |
| Adversarial data | Online, current-policy attacks | Offline, saved strong attacks |
| Inference mode | Direct response | Non-thinking |
| Preference pairs | 19,157 | 19,157 archived; 19,145 after length filter |
| Epochs | 1 | 1 |
| DPO activation | Sigmoid | Sigmoid |
| Beta | 0.1 | 0.1 |
| Learning rate | 1.6e-4 | 1.6e-4 |
| Precision | bf16 | bf16 |
| LoRA rank / alpha / dropout | 64 / 8 / 0.1 | 64 / 8 / 0.1 |
| Per-device batch | 2 | 1 |
| Gradient accumulation | 16 | 8 |
| Training processes | 8 | 8 |
| Maximum sequence length | 2,048 | 4,608 |
| Warmup ratio | 0 | 0.03 |
| LR schedule | Cosine | Linear |
| Gradient clipping | Disabled | Norm 1.0 |
| Release seed | 42 (see note) | 42 |

The shared DeepSpeed configuration intentionally leaves scheduler construction
to Transformers so that each model JSON selects the schedule shown above.

The original Llama Torchtune configuration used a synchronized
runtime-generated seed and did not retain a fixed value. This release pins 42
to make reruns of the clean wrapper deterministic; it cannot recover the
original run's seed. The Qwen checkpoint metadata records seed 42.

The Llama run used eight additional GPUs for current-policy inference during
attack generation, for 16 H200 GPUs in total. The paper reports approximately
18 hours for this setting. The Qwen offline run used eight H200 GPUs and took
approximately seven hours.

The paper's shared training paragraph lists query/value attention projections
plus the MLP projections. The executable Qwen implementation and the retained
Qwen checkpoint metadata also include key/output attention projections. The
Qwen configuration in this release follows that executable implementation:
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and
`down_proj`. The Llama configuration uses `q_proj`, `v_proj`, and the three
MLP projections.

Qwen non-thinking training uses prompts rendered with an empty reasoning
block before the visible answer. Reasoning-on training variants are not part
of this release.

## Data construction

Start with Cleaned-Alpaca examples that have both a trusted instruction and an
untrusted data field. For each example:

1. Place another example's instruction before or after the untrusted data.
2. Render the trusted instruction with role `user` and the data with role
   `input`.
3. Use the undefended model's response to the clean trusted task as `chosen`.
4. Use a response to the injected instruction as `rejected`.
5. For the Llama main run, replace the weak injection with the best
   current-policy adaptive injection produced by the online worker.
6. For the Qwen main run, reuse the saved strong Llama-targeted injection and
   rerender the prompt with the Qwen chat template.

In the exact released Qwen archive, the optimized prompt and
`rejected_input_whole` come from that saved-attack replacement.
`chosen` and `rejected` were carried over by record index from the earlier
Qwen generation stage rather than regenerated after the replacement.
`rejected_input` retains the earlier seed and is not consumed by training.
These details are documented to prevent the archive from being
mischaracterized as freshly generated responses to every optimized prompt.

The trainer requires both the attacked and clean rendered prompt. Supplying
them explicitly is preferred because it avoids ambiguous text replacement.
For fail-closed preprocessing, this release requires the clean prompt to be
obtainable by one contiguous deletion and rejects multiple exact injection
matches. The experimental scripts used unrestricted string replacement, so
this validation rule is intentionally stricter.

The exact Qwen raw dataset is included as a deterministic split gzip archive.
It contains 19,157 records; the configured 512-token prompt filter removes 12
and trains on 19,145. See `data/DATASET_CARD.md` for checksums, known
auxiliary-field inconsistencies, and the required preparation commands.

## SecPO-only main results reported in the submission

All values are percentages.

### Llama-3.1-8B-Instruct, online SecPO

| Metric | SecPO |
| --- | ---: |
| MMLU-Pro | 46.3 |
| GPQA Diamond | 24.2 |
| GSM8K | 81.3 |
| MATH | 50.1 |
| AlpacaEval2 Utility | 30.8 |
| SEP Utility | 50.1 |
| SEP ASR | 1.4 |
| SEP Basic Adaptive ASR | 10.6 |
| SEP Genetic Adaptive ASR | 12.3 |
| SEP TAP Adaptive ASR | 12.2 |
| SEP PISmith ASR | 7.0 |
| AlpacaFarm GCG ASR | 5.8 |

### Qwen3.6-27B, offline non-thinking SecPO

| Metric | SecPO |
| --- | ---: |
| MMLU-Pro | 84.8 |
| GPQA Diamond | 81.8 |
| GSM8K | 95.8 |
| MATH | 95.9 |
| AlpacaEval2 Utility | 73.7 |
| SEP Utility | 84.6 |
| AgentDojo Utility | 88.7 |
| AgentDojo ASR | 0.0 |
| AgentDojo Genetic Adaptive ASR | 7.5 |
| SEP ASR | 1.4 |
| SEP Genetic Adaptive ASR | 5.8 |
| SEP TAP Adaptive ASR | 6.7 |
| SEP PISmith ASR | 0.0 |

Evaluation implementations and non-SecPO result columns are outside the
requested release scope.

## Reproduction boundary

The supplied code, data validator, and configurations implement the
benign-reference objective and main hyperparameters in a clean, shared Trainer
wrapper. The original Llama executable used Torchtune, so this is not a
bit-for-bit runtime reproduction. End-to-end reproduction additionally
requires the concurrent adaptive-attack worker and generated corpus for the
Llama experiment; those components and attacker credentials/checkpoints are
not included. The exact Qwen offline preference corpus is included, but model
weights and the original cluster/runtime environment remain external.
