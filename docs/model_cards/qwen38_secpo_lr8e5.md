---
library_name: transformers
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: finetune
pipeline_tag: image-text-to-text
datasets:
  - Sizhe-Chen/Qwen3.8-27B-Instruct-SecPO-trainset
tags:
  - secpo
  - prompt-injection-defense
  - tinker
  - merged
---

# Qwen3.8-27B-SecPO-lr8e-5

This is the **full, merged BF16 checkpoint** from our original learning-rate
`8e-5` SecPO training run on Tinker. It is not an adapter-only repository.
Neither Tinker access nor a separate LoRA adapter is needed for inference.

The base is `Qwen/Qwen3.8-27B`, pinned to revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
The saved architecture is `Qwen3_5ForConditionalGeneration` with model type
`qwen3_5`; these are the original base-model architecture identifiers.

## Training

| Setting | Value |
| --- | --- |
| Training method | SecPO, implemented on Tinker |
| Dataset | Sizhe-Chen/Qwen3.8-27B-Instruct-SecPO-trainset |
| Dataset revision | 7fd85bf8b670370ad396a049ad11f0dad36364af |
| Preference pairs | 18,951 |
| Epochs / optimizer steps | 1 / 297 |
| Learning rate | 8e-5 |
| Schedule / warmup fraction | Linear / 0.03 |
| Logical batch | 64 preference pairs; final partial batch retained |
| SecPO beta | 0.1 |
| LoRA rank | 64 |
| Trainable groups | Attention and MLP; not unembedding |
| Adam betas / epsilon | 0.9, 0.999 / 1e-8 |
| Weight decay / gradient clipping norm | 0 / 1 |
| Training maximum sequence / prompt length | 6,144 / 512 tokens |
| Seed | 42 |

Training used the supplied pre-rendered, thinking-off preference data. SecPO
scores the policy on attacked prompts and its frozen reference on corresponding
clean prompts, using response-token log probabilities.

The exported adapter records LoRA alpha 32 (scaling 32/64 = 0.5) and dropout 0.
These were not independently configurable in this Tinker run. This checkpoint
is therefore not claimed to exactly reproduce the original local PEFT recipe's
LoRA settings.

The adapter was merged using Tinker Cookbook commit
`dfe4d77e8e8c7df7e9cd9539220b54a01d1eaa40`.
The export's adapter SHA-256 is
`a15e0c5fe8cb54a45cce05d98c4fc7fa8863e42dd466685dae6ad431a3bd0a3f`.
`merge_validation.json` records the checks performed during the original merge.

## Inference

Load this repository as a complete Hugging Face model using software supporting
the base model's Qwen3.5-family architecture. The saved tokenizer and chat
template are included unchanged. Our non-thinking evaluations explicitly pass
`enable_thinking=False` to the chat template; downloading this checkpoint alone
does **not** disable thinking.

Our subsequent non-thinking evaluation settings were:

```json
{
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "min_p": 0.0,
  "presence_penalty": 1.5,
  "repetition_penalty": 1.0,
  "frequency_penalty": 0.0
}
```

These are documented evaluation settings, not new generation defaults baked
into the checkpoint. Output caps and evaluation harnesses vary by benchmark;
training sequence length is not an inference output cap. Earlier experiments
also used greedy decoding, so compare results only with matched settings.

The model weights total approximately 55.6 GB across 18 safetensors shards.
Inference requires additional memory for activations and the KV cache.

## Scope and limitations

This is a research prompt-injection-defense fine-tune. It does not guarantee
resistance to arbitrary prompt injection, and utility/security trade-offs must
be evaluated for the intended application. The base model's vision components
and processing assets are preserved, but this release does not claim validated
multimodal performance after fine-tuning. No benchmark scores are asserted here
without their corresponding harness, generation, and judging configuration.

`training_summary.json` provides the run settings and provenance;
`SHA256SUMS` identifies the uploaded artifacts. No training examples, evaluation
datasets, credentials, or optimizer state are included.

The original Alibaba Cloud Apache-2.0 license is included in `LICENSE`.
