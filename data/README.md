# Preference data

This directory contains two distinct artifacts:

- `qwen36_offline_dataset/`: five binary parts plus a manifest that
  reconstruct the exact 19,157-record Qwen offline main-experiment dataset.
- `example_preferences.jsonl`: a tiny synthetic schema fixture used only by
  tests.

Read `DATASET_CARD.md` before sharing or processing the experimental data.
It records checksums, provenance, known artifacts, content-review findings,
and unresolved licensing requirements.

## Trainer contract

The trainer reads JSON arrays, JSON Lines, and their `.json.gz` or
`.jsonl.gz` equivalents. Each prepared record must provide the four strings
below:

- `prompt`: fully rendered attacked prompt ending immediately before the
  assistant response.
- `clean_prompt`: the same prompt with the adversarial injection removed and
  all other bytes unchanged.
- `chosen`: response to the trusted instruction.
- `rejected`: response to the injected instruction.

An optional string `id` is retained for provenance checks.

Raw generator output can use `rejected_input_whole` instead of
`clean_prompt`. `scripts/prepare_preferences.py` removes exactly one occurrence
from `prompt`. It rejects zero or multiple matches, empty injections, empty
responses, or an explicitly supplied clean prompt that is identical to the
attacked prompt.

Prompts are expected to be pre-rendered. The trainer does not invent or modify
chat roles. Main-experiment prompts use `user` for the trusted instruction and
the model-specific `input` role for untrusted data.

## Exact Qwen data workflow

```bash
python scripts/reconstruct_dataset.py \
  --manifest data/qwen36_offline_dataset/manifest.json \
  --output /tmp/qwen36_offline_preferences.raw.json
python scripts/prepare_preferences.py \
  --input /tmp/qwen36_offline_preferences.raw.json \
  --output /tmp/qwen36_offline_preferences.jsonl.gz \
  --expected-records 19157
DATA_FILE=/tmp/qwen36_offline_preferences.jsonl.gz \
  bash scripts/train_qwen36_offline.sh
```

Preparation retains the exact attacked prompt and responses but drops
auxiliary generator-only fields. It derives `clean_prompt` by one literal
deletion, matching the benign-reference training path.
