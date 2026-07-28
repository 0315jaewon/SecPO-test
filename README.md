# SecPO: anonymous training-code supplement

This folder is a minimal, standalone clean reimplementation of the
benign-reference Security Preference Optimization (SecPO) training objective
from the anonymous submission. It contains only the defense-side code and
configurations for the two main training settings:

- Llama-3.1-8B-Instruct with online adversarial examples supplied by the
  training-time attack worker.
- Qwen3.6-27B in the offline, non-thinking setting using saved strong
  adversarial examples. The exact Qwen offline preference dataset is included
  as a verified split archive.

Comparison methods, component studies, evaluation harnesses, checkpoints,
the Llama online attack corpus and worker, cluster launch files, credentials,
and experiment logs are intentionally not included.

## SecPO objective

For a preferred response `chosen`, a dispreferred response `rejected`, an
attacked prompt `x_adv`, and the corresponding injection-free prompt
`x_clean`, the implementation minimizes

```text
-log sigmoid(beta * (
    [log pi(chosen | x_adv)   - log pi(rejected | x_adv)]
  - [log ref(chosen | x_clean) - log ref(rejected | x_clean)]
))
```

`pi` is the trainable LoRA policy. `ref` is the same base model with its LoRA
adapter disabled. The important difference from ordinary DPO is that the
policy is evaluated on the attacked prompt while the reference is evaluated
on the clean prompt.

## Contents

```text
configs/                  Main experiment and distributed-training configs
data/                     Qwen dataset archive, card, and synthetic fixture
scripts/                  Data verification, preparation, launch, and audits
src/secpo/                Standalone SecPO loss, collator, and trainer
tests/                    Unit tests that do not load model weights
REPRODUCIBILITY.md        Exact released settings and reported main results
RELEASE_CHECKLIST.md      Anonymous-release gate
ANONYMOUS_4OPEN_SCIENCE.md  Service-specific publication procedure
```

## Environment

Python 3.13 and the package versions in `requirements.txt` match the clean
training environment used by the Qwen path.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Access to the gated Llama weights and to any model-specific Qwen weights is
the reviewer's responsibility. Keep `model_name_or_path` as a public model ID;
do not replace it with a private cache path, because PEFT may serialize that
path into checkpoint metadata.

## Preference data

The exact Qwen offline raw dataset is bundled as five lossless archive parts
under `data/qwen36_offline_dataset/`. It has 19,157 records and reconstructs
to SHA-256
`cd349baa406eae0bf7e4ee2d938d087d6656469d88076d68ba76baeb7e5ec669`.
Read `data/DATASET_CARD.md` before publishing or using it: redistribution
licensing and review of synthetic PII-like text are explicit release gates.

Reconstruct, audit, and prepare the smaller trainer schema:

```bash
python scripts/reconstruct_dataset.py \
  --manifest data/qwen36_offline_dataset/manifest.json \
  --output /tmp/qwen36_offline_preferences.raw.json
python scripts/audit_release_dataset.py \
  /tmp/qwen36_offline_preferences.raw.json \
  --schema raw \
  --expected-records 19157
python scripts/prepare_preferences.py \
  --input /tmp/qwen36_offline_preferences.raw.json \
  --output /tmp/qwen36_offline_preferences.jsonl.gz \
  --expected-records 19157
```

The prepared input records contain:

```json
{
  "id": "stable-example-id",
  "prompt": "fully rendered attacked chat prompt",
  "clean_prompt": "the same rendered prompt with only the injection removed",
  "chosen": "response to the trusted user instruction",
  "rejected": "response that follows the injected instruction"
}
```

For compatibility with the internal generator, `clean_prompt` may be omitted
when the record instead contains `rejected_input_whole`. The preparation
script then removes exactly one occurrence of that string and fails closed if
the match is missing or ambiguous.

See `data/README.md`, `data/DATASET_CARD.md`, and
`data/example_preferences.jsonl`.

To convert another preference JSON array or JSONL file:

```bash
python scripts/prepare_preferences.py \
  --input /path/to/source_preferences.json \
  --output data/main_preferences.jsonl
```

The prompts must already use the model's chat template, including the
additional `input` role for untrusted data.

## Main training commands

Run commands from this folder.

Llama defense-side training:

```bash
DATA_FILE=/path/to/llama31_online_preferences.jsonl \
  bash scripts/train_llama31_online.sh
```

Qwen offline training:

```bash
DATA_FILE=/path/to/qwen36_offline_preferences.jsonl \
  bash scripts/train_qwen36_offline.sh
```

For the released Qwen corpus, set `DATA_FILE` to the prepared
`/tmp/qwen36_offline_preferences.jsonl.gz` path above. The trainer accepts
plain JSON/JSONL and deterministic `.json.gz`/`.jsonl.gz` inputs.

Both launchers use eight training processes and DeepSpeed ZeRO-3. The Llama
main experiment additionally used eight GPUs in a separate online attack
worker. That private orchestration stack is deliberately outside this small
defense-code release; its output contract is the preference schema above.
The original Llama run used a custom Torchtune online recipe; this compact
release expresses the same objective and listed hyperparameters through
Hugging Face Trainer, Accelerate, and DeepSpeed. It is not a bit-for-bit copy
of that runtime and does not independently regenerate the adaptive injections.

## Model-weight release commitment

No model weights are included in this anonymous review artifact. After the
anonymous review process concludes, the authors commit to publicly releasing
the trained SecPO checkpoints for both main experiments:

- Llama-3.1-8B-Instruct with online SecPO training.
- Qwen3.6-27B with offline non-thinking SecPO training.

The post-review release will include the trained weights, exact base-model
revisions, integrity checksums, loading instructions, and applicable license
and attribution notices.

## Tests and release audit

```bash
python -m pytest -q
python -m compileall -q src scripts tests
bash -n scripts/*.sh
bash scripts/audit_anonymity.sh
```

No telemetry integration is enabled. Training logs and checkpoints remain
local under the configured output directory and must be reviewed separately
before sharing.

## Anonymous-review status

This is a review artifact by Anonymous Authors. Do not publish it from the
parent repository or reuse the parent Git history. Follow
`ANONYMOUS_4OPEN_SCIENCE.md` and complete `RELEASE_CHECKLIST.md` from a fresh,
dedicated repository.

No project or dataset license is asserted in this review copy. Resolve the
separate code and data license gates before uploading even an anonymous public
review repository.
