# Qwen3.8-27B GPQA Diamond on Lambda / vLLM

This branch publishes the code used for the September 14, 2026 Lambda run:
**159/198 correct (80.3030%)**, with one response reaching the 16,384-token cap.
It evaluates the **base model**, not a SecPO adapter. It does not include other
benchmarks or Tinker training integrations.

## What is original, and what was added for sharing?

- `scripts/lambda_gpqa.py` is byte-for-byte the runner used for the recorded
  evaluation. Its SHA-256 is
  `e9c1fbfdc6162b26cc7de9a51a295806df33c8c4e7fc72e6f876c83de34be3d2`.
  `prepare` validates and freezes the original prompts; `generate` runs vLLM;
  `score` decodes outputs and uses the original lm-eval extraction/scoring code.
- `scripts/export_lambda_gpqa_bundle.py` is a new, standard-library-only
  packaging helper. It exports the private files the original runner needs,
  without changing prompts, token IDs, generation, or scoring.
- `requirements/lambda-gpqa-*.txt` pin the core generation/scoring packages.
- `tests/test_lambda_gpqa*.py` check protocol integrity, extraction/scoring,
  and safe packaging. They do not launch GPU inference.

The old SSH collector is intentionally not included: it was tied to the
original host and SSH identity. It only transferred files and invoked `score`;
it was not part of model generation or answer grading.

## Exact evaluation protocol

| Setting | Value |
| --- | --- |
| Model | `Qwen/Qwen3.8-27B`, no adapter |
| Model/tokenizer revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Task | `gpqa_diamond_cot_zeroshot`, all 198 questions |
| Prompting | Zero-shot CoT, thinking off, exact saved rendered prompts |
| Sampling | Temperature 0.0, top-p 1.0, requested top-k -1, seed 42 |
| Output cap | 16,384 tokens |
| Total context limit | 32,768 tokens; longest prompt is 2,774 tokens |
| Stops | `</s>`, `<\|im_end\|>`, `<\|endoftext\|>` (without backslashes) |
| Precision / hardware | BF16, tensor parallelism 1, one H100 80GB HBM3 |
| Generation environment | Python 3.10.12, vLLM 0.29.0, Transformers 5.10.4, Torch 2.13.0 (CUDA 13.0 build), tokenizers 0.22.2 |
| Engine | Batch size 4, max batched tokens 4096, GPU memory utilization 0.90, eager execution, no prefix caching |
| Scoring environment | Python 3.11, lm-eval 0.4.11, Transformers 4.57.1 |
| Reported metric | `exact_match,flexible-extract`, not `strict-match` |

The two initial smoke questions are retained in the total of 198. vLLM is fed
the frozen input token IDs directly, after checking tokenizer agreement for
every prompt. It does not reapply the chat template or use a reasoning parser.
`generation_config="vllm"` prevents Hugging Face generation defaults from
overriding the explicit settings. With temperature zero, vLLM 0.29.0 normalizes
top-k to 0 for greedy decoding; both 0 and -1 disable top-k filtering.

The primary extraction filter is the original lm-eval `multi_choice_regex`
with `regex_pattern=r"(\([A-Z]\))"`, `group_select=-1`, case/punctuation ignored,
then `take_first`. The actual filter specifications are read from the frozen
reference configuration. Scoring first checks that both original filters
reproduce the saved Tinker per-question extractions and correctness exactly.

Matching this protocol does not guarantee identical fresh model outputs on
different hardware/software. The original Tinker manifest pins the tokenizer
revision but does not independently verify the base-weight revision or dtype;
this is a protocol comparison, not proof of bit-identical backends.

## Required private data bundle

This is a **code-only branch**. Before running, obtain
`lambda_gpqa_reproduction_inputs.tar.gz` privately from the run owner.
The owner can create it with the exporter below. It contains:

```text
lambda_gpqa/
  inputs.json
  reference/
    manifest.json
    results.json
    worker.json
  tokenizer/
    chat_template.jinja
    merges.txt
    tokenizer.json
    tokenizer_config.json
    vocab.json
  SHA256SUMS
```

The bundle includes GPQA questions, gold answers, and the original Tinker
responses used to validate scoring. Share only with authorized collaborators
under the dataset's terms. Do not commit it. It contains no API/SSH credentials,
model weights, or Lambda-generated answers. Those answers are not needed to
perform a fresh evaluation. Only `inputs.json` is read for generation.

Do not substitute the Lambda `results.json` for `reference/results.json`:
they have different schemas. The old 6.4 MB results-backup archive alone is
not this reproduction-input bundle; it lacks the separate original reference
files and tokenizer files required by the scorer.

## Clone and unpack

```bash
git clone --branch lambda-test --single-branch https://github.com/0315jaewon/SecPO-test.git
cd SecPO-test
mkdir -p data
tar -xzf /path/to/lambda_gpqa_reproduction_inputs.tar.gz -C data
(cd data/lambda_gpqa && sha256sum --check SHA256SUMS)
```

All following commands run from this checkout's root. Use a GPU machine with
sufficient VRAM and compatible CUDA drivers (original: driver 580.105.08,
Ubuntu 22.04.5). Results and model caches are stored locally under `outputs/`
and `runtime/`; put the checkout on persistent storage or copy the results off
the instance before terminating it.

## 1. Install the generation environment

Use Python 3.10 and an isolated environment. Do **not** install the repository's
general requirements or `pip install -e .` for this standalone evaluation.

```bash
python3.10 -m venv runtime/lambda_gpqa/generate
runtime/lambda_gpqa/generate/bin/python -m pip install --upgrade pip
runtime/lambda_gpqa/generate/bin/python -m pip install -r requirements/lambda-gpqa-generate.txt
source runtime/lambda_gpqa/generate/bin/activate
```

Activation is important: the original startup failed because the environment's
`ninja` executable was not on PATH. These pins capture the core packages, not
a complete transitive lockfile; the original full inventory is in the private
results backup's `remote/environment.json`.

## 2. Generate all 198 responses

Run inside `tmux` if you need the process to survive SSH disconnection.
This is the original generation invocation with portable paths. Leave hub
access enabled for the first run so the pinned model can download. The original
worker used `HF_HUB_OFFLINE=1` only after the weights were already cached.

```bash
env \
  HF_HOME="$PWD/runtime/lambda_gpqa/hf_cache" \
  HF_HUB_OFFLINE=0 HF_HUB_DISABLE_TELEMETRY=1 \
  VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
  TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn \
  python -u scripts/lambda_gpqa.py generate \
  --inputs data/lambda_gpqa/inputs.json \
  --output outputs/lambda_gpqa_rerun \
  --batch-size 4
```

Choose a **fresh output directory** for an independent run. Reusing a matching
directory resumes cached completions; it does not regenerate them. Changed
source or environment identity in an existing directory is rejected.

Monitor `outputs/lambda_gpqa_rerun/progress.json`. After `generation_complete`,
the directory contains `generations.json`, all 198 individual cache records,
`manifest.json`, `environment.json`, and progress/smoke metadata. Generation
completion is not yet a scored result. The original run took about 4 h 53 min;
this is an observation, not an estimate for other hardware.

## 3. Install the separate scoring environment

Scoring can run on a CPU machine. It deliberately uses older Transformers than
the GPU environment, matching the original Tinker tokenizer/scoring stack.
Use a separate Python 3.11 environment; do not downgrade the vLLM environment.

```bash
python3.11 -m venv runtime/lambda_gpqa/score
runtime/lambda_gpqa/score/bin/python -m pip install --upgrade pip
runtime/lambda_gpqa/score/bin/python -m pip install -r requirements/lambda-gpqa-score.txt
```

If scoring elsewhere, copy the generated run directory and the private data
bundle to that machine. No Tinker or judge API key is used by either phase.

## 4. Validate scoring and score the new responses

First replay the historical Tinker scoring. This must report 159/198 correct:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
runtime/lambda_gpqa/score/bin/python scripts/lambda_gpqa.py score \
  --reference data/lambda_gpqa/reference \
  --inputs data/lambda_gpqa/inputs.json \
  --tokenizer data/lambda_gpqa/tokenizer \
  --self-test
```

Then score the fresh vLLM generations:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
runtime/lambda_gpqa/score/bin/python scripts/lambda_gpqa.py score \
  --reference data/lambda_gpqa/reference \
  --inputs data/lambda_gpqa/inputs.json \
  --tokenizer data/lambda_gpqa/tokenizer \
  --generations outputs/lambda_gpqa_rerun \
  --output outputs/lambda_gpqa_rerun_scored
```

Read `outputs/lambda_gpqa_rerun_scored/results.json`. The main score is
`results["exact_match,flexible-extract"]`. The file also contains each question,
target, prompt, new response, extracted answer, correctness, output token count,
finish reason, and corresponding Tinker response/correctness.

## Run-owner command: export the private inputs

On the machine holding the original artifacts, use Python 3.10+:

```bash
python3 scripts/export_lambda_gpqa_bundle.py \
  --inputs /path/to/original/lambda/inputs.json \
  --reference /path/to/original/tinker/gpqa_diamond \
  --tokenizer /path/to/pinned/tokenizer/snapshot \
  --output outputs/lambda_gpqa_reproduction_inputs.tar.gz
```

The exporter validates the frozen prompt identity, original reference hashes,
and every tokenizer file. It copies only an explicit allowlist, dereferences
tokenizer symlinks into real files, writes checksums, and refuses to overwrite
an existing archive. The original `prepare` command is unnecessary if the
already-frozen `inputs.json` is available; see `lambda_gpqa.py prepare --help`
if preparing from the original full Tinker run and its completion cache.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 HF_HUB_OFFLINE=1 \
runtime/lambda_gpqa/score/bin/python -m pytest \
  tests/test_lambda_gpqa.py tests/test_lambda_gpqa_bundle.py -q
```

For inspection only, the code on this branch and the private results backup
are sufficient; no GPU, package installation, or rerun is needed.
