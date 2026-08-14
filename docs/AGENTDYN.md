# Qwen3.6 AgentDyn integration

AgentDyn is installed externally and is never vendored into this repository.
The release pins upstream commit
`5353cf7615b135cace8d07c8f12dac53a16b6db3` (package `0.1.35`), its MIT
license, the reviewed five-file compatibility patch, the complete logical
patched-tree checksum, and the Qwen input-role chat template.

The adapter preserves the security boundary used by the experiment:

- trusted benchmark system text and the legitimate task become `user`;
- external tool results, including injected content, become `input`;
- assistant reasoning remains separate from visible/scored content;
- native tool calls are preferred, with a Qwen XML text fallback so valid
  calls are not discarded; and
- transient requests use at most three attempts with a finite exponential
  wait, while invalid requests are not retried.

No endpoint or credential is stored in the config. At runtime,
`AGENTDYN_BASE_URL` must identify the reviewed OpenAI-compatible `/v1`
service. Set `AGENTDYN_API_KEY` only when that service requires it. Never put
either resolved value in a config, command log, result, or commit.

## Setup and exact verification

```bash
git clone https://github.com/SaFo-Lab/AgentDyn.git /path/to/agentdyn
git -C /path/to/agentdyn checkout 5353cf7615b135cace8d07c8f12dac53a16b6db3
git -C /path/to/agentdyn apply \
  /path/to/secpo-release/third_party/patches/agentdyn.patch

python scripts/evaluate_agentdyn_qwen36_27b.py \
  --checkout /path/to/agentdyn \
  --patch third_party/patches/agentdyn.patch \
  --template templates/qwen36_agentdyn_input_role.jinja \
  verify

pip install /path/to/agentdyn
```

The verifier checks the commit and license, patch and template checksums,
tracked and untracked paths, and a complete logical tree hash independently
derived by applying the pinned patch to pristine blobs from the pinned
commit. An additional edit to any patched file therefore fails even when the
same five filenames remain modified. Verification also prints the required
endpoint-free vLLM argument fragment.

## Required Qwen server settings

From the release root, serve the evaluated base or merged checkpoint with the
reviewed template and native Qwen tool/reasoning parsers. The settings exposed in
`configs/evaluate_agentdyn_qwen36_27b.json` require this argument surface:

```bash
vllm serve /path/to/evaluated_qwen_checkpoint \
  --served-model-name Qwen/Qwen3.6-27B \
  --language-model-only \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --chat-template templates/qwen36_agentdyn_input_role.jinja \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}' \
  --generation-config vllm
```

Choose host, port, tensor parallelism, model length, and resource limits for
the local environment. Inject the resulting `/v1` address through
`AGENTDYN_BASE_URL`; the release deliberately provides no endpoint value or
implicit fallback. A server started without the reviewed chat template,
`qwen3_coder` tool-call parser, or `qwen3` reasoning parser is not the frozen
AgentDyn protocol.

## Evaluation

First inspect the generated commands; this read-only step does not require an
endpoint. `AGENTDYN_BASE_URL` becomes mandatory only with `--execute`:

```bash
python scripts/evaluate_agentdyn_qwen36_27b.py \
  --checkout /path/to/agentdyn \
  run \
  --log-dir /tmp/agentdyn-qwen36
```

The default config produces two upstream invocations in order:

1. a clean invocation with no `--attack` option; and
2. an `important_instructions` invocation with exactly one scalar
   `--attack` option.

Add `--execute` only after reviewing both commands and the required vLLM
arguments printed beside them. The benchmark version is `v1.2.2`, suites are
shopping/GitHub/daily-life, and the defense is repeat-user-prompt. Each server
response checks direct and extra `reasoning`/`reasoning_content` fields,
separates `<think>` content before scoring, and falls back to parsing Qwen XML
tool calls only when native calls are absent.

The exact model revision, sampling values, retry bounds, role contract,
server parsers, patch, patched tree, and template are pinned in
`configs/evaluate_agentdyn_qwen36_27b.json`. The release does not include the
original vLLM lifecycle manager, CUDA setup, watchdogs, Slurm wrappers, logs,
results, or provider routing.
