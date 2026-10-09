# SecPO on Tinker

This backend implements the benign-reference SecPO recipe with Tinker's
custom-loss API. It is separate from the existing Transformers/PEFT trainer;
both backends call the same tested SecPO equation.

## What is model-agnostic

For each preference pair, the runner sends two attacked-prompt sequences to
the trainable policy and two clean-prompt sequences to a frozen snapshot of
the initial policy:

```text
policy:    attacked prompt + chosen/rejected response
reference: clean prompt    + chosen/rejected response
```

Only response-token log-probabilities are summed. Tinker's
`forward_backward_custom` differentiates the SecPO loss through the policy
log-probabilities; reference scores are detached values from a sampling
client. One `optim_step` then applies AdamW to the LoRA parameters.

Thinking-off is not an algorithmic requirement. The mentor-provided Qwen3.8
dataset is explicitly rendered in thinking-off mode, so its strict adapter
rejects thinking tags and template drift. A thinking-on experiment is valid
when attacked and clean prompts, chosen and rejected responses, and evaluation
are rendered consistently for that mode.

## Install and preflight Qwen3.8-27B

Install the optional backend and log in to Hugging Face without putting a
credential in the repository:

```bash
pip install -e '.[tinker]'
huggingface-cli login
```

The checked-in Qwen configuration pins dataset commit
`7fd85bf8b670370ad396a049ad11f0dad36364af` and tokenizer commit
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Run the no-cost preflight first:

```bash
secpo-train-tinker \
  --config configs/train_qwen38_27b_tinker.json \
  --preflight-only
```

Preflight authenticates to Hugging Face, checks the exact 18,951-row schema,
tokenizes all four paths for every row, enforces the 512/6,144 token limits,
and constructs the deterministic 297-step plan. It does not import Tinker,
create a remote model, write a run directory, or make a billed training call.
The validated snapshot currently has maximum attacked-prompt, clean-prompt,
and full-sequence lengths of 510, 441, and 6,001 tokens respectively.

## Train

Set the Tinker credential outside the repository, choose a new `output_dir` in
the JSON for each run, and start with a one-step integration run:

```bash
export TINKER_API_KEY=your_runtime_value
secpo-train-tinker \
  --config configs/train_qwen38_27b_tinker.json \
  --max-steps 1
```

Because the configured 3% warmup rounds up to one step in this capped run,
that sole optimizer call uses learning rate zero: it validates the paid remote
API path but intentionally does not test learning. For a nonzero-update smoke
test, copy the config, set `warmup_ratio` to `0`, and keep its output directory
separate from the full run.

After reviewing that run, use the same command without `--max-steps` and a
fresh output directory for all 297 updates. Remote training and reference
scoring are billed Tinker operations. Consult the current
[Tinker model and pricing table](https://tinker-docs.thinkingmachines.ai/tinker/models/)
before starting.

The Qwen configuration uses beta 0.1, LoRA rank 64, one epoch, effective batch
size 64, 3% warmup with linear decay, AdamW betas 0.9/0.999, zero weight decay,
gradient norm clipping at 1.0, and peak learning rate `8e-5`. The smaller
learning rate reflects the observed utility regression at `1.2e-4`.
Tinker exposes LoRA rank and trainable layer groups but not the Transformers
PEFT alpha/dropout knobs, so those backend-specific settings are not claimed
to be identical.

Every update is appended to `metrics.jsonl`. The Qwen configuration and generic
example now use `save_every: 0`, which disables intermediate checkpoints. The
completed 297-step Qwen run used `save_every: 20` and saved 14 intermediate
training states; its archived config remains unchanged for provenance.
Set a positive `save_every` to restore periodic training/optimizer-state saves.
Without intermediate saves there is no saved mid-run recovery point if a run
fails before its final export.

The final training step is still exported in two forms: `secpo-final-state`
contains training and optimizer state, while `secpo-final-sampler` is used for
inference and evaluation. Both paths are written to `result.json`; these are
two artifacts for the same final model, not two training checkpoints at different
steps. The initial frozen reference snapshot required by SecPO is also unchanged.
The runner refuses to overwrite an existing run.

Retention is separate from save frequency. The Qwen configuration and generic
example now use `checkpoint_ttl_seconds: null`, so their final exports have no
scheduled expiration. The runner passes this field to both training-state and
sampler saves. If periodic saves are re-enabled, they inherit the same retention.
The short-lived smoke configuration is separate and remains unchanged.

The original Qwen run used `604800` (seven days) for both intermediate and final
exports. Changing a local config does not alter existing remote checkpoints;
remove their expiration with Tinker's
[`set_checkpoint_ttl_from_tinker_path(path, ttl_seconds=None)`](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/restclient/#tinker.RestClient.set_checkpoint_ttl_from_tinker_path).

On September 9, 2026, the user requested retention of the final SecPO model and
deletion of earlier checkpoints. The final sampler used by every SecPO evaluation
and its matching final training state were both verified remotely with
`expires_at: null`. All 14 intermediate states (steps 20 through 280, every 20
steps) were then deleted, and a fresh listing verified that only the two final
artifacts remain in training run
`fcfc57ba-d2ab-568d-8f90-b3f60ab70560:train:0`. No local weight backups were made;
other runs, training logs, and evaluation results were untouched. Before/after
inventories are saved in `outputs/qwen38_27b_tinker/checkpoint_cleanup_20260909.before.json`
and `checkpoint_cleanup_20260909.after.json`. Historical metrics still contain
the intermediate paths for provenance, but those remote states are now deleted.

## Use SecPO with another model

Copy `configs/train_generic_tinker.example.json`, select any current
Tinker-supported `model_name`, and point it at a Hugging Face dataset with the
`secpo_pre_rendered` four-column schema:

```json
{
  "prompt": "attacked chat through the assistant-generation prefix",
  "clean_prompt": "matching clean chat through the assistant-generation prefix",
  "chosen": "preferred response including the model terminator",
  "rejected": "dispreferred response including the model terminator"
}
```

These strings must already use the selected model's tokenizer and chat
template. The code intentionally does not guess or apply a template a second
time. Consequently, changing only `model_name` is not sufficient: the dataset
must also be rendered for that model and thinking mode. Pin
`dataset_revision` and `tokenizer_revision` to 40-character commits after the
exploratory phase and set `expected_records` to the intended exact split size.

The `qwen38_pre_rendered` format is a stricter dataset-specific adapter. It
validates all eight source columns, confirms the thinking-off suffix and
response terminators, and derives `clean_prompt` by removing exactly one
occurrence of `rejected_input_whole` from the attacked prompt.

## Implementation map

| Module | Responsibility |
| --- | --- |
| `tinker/data.py` | Dataset contracts, direct tokenization, causal shift, response masks |
| `tinker/bridge.py` | Conversion to Tinker `Datum` and `ModelInput` |
| `tinker/loss.py` | Response log-probability reduction and shared SecPO objective |
| `tinker/runtime.py` | Frozen reference scoring, custom backward, LR schedule, optimizer step |
| `tinker/runner.py` | Full preflight, deterministic batches, metrics, checkpoints, final export |
| `tinker/cli.py` | Strict command-line entry point |

SecOPD is a separate method and is not used by this implementation.
