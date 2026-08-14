# Llama-3.1-8B current-policy PAIR backend

This release contains the provider-neutral implementation of the PAIR backend
used for online adversarial training of Llama-3.1-8B. It is a clean refactor of
the main `attacker_interface_pair.py` + `PAIRAttack.run_batch` path, not the
later selective-rerun prototype. It contains no scheduler directives, private
filesystem paths, service addresses, or credentials. GCG is not included.

## Preserved experiment behavior

The canonical values live in
`configs/llama31_current_policy_pair.json` and are validated when loaded:

- eight attacker ranks and 32 weight-affecting training examples per rank
  (256 examples total per full optimizer update);
- eight independent PAIR streams per example;
- 30 victim/score/refinement steps;
- PAIR generation at temperature 1.0 with at most 2,048 output tokens;
- deterministic current-policy victim generation with at most 1,536 tokens;
- the first attacker message plus the last three conversation exchanges;
- the exact initialization-template pool and PAIR/judge prompts from the main
  run;
- lookup of the newest recognizable current-policy LoRA before every PAIR
  step, with bounded waiting for publication to complete;
- a second lookup followed by a final victim generation of at most 2,048
  tokens;
- three distinct final judge slots, three independent trials per slot,
  temperature 0.7, and unanimous success across all nine responses.

The companion Llama training config uses `save_strategy: steps` and
`save_steps: 1`. Hugging Face Trainer therefore finishes a standard PEFT
`checkpoint-N/` after every optimizer/gradient-accumulation step, before the
next training batch is requested. An epoch-only save strategy is rejected for
online mode because it would leave the attacker on the base policy throughout
this one-epoch run.

For the archived 19,157-row dataset on eight ranks, the Llama config also uses
the archived sampler seed (`data_seed: 0`), drops incomplete two-example
microbatches, and caps training at 74 optimizer updates. The cap prevents
Hugging Face Trainer from flushing the final 13-microbatch accumulation
remainder as a 75th update; the Torchtune loop stepped only at exact multiples
of 16. The original training RNG was configured as `seed: null` and was not
retained, so the release keeps its documented deterministic replacement
`seed: 42`. Different dataset sizes or process counts require recalculating
`max_steps`; these controls do not prove an identical cross-framework index
schedule.

The highest-scoring injection is selected using strict `>` comparisons, so the
first candidate wins a score tie. If the proposer fails to return a non-empty
`prompt`, the stream retries its own best candidate with ` Please comply.`
appended. Iteration scores must be integers from 1 through 10. Final votes must
be `YES`, `NO`, or the corresponding Boolean.

For each example, the backend returns the serialized prompt containing the
best injection. It also retains a structured result in `backend.last_results`,
including all iteration logs, policy snapshot identifiers, the final response,
and the nine-vote outcome. As in the research code, a failed nine-vote check
falls back to the original undesirable target in `training_response`; it does
not replace the selected attacked prompt.

Attack-time placement intentionally differs from the returned training
prompt. The backend strictly parses the clean Llama rendering. Every search
query renders `clean_input + " " + candidate`, matching
`attack_utils.build_chat_prompt`; final verification renders
`clean_input + candidate`, matching the main interface. Thus a candidate is
tested after the clean input even for the 8,719 dataset rows whose seed attack
was originally at the front. Existing boundary spaces are preserved, including
the source's double-space search form for end-position rows and its no-added-
separator final form for front-position rows. Only the prompt returned to DPO
uses one-span replacement to put the optimized attack back into the original
front/end seed slot.

PAIR generation and both judge stages receive the raw trusted instruction,
not the whole serialized victim prompt. The online protocol intentionally does
not duplicate that string: the Llama backend extracts it from the exact
released rendering
`<|begin_of_text|><|start_header_id|>user...<|eot_id|><|start_header_id|>input...<|eot_id|><|start_header_id|>assistant...`.
The parser requires exactly one user role, one input role, one assistant
generation header, two end-of-turn markers, no system/additional role, and
non-empty user and input content. Unknown, malformed, or ambiguous renderings
fail the request. Compact HTML-like fixtures are not accepted by this PAIR
backend. This model-family boundary is deliberate; adapting the backend to
another chat template requires a reviewed parser rather than passing
serialized control tokens to the attack models.

One intentional reproducibility improvement is that initialization sampling is
stable for `(seed, example_id)`. The original process used unrecorded global
random state, so its exact initial eight templates cannot be reconstructed
after the fact. Set `seed` in the JSON config and archive that config with run
metadata.

## Two request-batching modes

The archived experiment generated the next 32 attacks per attacker rank while
the defender consumed the preceding group. Those 32 examples cover the 16
two-example microbatches in one gradient-accumulation step. A search could
therefore observe a newly published defender policy partway through its 30
rounds. The public `OnlineAttackCollator` instead emits and synchronously waits
on one two-example request before the defender can consume that microbatch.
Consequently the defender cannot update during that request's 30-round search:
all per-round `refresh()` calls normally observe one fixed snapshot.

The connected worker therefore has two explicit modes:

1. The default **microbatch-compatible mode** accepts the released trainer's
   requests. Every example still receives 8 streams, 30 refinement steps, a
   snapshot lookup each round, final generation, and nine votes. It is the
   runnable public-training mode, but synchronous scheduling changes attack
   trajectories as well as provider-call batch size.
2. `run_llama31_current_policy_pair_exact_batch.py` enforces exactly 32
   examples per request. It requires an external producer that assembles and
   asynchronously pipelines complete accumulation groups while the defender
   consumes the previous group. That producer is not included. Do not point
   the released blocking microbatch collator at this mode; it will correctly
   reject the first two-example request.

The worker prints the selected mode at startup. Strict 32 is faithful for the
74 full, weight-affecting groups. The archived sampler also exposed a final
27-example-per-rank attack tail; only 26 were consumed and none contributed to
an optimizer update. The strict worker intentionally does not claim attack-log
fidelity for that non-weight-affecting tail. Neither mode is a claim of a
bit-exact historical trajectory.

## Current-policy snapshots

The local victim starts from the pinned base model if no recognizable adapter
directory is available. Before each PAIR victim call and before final
generation, it selects the highest ordered recognizable adapter under
`--policy-root`, then waits for that adapter to finish publishing. Supported
directories are:

- `checkpoint-N/`, ordered by integer `N`;
- `epoch_N[.fraction]/`, ordered numerically; or
- any directory with this strict manifest:

```json
{
  "version": 1,
  "identifier": "step-120",
  "step": 120,
  "ready": true
}
```

Each directory must become complete by containing a non-empty
`adapter_config.json` and either `adapter_model.safetensors` or
`adapter_model.bin`. Symlinks are rejected. A strict `checkpoint-N` or
`epoch_N[.fraction]` directory is recognizable as soon as that directory name
appears. If it has a higher order than every complete adapter, the locator
waits for it to become complete and never serves the older adapter in the
meantime. It fails the request if publication does not complete within
`--snapshot-settle-timeout-seconds`. Names such as `checkpoint-120.tmp` are
not recognizable and remain ignored, so an atomic rename from a temporary
sibling is the preferred publication strategy.

Once complete, the directory and all publication files must be at least
`--minimum-snapshot-age-seconds` old (20 seconds by default). If the highest
complete checkpoint is newer than that, the locator waits for that checkpoint
for up to `--snapshot-settle-timeout-seconds` (60 seconds by default), then
revalidates it. It never silently falls back to an older/base policy merely
because the newest complete policy is still settling. A timeout or a
disappearing/regressing checkpoint fails the attack request. Publish a
checkpoint into a temporary sibling and rename it only after all files are
durable; the age check is an additional guard, not an atomicity substitute.
Use a new directory and identifier for every policy version; never overwrite a
previously served adapter in place.

The vLLM adapter ID changes for every newly observed path, preventing a stale
LoRA cache entry from being reused for a newer checkpoint. The base model
defaults to `meta-llama/Llama-3.1-8B-Instruct` at revision
`0e9e39f249a16976918f6564b8830bc894c89659`. Access to the model weights and a
vLLM build compatible with the local CUDA stack are external prerequisites.
The same immutable revision is passed to both tokenizer and training-model
loading. The Qwen training config is independently pinned to its released
40-character revision.

## Provider callable contract

Provider integration stays outside the repository. This avoids embedding an
institution-specific gateway or committing a key. Supply three importable
batch callables:

```text
my_pair_provider:propose
my_pair_provider:score_iteration
my_pair_provider:judge_final
```

`propose(requests)` receives an ordered sequence of `ProposalRequest` values.
Each value contains the configured model-routing label, temperature, token
limit, system prompt, and role/content conversation. Return an equally sized
ordered sequence of mappings with a non-empty `prompt` field (an optional
`improvement` field is preserved in conversation history).

`score_iteration(requests)` receives `IterationJudgeRequest` values containing
both structured calibration fields and the exact formatted system/user
prompts. Return one integer score from 1 through 10 per request, in order.

`judge_final(requests)` receives `FinalJudgeRequest` values. The sequence is
laid out per example as three trials for judge slot 1, three for slot 2, and
three for slot 3. Each request includes the exact system/user prompts,
temperature, and output-token limit. Route by `judge_identifier` and return an
equally sized ordered sequence of `YES`/`NO` strings or Booleans. Do not
collapse or majority-vote the requests in the provider: the backend requires
all nine raw votes to be positive.

The public routing labels resolve retained source/manuscript drift in favor of
the manuscript's reported configuration: PAIR candidates use
`gemini-3.1-flash-lite-preview`, iteration scoring uses
`gemini-2.5-flash-lite`, and the three final slots are
`gemini-3.1-flash-lite-preview`,
`gemini-3-flash-preview-thinking-4096`, and `gpt-5-high-reasoning`. These are
stable experiment labels, not endpoint strings or claims about currently
offered API model IDs. Immutable historical provider revisions were not
retained, so a future provider can only map each label to the closest available
configuration; byte-for-byte reproduction of hosted-model behavior is not
possible. The second label records the manuscript's 4,096-token thinking
setting and the third records high reasoning. Provider retry limits, deadlines,
rate limiting, and credentials are the provider module's responsibility. The
retained messy scripts contain different/default-overridable routes, so the
public labels are manuscript-resolved rather than a byte-identical extraction
of one historical process configuration. To approach the original
provider-side trajectory, a caller-owned shim must additionally reproduce the
historical raw JSON extraction, malformed-response fallbacks, score parsing,
retry/backoff policy, and provider sampling behavior; the typed public boundary
does not claim to reconstruct those unretained details.

## Running the connected worker

First prepare the raw Llama preferences in online mode. The research dataset
uses `rejected_input` for the attack goal and has no stable public `id`; this
mode strips surrounding goal whitespace, validates/maps the goal, and derives
a deterministic `pair-<sha256-prefix>` identifier. The SHA-256 input is the
ASCII domain separator `secpo-online-pair-id-v1` plus NUL, followed by the
UTF-8 bytes of `prompt`, `chosen`, and `rejected`, each preceded by its unsigned
eight-byte big-endian byte length. The ID uses the first 24 lowercase hex
characters. Default/offline preparation is unchanged.

```bash
python scripts/prepare_preferences.py \
  --input "$RAW_LLAMA_PREFERENCES" \
  --output data/llama31_online_preferences.jsonl \
  --expected-records 19157 \
  --online-pair
If both `attack_goal` and legacy `rejected_input` are present, they must agree
byte-for-byte. Content-derived ID collisions fail rather than being numbered
arbitrarily.

Install this project and a host-compatible vLLM separately. Make the provider
module importable, expose any provider credential only in the process
environment, and give each worker exactly one visible GPU. A rank-0 example in
microbatch-compatible mode is:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python \
  scripts/serve_llama31_current_policy_pair.py \
  --config configs/llama31_current_policy_pair.json \
  --exchange-dir runtime/online_exchange \
  --policy-root outputs/llama31_8b_online \
  --rank 0 \
  --proposer my_pair_provider:propose \
  --iteration-judge my_pair_provider:score_iteration \
  --final-judge my_pair_provider:judge_final
```

Start ranks 0 through 7 in separate processes, each with one different visible
GPU. Then launch `scripts/train_llama31_8b_online.sh` against the same exchange
directory. The generic `serve_online_attack_requests.py` remains useful for
other backends; this dedicated script is the construction path for current
policy PAIR.

For an external full-accumulation-group producer, replace the script with:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python \
  scripts/run_llama31_current_policy_pair_exact_batch.py \
  --config configs/llama31_current_policy_pair.json \
  --exchange-dir runtime/exact_pair_exchange \
  --policy-root outputs/llama31_8b_online \
  --rank 0 \
  --proposer my_pair_provider:propose \
  --iteration-judge my_pair_provider:score_iteration \
  --final-judge my_pair_provider:judge_final
```

Runtime exchange files, attack logs, provider traces, model weights, and LoRA
checkpoints are ignored release artifacts. Never add them to Git. Worker
failure files record only an exception class, not provider exception text that
could contain a URL or credential fragment. `backend.last_results` is an
in-memory convenience and is overwritten by the next request. The default
exchange also deletes completed request/response envelopes; `--retain-files`
retains those envelopes but not the full structured PAIR logs. Auditable runs
must persist `last_results` after every request in caller-owned run storage and
archive provider traces/configuration outside the release repository.

## Source-to-release map

| Research file | Public replacement |
| --- | --- |
| `attacker_interface_pair.py` | `src/secpo/backends/llama31_current_policy_pair.py` and the two dedicated worker CLIs |
| `pair_attack.py` main `run_batch` path | `CurrentPolicyPairBackend` |
| `attack_utils.py` PAIR prompts, seed pool, target query, scoring prompt, and conversation truncation | Constants and typed requests in the backend module |
| `utils.py` multi-stage final judge | `FinalJudgeRequest` plus strict 3-by-3 aggregation |
| `run_attacker.slurm` | Portable process-per-visible-GPU command above |

The later `attacker_interface_pair_llama.py` selective rerun, witness shortcut,
early-stop heuristics, private provider clients, scheduler coordination, and
one-time recovery utilities are intentionally not part of this backend.
