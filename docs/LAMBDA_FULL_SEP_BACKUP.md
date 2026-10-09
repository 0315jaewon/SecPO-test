# September 27 evaluations: REDS backup before Lambda shutdown

Backup directory: `/data2/jaewonc/secpo_backups/lambda_full_sep_20261001`

Instance: `ubuntu@209.20.159.207`. The authoritative completion marker is
`status.json`: require `status: complete` and `safe_to_shutdown: true`.
The archival script does not terminate the instance or delete any files.

## Results and per-question/task artifacts

| Archived path | Model | Completed evaluation |
| --- | --- | --- |
| `runs/full_utility/secpo/mmlu_pro` | SecPO lr=8e-5 | Full 12,032-question MMLU-Pro |
| `runs/full_utility/secpo/minerva_math` | SecPO lr=8e-5 | Full 5,000-question Minerva Math; use `math_verify`, not legacy `exact_match` |
| `runs/sep_original_judges/secpo/sep_static` | SecPO lr=8e-5 | Six attacks on all 9,160 SEP examples; original three judges, three trials each |
| `runs/base_static_security/base/agentdojo` | Base Qwen3.8-27B | 949 attacked episodes plus 35 capability checks |
| `runs/base_static_security/base/agentdyn` | Base Qwen3.8-27B | 560 attacked episodes plus 28 capability checks |

Each benchmark directory includes its `results.json`, manifests, logs, and
completion caches. General utility `results.json` files also contain per-question
documents, answers, and scores. Agentic directories contain `tasks/` and
`episodes/`. SEP includes all 54,960 `rows/`, final `judgments/`, `judge_votes/`,
and provider responses in `judge_http_cache/`. Error/retry history is retained;
final results are checked to have no unresolved judgments. SEP aggregate ASR is
the proportion of examples for which at least one of the six attacks succeeded.

The intentionally cancelled earlier Luna SEP experiment has metadata only under
`runs/full_utility/secpo/sep_static`. Its historical supervisor failure does not
indicate an unfinished current evaluation. Deleted generations are not restored.

## Other preserved artifacts

- `lambda_project/`: fresh copies of both remote projects, server logs, exact
  launch helpers, templates, model-validation manifests, and stop metadata.
- `workspace/`: scripts, source, configs, docs, tests, and dependency metadata.
  Run-local `code_snapshot/` directories preserve the exact launch-time sources.
- `benchmark_sources/`: AgentDojo and AgentDyn sources, excluding Git internals.
- `environments/`: Python and installed-package inventories on REDS and Lambda.
- `datasets/SEP_dataset_test.json`: checksum-verified approved SEP test data.
- `prior_subset_manifests/`: the earlier utility manifests referenced by the
  full-split launcher. Full-split documents are also embedded in utility results.
- `audit/`: SHA-256 inventories of every copied research artifact, remote
  transfer checksums, completion checks, and model/checkpoint verification.

The existing large model backups remain at
`/data2/jaewonc/secpo_backups/lambda_20260919/models/{base,secpo_lr8e5,secpo_lr6e5,secpo_lr4e5}`.
Standalone sampler archives remain under the same backup's `checkpoints/`.
The current archive verifies all model file inventories and sizes against the
previously SHA-256-verified manifests, matches the two evaluated remote model
identities, and freshly hashes all three sampler archives. It does not freshly
rehash all 220+ GB of model weights. Optimizer/training-state checkpoints are on
Tinker, not Lambda; this backup makes no new checkpoint-retention API call.

Credentials, SSH keys, virtual environments, and rebuildable caches are excluded.
Preserved original result paths under `/data2/jaewonc/secpo_evals/` remain usable.

## Archive command

```bash
cd /home/jaewonc/SecPO
python3 -u scripts/archive_lambda_full_sep.py
```

This command needs SSH access while collecting the final remote artifacts and
write permission under `/data2/jaewonc`. After a successful archive, reading the
saved results does not require Lambda, Tinker, or judge API access.
