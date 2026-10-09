# SEP generation concurrency 1024

The user requested stopping the concurrency-8 SEP run and trying batch size
1024. This execution interprets 1024 as the global outstanding generation
request limit, not a fixed GPU microbatch and not judge API concurrency.

The verified original worker, PID 2047055, was deliberately stopped with
SIGTERM. Its supervisor cleaned up only our four model replicas. Existing
completion caches and **19,875 saved SEP response rows** were preserved.
The old worker's recorded `failed / exit_code -15` denotes this intentional
stop, not a newly discovered evaluation failure.

## Execution-only changes

- Global generation concurrency: 8 -> 1024.
- Four single-GPU replicas remain pinned by UUID to physical GPUs 0, 2, 3, 4.
- Per-server `--max-num-seqs`: 32 -> 256 (an upper limit; actual admission is
  constrained by the memory-aware scheduler).
- Per-server `--max-num-batched-tokens`: 4096 -> 8192 tokens per scheduler step.
- Local proxy listen backlog: 5 -> 2048.
- Each replica uses a separate private SSH tunnel.

The base weights/revision, BF16, original prompt rendering, thinking off,
16,384-token output cap, all sampling parameters and seeds, six attacks,
9,160 examples, cache identities, and original scoring functions are unchanged.
Judging still uses the original three models with three votes each and
concurrency **8**, not 1024. Completed full MMLU-Pro and Minerva results are not
rerun and their file hashes are checked.

The original evaluation and serving source files are not edited. Separate
wrappers call their existing functions with the user-authorized concurrency
and server batching overrides. New stochastic generations are not promised to
be bit-identical after changing execution batching.

## Capacity-test history

The first execution (`sep_concurrency1024_20261008`) passed 64/64 and 256/256
short requests. At 1024, 1017 succeeded and seven returned transport HTTP 502.
The SSH log recorded failures opening channels 1021 through 1027; no model
OOM was reported. This exposed the single shared tunnel's connection limit.
The attempt stopped before launching scored generation and its audit is kept.

The next execution, `sep_concurrency1024_20261008_tunnels4`, gives each replica
its own SSH connection and repeats all three capacity tests. It passed all
three stages, including **1024/1024** requests with no failures in 7.15 seconds
at 2026-10-08 18:59:39 UTC. The full evaluation worker (PID 2226584) then resumed
with `--concurrency 1024`; the supervisor PID is 2221436. The first 479 real
evaluation requests returned valid HTTP-200 completions across all four GPUs.

A full evaluation worker starts only if all capacity tests pass. Probes have a 16-token cap, are marked
`scored: false`, and never enter the SEP completion cache; actual SEP responses
retain the 16,384-token cap.

## Files and launch

```bash
cd /home/jaewonc/SecPO
runtime/eval_envs/lm/bin/python -u scripts/launch_brev_sep_batched.py --detach
```

Relevant files:

- `configs/brev_sep_concurrency1024_20261008.json`
- `scripts/launch_brev_sep_batched.py`: durable queue, original cache audit,
  capacity tests, separate tunnels, and unchanged judge-stage orchestration.
- `scripts/evaluate_brev_sep_batched.py`: invokes the original SEP evaluator
  with the approved generation concurrency, preserving manifest identity.
- `scripts/brev_sep_batched_remote.py`: only the two server batching overrides.
- `tests/test_brev_sep_batched.py`: original-source and recipe identity guards,
  separate judge concurrency, scoped tunnel cleanup, and GPU allocation.

Do not launch a duplicate: the existing queue lock prevents parallel evaluators.
New remote helpers/configuration must already be copied into the existing
`/data/jaewonc/secpo_brev_base_20261003/code/` project before launching.

Results remain under:

```text
/data2/jaewonc/secpo_evals/qwen38_brev_base_full_20261003_ready
```

The active execution's `supervisor.json`, `capacity_probes/`, `routing.jsonl`,
and per-replica tunnel logs record scheduling and transport status.
`base/sep_static/progress.json` records real evaluation generation progress;
`base/sep_static/judge_progress.json` appears once original judging begins.
