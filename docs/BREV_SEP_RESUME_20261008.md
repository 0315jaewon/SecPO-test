# SEP-only resumption, 2026-10-08

The Oct 8 inspection found H100 GPUs 0, 2, 3, 4 idle and GPUs 1, 5, 6, 7
occupied. The user previously authorized four available GPUs regardless of
index and now requested resuming SEP. Four single-GPU base replicas use the
idle IDs, each pinned by UUID and guarded against occupied-device startup.

Only unfinished SEP is run. Full MMLU-Pro (82.24734%) and Minerva Math
(96.04% Math-Verify) remain completed and their result files are SHA-256 checked.
SEP resumes the same 54,960 requests from 18,560 saved responses, with 36,400
remaining at launch. Original datasets, six attacks, model revision, thinking
off, 16,384-token cap, sampling, global generation concurrency 8, and original
three-judge/three-vote scoring are unchanged. No Luna judge substitution.

The prior Oct 4 failure was a broken SSH tunnel, logged as `Broken pipe`,
followed by HTTP 502 connection errors from the loopback proxy. This execution
can retry generation up to three total attempts for that diagnosed transport
failure only. Each retry reconnects, revalidates its replicas and reuses valid
cached responses. It does not retry accounting/protocol errors or failed judge
decisions, and it never records an aborted response as valid. As before, new
generations are not promised bit-identical across different physical placement
or batching.

Original source files and prior execution snapshots are unmodified. The new
launch wrappers import the prior serving and evaluation orchestration. Outputs
stay in `/data2/jaewonc/secpo_evals/qwen38_brev_base_full_20261003_ready`.
Current execution metadata and the before-resume audit live under
`resume_sep_20261008/`; `active_execution.json` points there. New per-response
routing records and failed HTTP responses are saved for diagnosis. Existing SEP
cache files are inventoried before resumption. Credentials remain on REDS.

```bash
cd /home/jaewonc/SecPO
runtime/eval_envs/lm/bin/python -u scripts/launch_brev_sep_resume.py --detach
```

Generation completes before semantic judging begins. The wrapper stops only
its own verified model replicas after generation; API judging then continues
on REDS without a GPU. See `pipelines/sep_static.json`, SEP `progress.json`,
and eventually `judge_progress.json` for the current phase.
