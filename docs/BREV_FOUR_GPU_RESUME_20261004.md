# Four-GPU resumption, 2026-10-04

User expanded permission from physical GPU IDs 4--7 to four available GPUs.
At inspection, IDs 1, 2, 5, 6, 7 were idle. This execution uses **1, 2, 5, 6**
and leaves 7 free; occupied GPUs 0, 3, 4 are not used or stopped.

| GPUs | Work |
| --- | --- |
| 1 and 2 | SEP static attacks: two independent single-GPU base replicas |
| 5 | Full MMLU-Pro |
| 6 | Full Minerva Math |

The prior runs actually started after GPUs became free. All three engine logs
then record SIGTERM at **2026-10-04 07:17:55 UTC**, aborting in-flight requests.
The sampler rejected those responses rather than treating them as valid answers.
The logs do not establish who sent the signal.

Resume starts with 3,029 SEP row/cache files, 4,248 MMLU completion cache files,
and 1,691 Math completion cache files (including smoke completions). Existing
files are SHA-256 inventoried in `existing_cache_sha256.json` before continuing.
No cached generation is discarded or edited. Frozen original evaluation source
hashes, benchmark manifests, dataset fingerprints and cache identities remain
unchanged.

An execution-only remote wrapper reuses the existing serving recipe with new
GPU UUIDs and ports. Four loopback SSH forwards expose the replicas on REDS.
A byte-preserving local proxy retains the original logical inference endpoints;
SEP requests alternate between its two replicas. Global concurrency remains
SEP 8, MMLU 12, Math 12. This is replication, not tensor parallelism.

The original benchmark scripts run unmodified. Base model/revision, BF16,
thinking off, sampling/seed, full test sets, 16,384-token output cap, original
three judges x three votes, and scoring are unchanged. Physical placement and
batching differ; new stochastic generations are not promised bit-identical.
Original benchmark manifests describe the first execution; the resume manifest
records this hardware/routing override and is the authoritative placement record.

Replica identity and CUDA UUID are verified before adoption. Each server has
a fresh, unscored four-token completion probe. Aborted/malformed responses are
saved under `transport_errors/` and passed unchanged to the original validator;
the proxy does not fix, accept, or silently resample them. Owned replicas are
stopped when their benchmark generation finishes. SEP then uses the original
API judge protocol on REDS, requiring no GPU.

Results and original caches remain in:
`/data2/jaewonc/secpo_evals/qwen38_brev_base_full_20261003_ready`

Current execution metadata, routing audit, logs and preserved prior status:
`resume_four_gpu_20261004/` within that run directory.

Launch from REDS:

```bash
cd /home/jaewonc/SecPO
runtime/eval_envs/lm/bin/python -u scripts/launch_brev_four_gpu.py --detach
```

The existing Brev model/environment is reused. Remote launcher:
`/data/jaewonc/secpo_brev_base_20261003/code/scripts/brev_four_gpu_remote.py`.
API credentials stay on REDS. No evaluation/judge code or system packages are
changed by this resumption.
