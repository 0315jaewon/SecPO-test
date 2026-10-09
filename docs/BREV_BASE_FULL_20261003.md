# Base Qwen3.8 full evaluations on Brev

Instance `brev-v6r091tdo`; SSH user `nvidia`, public endpoint
`global.prd.ga.run.brev.nvidia.com:26865`. Host ED25519 fingerprint was verified
out of band against `SHA256:iIvxruaezYBuUDAHUw//6/VXiR9DfSL7sRrmS2G6+jo`.

Only physical GPU IDs **4--7** are authorized. The queue pins one model replica
to each GPU by verified UUID: SEP uses **7**, MMLU-Pro **5**, Minerva Math **6**.
GPU 4 became occupied during preparation, so its pre-generation queue was
superseded and its audit files retained. GPUs 0--4 are never used or stopped.
The queue waits up to 24 hours for its assigned GPU if busy. GPU startup refuses an
occupied assigned device. Cleanup can signal only its own matching PID/process
group, start time, model path and exact launch command.

| Benchmark | Scope | Concurrency | Output cap |
| --- | --- | --- | --- |
| SEP static security | All six attacks x 9,160 examples = 54,960 generations | 8 | 16,384 |
| MMLU-Pro | All 12,032 test questions; 5-shot | 12 | 16,384 |
| Minerva Math | All 5,000 test questions; 4-shot; Math-Verify | 12 | 16,384 |

Base revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The 32 model files
are checked against the original REDS base backup's SHA-256 manifest.
Thinking/preserve-thinking are false. Sampling remains temperature 0.7,
top-p 0.8, top-k 20, min-p 0, presence penalty 1.5, repetition penalty 1,
frequency penalty 0 and seed 42. Server context is 65,536 tokens, BF16,
single-GPU, max sequences 32, scheduler token budget 4,096, eager mode,
prefix caching disabled. Model-provided generation defaults are disabled.

vLLM 0.29.0, Torch 2.13.0, Transformers 5.10.4, Tokenizers 0.22.2 and
FlashInfer 0.6.18 match the previous server. The new host initially has Python
3.12, Ubuntu 24.04 and driver 595.71.05; these platform differences from the
earlier Lambda instance are recorded, not claimed bit-identical. An isolated
uv-managed Python 3.10 is used for inference because system Python lacks
ensurepip; the exact patch version is recorded in `install_status.json`. The CPU
renderer/scorer uses the same pinned REDS environment as before.
`CUDA_HOME`/`CUDA_PATH` point to the private environment's
`lib/python3.10/site-packages/nvidia/cu13`, and its compiler directory is on
the server PATH. The host has no system CUDA toolkit. Failed startup logs
from before this environment-path fix are preserved; no samples were generated.
Compiler, CRT and NVVM are pinned to 13.0.88 to match Torch's CUDA 13.0 headers;
unconstrained dependency resolution initially installed a mismatched 13.4 compiler.
`FLASHINFER_WORKSPACE_BASE` is explicitly project-local (FlashInfer does not use
XDG_CACHE_HOME for this cache). The `prepare-cuda` helper builds/loads sampling
kernels on CPU with GPUs hidden before the evaluation queue is launched.
Once a benchmark's assigned GPU is free, `validate-cuda` tests actual sampling
on that GPU before starting its model server and benchmark smoke test.
It supplies project-local `lib64 -> lib` and `libcudart.so -> libcudart.so.13`
links needed for the wheel-installed toolkit's linker layout.

SEP retains the original witness gate, semantic prompt, three judge models and
three trials per model, requiring all nine YES. Original model identifiers are
`gemini-3.1-flash-lite-preview`, `gemini-3-flash-preview-thinking-4096`, and
`gpt-5-high-reasoning`; the same OpenRouter routes and reasoning settings are
used. No Luna substitution. A synthetic compatibility probe must pass before
generation. All final responses and votes are cached, including failed-attempt
history; unknown judgments do not become negative examples.

Original evaluators and prior results are unmodified. `evaluate_brev_base.py`
reuses the prior generator, judge loop, renderer, sampler, full-split validation
and scorer; it only supplies base-model routing and fresh output paths. Preflight
compares the new protocol with the completed lr8 full runs, including exact
dataset and attack-request hashes. No prior model responses are reused.

## Locations and launch

- REDS results: `/data2/jaewonc/secpo_evals/qwen38_brev_base_full_20261003_ready`
- Brev isolated environment, weights, and server logs:
  `/data/jaewonc/secpo_brev_base_20261003`
- Exact current research-code snapshot (including uncommitted wrappers):
  `repository_snapshot.tar.gz` and `repository_snapshot/` on Brev.
- API credentials remain on REDS and are excluded from transferred artifacts.

```bash
cd /home/jaewonc/SecPO
runtime/eval_envs/lm/bin/python scripts/launch_brev_base_full.py --detach
```

The detached supervisor performs preflights and the SEP judge probe, waits for
the isolated model/environment setup and assigned GPU capacity, opens loopback-only SSH tunnels, then
runs a smoke test and full evaluation on each assigned GPU. The three benchmark
pipelines are independent and concurrent. Each owned server is stopped when
its generation is finished; SEP judging then proceeds without a GPU. A failed
benchmark does not cancel the other benchmarks, but the final supervisor
status reports failure and preserves partial data for an explicit resume.

`supervisor.json` gives overall status; `base/<benchmark>/pipeline.json` gives
stage and errors. Worker JSONs contain the exact commands and exit codes.
`progress.json`, `judge_progress.json`, `results.json`, manifests, per-question
outputs, and caches are written directly to REDS. Remote logs are synchronized
periodically into `remote/`. Small `smoke/` results are not final scores.
