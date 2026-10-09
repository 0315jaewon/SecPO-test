#!/usr/bin/env python3
"""Launch durable, resumable workers for the approved checkpoint evaluations."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from secpo.tinker.evaluation_common import now, write_json
from secpo.tinker.evaluation_variants import MODEL_VARIANTS, evaluation_output_root

DEFAULT_BENCHMARKS = ("gpqa_diamond", "mmlu_pro", "minerva_math", "agentdojo", "agentdyn")
REFERENCE_BENCHMARKS = ("alpacaeval2", "sep_utility")
BENCHMARKS = DEFAULT_BENCHMARKS + REFERENCE_BENCHMARKS
OUTPUTS = {"secpo": ROOT / "outputs/qwen38_27b_tinker_utility",
           "base": ROOT / "outputs/qwen38_27b_base_tinker_utility"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=list(DEFAULT_BENCHMARKS))
    parser.add_argument("--worker", choices=BENCHMARKS)
    parser.add_argument("--model-variant", choices=MODEL_VARIANTS, default="secpo")
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-sha256")
    parser.add_argument("--sep-test-dataset", type=Path)
    args = parser.parse_args()
    requested = [args.worker] if args.worker else args.benchmarks
    if args.model_variant == "secpo_lr6e5" and (args.max_output_tokens is None
            or any(b in {"agentdojo", "agentdyn"} for b in requested)):
        parser.error("Use explicit non-agentic benchmarks and an output cap for secpo_lr6e5")
    if args.max_output_tokens is not None and any(b in {"agentdojo", "agentdyn"} for b in requested):
        parser.error("Output-cap variants do not apply to the agentic benchmarks")
    reference_requested = [b for b in requested if b in REFERENCE_BENCHMARKS]
    if reference_requested and (len(requested) != 1 or args.reference is None or args.max_output_tokens is None):
        parser.error("Launch one reference benchmark at a time with --reference and --max-output-tokens")
    reference_options = []
    for name in ("reference", "reference_sha256", "sep_test_dataset"):
        value = getattr(args, name)
        if value is not None:
            reference_options += ["--" + name.replace("_", "-"), str(value)]
    output = evaluation_output_root(args.model_variant, args.max_output_tokens)
    if args.worker:
        benchmark = args.worker
        directory = output / benchmark
        agent = benchmark in {"agentdojo", "agentdyn"}
        reference = benchmark in REFERENCE_BENCHMARKS
        env_name = benchmark if agent else "lm"
        runner = "reference" if reference else "agents" if agent else "lm"
        command = [str(ROOT / f"runtime/eval_envs/{env_name}/bin/python"), "-u",
                   str(ROOT / f"scripts/evaluate_tinker_{runner}.py"),
                   "--benchmark", benchmark, "--output-dir", str(directory),
                   "--model-variant", args.model_variant]
        if reference:
            command += reference_options
        if args.max_output_tokens is not None:
            command += ["--max-output-tokens", str(args.max_output_tokens)]
        if benchmark in {"mmlu_pro", "minerva_math"}:
            command += ["--indices", str(ROOT / "data/eval_indices" /
                                        f"{benchmark}_random100cat_seed20260517_sample_indices.json")]
        started = now()
        with (directory / "run.log").open("a", buffering=1) as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT)
            write_json(directory / "worker.json", dict(status="running", supervisor_pid=os.getpid(),
                       pid=process.pid, started_at=started, command=command, log=str(directory / "run.log")))
            code = process.wait()
        write_json(directory / "worker.json", dict(status="complete" if code == 0 else "failed",
                   supervisor_pid=os.getpid(), pid=process.pid, started_at=started, ended_at=now(),
                   exit_code=code, command=command, log=str(directory / "run.log")))
        raise SystemExit(code)
    for benchmark in args.benchmarks:
        directory = output / benchmark
        smoke_path = directory / "smoke/progress.json"
        if not smoke_path.exists() or json.loads(smoke_path.read_text()).get("status") != "complete":
            raise ValueError(f"{benchmark}: a successful smoke is required before full launch")
        directory.mkdir(parents=True, exist_ok=True)
        # The supervisor holds this lock until the worker exits; stale PIDs alone
        # are not sufficient to prevent duplicate billable launches.
        import fcntl
        with (directory / "launch.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f"{benchmark}: already running", flush=True)
                continue
            worker_path = directory / "worker.json"
            if worker_path.exists() and json.loads(worker_path.read_text()).get("status") == "complete":
                print(f"{benchmark}: already complete", flush=True)
                continue
            with (directory / "supervisor.log").open("a") as log:
                supervisor_command = [sys.executable, str(Path(__file__).resolve()), "--worker", benchmark,
                                      "--model-variant", args.model_variant]
                if args.max_output_tokens is not None:
                    supervisor_command += ["--max-output-tokens", str(args.max_output_tokens)]
                if benchmark in REFERENCE_BENCHMARKS:
                    supervisor_command += reference_options
                process = subprocess.Popen(supervisor_command,
                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, pass_fds=(lock.fileno(),))
            print(f"{benchmark}: launched supervisor PID {process.pid}; {directory / 'run.log'}", flush=True)


if __name__ == "__main__":
    main()
