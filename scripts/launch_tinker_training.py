#!/usr/bin/env python3
"""Launch one fresh Tinker training run with a durable supervisor and logs."""

import argparse
from dataclasses import asdict
import fcntl
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from secpo.tinker.config import load_tinker_config
from secpo.tinker.evaluation_common import digest, load_credentials, now, write_json
from secpo.tinker.runner import build_training_plan
from secpo.evaluation.reference_utility import sha256_file


def require_fresh_output(directory):
    for name in ("config.json", "metrics.jsonl", "result.json", "worker.json", "launch.json"):
        if (directory / name).exists():
            raise FileExistsError(f"Existing run artifact {name}; use a fresh output directory")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, default=ROOT / "secpo_eval_keys.env")
    parser.add_argument("--mode", choices=("launch", "supervise", "worker"), default="launch",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    config_path = args.config.resolve()
    config = load_tinker_config(config_path)
    directory = Path(config.output_dir).resolve()
    fingerprint = digest(asdict(config))
    common = ["--config", str(config_path), "--credentials", str(args.credentials.resolve())]

    if args.mode != "launch":
        launch = json.loads((directory / "launch.json").read_text())
        if launch["config_sha256"] != fingerprint:
            raise ValueError("Training configuration changed after launch")

    if args.mode == "worker":
        load_credentials(args.credentials)
        from secpo.tinker.cli import main as train
        return train(["--config", str(config_path)])

    if args.mode == "supervise":
        command = [sys.executable, "-u", str(Path(__file__).resolve()), *common, "--mode", "worker"]
        started = now()
        with (directory / "run.log").open("a", buffering=1) as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
            record = dict(status="running", supervisor_pid=os.getpid(), pid=process.pid,
                          started_at=started, command=command, config_sha256=fingerprint,
                          log=str(directory / "run.log"))
            write_json(directory / "worker.json", record)
            code = process.wait()
        valid_result = False
        result_path = directory / "result.json"
        if code == 0 and result_path.exists():
            result = json.loads(result_path.read_text())
            valid_result = (result.get("steps_completed") == launch["total_steps"]
                            and all(isinstance(result.get(k), str) and result[k].startswith("tinker://")
                                    for k in ("final_state_path", "final_sampler_path")))
        write_json(directory / "worker.json", {
            **record, "status": "complete" if code == 0 and valid_result else "failed",
            "ended_at": now(), "exit_code": code, "full_result_validated": bool(valid_result),
        })
        return 0 if code == 0 and valid_result else 1

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "launch.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Training run already active; no duplicate launched.", flush=True)
            return 0
        require_fresh_output(directory)
        plan = build_training_plan(record_count=config.expected_records, batch_size=config.batch_size,
                                   num_epochs=config.num_epochs, seed=config.seed, max_steps=config.max_steps)
        command = [sys.executable, "-u", str(Path(__file__).resolve()), *common, "--mode", "supervise"]
        source_paths = [Path(__file__).resolve(), *sorted((ROOT / "src/secpo/tinker").glob("*.py"))]
        write_json(directory / "launch.json", {
            "launched_at": now(), "config_path": str(config_path), "config": asdict(config),
            "config_sha256": fingerprint, "total_steps": plan.total_steps,
            "python_executable": sys.executable, "supervisor_command": command,
            "package_versions": {p: version(p) for p in ("tinker", "transformers", "datasets", "torch")},
            "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in source_paths},
            "offline_cache_settings": {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")},
        })
        with (directory / "supervisor.log").open("a") as log:
            supervisor = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
        print(json.dumps({"status": "launched", "supervisor_pid": supervisor.pid,
                          "total_steps": plan.total_steps, "output_dir": str(directory)}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Service exceptions may contain private headers. Keep diagnostics safe.
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}), flush=True)
        raise SystemExit(1) from None
