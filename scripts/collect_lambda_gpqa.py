#!/usr/bin/env python3
"""Collect this authorized Lambda GPQA run and score with the original environment.

This never launches/restarts remote generation or terminates the cloud instance.
The worker polls every 45 seconds and stops on a failed/paused generation run.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from lambda_gpqa import write_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/qwen38_27b_lambda_gpqa_max16384"
HOST = "ubuntu@68.209.75.163"
REMOTE = "/home/ubuntu/secpo_gpqa_lambda/run_bf16_20260914"
SSH_OPTIONS = ["-i", str(ROOT / "runtime/lambda_gpqa_ssh/id_ed25519"),
               "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
               "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=yes",
               "-o", f"UserKnownHostsFile={ROOT / 'runtime/lambda_gpqa_known_hosts'}",
               "-o", "HostKeyAlgorithms=ssh-ed25519"]


def now():
    return datetime.now(timezone.utc).isoformat()


def worker():
    status = dict(status="monitoring_generation", pid=os.getpid(), started_at=now(),
                  remote=HOST + ":" + REMOTE, interval_seconds=45, consecutive_connection_errors=0)
    path = OUTPUT / "collector.json"
    def update():
        write_json(path, {**status, "updated_at":now()})
    update()
    try:
        while True:
            fetched = subprocess.run(["ssh", *SSH_OPTIONS, HOST, f"cat {REMOTE}/progress.json"],
                                     capture_output=True, text=True, timeout=40)
            if fetched.returncode:
                status["consecutive_connection_errors"] += 1
                status["last_connection_error"] = fetched.stderr.strip()[-1000:]
                update()
                if status["consecutive_connection_errors"] >= 10:
                    raise RuntimeError("Unable to retrieve remote progress after ten attempts")
                time.sleep(45)
                continue
            progress = json.loads(fetched.stdout)
            status.update(remote_progress=progress, consecutive_connection_errors=0)
            update()
            if progress["status"] in {"failed", "paused"}:
                status["status"] = "remote_" + progress["status"]
                update()
                return 1
            if progress["status"] == "generation_complete":
                if progress["completed"] != 198 or progress["total"] != 198:
                    raise ValueError("Incomplete remote generation count")
                break
            time.sleep(45)
        status["status"] = "collecting"
        update()
        # Use explicit file destinations so rerunning after a transfer failure
        # cannot accidentally create a nested remote/remote directory.
        local_remote = OUTPUT / "remote"
        local_remote.mkdir(exist_ok=True)
        for name in ("manifest.json", "environment.json", "progress.json", "smoke.json", "generations.json"):
            subprocess.run(["scp", *SSH_OPTIONS, f"{HOST}:{REMOTE}/{name}", str(local_remote / name)],
                           check=True, timeout=300)
        subprocess.run(["scp", *SSH_OPTIONS, f"{HOST}:/home/ubuntu/secpo_gpqa_lambda/run_bf16_20260914.log",
                        str(local_remote / "run.log")], check=True, timeout=300)
        status["status"] = "scoring"
        update()
        command = [str(ROOT / "runtime/eval_envs/lm/bin/python"), str(ROOT / "scripts/lambda_gpqa.py"), "score",
                   "--reference", str(ROOT / "outputs/qwen38_27b_base_tinker_utility_max16384/gpqa_diamond"),
                   "--inputs", str(OUTPUT / "inputs.json"),
                   "--tokenizer", "/data2/jaewonc/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                   "--generations", str(local_remote), "--output", str(OUTPUT)]
        status["scoring_command"] = command
        update()
        env = {**os.environ, "HF_HUB_OFFLINE":"1", "HF_DATASETS_OFFLINE":"1"}
        subprocess.run(command, check=True, cwd=ROOT, env=env)
        result = json.loads((OUTPUT / "results.json").read_text())
        if result["status"] != "complete" or result["n"] != 198:
            raise RuntimeError("Scoring did not produce complete results")
        status.update(status="complete", completed_at=now(), results=result["results"])
        update()
        return 0
    except Exception as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error))
        update()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(worker())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "results.json").exists():
        print("Results already exist; not launching collector")
        return
    with (OUTPUT / "collector.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Collector already running")
            return
        with (OUTPUT / "collector.log").open("a") as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker"],
                                       cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       start_new_session=True, pass_fds=(lock.fileno(),))
        print(json.dumps(dict(collector_pid=process.pid, output=str(OUTPUT))),flush=True)


if __name__ == "__main__":
    main()
