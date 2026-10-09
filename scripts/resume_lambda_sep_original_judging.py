#!/usr/bin/env python3
"""Resume only the frozen SEP judging stage, without touching any GPU job."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from evaluate_lambda_sep_original import RUN, OUTPUT, snapshot, read
from launch_lambda_sep_original import worker, wait_worker
from secpo.subprocess_env import build_subprocess_environment
from secpo.tinker.evaluation_common import now, write_json


def validate_resume():
    snapshot()
    generation = read(OUTPUT/'progress.json')
    completed = read(OUTPUT/'generate_worker.json')
    if (generation['status'] != 'generation_complete'
            or generation['completed'] != generation['total']
            or completed['status'] != 'complete' or completed['exit_code'] != 0):
        raise ValueError('Cannot resume judging before successful generation completion')
    with (OUTPUT/'worker.lock').open('a+') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUTPUT/'results.json').exists() and read(OUTPUT/'results.json')['status'] == 'complete':
        raise ValueError('SEP judging has already completed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--lock-fd', type=int)
    args = parser.parse_args()
    lock = os.fdopen(args.lock_fd, 'a+') if args.lock_fd is not None else (RUN/'queue.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    validate_resume()
    if args.detach:
        command = [sys.executable,'-u',str(Path(__file__).resolve()),'--lock-fd',str(lock.fileno())]
        with (RUN/'judge_resume_supervisor.log').open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=(lock.fileno(),))
        write_json(RUN/'judge_resume_launch.json', dict(pid=process.pid, command=command, at=now()))
        print(json.dumps(dict(status='launched', pid=process.pid, stage='judge', output=str(RUN))), flush=True)
        return
    previous = read(RUN/'supervisor.json')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive = RUN/'resumes'/stamp
    archive.mkdir(parents=True, exist_ok=False)
    for path in (RUN/'supervisor.json', OUTPUT/'judge_worker.json', OUTPUT/'judge_progress.json'):
        if path.exists():
            shutil.copy2(path, archive/path.name)
    script = Path(__file__).resolve()
    shutil.copy2(script, archive/script.name)
    write_json(archive/'resume.json', dict(at=now(), previous_supervisor=previous.get('pid'),
        supervisor_pid=os.getpid(), launcher_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        policy='Judging only; unchanged original worker, judges, prompts, retries, vote count and concurrency. Reuse successful votes and completed rows. No GPU/server operations.'))
    state = dict(status='running', pid=os.getpid(), stage='judge', resumed_at=now(),
        started_at=previous.get('started_at'), model=previous['model'],
        judge_policy=previous['judge_policy'], generation_status='complete',
        generation_completed_at=previous['generation_completed_at'], resume_archive=str(archive),
        execution_location='REDS; OpenRouter judging only, no Lambda GPU requests')
    def stage(name, **values):
        state.update(stage=name, updated_at=now(), **values)
        write_json(RUN/'supervisor.json', state)
    env = build_subprocess_environment(overrides=dict(HF_HOME='/data2/jaewonc/huggingface',
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MPLBACKEND='Agg'))
    try:
        stage('judge')
        process, record = worker('judge', env)
        wait_worker(process, record, 'judge', stage)
        stage('complete', status='complete', completed_at=now())
    except Exception as error:
        stage('judge', status='failed', error_type=type(error).__name__,
            message=str(error), failed_at=now())
        raise


if __name__ == '__main__':
    main()
