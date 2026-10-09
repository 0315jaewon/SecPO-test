#!/usr/bin/env python3
"""Judge two saved benchmarks now and queue lr6e5 SEP after generation exits."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('lambda_launcher', ROOT / 'scripts/launch_lambda_evals.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
RUN = original.OUTPUT
OUTPUT = RUN / 'judging_resume'
JOBS = [('base', 'sep_utility'), ('secpo_lr6e5', 'alpacaeval2'), ('secpo_lr6e5', 'sep_utility')]
WORKER = ROOT / 'scripts/judge_lambda_cached_reference.py'
write_json, now = original.write_json, original.now


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def ready(run, variant, benchmark):
    directory = run / variant / benchmark
    worker = read(directory / 'full_worker.json')
    progress = read(directory / 'progress.json')
    if worker.get('status') not in {'failed', 'awaiting_judgment', 'complete'}:
        return False
    if (variant, benchmark) == ('secpo_lr6e5', 'sep_utility'):
        # The generation supervisor must have observed its worker's terminal
        # state before judging changes progress.json, avoiding a handoff race.
        if read(run / variant / 'supervisor.json').get('status') not in {'awaiting_judgment', 'complete'}:
            return False
    return (progress.get('completed') == read(directory / 'manifest.json').get('reference', {}).get('rows')
            and (directory / 'candidate_outputs.json').is_file() and (directory / 'generations.json').is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = os.fdopen(args.lock_fd, 'a+') if args.lock_fd is not None else (OUTPUT / 'launch.lock').open('a+')
    if args.lock_fd is None:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sources = [Path(__file__).resolve(), WORKER]
    inventory = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if (OUTPUT / 'source_identity.json').exists() and read(OUTPUT / 'source_identity.json') != inventory:
        raise ValueError('Judging wrapper sources changed since launch')
    write_json(OUTPUT / 'source_identity.json', inventory)
    for path in sources:
        target = OUTPUT / 'code_snapshot' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    if args.detach:
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--lock-fd', str(lock.fileno())]
        with (OUTPUT / 'supervisor.log').open('a') as log:
            process = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
        write_json(OUTPUT / 'launch.json', dict(at=now(), pid=process.pid, command=cmd))
        print(json.dumps(dict(status='launched', pid=process.pid, output=str(OUTPUT))), flush=True)
        return
    pending, active, failed = list(JOBS), {}, []
    status = dict(status='running', started_at=now(), pid=os.getpid(), generation_api_requests_allowed=0)
    last_sync = 0
    while pending or active:
        for job in list(pending):
            variant, benchmark = job
            directory = RUN / variant / benchmark
            result = read(directory / 'results.json')
            if result.get('status') == 'complete' and read(directory / 'progress.json').get('status') == 'complete':
                pending.remove(job)
                continue
            if not ready(RUN, *job):
                generation = read(directory / 'full_worker.json')
                if variant == 'secpo_lr6e5' and generation.get('status') == 'failed':
                    failed.append('/'.join(job))
                    pending.remove(job)
                continue
            cmd = [str(ROOT / 'runtime/eval_envs/lm/bin/python'), '-u', str(WORKER),
                   '--variant', variant, '--benchmark', benchmark]
            with (directory / 'judge_resume.log').open('a') as log:
                process = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, env=dict(os.environ, TOKENIZERS_PARALLELISM='false',
                    OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MPLBACKEND='Agg'))
            record = dict(status='running', pid=process.pid, started_at=now(), command=cmd,
                          variant=variant, benchmark=benchmark, mode='cached_answers_only')
            write_json(directory / 'judge_worker.json', record)
            active[job] = (process, record)
            pending.remove(job)
        for job, (process, record) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            directory = RUN / job[0] / job[1]
            valid = code == 0 and read(directory / 'results.json').get('status') == 'complete'
            record.update(status='complete' if valid else 'failed', exit_code=code, ended_at=now())
            write_json(directory / 'judge_worker.json', record)
            if not valid:
                failed.append('/'.join(job))
            print(json.dumps(dict(job='/'.join(job), status=record['status'])), flush=True)
            del active[job]
        status.update(updated_at=now(), active=['/'.join(job) for job in active],
                      waiting_for_generation=['/'.join(job) for job in pending], failed=failed)
        write_json(OUTPUT / 'supervisor.json', status)
        if time.monotonic() - last_sync > 60:
            original.sync()
            last_sync = time.monotonic()
        if pending or active:
            time.sleep(5)
    status.update(status='failed' if failed else 'complete', ended_at=now())
    write_json(OUTPUT / 'supervisor.json', status)
    original.sync()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(status='failed', error_type=type(error).__name__)), flush=True)
        raise SystemExit(1) from None
