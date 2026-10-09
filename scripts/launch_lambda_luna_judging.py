#!/usr/bin/env python3
"""Durably rejudge cached Alpaca/SEP answers, waiting for lr4e5 generation."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('lambda_original', ROOT / 'scripts/launch_lambda_evals.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
write, now = original.write_json, original.now
OUTPUT = ROOT / 'outputs/qwen38_lambda_luna_openai_20260918'
SOURCES = ['scripts/launch_lambda_luna_judging.py', 'scripts/judge_lambda_luna.py',
           'src/secpo/tinker/luna_judge.py', 'src/secpo/tinker/reference_eval.py',
           'src/secpo/evaluation/reference_utility.py']


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def snapshot():
    identity = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    target = OUTPUT / 'source_identity.json'
    if target.exists() and read(target) != identity:
        raise ValueError('Active Luna judging sources changed')
    for name in SOURCES:
        dest = OUTPUT / 'code_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    write(target, identity)


def sync():
    try:
        proc = subprocess.run(['rsync', '-a', '-e', shlex.join(original.SSH), str(OUTPUT) + '/',
            original.HOST + ':' + original.REMOTE + '/luna_openai_judging_20260918/'], capture_output=True, timeout=60)
        write(OUTPUT / 'backup_status.json', dict(at=now(), exit_code=proc.returncode))
    except subprocess.TimeoutExpired:
        write(OUTPUT / 'backup_status.json', dict(at=now(), status='timeout'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', nargs='+', choices=['base', 'secpo', 'secpo_lr6e5', 'secpo_lr4e5'], required=True)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--lock-fd', type=int)
    args = parser.parse_args()
    if len(set(args.variants)) != len(args.variants):
        raise ValueError('Duplicate models')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = os.fdopen(args.lock_fd, 'a+') if args.lock_fd is not None else (OUTPUT / 'supervisor.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    snapshot()
    if args.detach:
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--variants', *args.variants,
               '--lock-fd', str(lock.fileno())]
        with (OUTPUT / 'supervisor.log').open('a') as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
        write(OUTPUT / 'launch.json', dict(at=now(), pid=proc.pid, command=cmd))
        print(json.dumps(dict(status='launched', pid=proc.pid, variants=args.variants, output=str(OUTPUT))), flush=True)
        return
    jobs = [(v, b) for b in ('alpacaeval2', 'sep_utility') for v in args.variants]
    status = dict(status='running', pid=os.getpid(), started_at=now(), variants=args.variants,
        judge='gpt-5.6-luna', endpoint='https://api.openai.com/v1', max_parallel_jobs=2,
        concurrency_per_job=4, generation_api_requests=0)
    pending, failed = {}, []
    last_sync = time.monotonic()
    env = dict(os.environ, OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MPLBACKEND='Agg')
    try:
        while jobs or pending:
            snapshot()
            for variant, benchmark in list(jobs):
                if len(pending) >= 2:
                    break
                directory = OUTPUT / variant / benchmark
                if read(directory / 'results.json').get('status') == 'complete':
                    jobs.remove((variant, benchmark))
                    continue
                source = original.OUTPUT / variant / benchmark
                ready = read(source / 'progress.json').get('status') in {'complete', 'awaiting_judgment'}
                if not ready:
                    if read(original.OUTPUT / variant / 'supervisor.json').get('status') == 'failed':
                        failed.append(dict(variant=variant, benchmark=benchmark, reason='generation_failed'))
                        jobs.remove((variant, benchmark))
                    continue
                phase = 'full' if read(directory / 'smoke/results.json').get('status') == 'complete' else 'smoke'
                directory.mkdir(parents=True, exist_ok=True)
                cmd = [str(ROOT / 'runtime/eval_envs/lm/bin/python'), '-u', str(ROOT / 'scripts/judge_lambda_luna.py'),
                       '--variant', variant, '--benchmark', benchmark]
                if phase == 'smoke':
                    cmd += ['--smoke', '2']
                with (directory / (phase + '.log')).open('a') as log:
                    proc = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)
                record = dict(status='running', pid=proc.pid, started_at=now(), variant=variant, benchmark=benchmark,
                              phase=phase, command=cmd)
                write(directory / (phase + '_worker.json'), record)
                pending[(variant, benchmark)] = (proc, record, directory)
                jobs.remove((variant, benchmark))
            for key, (proc, record, directory) in list(pending.items()):
                code = proc.poll()
                if code is None:
                    continue
                active = directory / 'smoke' if record['phase'] == 'smoke' else directory
                valid = code == 0 and read(active / 'results.json').get('status') == 'complete'
                record.update(status='complete' if valid else 'failed', exit_code=code, ended_at=now())
                write(directory / (record['phase'] + '_worker.json'), record)
                if valid and record['phase'] == 'smoke':
                    jobs.insert(0, key)
                elif not valid:
                    failed.append(dict(variant=key[0], benchmark=key[1], phase=record['phase'], exit_code=code))
                del pending[key]
            status.update(updated_at=now(), active=['/'.join(k) for k in pending], queued=['/'.join(k) for k in jobs], failures=failed)
            write(OUTPUT / 'supervisor.json', status)
            if time.monotonic() - last_sync > 60:
                sync()
                last_sync = time.monotonic()
            time.sleep(3)
        status.update(status='failed' if failed else 'complete', ended_at=now())
    except Exception as error:
        status.update(status='failed', error_type=type(error).__name__, ended_at=now())
        raise
    finally:
        write(OUTPUT / 'supervisor.json', status)
        sync()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(status='failed', error_type=type(error).__name__)), flush=True)
        raise SystemExit(1) from None
