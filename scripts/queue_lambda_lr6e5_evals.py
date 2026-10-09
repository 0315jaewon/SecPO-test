#!/usr/bin/env python3
"""Append lr6e5 evaluations after the existing two-model queue, without interrupting it."""
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
sys.path.insert(0, str(ROOT / 'src'))
from secpo.tinker.evaluation_common import now, write_json

spec = importlib.util.spec_from_file_location('original_lambda_launcher', ROOT / 'scripts/launch_lambda_evals.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
RUN = original.OUTPUT
CODE = ROOT / 'runtime/lambda_suite_20260916/lr6e5_code'
VARIANT = 'secpo_lr6e5'
OUTPUT = RUN / VARIANT
BENCHMARKS = original.BENCHMARKS
CONFIG = ROOT / 'configs/lambda_qwen38_secpo_lr6e5_recommended.json'


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def predecessor_ready(run=RUN):
    """Success and an unlocked GPU lease are BOTH required before switching models."""
    if read_json(run / 'supervisor.json').get('status') != 'complete':
        return False
    for variant in ('secpo', 'base'):
        for benchmark in BENCHMARKS:
            directory = run / variant / benchmark
            worker = read_json(directory / 'full_worker.json')
            if worker.get('status') != 'complete' or worker.get('exit_code') != 0:
                return False
            if read_json(directory / 'progress.json').get('status') != 'complete':
                return False
    return True


def command(benchmark, phase):
    cmd = original.command('secpo', benchmark, phase)
    cmd[2] = str(CODE / 'scripts' / Path(cmd[2]).name)
    cmd[cmd.index('--model-variant') + 1] = VARIANT
    cmd[cmd.index('--output-dir') + 1] = str(OUTPUT / benchmark)
    cmd[cmd.index('--inference-config') + 1] = str(CONFIG)
    cmd += ['--credentials', str(ROOT / 'secpo_eval_keys.env')]
    if benchmark in {'alpacaeval2', 'sep_utility'}:
        cmd += ['--judge-credentials', str(ROOT / 'secpo_openrouter.env')]
    return cmd


def verify_sources():
    # Do not silently accept edits to the existing, still-running experiment.
    for name, sha in read_json(RUN / 'source_identity.json').items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
            raise ValueError(f'Predecessor source changed: {name}')
    provenance = read_json(CODE / 'provenance.json')
    if not provenance.get('files'):
        raise ValueError('Missing isolated source provenance')
    inventory = {}
    for name, item in provenance['files'].items():
        path = CODE / name
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != item['sha256']:
            raise ValueError(f'Isolated evaluation source changed: {name}')
        inventory['code/' + name] = sha
    for path in (Path(__file__).resolve(), ROOT / 'scripts/prepare_lambda_lr6e5_eval.py', CONFIG,
                 ROOT / 'runtime/lambda_suite_20260916/server_lr6e5_remote.py',
                 ROOT / 'runtime/lambda_suite_20260916/merge_lr6e5_remote.py'):
        inventory[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    saved = OUTPUT / 'source_identity.json'
    if saved.exists() and read_json(saved) != inventory:
        raise ValueError('lr6e5 queue sources/config changed since launch')
    write_json(saved, inventory)
    return provenance


def snapshot():
    provenance = verify_sources()
    for name in provenance['files']:
        dest = OUTPUT / 'code_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CODE / name, dest)
    for path in (Path(__file__).resolve(), ROOT / 'scripts/prepare_lambda_lr6e5_eval.py', CONFIG,
                 ROOT / 'runtime/lambda_suite_20260916/server_lr6e5_remote.py',
                 ROOT / 'runtime/lambda_suite_20260916/merge_lr6e5_remote.py'):
        dest = OUTPUT / 'launch_snapshot' / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    write_json(OUTPUT / 'code_snapshot/provenance.json', provenance)


def sync():
    cmd = ['rsync', '-a', '-e', shlex.join(original.SSH), str(OUTPUT) + '/',
           original.HOST + ':' + original.REMOTE + '/results/secpo_lr6e5/']
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        write_json(OUTPUT / 'backup_status.json', dict(updated_at=now(), exit_code=result.returncode))
    except subprocess.TimeoutExpired:
        write_json(OUTPUT / 'backup_status.json', dict(updated_at=now(), status='timeout'))


def run_phase(phase):
    verify_sources()
    pending = {}
    failed = []
    env = dict(os.environ, HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MPLBACKEND='Agg')
    for benchmark in BENCHMARKS:
        directory = OUTPUT / benchmark
        directory.mkdir(parents=True, exist_ok=True)
        status_path = directory / (phase + '_worker.json')
        if read_json(status_path).get('status') == 'complete':
            continue
        if phase == 'full' and read_json(directory / 'smoke/progress.json').get('status') != 'complete':
            raise ValueError(f'{benchmark}: smoke has not passed')
        cmd = command(benchmark, phase)
        with (directory / (phase + '.log')).open('a') as log:
            process = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, env=env)
        record = dict(status='running', phase=phase, variant=VARIANT, benchmark=benchmark,
                      pid=process.pid, started_at=now(), command=cmd)
        write_json(status_path, record)
        pending[benchmark] = (process, status_path, record)
    print(json.dumps(dict(phase=phase, workers=len(pending), at=now())), flush=True)
    last_sync = 0
    while pending:
        for benchmark, (process, path, record) in list(pending.items()):
            code = process.poll()
            if code is not None:
                record.update(status='complete' if code == 0 else 'failed', exit_code=code, ended_at=now())
                write_json(path, record)
                print(json.dumps(dict(benchmark=benchmark, phase=phase, exit_code=code)), flush=True)
                if code:
                    failed.append(benchmark)
                del pending[benchmark]
        if time.monotonic() - last_sync > 60:
            sync()
            last_sync = time.monotonic()
        time.sleep(2)
    sync()
    if failed:
        raise RuntimeError(f'{phase} failed: {failed}')


def switch_server():
    """Called only while holding the original suite's GPU lock after its success."""
    report = read_json(OUTPUT / 'merge_validation.json')
    config = read_json(CONFIG)
    if (report.get('status') != 'complete' or report.get('checkpoint') != config['checkpoint']
            or report.get('adapter_sha256') != config['adapter_sha256']):
        raise ValueError('lr6e5 merge validation is missing or mismatched')
    model = original.server_model()
    if model == 'secpo-suite-secpo_lr6e5':
        return
    if model not in {None, 'secpo-suite-base', 'secpo-suite-secpo'}:
        raise RuntimeError('Another model owns the GPU; refusing to interrupt it')
    if model is not None:
        subprocess.run(original.SSH + [original.HOST, f'tmux send-keys -t {model} C-c'], check=True)
    deadline = time.monotonic() + 180
    while original.server_model() is not None:
        if time.monotonic() > deadline:
            raise TimeoutError('Previous model server did not stop')
        time.sleep(3)
    time.sleep(10)
    remote_command = (f'/home/ubuntu/secpo_qwen38_runtime_20260916/venv/bin/python '
                      f'{original.REMOTE}/code/server_lr6e5_remote.py '
                      f'> {original.REMOTE}/server-secpo_lr6e5.log 2>&1')
    exists = subprocess.run(original.SSH + [original.HOST, 'tmux has-session -t secpo-suite-secpo_lr6e5'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if not exists:
        subprocess.run(original.SSH + [original.HOST, 'tmux new-session -d -s secpo-suite-secpo_lr6e5 '
                                       + shlex.quote(remote_command)], check=True)
    deadline = time.monotonic() + 1200
    while original.server_model() != 'secpo-suite-secpo_lr6e5':
        if time.monotonic() > deadline:
            raise TimeoutError('lr6e5 server readiness timed out')
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['preflight', 'queue'], default='queue')
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = os.fdopen(args.lock_fd, 'a+') if args.lock_fd is not None else (OUTPUT / 'queue.lock').open('a+')
    if args.lock_fd is None:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    snapshot()
    if args.detach:
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--mode', args.mode,
               '--lock-fd', str(lock.fileno())]
        with (OUTPUT / 'supervisor.log').open('a') as log:
            process = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
        write_json(OUTPUT / 'launch.json', dict(status='launched', pid=process.pid, command=cmd, at=now()))
        print(json.dumps(dict(status='queued', pid=process.pid, output=str(OUTPUT))), flush=True)
        return
    status = dict(status='preflight', pid=os.getpid(), started_at=now(), variant=VARIANT,
                  waits_for=['secpo', 'base'], benchmarks=list(BENCHMARKS))
    write_json(OUTPUT / 'supervisor.json', status)
    gpu_lock = None
    try:
        run_phase('preflight')
        if args.mode == 'queue':
            gpu_lock = (RUN / 'launch.lock').open('a+')
            while True:
                owned = False
                try:
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    owned = True
                except BlockingIOError:
                    pass
                if owned and predecessor_ready():
                    break
                if owned:
                    fcntl.flock(gpu_lock, fcntl.LOCK_UN)
                status.update(status='queued', updated_at=now(),
                              predecessor_status=read_json(RUN / 'supervisor.json').get('status'))
                write_json(OUTPUT / 'supervisor.json', status)
                sync()
                time.sleep(30)
            verify_sources()
            status.update(status='loading_model', updated_at=now())
            write_json(OUTPUT / 'supervisor.json', status)
            switch_server()
            for phase in ('smoke', 'full'):
                status.update(status=phase, updated_at=now())
                write_json(OUTPUT / 'supervisor.json', status)
                run_phase(phase)
        status.update(status='complete', ended_at=now())
    except Exception as exc:
        status.update(status='failed', error_type=type(exc).__name__, error=str(exc), ended_at=now())
        raise
    finally:
        write_json(OUTPUT / 'supervisor.json', status)
        sync()
        if gpu_lock is not None:
            gpu_lock.close()


if __name__ == '__main__':
    main()
