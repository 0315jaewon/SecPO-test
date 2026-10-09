#!/usr/bin/env python3
"""Durable seven-benchmark lr4e5 generation queue; prior experiments are immutable."""
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
spec = importlib.util.spec_from_file_location('previous_queue', ROOT / 'scripts/queue_lambda_lr6e5_evals.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
original = prior.original
read, write, now = prior.read_json, prior.write_json, prior.now
RUN = original.OUTPUT
CODE = ROOT / 'runtime/lambda_suite_20260916/lr4e5_code'
VARIANT = 'secpo_lr4e5'
OUTPUT = RUN / VARIANT
CONFIG = ROOT / 'configs/lambda_qwen38_secpo_lr4e5_recommended.json'
BENCHMARKS = original.BENCHMARKS
REFERENCES = {'alpacaeval2', 'sep_utility'}


def command(benchmark, phase):
    cmd = original.command('secpo', benchmark, phase)
    cmd[2] = str(CODE / 'scripts' / Path(cmd[2]).name)
    if benchmark in REFERENCES and phase != 'preflight':
        cmd[2] = str(ROOT / 'scripts/generate_lambda_lr4e5_reference.py')
    for flag, value in (('--model-variant', VARIANT), ('--output-dir', OUTPUT / benchmark), ('--inference-config', CONFIG)):
        cmd[cmd.index(flag) + 1] = str(value)
    cmd += ['--credentials', str(ROOT / 'secpo_eval_keys.env')]
    if benchmark in REFERENCES:
        cmd += ['--judge-credentials', str(ROOT / 'secpo_openrouter.env')]
    return cmd


def predecessor_ready():
    # Successful retry results supersede historical failed worker/supervisor files.
    for variant in ('base', 'secpo', 'secpo_lr6e5'):
        for benchmark in BENCHMARKS:
            directory = RUN / variant / benchmark
            if read(directory / 'progress.json').get('status') != 'complete' or not (directory / 'results.json').exists():
                return False
    return True


def snapshot():
    for name, sha in read(RUN / 'source_identity.json').items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
            raise ValueError('Predecessor source changed: ' + name)
    provenance = read(CODE / 'provenance.json')
    inventory = {}
    for name, item in provenance['files'].items():
        path = CODE / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Isolated source changed: ' + name)
        inventory['code/' + name] = item['sha256']
    extras = [Path(__file__).resolve(), ROOT / 'scripts/generate_lambda_lr4e5_reference.py',
        ROOT / 'scripts/prepare_lambda_lr4e5_eval.py', ROOT / 'scripts/queue_lambda_lr6e5_evals.py', CONFIG,
        ROOT / 'runtime/lambda_suite_20260916/merge_lr4e5_remote.py',
        ROOT / 'runtime/lambda_suite_20260916/server_lr4e5_remote.py']
    inventory.update({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in extras})
    if (OUTPUT / 'source_identity.json').exists() and read(OUTPUT / 'source_identity.json') != inventory:
        raise ValueError('lr4e5 sources changed since launch')
    for name in provenance['files']:
        target = OUTPUT / 'code_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CODE / name, target)
    for path in extras:
        target = OUTPUT / 'launch_snapshot' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write(OUTPUT / 'code_snapshot/provenance.json', provenance)
    write(OUTPUT / 'source_identity.json', inventory)


def sync():
    try:
        result = subprocess.run(['rsync', '-a', '-e', shlex.join(original.SSH), str(OUTPUT) + '/',
            original.HOST + ':' + original.REMOTE + '/results/' + VARIANT + '/'], capture_output=True, timeout=60)
        write(OUTPUT / 'backup_status.json', dict(at=now(), exit_code=result.returncode))
    except subprocess.TimeoutExpired:
        write(OUTPUT / 'backup_status.json', dict(at=now(), status='timeout'))


def switch_server():
    config, report = read(CONFIG), read(OUTPUT / 'merge_validation.json')
    if (report.get('status') != 'complete' or report.get('checkpoint') != config['checkpoint']
            or report.get('adapter_sha256') != config['adapter_sha256']):
        raise ValueError('Merge validation missing or mismatched')
    model = original.server_model()
    target = 'secpo-suite-' + VARIANT
    if model == target:
        return
    if model not in {'secpo-suite-base', 'secpo-suite-secpo', 'secpo-suite-secpo_lr6e5'}:
        raise ValueError('Cannot establish owned predecessor server identity')
    subprocess.run(original.SSH + [original.HOST, f'tmux send-keys -t {model} C-c'], check=True)
    deadline = time.monotonic() + 180
    while original.server_model() is not None:
        if time.monotonic() > deadline:
            raise TimeoutError('Previous model did not stop')
        time.sleep(3)
    time.sleep(10)
    remote = (f'/home/ubuntu/secpo_qwen38_runtime_20260916/venv/bin/python '
              f'{original.REMOTE}/code/server_lr4e5_remote.py > {original.REMOTE}/server-secpo_lr4e5.log 2>&1')
    subprocess.run(original.SSH + [original.HOST, f'tmux new-session -d -s {target} ' + shlex.quote(remote)], check=True)
    deadline = time.monotonic() + 1200
    while original.server_model() != target:
        if time.monotonic() > deadline:
            raise TimeoutError('lr4e5 model readiness timed out')
        time.sleep(5)


def expected_status(benchmark, phase):
    return 'awaiting_judgment' if benchmark in REFERENCES and phase != 'preflight' else 'complete'


def run_phase(phase, status):
    pending, failed = {}, []
    env = dict(os.environ, HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MPLBACKEND='Agg')
    for benchmark in BENCHMARKS:
        directory = OUTPUT / benchmark
        directory.mkdir(parents=True, exist_ok=True)
        worker = directory / (phase + '_worker.json')
        expected = expected_status(benchmark, phase)
        if read(worker).get('status') == expected and read(worker).get('exit_code') == 0:
            continue
        if phase == 'full' and read(directory / 'smoke/progress.json').get('status') != expected:
            raise ValueError('Required smoke did not pass: ' + benchmark)
        cmd = command(benchmark, phase)
        with (directory / (phase + '.log')).open('a') as log:
            process = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)
        record = dict(status='running', pid=process.pid, phase=phase, benchmark=benchmark, variant=VARIANT, command=cmd, started_at=now())
        write(worker, record)
        pending[benchmark] = (process, worker, record)
    last_sync = time.monotonic()
    while pending:
        for benchmark, (process, worker, record) in list(pending.items()):
            code = process.poll()
            if code is None:
                continue
            active = worker.parent / 'smoke' if phase == 'smoke' else worker.parent
            expected = expected_status(benchmark, phase)
            valid = code == 0 and (phase == 'preflight' or read(active / 'progress.json').get('status') == expected)
            record.update(status=expected if valid else 'failed', exit_code=code, ended_at=now())
            write(worker, record)
            print(json.dumps(dict(benchmark=benchmark, phase=phase, status=record['status'])), flush=True)
            if not valid:
                failed.append(benchmark)
            del pending[benchmark]
        status.update(updated_at=now(), active_benchmarks=list(pending))
        write(OUTPUT / 'supervisor.json', status)
        if time.monotonic() - last_sync > 60:
            sync()
            last_sync = time.monotonic()
        time.sleep(2)
    sync()
    if failed:
        raise RuntimeError('Failed phase: ' + ', '.join(failed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--queue-fd', type=int)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = os.fdopen(args.queue_fd, 'a+') if args.queue_fd is not None else (OUTPUT / 'queue.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    snapshot()
    if args.detach:
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--queue-fd', str(lock.fileno())]
        if args.preflight_only:
            cmd += ['--preflight-only']
        with (OUTPUT / 'supervisor.log').open('a') as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=(lock.fileno(),))
        write(OUTPUT / 'launch.json', dict(at=now(), pid=proc.pid, command=cmd))
        print(json.dumps(dict(status='launched', pid=proc.pid, output=str(OUTPUT))), flush=True)
        return
    status = dict(status='preflight', variant=VARIANT, pid=os.getpid(), started_at=now(), active_benchmarks=[],
                  reference_judging='separate direct-OpenAI Luna queue', benchmarks=list(BENCHMARKS))
    write(OUTPUT / 'supervisor.json', status)
    gpu = None
    try:
        run_phase('preflight', status)
        if not args.preflight_only:
            gpu = (RUN / 'launch.lock').open('a+')
            fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not predecessor_ready():
                raise ValueError('Predecessor generation/results incomplete')
            status.update(status='loading_model', updated_at=now())
            write(OUTPUT / 'supervisor.json', status)
            switch_server()
            for phase in ('smoke', 'full'):
                snapshot()
                status.update(status=phase, updated_at=now())
                run_phase(phase, status)
        status.update(status='preflight_passed' if args.preflight_only else 'awaiting_judgment', ended_at=now())
    except Exception as error:
        status.update(status='failed', error_type=type(error).__name__, ended_at=now())
        raise
    finally:
        write(OUTPUT / 'supervisor.json', status)
        sync()
        if gpu:
            gpu.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(status='failed', error_type=type(error).__name__)), flush=True)
        raise SystemExit(1) from None
