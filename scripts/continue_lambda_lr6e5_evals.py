#!/usr/bin/env python3
"""Advance lr6e5 after explicitly deferring base SEP and all new paid judging.

This is a separate orchestration layer: original source inventories, unfinished
base SEP artifacts, generation protocols, and scoring implementations stay intact.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('frozen_lr6e5_queue', ROOT / 'scripts/queue_lambda_lr6e5_evals.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)
read_json, write_json, now = queue.read_json, queue.write_json, queue.now
REFERENCES = {'alpacaeval2', 'sep_utility'}
GENERATOR = ROOT / 'scripts/generate_lambda_lr6e5_reference.py'


def validate_handoff(run):
    """Allow exactly one unfinished predecessor: fully generated base SEP."""
    for variant in ('base', 'secpo'):
        for benchmark in queue.BENCHMARKS:
            directory = run / variant / benchmark
            worker = read_json(directory / 'full_worker.json')
            progress = read_json(directory / 'progress.json')
            if (variant, benchmark) == ('base', 'sep_utility'):
                if (worker.get('status') != 'failed' or progress.get('status') != 'judging'
                        or progress.get('completed') != 9160 or progress.get('total') != 9160
                        or (directory / 'results.json').exists()):
                    raise ValueError('Base SEP is not the expected failed judging stage')
                for name in ('candidate_outputs.json', 'generations.json'):
                    if len(read_json(directory / name)) != 9160:
                        raise ValueError('Base SEP saved answers are incomplete')
            elif (worker.get('status') != 'complete' or worker.get('exit_code') != 0
                  or progress.get('status') != 'complete'):
                raise ValueError(f'Another predecessor is incomplete: {variant}/{benchmark}')
    if read_json(run / 'supervisor.json').get('status') != 'failed':
        raise ValueError('Original supervisor is not in its expected failed state')


def command(benchmark, phase):
    result = queue.command(benchmark, phase)
    if benchmark in REFERENCES:
        result[2] = str(GENERATOR)
    return result


def phase_succeeded(directory, benchmark, phase):
    record = read_json(directory / (phase + '_worker.json'))
    active = directory / 'smoke' if phase == 'smoke' else directory
    expected = 'awaiting_judgment' if benchmark in REFERENCES else 'complete'
    return (record.get('status') == expected and record.get('exit_code') == 0
            and read_json(active / 'progress.json').get('status') == expected)


def snapshot():
    queue.verify_sources()
    paths = [Path(__file__).resolve(), GENERATOR]
    inventory = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    target = queue.OUTPUT / 'continuation_source_identity.json'
    if target.exists() and read_json(target) != inventory:
        raise ValueError('Continuation code changed since launch')
    for path in paths:
        dest = queue.OUTPUT / 'continuation_code_snapshot' / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    write_json(target, inventory)


def stop_waiting_supervisor():
    """Replace only the exact idle original queue, never an evaluation worker."""
    previous = read_json(queue.OUTPUT / 'supervisor.json')
    pid = previous.get('pid')
    if previous.get('status') != 'queued' or not isinstance(pid, int):
        raise ValueError('lr6e5 supervisor is no longer merely queued')
    expected_script = str(ROOT / 'scripts/queue_lambda_lr6e5_evals.py')
    # Some Conda Python builds omit pidfd_open even on a supporting Linux host.
    pid_fd = os.pidfd_open(pid) if hasattr(os, 'pidfd_open') else None
    try:
        args = Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
        if expected_script not in args:
            raise ValueError('Queue PID does not match the expected command')
        for path in queue.OUTPUT.glob('*/*_worker.json'):
            if read_json(path).get('status') == 'running':
                raise ValueError('A lower-learning-rate evaluation is already running')
        write_json(queue.OUTPUT / 'queue_handoff.json', dict(at=now(), previous_supervisor=previous,
                   original_supervisor=read_json(queue.RUN / 'supervisor.json'),
                   reason='User authorized skipping unfinished base SEP and continuing lr6e5',
                   deferred_judging=['base/sep_utility', 'secpo_lr6e5/alpacaeval2', 'secpo_lr6e5/sep_utility']))
        if pid_fd is not None:
            signal.pidfd_send_signal(pid_fd, signal.SIGTERM)
        else:
            # Recheck the exact command immediately before the PID-based signal.
            if Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0') != args:
                raise ValueError('Queue process identity changed')
            os.kill(pid, signal.SIGTERM)
    finally:
        if pid_fd is not None:
            os.close(pid_fd)


def run_phase(phase, status):
    snapshot()
    pending, failed = {}, []
    env = dict(os.environ, HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MPLBACKEND='Agg')
    for benchmark in queue.BENCHMARKS:
        directory = queue.OUTPUT / benchmark
        if phase_succeeded(directory, benchmark, phase):
            continue
        if phase == 'full' and not phase_succeeded(directory, benchmark, 'smoke'):
            raise ValueError(f'{benchmark}: required smoke did not pass')
        cmd = command(benchmark, phase)
        with (directory / (phase + '.log')).open('a') as log:
            process = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, env=env)
        record = dict(status='running', phase=phase, variant=queue.VARIANT, benchmark=benchmark,
                      generation_only=benchmark in REFERENCES, pid=process.pid, started_at=now(), command=cmd)
        path = directory / (phase + '_worker.json')
        write_json(path, record)
        pending[benchmark] = (process, path, record)
    print(json.dumps(dict(phase=phase, workers=len(pending), at=now())), flush=True)
    last_sync = 0
    while pending:
        for benchmark, (process, path, record) in list(pending.items()):
            code = process.poll()
            if code is None:
                continue
            expected = 'awaiting_judgment' if benchmark in REFERENCES else 'complete'
            active = path.parent / 'smoke' if phase == 'smoke' else path.parent
            valid = code == 0 and read_json(active / 'progress.json').get('status') == expected
            record.update(status=expected if valid else 'failed', exit_code=code, ended_at=now())
            write_json(path, record)
            if not valid:
                failed.append(benchmark)
            print(json.dumps(dict(benchmark=benchmark, phase=phase, status=record['status'])), flush=True)
            del pending[benchmark]
        status.update(updated_at=now(), active_benchmarks=list(pending))
        write_json(queue.OUTPUT / 'supervisor.json', status)
        if time.monotonic() - last_sync > 60:
            queue.sync()
            last_sync = time.monotonic()
        time.sleep(2)
    queue.sync()
    if failed:
        raise RuntimeError(f'{phase} failed: {failed}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--gpu-lock-fd', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--queue-lock-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    queue.verify_sources()
    validate_handoff(queue.RUN)
    if args.check_only:
        print(json.dumps(dict(status='preflight_passed', original_sources_unchanged=True,
                              deferred_predecessor='base/sep_utility', reference_judging=False)))
        return
    if args.gpu_lock_fd is None:
        gpu_lock = (queue.RUN / 'launch.lock').open('a+')
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if queue.original.server_model() not in {'secpo-suite-base', 'secpo-suite-secpo', 'secpo-suite-secpo_lr6e5'}:
            raise RuntimeError('Cannot verify the owned model through the SSH tunnel')
        stop_waiting_supervisor()
        queue_lock = (queue.OUTPUT / 'queue.lock').open('a+')
        deadline = time.monotonic() + 30
        while True:
            try:
                fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Previous queue did not release its lock')
                time.sleep(0.2)
    else:
        gpu_lock = os.fdopen(args.gpu_lock_fd, 'a+')
        queue_lock = os.fdopen(args.queue_lock_fd, 'a+')
    snapshot()
    if args.detach:
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()),
               '--gpu-lock-fd', str(gpu_lock.fileno()), '--queue-lock-fd', str(queue_lock.fileno())]
        with (queue.OUTPUT / 'continuation_supervisor.log').open('a') as log:
            process = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=(gpu_lock.fileno(), queue_lock.fileno()))
        write_json(queue.OUTPUT / 'continuation_launch.json', dict(at=now(), pid=process.pid, command=cmd))
        print(json.dumps(dict(status='launched', pid=process.pid, output=str(queue.OUTPUT))), flush=True)
        return
    status = dict(status='loading_model', pid=os.getpid(), started_at=now(), variant=queue.VARIANT,
                  mode='continue_without_paid_judging', deferred_predecessor='base/sep_utility',
                  deferred_judging=sorted(REFERENCES), benchmarks=list(queue.BENCHMARKS))
    write_json(queue.OUTPUT / 'supervisor.json', status)
    try:
        queue.switch_server()
        for phase in ('smoke', 'full'):
            status.update(status=phase, updated_at=now())
            write_json(queue.OUTPUT / 'supervisor.json', status)
            run_phase(phase, status)
        status.update(status='awaiting_judgment', generation_finished_at=now(), active_benchmarks=[])
    except Exception as error:
        status.update(status='failed', error_type=type(error).__name__, ended_at=now())
        raise
    finally:
        write_json(queue.OUTPUT / 'supervisor.json', status)
        queue.original.sync()
        queue_lock.close()
        gpu_lock.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(status='failed', error_type=type(error).__name__)), flush=True)
        raise SystemExit(1) from None
