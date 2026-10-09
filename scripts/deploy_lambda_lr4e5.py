#!/usr/bin/env python3
"""Finish adapter transfer/validated merge and automatically launch seven evals.

The already-running export owns the archive request. This durable supervisor
waits for its verified manifest, and never transfers any API credentials.
"""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('original_lambda', ROOT / 'scripts/launch_lambda_evals.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
write, now = original.write_json, original.now
OUTPUT = original.OUTPUT / 'secpo_lr4e5'
EXPORT = ROOT / 'runtime/lambda_suite_20260916/secpo_lr4e5_export'


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--lock-fd', type=int)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = os.fdopen(args.lock_fd, 'a+') if args.lock_fd is not None else (OUTPUT / 'deployment.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.detach:
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--lock-fd', str(lock.fileno())]
        with (OUTPUT / 'deployment.log').open('a') as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
        write(OUTPUT / 'deployment_launch.json', dict(at=now(), pid=proc.pid, command=cmd))
        print(json.dumps(dict(status='launched', pid=proc.pid, output=str(OUTPUT))), flush=True)
        return
    status = dict(status='waiting_for_export', started_at=now(), pid=os.getpid(), variant='secpo_lr4e5')
    def update(stage, **kwargs):
        status.update(status=stage, updated_at=now(), **kwargs)
        write(OUTPUT / 'deployment.json', status)
        print(json.dumps(status), flush=True)
    try:
        update('waiting_for_export')
        deadline = time.monotonic() + 2100
        while not (EXPORT / 'export_manifest.json').exists():
            state = read(EXPORT / 'export_status.json')
            if state.get('status') == 'failed':
                raise RuntimeError('Export failed; deployment stopped')
            if time.monotonic() > deadline:
                raise TimeoutError('Export wait exceeded 35 minutes')
            update('waiting_for_export', export_status=state.get('status'))
            time.sleep(15)
        subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_lambda_lr4e5_eval.py'), '--config-only'], check=True, cwd=ROOT)
        update('uploading_adapter')
        subprocess.run(original.SSH + [original.HOST, 'mkdir -p ' + original.REMOTE + '/secpo_lr4e5_export ' + original.REMOTE + '/code'], check=True, timeout=60)
        subprocess.run(['rsync', '-a', '--include=export_manifest.json', '--include=adapter/***', '--exclude=*',
            '-e', shlex.join(original.SSH), str(EXPORT) + '/', original.HOST + ':' + original.REMOTE + '/secpo_lr4e5_export/'], check=True, timeout=900)
        subprocess.run(['rsync', '-a', '-e', shlex.join(original.SSH),
            str(ROOT / 'runtime/lambda_suite_20260916/merge_lr4e5_remote.py'),
            str(ROOT / 'runtime/lambda_suite_20260916/server_lr4e5_remote.py'),
            original.HOST + ':' + original.REMOTE + '/code/'], check=True, timeout=60)
        update('merging_and_validating')
        report_path = original.REMOTE + '/lr4e5_merge_validation.json'
        merge_cmd = ('/home/ubuntu/secpo_qwen38_runtime_20260916/merge-venv/bin/python '
                     + original.REMOTE + '/code/merge_lr4e5_remote.py > ' + original.REMOTE + '/merge-lr4e5.log 2>&1')
        exists = subprocess.run(original.SSH + [original.HOST, 'test -f ' + report_path], capture_output=True, timeout=30).returncode == 0
        if not exists:
            subprocess.run(original.SSH + [original.HOST, 'tmux new-session -d -s secpo-merge-lr4e5 ' + shlex.quote(merge_cmd)], check=True, timeout=60)
        deadline = time.monotonic() + 3600
        while True:
            proc = subprocess.run(original.SSH + [original.HOST, 'test -f ' + report_path + ' && cat ' + report_path], capture_output=True, text=True, timeout=30)
            if proc.returncode == 0:
                report = json.loads(proc.stdout)
                if report.get('status') != 'complete':
                    raise ValueError('Remote merge validation failed')
                write(OUTPUT / 'merge_validation.json', report)
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Remote merge timeout')
            alive = subprocess.run(original.SSH + [original.HOST, 'tmux has-session -t secpo-merge-lr4e5'], capture_output=True, timeout=30)
            if alive.returncode != 0:
                raise RuntimeError('Remote merge exited without validation report')
            update('merging_and_validating')
            time.sleep(20)
        update('launching_evaluations')
        subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/launch_lambda_lr4e5_evals.py'), '--detach'], check=True, cwd=ROOT)
        update('evaluations_launched', launch=read(OUTPUT / 'launch.json'))
    except Exception as error:
        update('failed', error_type=type(error).__name__)
        # Let the independent judge queue stop waiting if deployment fails
        # before a generation supervisor has even been created.
        if not (OUTPUT / 'supervisor.json').exists():
            write(OUTPUT / 'supervisor.json', dict(status='failed', stage='deployment',
                error_type=type(error).__name__, ended_at=now()))
        raise


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(status='failed', error_type=type(error).__name__)), flush=True)
        raise SystemExit(1) from None
