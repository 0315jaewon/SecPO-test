#!/usr/bin/env python3
"""Durable single-model thinking-on Alpaca run, with all results saved on REDS."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path('/data2/jaewonc/secpo_evals/qwen38_thinking_alpaca_20260923')
BACKUP = Path('/data2/jaewonc/secpo_backups/lambda_20260919')
HOST = 'ubuntu@209.20.159.162'
REMOTE = '/home/ubuntu/secpo_thinking_20260923'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'ConnectTimeout=10',
       '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=6',
       '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=' + str(ROOT / 'runtime/lambda_gpqa_known_hosts'),
       '-i', str(ROOT / 'runtime/lambda_gpqa_ssh/id_ed25519')]


def now():
    return datetime.now(timezone.utc).isoformat()


def write(name, value):
    path = OUTPUT / name
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def run(command, log_name):
    with (OUTPUT / log_name).open('a') as log:
        subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                       check=True, cwd=ROOT)


def remote(action):
    return SSH + [HOST, shlex.join(['python3', REMOTE + '/lambda_thinking_remote.py', action])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--from-prepared', action='store_true')
    parser.add_argument('--wait-for-preparation', action='store_true')
    parser.add_argument('--reuse-server', action='store_true')
    parser.add_argument('--reuse-tunnel', action='store_true')
    parser.add_argument('--resume-compatible', action='store_true')
    parser.add_argument('--lock-fd', type=int)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = os.fdopen(args.lock_fd, 'a+') if args.lock_fd is not None else (OUTPUT / 'supervisor.lock').open('a+')
    if args.wait_for_preparation and args.detach:
        # This continuation has its own process but takes ownership only when
        # the preparer releases its lock. Do not acquire a lock in the parent.
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--from-prepared', '--wait-for-preparation']
        with (OUTPUT / 'continuation.log').open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write('continuation_launch.json', dict(status='queued', pid=process.pid, command=command, at=now()))
        print(json.dumps(dict(status='queued_after_preparation', pid=process.pid, output=str(OUTPUT))))
        return
    fcntl.flock(lock, fcntl.LOCK_EX | (0 if args.wait_for_preparation else fcntl.LOCK_NB))
    if args.wait_for_preparation:
        previous = json.loads((OUTPUT / 'supervisor.json').read_text())
        if previous.get('status') != 'prepared':
            raise ValueError('Predecessor did not complete preparation; evaluation not started')
    if args.detach:
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--lock-fd', str(lock.fileno())]
        if args.prepare_only:
            command.append('--prepare-only')
        if args.from_prepared:
            command.append('--from-prepared')
        for flag in ('reuse_server','reuse_tunnel','resume_compatible'):
            if getattr(args,flag):
                command.append('--'+flag.replace('_','-'))
        with (OUTPUT / 'supervisor.log').open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
        write('launch.json', dict(pid=process.pid, command=command, at=now()))
        print(json.dumps(dict(status='launched', pid=process.pid, output=str(OUTPUT))))
        return
    status = dict(pid=os.getpid(), started_at=now(), host=HOST)
    def update(stage, **more):
        status.update(status=stage, updated_at=now(), **more)
        write('supervisor.json', status)
        print(json.dumps(status), flush=True)
    tunnel = None
    try:
        if not args.from_prepared:
            update('preparing_instance')
            run(SSH + [HOST, 'mkdir -p ' + REMOTE + '/models/base'], 'deployment.log')
            run(['rsync', '-a', '-e', shlex.join(SSH), str(ROOT / 'scripts/lambda_thinking_remote.py'),
                 str(BACKUP / 'SHA256SUMS.models'), HOST + ':' + REMOTE + '/'], 'deployment.log')
            update('installing_and_transferring')
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(run, remote('setup'), 'installation.log'),
                           pool.submit(run, ['rsync', '-a', '--partial', '--info=progress2', '-e', shlex.join(SSH),
                               str(BACKUP / 'models/base') + '/', HOST + ':' + REMOTE + '/models/base/'], 'transfer.log')]
                for future in futures:
                    future.result()
            update('verifying_model')
            run(remote('verify'), 'verification.log')
            run(['rsync', '-a', '-e', shlex.join(SSH), HOST + ':' + REMOTE + '/model_validation.json',
                 HOST + ':' + REMOTE + '/packages.json', str(OUTPUT) + '/'], 'deployment.log')
        elif not (OUTPUT / 'model_validation.json').is_file():
            raise ValueError('No verified prepared model is recorded')
        if args.prepare_only:
            update('prepared')
            return
        update('checking_existing_server' if args.reuse_server else 'starting_server')
        if not args.reuse_server:
            run(remote('start-server'), 'deployment.log')
        if not args.reuse_tunnel:
            with (OUTPUT / 'tunnel.log').open('a') as log:
                tunnel = subprocess.Popen(SSH + ['-o', 'ExitOnForwardFailure=yes', '-N', '-L',
                    '127.0.0.1:18680:127.0.0.1:28680', HOST], stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT)
            write('tunnel.json', dict(pid=tunnel.pid, at=now()))
        import requests
        with requests.Session() as session:
            session.trust_env = False
            deadline = time.monotonic() + 1800
            while True:
                if tunnel is not None and tunnel.poll() is not None:
                    raise RuntimeError('SSH tunnel exited')
                try:
                    response = session.get('http://127.0.0.1:18680/v1/models', timeout=10)
                    if response.ok:
                        models = [r['id'] for r in response.json()['data']]
                        if models != ['secpo-thinking-base-20260923']:
                            raise ValueError('Unexpected model behind tunnel')
                        break
                except requests.RequestException:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('Model startup exceeded 30 minutes')
                update('waiting_for_server')
                time.sleep(15)
        update('evaluating')
        command = [sys.executable, '-u', str(ROOT / 'scripts/evaluate_lambda_thinking.py'),
            '--output-dir', str(OUTPUT / 'base/alpacaeval2'), '--max-output-tokens', '32768']
        if args.resume_compatible:
            command.append('--resume-compatible')
        write('evaluation_command.json', dict(command=command, at=now()))
        run(command, 'evaluation.log')
        update('complete', completed_at=now())
    except Exception as error:
        update('failed', failed_stage=status.get('status'), error_type=type(error).__name__)
        raise
    finally:
        if tunnel is not None:
            tunnel.terminate()
            tunnel.wait(timeout=15)


if __name__ == '__main__':
    main()
