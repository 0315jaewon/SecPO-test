#!/usr/bin/env python3
"""Isolated preparation/server for the September 27 full utility and SEP runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/lambda/nfs/jaewon/secpo_full_sep_20260927')
RUNTIME = Path('/home/ubuntu/secpo_full_sep_20260927')
MODEL = Path('/lambda/nfs/jaewon/secpo_qwen38_eval_20260916/secpo_merged')
PYTHON = RUNTIME / 'venv/bin/python'


def now():
    return datetime.now(timezone.utc).isoformat()


def write(name, obj):
    temporary = ROOT / (name + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2) + '\n')
    temporary.replace(ROOT / name)


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def install():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, '-m', 'venv', str(RUNTIME / 'venv')], check=True)
    subprocess.run([str(PYTHON), '-m', 'pip', 'install', '--upgrade', 'pip==26.2.1'], check=True)
    subprocess.run([str(PYTHON), '-m', 'pip', 'install', 'vllm==0.29.0', 'torch==2.13.0',
                    'transformers==5.10.4', 'tokenizers==0.22.2', 'flashinfer-python==0.6.18'], check=True)
    write('packages.json', json.loads(subprocess.check_output([str(PYTHON), '-m', 'pip', 'list', '--format=json'])))


def verify():
    records = [json.loads(line) for line in (ROOT / 'model_manifest.jsonl').read_text().splitlines()]
    if len(records) != 27 or sum(r['path'].endswith('.safetensors') for r in records) != 18:
        raise ValueError('Unexpected model manifest')
    verified = []
    for row in records:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or row['kind'] != 'file':
            raise ValueError('Invalid model path')
        path = MODEL / relative
        if path.is_symlink() or path.stat().st_size != row['bytes']:
            raise ValueError('Model file missing or changed size')
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(block)
        if h.hexdigest() != row['sha256']:
            raise ValueError('Model checksum mismatch: ' + row['path'])
        verified.append(row)
        write('model_validation.json', dict(status='verifying', files=verified, total=len(records), at=now()))
    write('model_validation.json', dict(status='complete', directory=str(MODEL), files=verified, at=now()))


def setup():
    write('setup_status.json', dict(status='running', pid=os.getpid(), at=now()))
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(install), pool.submit(verify)]
            for future in futures:
                future.result()
        write('setup_status.json', dict(status='complete', at=now()))
    except Exception as error:
        write('setup_status.json', dict(status='failed', error_type=type(error).__name__, at=now()))
        raise


def detached(action):
    name = 'setup_status.json' if action == 'setup' else 'server.json'
    path = ROOT / name
    if path.exists():
        state = json.loads(path.read_text())
        if action == 'setup' and state['status'] == 'complete':
            print('already_prepared')
            return
        if state.get('pid') and alive(state['pid']):
            raise ValueError('An owned process is already running')
    if action == 'serve':
        if json.loads((ROOT / 'setup_status.json').read_text())['status'] != 'complete':
            raise ValueError('Setup has not passed')
        used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
        if any(int(value) > 1024 for value in used.splitlines()):
            raise ValueError('GPU is not idle')
    with (ROOT / (action + '.log')).open('a') as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), action],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(name, dict(status='starting', pid=process.pid, at=now()))
    print(json.dumps(dict(action=action, pid=process.pid)))


def serve():
    command = [str(PYTHON), '-m', 'vllm.entrypoints.openai.api_server', '--model', str(MODEL),
        '--served-model-name', 'secpo-suite-secpo', '--host', '127.0.0.1', '--port', '28782',
        '--dtype', 'bfloat16', '--max-model-len', '65536', '--gpu-memory-utilization', '0.90',
        '--max-num-seqs', '32', '--max-num-batched-tokens', '4096', '--enforce-eager',
        '--no-enable-prefix-caching', '--generation-config', 'vllm', '--seed', '42',
        '--language-model-only', '--no-enable-log-requests']
    env = dict(os.environ, HF_HOME=str(RUNTIME / 'hf_cache'), HF_HUB_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1', OMP_NUM_THREADS='12', TOKENIZERS_PARALLELISM='false',
        PATH=str(PYTHON.parent) + os.pathsep + os.environ['PATH'])
    write('server.json', dict(status='starting', pid=os.getpid(), at=now(), command=command))
    os.execvpe(command[0], command, env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start-setup', 'setup', 'start-server', 'serve'])
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.action == 'start-setup': detached('setup')
    elif args.action == 'start-server': detached('serve')
    elif args.action == 'setup': setup()
    else: serve()
