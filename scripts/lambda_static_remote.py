#!/usr/bin/env python3
"""Owned, private Lambda environment for the two SecPO static-security runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/lambda/nfs/jaewon/secpo_static_security_20260924')
RUNTIME = Path('/home/ubuntu/secpo_static_security_20260924')
PRIOR = Path('/lambda/nfs/jaewon/secpo_qwen38_eval_20260916')
MODELS = {'secpo_lr4e5': PRIOR / 'secpo_lr4e5_merged', 'secpo': PRIOR / 'secpo_merged'}
PYTHON = RUNTIME / 'venv/bin/python'


def write(name, value):
    path = ROOT / name
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def install():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, '-m', 'venv', str(RUNTIME / 'venv')], check=True)
    subprocess.run([str(PYTHON), '-m', 'pip', 'install', '--upgrade', 'pip==26.2.1'], check=True)
    subprocess.run([str(PYTHON), '-m', 'pip', 'install', 'vllm==0.29.0', 'transformers==5.10.4',
                    'torch==2.13.0', 'tokenizers==0.22.2', 'flashinfer-python==0.6.18'], check=True)
    write('packages.json', json.loads(subprocess.check_output([str(PYTHON), '-m', 'pip', 'list', '--format=json'])))


def verify():
    result = {}
    lines = (ROOT / 'SHA256SUMS.models').read_text().splitlines()
    for variant, directory in MODELS.items():
        prefix = 'models/' + ('secpo_lr8e5' if variant == 'secpo' else variant) + '/'
        files = []
        for line in lines:
            expected, relative = line.split('  ', 1)
            if not relative.startswith(prefix):
                continue
            p = directory / relative[len(prefix):]
            h = hashlib.sha256()
            with p.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    h.update(block)
            if h.hexdigest() != expected:
                raise ValueError('Model checksum mismatch: ' + relative)
            files.append(dict(path=p.name, bytes=p.stat().st_size, sha256=expected))
        if len([x for x in files if x['path'].endswith('.safetensors')]) != 18:
            raise ValueError('Expected 18 shards per model')
        result[variant] = dict(directory=str(directory), files=files, status='complete', at=now())
        write('model_validation.json', result)


def setup():
    write('setup_status.json', dict(status='running', at=now(), pid=os.getpid()))
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(install), pool.submit(verify)]
            for future in futures:
                future.result()
        write('setup_status.json', dict(status='complete', at=now()))
    except Exception as error:
        write('setup_status.json', dict(status='failed', at=now(), error_type=type(error).__name__))
        raise


def alive(pid):
    try:
        os.kill(pid, 0)
        return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0] != 'Z'
    except (ProcessLookupError, FileNotFoundError):
        return False


def start_setup():
    status = ROOT / 'setup_status.json'
    if status.exists():
        previous = json.loads(status.read_text())
        if previous['status'] == 'complete':
            print('already_prepared')
            return
        if previous.get('pid') and alive(previous['pid']):
            raise ValueError('Setup already running')
    with (ROOT / 'setup.log').open('a') as log:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'setup'],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(status='setup_launched', pid=proc.pid)))


def start(variant):
    if json.loads((ROOT / 'setup_status.json').read_text())['status'] != 'complete':
        raise ValueError('Preparation incomplete')
    state = ROOT / 'server.json'
    if state.exists():
        old = json.loads(state.read_text())
        if alive(old['pid']):
            observed = Path(f'/proc/{old["pid"]}/cmdline').read_bytes().split(b'\0')
            if b'secpo-static-' + old['variant'].encode() not in observed or str(PYTHON).encode() not in observed:
                raise ValueError('Existing server PID ownership mismatch')
            if old['variant'] == variant:
                print('already_started')
                return
            raise ValueError('An owned server is still running')
    used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    if any(int(line.strip()) > 1024 for line in used.splitlines()):
        raise ValueError('GPU not idle; refusing to interfere with another workload')
    command = [str(PYTHON), '-m', 'vllm.entrypoints.openai.api_server', '--model', str(MODELS[variant]),
        '--served-model-name', 'secpo-static-' + variant, '--host', '127.0.0.1', '--port', '28780',
        '--dtype', 'bfloat16', '--max-model-len', '65536', '--gpu-memory-utilization', '0.90',
        '--max-num-seqs', '32', '--max-num-batched-tokens', '4096', '--enforce-eager',
        '--no-enable-prefix-caching', '--generation-config', 'vllm', '--seed', '42',
        '--language-model-only', '--no-enable-log-requests']
    env = dict(os.environ, HF_HOME=str(RUNTIME / 'hf_cache'), HF_HUB_OFFLINE='1',
               HF_HUB_DISABLE_TELEMETRY='1', OMP_NUM_THREADS='12', TOKENIZERS_PARALLELISM='false',
               PATH=str(PYTHON.parent) + os.pathsep + os.environ['PATH'])
    with (ROOT / ('server-' + variant + '.log')).open('a') as log:
        proc = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
    write('server.json', dict(status='starting', variant=variant, pid=proc.pid, command=command, at=now()))


def stop(variant):
    state = json.loads((ROOT / 'server.json').read_text())
    if state['variant'] != variant:
        raise ValueError('Refusing to stop another variant')
    pid = state['pid']
    if not alive(pid):
        return
    observed = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    if b'secpo-static-' + variant.encode() not in observed or str(PYTHON).encode() not in observed:
        raise ValueError('PID ownership mismatch')
    if os.getpgid(pid) != pid:
        raise ValueError('Unexpected process group')
    os.killpg(pid, signal.SIGTERM)
    write('server.json', {**state, 'status':'stopping', 'at':now()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['setup', 'start-setup', 'start', 'stop'])
    parser.add_argument('--variant', choices=MODELS)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.action == 'setup': setup()
    elif args.action == 'start-setup': start_setup()
    elif args.variant is None: parser.error('--variant is required')
    elif args.action == 'start': start(args.variant)
    else: stop(args.variant)
