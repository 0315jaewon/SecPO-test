#!/usr/bin/env python3
"""Credential-free, instance-local preparation for the thinking-on experiment."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/ubuntu/secpo_thinking_20260923')
MODEL_NAME = 'secpo-thinking-base-20260923'


def write(name, value):
    path = ROOT / name
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def setup():
    write('setup_status.json', dict(status='installing', at=stamp()))
    subprocess.run([sys.executable, '-m', 'venv', str(ROOT / 'venv')], check=True)
    python = str(ROOT / 'venv/bin/python')
    subprocess.run([python, '-m', 'pip', 'install', '--upgrade', 'pip==26.2.1'], check=True)
    subprocess.run([python, '-m', 'pip', 'install', 'vllm==0.29.0',
                    'transformers==5.10.4', 'torch==2.13.0',
                    'tokenizers==0.22.2', 'flashinfer-python==0.6.18'], check=True)
    packages = json.loads(subprocess.check_output([python, '-m', 'pip', 'list', '--format=json']))
    write('packages.json', packages)
    write('setup_status.json', dict(status='complete', at=stamp(), python=python))


def verify():
    checked = []
    for line in (ROOT / 'SHA256SUMS.models').read_text().splitlines():
        expected, name = line.split('  ', 1)
        if not name.startswith('models/base/'):
            continue
        path = ROOT / name
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        if h.hexdigest() != expected:
            raise ValueError('Model checksum differs: ' + name)
        checked.append(dict(path=name, sha256=expected, bytes=path.stat().st_size))
    if len([r for r in checked if r['path'].endswith('.safetensors')]) != 18:
        raise ValueError('Expected all 18 base-model shards')
    write('model_validation.json', dict(status='complete', at=stamp(), files=checked))


def server():
    for name in ('setup_status.json', 'model_validation.json'):
        if json.loads((ROOT / name).read_text()).get('status') != 'complete':
            raise ValueError('Preparation is incomplete')
    path = ROOT / 'server.json'
    if path.exists():
        old = json.loads(path.read_text())
        try:
            os.kill(old['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('Recorded server process is still alive; do not duplicate it')
    python = str(ROOT / 'venv/bin/python')
    command = [python, '-m', 'vllm.entrypoints.openai.api_server',
        '--model', str(ROOT / 'models/base'), '--served-model-name', MODEL_NAME,
        '--host', '127.0.0.1', '--port', '28680', '--dtype', 'bfloat16',
        '--max-model-len', '65536', '--gpu-memory-utilization', '0.90',
        '--max-num-seqs', '16', '--max-num-batched-tokens', '4096', '--enforce-eager',
        '--no-enable-prefix-caching', '--generation-config', 'vllm', '--seed', '42',
        '--language-model-only', '--no-enable-log-requests']
    env = dict(os.environ, HF_HOME=str(ROOT / 'hf_cache'), HF_HUB_OFFLINE='1',
               HF_HUB_DISABLE_TELEMETRY='1', OMP_NUM_THREADS='12', TOKENIZERS_PARALLELISM='false',
               PATH=str(ROOT / 'venv/bin') + os.pathsep + os.environ['PATH'])
    with (ROOT / 'server.log').open('a') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write('server.json', dict(status='starting', pid=process.pid, at=stamp(), command=command))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['setup', 'verify', 'start-server'])
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    try:
        {'setup': setup, 'verify': verify, 'start-server': server}[args.action]()
    except Exception as error:
        write(args.action + '_failure.json', dict(status='failed', at=stamp(), error_type=type(error).__name__))
        raise
