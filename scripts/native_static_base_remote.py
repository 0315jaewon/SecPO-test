#!/usr/bin/env python3
"""Verify and serve the pinned base checkpoint after the lr8 GPU work finishes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/lambda/nfs/jaewon/secpo_native_base_static_20260927')
PREVIOUS = Path('/lambda/nfs/jaewon/secpo_full_sep_20260927')
RUNTIME = Path('/home/ubuntu/secpo_full_sep_20260927')
PYTHON = RUNTIME/'venv/bin/python'
MODEL = Path('/lambda/nfs/jaewon/secpo_qwen38_eval_20260916/hf_cache/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')
TEMPLATE_SHA = 'bd7892f90fc8b4d1d29feed188b7febdcd8eaac038e3da7b25d5b61ba6a0f3e7'


def read(path):
    return json.loads(Path(path).read_text())


def write(name, obj):
    temporary = ROOT/(name+'.tmp')
    temporary.write_text(json.dumps(obj, indent=2)+'\n')
    temporary.replace(ROOT/name)


def now():
    return datetime.now(timezone.utc).isoformat()


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def verify():
    rows = [json.loads(line) for line in (ROOT/'model_manifest.jsonl').read_text().splitlines()]
    if len(rows) != 32 or sum(r['path'].endswith('.safetensors') for r in rows) != 18:
        raise ValueError('Unexpected base manifest inventory')
    verified = []
    try:
        for row in rows:
            relative = Path(row['path'])
            if relative.is_absolute() or '..' in relative.parts or row['kind'] != 'file':
                raise ValueError('Invalid model manifest path')
            path = MODEL/relative
            # The original HF snapshot uses symlinks into its own blob cache.
            if not path.resolve().is_relative_to(MODEL.parents[1]) or path.stat().st_size != row['bytes']:
                raise ValueError('Base model path or size differs: '+row['path'])
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024*1024), b''):
                    h.update(block)
            if h.hexdigest() != row['sha256']:
                raise ValueError('Base checksum differs: '+row['path'])
            verified.append({**row, 'mtime_ns':path.stat().st_mtime_ns})
            write('model_validation.json', dict(status='verifying', pid=os.getpid(), files=verified, total=len(rows), at=now()))
        write('model_validation.json', dict(status='complete', directory=str(MODEL), files=verified, at=now()))
    except Exception as error:
        write('model_validation.json', dict(status='failed', message=str(error), at=now()))
        raise


def start_verify():
    state = read(ROOT/'model_validation.json') if (ROOT/'model_validation.json').exists() else {}
    if state.get('status') == 'complete' or (state.get('pid') and alive(state['pid'])):
        return
    with (ROOT/'verify.log').open('a') as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'verify'],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(pid=process.pid, status='verification_started')), flush=True)


def stop_owned(path, name, port):
    if not path.exists():
        return
    state = read(path)
    if not alive(state['pid']):
        return
    expected = state['command']
    observed = Path(f'/proc/{state["pid"]}/cmdline').read_bytes().rstrip(b'\0').split(b'\0')
    if (observed != [value.encode() for value in expected]
            or expected[0] != str(PYTHON) or expected[1:3] != ['-m','vllm.entrypoints.openai.api_server']
            or expected[expected.index('--served-model-name')+1] != name
            or expected[expected.index('--port')+1] != port
            or os.getpgid(state['pid']) != state['pid']):
        raise ValueError('Server ownership check failed; refusing to stop it')
    os.killpg(state['pid'], signal.SIGTERM)
    deadline = time.monotonic()+150
    while alive(state['pid']):
        if time.monotonic() > deadline:
            raise TimeoutError('Owned model server did not stop')
        time.sleep(2)
    write('handoff.json' if name=='secpo-suite-secpo' else 'stop.json',
          dict(status='stopped', pid=state['pid'], served_model=name, at=now()))


def start():
    path = ROOT/'server.json'
    if path.exists() and alive(read(path)['pid']):
        raise ValueError('Base server already running')
    validation = read(ROOT/'model_validation.json')
    if validation['status'] != 'complete' or read(PREVIOUS/'setup_status.json')['status'] != 'complete':
        raise ValueError('Preparation incomplete')
    for row in validation['files']:
        stat = (MODEL/row['path']).stat()
        if stat.st_size != row['bytes'] or stat.st_mtime_ns != row['mtime_ns']:
            raise ValueError('Base file changed after verification')
    template = ROOT/'qwen38_native_input.jinja'
    if hashlib.sha256(template.read_bytes()).hexdigest() != TEMPLATE_SHA:
        raise ValueError('Native tool template differs from earlier static runs')
    expected_packages = {'vllm':'0.29.0','torch':'2.13.0','transformers':'5.10.4','flashinfer-python':'0.6.18'}
    packages = json.loads(subprocess.check_output([str(PYTHON),'-m','pip','list','--format=json']))
    installed = {row['name']:row['version'] for row in packages}
    if any(installed.get(k) != v for k,v in expected_packages.items()):
        raise ValueError('Server package versions differ')
    write('packages.json', packages)
    used = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'], text=True)
    if any(int(value)>1024 for value in used.splitlines()):
        raise ValueError('GPU is not idle; refusing second model allocation')
    command = [str(PYTHON),'-m','vllm.entrypoints.openai.api_server','--model',str(MODEL),
        '--served-model-name','secpo-native-base','--host','127.0.0.1','--port','28783',
        '--dtype','bfloat16','--max-model-len','65536','--gpu-memory-utilization','0.90',
        '--max-num-seqs','32','--max-num-batched-tokens','4096','--enforce-eager',
        '--no-enable-prefix-caching','--generation-config','vllm','--seed','42',
        '--language-model-only','--no-enable-log-requests','--enable-auto-tool-choice',
        '--tool-call-parser','qwen3_coder','--reasoning-parser','qwen3','--chat-template',str(template),
        '--default-chat-template-kwargs','{"enable_thinking":false,"preserve_thinking":false}',
        '--override-generation-config','{"min_p":0.0,"repetition_penalty":1.0,"frequency_penalty":0.0}']
    env = dict(os.environ, HF_HOME=str(RUNTIME/'hf_cache'), HF_HUB_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1', OMP_NUM_THREADS='12', TOKENIZERS_PARALLELISM='false',
        PATH=str(PYTHON.parent)+os.pathsep+os.environ['PATH'])
    with (ROOT/'server-base.log').open('a') as log:
        process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
    write('server.json', dict(status='starting', variant='base', pid=process.pid, command=command,
        at=now(), template_sha256=TEMPLATE_SHA, checkpoint_verification=validation))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start-verify','verify','stop-previous','start','stop'))
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.action == 'start-verify': start_verify()
    elif args.action == 'verify': verify()
    elif args.action == 'stop-previous': stop_owned(PREVIOUS/'server.json','secpo-suite-secpo','28782')
    elif args.action == 'stop': stop_owned(ROOT/'server.json','secpo-native-base','28783')
    else: start()
