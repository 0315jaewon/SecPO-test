#!/usr/bin/env python3
"""Serve unchanged checkpoints with vLLM's built-in native Qwen parsers."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/lambda/nfs/jaewon/secpo_native_static_20260924')
PREPARED = Path('/lambda/nfs/jaewon/secpo_static_security_20260924')
RUNTIME = Path('/home/ubuntu/secpo_static_security_20260924')
PYTHON = RUNTIME / 'venv/bin/python'


def read(path):
    return json.loads(Path(path).read_text())


def write(name, data):
    temporary = ROOT / (name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(ROOT / name)


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def verify_owner(state):
    observed = Path(f'/proc/{state["pid"]}/cmdline').read_bytes().split(b'\0')
    if str(PYTHON).encode() not in observed or ('secpo-native-'+state['variant']).encode() not in observed:
        raise ValueError('Model server PID ownership mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start','stop'))
    parser.add_argument('--variant', choices=('secpo_lr4e5','secpo'), required=True)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    state = read(ROOT / 'server.json') if (ROOT / 'server.json').exists() else None
    if state and alive(state['pid']):
        verify_owner(state)
        if state['variant'] != args.variant:
            raise ValueError('Another owned variant is running')
        if args.action=='start':
            print('already_started')
            return
        if os.getpgid(state['pid']) != state['pid']:
            raise ValueError('Unexpected process group')
        os.killpg(state['pid'], signal.SIGTERM)
        state.update(status='stopping')
        write('server.json', state)
        return
    if args.action=='stop':
        return
    if read(PREPARED / 'setup_status.json')['status']!='complete':
        raise ValueError('Original preparation not complete')
    verification = read(PREPARED / 'model_validation.json')[args.variant]
    if verification['status']!='complete':
        raise ValueError('Checkpoint verification incomplete')
    model = Path(verification['directory'])
    for row in verification['files']:
        if (model / row['path']).stat().st_size != row['bytes']:
            raise ValueError('Previously verified checkpoint changed')
    used = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'], text=True)
    if any(int(x) > 1024 for x in used.splitlines()):
        raise ValueError('GPU is not idle')
    template = ROOT / 'qwen38_native_input.jinja'
    command = [str(PYTHON), '-m','vllm.entrypoints.openai.api_server','--model',str(model),
        '--served-model-name','secpo-native-'+args.variant,'--host','127.0.0.1','--port','28781',
        '--dtype','bfloat16','--max-model-len','65536','--gpu-memory-utilization','0.90',
        '--max-num-seqs','32','--max-num-batched-tokens','4096','--enforce-eager',
        '--no-enable-prefix-caching','--generation-config','vllm','--seed','42',
        '--language-model-only','--no-enable-log-requests','--enable-auto-tool-choice',
        '--tool-call-parser','qwen3_coder','--reasoning-parser','qwen3','--chat-template',str(template),
        '--default-chat-template-kwargs','{"enable_thinking":false,"preserve_thinking":false}',
        '--override-generation-config','{"min_p":0.0,"repetition_penalty":1.0,"frequency_penalty":0.0}']
    env = dict(os.environ, HF_HOME=str(RUNTIME / 'hf_cache'), HF_HUB_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1', OMP_NUM_THREADS='12', TOKENIZERS_PARALLELISM='false',
        PATH=str(PYTHON.parent)+os.pathsep+os.environ['PATH'])
    with (ROOT / ('server-'+args.variant+'.log')).open('a') as log:
        proc = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
    write('server.json', dict(status='starting', variant=args.variant, pid=proc.pid, command=command,
        at=datetime.now(timezone.utc).isoformat(), checkpoint_verification=verification,
        template_sha256=hashlib.sha256(template.read_bytes()).hexdigest()))


if __name__=='__main__':
    main()
