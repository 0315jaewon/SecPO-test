#!/usr/bin/env python3
"""Change only the waiting base queue's dependency after SEP replacement."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from secpo.tinker.evaluation_common import now,write_json

OUTPUT=Path('/data2/jaewonc/secpo_evals/qwen38_native_base_static_20260927')


def main():
    state=json.loads((OUTPUT/'supervisor.json').read_text())
    launch=json.loads((OUTPUT/'launch.json').read_text())
    if state['stage']!='waiting_for_lr8_generation_and_utility' or state['status']!='queued':
        raise ValueError('Base queue is no longer waiting; refusing to interrupt an evaluation')
    if state['pid']!=launch['pid']:
        raise ValueError('Queue PID identity differs')
    pid=state['pid']
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes().rstrip(b'\0').split(b'\0')
    if cmd!=[v.encode() for v in launch['command']] or str(ROOT/'scripts/launch_native_static_base.py').encode() not in cmd:
        raise ValueError('Base queue ownership mismatch')
    replacement=Path('/data2/jaewonc/secpo_evals/qwen38_sep_original_20260927/supervisor.json')
    if not replacement.exists() or json.loads(replacement.read_text())['status']!='running':
        raise ValueError('Replacement SEP queue is not running')
    write_json(OUTPUT/'queue_before_sep_replacement.json',dict(at=now(),state=state,launch=launch))
    os.kill(pid,signal.SIGTERM)
    deadline=time.monotonic()+15
    while Path(f'/proc/{pid}').exists():
        if Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]=='Z':
            break
        if time.monotonic()>deadline:
            raise TimeoutError('Waiting base queue did not stop')
        time.sleep(.2)
    subprocess.run([str(ROOT/'runtime/eval_envs/lm/bin/python'),'-u',
        str(ROOT/'scripts/launch_native_static_base_after_sep.py'),'--detach'],check=True,cwd=ROOT)


if __name__=='__main__':
    main()
