#!/usr/bin/env python3
"""Queue-only handoff update after user cancellation/replacement of SEP."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import launch_native_static_base as original
from secpo.tinker.evaluation_common import now,write_json

REPLACEMENT=Path('/data2/jaewonc/secpo_evals/qwen38_sep_original_20260927')


def dependency_ready(directory=original.DEPENDENCY,replacement=REPLACEMENT):
    cancellation=original.read(directory/'sep_cancellation.json')
    if cancellation['status']!='deleted':
        raise ValueError('SEP cancellation not confirmed')
    # Keep the original two utility workers and their output files unchanged.
    state=original.read(directory/'supervisor.json')
    unexpected=set(state.get('failed_benchmarks',[]))-{'sep_static'}
    if unexpected:
        raise RuntimeError('A utility benchmark failed; leaving the server untouched')
    if state['status']=='failed' and state.get('message')!='Failed benchmarks: sep_static':
        raise RuntimeError('Unexpected failure in earlier queue')
    for name in ('mmlu_pro','minerva_math'):
        worker=original.read(directory/'secpo'/name/'full_worker.json')
        if worker['status']=='failed':
            raise RuntimeError('Utility worker failed: '+name)
        if worker['status']!='complete':
            return False
        if worker.get('exit_code')!=0 or original.read(directory/'secpo'/name/'progress.json')['status']!='complete':
            raise ValueError('Utility completion metadata inconsistent')
    path=replacement/'supervisor.json'
    if not path.exists():
        return False
    state=original.read(path)
    if state.get('generation_status')!='complete':
        if state['status']=='failed':
            raise RuntimeError('Replacement SEP generation failed; leaving server untouched')
        return False
    directory=replacement/'secpo/sep_static'
    worker=original.read(directory/'generate_worker.json')
    if (worker['status']!='complete' or worker['exit_code']!=0
            or original.read(directory/'progress.json')['status']!='generation_complete'):
        raise ValueError('Replacement SEP generation completion inconsistent')
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--lock-fd',type=int)
    args=parser.parse_args()
    root=original.OUTPUT
    lock=os.fdopen(args.lock_fd,'a+') if args.lock_fd is not None else (root/'queue.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=Path(__file__).resolve()
    identity=dict(source=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        replacement_sep=str(REPLACEMENT),policy='Queue readiness only; all base benchmark settings and sources unchanged.')
    manifest=root/'queue_handoff_override.json'
    if manifest.exists() and original.read(manifest)!=identity:
        raise ValueError('Queue handoff source changed')
    write_json(manifest,identity)
    snapshot=root/'queue_handoff_override.py'
    if not snapshot.exists():
        snapshot.write_bytes(path.read_bytes())
    if args.detach:
        command=[sys.executable,'-u',str(path),'--lock-fd',str(lock.fileno())]
        with (root/'supervisor.log').open('a') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
        write_json(root/'launch_after_sep_replacement.json',dict(pid=process.pid,command=command,at=now()))
        print(json.dumps(dict(status='launched',pid=process.pid,output=str(root))),flush=True)
        return
    original.dependency_ready=dependency_ready
    # Pass a duplicate of the held descriptor; original.main owns that handle.
    sys.argv=[str(ROOT/'scripts/launch_native_static_base.py'),'--lock-fd',str(os.dup(lock.fileno()))]
    original.main()


if __name__=='__main__':
    main()
