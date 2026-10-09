#!/usr/bin/env python3
"""Fresh SEP-only queue on the already running lr8 server; no utility changes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_lambda_sep_original import RUN, OUTPUT, snapshot, read
from launch_lambda_full_sep import SSH, HOST
from secpo.subprocess_env import build_subprocess_environment
from secpo.tinker.evaluation_common import now, write_json

PYTHON = ROOT/'runtime/eval_envs/lm/bin/python'


def health():
    import requests
    try:
        with requests.Session() as session:
            session.trust_env=False
            response=session.get('http://127.0.0.1:18784/v1/models',timeout=5)
            return response.ok and [r['id'] for r in response.json()['data']]==['secpo-suite-secpo']
    except (requests.RequestException,ValueError,KeyError):
        return False


def worker(phase,env):
    command=[str(PYTHON),'-u',str(ROOT/'scripts/evaluate_lambda_sep_original.py'),
             '--phase',phase,'--concurrency','8']
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/(phase+'.log')).open('a') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(status='running',pid=process.pid,command=command,phase=phase,started_at=now())
    write_json(OUTPUT/(phase+'_worker.json'),record)
    return process,record


def wait_worker(process,record,phase,stage):
    while process.poll() is None:
        snapshot()
        stage(phase)
        time.sleep(15)
    expected='generation_complete' if phase in ('smoke','generate') else 'complete'
    path=(OUTPUT/'smoke/progress.json' if phase=='smoke' else OUTPUT/'progress.json' if phase=='generate'
          else OUTPUT/'judge_progress.json')
    valid=process.returncode==0 and path.exists() and read(path)['status']==expected
    record.update(status='complete' if valid else 'failed',exit_code=process.returncode,ended_at=now())
    write_json(OUTPUT/(phase+'_worker.json'),record)
    if not valid:
        raise RuntimeError('Original SEP '+phase+' worker failed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--lock-fd',type=int)
    args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    lock=os.fdopen(args.lock_fd,'a+') if args.lock_fd is not None else (RUN/'queue.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    snapshot()
    if args.detach:
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--lock-fd',str(lock.fileno())]
        with (RUN/'supervisor.log').open('a') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
        write_json(RUN/'launch.json',dict(pid=process.pid,command=command,at=now()))
        print(json.dumps(dict(status='launched',pid=process.pid,output=str(RUN))),flush=True)
        return
    state=dict(status='running',pid=os.getpid(),host=HOST,started_at=now(),
        model='SecPO lr=8e-5',judge_policy='original three judges by three trials; unanimous nine YES')
    def stage(name,**kwargs):
        state.update(stage=name,updated_at=now(),**kwargs)
        write_json(RUN/'supervisor.json',state)
    env=build_subprocess_environment(overrides=dict(HF_HOME='/data2/jaewonc/huggingface',
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLBACKEND='Agg'))
    tunnel=None
    try:
        if read(RUN/'judge_probe/results.json')['status']!='complete' or read(OUTPUT/'preflight.json')['status']!='complete':
            raise ValueError('Preflight/probe incomplete')
        if read(Path('/data2/jaewonc/secpo_evals/qwen38_full_sep_20260927/sep_cancellation.json'))['status']!='deleted':
            raise ValueError('Old SEP outputs have not been deleted')
        if health():
            raise ValueError('SEP replacement tunnel port already in use')
        stage('opening_independent_sep_tunnel')
        with (RUN/'tunnel.log').open('a') as log:
            tunnel=subprocess.Popen(SSH+['-o','ExitOnForwardFailure=yes','-N','-L',
                '127.0.0.1:18784:127.0.0.1:28782',HOST],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        write_json(RUN/'tunnel.json',dict(pid=tunnel.pid,at=now()))
        deadline=time.monotonic()+60
        while not health():
            if tunnel.poll() is not None or time.monotonic()>deadline:
                raise RuntimeError('Existing lr8 server unavailable on replacement tunnel')
            time.sleep(2)
        for phase in ('smoke','generate'):
            stage(phase)
            process,record=worker(phase,env)
            wait_worker(process,record,phase,stage)
        # GPU work is complete. The base queue may now switch checkpoints once
        # the two utility workers also finish. Judging uses only OpenRouter.
        stage('judge',generation_status='complete',generation_completed_at=now())
        tunnel.terminate()
        tunnel.wait(timeout=15)
        tunnel=None
        process,record=worker('judge',env)
        wait_worker(process,record,'judge',stage)
        stage('complete',status='complete',completed_at=now())
    except Exception as error:
        stage(state.get('stage','failed'),status='failed',error_type=type(error).__name__,
            message=str(error),failed_at=now())
        raise
    finally:
        if tunnel is not None:
            tunnel.terminate()
            tunnel.wait(timeout=15)


if __name__=='__main__':
    main()
