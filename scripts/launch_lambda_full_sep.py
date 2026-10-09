#!/usr/bin/env python3
"""Durable queue for full MMLU-Pro, full Minerva Math, and six SEP static attacks."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from secpo.tinker.evaluation_common import now, write_json
from secpo.subprocess_env import build_subprocess_environment

OUTPUT = Path('/data2/jaewonc/secpo_evals/qwen38_full_sep_20260927')
REMOTE = '/lambda/nfs/jaewon/secpo_full_sep_20260927'
HOST = 'ubuntu@209.20.159.207'
PYTHON = ROOT / 'runtime/eval_envs/lm/bin/python'
SSH = ['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=10',
    '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6','-o','StrictHostKeyChecking=yes',
    '-o','UserKnownHostsFile='+str(ROOT/'runtime/lambda_gpqa_known_hosts'),
    '-i',str(ROOT/'runtime/lambda_gpqa_ssh/id_ed25519')]
FILES = ['scripts/launch_lambda_full_sep.py','scripts/lambda_full_sep_remote.py',
    'scripts/evaluate_lambda_full_lm.py','scripts/evaluate_lambda_sep_static.py',
    'configs/lambda_qwen38_lr8e5_full_20260927.json','src/secpo/inference/sep_openrouter.py',
    'src/secpo/inference/lambda_eval.py','src/secpo/tinker/evaluation_common.py',
    'src/secpo/tinker/agent_eval.py','src/secpo/tinker/lm_eval_backend.py',
    'src/secpo/evaluation/sep_static.py','src/secpo/evaluation/sep_judge.py',
    'src/secpo/evaluation/witness.py','src/secpo/subprocess_env.py']


def read(path):
    return json.loads(Path(path).read_text())


def snapshot():
    identity = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES}
    path = OUTPUT / 'source_identity.json'
    if path.exists():
        if read(path) != identity:
            raise ValueError('Launch sources changed after queue started')
        return
    for name in FILES:
        target = OUTPUT / 'code_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    write_json(path, identity)


def remote(command):
    return subprocess.run(SSH+[HOST,shlex.join(command)],check=True,capture_output=True,text=True,timeout=60).stdout


def sync():
    (OUTPUT/'remote').mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-a','--timeout=30','-e',shlex.join(SSH),
        HOST+':'+REMOTE+'/',str(OUTPUT/'remote')+'/'],check=True,timeout=60)


def health():
    import requests
    try:
        with requests.Session() as session:
            session.trust_env=False
            r=session.get('http://127.0.0.1:18782/v1/models',timeout=5)
            return r.ok and [m['id'] for m in r.json()['data']]==['secpo-suite-secpo']
    except (requests.RequestException,ValueError,KeyError):
        return False


def command(benchmark, phase):
    if benchmark=='sep_static':
        return [str(PYTHON),'-u',str(ROOT/'scripts/evaluate_lambda_sep_static.py'),
            '--phase','generate' if phase=='full' else phase,'--concurrency','8']
    return [str(PYTHON),'-u',str(ROOT/'scripts/evaluate_lambda_full_lm.py'),
            '--benchmark',benchmark,'--phase',phase,'--concurrency','12']


def phase(name, stage):
    processes = {}
    env = build_subprocess_environment(overrides=dict(HF_HOME='/data2/jaewonc/huggingface',
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLBACKEND='Agg'))
    for benchmark in ('mmlu_pro','minerva_math','sep_static'):
        directory=OUTPUT/'secpo'/benchmark
        if read(directory/'preflight.json')['status']!='complete':
            raise ValueError('Preflight has not passed')
        if name=='full':
            expected = 'generation_complete' if benchmark=='sep_static' else 'complete'
            if read(directory/'smoke/progress.json')['status']!=expected:
                raise ValueError('Smoke has not passed')
            if (directory/'progress.json').exists() and read(directory/'progress.json')['status']==expected:
                continue
        cmd=command(benchmark,name)
        with (directory/(name+'.log')).open('a') as log:
            process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT)
        record=dict(status='running',pid=process.pid,command=cmd,started_at=now(),phase=name)
        write_json(directory/(name+'_worker.json'),record)
        processes[benchmark]=(process,directory,record)
    last_sync=time.monotonic(); failures=[]
    while processes:
        snapshot()
        for benchmark,(process,directory,record) in list(processes.items()):
            code=process.poll()
            if code is None:
                continue
            active=directory/'smoke' if name=='smoke' else directory
            expected='generation_complete' if benchmark=='sep_static' else 'complete'
            valid=code==0 and (active/'progress.json').exists() and read(active/'progress.json')['status']==expected
            record.update(status='complete' if valid else 'failed',exit_code=code,ended_at=now())
            write_json(directory/(name+'_worker.json'),record)
            if not valid:
                failures.append(benchmark)
            del processes[benchmark]
        if time.monotonic()-last_sync>120:
            sync(); last_sync=time.monotonic()
        stage(name,active_benchmarks=list(processes),failed_benchmarks=failures)
        if processes:
            time.sleep(15)
    if failures:
        raise RuntimeError('Failed benchmarks: '+','.join(failures))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--lock-fd',type=int)
    args=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    lock=os.fdopen(args.lock_fd,'a+') if args.lock_fd is not None else (OUTPUT/'queue.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    snapshot()
    if args.detach:
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--lock-fd',str(lock.fileno())]
        with (OUTPUT/'supervisor.log').open('a') as log:
            process=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
        write_json(OUTPUT/'launch.json',dict(pid=process.pid,command=cmd,at=now()))
        print(json.dumps(dict(status='launched',pid=process.pid,output=str(OUTPUT))),flush=True)
        return
    state=dict(status='running',pid=os.getpid(),host=HOST,started_at=now(),judge_status='awaiting_trial_count_confirmation')
    def stage(name,**values):
        state.update(stage=name,updated_at=now(),**values)
        write_json(OUTPUT/'supervisor.json',state)
    tunnel=None
    try:
        stage('waiting_for_remote_setup')
        deadline=time.monotonic()+3600
        while True:
            sync()
            setup=read(OUTPUT/'remote/setup_status.json')
            if setup['status']=='complete':break
            if setup['status']=='failed':raise RuntimeError('Remote setup failed')
            if time.monotonic()>deadline:raise TimeoutError('Remote setup timeout')
            time.sleep(30)
        if health():raise ValueError('Local tunnel port already serves a model')
        with (OUTPUT/'tunnel.log').open('a') as log:
            tunnel=subprocess.Popen(SSH+['-o','ExitOnForwardFailure=yes','-N','-L',
                '127.0.0.1:18782:127.0.0.1:28782',HOST],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        write_json(OUTPUT/'tunnel.json',dict(pid=tunnel.pid,at=now()))
        stage('starting_model')
        remote(['python3',REMOTE+'/lambda_full_sep_remote.py','start-server'])
        deadline=time.monotonic()+2400
        while not health():
            if tunnel.poll() is not None:raise RuntimeError('SSH tunnel exited')
            if time.monotonic()>deadline:raise TimeoutError('Model startup timed out')
            sync();time.sleep(15)
        sync()
        stage('smoke')
        phase('smoke',stage)
        stage('full')
        phase('full',stage)
        sync()
        state.update(status='generation_and_utility_complete',stage='awaiting_judge_trial_count_confirmation',completed_at=now())
    except Exception as error:
        state.update(status='failed',error_type=type(error).__name__,message=str(error),failed_at=now())
        raise
    finally:
        write_json(OUTPUT/'supervisor.json',state)
        if tunnel is not None:
            tunnel.terminate();tunnel.wait(timeout=15)
        try:sync()
        except Exception:pass


if __name__=='__main__':
    main()
