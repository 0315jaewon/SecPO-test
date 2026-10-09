#!/usr/bin/env python3
"""Durable two-model static-security queue; all evaluation data stays on REDS."""
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from secpo.tinker.evaluation_common import now, write_json
from secpo.inference.static_agent_eval import OUTPUT, VARIANTS, FILES, read

HOST='ubuntu@209.20.159.198'
REMOTE='/lambda/nfs/jaewon/secpo_static_security_20260924'
SSH=['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=10',
     '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6','-o','StrictHostKeyChecking=yes',
     '-o','UserKnownHostsFile='+str(ROOT/'runtime/lambda_gpqa_known_hosts'),
     '-i',str(ROOT/'runtime/lambda_gpqa_ssh/id_ed25519')]


def snapshot():
    identity={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES}
    path=OUTPUT/'source_identity.json'
    if path.exists() and read(path)!=identity:
        raise ValueError('Static experiment sources changed')
    if not path.exists():
        for name in FILES:
            p=OUTPUT/'code_snapshot'/name
            p.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,p)
        write_json(path,identity)


def remote_read(name):
    command=shlex.join(['python3','-c',
        'from pathlib import Path; import sys; p=Path(sys.argv[1]); print(p.read_text() if p.exists() else "{}")',
        REMOTE+'/'+name])
    proc=subprocess.run(SSH+[HOST,command],capture_output=True,text=True,timeout=30)
    if proc.returncode:raise RuntimeError('Remote status check failed')
    return json.loads(proc.stdout)


def remote(action,variant):
    command=shlex.join(['python3',REMOTE+'/lambda_static_remote.py',action,'--variant',variant])
    subprocess.run(SSH+[HOST,command],check=True,timeout=60)


def sync_remote_logs():
    (OUTPUT/'remote').mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-a','--timeout=30','--exclude=results/','-e',shlex.join(SSH),
                    HOST+':'+REMOTE+'/',str(OUTPUT/'remote')+'/'],check=True,timeout=60)


def health():
    import requests
    try:
        with requests.Session() as session:
            session.trust_env=False
            response=session.get('http://127.0.0.1:18780/v1/models',timeout=5)
            return [x['id'] for x in response.json()['data']] if response.ok else []
    except (requests.RequestException,ValueError,KeyError):return []


def wait_server(variant):
    expected=['secpo-static-'+variant]
    deadline=time.monotonic()+1800
    while health()!=expected:
        if time.monotonic()>deadline:raise TimeoutError('Model startup timed out')
        time.sleep(5)
    sync_remote_logs()


def run_phase(variant,phase):
    snapshot()
    pending={}
    env=dict(os.environ,HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',
             OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MPLBACKEND='Agg')
    # No provider keys are needed: benchmarks/tools are local simulations and
    # model inference uses only the private Lambda tunnel.
    for key in ('HF_TOKEN','TINKER_API_KEY','OPENAI_API_KEY','OPENROUTER_API_KEY'):
        env.pop(key,None)
    for benchmark in ('agentdojo','agentdyn'):
        directory=OUTPUT/variant/benchmark
        directory.mkdir(parents=True,exist_ok=True)
        if phase=='full':
            smoke=read(directory/'smoke/results.json')
            if smoke['status']!='complete':raise ValueError('Smoke incomplete')
        command=[str(ROOT/f'runtime/eval_envs/{benchmark}/bin/python'),'-u',
            str(ROOT/'scripts/evaluate_lambda_static_agents.py'),'--benchmark',benchmark,
            '--variant',variant,'--phase',phase]
        with (directory/(phase+'.log')).open('a') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT)
        record=dict(status='running',phase=phase,pid=process.pid,command=command,started_at=now())
        write_json(directory/(phase+'_worker.json'),record)
        pending[benchmark]=(process,directory,record)
    failures=[]
    last_sync=time.monotonic()
    while pending:
        snapshot()
        for benchmark,(process,directory,record) in list(pending.items()):
            code=process.poll()
            if code is None:continue
            result_path=(directory/'preflight.json' if phase=='preflight' else
                         directory/'smoke/results.json' if phase=='smoke' else directory/'results.json')
            valid=code==0 and result_path.exists() and read(result_path).get('status')=='complete'
            record.update(status='complete' if valid else 'failed',exit_code=code,ended_at=now())
            write_json(directory/(phase+'_worker.json'),record)
            if not valid:failures.append(benchmark)
            del pending[benchmark]
        if time.monotonic()-last_sync>120:
            sync_remote_logs()
            last_sync=time.monotonic()
        if pending:time.sleep(5)
    if failures:raise RuntimeError(f'{variant}/{phase} failed: {failures}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--lock-fd',type=int)
    args=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    lock=os.fdopen(args.lock_fd,'a+') if args.lock_fd is not None else (OUTPUT/'supervisor.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    snapshot()
    if args.detach:
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--lock-fd',str(lock.fileno())]
        with (OUTPUT/'supervisor.log').open('a') as log:
            proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
        write_json(OUTPUT/'launch.json',dict(pid=proc.pid,command=command,at=now()))
        print(json.dumps(dict(status='launched',pid=proc.pid,output=str(OUTPUT))),flush=True)
        return
    status=dict(status='running',pid=os.getpid(),host=HOST,variants=VARIANTS,started_at=now())
    def update(stage,**kwargs):
        status.update(status='running',stage=stage,updated_at=now(),**kwargs)
        write_json(OUTPUT/'supervisor.json',status)
        print(json.dumps(status),flush=True)
    tunnel=None
    try:
        update('preflight')
        for variant in VARIANTS:run_phase(variant,'preflight')
        update('waiting_for_environment_and_checkpoint_verification')
        deadline=time.monotonic()+7200
        while True:
            prepared=remote_read('setup_status.json')
            if prepared.get('status')=='complete':break
            if prepared.get('status')=='failed':raise RuntimeError('Remote setup failed')
            if time.monotonic()>deadline:raise TimeoutError('Remote setup exceeded two hours')
            time.sleep(15)
        sync_remote_logs()
        if health():raise ValueError('Local tunnel port already serves a model; refusing overlap')
        with (OUTPUT/'tunnel.log').open('a') as log:
            tunnel=subprocess.Popen(SSH+['-o','ExitOnForwardFailure=yes','-N','-L',
                '127.0.0.1:18780:127.0.0.1:28780',HOST],stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT)
        time.sleep(2)
        if tunnel.poll() is not None:raise RuntimeError('Tunnel failed')
        write_json(OUTPUT/'tunnel.json',dict(pid=tunnel.pid,at=now()))
        for variant in VARIANTS:
            if all((OUTPUT/variant/b/'results.json').exists() and
                   read(OUTPUT/variant/b/'results.json').get('status')=='complete'
                   for b in ('agentdojo','agentdyn')):
                update('using_completed_results',variant=variant)
                continue
            update('starting_model',variant=variant)
            remote('start',variant)
            wait_server(variant)
            update('smoke',variant=variant)
            run_phase(variant,'smoke')
            update('full',variant=variant)
            run_phase(variant,'full')
            update('stopping_owned_model_server',variant=variant)
            remote('stop',variant)
            deadline=time.monotonic()+180
            while health():
                if time.monotonic()>deadline:raise TimeoutError('Owned server did not stop')
                time.sleep(3)
            time.sleep(15)
            sync_remote_logs()
        status.update(status='complete',stage='complete',completed_at=now())
    except Exception as error:
        status.update(status='failed',error_type=type(error).__name__,message=str(error),failed_at=now())
        raise
    finally:
        write_json(OUTPUT/'supervisor.json',status)
        if tunnel is not None:
            tunnel.terminate()
            tunnel.wait(timeout=15)
        try:sync_remote_logs()
        except Exception:pass


if __name__=='__main__':main()
