#!/usr/bin/env python3
"""Detached queue using original SecPO adapters and native vLLM tool parsing."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from secpo.inference.native_static import (OUTPUT, LEGACY, VARIANTS, FILES, read, sha,
    now, write_json, verify_originals, template, served)
from secpo.subprocess_env import build_subprocess_environment

HOST = 'ubuntu@209.20.159.198'
REMOTE = '/lambda/nfs/jaewon/secpo_native_static_20260924'
OLD_REMOTE = '/lambda/nfs/jaewon/secpo_static_security_20260924'
SSH = ['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=10',
    '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6','-o','StrictHostKeyChecking=yes',
    '-o','UserKnownHostsFile='+str(ROOT/'runtime/lambda_gpqa_known_hosts'),
    '-i',str(ROOT/'runtime/lambda_gpqa_ssh/id_ed25519')]


def remote_command(command):
    return subprocess.run(SSH+[HOST,shlex.join(command)],check=True,capture_output=True,text=True,timeout=60).stdout


def snapshot():
    verify_originals()
    identity = {p:sha(ROOT/p) for p in FILES}
    path = OUTPUT/'source_identity.json'
    if path.exists() and read(path)!=identity:
        raise ValueError('Launch sources changed')
    if not path.exists():
        for name in FILES:
            target = OUTPUT/'code_snapshot'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,target)
        write_json(path,identity)


def sync():
    (OUTPUT/'remote').mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-a','--timeout=30','-e',shlex.join(SSH),
                    HOST+':'+REMOTE+'/',str(OUTPUT/'remote')+'/'],check=True,timeout=60)


def health():
    import requests
    try:
        with requests.Session() as session:
            session.trust_env=False
            response=session.get('http://127.0.0.1:18781/v1/models',timeout=5)
            return [m['id'] for m in response.json()['data']] if response.ok else []
    except (requests.RequestException,KeyError,ValueError):
        return []


def stop_legacy():
    """Stop only recorded processes owned by the superseded custom-parser run."""
    retired=[]
    for path in (LEGACY/'secpo_lr4e5/agentdyn/full_worker.json',
                 LEGACY/'secpo_lr4e5/agentdojo/full_worker.json', LEGACY/'supervisor.json', LEGACY/'tunnel.json'):
        if not path.exists():
            continue
        record=read(path); pid=record.get('pid')
        if not pid or not Path(f'/proc/{pid}/cmdline').exists():
            continue
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        if path.name=='tunnel.json':
            owned=b'ssh' in cmd and b'127.0.0.1:18780:127.0.0.1:28780' in cmd and HOST.encode() in cmd
        else:
            script='launch_lambda_static.py' if path.name=='supervisor.json' else 'evaluate_lambda_static_agents.py'
            owned=str(ROOT/'scripts'/script).encode() in cmd
        if not owned:
            if not any(cmd):
                continue
            raise ValueError(f'Legacy PID ownership mismatch: {pid}')
        os.kill(pid,signal.SIGTERM)
        retired.append(dict(pid=pid,record=str(path)))
    # Give the old supervisor a chance to clean up, then stop its owned server.
    time.sleep(3)
    remote_command(['python3',OLD_REMOTE+'/lambda_static_remote.py','stop','--variant','secpo_lr4e5'])
    write_json(OUTPUT/'legacy_replacement.json',dict(at=now(),previous_output=str(LEGACY),
        terminated=retired,policy='old outputs preserved; no old trajectories reused'))
    write_json(LEGACY/'superseded.json',dict(at=now(),replacement_output=str(OUTPUT),
        reason='user authorized original SecPO native adapters instead of the custom parser'))


def phase(variant,name):
    snapshot()
    env=build_subprocess_environment(overrides=dict(HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLBACKEND='Agg'))
    pending={}; failures=[]; last_sync=time.monotonic()
    for benchmark in ('agentdojo','agentdyn'):
        directory=OUTPUT/variant/benchmark
        directory.mkdir(parents=True,exist_ok=True)
        if name=='full' and read(directory/'smoke/results.json')['status']!='complete':
            raise ValueError('Smoke incomplete')
        command=[str(ROOT/f'runtime/eval_envs/{benchmark}/bin/python'),'-u',
            str(ROOT/'scripts/evaluate_native_static.py'),'--benchmark',benchmark,
            '--variant',variant,'--phase',name]
        with (directory/(name+'.log')).open('a') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT)
        record=dict(status='running',pid=process.pid,command=command,started_at=now(),phase=name)
        write_json(directory/(name+'_worker.json'),record)
        pending[benchmark]=(process,directory,record)
    while pending:
        snapshot()
        for benchmark,(process,directory,record) in list(pending.items()):
            code=process.poll()
            if code is None:
                continue
            file='preflight.json' if name=='preflight' else 'smoke/results.json' if name=='smoke' else 'results.json'
            valid=code==0 and (directory/file).exists() and read(directory/file).get('status')=='complete'
            record.update(status='complete' if valid else 'failed',exit_code=code,ended_at=now())
            write_json(directory/(name+'_worker.json'),record)
            if not valid:
                failures.append(benchmark)
            del pending[benchmark]
        if name!='preflight' and time.monotonic()-last_sync>120:
            sync(); last_sync=time.monotonic()
        if pending:
            time.sleep(5)
    if failures:
        raise RuntimeError(f'{variant}/{name} failed: {failures}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--replace-legacy',action='store_true')
    parser.add_argument('--lock-fd',type=int)
    args=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    lock=os.fdopen(args.lock_fd,'a+') if args.lock_fd is not None else (OUTPUT/'supervisor.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    snapshot()
    if args.detach:
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--lock-fd',str(lock.fileno())]
        if args.replace_legacy:
            command.append('--replace-legacy')
        with (OUTPUT/'supervisor.log').open('a') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
        write_json(OUTPUT/'launch.json',dict(pid=process.pid,command=command,at=now()))
        print(json.dumps(dict(status='launched',pid=process.pid,output=str(OUTPUT))),flush=True)
        return
    state=dict(status='running',pid=os.getpid(),host=HOST,started_at=now(),variants=VARIANTS)
    def update(stage,**values):
        state.update(stage=stage,updated_at=now(),**values)
        write_json(OUTPUT/'supervisor.json',state)
        print(json.dumps(state),flush=True)
    tunnel=None
    try:
        update('preflight')
        for variant in VARIANTS:
            phase(variant,'preflight')
        update('deploying_native_server_configuration')
        rendered=OUTPUT/'qwen38_native_input.jinja'
        rendered.write_text(template())
        remote_command(['mkdir','-p',REMOTE])
        subprocess.run(['rsync','-a','-e',shlex.join(SSH),str(ROOT/'scripts/native_static_remote.py'),
                        str(rendered),HOST+':'+REMOTE+'/'],check=True,timeout=60)
        if args.replace_legacy:
            update('retiring_custom_parser_queue')
            stop_legacy()
        if health():
            raise ValueError('Native tunnel port already serves a model')
        with (OUTPUT/'tunnel.log').open('a') as log:
            tunnel=subprocess.Popen(SSH+['-o','ExitOnForwardFailure=yes','-N','-L',
                '127.0.0.1:18781:127.0.0.1:28781',HOST],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        time.sleep(3)
        if tunnel.poll() is not None:
            raise RuntimeError('Native tunnel failed')
        write_json(OUTPUT/'tunnel.json',dict(pid=tunnel.pid,at=now()))
        for variant in VARIANTS:
            if all((OUTPUT/variant/b/'results.json').exists() and read(OUTPUT/variant/b/'results.json')['status']=='complete'
                   for b in ('agentdojo','agentdyn')):
                continue
            update('starting_model',variant=variant)
            time.sleep(15)
            remote_command(['python3',REMOTE+'/native_static_remote.py','start','--variant',variant])
            deadline=time.monotonic()+1800
            last_sync=time.monotonic()
            while health()!=[served(variant)]:
                if time.monotonic()>deadline:
                    raise TimeoutError('Native model startup exceeded 30 minutes')
                if time.monotonic()-last_sync>60:
                    sync(); last_sync=time.monotonic()
                    command=['python3','-c', 'import json,os,sys; d=json.load(open(sys.argv[1])); os.kill(d["pid"],0)',REMOTE+'/server.json']
                    remote_command(command)
                time.sleep(5)
            sync()
            update('smoke',variant=variant)
            phase(variant,'smoke')
            update('full',variant=variant)
            phase(variant,'full')
            update('stopping_owned_model',variant=variant)
            remote_command(['python3',REMOTE+'/native_static_remote.py','stop','--variant',variant])
            deadline=time.monotonic()+180
            while health():
                if time.monotonic()>deadline:
                    raise TimeoutError('Model server did not stop')
                time.sleep(3)
            sync()
        state.update(status='complete',stage='complete',completed_at=now())
    except Exception as error:
        state.update(status='failed',error_type=type(error).__name__,message=str(error),failed_at=now())
        raise
    finally:
        write_json(OUTPUT/'supervisor.json',state)
        if tunnel is not None:
            tunnel.terminate(); tunnel.wait(timeout=15)
        try:
            sync()
        except Exception:
            pass


if __name__=='__main__':
    main()
