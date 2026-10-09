#!/usr/bin/env python3
"""Durable REDS queue; Brev inference is restricted to GPUs 5, 6, and 7."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'src'))
from evaluate_brev_base import SETTINGS, RUN, SOURCE_FILES, snapshot, read, identity
from secpo.subprocess_env import build_subprocess_environment
from secpo.tinker.evaluation_common import now, write_json

PYTHON = ROOT/'runtime/eval_envs/lm/bin/python'
REMOTE = SETTINGS['remote_root']
HOST = 'nvidia@global.prd.ga.run.brev.nvidia.com'
SSH = ['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=15',
       '-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3','-o','StrictHostKeyChecking=yes',
       '-o','HostKeyAlgorithms=ssh-ed25519','-o','UserKnownHostsFile='+str(ROOT/'runtime/brev_v6r091tdo_known_hosts'),
       '-i',str(ROOT/'runtime/lambda_gpqa_ssh/id_ed25519'),'-p','26865']
REMOTE_HELPER = REMOTE+'/code/scripts/brev_base_remote.py'


def remote(command, timeout=60):
    return subprocess.run(SSH+[HOST,shlex.join(command)],check=True,capture_output=True,
                          text=True,timeout=timeout).stdout


def environment():
    return build_subprocess_environment(overrides=dict(HF_HOME='/data2/jaewonc/huggingface',
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        MPLBACKEND='Agg',CUDA_VISIBLE_DEVICES=''))


def deploy_snapshot():
    """Copy exact tracked/untracked research sources, never runtime credentials."""
    artifact = RUN/'repository_snapshot.tar.gz'
    paths = []
    for folder in ('scripts','src','configs','docs','tests','third_party'):
        for path in sorted((ROOT/folder).rglob('*')):
            if path.is_file() and not path.is_symlink() and not any(
                    p in ('__pycache__','.pytest_cache','.git') for p in path.parts):
                if path.suffix not in ('.pyc','.env','.pem','.key'):
                    paths.append(path)
    paths.extend(ROOT/name for name in ('README.md','pyproject.toml','release_manifest.json','.gitignore'))
    inventory = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    if not artifact.exists():
        with tarfile.open(artifact,'w:gz') as archive:
            for path in paths:
                archive.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
        write_json(RUN/'repository_snapshot.json',dict(commit=subprocess.check_output(
            ['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),files=inventory,
            note='Includes current local wrappers; credential/runtime/output directories excluded.',at=now()))
    elif read(RUN/'repository_snapshot.json')['files'] != inventory:
        raise ValueError('Repository snapshot sources changed; do not overwrite the run')
    source_snapshot = str(artifact)
    subprocess.run(['rsync','-a','--timeout=60','-e',shlex.join(SSH),source_snapshot,
                    HOST+':'+REMOTE+'/repository_snapshot.tar.gz'],check=True,timeout=180)
    remote(['mkdir','-p',REMOTE+'/repository_snapshot'])
    remote(['tar','-xzf',REMOTE+'/repository_snapshot.tar.gz','-C',REMOTE+'/repository_snapshot'])
    # The running helper lives in a separate minimal code tree; never replace it.


def sync_remote():
    destination = RUN/'remote'
    destination.mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-a','--timeout=60','--exclude=venv','--exclude=download_venv',
        '--exclude=venv310','--exclude=download_venv310','--exclude=uv_cache','--exclude=python',
        '--exclude=models','--exclude=hf_cache','--exclude=pip_cache','--exclude=cache',
        '--exclude=tmp','--exclude=repository_snapshot','--exclude=repository_snapshot.tar.gz',
        '-e',shlex.join(SSH),HOST+':'+REMOTE+'/',str(destination)+'/'],check=True,timeout=180)


def health(benchmark):
    import requests
    endpoint = f"http://127.0.0.1:{SETTINGS['jobs'][benchmark]['local_port']}/v1/models"
    try:
        with requests.Session() as session:
            session.trust_env = False
            response = session.get(endpoint,timeout=5)
            return response.ok and [r['id'] for r in response.json()['data']] == ['secpo-suite-base']
    except (requests.RequestException,ValueError,KeyError):
        return False


def worker_command(benchmark, phase):
    return [str(PYTHON),'-u',str(ROOT/'scripts/evaluate_brev_base.py'),
            '--benchmark',benchmark,'--phase',phase,'--concurrency',str(SETTINGS['jobs'][benchmark]['concurrency'])]


def is_complete(benchmark,phase):
    directory = RUN/'base'/benchmark
    if phase == 'probe': path,expected = RUN/'judge_probe/results.json','complete'
    elif phase == 'preflight': path,expected = directory/'preflight.json','complete'
    elif phase == 'judge': path,expected = directory/'judge_progress.json','complete'
    else:
        path = (directory/'smoke' if phase=='smoke' else directory)/'progress.json'
        expected = 'generation_complete' if benchmark=='sep_static' else 'complete'
    return path.exists() and read(path).get('status') == expected


def run_worker(benchmark,phase):
    directory = RUN/'base'/benchmark
    directory.mkdir(parents=True,exist_ok=True)
    if is_complete(benchmark,phase):
        return dict(status='complete',benchmark=benchmark,phase=phase,reused_completed_stage=True)
    command = worker_command(benchmark,phase)
    with (directory/(phase+'.log')).open('a') as log:
        process = subprocess.Popen(command,cwd=ROOT,env=environment(),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record = dict(status='running',pid=process.pid,benchmark=benchmark,phase=phase,
                  command=command,started_at=now())
    write_json(directory/(phase+'_worker.json'),record)
    while process.poll() is None:
        # Never silently continue with changed launch/scoring sources.
        try:
            if identity() != read(RUN/'source_identity.json'):
                raise ValueError('Frozen experiment sources changed')
        except Exception:
            process.terminate()
            process.wait(timeout=60)
            raise
        time.sleep(15)
    valid = process.returncode == 0 and is_complete(benchmark,phase)
    record.update(status='complete' if valid else 'failed',exit_code=process.returncode,ended_at=now())
    write_json(directory/(phase+'_worker.json'),record)
    if not valid:
        raise RuntimeError(benchmark+' '+phase+' failed; partial responses/votes are preserved')
    return record


def server_stage(benchmark):
    directory = RUN/'base'/benchmark
    state = dict(benchmark=benchmark,status='running',gpu_id=SETTINGS['jobs'][benchmark]['gpu_id'],started_at=now())
    def update(stage,**kwargs):
        state.update(stage=stage,updated_at=now(),**kwargs)
        write_json(directory/'pipeline.json',state)
    started = False
    try:
        if not is_complete(benchmark,'full'):
            update('waiting_for_assigned_gpu')
            deadline = time.monotonic()+24*3600
            while json.loads(remote(['python3',REMOTE_HELPER,'check-gpu','--benchmark',benchmark]))['status'] != 'available':
                if time.monotonic()>deadline:
                    raise TimeoutError('Assigned GPU remained occupied for 24 hours; no other process was stopped')
                time.sleep(30)
            update('cuda_smoke')
            remote(['python3',REMOTE_HELPER,'validate-cuda','--benchmark',benchmark],timeout=600)
            update('starting_gpu_server')
            remote(['python3',REMOTE_HELPER,'start-server','--benchmark',benchmark])
            started = True
            deadline = time.monotonic()+2400
            while not health(benchmark):
                if time.monotonic()>deadline:
                    raise TimeoutError('Model startup timed out')
                time.sleep(15)
            write_json(directory/'server_ready.json',dict(status='ready',at=now(),gpu=SETTINGS['jobs'][benchmark]))
            update('smoke')
            run_worker(benchmark,'smoke')
            update('full')
            run_worker(benchmark,'full')
            update('stopping_owned_server')
            remote(['python3',REMOTE_HELPER,'stop','--benchmark',benchmark],timeout=150)
            started = False
        if benchmark == 'sep_static':
            update('judge',generation_status='complete')
            run_worker(benchmark,'judge')
        update('complete',status='complete',completed_at=now())
        return state
    except Exception as error:
        update(state.get('stage','failed'),status='failed',error_type=type(error).__name__,
               message=str(error),failed_at=now())
        if started:
            try:
                remote(['python3',REMOTE_HELPER,'stop','--benchmark',benchmark],timeout=150)
            except Exception as stop_error:
                update(state['stage'],cleanup_error=type(stop_error).__name__)
        raise


def open_tunnel():
    for job in SETTINGS['jobs'].values():
        with socket.socket() as probe:
            probe.bind(('127.0.0.1',job['local_port']))
    command = SSH+['-o','ExitOnForwardFailure=yes','-N']
    for job in SETTINGS['jobs'].values():
        command.extend(['-L',f"127.0.0.1:{job['local_port']}:127.0.0.1:{job['remote_port']}"])
    command.append(HOST)
    with (RUN/'tunnel.log').open('a') as log:
        process = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    write_json(RUN/'tunnel.json',dict(pid=process.pid,command=command,at=now()))
    time.sleep(3)
    if process.poll() is not None:
        raise RuntimeError('SSH forwarding failed')
    return process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--lock-fd',type=int)
    args = parser.parse_args()
    os.umask(0o077)
    RUN.mkdir(parents=True,exist_ok=True)
    lock = os.fdopen(args.lock_fd,'a+') if args.lock_fd is not None else (RUN/'queue.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    snapshot()
    if args.detach:
        command = [str(PYTHON),'-u',str(Path(__file__).resolve()),'--lock-fd',str(lock.fileno())]
        with (RUN/'supervisor.log').open('a') as log:
            process = subprocess.Popen(command,cwd=ROOT,env=environment(),stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
        write_json(RUN/'launch.json',dict(pid=process.pid,command=command,at=now()))
        print(json.dumps(dict(status='queued',pid=process.pid,output=str(RUN))),flush=True)
        return
    state = dict(status='running',pid=os.getpid(),host=HOST,started_at=now(),model='Qwen3.8-27B base',
        allowed_gpu_ids=[4,5,6,7],assigned_gpu_ids=sorted(j['gpu_id'] for j in SETTINGS['jobs'].values()),
        judge_policy='original three models x three trials; all nine YES')
    def stage(name,**kwargs):
        state.update(stage=name,updated_at=now(),**kwargs)
        write_json(RUN/'supervisor.json',state)
        print(json.dumps(state),flush=True)
    tunnel = None
    try:
        stage('preflight_and_judge_probe')
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(run_worker,name,'preflight') for name in SETTINGS['jobs']]
            futures.append(pool.submit(run_worker,'sep_static','probe'))
            for future in futures:
                future.result()
        stage('preserving_remote_repository_snapshot')
        deploy_snapshot()
        stage('waiting_for_remote_model_and_environment')
        deadline = time.monotonic()+6*3600
        while True:
            sync_remote()
            status = read(RUN/'remote/setup_status.json')
            if status['status']=='complete': break
            if status['status']=='failed':
                raise RuntimeError('Isolated remote preparation failed; see remote/setup.log')
            if time.monotonic()>deadline:
                raise TimeoutError('Remote preparation timed out')
            stage('waiting_for_remote_model_and_environment',remote_setup=status)
            time.sleep(30)
        stage('opening_private_inference_tunnels')
        tunnel = open_tunnel()
        stage('running_benchmarks',active_benchmarks=list(SETTINGS['jobs']))
        failures = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs = {name:pool.submit(server_stage,name) for name in SETTINGS['jobs']}
            last_sync = time.monotonic()
            while jobs:
                for name,future in list(jobs.items()):
                    if not future.done(): continue
                    try: future.result()
                    except Exception as error:
                        failures.append(dict(benchmark=name,error_type=type(error).__name__,message=str(error)))
                    del jobs[name]
                if time.monotonic()-last_sync>120:
                    try: sync_remote()
                    except (subprocess.SubprocessError,OSError):
                        stage('running_benchmarks',last_log_sync_error_at=now())
                    last_sync=time.monotonic()
                stage('running_benchmarks',active_benchmarks=list(jobs),failed_benchmarks=failures)
                if jobs: time.sleep(15)
        sync_remote()
        if failures:
            raise RuntimeError('One or more benchmark stages failed; independent completed results retained')
        stage('complete',status='complete',completed_at=now())
    except Exception as error:
        stage(state.get('stage','failed'),status='failed',error_type=type(error).__name__,
              message=str(error),failed_at=now())
        raise
    finally:
        if tunnel is not None:
            tunnel.terminate()
            tunnel.wait(timeout=20)


if __name__ == '__main__':
    main()
