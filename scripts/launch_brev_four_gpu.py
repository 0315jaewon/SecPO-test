#!/usr/bin/env python3
"""Resume unchanged evaluators/caches with four explicitly assigned GPU replicas."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time

import launch_brev_base_full as original
from brev_inference_proxy import ReplicaProxy,response_problem

ROOT, RUN = original.ROOT, original.RUN
CONFIG = ROOT/'configs/brev_resume_four_gpu_20261004.json'
SETTINGS = json.loads(CONFIG.read_text())
RESUME = RUN/SETTINGS['execution_id']
JOBS = SETTINGS['replicas']
REMOTE_HELPER = original.REMOTE+'/code/scripts/brev_four_gpu_remote.py'
FILES = ('scripts/launch_brev_four_gpu.py','scripts/brev_four_gpu_remote.py',
         'scripts/brev_inference_proxy.py','configs/brev_resume_four_gpu_20261004.json')
now, write = original.now, original.write_json


def remote(action,job,timeout=60):
    return json.loads(original.remote(['python3',REMOTE_HELPER,action,'--job',job],timeout=timeout) or '{}')


def freeze():
    if original.identity() != original.read(RUN/'source_identity.json'):
        raise ValueError('Original experiment sources changed; caches must not be mixed')
    hashes = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES}
    path = RESUME/'source_identity.json'
    if path.exists() and original.read(path) != hashes:
        raise ValueError('Four-GPU execution sources changed')
    write(path,hashes)
    for name in FILES:
        destination = RESUME/'code_snapshot'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists(): shutil.copy2(ROOT/name,destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != hashes[name]:
            raise ValueError('Execution snapshot changed')


def preserve_previous():
    if (RESUME/'baseline.json').exists(): return
    paths = [RUN/n for n in ('supervisor.json','launch.json','tunnel.json')]
    paths += list((RUN/'base').glob('*/*.json'))
    paths += list((RUN/'base').glob('*/smoke/progress.json'))
    for path in paths:
        if path.exists():
            destination = RESUME/'before'/path.relative_to(RUN)
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,destination)
    inventory = {}
    counts = {}
    for benchmark in original.SETTINGS['jobs']:
        counts[benchmark] = {}
        for kind in ('completion_cache','rows'):
            files = sorted((RUN/'base'/benchmark/kind).rglob('*.json'))
            counts[benchmark][kind] = len(files)
            for path in files:
                inventory[str(path.relative_to(RUN))] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(RESUME/'existing_cache_sha256.json',inventory)
    write(RESUME/'baseline.json',dict(at=now(),counts=counts,
        prior_failure='Engine cores received SIGTERM at 2026-10-04 07:17:55 UTC; aborted responses rejected.',
        note='Caches may include smoke completions; counts are not final scored question totals.'))
    manifest = dict(at=now(),allocation=SETTINGS,original_source_identity=original.read(RUN/'source_identity.json'),
        execution_source_identity=original.read(RESUME/'source_identity.json'),
        changes=['GPU allocation only','SEP uses two single-GPU replicas via byte-preserving loopback routing'],
        unchanged=['dataset/request identities','model/revision/BF16','thinking off','sampling and seed',
                   '16384 output cap','global concurrency SEP=8, MMLU=12, Math=12','original SEP judges/scoring'],
        cache_reuse='Same logical endpoints and exact cached request identities; no response or score changes.',
        numerical_note='GPU assignment and request batching changed; new generations are not promised bit-identical.')
    write(RESUME/'manifest.json',manifest)
    write(RUN/'active_execution.json',dict(path=str(RESUME),at=now(),allocation=SETTINGS))


def tunnel():
    command = original.SSH+['-o','ExitOnForwardFailure=yes','-N']
    for job in JOBS.values():
        with socket.socket() as probe: probe.bind(('127.0.0.1',job['local_port']))
        command += ['-L',f"127.0.0.1:{job['local_port']}:127.0.0.1:{job['remote_port']}"]
    command.append(original.HOST)
    with (RESUME/'tunnel.log').open('a') as log:
        child = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                 env=original.environment())
    write(RESUME/'tunnel.json',dict(pid=child.pid,command=command,at=now()))
    time.sleep(3)
    if child.poll() is not None: raise RuntimeError('Private SSH tunnel failed')
    return child


def health(job):
    connection = http.client.HTTPConnection('127.0.0.1',JOBS[job]['local_port'],timeout=5)
    try:
        connection.request('GET','/v1/models')
        response = connection.getresponse()
        return response.status == 200 and [r['id'] for r in json.loads(response.read())['data']] == ['secpo-suite-base']
    except (OSError,ValueError,KeyError,http.client.HTTPException): return False
    finally: connection.close()


def ensure_server(job):
    deadline = time.monotonic()+24*3600
    while remote('status',job)['status'] == 'stopped':
        if remote('check-gpu',job)['status'] == 'available':
            remote('start',job); break
        if time.monotonic()>deadline: raise TimeoutError('Assigned GPU remained occupied for 24 hours')
        time.sleep(30)
    deadline = time.monotonic()+1800
    while not health(job):
        if remote('status',job)['status'] == 'stopped':
            raise RuntimeError('Owned model server exited during startup; see '+job+' serve.log')
        if time.monotonic()>deadline: raise TimeoutError('Model startup timed out')
        time.sleep(10)
    # Exercise new servers even when benchmark smoke completions are already cached.
    cached = next((RUN/'base'/JOBS[job]['benchmark']/'completion_cache').glob('*.json'))
    sample = original.read(cached)
    payload = dict(model='secpo-suite-base',prompt=sample['prompt_token_ids'],n=1,
                   return_token_ids=True,skip_special_tokens=False,**{**sample['params'],'max_tokens':4})
    connection = http.client.HTTPConnection('127.0.0.1',JOBS[job]['local_port'],timeout=300)
    try:
        connection.request('POST','/v1/completions',json.dumps(payload),{'Content-Type':'application/json'})
        response = connection.getresponse()
        raw = response.read()
        result = json.loads(raw)
        problem = response_problem(payload,result)
        write(RESUME/'server_smoke'/(job+'.json'),dict(at=now(),status=response.status,
            problem=problem,payload=payload,response=result,scored=False))
        if response.status != 200 or problem:
            raise RuntimeError('Fresh model-server smoke failed: '+job)
    finally: connection.close()
    return dict(status='ready',job=job,gpu=JOBS[job],at=now())


AUDIT_LOCK = threading.Lock()


def audit(record,request,response):
    record['at'] = now()
    with AUDIT_LOCK:
        with (RESUME/'routing.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        if request is not None:
            write(RESUME/'transport_errors'/(record['request_sha256']+'.json'),dict(
                **record,request=request.decode(),response=response.decode(errors='replace')))


def pipeline(benchmark):
    names = [name for name,job in JOBS.items() if job['benchmark'] == benchmark]
    path = RESUME/'pipelines'/(benchmark+'.json')
    state = dict(status='running',benchmark=benchmark,gpu_ids=[JOBS[n]['gpu_id'] for n in names],at=now())
    def stage(value,**kwargs):
        state.update(stage=value,updated_at=now(),**kwargs); write(path,state)
    owned = []
    proxy = None
    try:
        if not original.is_complete(benchmark,'full'):
            stage('starting_or_waiting_for_replicas')
            with ThreadPoolExecutor(max_workers=len(names)) as pool:
                futures = {name:pool.submit(ensure_server,name) for name in names}
                for name,future in futures.items():
                    future.result(); owned.append(name)
            port = original.SETTINGS['jobs'][benchmark]['local_port']
            proxy = ReplicaProxy(port,[dict(name=n,**JOBS[n]) for n in names],audit)
            threading.Thread(target=proxy.serve_forever,daemon=True).start()
            stage('full')
            original.run_worker(benchmark,'full')
            stage('stopping_owned_replicas')
            for name in owned: remote('stop',name,timeout=150)
            owned.clear()
            proxy.shutdown(); proxy.server_close(); proxy = None
        if benchmark == 'sep_static':
            stage('judge')
            original.run_worker(benchmark,'judge')
        stage('complete',status='complete')
        return state
    except Exception as error:
        stage(state.get('stage','failed'),status='failed',error_type=type(error).__name__,message=str(error))
        raise
    finally:
        if proxy is not None: proxy.shutdown(); proxy.server_close()
        # Stop only these recorded replicas; never signal unrelated workloads.
        for name in names:
            try:
                if remote('status',name)['status'] == 'serving': remote('stop',name,timeout=150)
            except Exception as error:
                write(RESUME/'cleanup_errors'/(name+'.json'),dict(at=now(),error_type=type(error).__name__,message=str(error)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--lock-fd',type=int)
    args = parser.parse_args()
    os.umask(0o077)
    RESUME.mkdir(parents=True,exist_ok=True)
    lock = os.fdopen(args.lock_fd,'a+') if args.lock_fd is not None else (RUN/'queue.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    freeze()
    if args.detach:
        command = [str(original.PYTHON),'-u',str(Path(__file__).resolve()),'--lock-fd',str(lock.fileno())]
        with (RESUME/'supervisor.log').open('a') as log:
            child = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                env=original.environment(),start_new_session=True,pass_fds=(lock.fileno(),),cwd=ROOT)
        write(RESUME/'launch.json',dict(pid=child.pid,command=command,at=now()))
        print(json.dumps(dict(status='queued',pid=child.pid,execution=str(RESUME),results=str(RUN))))
        return
    preserve_previous()
    state = dict(status='running',pid=os.getpid(),execution=str(RESUME),gpu_ids=SETTINGS['allowed_gpu_ids'],started_at=now())
    def update(**kwargs):
        state.update(updated_at=now(),**kwargs)
        write(RESUME/'supervisor.json',state); write(RUN/'supervisor.json',state)
    forward = None
    try:
        update(stage='opening_tunnel')
        forward = tunnel()
        update(stage='running_benchmarks')
        failures = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs = {b:pool.submit(pipeline,b) for b in original.SETTINGS['jobs']}
            last_sync = 0
            while jobs:
                freeze()
                for benchmark,future in list(jobs.items()):
                    if future.done():
                        try: future.result()
                        except Exception as error: failures.append(dict(benchmark=benchmark,message=str(error)))
                        del jobs[benchmark]
                update(active_benchmarks=list(jobs),failed_benchmarks=failures)
                if time.monotonic()-last_sync>120:
                    try: original.sync_remote()
                    except Exception as error: update(log_sync_error=type(error).__name__)
                    last_sync = time.monotonic()
                if jobs: time.sleep(15)
        if failures: raise RuntimeError('Independent stages failed; cached progress preserved')
        update(status='complete',stage='complete',completed_at=now())
    except Exception as error:
        update(status='failed',error_type=type(error).__name__,message=str(error))
        raise
    finally:
        if forward is not None:
            forward.terminate(); forward.wait(timeout=20)
        try: original.sync_remote()
        except Exception: pass


if __name__ == '__main__':
    main()
