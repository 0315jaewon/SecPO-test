#!/usr/bin/env python3
"""Queue base AgentDojo/AgentDyn static attacks after the active lr8 inference."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from evaluate_native_static_base import OUTPUT, FILES, ENDPOINT, native
from secpo.subprocess_env import build_subprocess_environment

DEPENDENCY = Path('/data2/jaewonc/secpo_evals/qwen38_full_sep_20260927')
HOST = 'ubuntu@209.20.159.207'
REMOTE = '/lambda/nfs/jaewon/secpo_native_base_static_20260927'
REMOTE_SCRIPT = REMOTE+'/native_static_base_remote.py'
MODEL_MANIFEST = Path('/data2/jaewonc/secpo_backups/lambda_20260919/audit/base.remote_sha256.jsonl')
spec = importlib.util.spec_from_file_location('native_queue_helpers', ROOT/'scripts/launch_native_static.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
helper.OUTPUT, helper.FILES, helper.HOST, helper.REMOTE = OUTPUT, FILES, HOST, REMOTE
read, write_json, now = native.read, native.write_json, native.now


def dependency_ready(directory=DEPENDENCY):
    state = read(directory/'supervisor.json')
    if state['status'] == 'failed':
        raise RuntimeError('Earlier lr8 queue failed; leaving its server untouched')
    if state['status'] != 'generation_and_utility_complete':
        return False
    for benchmark in ('mmlu_pro','minerva_math','sep_static'):
        worker = read(directory/'secpo'/benchmark/'full_worker.json')
        progress = read(directory/'secpo'/benchmark/'progress.json')
        expected = 'generation_complete' if benchmark=='sep_static' else 'complete'
        if worker['status'] != 'complete' or worker['exit_code'] != 0 or progress['status'] != expected:
            raise ValueError('Earlier queue reports completion but a GPU worker is incomplete')
    return True


def health():
    import requests
    try:
        with requests.Session() as session:
            session.trust_env = False
            response = session.get(ENDPOINT+'/models', timeout=5)
            return [m['id'] for m in response.json()['data']] if response.ok else []
    except (requests.RequestException, ValueError, KeyError):
        return []


def phase(name, stage):
    helper.snapshot()
    env = build_subprocess_environment(overrides=dict(HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MPLBACKEND='Agg'))
    pending, failures = {}, []
    for benchmark in ('agentdojo','agentdyn'):
        directory = OUTPUT/'base'/benchmark
        directory.mkdir(parents=True, exist_ok=True)
        if name=='full' and read(directory/'smoke/results.json')['status']!='complete':
            raise ValueError('Static smoke incomplete')
        command = [str(ROOT/f'runtime/eval_envs/{benchmark}/bin/python'), '-u',
            str(ROOT/'scripts/evaluate_native_static_base.py'),'--benchmark',benchmark,'--phase',name]
        with (directory/(name+'.log')).open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT)
        record = dict(status='running', pid=process.pid, command=command, phase=name, started_at=now())
        write_json(directory/(name+'_worker.json'), record)
        pending[benchmark] = (process, directory, record)
    last_sync = time.monotonic()
    while pending:
        helper.snapshot()
        for benchmark, (process, directory, record) in list(pending.items()):
            code = process.poll()
            if code is None:
                continue
            path = directory/('preflight.json' if name=='preflight' else 'smoke/results.json' if name=='smoke' else 'results.json')
            valid = code==0 and path.exists() and read(path)['status']=='complete'
            record.update(status='complete' if valid else 'failed', exit_code=code, ended_at=now())
            write_json(directory/(name+'_worker.json'), record)
            if not valid:
                failures.append(benchmark)
            del pending[benchmark]
        stage(name, active_benchmarks=list(pending), failed_benchmarks=failures)
        if name!='preflight' and time.monotonic()-last_sync>120:
            helper.sync()
            last_sync = time.monotonic()
        if pending:
            time.sleep(10)
    if failures:
        raise RuntimeError(f'Base {name} failed: {failures}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--lock-fd', type=int)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = os.fdopen(args.lock_fd, 'a+') if args.lock_fd is not None else (OUTPUT/'queue.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    helper.snapshot()
    if args.detach:
        command = [sys.executable,'-u',str(Path(__file__).resolve()),'--lock-fd',str(lock.fileno())]
        with (OUTPUT/'supervisor.log').open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
        write_json(OUTPUT/'launch.json', dict(pid=process.pid, command=command, at=now()))
        print(json.dumps(dict(status='launched', pid=process.pid, output=str(OUTPUT))), flush=True)
        return
    state = dict(status='running', pid=os.getpid(), host=HOST, started_at=now(),
        model='base:Qwen/Qwen3.8-27B', dependency=str(DEPENDENCY))
    def stage(name, **values):
        state.update(stage=name, updated_at=now(), **values)
        write_json(OUTPUT/'supervisor.json', state)
    tunnel, dependency_lock = None, None
    try:
        stage('preflight')
        phase('preflight', stage)
        stage('deploying_and_verifying_base')
        rendered = OUTPUT/'qwen38_native_input.jinja'
        rendered.write_text(native.template())
        helper.remote_command(['mkdir','-p',REMOTE])
        subprocess.run(['rsync','-a','-e',shlex.join(helper.SSH),
            str(ROOT/'scripts/native_static_base_remote.py'),str(rendered),HOST+':'+REMOTE+'/'], check=True, timeout=60)
        subprocess.run(['rsync','-a','-e',shlex.join(helper.SSH),str(MODEL_MANIFEST),
            HOST+':'+REMOTE+'/model_manifest.jsonl'], check=True, timeout=60)
        helper.remote_command(['python3',REMOTE_SCRIPT,'start-verify'])
        deadline = time.monotonic()+3600
        while True:
            helper.sync()
            path = OUTPUT/'remote/model_validation.json'
            if path.exists():
                verification = read(path)
                if verification['status']=='complete':
                    break
                if verification['status']=='failed':
                    raise ValueError('Remote base checkpoint verification failed: '+verification['message'])
            if time.monotonic()>deadline:
                raise TimeoutError('Base verification timed out')
            time.sleep(30)
        # Retain the preceding queue lock through the model switch and base run,
        # so a normal restart of that queue cannot send calls to a retired server.
        stage('waiting_for_lr8_generation_and_utility', status='queued')
        dependency_lock = (DEPENDENCY/'queue.lock').open('a+')
        while True:
            helper.snapshot()
            if dependency_ready():
                try:
                    fcntl.flock(dependency_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    if not dependency_ready():
                        raise ValueError('Dependency status changed at handoff')
                    break
            stage('waiting_for_lr8_generation_and_utility', status='queued')
            time.sleep(30)
        stage('switching_to_base', status='running')
        helper.remote_command(['python3',REMOTE_SCRIPT,'stop-previous'])
        time.sleep(10)
        if health():
            raise ValueError('Base tunnel port already serves a model')
        with (OUTPUT/'tunnel.log').open('a') as log:
            tunnel = subprocess.Popen(helper.SSH+['-o','ExitOnForwardFailure=yes','-N','-L',
                '127.0.0.1:18783:127.0.0.1:28783',HOST],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        write_json(OUTPUT/'tunnel.json', dict(pid=tunnel.pid, at=now()))
        helper.remote_command(['python3',REMOTE_SCRIPT,'start'])
        stage('starting_base_server')
        deadline = time.monotonic()+2400
        while health()!=['secpo-native-base']:
            if tunnel.poll() is not None:
                raise RuntimeError('Base SSH tunnel exited')
            if time.monotonic()>deadline:
                raise TimeoutError('Base model startup timed out')
            helper.sync()
            time.sleep(15)
        helper.sync()
        phase('smoke', stage)
        phase('full', stage)
        stage('stopping_owned_base_server')
        helper.remote_command(['python3',REMOTE_SCRIPT,'stop'])
        helper.sync()
        stage('complete', status='complete', completed_at=now())
    except Exception as error:
        stage(state.get('stage','failed'), status='failed', error_type=type(error).__name__,
              message=str(error), failed_at=now())
        raise
    finally:
        if tunnel is not None:
            tunnel.terminate()
            tunnel.wait(timeout=15)
        if dependency_lock is not None:
            dependency_lock.close()


if __name__ == '__main__':
    main()
