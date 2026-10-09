#!/usr/bin/env python3
"""Resume SEP at global concurrency 1024, preserving all evaluation identities."""
import argparse
import asyncio
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import threading
import time

import launch_brev_sep_resume as prior
from brev_inference_proxy import ReplicaProxy, response_problem

previous = prior.previous
original = previous.original
ROOT, RUN = prior.ROOT, prior.RUN
CONFIG = ROOT/'configs/brev_sep_concurrency1024_20261008.json'
SETTINGS = json.loads(CONFIG.read_text())
EXECUTION = RUN/SETTINGS['execution_id']
FILES = (*prior.FILES,'scripts/launch_brev_sep_batched.py',
         'scripts/evaluate_brev_sep_batched.py','scripts/brev_sep_batched_remote.py',
         'configs/brev_sep_concurrency1024_20261008.json')
ORIGINAL_WORKER_COMMAND = original.worker_command
ORIGINAL_RUN_WORKER = original.run_worker


class BatchReplicaProxy(ReplicaProxy):
    # Avoid the stdlib server's default five-connection listen backlog under load.
    request_queue_size = 2048


class TunnelGroup:
    def __init__(self, processes):
        self.processes = processes

    def terminate(self):
        for process in self.processes:
            if process.poll() is None:
                process.terminate()

    def wait(self, timeout):
        deadline = time.monotonic()+timeout
        for process in self.processes:
            process.wait(timeout=max(0.1,deadline-time.monotonic()))


def open_replica_tunnels():
    # One SSH process cannot carry 1024 simultaneous forwarded connections on
    # this host: the initial unscored probe hit its channel/fd limit at ~1017.
    # Keep the same private ports and route each replica through its own tunnel.
    processes, records = [], {}
    try:
        for name, job in SETTINGS['replicas'].items():
            with socket.socket() as probe:
                probe.bind(('127.0.0.1',job['local_port']))
            command = original.SSH+['-o','ExitOnForwardFailure=yes','-N','-L',
                f"127.0.0.1:{job['local_port']}:127.0.0.1:{job['remote_port']}",original.HOST]
            with (EXECUTION/(name+'_tunnel.log')).open('a') as log:
                process = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,
                    stderr=subprocess.STDOUT,env=original.environment())
            processes.append(process)
            records[name] = dict(pid=process.pid,command=command,at=previous.now())
        previous.write(EXECUTION/'tunnel.json',dict(tunnels=records,at=previous.now()))
        time.sleep(3)
        if any(process.poll() is not None for process in processes):
            raise RuntimeError('Per-replica private SSH tunnel failed')
        return TunnelGroup(processes)
    except BaseException:
        group = TunnelGroup(processes)
        group.terminate(); group.wait(timeout=20)
        raise


def worker_command(benchmark, phase):
    if benchmark == 'sep_static' and phase == 'full':
        return [str(original.PYTHON),'-u',str(ROOT/'scripts/evaluate_brev_sep_batched.py'),
                '--benchmark',benchmark,'--phase',phase,
                '--concurrency',str(SETTINGS['generation_concurrency'])]
    return ORIGINAL_WORKER_COMMAND(benchmark,phase)


async def capacity_probe():
    """Short unscored capacity tests; never enter the evaluation completion cache."""
    import aiohttp
    sample = original.read(next((RUN/'base/sep_static/completion_cache').glob('*.json')))
    payload = dict(model='secpo-suite-base',prompt=sample['prompt_token_ids'],n=1,
                   return_token_ids=True,skip_special_tokens=False,
                   **{**sample['params'],'max_tokens':SETTINGS['capacity_probe_max_tokens']})
    endpoint = 'http://127.0.0.1:19440/v1/completions'
    for concurrency in SETTINGS['capacity_probe_stages']:
        started = time.monotonic()
        previous.write(EXECUTION/'capacity_probe.json',dict(status='running',scored=False,
            at=previous.now(),concurrency=concurrency,max_tokens=payload['max_tokens']))
        connector = aiohttp.TCPConnector(limit=concurrency)
        timeout = aiohttp.ClientTimeout(total=600,connect=60)
        async with aiohttp.ClientSession(connector=connector,timeout=timeout,trust_env=False) as session:
            async def one():
                try:
                    async with session.post(endpoint,json=payload) as response:
                        if response.status != 200:
                            return dict(problem='http_status',http_status=response.status)
                        result = await response.json()
                    return dict(problem=response_problem(payload,result),
                                tokens=result.get('usage',{}).get('completion_tokens',0))
                except Exception as error:
                    return dict(problem=type(error).__name__)
            outcomes = await asyncio.gather(*(one() for _ in range(concurrency)))
        problems = Counter(row['problem'] for row in outcomes if row['problem'])
        record = dict(status='failed' if problems else 'complete',scored=False,
            at=previous.now(),concurrency=concurrency,max_tokens=payload['max_tokens'],
            successes=sum(row['problem'] is None for row in outcomes),problems=dict(problems),
            elapsed_seconds=time.monotonic()-started,
            generated_tokens=sum(row.get('tokens',0) for row in outcomes))
        previous.write(EXECUTION/'capacity_probes'/f'{concurrency}.json',record)
        previous.write(EXECUTION/'capacity_probe.json',record)
        if problems:
            raise RuntimeError('Unscored concurrency capacity probe failed')


def run_worker(benchmark, phase):
    if benchmark == 'sep_static' and phase == 'full' and not original.is_complete(benchmark,phase):
        asyncio.run(capacity_probe())
    return ORIGINAL_RUN_WORKER(benchmark,phase)


def configure():
    prior.CONFIG = CONFIG
    prior.SETTINGS = SETTINGS
    prior.EXECUTION = EXECUTION
    prior.FILES = FILES
    prior.configure()
    previous.REMOTE_HELPER = original.REMOTE+'/code/scripts/brev_sep_batched_remote.py'
    previous.ReplicaProxy = BatchReplicaProxy
    previous.tunnel = open_replica_tunnels
    original.worker_command = worker_command
    original.run_worker = run_worker


def baseline():
    path = EXECUTION/'baseline.json'
    if path.exists():
        return original.read(path)
    sep = RUN/'base/sep_static'
    inventory = {str(p.relative_to(RUN)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for name in ('rows','completion_cache','judgments')
                 for p in (sep/name).rglob('*.json')}
    previous.write(EXECUTION/'existing_cache_sha256.json',inventory)
    for source in [RUN/'supervisor.json',RUN/'active_execution.json',sep/'progress.json',sep/'full_worker.json']:
        if source.exists():
            target = EXECUTION/'before'/source.relative_to(RUN)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
    record = dict(at=previous.now(),saved_sep_rows=sum(1 for _ in (sep/'rows').rglob('*.json')),
        utility_results_sha256=prior.utility_identity(),
        pause_reason='User requested concurrency 1024; original concurrency-8 worker deliberately stopped with SIGTERM',
        old_worker_pid=2047055,old_supervisor_pid=2033202)
    previous.write(path,record)
    previous.write(EXECUTION/'manifest.json',dict(at=previous.now(),allocation=SETTINGS,
        baseline=record,benchmark='sep_static',total=54960,
        original_source_identity=original.read(RUN/'source_identity.json'),
        execution_source_identity=original.read(EXECUTION/'source_identity.json'),
        global_generation_concurrency=SETTINGS['generation_concurrency'],
        judge_concurrency=SETTINGS['judge_concurrency'],
        max_output_tokens=16384,enable_thinking=False,
        unchanged=['Original six attacks and all 9160 examples','model and base revision, BF16',
                   'prompt rendering, sampling, seed, 16384 cap','cache keys and saved responses',
                   'original witness check, original three judges x three votes, judge concurrency 8'],
        changes=['generation concurrency 8 -> 1024','per-GPU max-num-seqs 32 -> 256',
                 'per-GPU max-num-batched-tokens 4096 -> 8192','proxy listen backlog 5 -> 2048',
                 'one private SSH tunnel per replica to avoid the single-tunnel connection limit'],
        numerical_note='New stochastic generations are not promised bit-identical after batching changes.',
        capacity_note='1024 is the global outstanding-request limit; GPU admission is dynamically memory-limited.',
        probes='Unscored short completions; never saved into evaluation caches'))
    previous.write(RUN/'active_execution.json',dict(path=str(EXECUTION),at=previous.now(),allocation=SETTINGS))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--lock-fd',type=int)
    args = parser.parse_args()
    configure()
    os.umask(0o077)
    EXECUTION.mkdir(parents=True,exist_ok=True)
    lock = os.fdopen(args.lock_fd,'a+') if args.lock_fd is not None else (RUN/'queue.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous.freeze()
    if args.detach:
        command = [str(original.PYTHON),'-u',str(Path(__file__).resolve()),'--lock-fd',str(lock.fileno())]
        with (EXECUTION/'supervisor.log').open('a') as log:
            process = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                env=original.environment(),cwd=ROOT,start_new_session=True,pass_fds=(lock.fileno(),))
        previous.write(EXECUTION/'launch.json',dict(pid=process.pid,command=command,at=previous.now()))
        print(json.dumps(dict(status='queued',pid=process.pid,execution=str(EXECUTION))))
        return
    saved = baseline()
    state = dict(status='running',pid=os.getpid(),execution=str(EXECUTION),benchmark='sep_static',
        gpu_ids=SETTINGS['allowed_gpu_ids'],saved_sep_rows=saved['saved_sep_rows'],
        generation_concurrency=SETTINGS['generation_concurrency'],judge_concurrency=8,
        started_at=previous.now(),completed_utilities_unchanged=True)
    def update(**kwargs):
        state.update(updated_at=previous.now(),**kwargs)
        previous.write(EXECUTION/'supervisor.json',state)
        previous.write(RUN/'supervisor.json',state)
    forward = None
    try:
        for attempt in range(1,SETTINGS['max_transport_attempts']+1):
            started = previous.now()
            update(stage='opening_tunnel',attempt=attempt,attempt_started_at=started)
            previous.freeze()
            forward = previous.tunnel()
            update(stage='sep_pipeline')
            try:
                previous.pipeline('sep_static')
                break
            except Exception as error:
                retry = prior.recoverable_transport_failure(started) and attempt < SETTINGS['max_transport_attempts']
                previous.write(EXECUTION/'attempts'/f'{attempt}.json',dict(at=previous.now(),
                    status='failed',error_type=type(error).__name__,message=str(error),will_retry=retry))
                if not retry:
                    raise
                update(stage='reconnecting_after_transport_failure')
            finally:
                if forward is not None:
                    forward.terminate(); forward.wait(timeout=20); forward=None
                try: original.sync_remote()
                except Exception as error: update(last_log_sync_error=type(error).__name__)
            time.sleep(10)
        if prior.utility_identity() != saved['utility_results_sha256']:
            raise ValueError('Completed utility results changed')
        if not original.is_complete('sep_static','judge'):
            raise ValueError('SEP final judging incomplete')
        update(status='complete',stage='complete',completed_at=previous.now())
    except Exception as error:
        update(status='failed',error_type=type(error).__name__,message=str(error))
        raise


if __name__ == '__main__':
    main()
