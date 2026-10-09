#!/usr/bin/env python3
"""Resume SEP only; preserve completed utilities and retry bounded transport failures."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import launch_brev_four_gpu as previous

ROOT,RUN = previous.ROOT,previous.RUN
CONFIG = ROOT/'configs/brev_sep_resume_20261008.json'
SETTINGS = json.loads(CONFIG.read_text())
EXECUTION = RUN/SETTINGS['execution_id']
FILES = (*previous.FILES,'scripts/launch_brev_sep_resume.py','scripts/brev_sep_resume_remote.py',
         'configs/brev_sep_resume_20261008.json')


def configure():
    previous.CONFIG = CONFIG
    previous.SETTINGS = SETTINGS
    previous.RESUME = EXECUTION
    previous.JOBS = SETTINGS['replicas']
    previous.REMOTE_HELPER = previous.original.REMOTE+'/code/scripts/brev_sep_resume_remote.py'
    previous.FILES = FILES


def utility_identity():
    return {b:hashlib.sha256((RUN/'base'/b/'results.json').read_bytes()).hexdigest()
            for b in ('mmlu_pro','minerva_math')}


def baseline():
    path = EXECUTION/'baseline.json'
    if path.exists(): return previous.original.read(path)
    inventory = {}
    sep = RUN/'base/sep_static'
    for directory in ('rows','completion_cache','judgments'):
        for source in (sep/directory).rglob('*.json'):
            inventory[str(source.relative_to(RUN))] = hashlib.sha256(source.read_bytes()).hexdigest()
    previous.write(EXECUTION/'existing_cache_sha256.json',inventory)
    for source in [RUN/'supervisor.json',RUN/'active_execution.json',sep/'progress.json',sep/'full_worker.json',sep/'judge_worker.json']:
        if source.exists():
            destination = EXECUTION/'before'/source.relative_to(RUN)
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,destination)
    record = dict(at=previous.now(),saved_sep_rows=sum(1 for _ in (sep/'rows').rglob('*.json')),
        utility_results_sha256=utility_identity(),
        prior_failure='2026-10-04 22:18:38 UTC: SSH client broken pipe; proxy HTTP 502 transport errors')
    previous.write(path,record)
    previous.write(EXECUTION/'manifest.json',dict(at=previous.now(),allocation=SETTINGS,
        original_source_identity=previous.original.read(RUN/'source_identity.json'),
        execution_source_identity=previous.original.read(EXECUTION/'source_identity.json'),
        baseline=record,benchmark='sep_static',total=54960,
        protocol='Unchanged original evaluator, six full attacks, same original three judges x three votes',
        max_output_tokens=16384,enable_thinking=False,global_generation_concurrency=8,
        changes=['Four single-GPU inference replicas on currently free IDs 0,2,3,4',
                 'Up to three generation attempts on diagnosed connection failures; preserve valid caches'],
        numerical_note='Placement/batching differs; new generations are not promised bit-identical.',
        utility_policy='Completed MMLU-Pro and Minerva are not rerun or modified'))
    previous.write(RUN/'active_execution.json',dict(path=str(EXECUTION),at=previous.now(),allocation=SETTINGS))
    return record


def recoverable_transport_failure(attempt_started):
    worker = RUN/'base/sep_static/full_worker.json'
    if not worker.exists(): return False
    state = previous.original.read(worker)
    if state.get('status') != 'failed' or state.get('started_at','') < attempt_started:
        return False
    # Do not turn protocol, accounting, source-identity, or judge failures into retries.
    log = (RUN/'base/sep_static/full.log').read_text(errors='replace')[-12000:]
    last_line = log.strip().splitlines()[-1] if log.strip() else ''
    if last_line != 'RuntimeError: Lambda inference failed with HTTP 502': return False
    for path in (EXECUTION/'transport_errors').glob('*.json'):
        record = previous.original.read(path)
        if record.get('at','') >= attempt_started and record.get('problem') == 'transport_error':
            return True
    return False


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
        command = [str(previous.original.PYTHON),'-u',str(Path(__file__).resolve()),'--lock-fd',str(lock.fileno())]
        with (EXECUTION/'supervisor.log').open('a') as log:
            child = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                env=previous.original.environment(),cwd=ROOT,start_new_session=True,pass_fds=(lock.fileno(),))
        previous.write(EXECUTION/'launch.json',dict(pid=child.pid,command=command,at=previous.now()))
        print(json.dumps(dict(status='queued',pid=child.pid,execution=str(EXECUTION))))
        return
    saved = baseline()
    state = dict(status='running',pid=os.getpid(),execution=str(EXECUTION),benchmark='sep_static',
        gpu_ids=SETTINGS['allowed_gpu_ids'],saved_sep_rows=saved['saved_sep_rows'],
        started_at=previous.now(),completed_utilities_unchanged=True)
    def update(**kwargs):
        state.update(updated_at=previous.now(),**kwargs)
        previous.write(EXECUTION/'supervisor.json',state)
        previous.write(RUN/'supervisor.json',state)
    forward = None
    try:
        # Reuse only the generation and original judge implementations, never rerun utilities.
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
                retry = recoverable_transport_failure(started) and attempt < SETTINGS['max_transport_attempts']
                previous.write(EXECUTION/'attempts'/f'{attempt}.json',dict(at=previous.now(),
                    status='failed',error_type=type(error).__name__,message=str(error),will_retry=retry))
                if not retry: raise
                update(stage='reconnecting_after_transport_failure',last_error=str(error))
            finally:
                if forward is not None:
                    forward.terminate(); forward.wait(timeout=20); forward=None
                try: previous.original.sync_remote()
                except Exception as error: update(last_log_sync_error=type(error).__name__)
            time.sleep(10)
        if utility_identity() != saved['utility_results_sha256']:
            raise ValueError('Completed utility results unexpectedly changed')
        if not previous.original.is_complete('sep_static','judge'):
            raise ValueError('SEP final judging is incomplete')
        update(status='complete',stage='complete',completed_at=previous.now())
    except Exception as error:
        update(status='failed',error_type=type(error).__name__,message=str(error))
        raise


if __name__ == '__main__':
    main()
