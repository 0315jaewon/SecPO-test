#!/usr/bin/env python3
"""Execution-only GPU reassignment; imports the unchanged serving recipe."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import brev_base_remote as base

CONFIG = Path(__file__).resolve().parents[1]/'configs/brev_resume_four_gpu_20261004.json'
ALLOCATION = json.loads(CONFIG.read_text())
JOBS = ALLOCATION['replicas']


def validate_allocation(settings=None):
    selected = list(JOBS.values())
    if (len(selected) != 4 or sorted(j['gpu_id'] for j in selected) != ALLOCATION['allowed_gpu_ids']
            or len({j['gpu_uuid'] for j in selected}) != 4
            or any(type(j['gpu_id']) is not int or not 0 <= j['gpu_id'] <= 7 for j in selected)
            or len({j['remote_port'] for j in selected}) != 4
            or len({j['local_port'] for j in selected}) != 4):
        raise ValueError('Invalid four-GPU execution allocation')
    if settings is not None and settings['jobs'] != JOBS:
        raise ValueError('Execution allocation differs from the pinned configuration')


base.SETTINGS = {**base.SETTINGS,'allowed_gpu_ids':ALLOCATION['allowed_gpu_ids'],'jobs':JOBS}
base.validate_allocation = validate_allocation


def launch_command(job):
    return [sys.executable,str(Path(__file__).resolve()),'serve','--job',job]


def process_command(pid):
    return Path(f'/proc/{pid}/cmdline').read_bytes().decode().strip('\0').split('\0')


def status(job):
    directory = base.ROOT/'servers'/job
    state = directory/'server.json'
    if state.exists():
        record = base.read(state)
        pid = record['pid']
        if base.alive(pid):
            ticks = Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19]
            if process_command(pid) != base.server_command(job) or ticks != record['start_ticks']:
                raise ValueError('Recorded server PID was reused; refusing adoption')
            environment = Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
            expected = ('CUDA_VISIBLE_DEVICES='+JOBS[job]['gpu_uuid']).encode()
            if expected not in environment or record['gpu_uuid'] != JOBS[job]['gpu_uuid']:
                raise ValueError('Owned server has a different GPU assignment')
            return dict(status='serving',pid=pid,gpu=JOBS[job])
    launch = directory/'launch.json'
    if launch.exists():
        record = base.read(launch)
        if base.alive(record['pid']):
            if process_command(record['pid']) != launch_command(job):
                raise ValueError('Recorded launcher PID was reused; refusing adoption')
            return dict(status='starting',pid=record['pid'],gpu=JOBS[job])
    return dict(status='stopped',gpu=JOBS[job])


def start(job):
    directory = base.ROOT/'servers'/job
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'start.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        current = status(job)
        if current['status'] != 'stopped':
            print(json.dumps(current)); return
        base.gpu_for(job)
        command = launch_command(job)
        with (directory/'serve.log').open('a') as log:
            child = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                     env=base.env(),start_new_session=True)
        base.write(directory/'launch.json',dict(pid=child.pid,command=command,at=base.now(),
            allocation_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest()))
        print(json.dumps(dict(status='starting',pid=child.pid,gpu=JOBS[job])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['start','serve','status','check-gpu','stop'])
    parser.add_argument('--job',choices=list(JOBS),required=True)
    args = parser.parse_args()
    os.umask(0o077)
    validate_allocation()
    if args.action == 'start': start(args.job)
    elif args.action == 'serve':
        base.validate_cuda(args.job)
        base.serve(args.job)
    elif args.action == 'status': print(json.dumps(status(args.job)))
    elif args.action == 'stop': base.stop(args.job)
    else:
        try:
            print(json.dumps(dict(status='available',gpu=base.gpu_for(args.job))))
        except base.GpuBusy as error:
            print(json.dumps(dict(status='busy',message=str(error))))


if __name__ == '__main__':
    main()
