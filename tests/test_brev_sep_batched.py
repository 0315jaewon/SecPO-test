"""Concurrency-only changes must not alter the frozen evaluation recipe."""
import ast
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import launch_brev_sep_batched as batch


def test_global_concurrency_and_separate_judge_limit():
    assert batch.SETTINGS['generation_concurrency'] == 1024
    assert batch.worker_command('sep_static','full')[-1] == '1024'
    assert batch.worker_command('sep_static','judge')[-1] == '8'
    assert batch.worker_command('mmlu_pro','full')[-1] == '12'
    assert batch.BatchReplicaProxy.request_queue_size >= 1024
    assert batch.SETTINGS['ssh_tunnels_per_replica'] is True


def test_only_authorized_four_devices_used():
    jobs = batch.SETTINGS['replicas']
    assert sorted(j['gpu_id'] for j in jobs.values()) == [0,2,3,4]
    assert len({j['gpu_uuid'] for j in jobs.values()}) == 4
    assert {j['benchmark'] for j in jobs.values()} == {'sep_static'}


def test_only_batching_server_flags_change():
    code = '''
import sys
sys.path.insert(0,'scripts')
import brev_base_remote as base
before=base.server_command('sep_static')
import brev_sep_batched_remote as remote
remote.configure()
job=next(iter(remote.previous.JOBS))
after=base.server_command(job)
assert after[after.index('--max-num-seqs')+1]=='256'
assert after[after.index('--max-num-batched-tokens')+1]=='8192'
for flag in ('--port','--max-num-seqs','--max-num-batched-tokens'):
    after[after.index(flag)+1]=before[before.index(flag)+1]
assert before==after
assert base.env()['CUDA_VISIBLE_DEVICES']==''
assert remote.previous.launch_command(job)[1].endswith('brev_sep_batched_remote.py')
'''
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)


def test_original_sources_are_unchanged():
    assert batch.original.identity() == batch.original.read(batch.RUN/'source_identity.json')


def test_generation_delegates_to_original_sep():
    tree = ast.parse((ROOT/'scripts/evaluate_brev_sep_batched.py').read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call)
             and isinstance(n.func,ast.Attribute) and n.func.attr=='sep']
    assert len(calls)==1
    assert ast.unparse(calls[0]) == 'original.sep(args.phase, args.concurrency)'
    assert batch.SETTINGS['capacity_probe_stages'] == [64,256,1024]
    assert batch.SETTINGS['capacity_probe_max_tokens'] == 16


def test_configuration_targets_new_execution_without_modifying_original_settings():
    code = '''
import sys
sys.path.insert(0,'scripts')
import launch_brev_sep_batched as batch
batch.configure()
assert batch.previous.RESUME==batch.EXECUTION
assert batch.previous.REMOTE_HELPER.endswith('/brev_sep_batched_remote.py')
assert batch.previous.tunnel is batch.open_replica_tunnels
assert batch.original.worker_command('sep_static','full')[-1]=='1024'
assert batch.original.worker_command('sep_static','judge')[-1]=='8'
assert batch.original.SETTINGS['jobs']['sep_static']['concurrency']==8
assert batch.original.SETTINGS['jobs']['sep_static']['local_port']==19440
assert batch.original.identity()==batch.original.read(batch.RUN/'source_identity.json')
'''
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)


def test_tunnel_group_stops_only_its_processes():
    class Process:
        def __init__(self): self.stopped=False; self.waited=False
        def poll(self): return 0 if self.stopped else None
        def terminate(self): self.stopped=True
        def wait(self,timeout): assert timeout>0; self.waited=True
    processes=[Process() for _ in range(4)]
    group=batch.TunnelGroup(processes)
    group.terminate(); group.wait(timeout=10)
    assert all(p.stopped and p.waited for p in processes)
