#!/usr/bin/env python3
"""Preserve the September 24 static-security experiments before Lambda shutdown.

Copies research artifacts only. Does not edit evaluators, remove source files,
stop workloads, change checkpoint retention, or terminate the Lambda instance.
The previously checksummed standalone models remain in their REDS backup.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import base64
import fcntl
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEST = Path('/data2/jaewonc/secpo_backups/lambda_static_20260924')
OLD = Path('/data2/jaewonc/secpo_backups/lambda_20260919')
RUNS = {
    'native_run': Path('/data2/jaewonc/secpo_evals/qwen38_native_static_20260924'),
    'superseded_run': Path('/data2/jaewonc/secpo_evals/qwen38_lambda_static_20260924'),
}
HOST = 'ubuntu@209.20.159.198'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes',
       '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=30',
       '-o', 'ServerAliveCountMax=6', '-o', 'StrictHostKeyChecking=yes',
       '-o', 'UserKnownHostsFile=' + str(ROOT / 'runtime/lambda_gpqa_known_hosts'),
       '-i', str(ROOT / 'runtime/lambda_gpqa_ssh/id_ed25519')]
EXCLUDE = ('.git', '.ssh', '__pycache__', '.pytest_cache', '.env', '*.env',
           'id_ed25519', 'id_rsa', '*.pyc')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stage(name, **values):
    state = dict(status='running', safe_to_shutdown=False, stage=name,
                 host=HOST, updated_at=now(), **values)
    write(DEST / 'status.json', state)
    print(json.dumps(state), flush=True)


REMOTE_SNAPSHOT = r'''
import base64, hashlib, json, os, platform, subprocess
from pathlib import Path
roots = {
    'native_server': Path('/lambda/nfs/jaewon/secpo_native_static_20260924'),
    'superseded_server': Path('/lambda/nfs/jaewon/secpo_static_security_20260924'),
}
files=[]
for label,root in roots.items():
    for base,dirs,names in os.walk(root):
        dirs.sort()
        for name in sorted(names):
            p=Path(base)/name
            if p.is_symlink(): raise ValueError('Unexpected source symlink')
            before=p.stat()
            if before.st_size>100*1024*1024: raise ValueError('Unexpected oversized artifact')
            data=p.read_bytes()
            after=p.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                raise ValueError('Source changed during snapshot')
            files.append(dict(path=label+'/'+str(p.relative_to(root)),bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),data_b64=base64.b64encode(data).decode()))
runtime=Path('/home/ubuntu/secpo_static_security_20260924')
if set(p.name for p in runtime.iterdir())!={'venv'}:
    raise ValueError('New runtime artifacts require explicit preservation')
prior=Path('/lambda/nfs/jaewon/secpo_qwen38_eval_20260916')
validation=json.loads((roots['superseded_server']/'model_validation.json').read_text())
for item in validation.values():
    model=Path(item['directory'])
    if set(p.name for p in model.iterdir())!={r['path'] for r in item['files']}:
        raise ValueError('Model inventory changed since initial checksum validation')
    for row in item['files']:
        p=model/row['path']
        if p.is_symlink() or p.stat().st_size!=row['bytes']:
            raise ValueError('Previously validated model file missing or changed size')
active=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
    try: args=p.read_bytes().split(b'\0')
    except (FileNotFoundError,PermissionError): continue
    if b'vllm.entrypoints.openai.api_server' in args and any(a.startswith(b'secpo-') for a in args):
        active.append(int(p.parent.name))
if active: raise ValueError('Model server still running; snapshot not quiescent')
packages=json.loads(subprocess.check_output([str(runtime/'venv/bin/python'),'-m','pip','list','--format=json']))
print(json.dumps(dict(files=files,models=validation,packages=packages,
    environment=dict(platform=platform.platform(),python=platform.python_version(),
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True)),
    project_inventory={str(p):sorted(c.name for c in p.iterdir()) for p in [Path('/lambda/nfs/jaewon'),prior,runtime]},
    active_model_servers=active,
    model_check='Fresh inventory/size check; full SHA-256 was verified when this instance was prepared on September 24.')))
'''


def snapshot_remote():
    proc = subprocess.run(SSH + [HOST, shlex.join(['python3', '-c', REMOTE_SNAPSHOT])],
                          check=True, capture_output=True, text=True, timeout=300)
    snapshot = json.loads(proc.stdout)
    for item in snapshot['files']:
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe remote artifact path')
        payload = base64.b64decode(item.pop('data_b64'), validate=True)
        if len(payload) != item['bytes'] or hashlib.sha256(payload).hexdigest() != item['sha256']:
            raise ValueError('Transferred artifact checksum mismatch')
        target = DEST / 'lambda_project' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        if sha(target) != item['sha256']:
            raise ValueError('Persisted artifact checksum mismatch')
    if not snapshot['files']:
        raise ValueError('Empty remote snapshot')
    write(DEST / 'audit/remote_snapshot.json', snapshot)
    return snapshot


def blocked(name):
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in EXCLUDE)


def copy_verified(label, source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    command = ['rsync', '-a', '--no-owner', '--no-group']
    command.extend('--exclude=' + pattern for pattern in EXCLUDE)
    command.extend([str(source) + '/', str(destination) + '/'])
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    inventory = []
    for base, dirs, names in os.walk(source):
        dirs[:] = sorted(d for d in dirs if not blocked(d))
        for directory in dirs:
            if (Path(base) / directory).is_symlink():
                raise ValueError('Directory symlink requires explicit preservation')
        for name in sorted(names):
            if blocked(name):
                continue
            path = Path(base) / name
            relative = path.relative_to(source)
            target = destination / relative
            if path.is_symlink() or target.is_symlink():
                raise ValueError('Unexpected artifact symlink')
            before = path.stat()
            checksum = sha(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Source changed during archive verification')
            if not target.is_file() or target.stat().st_size != after.st_size or sha(target) != checksum:
                raise ValueError('Backup checksum mismatch: ' + str(relative))
            inventory.append(dict(path=str(relative), bytes=after.st_size, sha256=checksum))
    if not inventory:
        raise ValueError('Empty snapshot: ' + label)
    record = dict(source=str(source), destination=str(destination), files=inventory, at=now())
    write(DEST / 'audit' / (label + '.json'), record)
    return dict(source=str(source), destination=str(destination), files=len(inventory),
                bytes=sum(row['bytes'] for row in inventory), verification='fresh SHA-256 of source and copy')


def check_models(remote):
    models = {}
    for variant in ('base', 'secpo_lr8e5', 'secpo_lr6e5', 'secpo_lr4e5'):
        previous = read(OLD / 'audit/tasks' / (variant + '.json'))
        manifest = OLD / 'audit' / (variant + '.remote_sha256.jsonl')
        if previous['status'] != 'complete' or sha(manifest) != previous['manifest_sha256']:
            raise ValueError('Previous verified model manifest changed')
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        directory = OLD / 'models' / variant
        if set(p.name for p in directory.iterdir()) != {row['path'] for row in rows}:
            raise ValueError('Local model inventory differs')
        for row in rows:
            path = directory / row['path']
            if row['kind'] != 'file' or path.is_symlink() or not path.is_file() or path.stat().st_size != row['bytes']:
                raise ValueError('Previously backed-up model file missing or changed size')
        remote_key = {'secpo_lr8e5': 'secpo', 'secpo_lr4e5': 'secpo_lr4e5'}.get(variant)
        if remote_key:
            expected = [{k: row[k] for k in ('path', 'bytes', 'sha256')} for row in rows]
            current = remote['models'][remote_key]
            if current['status'] != 'complete' or sorted(current['files'], key=lambda r: r['path']) != sorted(expected, key=lambda r: r['path']):
                raise ValueError('Evaluated remote model differs from REDS backup identity')
        models[variant] = dict(directory=str(directory), files=len(rows), bytes=sum(r['bytes'] for r in rows),
            earlier_sha256_verified_at=previous['verified_at'], fresh_inventory_and_size_check=True,
            fresh_full_weight_sha256_check=False, matches_evaluated_remote_manifest=bool(remote_key))
    checkpoints = read(OLD / 'checkpoints/manifest.json')
    if checkpoints['status'] != 'complete':
        raise ValueError('Prior checkpoint backup incomplete')
    for item in checkpoints['backups']:
        path = Path(item['archive'])
        if not path.is_file() or path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Sampler archive missing or corrupt')
    record = dict(models=models, checkpoint_manifest=str(OLD / 'checkpoints/manifest.json'),
        freshly_verified_sampler_archives=len(checkpoints['backups']),
        checkpoint_retention_last_checked=checkpoints['verified_at'],
        training_states='On Tinker, not on Lambda; retention unchanged by this backup.', at=now())
    write(DEST / 'audit/models.json', record)
    return record


def check_results():
    runs = {}
    for variant in ('secpo', 'secpo_lr4e5'):
        for benchmark, attacks, total in [('agentdojo', 949, 984), ('agentdyn', 560, 588)]:
            directory = RUNS['native_run'] / variant / benchmark
            result = read(directory / 'results.json')
            tasks = list((directory / 'tasks').glob('*.json'))
            episodes = list((directory / 'episodes').rglob('*.json'))
            if (result['status'] != 'complete' or result['errors'] != 0 or
                    result['attacked_completed'] != attacks or result['completed'] != total or
                    len(tasks) != total or len(episodes) != total):
                raise ValueError('Evaluation results are incomplete')
            for path in tasks:
                if read(path)['status'] != 'complete':
                    raise ValueError('Incomplete task record')
            runs[variant + '/' + benchmark] = dict(episodes=total, attacked_episodes=attacks,
                cached_responses=len(list((directory / 'completion_cache').glob('*.json'))),
                macro_asr=result['macro_attack_success_rate'], micro_asr=result['micro_attack_success_rate'])
    if read(RUNS['native_run'] / 'supervisor.json')['status'] != 'complete':
        raise ValueError('Native supervisor is not complete')
    write(DEST / 'audit/evaluations.json', dict(at=now(), verified=True, runs=runs))
    return runs


def capture_environments():
    for benchmark in ('agentdojo', 'agentdyn'):
        python = ROOT / 'runtime/eval_envs' / benchmark / 'bin/python'
        packages = json.loads(subprocess.check_output([str(python), '-m', 'pip', 'list', '--format=json']))
        write(DEST / 'environments' / ('reds_' + benchmark + '.json'), packages)


def main():
    os.umask(0o077)
    DEST.mkdir(parents=True, exist_ok=True)
    with (DEST / 'archive.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            stage('checking_evaluations')
            evaluations = check_results()
            stage('collecting_remote_artifacts')
            remote = snapshot_remote()
            stage('verifying_saved_checkpoints_and_snapshotting_results')
            copies = {}
            with ThreadPoolExecutor(max_workers=2) as pool:
                future = pool.submit(check_models, remote)
                for label, source in RUNS.items():
                    copies[label] = copy_verified(label, source, DEST / label)
                models = future.result()
            stage('snapshotting_code_and_benchmark_sources')
            for name in ('scripts', 'src', 'configs', 'docs', 'tests', 'third_party'):
                copies['workspace_' + name] = copy_verified('workspace_' + name, ROOT / name, DEST / 'workspace' / name)
            for benchmark in ('agentdojo', 'agentdyn'):
                label = 'benchmark_' + benchmark
                copies[label] = copy_verified(label, ROOT / 'runtime/benchmarks' / benchmark,
                                              DEST / 'benchmark_sources' / benchmark)
            for name in ('README.md', 'pyproject.toml', 'release_manifest.json'):
                shutil.copy2(ROOT / name, DEST / 'workspace' / name)
                if sha(ROOT / name) != sha(DEST / 'workspace' / name):
                    raise ValueError('Workspace metadata checksum mismatch')
            capture_environments()
            write(DEST / 'environments/lambda_vllm.json', dict(packages=remote['packages'], environment=remote['environment']))
            shutil.copy2(ROOT / 'docs/LAMBDA_STATIC_BACKUP.md', DEST / 'README.md')
            status = dict(status='complete', safe_to_shutdown=True, host=HOST, completed_at=now(),
                instance_terminated=False, files_deleted=False, evaluations=evaluations, copies=copies,
                remote_files=len(remote['files']), remote_bytes=sum(f['bytes'] for f in remote['files']),
                models=models, prior_backup=str(OLD),
                prior_project_reuse='September 16-19 project already archived under prior_backup/lambda_project; not rerun or modified by the static-security experiments.',
                exclusions=['Credentials and SSH keys', 'Reinstallable virtual environments (package inventories preserved)',
                            'Rebuildable compilation/download caches', 'Duplicate weights (existing REDS backup retained)'])
            write(DEST / 'status.json', status)
            print(json.dumps(status), flush=True)
        except Exception as error:
            write(DEST / 'status.json', dict(status='failed', safe_to_shutdown=False, at=now(),
                error_type=type(error).__name__, message=str(error)))
            raise


if __name__ == '__main__':
    main()
