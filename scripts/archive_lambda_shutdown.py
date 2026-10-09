#!/usr/bin/env python3
"""Verified, resumable Lambda shutdown backup, with all large files on /data2.

Copies data only: never stops a server, deletes a source, or terminates Lambda.
Each model is verified against SHA-256 hashes computed on the source instance.
Authoritative local results are preserved separately from the older remote mirror.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
DEST = Path('/data2/jaewonc/secpo_backups/lambda_20260919')
POINTER = ROOT / 'outputs/lambda_shutdown_backup_20260919.json'
REMOTE = '/lambda/nfs/jaewon/secpo_qwen38_eval_20260916'
HOST = 'ubuntu@209.20.156.254'
SSH = ['ssh', '-i', str(ROOT / 'runtime/lambda_gpqa_ssh/id_ed25519'),
       '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
       '-o', 'StrictHostKeyChecking=yes', '-o', 'HostKeyAlgorithms=ssh-ed25519',
       '-o', 'UserKnownHostsFile=' + str(ROOT / 'runtime/lambda_gpqa_known_hosts'),
       '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=6']
REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
MODELS = {
    'base': REMOTE + '/hf_cache/hub/models--Qwen--Qwen3.8-27B/snapshots/' + REVISION,
    'secpo_lr8e5': REMOTE + '/secpo_merged',
    'secpo_lr6e5': REMOTE + '/secpo_lr6e5_merged',
    'secpo_lr4e5': REMOTE + '/secpo_lr4e5_merged',
}
EXCLUDED_NAMES = ['.git', '.ssh', '__pycache__', '.pytest_cache', '.rsync-partial',
                  '.env', '*.env', 'id_ed25519', 'id_rsa']
PROJECT_EXCLUDES = ['hf_cache', 'secpo_merged', 'secpo_lr6e5_merged', 'secpo_lr4e5_merged']
IO_LOCK = threading.Lock()
# The bulk destination is a rotating disk. Serial reads avoid seek contention
# between several simultaneous full-model SHA-256 passes.
VERIFY_LOCK = threading.Lock()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial-json')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def task_status(name, **values):
    with IO_LOCK:
        path = DEST / 'audit/tasks' / (name + '.json')
        previous = read(path)
        if values.get('status') and values['status'] != 'failed':
            previous.pop('error_type', None)
        if values.get('status') == 'complete':
            for field in ('current_file', 'verified_files', 'verified_bytes'):
                previous.pop(field, None)
        write(path, {**previous, 'task': name, 'updated_at': now(), **values})


def ssh(command, **kwargs):
    return subprocess.run(SSH + [HOST, command], **kwargs)


HASH_CODE = r'''
import fnmatch, hashlib, json, os, sys
from pathlib import Path
root=Path(sys.argv[1]); mode=sys.argv[2]
excluded=json.loads(sys.argv[3]); top_excluded=json.loads(sys.argv[4])
def blocked(name):
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in excluded)
for base, dirs, files in os.walk(root, followlinks=False):
    base=Path(base)
    dirs[:]=sorted(d for d in dirs if not blocked(d) and not (base==root and d in top_excluded))
    for name in sorted(files):
        if blocked(name): continue
        p=base/name
        row={'path':str(p.relative_to(root))}
        if p.is_symlink() and mode=='links':
            row.update(kind='symlink',target=os.readlink(p))
        else:
            before=p.stat()
            with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
            after=p.stat()
            if before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns:
                raise RuntimeError('Source changed while hashing')
            row.update(kind='file',bytes=after.st_size,sha256=digest)
        print(json.dumps(row),flush=True)
    for name in dirs:
        p=base/name
        if p.is_symlink():
            print(json.dumps({'path':str(p.relative_to(root)),'kind':'symlink','target':os.readlink(p)}),flush=True)
'''


def verify_manifest(manifest, destination, *, progress=None):
    count = size = 0
    last_update = 0.0
    with manifest.open() as records:
        for line in records:
            item = json.loads(line)
            relative = Path(item['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe manifest path')
            path = destination / relative
            if progress is not None and time.monotonic() - last_update >= 5:
                progress(verified_files=count, verified_bytes=size, current_file=item['path'])
                last_update = time.monotonic()
            if item['kind'] == 'symlink':
                if not path.is_symlink() or os.readlink(path) != item['target']:
                    raise ValueError('Symlink differs: ' + item['path'])
            else:
                if path.is_symlink() or not path.is_file() or path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
                    raise ValueError('File checksum differs: ' + item['path'])
                size += item['bytes']
            count += 1
    if not count:
        raise ValueError('Empty source inventory')
    return dict(files=count, bytes=size, manifest_sha256=sha(manifest))


def transfer_remote(name, source, destination, *, project=False):
    destination.mkdir(parents=True, exist_ok=True)
    cmd = ['rsync', '-a', '--no-owner', '--no-group', '--partial', '--partial-dir=.rsync-partial',
           '--timeout=300', '--info=progress2', '--stats']
    if not project:
        cmd += ['--copy-links']
    for pattern in EXCLUDED_NAMES:
        cmd += ['--exclude=' + pattern]
    if project:
        cmd += ['--exclude=/' + name + '/' for name in PROJECT_EXCLUDES]
    cmd += ['-e', shlex.join(SSH), HOST + ':' + source + '/', str(destination) + '/']
    log_path = DEST / 'logs' / (name + '.rsync.log')
    for attempt in range(1, 4):
        task_status(name, status='transferring', started_at=now(), destination=str(destination), attempt=attempt)
        with log_path.open('a') as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode == 0:
            break
        if attempt == 3:
            raise RuntimeError('Transfer failed: ' + name)
        time.sleep(10)
    task_status(name, status='hashing_remote')
    manifest = DEST / 'audit' / (name + '.remote_sha256.jsonl')
    hash_cmd = shlex.join(['/home/ubuntu/secpo_qwen38_runtime_20260916/merge-venv/bin/python', '-c', HASH_CODE, source,
        'links' if project else 'dereference', json.dumps(EXCLUDED_NAMES), json.dumps(PROJECT_EXCLUDES if project else [])])
    with manifest.open('w') as output, (DEST / 'logs' / (name + '.hash.log')).open('a') as log:
        proc = ssh(hash_cmd, stdout=output, stderr=log)
    if proc.returncode:
        raise RuntimeError('Remote hashing failed: ' + name)
    task_status(name, status='waiting_for_local_verification')
    with VERIFY_LOCK:
        task_status(name, status='verifying_local')
        verified = verify_manifest(manifest, destination,
            progress=lambda **values: task_status(name, **values))
    if not project:
        index = read(destination / 'model.safetensors.index.json')
        if not index.get('weight_map'):
            raise ValueError('Missing model tensor index')
        for filename in set(index['weight_map'].values()):
            if not (destination / filename).is_file():
                raise ValueError('Model shard missing')
        verified['model_shards'] = len(set(index['weight_map'].values()))
    task_status(name, status='complete', verified_at=now(), **verified)
    return verified


def copy_verified_local(name, source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    excluded = EXCLUDED_NAMES + ['lambda_shutdown_backup_20260919.json']
    cmd = ['rsync', '-a', '--no-owner', '--no-group'] + ['--exclude=' + n for n in excluded]
    cmd += [str(source) + '/', str(destination) + '/']
    with (DEST / 'logs/local_copy.log').open('a') as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    manifest = DEST / 'audit' / (name + '.local_source_sha256.jsonl')
    with manifest.open('w') as output:
        subprocess.run([sys.executable, '-c', HASH_CODE, str(source), 'links', json.dumps(excluded), '[]'],
                       check=True, stdout=output)
    return verify_manifest(manifest, destination)


def local_snapshot():
    name = 'authoritative_local_data'
    task_status(name, status='copying')
    records = {}
    for directory in ('outputs', 'scripts', 'src', 'configs', 'docs', 'tests', 'data', 'third_party'):
        records[directory] = copy_verified_local('workspace_' + directory, ROOT / directory, DEST / 'workspace' / directory)
    for filename in ('README.md', 'pyproject.toml', 'release_manifest.json'):
        source = ROOT / filename
        shutil.copy2(source, DEST / 'workspace' / filename)
        if sha(source) != sha(DEST / 'workspace' / filename):
            raise ValueError('Workspace metadata copy differs')
    for variant in ('lr4e5', 'lr6e5'):
        code = ROOT / 'runtime/lambda_suite_20260916' / (variant + '_code')
        provenance = read(code / 'provenance.json')
        for filename, item in provenance['files'].items():
            source = code / filename
            target = DEST / 'isolated_code' / variant / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if sha(source) != item['sha256']:
                raise ValueError('Isolated code identity changed')
            shutil.copy2(source, target)
            if sha(target) != item['sha256']:
                raise ValueError('Isolated code copy differs')
        write(DEST / 'isolated_code' / variant / 'provenance.json', provenance)
    for filename in ('SEP_dataset_test.json', 'SEP_dataset_test_Meta-Llama-3-8B-Instruct.json'):
        source = Path('/home/jaewonc/SecAlign_cont/data') / filename
        target = DEST / 'private_references' / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha(source) != sha(target):
            raise ValueError('SEP reference copy differs')
    for dataset in ('EleutherAI___hendrycks_math', 'Idavidrein___gpqa', 'TIGER-Lab___mmlu-pro',
                    'Sizhe-Chen___qwen3.8-27_b-instruct-sec_po-trainset'):
        source = Path('/data2/jaewonc/huggingface/datasets') / dataset
        records[dataset] = copy_verified_local('dataset_' + dataset, source, DEST / 'datasets' / dataset)
    for benchmark in ('agentdojo', 'agentdyn'):
        source = ROOT / 'runtime/benchmarks' / benchmark
        records[benchmark] = copy_verified_local('benchmark_' + benchmark, source, DEST / 'benchmark_sources' / benchmark)
    write(DEST / 'audit/local_snapshot.json', dict(status='complete', at=now(), groups=records))
    task_status(name, status='complete', verified_at=now(), groups=len(records), bytes=sum(r['bytes'] for r in records.values()))


def preserve_checkpoints():
    task_status('tinker_checkpoints', status='checking_retention')
    sys.path.insert(0, str(ROOT / 'src'))
    from secpo.tinker.evaluation_common import load_credentials
    import tinker
    spec = importlib.util.spec_from_file_location('preserve', ROOT / 'scripts/preserve_tinker_checkpoints.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    load_credentials(ROOT / 'secpo_eval_keys.env')
    rest = tinker.ServiceClient(timeout=120, max_retries=2).create_rest_client()
    checkpoints, backups = [], []
    for label, training, export in [('lr8e5', 'qwen38_27b_tinker', 'secpo_export'),
        ('lr6e5', 'qwen38_27b_tinker_lr6e5', 'secpo_lr6e5_export'),
        ('lr4e5', 'qwen38_27b_tinker_lr4e5', 'secpo_lr4e5_export')]:
        result = read(ROOT / 'outputs' / training / 'result.json')
        if result.get('steps_completed') != 297:
            raise ValueError('Training is not complete')
        for field in ('final_state_path', 'final_sampler_path'):
            path = result[field]
            run_id = path.removeprefix('tinker://').split('/', 1)[0]
            found = [c for c in rest.list_checkpoints(run_id).result(timeout=120).checkpoints if c.tinker_path == path]
            if len(found) != 1 or found[0].public:
                raise ValueError('Final checkpoint missing or unexpectedly public')
            if found[0].expires_at is not None:
                rest.set_checkpoint_ttl_from_tinker_path(path, ttl_seconds=None).result(timeout=120)
            checked = next(c for c in rest.list_checkpoints(run_id).result(timeout=120).checkpoints if c.tinker_path == path)
            if checked.expires_at is not None:
                raise ValueError('Expiration remains enabled')
            checkpoints.append(dict(model=label, **checked.model_dump(mode='json')))
        backups.append(module.backup_one(rest, result['final_sampler_path'], DEST / 'checkpoints' / label / 'sampler.tar',
            ROOT / 'runtime/lambda_suite_20260916' / export))
    write(DEST / 'checkpoints/manifest.json', dict(status='complete', verified_at=now(), checkpoints=checkpoints, backups=backups,
        training_state_scope='Remote Tinker finals, private, expires_at=null; Tinker does not support training-state archive downloads.',
        inference_scope='Locally verified sampler archives plus exact merged HF models and pinned base.'))
    task_status('tinker_checkpoints', status='complete', final_checkpoints=len(checkpoints), local_sampler_backups=len(backups))


ENV_CODE = "import importlib.metadata as m,json,platform,sys; print(json.dumps({'python':sys.version,'platform':platform.platform(),'packages':sorted([{'name':d.metadata.get('Name'),'version':d.version} for d in m.distributions()],key=lambda d:(d['name'] or '').lower())}))"


def environments():
    task_status('environments', status='collecting')
    for label in ('venv', 'merge-venv'):
        command = shlex.join(['/home/ubuntu/secpo_qwen38_runtime_20260916/' + label + '/bin/python', '-c', ENV_CODE])
        result = ssh(command, capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise RuntimeError('Remote environment capture failed')
        write(DEST / 'environments' / ('lambda_' + label + '.json'), json.loads(result.stdout))
    for label in ('lm', 'agentdojo', 'agentdyn'):
        result = subprocess.run([str(ROOT / 'runtime/eval_envs' / label / 'bin/python'), '-c', ENV_CODE], capture_output=True, text=True, check=True)
        write(DEST / 'environments' / ('reds_' + label + '.json'), json.loads(result.stdout))
    result = ssh('nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader', capture_output=True, text=True, timeout=30)
    write(DEST / 'environments/lambda_gpu.json', dict(at=now(), description=result.stdout.strip(), exit_code=result.returncode))
    task_status('environments', status='complete')


def finalize_backup():
    """Require all full benchmark scores and provide a portable checksum list."""
    models = ('base', 'secpo', 'secpo_lr6e5', 'secpo_lr4e5')
    run = DEST / 'workspace/outputs/qwen38_lambda_recommended_20260916'
    luna = DEST / 'workspace/outputs/qwen38_lambda_luna_openai_20260918'
    counts = dict(gpqa_diamond=198, mmlu_pro=1400, minerva_math=700,
                  agentdojo=97, agentdyn=60, alpacaeval2=805, sep_utility=9160)
    results = []
    for model in models:
        for benchmark, count in counts.items():
            directory = (luna if benchmark in {'alpacaeval2', 'sep_utility'} else run) / model / benchmark
            progress, result = read(directory / 'progress.json'), read(directory / 'results.json')
            if progress.get('status') != 'complete' or progress.get('completed') != count or not result:
                raise ValueError('A full evaluation result is missing from the backup')
            results.append(dict(model=model, benchmark=benchmark, samples=count,
                result=str((directory / 'results.json').relative_to(DEST)),
                sha256=sha(directory / 'results.json')))
    lines = []
    for name in MODELS:
        for line in (DEST / 'audit' / (name + '.remote_sha256.jsonl')).read_text().splitlines():
            item = json.loads(line)
            if item['kind'] == 'file':
                lines.append(item['sha256'] + '  ' + str(Path('models') / name / item['path']))
    (DEST / 'SHA256SUMS.models').write_text('\n'.join(lines) + '\n')
    guide = ROOT / 'docs/LAMBDA_SHUTDOWN_BACKUP.md'
    shutil.copy2(guide, DEST / 'README.md')
    write(DEST / 'audit/shutdown_readiness.json', dict(status='complete', verified_at=now(),
        models=list(MODELS), full_benchmark_results=len(results), results=results,
        local_sampler_archives=3, nonexpiring_remote_training_states=3,
        guide_sha256=sha(DEST / 'README.md'), model_checksum_list_sha256=sha(DEST / 'SHA256SUMS.models')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--lock-fd', type=int)
    parser.add_argument('--wait-for-lock', action='store_true', help='Finish an interrupted copy after the current worker releases its lease')
    args = parser.parse_args()
    if not Path('/data2/jaewonc').is_dir() or DEST.resolve() != DEST:
        raise ValueError('Unexpected bulk-storage location')
    if shutil.disk_usage('/data2/jaewonc').free < 260 * 1024**3:
        raise ValueError('Insufficient space for full backup')
    DEST.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name in ('audit/tasks', 'logs', 'models', 'checkpoints', 'environments'):
        (DEST / name).mkdir(parents=True, exist_ok=True)
    lock = os.fdopen(args.lock_fd, 'a+') if args.lock_fd is not None else (DEST / 'archive.lock').open('a+')
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if not args.wait_for_lock:
                raise
            time.sleep(15)
    if args.detach:
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--lock-fd', str(lock.fileno())]
        with (DEST / 'logs/supervisor.log').open('a') as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=(lock.fileno(),))
        write(POINTER, dict(status='launched', destination=str(DEST), pid=proc.pid, at=now()))
        print(json.dumps(read(POINTER)), flush=True)
        return
    identity = dict(remote=HOST + ':' + REMOTE, destination=str(DEST), base_revision=REVISION,
                    models=MODELS, credentials_copied=False, source_deletions=False)
    if (DEST / 'identity.json').exists() and read(DEST / 'identity.json') != identity:
        raise ValueError('Existing backup identity differs')
    write(DEST / 'identity.json', identity)
    if read(ROOT / 'outputs/qwen38_lambda_luna_openai_20260918/supervisor.json').get('status') != 'complete':
        raise ValueError('Judging is not complete')
    status = dict(status='running', started_at=now(), pid=os.getpid(), destination=str(DEST), safe_to_shutdown=False)
    failures = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = {pool.submit(transfer_remote, name, source, DEST / 'models' / name): name for name, source in MODELS.items()}
        jobs[pool.submit(transfer_remote, 'lambda_project', REMOTE, DEST / 'lambda_project', project=True)] = 'lambda_project'
        jobs[pool.submit(local_snapshot)] = 'authoritative_local_data'
        jobs[pool.submit(preserve_checkpoints)] = 'tinker_checkpoints'
        jobs[pool.submit(environments)] = 'environments'
        while jobs:
            for future, name in list(jobs.items()):
                if not future.done():
                    continue
                try:
                    future.result()
                except Exception as error:
                    task_status(name, status='failed', error_type=type(error).__name__)
                    failures.append(dict(task=name, error_type=type(error).__name__))
                del jobs[future]
            status.update(updated_at=now(), active_tasks=list(jobs.values()), failures=failures)
            write(DEST / 'status.json', status)
            write(POINTER, status)
            if jobs:
                time.sleep(15)
    if not failures:
        try:
            finalize_backup()
        except Exception as error:
            failures.append(dict(task='shutdown_readiness', error_type=type(error).__name__))
    status.update(status='failed' if failures else 'complete', ended_at=now(), safe_to_shutdown=not failures)
    write(DEST / 'status.json', status)
    write(POINTER, status)
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(status='failed', error_type=type(error).__name__)), flush=True)
        raise SystemExit(1) from None
