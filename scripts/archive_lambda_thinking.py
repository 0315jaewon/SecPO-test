#!/usr/bin/env python3
"""Snapshot the completed thinking/GPT-6 experiments before Lambda shutdown.

Copies research artifacts only. Never stops processes, deletes originals, or
terminates an instance. Existing standalone weights remain in the verified
September 19 REDS backup instead of being duplicated.
"""
from concurrent.futures import ThreadPoolExecutor
import base64
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from secpo.tinker.evaluation_common import digest, now, write_json
from secpo.tinker import reference_eval as ref
from secpo.evaluation.reference_utility import sha256_file

spec = importlib.util.spec_from_file_location('thinking_launch', ROOT / 'scripts/launch_lambda_thinking.py')
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)
DEST = Path('/data2/jaewonc/secpo_backups/lambda_thinking_20260923')
OLD = launch.BACKUP
THINKING = launch.OUTPUT
GPT6 = Path('/data2/jaewonc/secpo_evals/gpt6_luna_alpaca_20260923')
EXCLUDE = ('.git', '.ssh', '__pycache__', '.pytest_cache', '.env', '*.env', 'id_ed25519', 'id_rsa')


def read(path):
    return json.loads(path.read_text())


def stage(name, **kwargs):
    record = dict(status='running', safe_to_shutdown=False, stage=name, updated_at=now(), **kwargs)
    write_json(DEST / 'status.json', record)
    print(json.dumps(record), flush=True)


REMOTE_SNAPSHOT = r'''
import base64, hashlib, json, os, platform, subprocess, sys
from pathlib import Path
root=Path(sys.argv[1])
files=[]
for base, dirs, names in os.walk(root):
    base=Path(base)
    dirs[:]=sorted(d for d in dirs if d not in {'venv','models','hf_cache','__pycache__','.git','.ssh'})
    for name in sorted(names):
        if name=='.env' or name.endswith('.env') or name in {'id_ed25519','id_rsa'}: continue
        p=base/name
        if p.is_symlink(): raise ValueError('Unexpected project symlink')
        before=p.stat()
        if before.st_size>100*1024*1024: raise ValueError('Unexpected oversized artifact')
        # Capture exactly the observed prefix of an actively appended server log.
        with p.open('rb') as f: data=f.read(before.st_size)
        after=p.stat()
        if len(data)!=before.st_size or after.st_size<before.st_size:
            raise ValueError('Source truncated while snapshotting')
        if p.suffix!='.log' and (after.st_size,after.st_mtime_ns)!=(before.st_size,before.st_mtime_ns):
            raise ValueError('Non-log artifact changed during snapshot')
        files.append(dict(path=str(p.relative_to(root)),bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),data_b64=base64.b64encode(data).decode(),
            source_mtime_ns=before.st_mtime_ns,log_snapshot=p.suffix=='.log'))
model_files=[]
for line in (root/'SHA256SUMS.models').read_text().splitlines():
    expected,relative=line.split('  ',1)
    if not relative.startswith('models/base/'): continue
    p=root/relative
    if p.is_symlink(): raise ValueError('Unexpected model symlink')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    if h.hexdigest()!=expected: raise ValueError('Remote model checksum mismatch')
    model_files.append(dict(path=relative,bytes=p.stat().st_size,sha256=expected))
packages=json.loads(subprocess.check_output([str(root/'venv/bin/python'),'-m','pip','list','--format=json']))
print(json.dumps(dict(files=files,model_files=model_files,packages=packages,
    environment=dict(platform=platform.platform(),python=sys.version,
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True)),
    excluded=['installed venv (package inventory and setup script retained)','duplicate model weights (verified on REDS)',
              'rebuildable compilation/download caches','credentials and SSH keys'])))
'''


def snapshot_remote():
    command = launch.SSH + [launch.HOST, shlex.join(['python3', '-c', REMOTE_SNAPSHOT, launch.REMOTE])]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if proc.returncode:
        raise RuntimeError('Remote snapshot failed; no shutdown clearance')
    result = json.loads(proc.stdout)
    for item in result['files']:
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe remote snapshot path')
        data = base64.b64decode(item.pop('data_b64'), validate=True)
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('Transferred snapshot checksum mismatch')
        path = DEST / 'lambda_project' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        if sha256_file(path) != item['sha256']:
            raise ValueError('Persisted snapshot checksum mismatch')
    if not result['files'] or len(result['model_files']) != 32:
        raise ValueError('Unexpected remote inventory')
    write_json(DEST / 'audit/remote_snapshot.json', dict(at=now(), host=launch.HOST, root=launch.REMOTE, **result))
    return result


def verify_saved_models():
    # Rehash the model on the current instance's critical path. Other models
    # were fully checksummed September 19; check inventory/size again here.
    records = {}
    for variant in ('base', 'secpo_lr8e5', 'secpo_lr6e5', 'secpo_lr4e5'):
        prior = read(OLD / 'audit/tasks' / (variant + '.json'))
        if prior['status'] != 'complete':
            raise ValueError('Earlier model backup incomplete')
        expected = [json.loads(line) for line in (OLD / 'audit' / (variant + '.remote_sha256.jsonl')).read_text().splitlines()]
        for item in expected:
            p = OLD / 'models' / variant / item['path']
            if item['kind'] != 'file' or p.is_symlink() or not p.is_file() or p.stat().st_size != item['bytes']:
                raise ValueError('Previously backed-up model file missing or changed in size')
            if variant == 'base':
                stage('verifying_base_weights', file=item['path'])
                if sha256_file(p) != item['sha256']:
                    raise ValueError('Local base weights failed fresh checksum verification')
        records[variant] = dict(directory=str(OLD / 'models' / variant), files=len(expected),
            bytes=sum(r['bytes'] for r in expected), earlier_sha256_verified_at=prior['verified_at'],
            fresh_sha256_check=variant=='base', current_inventory_and_size_check=True)
    checkpoints = read(OLD / 'checkpoints/manifest.json')
    for item in checkpoints['backups']:
        p = Path(item['archive'])
        if not p.is_file() or p.stat().st_size != item['bytes'] or sha256_file(p) != item['sha256']:
            raise ValueError('Saved adapter archive missing or corrupt')
    write_json(DEST / 'audit/local_models.json', dict(at=now(), models=records,
        adapter_archives_verified=len(checkpoints['backups']), checkpoint_manifest=str(OLD / 'checkpoints/manifest.json'),
        training_states='Remain on Tinker; not on this Lambda instance. See existing preservation manifest.'))
    return records


def audit_results():
    source = THINKING / 'base/alpacaeval2'
    candidates, generations = read(source / 'candidate_outputs.json'), read(source / 'generations.json')
    if len(candidates) != 805 or len(generations) != 805:
        raise ValueError('Incomplete thinking-on generations')
    for i, (c, g) in enumerate(zip(candidates, generations)):
        cache = read(source / 'completion_cache' / (g['request_hash'] + '.json'))
        if (g['index'] != i or c['output'] != g['text'] or cache['text'] != g['text']
                or cache['request_hash'] != digest(cache['request_identity'])
                or len(cache['output_token_ids']) != g['output_tokens']):
            raise ValueError('Raw generation cache is incomplete or misaligned')
    reference = ROOT / 'data/eval_references/alpaca_eval2_gpt-4o-2024-05-13_official_reference_outputs.json'
    config, references, info = ref.reference_inputs('alpacaeval2', reference)
    combined = read(source / 'results.json')
    if combined['status'] != 'complete' or combined['candidate_sha256'] != sha256_file(source / 'candidate_outputs.json'):
        raise ValueError('Thinking-on combined result incomplete or changed')
    jobs = [(name, source / name, source) for name in ('mini', 'luna')]
    for variant in ('base', 'secpo', 'secpo_lr6e5', 'secpo_lr4e5', 'base_thinking'):
        inputs = source if variant == 'base_thinking' else ROOT / 'outputs/qwen38_lambda_recommended_20260916' / variant / 'alpacaeval2'
        jobs.append(('gpt6/' + variant, GPT6 / variant / 'alpacaeval2', inputs))
    results = {}
    for name, directory, inputs in jobs:
        result, manifest = read(directory / 'results.json'), read(directory / 'manifest.json')
        answers = read(inputs / 'candidate_outputs.json')
        annotations = read(directory / 'annotations.json')
        if result['status'] != 'complete' or result['n'] != 805 or result['smoke']:
            raise ValueError('Incomplete full judge result')
        if result['manifest_sha256'] != digest(manifest):
            raise ValueError('Judge manifest mismatch')
        expected = digest(answers) if name in ('mini', 'luna') else sha256_file(inputs / 'candidate_outputs.json')
        if manifest['candidate_sha256'] != expected:
            raise ValueError('Judge candidate hash mismatch')
        ref.validate_annotations(annotations, answers, references, config.identity_keys)
        recomputed = ref.score_annotations('alpacaeval2', annotations,
            metric_asset=ROOT / 'data/eval_references/alpacaeval_df_gamed.csv',
            metric_asset_sha256=ref.ALPACA_METRIC_SHA256)
        for key in ('win_rate', 'discrete_win_rate', 'length_controlled_winrate'):
            if abs(float(result['metrics'][key]) - float(recomputed[key])) > 1e-8:
                raise ValueError('Saved metric differs from recomputation')
        results[name] = dict(n=805, directory=str(directory), metrics=result['metrics'])
    note = ('Thinking supervisor reports failure because final stdout json.dumps raised TypeError '
            'after complete results/progress were saved. All seven judge results independently '
            'validated and metrics recomputed; original status/logs retained unchanged.')
    report = dict(at=now(), verified=True, generations=805, raw_caches=805,
        length_limited=sum(g['finish_reason']=='length' for g in generations),
        empty_final=sum(g['text']=='' for g in generations), reference=info, results=results,
        supervisor_note=note)
    write_json(DEST / 'audit/evaluation_readiness.json', report)
    return report


def copy_verified(source, destination, label):
    destination.mkdir(parents=True, exist_ok=True)
    command = ['rsync', '-a', '--no-owner', '--no-group']
    command += ['--exclude=' + pattern for pattern in EXCLUDE]
    command += [str(source) + '/', str(destination) + '/']
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    import fnmatch
    inventory = []
    for base, dirs, files in os.walk(source):
        dirs[:] = sorted(d for d in dirs if not any(fnmatch.fnmatchcase(d, p) for p in EXCLUDE))
        for name in sorted(files):
            if any(fnmatch.fnmatchcase(name, p) for p in EXCLUDE):
                continue
            p = Path(base) / name
            relative = p.relative_to(source)
            out = destination / relative
            if p.is_symlink() or out.is_symlink():
                raise ValueError('Unexpected artifact symlink; explicit preservation needed')
            checksum = sha256_file(p)
            if checksum != sha256_file(out):
                raise ValueError('Backup/source checksum mismatch')
            inventory.append(dict(path=str(relative), bytes=p.stat().st_size, sha256=checksum))
    if not inventory:
        raise ValueError('Empty artifact snapshot')
    write_json(DEST / 'audit' / (label + '.json'), dict(source=str(source), destination=str(destination),
        at=now(), files=inventory))
    return dict(source=str(source), destination=str(destination), files=len(inventory),
                bytes=sum(r['bytes'] for r in inventory))


def main():
    os.umask(0o077)
    DEST.mkdir(parents=True, exist_ok=True)
    with (DEST / 'archive.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stage('validating_complete_evaluations')
        evaluation = audit_results()
        stage('collecting_remote_snapshot_and_verifying_models')
        with ThreadPoolExecutor(max_workers=2) as pool:
            remote_future = pool.submit(snapshot_remote)
            models_future = pool.submit(verify_saved_models)
            models = models_future.result()
            remote = remote_future.result()
        expected = read(THINKING / 'model_validation.json')['files']
        if remote['model_files'] != expected:
            raise ValueError('Remote model differs from the evaluated model inventory')
        stage('snapshotting_local_artifacts')
        copies = {}
        for label, source in (
            ('thinking_run', THINKING), ('gpt6_judging', GPT6),
            ('judge_shift_analysis', Path('/data2/jaewonc/secpo_analysis/judge_shift_20260923'))):
            copies[label] = copy_verified(source, DEST / label, label)
        for name in ('scripts','src','configs','docs','tests','data'):
            copies['workspace_' + name] = copy_verified(ROOT / name, DEST / 'workspace' / name, 'workspace_' + name)
        for name in ('pyproject.toml', 'README.md', 'release_manifest.json'):
            shutil.copy2(ROOT / name, DEST / 'workspace' / name)
            if sha256_file(ROOT / name) != sha256_file(DEST / 'workspace' / name):
                raise ValueError('Workspace metadata differs')
        shutil.copy2(ROOT / 'docs/LAMBDA_THINKING_BACKUP.md', DEST / 'README.md')
        status = dict(status='complete', safe_to_shutdown=True, host=launch.HOST,
            completed_at=now(), instance_terminated=False, files_deleted=False,
            evaluation=evaluation, copied=copies, existing_models=models,
            prior_complete_backup=str(OLD), remote_snapshot_files=len(remote['files']),
            remote_model_fresh_sha256_verified=True, failures=[], exclusions=remote['excluded'])
        write_json(DEST / 'status.json', status)
        write_json(ROOT / 'outputs/lambda_thinking_shutdown_backup_20260923.json',
            dict(status='complete', safe_to_shutdown=True, backup=str(DEST), completed_at=now()))
        print(json.dumps(dict(status='complete', safe_to_shutdown=True, backup=str(DEST))), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if DEST.exists():
            write_json(DEST / 'status.json', dict(status='failed', safe_to_shutdown=False,
                at=now(), error_type=type(error).__name__,
                message=str(error) if isinstance(error, ValueError) else 'Inspect collector output'))
        raise
