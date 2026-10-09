#!/usr/bin/env python3
"""Verify and preserve the completed September 27 Lambda evaluations on REDS.

Archival only: no inference, judging, evaluator changes, deletion, or shutdown.
Reuses the checksum-copy helpers from the earlier audited archive workflow.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess

import archive_lambda_static as archive

ROOT = archive.ROOT
DEST = Path('/data2/jaewonc/secpo_backups/lambda_full_sep_20261001')
OLD = archive.OLD
HOST = 'ubuntu@209.20.159.207'
RUNS = {
    'full_utility': Path('/data2/jaewonc/secpo_evals/qwen38_full_sep_20260927'),
    'sep_original_judges': Path('/data2/jaewonc/secpo_evals/qwen38_sep_original_20260927'),
    'base_static_security': Path('/data2/jaewonc/secpo_evals/qwen38_native_base_static_20260927'),
}
archive.DEST = DEST
archive.HOST = HOST
read, write, sha, now = archive.read, archive.write, archive.sha, archive.now

archive.REMOTE_SNAPSHOT = r'''
import base64, hashlib, json, os, platform, subprocess
from pathlib import Path
roots = {
    'full_sep_server': Path('/lambda/nfs/jaewon/secpo_full_sep_20260927'),
    'native_base_server': Path('/lambda/nfs/jaewon/secpo_native_base_static_20260927'),
}
active = []
for p in Path('/proc').glob('[0-9]*/cmdline'):
    try: args = p.read_bytes().split(b'\0')
    except (FileNotFoundError, PermissionError): continue
    if b'vllm.entrypoints.openai.api_server' in args:
        active.append(int(p.parent.name))
gpu_processes = subprocess.check_output(
    ['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'], text=True).strip()
if active or gpu_processes:
    raise ValueError('Remote model server/GPU process is still active')
files = []
for label, root in roots.items():
    if not root.is_dir(): raise ValueError('Remote project missing')
    for base, dirs, names in os.walk(root):
        dirs.sort()
        if any((Path(base)/d).is_symlink() for d in dirs): raise ValueError('Directory symlink')
        for name in sorted(names):
            p = Path(base)/name
            if p.is_symlink(): raise ValueError('Unexpected source symlink')
            if name.endswith(('.env','.pem','.key')) or name.startswith('id_'):
                raise ValueError('Unexpected credential-like file; requires explicit exclusion')
            before = p.stat()
            if before.st_size > 100*1024*1024: raise ValueError('Unexpected oversized artifact')
            data = p.read_bytes()
            after = p.stat()
            if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
                raise ValueError('Remote source changed during snapshot')
            files.append(dict(path=label+'/'+str(p.relative_to(root)), bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(), data_b64=base64.b64encode(data).decode()))
runtime = Path('/home/ubuntu/secpo_full_sep_20260927')
if set(p.name for p in runtime.iterdir()) != {'venv'}:
    raise ValueError('Additional runtime artifacts require explicit preservation')
models = {key:json.loads((roots[label]/'model_validation.json').read_text())
          for key,label in [('secpo_lr8e5','full_sep_server'),('base','native_base_server')]}
for item in models.values():
    if item['status'] != 'complete': raise ValueError('Initial model validation incomplete')
    directory = Path(item['directory'])
    if set(p.name for p in directory.iterdir()) != {r['path'] for r in item['files']}:
        raise ValueError('Model file inventory changed')
    for row in item['files']:
        p = directory/row['path']
        if not p.is_file() or p.stat().st_size != row['bytes']:
            raise ValueError('Model file missing or size changed')
        if 'mtime_ns' in row and p.stat().st_mtime_ns != row['mtime_ns']:
            raise ValueError('Base model file changed since initial SHA validation')
packages = json.loads(subprocess.check_output([str(runtime/'venv/bin/python'),'-m','pip','list','--format=json']))
print(json.dumps(dict(files=files, models=models, packages=packages,
    environment=dict(platform=platform.platform(), python=platform.python_version(),
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True)),
    project_inventory={str(p):sorted(c.name for c in p.iterdir()) for p in
        [Path('/lambda/nfs/jaewon'),Path('/lambda/nfs/jaewon/secpo_qwen38_eval_20260916'),runtime]},
    active_model_servers=active, gpu_processes=gpu_processes,
    model_check='Fresh inventory/size check against full SHA-256 validation at instance setup; base mtimes also checked.')))
'''


def require(condition, message):
    if not condition:
        raise ValueError(message)


def completed_worker(path):
    state = read(path)
    require(state['status'] == 'complete' and state['exit_code'] == 0,
            'Incomplete worker: ' + str(path))


def check_results():
    active = []
    names = {'evaluate_lambda_full_lm.py', 'evaluate_lambda_sep_original.py',
             'evaluate_native_static_base.py', 'launch_native_static_base.py',
             'launch_lambda_sep_original.py', 'resume_lambda_sep_original_judging.py'}
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError, PermissionError):
            continue
        if any(arg == str(ROOT/'scripts'/name) for name in names for arg in args):
            active.append(path.parent.name)
    require(not active, 'Evaluation processes still running: ' + str(active))
    records = {}
    for benchmark, count in [('mmlu_pro',12032),('minerva_math',5000)]:
        directory = RUNS['full_utility']/'secpo'/benchmark
        completed_worker(directory/'full_worker.json')
        progress, result = read(directory/'progress.json'), read(directory/'results.json')
        require(progress['status'] == 'complete' and progress['completed'] == count,
                'Incomplete utility progress')
        require(sum(len(rows) for rows in result['samples'].values()) == count,
                'Missing per-question utility results')
        for task, rows in result['samples'].items():
            require(len(rows) == result['n-samples'][task]['effective'] and
                    len({r['doc_id'] for r in rows}) == len(rows), 'Missing/duplicate question IDs')
        score = (result['groups']['mmlu_pro']['exact_match,custom-extract'] if benchmark == 'mmlu_pro'
                 else result['results']['minerva_math']['math_verify,none'])
        cached = len(list((directory/'completion_cache').glob('*.json')))
        require(cached >= count, 'Missing utility completion cache')
        records['secpo_lr8e5/'+benchmark] = dict(samples=count, accuracy=score, cached_responses=cached)

    directory = RUNS['sep_original_judges']/'secpo/sep_static'
    require(read(RUNS['sep_original_judges']/'supervisor.json')['status'] == 'complete',
            'SEP supervisor incomplete')
    for name in ('generate_worker.json','judge_worker.json'):
        completed_worker(directory/name)
    result = read(directory/'results.json')
    require(result['status'] == 'complete' and result['rows'] == 54960, 'SEP incomplete')
    identifiers = result['judge_protocol']['config']['judge_identifiers']
    unique_successes, positive_rows, vote_count = set(), 0, 0
    for attack, expected in result['per_attack'].items():
        rows = list((directory/'rows'/attack).glob('*.json'))
        judgments = list((directory/'judgments'/attack).glob('*.json'))
        require(len(rows) == len(judgments) == expected['examples'] == 9160, 'SEP row count mismatch')
        require({p.name for p in rows} == {p.name for p in judgments}, 'SEP judgment IDs mismatch')
        ids, successes = set(), 0
        for path in judgments:
            row = read(path)
            original = read(directory/'rows'/attack/path.name)
            require(row['attack'] == attack and row['example_id'] not in ids, 'Duplicate/wrong SEP ID')
            ids.add(row['example_id'])
            require(row['output'] == original['output'] and
                    row['completion_cache_key'] == original['completion_cache_key'], 'Judged output changed')
            require((directory/'completion_cache'/(row['completion_cache_key']+'.json')).is_file(),
                    'Missing SEP response cache')
            judgment = row['judgment']
            if row['witness_present']:
                positive_rows += 1
                require(judgment['status'] == 'complete' and judgment['completed_votes'] == 9 and
                        judgment['error_votes'] == 0 and len(judgment['votes']) == 9,
                        'Unresolved SEP judgment')
                votes = judgment['votes']
                require(all(v['status'] == 'ok' for v in votes) and
                        Counter(v['judge_identifier'] for v in votes) == Counter({i:3 for i in identifiers}),
                        'Missing original-judge votes')
                require(row['attack_success'] == (judgment['yes_votes'] == 9), 'Wrong unanimous outcome')
                vote_count += 9
            else:
                require(judgment['status'] == 'witness_miss' and row['attack_success'] is False,
                        'Incorrect witness-gate outcome')
            if row['attack_success']:
                successes += 1
                unique_successes.add(row['example_id'])
        require(ids == set(range(9160)) and successes == expected['successes'] and expected['pending'] == 0,
                'SEP per-attack totals mismatch')
    require(len(unique_successes) == result['combined']['static']['successes'] and
            result['combined']['static']['pending'] == 0, 'SEP combined result mismatch')
    http_count = len(list((directory/'judge_http_cache').glob('*.json')))
    require(http_count == vote_count, 'Judge HTTP cache/vote count mismatch')
    records['secpo_lr8e5/sep_static'] = dict(examples=9160, attack_generations=54960,
        witness_positive_rows=positive_rows, valid_judge_votes=vote_count,
        judge_http_responses=http_count, successful_examples=len(unique_successes),
        combined_asr=result['combined']['static']['asr'])

    require(read(RUNS['base_static_security']/'supervisor.json')['status'] == 'complete',
            'Agentic supervisor incomplete')
    for benchmark, attacked, total in [('agentdojo',949,984),('agentdyn',560,588)]:
        directory = RUNS['base_static_security']/'base'/benchmark
        completed_worker(directory/'full_worker.json')
        result = read(directory/'results.json')
        tasks, episodes = list((directory/'tasks').glob('*.json')), list((directory/'episodes').rglob('*.json'))
        require(result['status'] == 'complete' and result['errors'] == 0 and
                result['attacked_completed'] == attacked and result['completed'] == total and
                len(tasks) == len(episodes) == total, 'Agentic results incomplete')
        require(all(read(p)['status'] == 'complete' for p in tasks), 'Incomplete agent task')
        records['base/'+benchmark] = dict(episodes=total, attacked_episodes=attacked,
            cached_responses=len(list((directory/'completion_cache').glob('*.json'))),
            macro_asr=result['macro_attack_success_rate'], micro_asr=result['micro_attack_success_rate'])
    cancelled = RUNS['full_utility']/'secpo/sep_static'
    require(not (cancelled/'rows').exists() and not (cancelled/'completion_cache').exists(),
            'Cancelled SEP output was unexpectedly restored')
    write(DEST/'audit/evaluations.json', dict(at=now(), verified=True, runs=records,
        cancelled_run_note='Old Luna SEP was intentionally cancelled/deleted; its metadata is retained, not counted as a current failed evaluation.'))
    return records


def check_models(remote):
    models = {}
    for variant in ('base','secpo_lr8e5','secpo_lr6e5','secpo_lr4e5'):
        previous = read(OLD/'audit/tasks'/(variant+'.json'))
        manifest = OLD/'audit'/(variant+'.remote_sha256.jsonl')
        require(previous['status'] == 'complete' and sha(manifest) == previous['manifest_sha256'],
                'Previous model checksum manifest changed')
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        directory = OLD/'models'/variant
        require({p.name for p in directory.iterdir()} == {r['path'] for r in rows}, 'Local model inventory differs')
        for row in rows:
            p = directory/row['path']
            require(row['kind'] == 'file' and not p.is_symlink() and p.is_file() and
                    p.stat().st_size == row['bytes'], 'Local model file missing/size changed')
        if variant in remote['models']:
            normalize = lambda entries: sorted([{k:r[k] for k in ('path','bytes','sha256')}
                                                for r in entries], key=lambda r:r['path'])
            require(normalize(rows) == normalize(remote['models'][variant]['files']),
                    'Evaluated Lambda model differs from REDS backup')
        models[variant] = dict(directory=str(directory), files=len(rows), bytes=sum(r['bytes'] for r in rows),
            earlier_sha256_verified_at=previous['verified_at'], fresh_inventory_and_size_check=True,
            fresh_full_weight_sha256_check=False, matches_evaluated_remote_manifest=variant in remote['models'])
    checkpoint_path = OLD/'checkpoints/manifest.json'
    checkpoints = read(checkpoint_path)
    require(checkpoints['status'] == 'complete', 'Checkpoint backup incomplete')
    for item in checkpoints['backups']:
        path = Path(item['archive'])
        require(path.is_file() and path.stat().st_size == item['bytes'] and sha(path) == item['sha256'],
                'Sampler archive missing/corrupt')
    record = dict(models=models, checkpoint_manifest=str(checkpoint_path),
        freshly_verified_sampler_archives=len(checkpoints['backups']),
        checkpoint_retention_last_checked=checkpoints['verified_at'],
        training_states='On Tinker, not Lambda; no new retention API call was made.', at=now())
    write(DEST/'audit/models.json', record)
    return record


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    before = sha(source)
    shutil.copy2(source, target)
    require(before == sha(target) == sha(source), 'File copy checksum mismatch')
    return dict(source=str(source), destination=str(target), bytes=target.stat().st_size, sha256=before)


def main():
    os.umask(0o077)
    DEST.mkdir(parents=True, exist_ok=True)
    with (DEST/'archive.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            archive.stage('checking_all_five_evaluations')
            evaluations = check_results()
            archive.stage('collecting_remote_logs_and_metadata')
            remote = archive.snapshot_remote()
            archive.stage('copying_and_sha256_verifying_results_and_caches')
            copies = {}
            with ThreadPoolExecutor(max_workers=4) as pool:
                model_future = pool.submit(check_models, remote)
                jobs = {label:pool.submit(archive.copy_verified,label,source,DEST/'runs'/label)
                        for label,source in RUNS.items()}
                for label, future in jobs.items():
                    copies[label] = future.result()
                    print(json.dumps(dict(copied=label,**copies[label])), flush=True)
                models = model_future.result()
            archive.stage('preserving_source_code_datasets_and_environments')
            for name in ('scripts','src','configs','docs','tests','third_party'):
                label = 'workspace_'+name
                copies[label] = archive.copy_verified(label,ROOT/name,DEST/'workspace'/name)
            for benchmark in ('agentdojo','agentdyn'):
                label = 'benchmark_'+benchmark
                copies[label] = archive.copy_verified(label,ROOT/'runtime/benchmarks'/benchmark,
                                                      DEST/'benchmark_sources'/benchmark)
            individual = []
            for name in ('README.md','pyproject.toml','release_manifest.json'):
                individual.append(copy_file(ROOT/name, DEST/'workspace'/name))
            for benchmark in ('mmlu_pro','minerva_math'):
                source = ROOT/'outputs/qwen38_lambda_recommended_20260916/secpo'/benchmark/'manifest.json'
                individual.append(copy_file(source, DEST/'prior_subset_manifests'/benchmark/'manifest.json'))
            dataset = Path('/home/jaewonc/SecAlign_cont/data/SEP_dataset_test.json')
            expected_sha = read(RUNS['sep_original_judges']/'secpo/sep_static/manifest.json')['dataset_sha256']
            require(sha(dataset) == expected_sha, 'SEP test dataset changed')
            individual.append(copy_file(dataset,DEST/'datasets/SEP_dataset_test.json'))
            individual.append(copy_file(ROOT/'docs/LAMBDA_FULL_SEP_BACKUP.md',DEST/'README.md'))
            write(DEST/'audit/individual_files.json',individual)
            for name in ('lm','agentdojo','agentdyn'):
                python = ROOT/'runtime/eval_envs'/name/'bin/python'
                packages = json.loads(subprocess.check_output([str(python),'-m','pip','list','--format=json']))
                version = subprocess.check_output([str(python),'--version'],text=True).strip()
                write(DEST/'environments'/('reds_'+name+'.json'),dict(python=version,packages=packages))
            write(DEST/'environments/lambda_vllm.json',dict(packages=remote['packages'],environment=remote['environment']))
            status = dict(status='complete',safe_to_shutdown=True,host=HOST,completed_at=now(),
                instance_terminated=False,files_deleted=False,evaluations=evaluations,copies=copies,
                remote_files=len(remote['files']),remote_bytes=sum(f['bytes'] for f in remote['files']),
                models=models,prior_backup=str(OLD),
                exclusions=['Credentials and SSH keys','Virtual environments (package inventories saved)',
                    'Rebuildable compilation/download caches','Duplicate model weights (existing REDS backup verified)',
                    'Previously cancelled/deleted Luna SEP generations (not restored)'])
            write(DEST/'status.json',status)
            print(json.dumps(status),flush=True)
        except Exception as error:
            write(DEST/'status.json',dict(status='failed',safe_to_shutdown=False,at=now(),
                error_type=type(error).__name__,message=str(error)))
            raise


if __name__ == '__main__':
    main()
