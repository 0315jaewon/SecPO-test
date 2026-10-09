#!/usr/bin/env python3
"""Publish the verified lr8e-5 merged backup; credentials stay in memory only."""

from __future__ import annotations

import argparse
import fcntl
import getpass
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone


PROJECT = Path(__file__).resolve().parents[1]
BACKUP = Path('/data2/jaewonc/secpo_backups/lambda_20260919')
SOURCE = BACKUP / 'models/secpo_lr8e5'
ROOT = Path('/data2/jaewonc/secpo_hf_uploads/qwen38_secpo_lr8e5')
STAGE = ROOT / 'model'
REPO = 'promptinjections/Qwen3.8-27B-SecPO-lr8e-5'
STATE = ROOT / 'status.json'
ADAPTER_SHA = 'a15e0c5fe8cb54a45cce05d98c4fc7fa8863e42dd466685dae6ad431a3bd0a3f'
BASE_REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'


def record(phase: str, **details):
    state = dict(phase=phase, updated_at=datetime.now(timezone.utc).isoformat(),
                 pid=os.getpid(), repo_id=REPO, **details)
    temporary = ROOT / 'status.json.tmp'
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(STATE)
    print(json.dumps(state), flush=True)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare():
    rows = [json.loads(line) for line in
            (BACKUP / 'audit/secpo_lr8e5.remote_sha256.jsonl').read_text().splitlines()]
    names = {row['path'] for row in rows}
    require(len(rows) == len(names) == 27, 'Unexpected source file manifest')
    require(names == {p.name for p in SOURCE.iterdir()}, 'Source file set changed')
    require(sum(row['bytes'] for row in rows) == 55586035734, 'Source size changed')
    require(all(Path(name).name == name for name in names), 'Unsafe manifest path')
    shards = {f'model-{i:05d}-of-00018.safetensors' for i in range(1, 19)}
    index = json.loads((SOURCE / 'model.safetensors.index.json').read_text())
    require(set(index['weight_map'].values()) == shards, 'Unexpected model index')
    STAGE.mkdir(mode=0o700, exist_ok=True)
    for i, row in enumerate(rows, 1):
        source = SOURCE / row['path']
        require(not source.is_symlink() and source.is_file(), 'Unsafe source')
        require(source.stat().st_size == row['bytes'], 'Source size mismatch')
        require(sha256(source) == row['sha256'], 'Backup checksum mismatch')
        target = STAGE / source.name
        if target.exists():
            require(os.path.samefile(source, target), 'Unexpected staged file')
        else:
            os.link(source, target)
        record('verifying_local_files', verified_files=i, total_files=len(rows))

    training = json.loads((PROJECT / 'outputs/qwen38_27b_tinker/config.json').read_text())
    result = json.loads((PROJECT / 'outputs/qwen38_27b_tinker/result.json').read_text())
    merge = json.loads((BACKUP / 'lambda_project/merge_complete.json').read_text())
    validation_path = BACKUP / 'lambda_project/merge_validation.json'
    validation = json.loads(validation_path.read_text())
    require(training['learning_rate'] == 8e-5 and training['lora_rank'] == 64,
            'Wrong training run')
    require(training['model_name'] == 'Qwen/Qwen3.8-27B', 'Wrong base model')
    require(training['expected_records'] == 18951 and result['steps_completed'] == 297,
            'Unexpected training counts')
    require(merge['status'] == validation['status'] == 'complete', 'Incomplete merge')
    require(merge['base_revision'] == BASE_REVISION, 'Wrong base revision')
    require(validation['adapter_sha256'] == ADAPTER_SHA, 'Wrong adapter')
    config = {k: v for k, v in training.items()
              if k not in {'output_dir', 'checkpoint_ttl_seconds', 'save_every'}}
    summary = dict(training_config=config, steps_completed=result['steps_completed'],
                   base_revision=BASE_REVISION, adapter_sha256=ADAPTER_SHA,
                   exported_lora_alpha=32, exported_lora_dropout=0,
                   merge_cookbook_commit=merge['cookbook_commit'],
                   artifact_format='full merged Hugging Face BF16 model')
    (STAGE / 'training_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    shutil.copyfile(PROJECT / 'docs/model_cards/qwen38_secpo_lr8e5.md', STAGE / 'README.md')
    shutil.copyfile(BACKUP / 'models/base/LICENSE', STAGE / 'LICENSE')
    shutil.copyfile(validation_path, STAGE / 'merge_validation.json')
    names.update({'README.md', 'LICENSE', 'training_summary.json', 'merge_validation.json'})
    checksums = {name: sha256(STAGE / name) for name in names - {r['path'] for r in rows}}
    checksums.update({row['path']: row['sha256'] for row in rows})
    (STAGE / 'SHA256SUMS').write_text(''.join(
        f'{checksums[name]}  {name}\n' for name in sorted(checksums)))
    checksums['SHA256SUMS'] = sha256(STAGE / 'SHA256SUMS')
    (ROOT / 'expected_checksums.json').write_text(json.dumps(checksums, indent=2) + '\n')
    return checksums


def verify_remote(api, checksums, *, complete):
    info = api.model_info(REPO)
    require(info.private, 'Refusing to upload into a public repository')
    files = {entry.path: entry for entry in api.list_repo_tree(REPO, revision=info.sha)
             if hasattr(entry, 'blob_id') and entry.path != '.gitattributes'}
    require(set(files) <= set(checksums), 'Unexpected existing repository files')
    if complete:
        require(set(files) == set(checksums), 'Remote files incomplete')
    for name, entry in files.items():
        local = STAGE / name
        require(entry.size == local.stat().st_size, 'Remote file size mismatch')
        if entry.lfs:
            require(entry.lfs.sha256 == checksums[name], 'Remote LFS checksum mismatch')
        else:
            # Git blobs use SHA-1 over their header and bytes, unlike LFS SHA-256.
            require(entry.size < 32 * 1024 * 1024, 'Unexpected large non-LFS file')
            data = local.read_bytes()
            blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
            require(entry.blob_id == blob, 'Remote Git blob checksum mismatch')
    return info.sha


def worker(token):
    os.environ['HF_HOME'] = str(ROOT / 'hf_cache')
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    # Do not emit HTTP headers, presigned URLs, or credentials into job logs.
    logging.disable(logging.CRITICAL)
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    with (ROOT / 'upload.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        api = HfApi(token=token)
        who = api.whoami()
        membership = next((o for o in who.get('orgs', [])
                           if o.get('name') == 'promptinjections'), None)
        require(membership is not None, 'Missing organization membership')
        require(who.get('auth', {}).get('accessToken', {}).get('role') != 'read',
                'Token is read-only')
        record('authenticated', account=who.get('name'))
        checksums = prepare()
        try:
            api.model_info(REPO)
        except RepositoryNotFoundError as error:
            require(error.response is not None and error.response.status_code == 404,
                    'Repository lookup failed; not a missing repo')
            api.create_repo(REPO, repo_type='model', private=True, exist_ok=False)
        verify_remote(api, checksums, complete=False)
        # A small first commit checks actual write permission before large transfer.
        api.upload_folder(repo_id=REPO, folder_path=STAGE, repo_type='model',
                          allow_patterns=['README.md', 'LICENSE', 'training_summary.json',
                                          'merge_validation.json', 'SHA256SUMS'],
                          commit_message='Document original SecPO lr8e-5 merged checkpoint')
        record('uploading', total_files=len(checksums),
               total_bytes=sum((STAGE / name).stat().st_size for name in checksums),
               private=True)
        api.upload_large_folder(repo_id=REPO, folder_path=STAGE, repo_type='model',
                                private=True, allow_patterns=sorted(checksums),
                                num_workers=4, print_report=True, print_report_every=60)
        record('verifying_remote_files')
        revision = verify_remote(api, checksums, complete=True)
        record('complete', revision=revision, private=True, verified_files=len(checksums),
               verified_weight_shards=18, url=f'https://huggingface.co/{REPO}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    if args.worker:
        token = sys.stdin.readline().strip()
        sys.stdin.close()
        try:
            require(token.startswith('hf_'), 'Invalid token format')
            worker(token)
        except Exception as error:
            # Never serialize raw API exceptions (they can contain signed URLs).
            response = getattr(error, 'response', None)
            record('failed', error_type=type(error).__name__,
                   http_status=getattr(response, 'status_code', None),
                   detail=str(error) if isinstance(error, ValueError) else None)
            raise SystemExit(1) from None
    else:
        token = getpass.getpass('HF upload token (hidden; not saved): ')
        with (ROOT / 'upload.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--worker'],
                                     stdin=subprocess.PIPE, stdout=log, stderr=log,
                                     start_new_session=True, close_fds=True, text=True)
            child.stdin.write(token + '\n')
            child.stdin.close()
        print(json.dumps(dict(started_pid=child.pid, status_file=str(STATE),
                              log_file=str(ROOT / 'upload.log'))))


if __name__ == '__main__':
    main()
