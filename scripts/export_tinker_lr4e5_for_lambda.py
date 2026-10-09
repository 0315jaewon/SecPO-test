#!/usr/bin/env python3
"""Use the verified exporter with explicit completed lr4e5 checkpoint routing."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from secpo.tinker.evaluation_common import load_credentials, now, write_json

DEST = ROOT / 'runtime/lambda_suite_20260916/secpo_lr4e5_export'


def stage(name, **kwargs):
    record = dict(status=name, at=now(), **kwargs)
    write_json(DEST / 'export_status.json', record)
    print(json.dumps(record), flush=True)


def main():
    import tinker
    import requests
    DEST.mkdir(parents=True, exist_ok=True)
    load_credentials(ROOT / 'secpo_eval_keys.env')
    result = json.loads((ROOT / 'outputs/qwen38_27b_tinker_lr4e5/result.json').read_text())
    if result['steps_completed'] != 297:
        raise ValueError('Incomplete training')
    checkpoint = result['final_sampler_path']
    manifest = DEST / 'export_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['checkpoint'] != checkpoint:
        raise ValueError('Different checkpoint already exported')
    stage('checking_metadata', checkpoint=checkpoint)
    rest = tinker.ServiceClient(timeout=600, max_retries=2).create_rest_client()
    metadata = rest.get_weights_info_by_tinker_path(checkpoint).result(timeout=120).model_dump(mode='json')
    if (metadata['base_model'] != 'Qwen/Qwen3.8-27B' or metadata['lora_rank'] != 64
            or not metadata['is_lora'] or metadata['train_unembed']
            or not metadata['train_mlp'] or not metadata['train_attn']):
        raise ValueError('Unexpected adapter architecture')
    archive = DEST / 'checkpoint.tar'
    if not archive.exists():
        stage('waiting_for_tinker_archive', checkpoint=checkpoint)
        future = rest.get_checkpoint_archive_url_from_tinker_path(checkpoint)
        start = time.monotonic()
        while not future.done():
            if time.monotonic() - start > 1800:
                raise TimeoutError('Tinker archive preparation exceeded 30 minutes')
            time.sleep(15)
            stage('waiting_for_tinker_archive', elapsed_seconds=round(time.monotonic() - start))
        response = future.result()
        stage('downloading_archive')
        with requests.Session() as session:
            session.trust_env = False
            with session.get(response.url, stream=True, timeout=(20, 120)) as stream:
                stream.raise_for_status()
                with archive.with_suffix('.partial').open('wb') as output:
                    for chunk in stream.iter_content(1024 * 1024):
                        output.write(chunk)
        archive.with_suffix('.partial').rename(archive)
    stage('validating_archive')
    unpacked = DEST / 'adapter'
    unpacked.mkdir(exist_ok=True)
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            target = (unpacked / member.name).resolve()
            if (not target.is_relative_to(unpacked.resolve()) or member.issym() or member.islnk()
                    or not (member.isdir() or member.isfile())):
                raise ValueError('Unsafe archive entry')
        bundle.extractall(unpacked, filter='data')
    files = {}
    for path in unpacked.rglob('*'):
        if path.is_file():
            with path.open('rb') as stream:
                files[str(path.relative_to(unpacked))] = hashlib.file_digest(stream, 'sha256').hexdigest()
    write_json(manifest, dict(checkpoint=checkpoint, weights_info=metadata, exported_at=now(), files=files))
    stage('complete', checkpoint=checkpoint, files=list(files))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        stage('failed', error_type=type(error).__name__)
        raise SystemExit(1) from None
