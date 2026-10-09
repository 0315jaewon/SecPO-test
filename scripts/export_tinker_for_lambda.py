#!/usr/bin/env python3
"""Download a saved SecPO inference adapter without logging signed URLs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from secpo.tinker.evaluation_common import load_credentials, final_checkpoint, write_json, now
from secpo.tinker.evaluation_variants import evaluation_model_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-variant', choices=['secpo', 'secpo_lr6e5'], default='secpo')
    args = parser.parse_args()
    import tinker
    load_credentials(ROOT / 'secpo_eval_keys.env')
    dest = ROOT / 'runtime/lambda_suite_20260916' / (args.model_variant + '_export')
    dest.mkdir(parents=True, exist_ok=True)
    checkpoint = evaluation_model_id(args.model_variant)
    manifest = dest / 'export_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['checkpoint'] != checkpoint:
        raise ValueError('Existing export belongs to a different checkpoint')
    rest = tinker.ServiceClient(timeout=600, max_retries=2).create_rest_client()
    info = rest.get_weights_info_by_tinker_path(checkpoint).result(timeout=120)
    metadata = info.model_dump(mode='json')
    if (metadata['base_model'] != 'Qwen/Qwen3.8-27B' or metadata['lora_rank'] != 64
            or not metadata['is_lora'] or metadata['train_unembed']
            or not metadata['train_mlp'] or not metadata['train_attn']):
        raise ValueError('Unexpected checkpoint architecture or adapter configuration')
    print(json.dumps({'checkpoint': checkpoint, 'weights_info': metadata}), flush=True)
    archive = dest / 'checkpoint.tar'
    if not archive.exists():
        response = rest.get_checkpoint_archive_url_from_tinker_path(checkpoint).result(timeout=600)
        # The signed URL and credential never enter logs or the saved manifest.
        with urllib.request.urlopen(response.url, timeout=120) as stream, archive.with_suffix('.partial').open('wb') as output:
            while chunk := stream.read(1024 * 1024):
                output.write(chunk)
        archive.with_suffix('.partial').rename(archive)
    unpacked = dest / 'adapter'
    unpacked.mkdir(exist_ok=True)
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            target = (unpacked / member.name).resolve()
            if not target.is_relative_to(unpacked.resolve()) or member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
                raise ValueError('Unsafe checkpoint archive entry')
        bundle.extractall(unpacked, filter='data')
    files = {str(p.relative_to(unpacked)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in unpacked.rglob('*') if p.is_file()}
    write_json(dest / 'export_manifest.json', dict(checkpoint=checkpoint, weights_info=metadata,
               exported_at=now(), files=files))
    print(json.dumps({'status': 'exported', 'files': list(files), 'output': str(dest)}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__}), flush=True)
        raise SystemExit(1) from None
