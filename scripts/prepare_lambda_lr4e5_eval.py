#!/usr/bin/env python3
"""Copy the frozen lr6e5 implementation, changing only lr4e5 model routing.

Never edits the predecessor's sources or artifacts. Explicit inventories avoid
copying credentials, cache directories, or following runtime symlinks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from secpo.tinker.evaluation_common import write_json

RUN = ROOT / 'outputs/qwen38_lambda_recommended_20260916'
SOURCE = RUN / 'secpo_lr6e5/code_snapshot'
CODE = ROOT / 'runtime/lambda_suite_20260916/lr4e5_code'


def remap(text):
    return text.replace('lr6e5', 'lr4e5').replace('lr6e-5', 'lr4e-5')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-only', action='store_true')
    args = parser.parse_args()
    result = json.loads((ROOT / 'outputs/qwen38_27b_tinker_lr4e5/result.json').read_text())
    if result['steps_completed'] != 297:
        raise ValueError('Training is not complete')
    checkpoint = result['final_sampler_path']
    if not args.config_only:
        if CODE.exists():
            raise FileExistsError('Do not overwrite an isolated experiment')
        prior = json.loads((SOURCE / 'provenance.json').read_text())
        inventory = {}
        for name, item in prior['files'].items():
            source = SOURCE / name
            raw = source.read_bytes()
            if hashlib.sha256(raw).hexdigest() != item['sha256']:
                raise ValueError('Frozen predecessor changed: ' + name)
            transformed = remap(raw.decode()).encode()
            dest = CODE / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(transformed)
            inventory[name] = dict(source_sha256=item['sha256'],
                sha256=hashlib.sha256(transformed).hexdigest(), model_routing_only=raw != transformed)
        for name in ('data', 'outputs', 'runtime', 'third_party'):
            (CODE / name).symlink_to(ROOT / name, target_is_directory=True)
        write_json(CODE / 'provenance.json', dict(original_run=str(RUN), source=str(SOURCE),
            files=inventory, changed_sources=[n for n, v in inventory.items() if v['model_routing_only']],
            credential_files_copied=False))
        old_checkpoint = json.loads((ROOT / 'configs/lambda_qwen38_secpo_lr6e5_recommended.json').read_text())['checkpoint']
        for name in ('merge_lr6e5_remote.py', 'server_lr6e5_remote.py'):
            source = ROOT / 'runtime/lambda_suite_20260916' / name
            target = source.with_name(remap(name))
            if target.exists():
                raise FileExistsError(target)
            target.write_text(remap(source.read_text()).replace(old_checkpoint, checkpoint))
    export = ROOT / 'runtime/lambda_suite_20260916/secpo_lr4e5_export/export_manifest.json'
    if export.exists():
        metadata = json.loads(export.read_text())
        if metadata['checkpoint'] != checkpoint:
            raise ValueError('Wrong adapter export')
        prior = json.loads((ROOT / 'configs/lambda_qwen38_secpo_lr6e5_recommended.json').read_text())
        config = {**prior, 'variant': 'secpo_lr4e5', 'served_model': 'secpo-suite-secpo_lr4e5',
                  'checkpoint': checkpoint, 'adapter_sha256': metadata['files']['adapter_model.safetensors']}
        target = ROOT / 'configs/lambda_qwen38_secpo_lr4e5_recommended.json'
        if target.exists() and json.loads(target.read_text()) != config:
            raise ValueError('Existing configuration differs')
        write_json(target, config)
    print(json.dumps(dict(status='prepared', code=str(CODE), checkpoint=checkpoint)))


if __name__ == '__main__':
    main()
