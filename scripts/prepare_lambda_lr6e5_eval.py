#!/usr/bin/env python3
"""Materialize isolated lr6e5 routing without editing an active run's sources.

Only model routing and protocol-comparison support differ from the original
Lambda code. All rendering, sampling, scoring and judging code is copied
byte-for-byte. Credentials are neither copied nor linked into this snapshot.
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from secpo.tinker.evaluation_common import write_json

CODE = ROOT / 'runtime/lambda_suite_20260916/lr6e5_code'
RUN = ROOT / 'outputs/qwen38_lambda_recommended_20260916'


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Frozen routing source differs from the reviewed template')
    return text.replace(old, new)


def adapt_backend(text):
    text = replace_once(text,
        'cut_at_stop, digest, evaluation_model_id, evaluation_output_root, write_json,',
        'cut_at_stop, digest, evaluation_output_root, write_json,')
    text = replace_once(text,
        'from secpo.tinker.evaluation_variants import sampler_model_kwargs',
        'from secpo.tinker.evaluation_variants import sampler_model_kwargs, evaluation_model_id')
    text = replace_once(text, "variant not in {'base', 'secpo'}", "variant not in {'base', 'secpo', 'secpo_lr6e5'}")
    text = replace_once(text, "if variant == 'secpo' and not config.get('adapter_sha256'):",
                        "if variant != 'base' and not config.get('adapter_sha256'):")
    text = replace_once(text,
        "prior_path = evaluation_output_root(variant, None if agent else 16384) / benchmark / 'manifest.json'",
        "comparison_variant = 'secpo' if variant == 'secpo_lr6e5' else variant\n"
        "    prior_path = evaluation_output_root(comparison_variant, None if agent else 16384) / benchmark / 'manifest.json'")
    text = replace_once(text,
        "stable = lambda m: {k: v for k, v in m.items() if k != 'implementation_sha256'}\n    if stable(prior) != stable(manifest):",
        "stable = lambda m: {k: v for k, v in m.items() if k != 'implementation_sha256'}\n"
        "    expected = {**prior, 'checkpoint': evaluation_model_id(variant)}\n"
        "    if prior['checkpoint'] != evaluation_model_id(comparison_variant):\n"
        "        raise ValueError('Comparison checkpoint identity differs')\n"
        "    if stable(expected) != stable(manifest):")
    text = replace_once(text, '    return config\n\n\ndef make_sampler',
        "    if variant == 'secpo_lr6e5':\n"
        "        peer_path = ROOT / 'outputs/qwen38_lambda_recommended_20260916/secpo' / benchmark / 'manifest.json'\n"
        "        peer = json.loads(peer_path.read_text())\n"
        "        exclude = {'checkpoint', 'inference', 'comparison_manifest_sha256', 'implementation_sha256'}\n"
        "        protocol = lambda value: {k: v for k, v in value.items() if k not in exclude}\n"
        "        if protocol(peer) != protocol(manifest):\n"
        "            raise ValueError('lr6e5 protocol differs from the active SecPO experiment')\n"
        "        weight_keys = {'variant', 'checkpoint', 'served_model', 'adapter_sha256'}\n"
        "        engine = lambda value: {k: v for k, v in value.items() if k not in weight_keys}\n"
        "        if engine(peer['inference']) != engine(config):\n"
        "            raise ValueError('lr6e5 inference differs beyond model weights')\n"
        "    return config\n\n\ndef make_sampler")
    return text


def adapt_agents(text):
    text = replace_once(text, 'CheckpointSampler, MODEL, TOKENIZER_REVISION, digest, evaluation_model_id,',
                        'CheckpointSampler, MODEL, TOKENIZER_REVISION, digest,')
    text = replace_once(text, 'from secpo.tinker.agent_eval import BENCHMARKS',
                        'from secpo.tinker.evaluation_variants import MODEL_VARIANTS, evaluation_model_id\n'
                        'from secpo.tinker.agent_eval import BENCHMARKS')
    return replace_once(text, 'choices=["secpo", "base"], default="secpo"',
                        'choices=MODEL_VARIANTS, default="secpo"')


def main():
    frozen = json.loads((RUN / 'source_identity.json').read_text())
    for name, sha in frozen.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
            raise ValueError(f'Active experiment source changed: {name}')
    if CODE.exists():
        raise FileExistsError('The isolated code snapshot already exists; do not overwrite it')
    sources = list((ROOT / 'src/secpo').rglob('*.py'))
    sources += [ROOT / f'scripts/evaluate_tinker_{kind}.py' for kind in ('lm', 'agents', 'reference')]
    sources += [ROOT / f'configs/evaluate_{kind}_utility.json' for kind in ('alpacaeval2', 'sep')]
    changes = {'src/secpo/inference/lambda_eval.py': adapt_backend,
               'scripts/evaluate_tinker_agents.py': adapt_agents}
    provenance = {}
    for path in sources:
        name = str(path.relative_to(ROOT))
        source = path.read_text()
        transformed = changes[name](source) if name in changes else source
        target = CODE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name in changes:
            target.write_text(transformed)
        else:
            shutil.copy2(path, target)
        provenance[name] = dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                                model_routing_only=name in changes)
    # Links exist only in the local execution workspace. They are not followed
    # when the explicit source inventory is copied into result artifacts.
    for name in ('data', 'outputs', 'runtime', 'third_party'):
        (CODE / name).symlink_to(ROOT / name, target_is_directory=True)
    write_json(CODE / 'provenance.json', dict(original_run=str(RUN), files=provenance,
               changed_sources=sorted(changes), credential_files_copied=False))
    print(json.dumps(dict(code=str(CODE), files=len(provenance), changed_sources=sorted(changes))))


if __name__ == '__main__':
    main()
