#!/usr/bin/env python3
"""Base-only launch configuration for the unchanged native static worker."""
import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from secpo.inference import native_static as native
from secpo.inference.lambda_eval import QWEN_SAMPLING
from secpo.tinker.evaluation_common import MODEL, TOKENIZER_REVISION

OUTPUT = Path('/data2/jaewonc/secpo_evals/qwen38_native_base_static_20260927')
PREVIOUS = Path('/data2/jaewonc/secpo_evals/qwen38_native_static_20260924')
ENDPOINT = 'http://127.0.0.1:18783/v1'
FILES = (*native.FILES, 'configs/lambda_qwen38_base_recommended.json',
         'scripts/evaluate_native_static_base.py', 'scripts/launch_native_static_base.py',
         'scripts/native_static_base_remote.py')


def inference_config(variant):
    config = native.read(ROOT / 'configs/lambda_qwen38_base_recommended.json')
    if (variant != 'base' or config['variant'] != 'base'
            or config['checkpoint'] != 'base:' + MODEL
            or config['base_revision'] != TOKENIZER_REVISION
            or config['sampling'] != QWEN_SAMPLING
            or config['max_model_len'] != 65536 or config['weight_dtype'] != 'bfloat16'
            or config['backend'] != 'vllm-0.29.0' or config.get('adapter_sha256')):
        raise ValueError('Base checkpoint/sampling identity mismatch')
    return {**config, 'endpoint': ENDPOINT, 'served_model': 'secpo-native-base'}


def configure():
    # Only launch constants change in this process; no benchmark or adapter
    # functions, attack construction, checkers, or files are patched.
    native.OUTPUT = OUTPUT
    native.VARIANTS = ('base',)
    native.ENDPOINT = ENDPOINT
    native.FILES = FILES
    native.inference_config = inference_config


def compare_protocol(current, previous):
    permitted = {'variant', 'inference', 'implementation_sha256'}
    if {k:v for k,v in current.items() if k not in permitted} != {
            k:v for k,v in previous.items() if k not in permitted}:
        raise ValueError('Static evaluation protocol differs from previous SecPO run')
    identity_keys = {'variant', 'endpoint', 'served_model', 'checkpoint',
                     'adapter_sha256', 'merge', 'cookbook_commit'}
    if {k:v for k,v in current['inference'].items() if k not in identity_keys} != {
            k:v for k,v in previous['inference'].items() if k not in identity_keys}:
        raise ValueError('Inference settings differ from previous SecPO run')
    # Every shared implementation file must remain byte-identical as well.
    for name, value in previous['implementation_sha256'].items():
        if current['implementation_sha256'].get(name) != value:
            raise ValueError('Shared implementation changed: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=('agentdojo', 'agentdyn'), required=True)
    parser.add_argument('--phase', choices=('preflight', 'smoke', 'full'), required=True)
    args = parser.parse_args()
    configure()
    spec = importlib.util.spec_from_file_location('unchanged_native_worker', ROOT / 'scripts/evaluate_native_static.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    saved_argv = sys.argv
    def invoke(phase):
        sys.argv = [str(ROOT/'scripts/evaluate_native_static.py'), '--benchmark', args.benchmark,
                    '--variant', 'base', '--phase', phase]
        try:
            worker.main()
        finally:
            sys.argv = saved_argv
    invoke('preflight')
    directory = OUTPUT/'base'/args.benchmark
    current = native.read(directory/'manifest.json')
    comparisons = {}
    for variant in ('secpo', 'secpo_lr4e5'):
        path = PREVIOUS/variant/args.benchmark/'manifest.json'
        previous = native.read(path)
        compare_protocol(current, previous)
        comparisons[variant] = dict(path=str(path), sha256=native.sha(path))
    native.write_json(directory/'protocol_comparison.json', dict(status='complete',
        at=native.now(), comparisons=comparisons,
        policy='Only model identity and endpoint differ; shared sources and evaluation settings identical.'))
    if args.phase != 'preflight':
        invoke(args.phase)


if __name__ == '__main__':
    main()
