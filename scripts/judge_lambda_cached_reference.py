#!/usr/bin/env python3
"""Resume the frozen utility evaluator using verified local completions only.

No Lambda/Tinker inference client is constructed. The original judge, prompt
ordering, cache, validation, and scoring code run without modifications.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/qwen38_lambda_recommended_20260916'
CODE = ROOT / 'runtime/lambda_suite_20260916/lr6e5_code'
JOBS = {('base', 'sep_utility'), ('secpo_lr6e5', 'alpacaeval2'), ('secpo_lr6e5', 'sep_utility')}


def read(path):
    return json.loads(path.read_text())


def verify_sources(variant):
    for name, expected in read(RUN / 'source_identity.json').items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Original experiment source changed')
    if variant == 'secpo_lr6e5':
        for name, item in read(CODE / 'provenance.json')['files'].items():
            if hashlib.sha256((CODE / name).read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Frozen lr6e5 source changed')


def validate_saved(directory, manifest, prompts, common):
    """Fail before paid judging if any saved answer/cache/identity is missing."""
    candidates = read(directory / 'candidate_outputs.json')
    generations = read(directory / 'generations.json')
    if len(candidates) != manifest['reference']['rows'] or common.digest(prompts) != manifest['prompt_sha256']:
        raise ValueError('Candidate count or prompt identity changed')
    params = dict(max_tokens=manifest['default_max_gen_toks'], stop=['<|im_end|>', '<|endoftext|>'],
                  **manifest['generation'])
    records = {}
    for i, (prompt, candidate, generation) in enumerate(zip(prompts, candidates, generations, strict=True)):
        key = common.digest(dict(inference=manifest['inference'], render_version=common.RENDER_VERSION,
                                 tokenizer_revision=common.TOKENIZER_REVISION, prompt=prompt, params=params))
        if key not in records:
            cached = read(directory / 'completion_cache' / (key + '.json'))
            if (cached['request_hash'] != key or cached['params'] != params
                    or cached['generation']['input_tokens'] != len(cached['prompt_token_ids'])
                    or cached['generation']['output_tokens'] != len(cached['output_token_ids'])):
                raise ValueError('Completion cache identity or token counts differ')
            records[key] = cached['generation']
        if (candidate['output'] != records[key]['text'] or generation['index'] != i
                or {k: v for k, v in generation.items() if k != 'index'} != records[key]):
            raise ValueError('Saved answer differs from its original cached generation')
    return records


class CacheOnlySampler:
    def __init__(self, manifest, records, common):
        self.manifest, self.records, self.common = manifest, records, common
        self.lock, self.cached = threading.Lock(), 0

    def generate(self, prompt, *, max_tokens, stops=(), seed=42, **sampling):
        params = dict(max_tokens=max_tokens, stop=list(dict.fromkeys([*stops, '<|im_end|>', '<|endoftext|>'])),
                      seed=seed, **sampling)
        key = self.common.digest(dict(inference=self.manifest['inference'], render_version=self.common.RENDER_VERSION,
                    tokenizer_revision=self.common.TOKENIZER_REVISION, prompt=prompt, params=params))
        # A miss is a hard error. There is intentionally no API/network fallback.
        result = self.common.Generation(**self.records[key])
        with self.lock:
            self.cached += 1
        return result

    def stats(self):
        with self.lock:
            return dict(completed_api_requests=0, cached_requests=self.cached, input_tokens=0, output_tokens=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True, choices=['base', 'secpo_lr6e5'])
    parser.add_argument('--benchmark', required=True, choices=['sep_utility', 'alpacaeval2'])
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if (args.variant, args.benchmark) not in JOBS:
        parser.error('Not one of the pending judgments')
    directory = RUN / args.variant / args.benchmark
    lock = (directory / 'judge_resume.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_sources(args.variant)
    source = CODE if args.variant == 'secpo_lr6e5' else ROOT
    sys.path.insert(0, str(source / 'src'))
    from secpo.tinker import evaluation_common as common, reference_eval as reference
    from secpo.inference import lambda_eval as backend
    spec = importlib.util.spec_from_file_location('frozen_reference_runner', source / 'scripts/evaluate_tinker_reference.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    previous_worker = read(directory / 'full_worker.json')
    if previous_worker['status'] not in {'failed', 'awaiting_judgment', 'complete'}:
        raise ValueError('Generation worker has not stopped')
    argv = previous_worker['command'][3:]
    def value(flag):
        return argv[argv.index(flag) + 1]
    if ('--smoke' in argv or '--preflight-only' in argv or value('--model-variant') != args.variant
            or Path(value('--output-dir')).resolve() != directory or value('--benchmark') != args.benchmark):
        raise ValueError('Saved command does not match this complete benchmark')
    manifest = read(directory / 'manifest.json')
    if read(directory / 'progress.json')['completed'] != manifest['reference']['rows']:
        raise ValueError('Generation is incomplete')
    # Full original preflight also checks the manifest, judge, references and scoring assets.
    common.load_credentials(ROOT / 'secpo_eval_keys.env')
    os.environ['HF_HUB_OFFLINE'] = '1'
    sys.argv = [str(source / 'scripts/evaluate_tinker_reference.py'), *argv, '--preflight-only']
    runner.main()
    suite = 'sep' if args.benchmark == 'sep_utility' else 'alpacaeval2'
    reference_path = Path(value('--reference'))
    config, references, _ = reference.reference_inputs(suite, reference_path,
        approved_sha256=manifest['reference']['sha256'] if suite == 'sep' else None,
        sep_test_path=Path(value('--sep-test-dataset')) if suite == 'sep' else None)
    runner.preflight_reference_utility(config, candidate_path=directory / 'candidate_outputs.json', reference_path=reference_path)
    tokenizer = common.load_tokenizer()
    prompts = [reference.render_clean_prompt(tokenizer, suite, row) for row in references]
    records = validate_saved(directory, manifest, prompts, common)
    hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
              for name in ('manifest.json', 'candidate_outputs.json', 'generations.json')}
    common.write_json(directory / 'judge_resume_audit.json', dict(at=common.now(), status='passed',
        rows=len(references), unique_completions=len(records), saved_sha256=hashes,
        generation_api_requests_allowed=0, wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(dict(status='audit_passed', rows=len(references), unique_completions=len(records))), flush=True)
    if args.audit_only:
        return
    previous_path = directory / 'judge_resume_before.json'
    if not previous_path.exists():
        common.write_json(previous_path, dict(at=common.now(), full_worker=previous_worker,
            progress=read(directory / 'progress.json'), saved_sha256=hashes))
    sampler = CacheOnlySampler(manifest, records, common)
    def cache_factory(cache_dir, *, tokenizer, variant, config_path=None):
        if (Path(cache_dir).resolve() != directory / 'completion_cache' or variant != args.variant
                or backend.load_config(config_path, variant) != manifest['inference']):
            raise ValueError('Unexpected sampler identity')
        return sampler
    backend.make_sampler = cache_factory
    sys.argv = [str(source / 'scripts/evaluate_tinker_reference.py'), *argv]
    runner.main()
    for name, sha in hashes.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != sha:
            raise ValueError('Saved generation artifacts changed during judging')
    print(json.dumps(dict(status='judging_complete', generation_api_requests=0)), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        detail = str(error)
        if not re.fullmatch(r'OpenRouter (?:HTTP \d{3}|failed after three attempts \((?:\d{3}|network_error)\)); body withheld', detail):
            detail = type(error).__name__
        print(json.dumps(dict(status='failed', error_type=type(error).__name__, detail=detail)), flush=True)
        raise SystemExit(1) from None
