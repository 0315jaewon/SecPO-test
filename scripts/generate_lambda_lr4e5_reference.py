#!/usr/bin/env python3
"""Frozen lr4e5 answer generation only; Luna judging is a separate saved run."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'runtime/lambda_suite_20260916/lr4e5_code'


def main():
    argv = sys.argv[1:]
    def value(flag, default=None):
        return argv[argv.index(flag) + 1] if flag in argv else default
    benchmark = value('--benchmark')
    directory = Path(value('--output-dir')).resolve()
    if (value('--model-variant') != 'secpo_lr4e5' or '--preflight-only' in argv
            or benchmark not in {'alpacaeval2', 'sep_utility'}
            or directory != ROOT / 'outputs/qwen38_lambda_recommended_20260916/secpo_lr4e5' / benchmark):
        raise ValueError('Unexpected generation routing')
    for name, item in json.loads((CODE / 'provenance.json').read_text())['files'].items():
        if hashlib.sha256((CODE / name).read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Frozen source changed')
    sys.path.insert(0, str(CODE / 'src'))
    spec = importlib.util.spec_from_file_location('frozen_reference_runner', CODE / 'scripts/evaluate_tinker_reference.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    class JudgmentDeferred(Exception):
        pass
    def defer(*args, **kwargs):
        smoke = int(value('--smoke', '0'))
        active = directory / 'smoke' if smoke else directory
        read = lambda path: json.loads(path.read_text())
        manifest, progress = read(directory / 'manifest.json'), read(active / 'progress.json')
        candidates, generations = read(active / 'candidate_outputs.json'), read(active / 'generations.json')
        count = smoke or manifest['reference']['rows']
        if not (progress['completed'] == progress['total'] == len(candidates) == len(generations) == count):
            raise ValueError('Generation incomplete')
        if any(g['index'] != i or g['text'] != c['output'] for i, (c, g) in enumerate(zip(candidates, generations))):
            raise ValueError('Saved answer identity mismatch')
        if (active / 'results.json').exists():
            raise ValueError('Refusing to overwrite scored results')
        runner.write_json(active / 'judging_deferred.json', dict(status='awaiting_judgment', n=count,
            deferred_at=runner.now(), smoke=bool(smoke), judge_api_requests=0,
            reason='Answers saved for separate direct-OpenAI Luna judging; legacy manifest judge was not invoked',
            intended_judge='gpt-5.6-luna',
            judging_output=str(ROOT / 'outputs/qwen38_lambda_luna_openai_20260918/secpo_lr4e5' / benchmark),
            candidate_sha256=hashlib.sha256((active / 'candidate_outputs.json').read_bytes()).hexdigest()))
        runner.write_json(active / 'progress.json', {**progress, 'status': 'awaiting_judgment', 'updated_at': runner.now()})
        raise JudgmentDeferred()
    runner.OpenRouterJudge = defer
    try:
        runner.main()
    except JudgmentDeferred:
        print(json.dumps(dict(status='awaiting_judgment', judge_api_requests=0)), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(status='failed', error_type=type(error).__name__)), flush=True)
        raise SystemExit(1) from None
