#!/usr/bin/env python3
"""Run the frozen reference evaluator up to, but never into, paid judging.

The original evaluator still renders, samples, validates, and saves every answer.
Only construction of its judge is intercepted. No scores or completed benchmark
results are fabricated. The original evaluator can later reuse these caches.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'runtime/lambda_suite_20260916/lr6e5_code'


class JudgmentDeferred(Exception):
    pass


def save_deferral(directory, *, smoke, write_json, now):
    active = directory / 'smoke' if smoke else directory
    manifest = json.loads((directory / 'manifest.json').read_text())
    progress = json.loads((active / 'progress.json').read_text())
    candidates = json.loads((active / 'candidate_outputs.json').read_text())
    generations = json.loads((active / 'generations.json').read_text())
    expected = smoke or manifest['reference']['rows']
    if not (progress['completed'] == progress['total'] == len(candidates)
            == len(generations) == expected):
        raise ValueError('Cannot defer judging before all answers have been saved')
    if any(row['index'] != i or row['text'] != candidates[i]['output']
           for i, row in enumerate(generations)):
        raise ValueError('Saved generation identities differ')
    if (active / 'results.json').exists():
        raise ValueError('Refusing to replace a completed evaluation')
    record = dict(status='awaiting_judgment', deferred_at=now(), n=expected,
                  smoke=bool(smoke), reason='OpenRouter credits exhausted; user requested queue continuation',
                  judge_api_requests=0, original_evaluator_unchanged=True,
                  candidate_sha256=hashlib.sha256((active / 'candidate_outputs.json').read_bytes()).hexdigest())
    write_json(active / 'judging_deferred.json', record)
    write_json(active / 'progress.json', {**progress, 'status': 'awaiting_judgment', 'updated_at': now()})
    return record


def main():
    argv = sys.argv[1:]
    def value(flag, default=None):
        return argv[argv.index(flag) + 1] if flag in argv else default
    if value('--model-variant') != 'secpo_lr6e5' or '--preflight-only' in argv:
        raise ValueError('This wrapper is only for lr6e5 reference answer generation')
    directory = Path(value('--output-dir')).resolve()
    benchmark = value('--benchmark')
    expected = ROOT / 'outputs/qwen38_lambda_recommended_20260916/secpo_lr6e5' / benchmark
    if benchmark not in {'alpacaeval2', 'sep_utility'} or directory != expected:
        raise ValueError('Unexpected benchmark output directory')
    provenance = json.loads((CODE / 'provenance.json').read_text())
    for name, item in provenance['files'].items():
        if hashlib.sha256((CODE / name).read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Frozen evaluation source changed')
    # Import from the frozen snapshot before importing any secpo modules.
    sys.path.insert(0, str(CODE / 'src'))
    spec = importlib.util.spec_from_file_location('frozen_reference_runner', CODE / 'scripts/evaluate_tinker_reference.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    smoke = int(value('--smoke', '0'))

    def defer_judge(*args, **kwargs):
        record = save_deferral(directory, smoke=smoke, write_json=runner.write_json, now=runner.now)
        print(json.dumps(record), flush=True)
        raise JudgmentDeferred()

    runner.OpenRouterJudge = defer_judge
    try:
        runner.main()
    except JudgmentDeferred:
        pass


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(status='failed', error_type=type(error).__name__)), flush=True)
        raise SystemExit(1) from None
