#!/usr/bin/env python3
"""Audit saved SEP generations and probe a bounded number of uncached judgments.

This does not edit the pinned evaluation code or produce partial benchmark
scores. Successful probe judgments go into the original content-addressed cache
and are reused by the normal resume. Credentials and response bodies are never
logged. The original model-queue lock prevents concurrent resumes.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from secpo.tinker.evaluation_common import (
    RENDER_VERSION, TOKENIZER_REVISION, digest, load_credentials, load_tokenizer, now, write_json,
)
from secpo.tinker.reference_eval import (
    OpenRouterJudge, make_annotator, reference_inputs, render_clean_prompt,
)
from secpo.evaluation.reference_utility import preflight_reference_utility

RUN = ROOT / 'outputs/qwen38_lambda_recommended_20260916'
OUTPUT = RUN / 'base/sep_utility'


class ProbeFinished(Exception):
    pass


def safe_error(error):
    message = str(error)
    allowed = (r'OpenRouter HTTP \d{3}; body withheld',
               r'OpenRouter failed after three attempts \((?:\d{3}|network_error)\); body withheld')
    return message if any(re.fullmatch(pattern, message) for pattern in allowed) else type(error).__name__


def audit():
    for name, sha in json.loads((RUN / 'source_identity.json').read_text()).items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
            raise ValueError('Pinned source differs: ' + name)
    manifest = json.loads((OUTPUT / 'manifest.json').read_text())
    reference_path = Path('/home/jaewonc/SecAlign_cont/data/SEP_dataset_test_Meta-Llama-3-8B-Instruct.json')
    config, references, _ = reference_inputs('sep', reference_path,
        approved_sha256=manifest['reference']['sha256'],
        sep_test_path=Path('/home/jaewonc/SecAlign_cont/data/SEP_dataset_test.json'))
    candidate_path = OUTPUT / 'candidate_outputs.json'
    preflight_reference_utility(config, candidate_path=candidate_path, reference_path=reference_path)
    candidates = json.loads(candidate_path.read_text())
    generations = json.loads((OUTPUT / 'generations.json').read_text())
    load_credentials(ROOT / 'secpo_eval_keys.env')
    os.environ['HF_HUB_OFFLINE'] = '1'
    tokenizer = load_tokenizer()
    prompts = [render_clean_prompt(tokenizer, 'sep', row) for row in references]
    if digest(prompts) != manifest['prompt_sha256']:
        raise ValueError('Prompt identity changed')
    params = dict(max_tokens=manifest['default_max_gen_toks'],
                  stop=['<|im_end|>', '<|endoftext|>'], **manifest['generation'])
    seen = {}
    for index, (prompt, candidate, generation) in enumerate(zip(prompts, candidates, generations, strict=True)):
        key = digest(dict(inference=manifest['inference'], render_version=RENDER_VERSION,
                          tokenizer_revision=TOKENIZER_REVISION, prompt=prompt, params=params))
        if key not in seen:
            cached = json.loads((OUTPUT / 'completion_cache' / (key + '.json')).read_text())
            if cached['request_hash'] != key or cached['params'] != params:
                raise ValueError('Completion cache identity mismatch')
            if (cached['generation']['output_tokens'] != len(cached['output_token_ids'])
                    or cached['generation']['input_tokens'] != len(cached['prompt_token_ids'])):
                raise ValueError('Completion token accounting mismatch')
            seen[key] = cached['generation']
        expected = seen[key]
        if (candidate['output'] != expected['text'] or generation['index'] != index
                or {k: v for k, v in generation.items() if k != 'index'} != expected):
            raise ValueError('Saved answers differ from original cached generations')
    report = dict(checked_at=now(), rows=len(candidates), unique_completions=len(seen),
                  candidate_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                  manifest_sha256=digest(manifest), all_completions_cached=True,
                  judge_cache_before=len(list((OUTPUT / 'judge_cache').glob('*.json'))),
                  pinned_sources_unchanged=True)
    return config, references, candidates, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-new-judgments', type=int, default=8)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.max_new_judgments <= 16:
        parser.error('Probe must be limited to 1–16 new judgments')
    lock = (RUN / 'launch.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config, references, candidates, report = audit()
    print(json.dumps(report), flush=True)
    destination = OUTPUT / 'diagnostics' / ('judge_probe_' + now().replace(':', '-') + '.json')
    if args.audit_only:
        report['status'] = 'audit_passed'
        write_json(destination, report)
        return

    class ProbeJudge(OpenRouterJudge):
        def _post(self, payload):
            if self.stats['api_requests_completed'] >= args.max_new_judgments:
                raise ProbeFinished()
            event = dict(at=now(), model=payload['model'], request_sha256=digest(payload),
                         prompt_characters=sum(len(m.get('content', '')) for m in payload['messages']))
            try:
                body = super()._post(payload)
            except Exception as error:
                event.update(status='failed', error_type=type(error).__name__, detail=safe_error(error),
                             stack=[dict(file=Path(f.filename).name, line=f.lineno, function=f.name)
                                    for f in traceback.extract_tb(error.__traceback__)])
                report['requests'].append(event)
                print(json.dumps(event), flush=True)
                raise
            event.update(status='returned', returned_model=body.get('model'),
                         provider=body.get('provider'))
            report['requests'].append(event)
            print(json.dumps(event), flush=True)
            return body

    report['requests'] = []
    judge = ProbeJudge(OUTPUT / 'judge_cache', ROOT / 'secpo_openrouter.env', concurrency=1)
    try:
        annotator = make_annotator(judge, config)
        annotator.annotate_head2head(outputs_1=references, outputs_2=candidates, is_ordered=True)
        report['status'] = 'all_judgments_cached'
    except ProbeFinished:
        report['status'] = 'probe_passed'
    except Exception as error:
        report.update(status='probe_failed', error_type=type(error).__name__, detail=safe_error(error))
    report.update(stats=judge.stats, ended_at=now(),
                  judge_cache_after=len(list((OUTPUT / 'judge_cache').glob('*.json'))))
    write_json(destination, report)
    print(json.dumps(dict(status=report['status'], stats=judge.stats, report=str(destination))), flush=True)
    if report['status'] == 'probe_failed':
        raise SystemExit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps(dict(status='failed', error_type=type(exc).__name__, detail=safe_error(exc))), flush=True)
        raise SystemExit(1) from None
