#!/usr/bin/env python3
"""Bounded direct-OpenAI compatibility probes; no secrets in output artifacts."""
import json
from pathlib import Path
import shlex
import sys
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from secpo.tinker.reference_eval import reference_inputs, make_annotator, validate_judge_choice
from secpo.tinker.evaluation_common import write_json, now


def credentials():
    values = {}
    for line in (ROOT / 'secpo_openai.env').read_text().splitlines():
        parts = shlex.split(line, comments=True)
        if parts and parts[0] == 'export':
            parts = parts[1:]
        if not parts:
            continue
        if len(parts) != 1 or '=' not in parts[0]:
            raise ValueError('Invalid literal credential file')
        name, value = parts[0].split('=', 1)
        if name in values:
            raise ValueError('Duplicate credential field')
        values[name] = value
    if values.get('OPENAI_BASE_URL', 'https://api.openai.com/v1').rstrip('/') != 'https://api.openai.com/v1':
        raise ValueError('Refusing alternate API endpoint')
    key = values.get('OPENAI_API_KEY', '').strip()
    if not key or any(c.isspace() for c in key):
        raise ValueError('Missing API key')
    return key


def main():
    key = credentials()
    reference = ROOT / 'data/eval_references/alpaca_eval2_gpt-4o-2024-05-13_official_reference_outputs.json'
    config, references, _ = reference_inputs('alpacaeval2', reference)
    candidates = json.loads((ROOT / 'outputs/qwen38_lambda_recommended_20260916/base/alpacaeval2/candidate_outputs.json').read_text())
    prompts = []
    class Captured(Exception):
        pass
    def capture(**kwargs):
        prompts.extend(kwargs['prompts'])
        raise Captured()
    try:
        make_annotator(capture, config).annotate_head2head(outputs_1=references[:2], outputs_2=candidates[:2], is_ordered=True)
    except Captured:
        pass
    from alpaca_eval.utils import prompt_to_chatml
    tests = [('real_pair_cap1', prompt_to_chatml(prompts[0].strip()), 1),
             ('real_pair_cap16', prompt_to_chatml(prompts[1].strip()), 16),
             ('equal_answers', [{'role': 'user', 'content': 'Choose which greeting is better: m: Hello! M: Hi! These are equally appropriate. Output exactly m or M.'}], 16)]
    records = []
    for name, messages, cap in tests:
        payload = dict(model='gpt-5.6-luna', messages=messages, reasoning_effort='none',
                       max_completion_tokens=cap, temperature=1, logprobs=True, top_logprobs=5)
        with requests.Session() as session:
            session.trust_env = False
            response = session.post('https://api.openai.com/v1/chat/completions',
                headers={'Authorization': 'Bearer ' + key}, json=payload, timeout=(15, 90), allow_redirects=False)
        record = dict(test=name, http_status=response.status_code, request=payload)
        if response.status_code == 200:
            body = response.json()
            record.update(returned_model=body.get('model'), choices=body.get('choices'), usage=body.get('usage'))
            try:
                record['preference'] = validate_judge_choice(body['choices'][0])
            except ValueError:
                record['parser_failed'] = True
        records.append(record)
        write_json(ROOT / 'outputs/judge_luna_openai_preflight_20260918/real_format.json', dict(at=now(), probes=records))
        print(json.dumps({k: v for k, v in record.items() if k != 'request'}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(status='failed', error_type=type(error).__name__)), flush=True)
        raise SystemExit(1) from None
