import importlib.util
import json
from pathlib import Path

import pytest

from secpo.tinker.gpt6_luna_judge import GPT6LunaJudge, MODEL, PARAMS
from secpo.tinker.luna_judge import LunaJudge, PARAMS as OLD_PARAMS, validate_choice

ROOT = Path(__file__).resolve().parents[1]


def verdict():
    return dict(message={'content':'m'}, finish_reason='stop', logprobs={'content':[
        dict(token='m', logprob=-0.4, top_logprobs=[
            dict(token='m', logprob=-0.4), dict(token='M', logprob=-1.1)])]})


def setup_judge(tmp_path, model=MODEL):
    credentials = tmp_path / 'credential.env'
    credentials.write_text('OPENAI_API_KEY="test-only-key"\n')
    payloads = []
    def post(payload):
        payloads.append(payload)
        return dict(model=model, choices=[verdict()], usage=dict(prompt_tokens=20, completion_tokens=4))
    return GPT6LunaJudge(tmp_path / 'cache', credentials, post=post), payloads, credentials


def test_weighted_scoring_and_isolated_cache(tmp_path):
    judge, payloads, credentials = setup_judge(tmp_path)
    prompt = '<|im_start|>user\nCompare answers.<|im_end|>'
    result = judge([prompt, prompt], **PARAMS)
    assert len(payloads) == 1
    assert 1 < validate_choice(result['completions_all'][0]) < 2
    assert payloads[0]['model'] == MODEL
    assert payloads[0]['reasoning_effort'] == 'none'
    assert payloads[0]['temperature'] == 1
    assert payloads[0]['logprobs'] is True and payloads[0]['top_logprobs'] == 5
    judge([prompt], **PARAMS)
    assert len(payloads) == 1 and judge.stats['cached_requests'] == 1
    old_calls = []
    def old_post(payload):
        old_calls.append(payload)
        return dict(model='gpt-5.6-luna', choices=[verdict()], usage={})
    LunaJudge(tmp_path / 'cache', credentials, post=old_post)([prompt], **OLD_PARAMS)
    assert len(old_calls) == 1
    assert len(list((tmp_path / 'cache').glob('*.json'))) == 2
    assert all('test-only-key' not in p.read_text() for p in (tmp_path / 'cache').glob('*.json'))


def test_model_and_parameter_guards(tmp_path):
    judge, _, _ = setup_judge(tmp_path, model='gpt-5.6-luna')
    with pytest.raises(ValueError, match='Unexpected judge model'):
        judge(['<|im_start|>user\nJudge<|im_end|>'], **PARAMS)
    with pytest.raises(ValueError, match='decoder settings'):
        judge([], **{**PARAMS, 'reasoning_effort':'high'})
    assert not list((tmp_path / 'cache').glob('*.json'))


def test_corrupt_cache_rejected(tmp_path):
    judge, payloads, _ = setup_judge(tmp_path)
    prompt = '<|im_start|>user\nJudge<|im_end|>'
    judge([prompt], **PARAMS)
    path = next((tmp_path / 'cache').glob('*.json'))
    record = json.loads(path.read_text())
    record['returned_model'] = 'gpt-5.6-luna'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='cache identity'):
        judge([prompt], **PARAMS)
    assert len(payloads) == 1


def runner():
    spec = importlib.util.spec_from_file_location('gpt6_queue', ROOT / 'scripts/judge_gpt6_luna.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_rows_required_and_final_answer_alignment():
    module = runner()
    candidates = [dict(instruction=str(i), output='final') for i in range(805)]
    generations = [dict(index=i, text='final', raw_text='reasoning and final') for i in range(805)]
    references = [dict(instruction=str(i), output='reference') for i in range(805)]
    module.validate_rows(candidates, generations, references, ['instruction'])
    with pytest.raises(ValueError, match='805'):
        module.validate_rows(candidates[:-1], generations, references, ['instruction'])
    candidates[5]['output'] = 'reasoning and final'
    with pytest.raises(ValueError, match='mismatch'):
        module.validate_rows(candidates, generations, references, ['instruction'])


def test_ready_requires_completed_files_not_other_judges(tmp_path):
    module = runner()
    (tmp_path / 'progress.json').write_text(json.dumps(dict(status='judging', completed=805)))
    assert not module.generation_ready(tmp_path)
    for name in ('manifest.json', 'candidate_outputs.json', 'generations.json'):
        (tmp_path / name).write_text('{}')
    assert module.generation_ready(tmp_path)
    (tmp_path / 'progress.json').write_text(json.dumps(dict(status='generating', completed=804)))
    assert not module.generation_ready(tmp_path)
