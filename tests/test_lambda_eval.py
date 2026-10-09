import json
from types import SimpleNamespace
import pytest
from secpo.inference.lambda_eval import LambdaSampler, QWEN_SAMPLING, load_config
from secpo.tinker.evaluation_common import Generation, ROOT


def test_pinned_configs_and_wrong_variant():
    for variant in ('base', 'secpo'):
        p = ROOT / f'configs/lambda_qwen38_{variant}_recommended.json'
        assert load_config(p, variant)['sampling'] == QWEN_SAMPLING
        with pytest.raises(ValueError):
            load_config(p, 'secpo' if variant == 'base' else 'base')


def test_loopback_only(tmp_path):
    obj = json.loads((ROOT / 'configs/lambda_qwen38_base_recommended.json').read_text())
    for url in ('https://example.com/v1', 'http://127.0.0.1.evil/v1', 'http://user@127.0.0.1/v1'):
        obj['endpoint'] = url
        path = tmp_path / 'config.json'
        path.write_text(json.dumps(obj))
        with pytest.raises(ValueError):
            load_config(path, 'base')


def test_lm_passes_all_sampling_settings(tmp_path):
    from secpo.tinker.lm_eval_backend import TinkerEvaluationLM
    class Sampler:
        tokenizer = SimpleNamespace(chat_template='template')
        def generate(self, prompt, *, max_tokens, stops, **kwargs):
            assert kwargs == QWEN_SAMPLING
            assert max_tokens == 16384
            return Generation('answer', 4, 2, 'stop', .01)
        def stats(self):
            return {}
    lm = TinkerEvaluationLM(Sampler(), tmp_path)
    req = SimpleNamespace(args=('prompt', dict(QWEN_SAMPLING, do_sample=True, max_gen_toks=16384)))
    assert lm.generate_until([req]) == ['answer']


def test_sampler_rejects_silent_sampling_fallback():
    sampler = object.__new__(LambdaSampler)
    with pytest.raises(ValueError, match='explicit Qwen'):
        sampler.generate('prompt', max_tokens=16384, temperature=0)


def test_raw_tokens_stop_trimming_and_cache(tmp_path, monkeypatch):
    import requests
    posts = []
    config = load_config(ROOT / 'configs/lambda_qwen38_base_recommended.json', 'base')
    class Response:
        status_code = 200
        def __init__(self, data):
            self.data = data
        def raise_for_status(self):
            pass
        def json(self):
            return self.data
    class Session:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def get(self, *args, **kwargs):
            return Response({'data': [{'id': config['served_model']}]})
        def post(self, url, *, json, timeout):
            assert self.trust_env is False
            assert json['prompt'] == [9, 8, 7]
            assert json['return_token_ids'] is True
            assert {k: json[k] for k in QWEN_SAMPLING} == QWEN_SAMPLING
            posts.append(json)
            return Response({'model': config['served_model'],
                             'usage': {'prompt_tokens': 3, 'completion_tokens': 2},
                             'choices': [{'token_ids': [1, 2], 'finish_reason': 'stop', 'text': 'answer'}]})
    monkeypatch.setattr(requests, 'Session', Session)
    tokenizer = SimpleNamespace(encode=lambda *a, **k: [9, 8, 7],
                                decode=lambda *a, **k: 'answer<|im_end|>')
    sampler = LambdaSampler(tmp_path, tokenizer=tokenizer, config=config)
    first = sampler.generate('prompt', max_tokens=16384, **QWEN_SAMPLING)
    second = sampler.generate('prompt', max_tokens=16384, **QWEN_SAMPLING)
    assert first == second
    assert first.text == 'answer'
    assert first.output_tokens == 2
    assert len(posts) == 1
    assert sampler.stats()['cached_requests'] == 1
    saved = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert saved['output_token_ids'] == [1, 2]
    assert saved['prompt_token_ids'] == [9, 8, 7]


def test_context_and_output_budgets_checked_before_request(tmp_path, monkeypatch):
    sampler = object.__new__(LambdaSampler)
    sampler.config = {'max_model_len': 65536}
    sampler.tokenizer = SimpleNamespace(encode=lambda *a, **k: [1] * 50000)
    with pytest.raises(ValueError, match='context budget'):
        sampler.generate('prompt', max_tokens=16384, **QWEN_SAMPLING)
    with pytest.raises(ValueError, match='Unapproved'):
        sampler.generate('prompt', max_tokens=2048, **QWEN_SAMPLING)
