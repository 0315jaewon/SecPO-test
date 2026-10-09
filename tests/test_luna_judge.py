import importlib.util
import json
from pathlib import Path
import pytest

from secpo.tinker.luna_judge import LunaJudge, PARAMS, read_credentials, validate_choice

ROOT = Path(__file__).resolve().parents[1]


def choice():
    return dict(message={'content': 'm'}, finish_reason='stop', logprobs={'content': [
        dict(token='m', logprob=-0.4, top_logprobs=[{'token': 'm', 'logprob': -0.4}, {'token': 'M', 'logprob': -1.1}])]})


def test_fractional_original_parser():
    assert 1 < validate_choice(choice()) < 2
    invalid = choice()
    invalid['message']['content'] = 'Explanation: m'
    with pytest.raises(ValueError):
        validate_choice(invalid)


def test_endpoint_and_literal_credentials(tmp_path):
    path = tmp_path / 'key.env'
    path.write_text('OPENAI_API_KEY="test-key"\n')
    assert read_credentials(path) == 'test-key'
    path.write_text('OPENAI_API_KEY="test-key"\nOPENAI_BASE_URL="https://example.org"\n')
    with pytest.raises(ValueError, match='official endpoint'):
        read_credentials(path)


def test_deduplication_cache_and_request_params(tmp_path):
    path = tmp_path / 'key.env'
    path.write_text('OPENAI_API_KEY="test-key"\n')
    payloads = []
    def post(payload):
        payloads.append(payload)
        return dict(model='gpt-5.6-luna', choices=[choice()], usage={'prompt_tokens': 10, 'completion_tokens': 4})
    judge = LunaJudge(tmp_path / 'cache', path, post=post)
    prompt = '<|im_start|>user\nWhich answer?<|im_end|>'
    result = judge([prompt, prompt], **PARAMS)
    assert len(result['completions_all']) == 2
    assert len(payloads) == 1
    assert payloads[0]['max_completion_tokens'] == 16
    assert payloads[0]['reasoning_effort'] == 'none'
    assert 'provider' not in payloads[0] and 'max_tokens' not in payloads[0]
    judge([prompt], **PARAMS)
    assert len(payloads) == 1 and judge.stats['cached_requests'] == 1
    assert judge.stats['reported_cost_usd'] is None
    assert all('test-key' not in p.read_text() for p in (tmp_path / 'cache').glob('*.json'))


def test_existing_experiment_sources_unchanged():
    import hashlib
    run = ROOT / 'outputs/qwen38_lambda_recommended_20260916'
    for name, sha in json.loads((run / 'source_identity.json').read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha


def test_lr4_commands_keep_all_original_protocol_flags():
    spec = importlib.util.spec_from_file_location('new_queue', ROOT / 'scripts/launch_lambda_lr4e5_evals.py')
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
    for benchmark in queue.BENCHMARKS:
        for phase in ('preflight', 'smoke', 'full'):
            old = queue.original.command('secpo', benchmark, phase)
            new = queue.command(benchmark, phase)
            normalized = new[:len(old)]
            normalized[2] = old[2]
            for flag in ('--model-variant', '--output-dir', '--inference-config'):
                normalized[old.index(flag) + 1] = old[old.index(flag) + 1]
            assert normalized == old
            assert new[new.index('--model-variant') + 1] == 'secpo_lr4e5'
    assert queue.expected_status('sep_utility', 'full') == 'awaiting_judgment'
    assert queue.expected_status('agentdojo', 'full') == 'complete'
