import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def saved(tmp_path):
    from secpo.tinker import evaluation_common as common
    manifest = {'inference': {'variant': 'base'}, 'reference': {'rows': 2},
                'prompt_sha256': common.digest(['prompt', 'prompt']), 'default_max_gen_toks': 16384,
                'generation': {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'seed': 42}}
    params = dict(max_tokens=16384, stop=['<|im_end|>', '<|endoftext|>'], **manifest['generation'])
    key = common.digest(dict(inference=manifest['inference'], render_version=common.RENDER_VERSION,
                             tokenizer_revision=common.TOKENIZER_REVISION, prompt='prompt', params=params))
    generation = dict(text='answer', input_tokens=1, output_tokens=1, stop_reason='stop', elapsed_seconds=1.5)
    write(tmp_path / 'candidate_outputs.json', [{'output': 'answer'}, {'output': 'answer'}])
    write(tmp_path / 'generations.json', [dict(index=i, **generation) for i in range(2)])
    path = tmp_path / 'completion_cache' / (key + '.json')
    write(path, dict(request_hash=key, params=params, generation=generation, prompt_token_ids=[1], output_token_ids=[2]))
    return tmp_path, manifest, common, path


def test_cache_only_sampler_reuses_all_rows_and_never_generates(saved):
    worker = load('judge_lambda_cached_reference')
    directory, manifest, common, _ = saved
    records = worker.validate_saved(directory, manifest, ['prompt', 'prompt'], common)
    assert len(records) == 1
    sampler = worker.CacheOnlySampler(manifest, records, common)
    result = sampler.generate('prompt', max_tokens=16384, **manifest['generation'])
    assert result.text == 'answer'
    assert sampler.stats() == dict(completed_api_requests=0, cached_requests=1, input_tokens=0, output_tokens=0)
    with pytest.raises(KeyError):
        sampler.generate('uncached', max_tokens=16384, **manifest['generation'])
    with pytest.raises(KeyError):
        sampler.generate('prompt', max_tokens=8192, **manifest['generation'])


def test_mismatched_answer_fails_before_judging(saved):
    worker = load('judge_lambda_cached_reference')
    directory, manifest, common, _ = saved
    write(directory / 'candidate_outputs.json', [{'output': 'changed'}, {'output': 'answer'}])
    with pytest.raises(ValueError, match='Saved answer differs'):
        worker.validate_saved(directory, manifest, ['prompt', 'prompt'], common)


def test_missing_completion_has_no_network_fallback(saved):
    worker = load('judge_lambda_cached_reference')
    directory, manifest, common, cache_path = saved
    cache_path.unlink()
    with pytest.raises(FileNotFoundError):
        worker.validate_saved(directory, manifest, ['prompt', 'prompt'], common)


def test_incorrect_generation_length_is_rejected(saved):
    worker = load('judge_lambda_cached_reference')
    directory, manifest, common, _ = saved
    write(directory / 'generations.json', [])
    with pytest.raises(ValueError):
        worker.validate_saved(directory, manifest, ['prompt', 'prompt'], common)


def test_sep_waits_for_generation_supervisor_acknowledgement(tmp_path):
    launcher = load('resume_lambda_judging')
    directory = tmp_path / 'secpo_lr6e5/sep_utility'
    write(directory / 'manifest.json', {'reference': {'rows': 9160}})
    write(directory / 'progress.json', {'completed': 9160})
    write(directory / 'candidate_outputs.json', [])
    write(directory / 'generations.json', [])
    write(directory / 'full_worker.json', {'status': 'running'})
    write(tmp_path / 'secpo_lr6e5/supervisor.json', {'status': 'full'})
    assert not launcher.ready(tmp_path, 'secpo_lr6e5', 'sep_utility')
    write(directory / 'full_worker.json', {'status': 'awaiting_judgment'})
    assert not launcher.ready(tmp_path, 'secpo_lr6e5', 'sep_utility')
    write(tmp_path / 'secpo_lr6e5/supervisor.json', {'status': 'awaiting_judgment'})
    assert launcher.ready(tmp_path, 'secpo_lr6e5', 'sep_utility')


def test_frozen_protocol_sources_still_match():
    worker = load('judge_lambda_cached_reference')
    worker.verify_sources('base')
    worker.verify_sources('secpo_lr6e5')
