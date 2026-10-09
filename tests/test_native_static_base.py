"""No-network checks for base protocol matching and GPU handoff gating."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import evaluate_native_static_base as worker
import launch_native_static_base as queue


def sample():
    old = dict(attack='important_instructions', defense='repeat_user_prompt', max_output_tokens=8192,
        tasks=[{'task':'one'}], variant='secpo', payloads_sha256='identical',
        inference=dict(sampling=dict(temperature=.7), checkpoint='tinker://example', adapter_sha256='adapter'),
        implementation_sha256={'unchanged.py':'original'})
    current = copy.deepcopy(old)
    current['variant'] = 'base'
    current['inference'] = dict(sampling=dict(temperature=.7), checkpoint='base:Qwen/Qwen3.8-27B')
    current['implementation_sha256']['base_wrapper.py'] = 'new'
    return current, old


def test_allows_only_model_identity_and_extra_wrapper():
    current, old = sample()
    worker.compare_protocol(current, old)


@pytest.mark.parametrize('key,value', [('max_output_tokens',16384),('attack','other'),('payloads_sha256','different')])
def test_rejects_changed_protocol(key, value):
    current, old = sample()
    current[key] = value
    with pytest.raises(ValueError):
        worker.compare_protocol(current, old)


def test_rejects_sampling_and_original_code_change():
    current, old = sample()
    current['inference']['sampling']['temperature'] = 0
    with pytest.raises(ValueError):
        worker.compare_protocol(current, old)
    current, old = sample()
    current['implementation_sha256']['unchanged.py'] = 'modified'
    with pytest.raises(ValueError):
        worker.compare_protocol(current, old)


def test_base_model_identity():
    config = worker.inference_config('base')
    assert config['served_model']=='secpo-native-base'
    assert config['sampling']['top_k']==20
    with pytest.raises(ValueError):
        worker.inference_config('secpo')


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_dependency_does_not_allow_early_or_failed_handoff(tmp_path):
    write(tmp_path/'supervisor.json', dict(status='running'))
    assert queue.dependency_ready(tmp_path) is False
    write(tmp_path/'supervisor.json', dict(status='failed'))
    with pytest.raises(RuntimeError):
        queue.dependency_ready(tmp_path)


def test_dependency_checks_all_worker_results(tmp_path):
    write(tmp_path/'supervisor.json', dict(status='generation_and_utility_complete'))
    for name in ('mmlu_pro','minerva_math','sep_static'):
        directory = tmp_path/'secpo'/name
        write(directory/'full_worker.json', dict(status='complete', exit_code=0))
        write(directory/'progress.json', dict(status='generation_complete' if name=='sep_static' else 'complete'))
    assert queue.dependency_ready(tmp_path)
    write(tmp_path/'secpo/sep_static/progress.json', dict(status='running'))
    with pytest.raises(ValueError):
        queue.dependency_ready(tmp_path)
