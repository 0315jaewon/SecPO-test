import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

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
def continuation():
    return load('continue_lambda_lr6e5_evals')


def predecessors(module, root):
    write(root / 'supervisor.json', {'status': 'failed'})
    for variant in ('base', 'secpo'):
        for benchmark in module.queue.BENCHMARKS:
            directory = root / variant / benchmark
            write(directory / 'full_worker.json', {'status': 'complete', 'exit_code': 0})
            write(directory / 'progress.json', {'status': 'complete'})
    directory = root / 'base/sep_utility'
    write(directory / 'full_worker.json', {'status': 'failed', 'exit_code': 1})
    write(directory / 'progress.json', {'status': 'judging', 'completed': 9160, 'total': 9160})
    for name in ('candidate_outputs.json', 'generations.json'):
        write(directory / name, [{}] * 9160)


def test_only_base_sep_judging_can_be_bypassed(continuation, tmp_path):
    predecessors(continuation, tmp_path)
    before = (tmp_path / 'base/sep_utility/full_worker.json').read_bytes()
    continuation.validate_handoff(tmp_path)
    assert (tmp_path / 'base/sep_utility/full_worker.json').read_bytes() == before
    write(tmp_path / 'secpo/gpqa_diamond/progress.json', {'status': 'running'})
    with pytest.raises(ValueError, match='Another predecessor'):
        continuation.validate_handoff(tmp_path)


def test_incomplete_base_generation_cannot_be_bypassed(continuation, tmp_path):
    predecessors(continuation, tmp_path)
    write(tmp_path / 'base/sep_utility/generations.json', [{}])
    with pytest.raises(ValueError, match='answers are incomplete'):
        continuation.validate_handoff(tmp_path)


def test_commands_change_only_reference_entrypoint(continuation):
    for benchmark in continuation.queue.BENCHMARKS:
        for phase in ('smoke', 'full'):
            expected = continuation.queue.command(benchmark, phase)
            actual = continuation.command(benchmark, phase)
            if benchmark in continuation.REFERENCES:
                assert actual[2] == str(continuation.GENERATOR)
                actual[2] = expected[2]
            assert actual == expected


def test_generation_success_is_not_benchmark_completion(continuation, tmp_path):
    write(tmp_path / 'smoke_worker.json', {'status': 'awaiting_judgment', 'exit_code': 0})
    write(tmp_path / 'smoke/progress.json', {'status': 'awaiting_judgment'})
    assert continuation.phase_succeeded(tmp_path, 'sep_utility', 'smoke')
    assert not continuation.phase_succeeded(tmp_path, 'gpqa_diamond', 'smoke')
    write(tmp_path / 'smoke_worker.json', {'status': 'awaiting_judgment', 'exit_code': 1})
    assert not continuation.phase_succeeded(tmp_path, 'sep_utility', 'smoke')


@pytest.mark.parametrize('smoke', [0, 2])
def test_deferral_preserves_answers_and_never_writes_scores(tmp_path, smoke):
    wrapper = load('generate_lambda_lr6e5_reference')
    active = tmp_path / 'smoke' if smoke else tmp_path
    write(tmp_path / 'manifest.json', {'reference': {'rows': 2}})
    write(active / 'progress.json', {'status': 'generating', 'completed': 2, 'total': 2})
    write(active / 'candidate_outputs.json', [{'output': 'A'}, {'output': 'B'}])
    write(active / 'generations.json', [{'index': 0, 'text': 'A'}, {'index': 1, 'text': 'B'}])
    before = (active / 'candidate_outputs.json').read_bytes()
    report = wrapper.save_deferral(tmp_path, smoke=smoke, write_json=write, now=lambda: 'test')
    assert report['status'] == 'awaiting_judgment'
    assert report['judge_api_requests'] == 0
    assert (active / 'candidate_outputs.json').read_bytes() == before
    assert not (active / 'results.json').exists()
    assert not (active / 'annotations.json').exists()
    assert json.loads((active / 'progress.json').read_text())['status'] == 'awaiting_judgment'


def test_deferral_rejects_incomplete_answers(tmp_path):
    wrapper = load('generate_lambda_lr6e5_reference')
    write(tmp_path / 'manifest.json', {'reference': {'rows': 2}})
    write(tmp_path / 'progress.json', {'status': 'generating', 'completed': 1, 'total': 2})
    write(tmp_path / 'candidate_outputs.json', [{'output': 'A'}])
    write(tmp_path / 'generations.json', [{'index': 0, 'text': 'A'}])
    with pytest.raises(ValueError, match='before all answers'):
        wrapper.save_deferral(tmp_path, smoke=0, write_json=write, now=lambda: 'test')
    assert not (tmp_path / 'judging_deferred.json').exists()


def test_frozen_sources_still_match(continuation):
    continuation.queue.verify_sources()


def test_generation_wrapper_stops_before_original_judge(monkeypatch, tmp_path):
    wrapper = load('generate_lambda_lr6e5_reference')
    code = tmp_path / 'frozen_code'
    output = tmp_path / 'outputs/qwen38_lambda_recommended_20260916/secpo_lr6e5/alpacaeval2'
    write(code / 'provenance.json', {'files': {}})
    write(output / 'manifest.json', {'reference': {'rows': 2}})
    write(output / 'progress.json', {'status': 'generating', 'completed': 2, 'total': 2})
    write(output / 'candidate_outputs.json', [{'output': 'A'}, {'output': 'B'}])
    write(output / 'generations.json', [{'index': 0, 'text': 'A'}, {'index': 1, 'text': 'B'}])
    reached = []

    def paid_judge(*args, **kwargs):
        pytest.fail('Original paid judge must not be instantiated')

    runner = SimpleNamespace(write_json=write, now=lambda: 'test', OpenRouterJudge=paid_judge)

    def frozen_main():
        reached.append('saved_answers')
        runner.OpenRouterJudge(output / 'judge_cache', tmp_path / 'credentials')
        pytest.fail('Scoring must not be reached')

    runner.main = frozen_main
    fake_spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
    monkeypatch.setattr(wrapper, 'ROOT', tmp_path)
    monkeypatch.setattr(wrapper, 'CODE', code)
    monkeypatch.setattr(wrapper.sys, 'argv', ['wrapper', '--model-variant', 'secpo_lr6e5',
                                            '--benchmark', 'alpacaeval2', '--output-dir', str(output)])
    monkeypatch.setattr(wrapper.sys, 'path', wrapper.sys.path.copy())
    monkeypatch.setattr(wrapper.importlib.util, 'spec_from_file_location', lambda *args: fake_spec)
    monkeypatch.setattr(wrapper.importlib.util, 'module_from_spec', lambda spec: runner)
    wrapper.main()
    assert reached == ['saved_answers']
    assert json.loads((output / 'judging_deferred.json').read_text())['judge_api_requests'] == 0
    assert not (output / 'results.json').exists()
