import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def queue():
    return load_module(ROOT / 'scripts/queue_lambda_lr6e5_evals.py', 'lr6e5_queue_test')


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def test_no_gpu_handoff_before_every_predecessor_completes(queue, tmp_path):
    assert not queue.predecessor_ready(tmp_path)
    write(tmp_path / 'supervisor.json', {'status': 'complete'})
    assert not queue.predecessor_ready(tmp_path)
    for variant in ('secpo', 'base'):
        for benchmark in queue.BENCHMARKS:
            directory = tmp_path / variant / benchmark
            write(directory / 'full_worker.json', {'status': 'complete', 'exit_code': 0})
            write(directory / 'progress.json', {'status': 'complete'})
    assert queue.predecessor_ready(tmp_path)
    write(tmp_path / 'base/sep_utility/full_worker.json', {'status': 'failed', 'exit_code': 1})
    assert not queue.predecessor_ready(tmp_path)
    write(tmp_path / 'base/sep_utility/full_worker.json', {'status': 'complete', 'exit_code': 0})
    write(tmp_path / 'base/sep_utility/progress.json', {'status': 'judging'})
    assert not queue.predecessor_ready(tmp_path)


def test_commands_keep_caps_indices_references_and_concurrency(queue):
    for benchmark in queue.BENCHMARKS:
        for phase in ('preflight', 'smoke', 'full'):
            old = queue.original.command('secpo', benchmark, phase)
            new = queue.command(benchmark, phase)
            # Normalize ONLY the explicitly reviewed routing/credential-path changes.
            equivalent = list(new[:len(old)])
            equivalent[2] = old[2]
            for flag in ('--model-variant', '--output-dir', '--inference-config'):
                index = old.index(flag) + 1
                equivalent[index] = old[index]
            assert equivalent == old
            assert new[new.index('--model-variant') + 1] == 'secpo_lr6e5'
            assert '/lr6e5_code/scripts/' in new[2]
            assert '/secpo_lr6e5/' in new[new.index('--output-dir') + 1]
            assert '--preflight-only' in new if phase == 'preflight' else '--preflight-only' not in new


def test_engine_settings_identical_except_weights():
    old = json.loads((ROOT / 'configs/lambda_qwen38_secpo_recommended.json').read_text())
    new = json.loads((ROOT / 'configs/lambda_qwen38_secpo_lr6e5_recommended.json').read_text())
    differences = {key for key in old if old[key] != new[key]}
    assert differences == {'variant', 'served_model', 'checkpoint', 'adapter_sha256'}


def test_isolated_backend_checks_all_seven_protocols(queue):
    module = load_module(queue.CODE / 'src/secpo/inference/lambda_eval.py', 'isolated_lambda_test')
    new_id = module.evaluation_model_id('secpo_lr6e5')
    for benchmark in queue.BENCHMARKS:
        cap = None if benchmark in ('agentdojo', 'agentdyn') else 16384
        path = module.evaluation_output_root('secpo', cap) / benchmark / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['checkpoint'] = new_id
        module.configure_protocol(manifest, queue.CONFIG, 'secpo_lr6e5')
        assert manifest['checkpoint'] == new_id
        assert manifest['inference']['variant'] == 'secpo_lr6e5'
        bad = json.loads(path.read_text())
        bad['checkpoint'] = new_id
        bad['enable_thinking'] = True
        with pytest.raises(ValueError, match='protocol changes'):
            module.configure_protocol(bad, queue.CONFIG, 'secpo_lr6e5')


def test_isolated_snapshot_changes_only_routing(queue):
    provenance = json.loads((queue.CODE / 'provenance.json').read_text())
    changed = {name for name, item in provenance['files'].items()
               if item['sha256'] != item['source_sha256']}
    assert changed == {'src/secpo/inference/lambda_eval.py', 'scripts/evaluate_tinker_agents.py'}
    assert provenance['credential_files_copied'] is False
    queue.verify_sources()
