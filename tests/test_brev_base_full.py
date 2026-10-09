import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import evaluate_brev_base as evaluation
import brev_base_remote as remote
import launch_brev_base_full as queue


def test_gpu_allocation_is_physically_restricted():
    remote.validate_allocation()
    for index in (0,1,2,3,4):
        invalid = deepcopy(remote.SETTINGS)
        invalid['jobs']['sep_static']['gpu_id'] = index
        with pytest.raises(ValueError): remote.validate_allocation(invalid)
    invalid = deepcopy(remote.SETTINGS)
    invalid['allowed_gpu_ids'].append(0)
    with pytest.raises(ValueError): remote.validate_allocation(invalid)


def test_gpu_uuid_mapping_and_busy_guards(monkeypatch):
    job = remote.SETTINGS['jobs']['sep_static']
    def output(command,**kwargs):
        return (f"7, {job['gpu_uuid']}, 0\n" if 'index,uuid,memory.used' in command[1] else '')
    monkeypatch.setattr(remote.subprocess,'check_output',output)
    assert remote.gpu_for('sep_static') == job
    monkeypatch.setattr(remote.subprocess,'check_output',lambda *a,**kw:'7, GPU-wrong, 0\n')
    with pytest.raises(ValueError): remote.gpu_for('sep_static')
    monkeypatch.setattr(remote.subprocess,'check_output',lambda *a,**kw:f"7, {job['gpu_uuid']}, 4000\n")
    with pytest.raises(remote.GpuBusy): remote.gpu_for('sep_static')


def test_server_configuration_matches_prior_single_gpu_recipe():
    for benchmark in remote.SETTINGS['jobs']:
        command = remote.server_command(benchmark)
        for flag,value in {'--max-model-len':'65536','--max-num-seqs':'32',
                           '--max-num-batched-tokens':'4096','--dtype':'bfloat16',
                           '--seed':'42','--served-model-name':'secpo-suite-base'}.items():
            assert command[command.index(flag)+1] == value
        assert '--enforce-eager' in command and '--no-enable-prefix-caching' in command
        assert '--tensor-parallel-size' not in command
        assert remote.env(remote.SETTINGS['jobs'][benchmark]['gpu_uuid'])['CUDA_VISIBLE_DEVICES'].startswith('GPU-')
    assert remote.env()['CUDA_VISIBLE_DEVICES'] == ''
    assert remote.env()['CUDA_HOME'] == str(remote.CUDA)
    assert str(remote.CUDA/'bin') in remote.env()['PATH'].split(':')
    assert remote.env()['FLASHINFER_WORKSPACE_BASE'].startswith(str(remote.ROOT))
    assert remote.PACKAGES[-3:] == ['nvidia-cuda-nvcc==13.0.88','nvidia-cuda-crt==13.0.88','nvidia-nvvm==13.0.88']


def test_queue_commands_and_credential_exclusion(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY','test-not-a-real-secret')
    monkeypatch.setenv('HF_TOKEN','test-not-a-real-secret')
    for benchmark,count in [('sep_static',8),('mmlu_pro',12),('minerva_math',12)]:
        command=queue.worker_command(benchmark,'full')
        assert command[-1] == str(count)
        assert command[command.index('--benchmark')+1] == benchmark
    assert 'OPENROUTER_API_KEY' not in queue.environment()
    assert 'HF_TOKEN' not in remote.env()
    assert queue.environment()['CUDA_VISIBLE_DEVICES'] == ''


def test_protocol_comparison_fails_closed():
    current={'a':[1,2],'b':{'sample':42}}
    evaluation.assert_same_fields(current,deepcopy(current),('a','b'))
    with pytest.raises(ValueError):
        evaluation.assert_same_fields(current,{'a':[1,3],'b':{'sample':42}},('a','b'))


def test_lm_evaluation_invocation_matches_original():
    def call(path):
        tree=ast.parse(path.read_text())
        matches=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                 and isinstance(n.func.value,ast.Name) and n.func.value.id=='evaluator' and n.func.attr=='evaluate']
        assert len(matches)==1
        # Model variable names differ; every evaluation option must be identical.
        return {kw.arg:ast.dump(kw.value) for kw in matches[0].keywords if kw.arg!='lm'}
    assert call(ROOT/'scripts/evaluate_brev_base.py') == call(ROOT/'scripts/evaluate_lambda_full_lm.py')


def test_original_sep_implementation_not_modified():
    assert len(evaluation.verify_originals()) == 4
    settings=evaluation.protocol()
    assert settings['config']['trials_per_judge']==3
    assert settings['aggregation']=='original evaluate_rows: witness gate then all nine YES'
