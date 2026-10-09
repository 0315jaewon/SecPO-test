import pytest
from secpo.inference.static_agent_eval import inference_config, summarize, INJECTION_COUNTS
from secpo.tinker.agent_eval import BENCHMARKS
from secpo.inference.lambda_eval import QWEN_SAMPLING


def test_static_pair_inventory():
    assert sum(n*INJECTION_COUNTS['agentdojo'][s] for s,n in BENCHMARKS['agentdojo']['suites'].items())==949
    assert sum(n*INJECTION_COUNTS['agentdyn'][s] for s,n in BENCHMARKS['agentdyn']['suites'].items())==560
    assert sum(INJECTION_COUNTS['agentdojo'].values())==35
    assert sum(INJECTION_COUNTS['agentdyn'].values())==28


@pytest.mark.parametrize('variant',['secpo','secpo_lr4e5'])
def test_saved_model_protocol(variant):
    config=inference_config(variant)
    assert config['sampling']==QWEN_SAMPLING
    assert config['endpoint']=='http://127.0.0.1:18780/v1'
    assert config['served_model']=='secpo-static-'+variant
    assert 'secpo-final-sampler' in config['checkpoint']


def test_unauthorized_variant_rejected():
    with pytest.raises(ValueError):inference_config('base')


def row(suite,success,kind='attack',status='complete'):
    return dict(suite=suite,attack_success=success,utility=False,kind=kind,status=status)


def test_asr_polarity_macro_micro_and_calibration_denominator():
    rows=[row('a',True),row('a',False),row('a',False),row('b',True),row('b',None,'calibration')]
    result=summarize(rows,['a','b'],5)
    assert result['status']=='complete'
    assert result['macro_attack_success_rate']==pytest.approx(2/3)
    assert result['micro_attack_success_rate']==.5
    assert result['attacked_completed']==4


def test_error_or_partial_cannot_report_final_score():
    result=summarize([row('a',True),row('a',None,status='error')],['a'],2)
    assert result['status']=='failed' and result['macro_attack_success_rate'] is None
    result=summarize([row('a',True)],['a'],2)
    assert result['status']=='running' and result['micro_attack_success_rate'] is None
