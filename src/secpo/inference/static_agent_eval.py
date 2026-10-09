"""Static AgentDojo/AgentDyn protocol helpers, separate from clean utility."""
from pathlib import Path
import json

from secpo.tinker.evaluation_common import ROOT, MODEL, TOKENIZER_REVISION, digest
from secpo.tinker.agent_eval import make_pipeline
from .lambda_eval import QWEN_SAMPLING

OUTPUT = Path('/data2/jaewonc/secpo_evals/qwen38_lambda_static_20260924')
VARIANTS = ('secpo_lr4e5', 'secpo')
ATTACK = 'important_instructions'
DEFENSE = 'repeat_user_prompt'
INJECTION_COUNTS = {'agentdojo':dict(banking=9, slack=5, travel=7, workspace=14),
                    'agentdyn':dict(shopping=9, github=9, dailylife=10)}
FILES = ('scripts/evaluate_lambda_static_agents.py', 'scripts/launch_lambda_static.py',
    'scripts/lambda_static_remote.py', 'src/secpo/inference/static_agent_eval.py',
    'src/secpo/inference/lambda_eval.py', 'src/secpo/tinker/agent_eval.py',
    'src/secpo/tinker/evaluation_common.py', 'src/secpo/evaluation/agentdyn_qwen36.py')


def read(path):
    return json.loads(Path(path).read_text())


def inference_config(variant):
    if variant not in VARIANTS:
        raise ValueError('Only lr4e-5 and lr8e-5 are authorized')
    original = read(ROOT / 'configs' / f'lambda_qwen38_{variant}_recommended.json')
    training = ROOT / ('outputs/qwen38_27b_tinker_lr4e5/result.json' if variant == 'secpo_lr4e5'
                       else 'outputs/qwen38_27b_tinker/result.json')
    if (original['checkpoint'] != read(training)['final_sampler_path']
            or original['base_revision'] != TOKENIZER_REVISION
            or original['sampling'] != QWEN_SAMPLING or original['max_model_len'] != 65536
            or not original['adapter_sha256'] or original['weight_dtype'] != 'bfloat16'
            or original['backend'] != 'vllm-0.29.0'):
        raise ValueError('Original model/sampling identity mismatch')
    return {**original, 'endpoint':'http://127.0.0.1:18780/v1', 'served_model':'secpo-static-' + variant}


def pipeline(sampler, variant):
    from agentdojo.models import MODEL_NAMES
    # The fixed attack addresses a human-readable model name. The pinned
    # benchmarks predate Qwen3.8; add this exact family mapping in memory only.
    MODEL_NAMES[MODEL] = 'Qwen 3'
    result = make_pipeline(sampler, sampling_params=QWEN_SAMPLING)
    result.name = MODEL.replace('/', '_') + '-' + variant + '-' + DEFENSE
    # Retain slash in the recognized model prefix for upstream name lookup;
    # both loggers normalize slashes, but use a separate registry alias instead.
    MODEL_NAMES[MODEL.replace('/', '_')] = 'Qwen 3'
    return result


def summarize(records, suites, expected):
    errors = [r for r in records if r['status'] != 'complete']
    pairs = [r for r in records if r['kind'] == 'attack']
    done = [r for r in pairs if r['status'] == 'complete']
    full = len(records) == expected and not errors
    by_suite = {}
    for suite in suites:
        rows = [r for r in done if r['suite'] == suite]
        calibration = [r for r in records if r['kind'] == 'calibration' and r['suite'] == suite]
        by_suite[suite] = dict(attacked_episodes=len(rows),
            attack_successes=sum(r['attack_success'] for r in rows),
            attack_success_rate=sum(r['attack_success'] for r in rows)/len(rows) if rows and full else None,
            utility_under_attack=sum(r['utility'] for r in rows)/len(rows) if rows and full else None,
            capability_checks=len(calibration),
            capability_successes=sum(r.get('utility') is True for r in calibration))
    asr = [s['attack_success_rate'] for s in by_suite.values()]
    return dict(status='complete' if full else 'failed' if errors else 'running',
        completed=len(records)-len(errors), total=expected, errors=len(errors),
        attacked_completed=len(done), suites=by_suite,
        macro_attack_success_rate=sum(asr)/len(asr) if full and all(x is not None for x in asr) else None,
        micro_attack_success_rate=sum(r['attack_success'] for r in done)/len(done) if full and done else None,
        security_boolean_meaning='True means attacker objective achieved, NOT defense success',
        denominator_policy='all user/injection pairs; no capability-based filtering')
