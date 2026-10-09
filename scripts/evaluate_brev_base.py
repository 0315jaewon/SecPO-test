#!/usr/bin/env python3
"""Base-only wrappers for the unchanged full-split and original SEP protocols."""
import argparse
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from secpo.tinker.evaluation_common import MODEL, TOKENIZER_REVISION, digest, now, write_json
from secpo.inference.lambda_eval import load_config, LambdaSampler, QWEN_SAMPLING
from secpo.inference.sep_original_judges import protocol, verify_originals
import evaluate_lambda_full_lm as original_lm
import evaluate_lambda_sep_static as generation
import evaluate_lambda_sep_original as original_sep

SETTINGS_PATH = ROOT/'configs/brev_qwen38_base_full_20261003.json'
SETTINGS = json.loads(SETTINGS_PATH.read_text())
RUN = Path('/data2/jaewonc/secpo_evals')/SETTINGS['run_id']
PRIOR_LM = Path('/data2/jaewonc/secpo_evals/qwen38_full_sep_20260927/secpo')
PRIOR_SEP = Path('/data2/jaewonc/secpo_evals/qwen38_sep_original_20260927/secpo/sep_static')
SOURCE_FILES = (
    'scripts/evaluate_brev_base.py','scripts/launch_brev_base_full.py','scripts/brev_base_remote.py',
    'configs/brev_qwen38_base_full_20261003.json','scripts/evaluate_lambda_full_lm.py',
    'scripts/evaluate_lambda_sep_static.py','scripts/evaluate_lambda_sep_original.py',
    'src/secpo/tinker/evaluation_common.py','src/secpo/tinker/lm_eval_backend.py',
    'src/secpo/tinker/agent_eval.py','src/secpo/tinker/data.py','src/secpo/inference/lambda_eval.py',
    'src/secpo/inference/sep_original_judges.py','src/secpo/inference/sep_openrouter.py',
    'src/secpo/subprocess_env.py','configs/evaluate_sep_judges.json',
    'src/secpo/evaluation/sep_judge.py','src/secpo/evaluation/sep_static.py','src/secpo/evaluation/witness.py')


def read(path):
    return json.loads(Path(path).read_text())


def identity():
    verify_originals()
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def snapshot():
    hashes = identity()
    generation.persist_manifest(RUN/'source_identity.json',hashes)
    for name, checksum in hashes.items():
        target = RUN/'code_snapshot'/name
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != checksum:
            raise ValueError('Frozen source differs: '+name)
    return hashes


def inference(benchmark):
    config = {**SETTINGS['inference'],
              'endpoint':f"http://127.0.0.1:{SETTINGS['jobs'][benchmark]['local_port']}/v1"}
    path = RUN/'configs'/(benchmark+'.json')
    generation.persist_manifest(path,config)
    return load_config(path,'base')


def assert_same_fields(current, previous, fields):
    normalize = lambda value:json.loads(json.dumps(value))
    changed = [key for key in fields if normalize(current[key]) != normalize(previous[key])]
    if changed:
        raise ValueError('Evaluation protocol differs from completed lr8 run: '+','.join(changed))


def bind_sep():
    # Output routing only; imported generator, witness/scoring, and judge loops
    # remain unchanged. No old generations or judge outcomes are reused.
    original_sep.RUN = RUN
    original_sep.OUTPUT = RUN/'base/sep_static'
    original_sep.FILES = SOURCE_FILES
    generation.OUTPUT = original_sep.OUTPUT


def sep_inputs(tokenizer, config):
    if hashlib.sha256(generation.DATA.read_bytes()).hexdigest() != generation.DATA_SHA:
        raise ValueError('Approved SEP dataset changed')
    records = read(generation.DATA)
    if len(records) != 9160:
        raise ValueError('Expected all 9160 SEP examples')
    rows = generation.build_static_sep_requests(records,model_family='qwen36_27b',
        attack_names=generation.PAPER_STATIC_ATTACKS,seed=20260630,instruction_hierarchy=True)
    if len(rows) != 54960:
        raise ValueError('Expected all six attacks for every SEP example')
    previous = read(PRIOR_SEP/'manifest.json')
    template = generation.trust_role_template(tokenizer)
    manifest = dict(schema='secpo-brev-base-original-three-judges-v1',model=MODEL,
        checkpoint=config['checkpoint'],inference=config,dataset_path=str(generation.DATA),
        dataset_sha256=generation.DATA_SHA,examples=9160,generations=len(rows),
        attacks=generation.PAPER_STATIC_ATTACKS,attack_seed=20260630,generation_seed=42,
        enable_thinking=False,preserve_thinking=False,max_output_tokens=16384,
        template=template,tokenizer_revision=TOKENIZER_REVISION,request_rows_sha256=digest(rows),
        attack_builder_family='qwen36_27b',
        family_note=previous['family_note'],judge_protocol=protocol(),
        judge_policy='Original sep_judge.evaluate_rows with unchanged evaluate_sep_judges.json',
        comparison_manifest_sha256=hashlib.sha256((PRIOR_SEP/'manifest.json').read_bytes()).hexdigest(),
        protocol_changes=['base weights instead of lr8 adapter',
                          f"Brev H100; single-GPU server on physical GPU {SETTINGS['jobs']['sep_static']['gpu_id']}"],
        implementation_sha256=identity())
    fields = ('model','dataset_sha256','examples','generations','attacks','attack_seed','generation_seed',
              'enable_thinking','preserve_thinking','max_output_tokens','template','tokenizer_revision',
              'request_rows_sha256','attack_builder_family','judge_protocol','judge_policy')
    assert_same_fields(manifest,previous,fields)
    assert_same_fields(config,previous['inference'],('backend','base_revision','max_model_len','weight_dtype','sampling'))
    return rows,manifest,template


def sep(phase, concurrency):
    bind_sep()
    if phase == 'probe':
        with (RUN/'probe.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            original_sep.probe()
        return
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL,revision=TOKENIZER_REVISION,trust_remote_code=False)
    output = RUN/'base/sep_static'
    output.mkdir(parents=True,exist_ok=True)
    with (output/'worker.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config = inference('sep_static')
        rows,manifest,template = sep_inputs(tokenizer,config)
        generation.persist_manifest(output/'manifest.json',manifest)
        if phase == 'preflight':
            # Identical full request-row hash, tokenizer, template and output cap
            # make the earlier exhaustive context-length check applicable.
            prior = read(PRIOR_SEP/'preflight.json')
            if prior['status'] != 'complete' or prior['max_input_tokens']+16384>65536:
                raise ValueError('Prior identical-prompt context check incomplete')
            write_json(output/'preflight.json',{**prior,'at':now(),
                'note':'All request identities matched; context-length metadata reused, not generations.',
                'model':MODEL,'checkpoint':config['checkpoint']})
        elif phase == 'judge':
            original_sep.judge(rows,manifest,concurrency)
        else:
            if read(RUN/'judge_probe/results.json')['status'] != 'complete':
                raise ValueError('Three-judge probe must pass first')
            generation.generate(rows,manifest,config,tokenizer,template,phase=='smoke',concurrency)


def lm(benchmark, phase, concurrency):
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager, get_task_dict
    from lm_eval.evaluator_utils import get_task_list
    from secpo.tinker.lm_eval_backend import TinkerEvaluationLM
    if version('lm_eval') != '0.4.11':
        raise ValueError('Wrong harness version')
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    output = RUN/'base'/benchmark
    output.mkdir(parents=True,exist_ok=True)
    with (output/'worker.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config = inference(benchmark)
        prior_path = PRIOR_LM/benchmark/'manifest.json'
        prior = read(prior_path)
        task_dict = get_task_dict([benchmark],TaskManager())
        tasks = get_task_list(task_dict)
        if {t.task_name for t in tasks} != set(prior['tasks']):
            raise ValueError('Task categories changed')
        fingerprints = {t.task_name:original_lm.validate_task(t.task,prior['tasks'][t.task_name],t.task_name)
                        for t in tasks}
        if sum(r['num_docs'] for r in fingerprints.values()) != original_lm.COUNTS[benchmark]:
            raise ValueError('Full test count changed')
        for item in tasks:
            item.task.set_fewshot_seed(42)
        manifest = dict(benchmark=benchmark,scope='full_test_split',model=MODEL,checkpoint=config['checkpoint'],
            tokenizer_revision=TOKENIZER_REVISION,inference=config,enable_thinking=False,preserve_thinking=False,
            lm_eval_version=version('lm_eval'),math_verify_version=version('math-verify'),
            default_max_gen_toks=16384,random_seed=42,numpy_seed=42,torch_seed=42,fewshot_seed=42,
            generation_seed=42,tasks=fingerprints,comparison_manifest_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest(),
            protocol_changes=['base weights instead of lr8 adapter','Brev H100; dedicated single-GPU server'],
            implementation_sha256=identity())
        assert_same_fields(manifest,prior,('benchmark','scope','model','tokenizer_revision','enable_thinking',
            'preserve_thinking','lm_eval_version','math_verify_version','default_max_gen_toks','random_seed',
            'numpy_seed','torch_seed','fewshot_seed','generation_seed','tasks'))
        assert_same_fields(config,prior['inference'],('backend','base_revision','max_model_len','weight_dtype','sampling'))
        generation.persist_manifest(output/'manifest.json',manifest)
        if phase == 'preflight':
            write_json(output/'preflight.json',dict(status='complete',at=now(),total=original_lm.COUNTS[benchmark]))
            return
        tokenizer = AutoTokenizer.from_pretrained(MODEL,revision=TOKENIZER_REVISION,trust_remote_code=False)
        samples = {name:values['indices'] for name,values in fingerprints.items()}
        if phase == 'smoke':
            samples = {t.task_name:[0] for t in tasks}
            task_dict = {t.task_name:t.task for t in tasks}
        active = output/'smoke' if phase == 'smoke' else output
        active.mkdir(parents=True,exist_ok=True)
        sampler = LambdaSampler(output/'completion_cache',tokenizer=tokenizer,config=config)
        adapter = TinkerEvaluationLM(sampler,active,concurrency=concurrency,default_max_tokens=16384)
        total = sum(map(len,samples.values()))
        write_json(active/'progress.json',dict(status='starting',total=total,at=now()))
        result = evaluator.evaluate(lm=adapter,task_dict=task_dict,samples=samples,
            log_samples=True,apply_chat_template=True,bootstrap_iters=1000)
        for item in get_task_list(task_dict):
            observed = result['samples'][item.task_name]
            expected = [item.task.eval_docs[i] for i in samples[item.task_name]]
            if not observed:
                raise ValueError('No scored samples')
            for filter_name in {r['filter'] for r in observed}:
                rows = [r for r in observed if r['filter']==filter_name]
                if [r['doc_id'] for r in rows] != samples[item.task_name] or digest([r['doc'] for r in rows]) != digest(expected):
                    raise ValueError('Scored document identities changed')
            write_json(active/'samples'/(item.task_name+'.json'),observed)
        write_json(active/'results.json',result)
        write_json(active/'progress.json',dict(status='complete',completed_at=now(),total=total,
            completed=total,length_limited=adapter.length_limited,**sampler.stats()))
        print(json.dumps(dict(status='complete',results=result['results']),default=str),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark',choices=list(SETTINGS['jobs']),required=True)
    parser.add_argument('--phase',choices=['probe','preflight','smoke','full','judge'],required=True)
    parser.add_argument('--concurrency',type=int)
    args = parser.parse_args()
    if args.benchmark != 'sep_static' and args.phase in ('probe','judge'):
        parser.error('Only SEP uses a judge')
    os.environ.update(HF_HOME='/data2/jaewonc/huggingface',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',CUDA_VISIBLE_DEVICES='')
    snapshot()
    concurrency = args.concurrency or SETTINGS['jobs'][args.benchmark]['concurrency']
    if concurrency != SETTINGS['jobs'][args.benchmark]['concurrency']:
        raise ValueError('Concurrency differs from approved prior run')
    if args.benchmark == 'sep_static': sep(args.phase,concurrency)
    else: lm(args.benchmark,args.phase,concurrency)
