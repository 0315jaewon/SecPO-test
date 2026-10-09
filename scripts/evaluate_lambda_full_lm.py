#!/usr/bin/env python3
"""Full test splits; reuse the prior lm-eval tasks, renderer, and scorer unchanged."""
import argparse
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from secpo.tinker.evaluation_common import MODEL, TOKENIZER_REVISION, digest, now, write_json
from secpo.inference.lambda_eval import load_config, LambdaSampler, QWEN_SAMPLING

OUTPUT = Path('/data2/jaewonc/secpo_evals/qwen38_full_sep_20260927/secpo')
CONFIG = ROOT / 'configs/lambda_qwen38_lr8e5_full_20260927.json'
COUNTS = {'mmlu_pro': 12032, 'minerva_math': 5000}


def validate_task(task, previous, name):
    docs = task.eval_docs
    generation = {**(task.get_config('generation_kwargs') or {}),
                  'max_gen_toks': 16384, **QWEN_SAMPLING, 'do_sample': True}
    if (len(docs) != previous['num_docs'] or
            getattr(docs, '_fingerprint', None) != previous['dataset_fingerprint'] or
            task.get_config('num_fewshot') != previous['num_fewshot'] or
            generation != previous['generation_kwargs'] or
            digest([docs[i] for i in previous['indices']]) != previous['selected_documents_sha256']):
        raise ValueError('Task differs from the earlier evaluated dataset/protocol: ' + name)
    task.set_config('generation_kwargs', generation)
    return dict(num_docs=len(docs), dataset_fingerprint=getattr(docs, '_fingerprint', None),
        indices=list(range(len(docs))), selected_documents_sha256=digest(list(docs)),
        generation_kwargs=generation, num_fewshot=task.get_config('num_fewshot'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=COUNTS, required=True)
    parser.add_argument('--phase', choices=['preflight', 'smoke', 'full'], required=True)
    parser.add_argument('--concurrency', type=int, default=12)
    args = parser.parse_args()
    os.environ.update(HF_HOME='/data2/jaewonc/huggingface', HF_HUB_OFFLINE='1',
        HF_DATASETS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false')
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
    directory = OUTPUT / args.benchmark
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = load_config(CONFIG, 'secpo')
        prior_path = ROOT / 'outputs/qwen38_lambda_recommended_20260916/secpo' / args.benchmark / 'manifest.json'
        previous = json.loads(prior_path.read_text())
        if previous['checkpoint'] != config['checkpoint'] or previous['enable_thinking'] is not False:
            raise ValueError('Prior model identity differs')
        task_dict = get_task_dict([args.benchmark], TaskManager())
        tasks = get_task_list(task_dict)
        if {t.task_name for t in tasks} != set(previous['tasks']):
            raise ValueError('Task categories changed')
        fingerprints = {t.task_name: validate_task(t.task, previous['tasks'][t.task_name], t.task_name) for t in tasks}
        if sum(r['num_docs'] for r in fingerprints.values()) != COUNTS[args.benchmark]:
            raise ValueError('Unexpected full-dataset size')
        for item in tasks:
            item.task.set_fewshot_seed(42)
        manifest = dict(benchmark=args.benchmark, scope='full_test_split', model=MODEL,
            checkpoint=config['checkpoint'], tokenizer_revision=TOKENIZER_REVISION,
            inference=config, enable_thinking=False, preserve_thinking=False,
            lm_eval_version=version('lm_eval'), math_verify_version=version('math-verify'),
            default_max_gen_toks=16384, random_seed=42, numpy_seed=42, torch_seed=42,
            fewshot_seed=42, generation_seed=42, tasks=fingerprints,
            prior_subset_manifest_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest(),
            protocol_changes=['all test rows instead of selected subsets', 'private inference endpoint'],
            implementation_sha256={name:hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                ['scripts/evaluate_lambda_full_lm.py', 'src/secpo/inference/lambda_eval.py',
                 'src/secpo/tinker/lm_eval_backend.py', 'src/secpo/tinker/evaluation_common.py']})
        path = directory / 'manifest.json'
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError('Existing full-dataset manifest differs')
        write_json(path, manifest)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=TOKENIZER_REVISION, trust_remote_code=False)
        print(json.dumps(dict(benchmark=args.benchmark, selected=COUNTS[args.benchmark], manifest_sha256=digest(manifest))), flush=True)
        if args.phase == 'preflight':
            write_json(directory / 'preflight.json', dict(status='complete', at=now(), total=COUNTS[args.benchmark]))
            return
        samples = {name: values['indices'] for name, values in fingerprints.items()}
        if args.phase == 'smoke':
            samples = {t.task_name:[0] for t in tasks}
            task_dict = {t.task_name:t.task for t in tasks}
        active = directory / 'smoke' if args.phase == 'smoke' else directory
        active.mkdir(parents=True, exist_ok=True)
        sampler = LambdaSampler(directory / 'completion_cache', tokenizer=tokenizer, config=config)
        lm = TinkerEvaluationLM(sampler, active, concurrency=args.concurrency, default_max_tokens=16384)
        total = sum(map(len, samples.values()))
        write_json(active / 'progress.json', dict(status='starting', total=total, at=now()))
        result = evaluator.evaluate(lm=lm, task_dict=task_dict, samples=samples,
            log_samples=True, apply_chat_template=True, bootstrap_iters=1000)
        for item in get_task_list(task_dict):
            observed = result['samples'][item.task_name]
            expected = [item.task.eval_docs[i] for i in samples[item.task_name]]
            if not observed:
                raise ValueError('No scored samples')
            for filter_name in {r['filter'] for r in observed}:
                rows = [r for r in observed if r['filter'] == filter_name]
                if [r['doc_id'] for r in rows] != samples[item.task_name] or digest([r['doc'] for r in rows]) != digest(expected):
                    raise ValueError('Scored document identities differ')
            write_json(active / 'samples' / (item.task_name + '.json'), observed)
        write_json(active / 'results.json', result)
        write_json(active / 'progress.json', dict(status='complete', completed_at=now(), total=total,
            completed=total, length_limited=lm.length_limited, **sampler.stats()))
        print(json.dumps(dict(status='complete', results=result['results']), default=str), flush=True)


if __name__ == '__main__':
    main()
