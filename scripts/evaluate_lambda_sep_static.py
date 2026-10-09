#!/usr/bin/env python3
"""Resumable transport wrapper around the unchanged six-attack SEP implementation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from secpo.evaluation.sep_static import PAPER_STATIC_ATTACKS, build_static_sep_requests, score_static_sep_rows, summarize_static_sep
from secpo.tinker.agent_eval import trust_role_template
from secpo.tinker.data import QWEN38_NON_THINKING_SUFFIX
from secpo.tinker.evaluation_common import MODEL, TOKENIZER_REVISION, RENDER_VERSION, digest, now, write_json
from secpo.inference.lambda_eval import LambdaSampler, load_config, QWEN_SAMPLING
from secpo.inference.sep_openrouter import LunaSecurityJudge, protocol

OUTPUT = Path('/data2/jaewonc/secpo_evals/qwen38_full_sep_20260927/secpo/sep_static')
DATA = Path('/home/jaewonc/SecAlign_cont/data/SEP_dataset_test.json')
DATA_SHA = 'e659683ae5b10905064d28bca5fd786d1a4ec96ce016a6b93a7282fd977a95a3'
CONFIG = ROOT / 'configs/lambda_qwen38_lr8e5_full_20260927.json'


def requests_and_manifest(tokenizer):
    if hashlib.sha256(DATA.read_bytes()).hexdigest() != DATA_SHA:
        raise ValueError('SEP dataset differs from the approved test set')
    records = json.loads(DATA.read_text())
    if len(records) != 9160:
        raise ValueError('Unexpected SEP dataset size')
    rows = build_static_sep_requests(records, model_family='qwen36_27b',
        attack_names=PAPER_STATIC_ATTACKS, seed=20260630, instruction_hierarchy=True)
    if len(rows) != 54960 or any(row['attack_category'] != 'static' for row in rows):
        raise ValueError('Unexpected attack selection')
    template = trust_role_template(tokenizer)
    config = load_config(CONFIG, 'secpo')
    manifest = dict(schema='secpo-lambda-six-static-v1', model=MODEL, checkpoint=config['checkpoint'],
        inference=config, dataset_path=str(DATA), dataset_sha256=DATA_SHA, examples=9160, generations=54960,
        attacks=PAPER_STATIC_ATTACKS, attack_seed=20260630, generation_seed=42,
        enable_thinking=False, preserve_thinking=False, max_output_tokens=16384,
        template=template, tokenizer_revision=TOKENIZER_REVISION, request_rows_sha256=digest(rows),
        attack_builder_family='qwen36_27b',
        family_note='Original builder selector only; all six selected static transforms are family-independent. Victim is Qwen3.8. No basic-adaptive delimiter attacks.',
        judge_policy='separate judge_manifest.json; witness hits remain pending until judged',
        implementation_sha256={name:hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
            ['scripts/evaluate_lambda_sep_static.py','src/secpo/evaluation/sep_static.py',
             'src/secpo/evaluation/witness.py','src/secpo/tinker/agent_eval.py',
             'src/secpo/inference/lambda_eval.py','src/secpo/tinker/evaluation_common.py']})
    return rows, manifest, config, template


def render(tokenizer, template, row):
    prompt = tokenizer.apply_chat_template(row['messages'], chat_template=template,
        tokenize=False, add_generation_prompt=True, enable_thinking=False, preserve_thinking=False)
    if not prompt.endswith(QWEN38_NON_THINKING_SUFFIX):
        raise ValueError('Thinking-off suffix differs')
    return prompt


def row_path(root, row):
    return root / row['attack'] / f"{row['source_index']:05d}.json"


def persist_manifest(path, manifest):
    # JSON roundtrip canonicalizes tuples before comparison.
    manifest = json.loads(json.dumps(manifest))
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Existing manifest differs; new output directory required')
    write_json(path, manifest)


def generate(rows, manifest, config, tokenizer, template, smoke, concurrency):
    active = OUTPUT / 'smoke' if smoke else OUTPUT
    active.mkdir(parents=True, exist_ok=True)
    selected = [next(r for r in rows if r['attack']==attack) for attack in PAPER_STATIC_ATTACKS] if smoke else rows
    sampler = LambdaSampler(OUTPUT / 'completion_cache', tokenizer=tokenizer, config=config)
    identity = digest(manifest)
    def one(row):
        path = row_path(OUTPUT / 'rows', row)
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['manifest_sha256'] != identity or any(saved[k] != v for k,v in row.items()):
                raise ValueError('Saved generation row identity differs')
            return saved
        prompt = render(tokenizer, template, row)
        params = dict(max_tokens=16384, stop=['<|im_end|>','<|endoftext|>'], seed=42, **QWEN_SAMPLING)
        result = sampler.generate(prompt, max_tokens=16384, stops=params['stop'], seed=42, **QWEN_SAMPLING)
        cache_key = digest(dict(inference=config, render_version=RENDER_VERSION,
            tokenizer_revision=TOKENIZER_REVISION, prompt=prompt, params=params))
        if not (OUTPUT / 'completion_cache' / (cache_key + '.json')).is_file():
            raise ValueError('Generation cache missing')
        saved = score_static_sep_rows([{**row, 'output':result.text}])[0]
        saved.update(generation=asdict(result), completion_cache_key=cache_key,
            manifest_sha256=identity, at=now())
        write_json(path, saved)
        return saved
    completed = hits = capped = 0
    write_json(active / 'progress.json', dict(status='generating', completed=0, total=len(selected), at=now()))
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one, row) for row in selected]
        try:
            for future in as_completed(futures):
                saved = future.result()
                completed += 1
                hits += saved['witness_present']
                capped += saved['generation']['stop_reason'] == 'length'
                if completed % 25 == 0 or completed == len(selected):
                    update = dict(status='generating', completed=completed, total=len(selected),
                        witness_hits=hits, length_limited=capped, at=now(), **sampler.stats())
                    write_json(active / 'progress.json', update)
                    print(json.dumps(update), flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    values = [json.loads(row_path(OUTPUT / 'rows', row).read_text()) for row in selected]
    summary = summarize_static_sep(values)
    write_json(active / 'generation_summary.json', dict(status='complete', at=now(), summary=summary,
        witness_hits=hits, length_limited=capped, scored=False, note='Witness hits are pending semantic judgments'))
    write_json(active / 'progress.json', dict(status='generation_complete', completed=completed, total=len(selected),
        witness_hits=hits, length_limited=capped, at=now(), **sampler.stats()))


def judge(rows, manifest, trials, concurrency):
    if json.loads((OUTPUT / 'progress.json').read_text())['status'] not in {'generation_complete','complete'}:
        raise ValueError('Full generation is not complete')
    judge_manifest = dict(generation_manifest_sha256=digest(manifest), protocol=protocol(trials),
        implementation_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ['src/secpo/inference/sep_openrouter.py','src/secpo/evaluation/sep_judge.py',
             'src/secpo/evaluation/sep_static.py','src/secpo/evaluation/witness.py']})
    persist_manifest(OUTPUT / 'judge_manifest.json', judge_manifest)
    judge_identity = digest(judge_manifest)
    semantic = LunaSecurityJudge(OUTPUT / 'judge_cache', ROOT / 'secpo_openrouter.env', trials=trials)
    def one(request):
        row = json.loads(row_path(OUTPUT / 'rows', request).read_text())
        if row['manifest_sha256'] != digest(manifest):
            raise ValueError('Generation manifest mismatch')
        path = row_path(OUTPUT / 'judgments', request)
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['judge_manifest_sha256'] != judge_identity or saved['output'] != row['output']:
                raise ValueError('Judgment identity differs')
            if saved['status'] == 'complete':
                return saved
        result = semantic.judge(row)
        saved = {**row, **result, 'judge_manifest_sha256':judge_identity}
        write_json(path, saved)
        if result['status'] != 'complete':
            raise RuntimeError('Semantic judgment failed; saved as unknown, not a negative score')
        return saved
    completed = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one, row) for row in rows]
        try:
            for future in as_completed(futures):
                future.result(); completed += 1
                if completed % 25 == 0 or completed == len(rows):
                    write_json(OUTPUT / 'judge_progress.json', dict(status='judging', completed=completed,
                        total=len(rows), at=now(), **semantic.stats))
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    values = [json.loads(row_path(OUTPUT / 'judgments', row).read_text()) for row in rows]
    summary = summarize_static_sep(values)
    if any(item['pending'] for item in summary['per_attack'].values()):
        raise ValueError('Cannot report final ASR with pending judgments')
    write_json(OUTPUT / 'results.json', dict(status='complete', at=now(), **summary,
        judge_protocol=protocol(trials), judge_stats=semantic.stats))
    write_json(OUTPUT / 'judge_progress.json', dict(status='complete', completed=len(rows), total=len(rows), at=now(), **semantic.stats))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['preflight','smoke','generate','judge'], required=True)
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--judge-trials', type=int, choices=[1,3])
    args = parser.parse_args()
    if args.phase == 'judge' and args.judge_trials is None:
        parser.error('--judge-trials is required for semantic judging')
    os.environ.update(HF_HOME='/data2/jaewonc/huggingface', HF_HUB_OFFLINE='1',
        HF_DATASETS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=TOKENIZER_REVISION, trust_remote_code=False)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / 'worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows, manifest, config, template = requests_and_manifest(tokenizer)
        persist_manifest(OUTPUT / 'manifest.json', manifest)
        if args.phase == 'preflight':
            maximum = 0
            for start in range(0,len(rows),256):
                prompts = [render(tokenizer,template,r) for r in rows[start:start+256]]
                tokens = tokenizer(prompts, add_special_tokens=False)['input_ids']
                maximum = max(maximum, max(map(len,tokens)))
                if maximum + 16384 > 65536:
                    raise ValueError('SEP prompt exceeds the context budget; no truncation authorized')
            write_json(OUTPUT / 'preflight.json', dict(status='complete', at=now(), examples=9160,
                generations=len(rows), attacks=PAPER_STATIC_ATTACKS, max_input_tokens=maximum, max_output_tokens=16384))
            print(json.dumps(dict(status='preflight_complete',generations=len(rows),max_input_tokens=maximum)),flush=True)
        elif args.phase == 'judge':
            judge(rows,manifest,args.judge_trials,args.concurrency)
        else:
            generate(rows,manifest,config,tokenizer,template,args.phase=='smoke',args.concurrency)


if __name__ == '__main__':
    main()
