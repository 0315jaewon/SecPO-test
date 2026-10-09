#!/usr/bin/env python3
"""Generate base thinking-on answers once, then score identical answers twice."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import fcntl
import json
import math
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from secpo.inference.thinking_eval import ThinkingSampler, render_prompt, SAMPLING, MODEL_NAME, RENDER_VERSION
from secpo.tinker.evaluation_common import MODEL, TOKENIZER_REVISION, digest, now, write_json
from secpo.tinker import reference_eval as ref
from secpo.tinker.luna_judge import LunaJudge, PARAMS as LUNA_PARAMS, read_credentials, ENDPOINT as LUNA_ENDPOINT
from secpo.evaluation.reference_utility import sha256_file, preflight_reference_utility

BACKUP = Path('/data2/jaewonc/secpo_backups/lambda_20260919')
REFERENCE = ROOT / 'data/eval_references/alpaca_eval2_gpt-4o-2024-05-13_official_reference_outputs.json'
ASSET = ROOT / 'data/eval_references/alpacaeval_df_gamed.csv'
SOURCES = ['scripts/evaluate_lambda_thinking.py', 'src/secpo/inference/thinking_eval.py',
    'scripts/launch_lambda_thinking.py', 'scripts/lambda_thinking_remote.py',
    'src/secpo/tinker/reference_eval.py', 'src/secpo/tinker/luna_judge.py',
    'src/secpo/tinker/evaluation_common.py', 'src/secpo/evaluation/reference_utility.py']
RUN_DIR = None


def read(path):
    return json.loads(path.read_text())


def score(judge_name, directory, candidates, references, config, manifest, *, smoke):
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(generation_manifest_sha256=digest(manifest), candidate_sha256=digest(candidates),
                    reference_sha256=sha256_file(REFERENCE), judge=manifest['judges'][judge_name],
                    evaluator_sources=manifest['evaluator_sources'], smoke=smoke)
    path = directory / 'manifest.json'
    if path.exists() and read(path) != identity:
        raise ValueError('Judge run identity differs')
    write_json(path, identity)
    if (directory / 'results.json').exists():
        result = read(directory / 'results.json')
        if result['manifest_sha256'] != digest(identity):
            raise ValueError('Existing judge result differs')
        return result
    progress = dict(status='judging', total=len(candidates), started_at=now())
    def update(stats):
        write_json(directory / 'progress.json', {**progress, 'updated_at': now(), 'api': stats})
    # Both smoke and full passes use the same prompt-addressed cache.
    cache = directory.parent.parent / 'judge_cache' / judge_name if smoke else directory.parent / 'judge_cache' / judge_name
    if judge_name == 'mini':
        judge = ref.OpenRouterJudge(cache, ROOT / 'secpo_openrouter.env', concurrency=4, progress=update)
        params = ref.JUDGE_PARAMS
    else:
        judge = LunaJudge(cache, ROOT / 'secpo_openai.env', concurrency=4, progress=update)
        params = LUNA_PARAMS
    update(judge.stats)
    annotator = ref.make_annotator(judge, config)
    next(iter(annotator.annotators.values())).completions_kwargs = dict(params)
    annotations = annotator.annotate_head2head(outputs_1=references, outputs_2=candidates, is_ordered=True)
    ref.validate_annotations(annotations, candidates, references, config.identity_keys)
    for row in annotations:
        row['source_annotator_template'] = ref.ANNOTATOR_NAME
        if judge_name == 'luna':
            row['annotator'] = 'weighted_alpaca_eval_gpt-5.6-luna'
    write_json(directory / 'annotations.json', annotations)
    progress['status'] = 'scoring'
    update(judge.stats)
    metrics = ref.score_annotations('sep' if smoke else 'alpacaeval2', annotations,
        metric_asset=ASSET, metric_asset_sha256=ref.ALPACA_METRIC_SHA256)
    if not math.isfinite(metrics['win_rate' if smoke else 'length_controlled_winrate']):
        raise ValueError('Nonfinite score')
    result = dict(status='complete', judge=judge_name, n=len(candidates), smoke=smoke,
        manifest_sha256=digest(identity), metrics=metrics, completed_at=now(), api_this_process=judge.stats,
        pricing_note='Luna billing dollars unavailable; zero package placeholders are not billed costs')
    write_json(directory / 'results.json', result)
    progress.update(status='complete', completed=len(candidates), completed_at=now())
    update(judge.stats)
    return result


def main():
    global RUN_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--endpoint', default='http://127.0.0.1:18680/v1')
    parser.add_argument('--max-output-tokens', type=int, default=32768)
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--resume-compatible', action='store_true',
                        help='Record a transport/recovery repair without altering the frozen experiment')
    args = parser.parse_args()
    if args.max_output_tokens != 32768 or not 1 <= args.concurrency <= 16:
        parser.error('Expected cap 32768 and concurrency 1–16')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    RUN_DIR = args.output_dir
    lock = (args.output_dir / 'worker.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config, references, ref_info = ref.reference_inputs('alpacaeval2', REFERENCE)
    ref.read_openrouter_credentials(ROOT / 'secpo_openrouter.env')
    read_credentials(ROOT / 'secpo_openai.env')
    if sha256_file(ASSET) != ref.ALPACA_METRIC_SHA256:
        raise ValueError('LC asset checksum differs')
    old = read(ROOT / 'outputs/qwen38_lambda_recommended_20260916/base/alpacaeval2/manifest.json')
    sources = ref.evaluator_sources()
    if sources != old['evaluator_sources'] or ref_info != old['reference']:
        raise ValueError('Reference or scoring differs from the non-thinking run')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BACKUP / 'models/base', local_files_only=True, trust_remote_code=False)
    prompts = [render_prompt(tokenizer, r['instruction']) for r in references]
    lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    if max(lengths) + args.max_output_tokens > 65536:
        raise ValueError('Insufficient context budget')
    expected_files = {}
    for line in (BACKUP / 'SHA256SUMS.models').read_text().splitlines():
        sha, name = line.split('  ', 1)
        if name.startswith('models/base/'):
            expected_files[name] = sha
            if not name.endswith('.safetensors') and sha256_file(BACKUP / name) != sha:
                raise ValueError('Local model/tokenizer asset differs')
    model_identity = dict(model=MODEL, revision=TOKENIZER_REVISION, files_sha256=expected_files,
                          backend='vllm-0.29.0', weight_dtype='bfloat16')
    manifest = dict(schema='secpo-thinking-alpaca-v1', model_identity=model_identity, model_variant='base',
        benchmark='alpacaeval2', reference=ref_info, rows=len(references),
        thinking=dict(enable_thinking=True, preserve_thinking=True, reasoning_effort='xhigh'),
        render_version=RENDER_VERSION, prompts_sha256=digest(prompts), max_prompt_tokens=max(lengths),
        max_output_tokens=args.max_output_tokens, max_model_len=65536, sampling={**SAMPLING, 'seed':42},
        output_policy='cap includes reasoning and answer; judge final answer only; unfinished reasoning becomes empty answer, never dropped',
        token_policy='raw IDs retained; after_reasoning_tokens includes boundary whitespace/termination tokens if returned',
        endpoint=args.endpoint, served_model=MODEL_NAME, concurrency=args.concurrency,
        judges=dict(mini=dict(model=ref.JUDGE_MODEL, endpoint=ref.JUDGE_BASE_URL, params=ref.JUDGE_PARAMS),
                    luna=dict(model=LUNA_PARAMS['model_name'], endpoint=LUNA_ENDPOINT, params=LUNA_PARAMS)),
        judge_protocol=dict(template=ref.ANNOTATOR_NAME, seed=0, randomized_order=True, scoring='probability-weighted'),
        evaluator_sources=sources, metric_asset_sha256=ref.ALPACA_METRIC_SHA256,
        implementation_sha256={name:sha256_file(ROOT/name) for name in SOURCES},
        nonthinking_comparator=str(ROOT/'outputs/qwen38_lambda_recommended_20260916/base/alpacaeval2'))
    mp = args.output_dir / 'manifest.json'
    snapshot_dir = args.output_dir / 'code_snapshot'
    history = None
    if mp.exists() and read(mp) != manifest:
        previous = read(mp)
        stable = lambda value: {k:v for k,v in value.items() if k!='implementation_sha256'}
        changed = {name for name in manifest['implementation_sha256']
                   if manifest['implementation_sha256'][name] != previous['implementation_sha256'].get(name)}
        allowed = {'scripts/evaluate_lambda_thinking.py', 'scripts/launch_lambda_thinking.py',
                   'src/secpo/inference/thinking_eval.py'}
        if not args.resume_compatible or stable(previous)!=stable(manifest) or not changed<=allowed:
            raise ValueError('Experiment manifest differs; do not mix runs')
        # The original protocol manifest and source snapshot stay immutable.
        for name, expected in previous['implementation_sha256'].items():
            if sha256_file(args.output_dir/'code_snapshot'/name) != expected:
                raise ValueError('Original source snapshot changed')
        history = args.output_dir/'resume_history'/digest(manifest['implementation_sha256'])
        if not (history/'resume_manifest.json').exists():
            history.mkdir(parents=True,exist_ok=True)
            for name in ('manifest.json','progress.json','failure.json','preflight.json'):
                if (args.output_dir/name).exists():
                    shutil.copy2(args.output_dir/name,history/name)
            cache_hashes = {p.name:sha256_file(p) for p in (args.output_dir/'completion_cache').glob('*.json')}
            write_json(history/'resume_manifest.json',dict(at=now(),original_manifest_sha256=digest(previous),
                runtime_manifest=manifest,changed_source_files=sorted(changed),
                repair='Preserve raw responses; treat early stop without final boundary as empty answer; resume existing server/cache; improve diagnostics',
                existing_cache_sha256=cache_hashes))
        snapshot_dir = history/'code_snapshot'
        manifest = previous  # Retain the identity used by the completed judge smoke caches.
    elif not mp.exists():
        write_json(mp, manifest)
    for name in SOURCES:
        destination = snapshot_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, destination)
    write_json(args.output_dir/'preflight.json', dict(status='passed', rows=len(references),
        max_prompt_tokens=max(lengths), manifest_sha256=digest(manifest), at=now(),
        resume_history=str(history) if history else None))
    print(json.dumps(dict(status='preflight_passed', rows=len(references), max_prompt_tokens=max(lengths))), flush=True)
    if args.preflight_only:
        return
    verified = read(args.output_dir.parents[1] / 'model_validation.json')
    if verified['status'] != 'complete' or {r['path']:r['sha256'] for r in verified['files']} != expected_files:
        raise ValueError('Remote model verification missing or differs')
    sampler = ThinkingSampler(args.output_dir/'completion_cache', tokenizer, args.endpoint, args.max_output_tokens, model_identity)
    generations = [None] * len(references)
    candidates = [None] * len(references)
    progress = dict(status='smoke', total=len(references), completed=0, started_at=now(),
                    length_limited=0, empty_final=0, output_tokens=0, stopped_without_final_boundary=0,
                    resume_history=str(history) if history else None)
    def accept(index, record):
        generations[index] = {k:v for k,v in record.items() if k not in ('output_token_ids','request_identity')}
        generations[index]['index'] = index
        candidates[index] = {**references[index], 'output':record['text'], 'generator':'Qwen3.8-27B-base-thinking-xhigh'}
        progress['completed'] += 1
        progress['length_limited'] += int(record['finish_reason']=='length')
        progress['empty_final'] += int(record['empty_final'])
        progress['stopped_without_final_boundary'] += int(record.get('stopped_without_final_boundary',False))
        progress['output_tokens'] += record['output_tokens']
        write_json(args.output_dir/'progress.json', {**progress, 'updated_at':now(), 'generation_api':sampler.stats()})
    # Smoke calls are retained as the first two rows of the full experiment.
    for index in range(2):
        accept(index, sampler.generate(prompts[index]))
    if any(generations[i]['empty_final'] or not generations[i]['reasoning_complete'] for i in range(2)):
        raise ValueError('Smoke produced no usable final answer; inspect before running all 805')
    for judge_name in ('mini','luna'):
        score(judge_name, args.output_dir/'smoke'/judge_name, candidates[:2], references[:2], config, manifest, smoke=True)
    write_json(args.output_dir/'smoke/results.json', dict(status='complete', rows=2, at=now()))
    progress['status']='generating'
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        by_prompt = {}
        for index in range(2,len(prompts)):
            by_prompt.setdefault(prompts[index],[]).append(index)
        jobs = {pool.submit(sampler.generate,prompt):indices for prompt,indices in by_prompt.items()}
        future = None
        try:
            for future in as_completed(jobs):
                for index in jobs[future]:
                    accept(index,future.result())
        except BaseException as error:
            # Write immediately, before waiting for already-running requests.
            detail = dict(status='failed', error_type=type(error).__name__, at=now(),
                indices=jobs.get(future,[]), message=str(error) if isinstance(error,ValueError) else 'See raw response cache')
            write_json(args.output_dir/'generation_failure.json',detail)
            write_json(args.output_dir/'progress.json',{**progress,**detail})
            pool.shutdown(wait=True, cancel_futures=True)
            raise
    write_json(args.output_dir/'candidate_outputs.json', candidates)
    write_json(args.output_dir/'generations.json', generations)
    report = preflight_reference_utility(config, candidate_path=args.output_dir/'candidate_outputs.json', reference_path=REFERENCE)
    write_json(args.output_dir/'reference_preflight.json', asdict(report))
    progress['status']='judging'
    write_json(args.output_dir/'progress.json', {**progress, 'updated_at':now()})
    results = {}
    # Sequential judge jobs avoid the package's process-global LC asset hook
    # overlapping. Within each job, bounded API requests run concurrently.
    for judge_name in ('mini','luna'):
        results[judge_name] = score(judge_name,args.output_dir/judge_name,candidates,references,config,manifest,smoke=False)
    write_json(args.output_dir/'results.json', dict(status='complete', n=len(candidates),
        manifest_sha256=digest(manifest), candidate_sha256=sha256_file(args.output_dir/'candidate_outputs.json'),
        metrics={name:r['metrics'] for name,r in results.items()}, generation=progress, completed_at=now()))
    progress.update(status='complete', completed_at=now())
    write_json(args.output_dir/'progress.json',progress)
    print(json.dumps(dict(status='complete', metrics={name:r['metrics'] for name,r in results.items()})),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        failure = dict(status='failed', error_type=type(error).__name__, at=now(),
            message=str(error) if isinstance(error,ValueError) else 'External error details withheld',
            frames=[dict(file=f.filename,line=f.lineno,function=f.name) for f in traceback.extract_tb(error.__traceback__)])
        if RUN_DIR is not None:
            write_json(RUN_DIR/'failure.json',failure)
        print(json.dumps(failure),flush=True)
        raise SystemExit(1) from None
