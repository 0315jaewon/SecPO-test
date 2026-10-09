#!/usr/bin/env python3
"""Rejudge verified Lambda answers with Luna; never regenerate model responses."""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from secpo.tinker.evaluation_common import digest, now, write_json
from secpo.tinker import reference_eval as reference
from secpo.tinker.luna_judge import LunaJudge, MODEL, ENDPOINT, PARAMS
from secpo.evaluation.reference_utility import preflight_reference_utility, sha256_file

RUN = ROOT / 'outputs/qwen38_lambda_recommended_20260916'
OUTPUT = ROOT / 'outputs/qwen38_lambda_luna_openai_20260918'


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True, choices=['base', 'secpo', 'secpo_lr6e5', 'secpo_lr4e5'])
    parser.add_argument('--benchmark', required=True, choices=['alpacaeval2', 'sep_utility'])
    parser.add_argument('--smoke', type=int, default=0)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if args.smoke < 0 or args.smoke > 805:
        raise ValueError('Invalid smoke size')
    source = RUN / args.variant / args.benchmark
    directory = OUTPUT / args.variant / args.benchmark
    active = directory / 'smoke' if args.smoke else directory
    active.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'judge.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old = read(source / 'manifest.json')
    progress = read(source / 'progress.json')
    if progress.get('status') not in {'complete', 'awaiting_judgment'}:
        raise ValueError('Generation not complete')
    suite = 'sep' if args.benchmark == 'sep_utility' else 'alpacaeval2'
    ref_path = (Path('/home/jaewonc/SecAlign_cont/data/SEP_dataset_test_Meta-Llama-3-8B-Instruct.json')
        if suite == 'sep' else ROOT / 'data/eval_references/alpaca_eval2_gpt-4o-2024-05-13_official_reference_outputs.json')
    config, references, info = reference.reference_inputs(suite, ref_path,
        approved_sha256=old['reference']['sha256'] if suite == 'sep' else None,
        sep_test_path=Path('/home/jaewonc/SecAlign_cont/data/SEP_dataset_test.json') if suite == 'sep' else None)
    candidate_path = source / 'candidate_outputs.json'
    preflight_reference_utility(config, candidate_path=candidate_path, reference_path=ref_path)
    candidates, generations = read(candidate_path), read(source / 'generations.json')
    if (info != old['reference'] or len(candidates) != len(generations) or len(candidates) != info['rows']
            or progress['completed'] != info['rows']):
        raise ValueError('Saved answer counts/reference identity differ')
    if any(g['index'] != i or g['text'] != c['output'] for i, (c, g) in enumerate(zip(candidates, generations))):
        raise ValueError('Candidate outputs differ from saved generations')
    if args.variant != 'secpo_lr4e5':
        previous = read(source / 'results.json')
        if previous['candidate_sha256'] != sha256_file(candidate_path):
            raise ValueError('Candidate changed since original scoring')
    elif read(source / 'judging_deferred.json')['candidate_sha256'] != sha256_file(candidate_path):
        raise ValueError('Candidate changed since generation finished')
    asset = ROOT / 'data/eval_references/alpacaeval_df_gamed.csv'
    if sha256_file(asset) != reference.ALPACA_METRIC_SHA256:
        raise ValueError('Length-control asset changed')
    sources = reference.evaluator_sources()
    if sources != old['evaluator_sources']:
        raise ValueError('Original evaluator package changed')
    manifest = dict(schema='secpo-luna-rejudge-v1', benchmark=args.benchmark, model_variant=args.variant,
        checkpoint=old['checkpoint'], generation_directory=str(source), generation_manifest_sha256=sha256_file(source / 'manifest.json'),
        candidate_sha256=sha256_file(candidate_path), generations_sha256=sha256_file(source / 'generations.json'),
        reference=info, judge=dict(model=MODEL, endpoint=ENDPOINT, params=PARAMS,
        scoring='original probability-weighted logprob_parser; no binarization',
        source_annotator_template=reference.ANNOTATOR_NAME, annotator_seed=0, randomized_output_order=True),
        evaluator_sources=sources, metric_asset_sha256=reference.ALPACA_METRIC_SHA256,
        implementation_sha256={name: sha256_file(ROOT / name) for name in
            ('scripts/judge_lambda_luna.py', 'src/secpo/tinker/luna_judge.py', 'src/secpo/tinker/reference_eval.py')},
        generation_api_requests_allowed=0)
    if (directory / 'manifest.json').exists() and read(directory / 'manifest.json') != manifest:
        raise ValueError('Judge configuration changed; use new outputs')
    write_json(directory / 'manifest.json', manifest)
    if args.audit_only:
        print(json.dumps(dict(status='audit_passed', rows=len(candidates))), flush=True)
        return
    if (active / 'results.json').exists():
        result = read(active / 'results.json')
        if result.get('manifest_sha256') != digest(manifest) or result.get('n') != (args.smoke or len(candidates)):
            raise ValueError('Existing completed result differs')
        print('already_complete', flush=True)
        return
    count = args.smoke or len(candidates)
    candidates, references = candidates[:count], references[:count]
    status = dict(status='judging', started_at=now(), total=count, model=MODEL, generation_api_requests=0)
    def update(stats):
        write_json(active / 'progress.json', {**status, 'updated_at': now(), 'openai': stats})
    judge = LunaJudge(directory / 'judge_cache', ROOT / 'secpo_openai.env', progress=update)
    update(judge.stats)
    annotator = reference.make_annotator(judge, config)
    next(iter(annotator.annotators.values())).completions_kwargs = dict(PARAMS)
    annotations = annotator.annotate_head2head(outputs_1=references, outputs_2=candidates, is_ordered=True)
    reference.validate_annotations(annotations, candidates, references, config.identity_keys)
    # Keep the source annotator name while rendering to retain its exact seeded
    # ordering, but label saved rows with the model that actually judged them.
    for row in annotations:
        row['annotator'] = 'weighted_alpaca_eval_gpt-5.6-luna'
        row['source_annotator_template'] = reference.ANNOTATOR_NAME
    write_json(active / 'annotations.json', annotations)
    status.update(status='scoring', judged=count)
    update(judge.stats)
    metrics = reference.score_annotations('sep' if args.smoke else suite, annotations,
        metric_asset=asset, metric_asset_sha256=reference.ALPACA_METRIC_SHA256)
    primary = 'win_rate' if args.smoke else config.metric.primary
    if not math.isfinite(float(metrics[primary])):
        raise ValueError('Nonfinite utility score')
    if sha256_file(candidate_path) != manifest['candidate_sha256']:
        raise ValueError('Source answers changed during judging')
    result = dict(status='complete', benchmark=args.benchmark, model_variant=args.variant, n=count,
        smoke=bool(args.smoke), metrics=metrics, judge_model=MODEL, manifest_sha256=digest(manifest),
        candidate_sha256=manifest['candidate_sha256'], reference=info, completed_at=now(),
        generation_api_requests=0, openai_this_process=judge.stats,
        pricing_note='API does not report cost; per-example package price fields are unavailable placeholders')
    write_json(active / 'results.json', result)
    status.update(status='complete', completed=count, completed_at=now())
    update(judge.stats)
    print(json.dumps(dict(status='complete', metrics=metrics), default=float), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(status='failed', error_type=type(error).__name__)), flush=True)
        raise SystemExit(1) from None
