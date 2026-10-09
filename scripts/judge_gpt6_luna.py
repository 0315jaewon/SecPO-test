#!/usr/bin/env python3
"""Rejudge five saved AlpacaEval2 answer sets; never generate new answers."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from secpo.tinker.evaluation_common import digest, now, write_json
from secpo.tinker import reference_eval as ref
from secpo.tinker.gpt6_luna_judge import GPT6LunaJudge, MODEL, PARAMS, ENDPOINT
from secpo.evaluation.reference_utility import preflight_reference_utility, sha256_file

OLD = ROOT / 'outputs/qwen38_lambda_recommended_20260916'
OLD_JUDGE = ROOT / 'outputs/qwen38_lambda_luna_openai_20260918'
THINKING = Path('/data2/jaewonc/secpo_evals/qwen38_thinking_alpaca_20260923/base/alpacaeval2')
OUTPUT = Path('/data2/jaewonc/secpo_evals/gpt6_luna_alpaca_20260923')
VARIANTS = ('base', 'secpo', 'secpo_lr6e5', 'secpo_lr4e5', 'base_thinking')
REFERENCE = ROOT / 'data/eval_references/alpaca_eval2_gpt-4o-2024-05-13_official_reference_outputs.json'
ASSET = ROOT / 'data/eval_references/alpacaeval_df_gamed.csv'
SOURCES = ('scripts/judge_gpt6_luna.py', 'src/secpo/tinker/gpt6_luna_judge.py',
    'src/secpo/tinker/luna_judge.py', 'src/secpo/tinker/reference_eval.py',
    'src/secpo/tinker/evaluation_common.py', 'src/secpo/evaluation/reference_utility.py')


def read(path):
    return json.loads(path.read_text())


def optional_read(path):
    return read(path) if path.exists() else {}


def source_dir(variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown model variant')
    return THINKING if variant == 'base_thinking' else OLD / variant / 'alpacaeval2'


def generation_ready(source):
    # Judging by other models need not finish first. These full files are only
    # published after all 805 generation futures finish; validate their contents.
    return (optional_read(source / 'progress.json').get('completed') == 805
            and all((source / name).is_file() for name in
                    ('manifest.json', 'candidate_outputs.json', 'generations.json')))


def validate_rows(candidates, generations, references, identity_keys):
    if len(candidates) != 805 or len(generations) != 805 or len(references) != 805:
        raise ValueError('Expected all 805 saved responses')
    for i, (candidate, generation, reference) in enumerate(zip(candidates, generations, references)):
        if (generation['index'] != i or generation['text'] != candidate['output']
                or any(candidate[k] != reference[k] for k in identity_keys)):
            raise ValueError('Saved answer text/order/reference mismatch')


def audit(variant):
    source = source_dir(variant)
    if not generation_ready(source):
        raise ValueError('Generation not complete')
    old = read(source / 'manifest.json')
    config, references, info = ref.reference_inputs('alpacaeval2', REFERENCE)
    candidate_path = source / 'candidate_outputs.json'
    preflight_reference_utility(config, candidate_path=candidate_path, reference_path=REFERENCE)
    candidates, generations = read(candidate_path), read(source / 'generations.json')
    validate_rows(candidates, generations, references, config.identity_keys)
    sources = ref.evaluator_sources()
    if old['reference'] != info or sources != old['evaluator_sources']:
        raise ValueError('Original reference/evaluator changed')
    if sha256_file(ASSET) != ref.ALPACA_METRIC_SHA256:
        raise ValueError('Length-control asset changed')
    if variant != 'base_thinking':
        previous = read(OLD_JUDGE / variant / 'alpacaeval2/results.json')
        if previous['candidate_sha256'] != sha256_file(candidate_path):
            raise ValueError('Saved answers changed since GPT-5.6 Luna scoring')
    else:
        if old['thinking']['enable_thinking'] is not True or old['max_output_tokens'] != 32768:
            raise ValueError('Unexpected thinking protocol')
        # generation.text is the extracted final answer, never raw_text/reasoning.
        if any(not g['reasoning_complete'] and c['output'] != ''
               for g, c in zip(generations, candidates)):
            raise ValueError('Unfinished reasoning must have an empty final answer')
    manifest = dict(schema='secpo-gpt6-luna-alpaca-v1', model_variant=variant,
        benchmark='alpacaeval2', generation_directory=str(source),
        generation_manifest_sha256=sha256_file(source / 'manifest.json'),
        candidate_sha256=sha256_file(candidate_path),
        generations_sha256=sha256_file(source / 'generations.json'),
        checkpoint=old.get('checkpoint', old.get('model_identity')), reference=info,
        judge=dict(model=MODEL, endpoint=ENDPOINT, params=PARAMS,
            scoring='original probability-weighted logprob_parser; no binarization',
            template=ref.ANNOTATOR_NAME, seed=0, randomized_output_order=True),
        evaluator_sources=sources, metric_asset_sha256=ref.ALPACA_METRIC_SHA256,
        implementation_sha256={name:sha256_file(ROOT / name) for name in SOURCES},
        generation_api_requests_allowed=0)
    return config, references, candidates, manifest


def snapshot():
    identity = {name:sha256_file(ROOT / name) for name in SOURCES}
    path = OUTPUT / 'source_identity.json'
    if path.exists() and read(path) != identity:
        raise ValueError('GPT-6 experiment sources changed; use a separate output directory')
    if not path.exists():
        for name in SOURCES:
            dest = OUTPUT / 'code_snapshot' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        write_json(path, identity)
    return identity


def worker(variant, smoke=0, audit_only=False):
    directory = OUTPUT / variant / 'alpacaeval2'
    active = directory / 'smoke' if smoke else directory
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'judge.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config, references, candidates, manifest = audit(variant)
        if audit_only:
            print(json.dumps(dict(status='audit_passed', variant=variant, rows=len(candidates))), flush=True)
            return
        snapshot()
        if (directory / 'manifest.json').exists() and read(directory / 'manifest.json') != manifest:
            raise ValueError('Judge experiment manifest changed')
        write_json(directory / 'manifest.json', manifest)
        count = smoke or len(candidates)
        if (active / 'results.json').exists():
            result = read(active / 'results.json')
            if result.get('manifest_sha256') != digest(manifest) or result.get('n') != count:
                raise ValueError('Existing result identity mismatch')
            print('already_complete', flush=True)
            return
        status = dict(status='judging', started_at=now(), total=count, model=MODEL,
                      generation_api_requests=0)
        def update(stats):
            write_json(active / 'progress.json', {**status, 'updated_at':now(), 'openai':stats})
        judge = GPT6LunaJudge(directory / 'judge_cache', ROOT / 'secpo_openai.env',
                             concurrency=4, progress=update)
        update(judge.stats)
        annotator = ref.make_annotator(judge, config)
        next(iter(annotator.annotators.values())).completions_kwargs = dict(PARAMS)
        candidates, references = candidates[:count], references[:count]
        annotations = annotator.annotate_head2head(outputs_1=references, outputs_2=candidates, is_ordered=True)
        ref.validate_annotations(annotations, candidates, references, config.identity_keys)
        for row in annotations:
            row['annotator'] = 'weighted_alpaca_eval_gpt-6-luna'
            row['source_annotator_template'] = ref.ANNOTATOR_NAME
        write_json(active / 'annotations.json', annotations)
        status.update(status='scoring', judged=count)
        update(judge.stats)
        metrics = ref.score_annotations('sep' if smoke else 'alpacaeval2', annotations,
            metric_asset=ASSET, metric_asset_sha256=ref.ALPACA_METRIC_SHA256)
        primary = 'win_rate' if smoke else config.metric.primary
        if not math.isfinite(float(metrics[primary])):
            raise ValueError('Nonfinite metric')
        if sha256_file(source_dir(variant) / 'candidate_outputs.json') != manifest['candidate_sha256']:
            raise ValueError('Source answers changed during scoring')
        write_json(active / 'results.json', dict(status='complete', model_variant=variant,
            benchmark='alpacaeval2', n=count, smoke=bool(smoke), metrics=metrics,
            judge_model=MODEL, manifest_sha256=digest(manifest),
            candidate_sha256=manifest['candidate_sha256'], completed_at=now(),
            generation_api_requests=0, openai_this_process=judge.stats,
            pricing_note='Package price placeholders are not billing estimates'))
        status.update(status='complete', completed=count, completed_at=now())
        update(judge.stats)
        print(json.dumps(dict(status='complete', variant=variant, smoke=bool(smoke), metrics=metrics), default=float), flush=True)


def supervise(args):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = os.fdopen(args.lock_fd, 'a+') if args.lock_fd is not None else (OUTPUT / 'supervisor.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    snapshot()
    if args.detach:
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--queue', '--lock-fd', str(lock.fileno())]
        with (OUTPUT / 'supervisor.log').open('a') as log:
            proc = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
        write_json(OUTPUT / 'launch.json', dict(at=now(), pid=proc.pid, command=command))
        print(json.dumps(dict(status='launched', pid=proc.pid, output=str(OUTPUT))), flush=True)
        return
    pending, queued, failed = {}, list(VARIANTS), []
    status = dict(status='running', pid=os.getpid(), started_at=now(), judge_model=MODEL,
        variants=VARIANTS, max_parallel_jobs=2, concurrency_per_job=4, generation_api_requests=0)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MPLBACKEND='Agg')
    try:
        while queued or pending:
            snapshot()
            for variant in list(queued):
                if len(pending) >= 2:
                    break
                directory = OUTPUT / variant / 'alpacaeval2'
                if optional_read(directory / 'results.json').get('status') == 'complete':
                    audit(variant)  # Still refuse mutated inputs on queue resume.
                    queued.remove(variant)
                    continue
                if not generation_ready(source_dir(variant)):
                    continue
                phase = 'full' if optional_read(directory / 'smoke/results.json').get('status') == 'complete' else 'smoke'
                directory.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, '-u', str(Path(__file__).resolve()), '--variant', variant]
                if phase == 'smoke':
                    command += ['--smoke', '2']
                with (directory / (phase + '.log')).open('a') as log:
                    proc = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, env=env)
                record = dict(status='running', pid=proc.pid, command=command, phase=phase, started_at=now())
                write_json(directory / (phase + '_worker.json'), record)
                pending[variant] = (proc, directory, record)
                queued.remove(variant)
            for variant, (proc, directory, record) in list(pending.items()):
                code = proc.poll()
                if code is None:
                    continue
                active = directory / 'smoke' if record['phase'] == 'smoke' else directory
                valid = code == 0 and optional_read(active / 'results.json').get('status') == 'complete'
                record.update(status='complete' if valid else 'failed', exit_code=code, ended_at=now())
                write_json(directory / (record['phase'] + '_worker.json'), record)
                if valid and record['phase'] == 'smoke':
                    queued.insert(0, variant)
                elif not valid:
                    failed.append(dict(variant=variant, phase=record['phase'], exit_code=code))
                del pending[variant]
            status.update(updated_at=now(), active=list(pending), queued=list(queued), failures=failed)
            if not pending and queued:
                status['status'] = 'waiting_for_saved_generations'
            else:
                status['status'] = 'running'
            write_json(OUTPUT / 'supervisor.json', status)
            if queued or pending:
                time.sleep(10)
        status.update(status='failed' if failed else 'complete', ended_at=now())
    except BaseException as error:
        status.update(status='failed', error_type=type(error).__name__, ended_at=now())
        raise
    finally:
        write_json(OUTPUT / 'supervisor.json', status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--variant', choices=VARIANTS)
    mode.add_argument('--queue', action='store_true')
    parser.add_argument('--smoke', type=int, default=0)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--lock-fd', type=int)
    args = parser.parse_args()
    if args.smoke not in (0, 2) or (not args.queue and (args.detach or args.lock_fd is not None)):
        parser.error('Use --smoke 2, and --detach/--lock-fd only with --queue')
    if args.queue:
        supervise(args)
    else:
        worker(args.variant, args.smoke, args.audit_only)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(status='failed', error_type=type(error).__name__,
            message=str(error) if isinstance(error, (ValueError, RuntimeError)) else 'Details withheld',
            frames=[dict(file=f.filename, line=f.lineno, function=f.name)
                    for f in traceback.extract_tb(error.__traceback__)])), flush=True)
        raise SystemExit(1) from None
