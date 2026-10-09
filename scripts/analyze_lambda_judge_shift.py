#!/usr/bin/env python3
"""Offline audit of the saved mini/Luna judgments. Makes no API requests.

Reconstructs judge prompts and reads their existing content-addressed caches,
replays the original parser/order correction, and independently recomputes
metrics. Source evaluations are read-only. New reports go to /data2 by default.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from secpo.tinker import reference_eval as ref
from secpo.tinker.luna_judge import MODEL as LUNA_MODEL, ENDPOINT, PARAMS
from secpo.tinker.evaluation_common import digest
from secpo.evaluation.reference_utility import load_reference_utility_config

RUNS = dict(mini=ROOT / 'outputs/qwen38_lambda_recommended_20260916',
            luna=ROOT / 'outputs/qwen38_lambda_luna_openai_20260918')
VARIANTS = ('base', 'secpo', 'secpo_lr6e5')
BENCHMARKS = ('alpacaeval2', 'sep_utility')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def mean(values):
    return statistics.mean(values) if values else None


def label(p):
    return 'win' if p > 0.5 else 'loss' if p < 0.5 else 'tie'


def bucket(ratio):
    return '<=1' if ratio <= 1 else '1-2' if ratio <= 2 else '2-4' if ratio <= 4 else '>4'


class CachedDecoder:
    def __init__(self, directory, judge):
        self.directory, self.judge = directory, judge
        self.prompt_hashes = []
        self.providers = Counter()
        self.returned_models = Counter()
        self.missing = []

    def __call__(self, prompts, **kwargs):
        from alpaca_eval.utils import prompt_to_chatml
        expected = ref.JUDGE_PARAMS if self.judge == 'mini' else PARAMS
        assert kwargs == expected
        records = []
        for prompt in prompts:
            messages = prompt_to_chatml(prompt.strip())
            self.prompt_hashes.append(digest(messages))
            payload = dict(model=expected['model_name'], messages=messages,
                           **{k: v for k, v in expected.items() if k != 'model_name'})
            if self.judge == 'mini':
                payload['provider'] = dict(require_parameters=True)
                key = digest(dict(version='secpo-openrouter-judge-v1', endpoint=ref.JUDGE_BASE_URL, request=payload))
            else:
                key = digest(dict(version='secpo-direct-openai-luna-weighted-v1', endpoint=ENDPOINT, request=payload))
            path = self.directory / f'{key}.json'
            if not path.is_file():
                self.missing.append(key)
                raise ValueError('Saved cache missing for a reconstructed prompt; no API fallback exists')
            record = read(path)
            assert record['request_hash'] == key
            assert record['requested_model'] == expected['model_name']
            self.providers[str(record.get('provider', 'direct OpenAI'))] += 1
            self.returned_models[record['returned_model']] += 1
            records.append(record)
        return dict(completions=[r['choice']['message']['content'] for r in records],
                    completions_all=[r['choice'] for r in records],
                    price_per_example=[0.0] * len(records), time_per_example=[0.0] * len(records))


def replay(directory, judge, benchmark, annotations):
    config = load_reference_utility_config(ROOT / 'configs' /
        ('evaluate_alpacaeval2_utility.json' if benchmark == 'alpacaeval2' else 'evaluate_sep_utility.json'))
    decoder = CachedDecoder(directory / 'judge_cache', judge)
    annotator = ref.make_annotator(decoder, config)
    next(iter(annotator.annotators.values())).completions_kwargs = dict(ref.JUDGE_PARAMS if judge == 'mini' else PARAMS)
    references = [{**r, 'output': r['output_1'], 'generator': r['generator_1']} for r in annotations]
    candidates = [{**r, 'output': r['output_2'], 'generator': r['generator_2']} for r in annotations]
    # Avoid passing old annotation/output columns to the fresh annotator.
    allowed = set(config.identity_keys) | {'instruction', 'output', 'generator'}
    references = [{k: v for k, v in r.items() if k in allowed} for r in references]
    candidates = [{k: v for k, v in r.items() if k in allowed} for r in candidates]
    repeated = annotator.annotate_head2head(outputs_1=references, outputs_2=candidates, is_ordered=True)
    ref.validate_annotations(repeated, candidates, references, config.identity_keys)
    maximum_error = max(abs(a['preference'] - b['preference']) for a, b in zip(annotations, repeated))
    assert len(repeated) == len(annotations) and maximum_error < 1e-12
    return dict(rows=len(repeated), reconstructed_requests=len(decoder.prompt_hashes),
                max_preference_error=maximum_error, providers=dict(decoder.providers),
                returned_models=dict(decoder.returned_models)), decoder.prompt_hashes


def raw_diagnostics(rows):
    first = second = no_choice = 0
    missing_mass_bound = []
    labels_missing = top_counts = None
    labels_missing, top_counts, finishes = Counter(), Counter(), Counter()
    for row in rows:
        choice = row.get('preference_raw_completion')
        if not isinstance(choice, dict):
            no_choice += 1
            continue
        finishes[choice.get('finish_reason')] += 1
        top = choice['logprobs']['content'][0]['top_logprobs']
        probs = {t['token']: math.exp(t['logprob']) for t in top}
        top_counts[len(top)] += 1
        present = set(probs) & {'m', 'M'}
        labels_missing[str(2-len(present))] += 1
        if len(present) == 0:
            raise ValueError('No valid verdict token probability')
        # With observed verdict mass v and unreported mass u, a missing
        # verdict can change the normalized binary score by at most u/(v+u).
        # This also allows for other, invalid tokens among the top logprobs.
        unreported_mass = max(0.0, 1 - sum(probs.values()))
        verdict_mass = sum(probs.get(k, 0) for k in ('m', 'M'))
        missing_mass_bound.append(
            unreported_mass / (verdict_mass + unreported_mass)
            if len(present) == 1 else 0.0
        )
        p_second = probs.get('M', 0) / sum(probs.get(k, 0) for k in ('m', 'M'))
        p_saved = row['preference'] - 1
        if abs(p_saved-p_second) < 1e-10 and abs(p_second-0.5) > 1e-10:
            second += 1
        elif abs(p_saved-(1-p_second)) < 1e-10 and abs(p_second-0.5) > 1e-10:
            first += 1
    return dict(top_logprobs_count=dict(top_counts), missing_verdict_labels=dict(labels_missing),
                missing_label_max_raw_score_shift_pp=100*sum(missing_mass_bound)/len(rows),
                finish_reasons=dict(finishes), no_raw_choice=no_choice,
                inferred_candidate_presented_first=first, inferred_candidate_presented_second=second)


def summarize_group(items):
    return dict(n=len(items), mini=100*mean([r['mini_p'] for r in items]),
                luna=100*mean([r['luna_p'] for r in items]),
                delta_pp=100*mean([r['luna_p']-r['mini_p'] for r in items]),
                mini_win_luna_loss=sum(r['mini_label']=='win' and r['luna_label']=='loss' for r in items),
                mini_loss_luna_win=sum(r['mini_label']=='loss' and r['luna_label']=='win' for r in items))


def audit_pair(variant, benchmark, out, replay_cache):
    paths = {judge: root/variant/benchmark for judge, root in RUNS.items()}
    data = {judge: read(p/'annotations.json') for judge,p in paths.items()}
    results = {judge: read(p/'results.json') for judge,p in paths.items()}
    manifests = {judge: read(p/'manifest.json') for judge,p in paths.items()}
    a,b = data['mini'],data['luna']
    assert len(a)==len(b)==results['mini']['n']==results['luna']['n']
    assert results['mini']['candidate_sha256']==results['luna']['candidate_sha256']
    assert results['mini']['reference']==results['luna']['reference']
    assert manifests['mini']['evaluator_sources']==manifests['luna']['evaluator_sources']==ref.evaluator_sources()
    ref_path = ROOT/'data/eval_references/alpaca_eval2_gpt-4o-2024-05-13_official_reference_outputs.json' if benchmark=='alpacaeval2' else Path('/home/jaewonc/SecAlign_cont/data/SEP_dataset_test_Meta-Llama-3-8B-Instruct.json')
    assert sha(ref_path)==results['mini']['reference']['sha256']
    assert sha(paths['mini']/'candidate_outputs.json')==results['mini']['candidate_sha256']
    entries=[]
    for i,(x,y) in enumerate(zip(a,b)):
        for field in ('instruction','output_1','output_2','generator_1','generator_2'):
            assert x[field]==y[field],(i,field)
        ratio=len(x['output_2'])/max(1,len(x['output_1']))
        entry=dict(index=i, dataset=x.get('dataset','SEP'), mini_p=x['preference']-1,
            luna_p=y['preference']-1, mini_label=label(x['preference']-1), luna_label=label(y['preference']-1),
            reference_chars=len(x['output_1']), candidate_chars=len(x['output_2']),
            reference_words=len(x['output_1'].split()), candidate_words=len(x['output_2'].split()),
            length_ratio=ratio,length_group=bucket(ratio), instruction=x['instruction'])
        entries.append(entry)
    directory=out/variant/benchmark
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'paired_examples.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(entries[0])); writer.writeheader(); writer.writerows(entries)
    # Full texts are local research data. Save disagreements for reproducible
    # qualitative inspection; no extra paid judge is called.
    flips=[dict(index=i,instruction=x['instruction'],reference=x['output_1'],candidate=x['output_2'],
                mini_p=x['preference']-1,luna_p=y['preference']-1,group=x.get('dataset','SEP'))
           for i,(x,y) in enumerate(zip(a,b)) if label(x['preference']-1)!=label(y['preference']-1)]
    (directory/'disagreements.json').write_text(json.dumps(flips,indent=2)+'\n')
    groups={}
    for field in ('length_group','dataset'):
        by=defaultdict(list)
        for r in entries: by[r[field]].append(r)
        groups[field]={key:summarize_group(value) for key,value in sorted(by.items())}
    from scipy.stats import spearmanr
    deltas=[r['luna_p']-r['mini_p'] for r in entries]
    corr=spearmanr([r['length_ratio'] for r in entries],deltas)
    transition=Counter(f"{r['mini_label']}->{r['luna_label']}" for r in entries)
    summary=dict(variant=variant,benchmark=benchmark,n=len(a),reference=results['mini']['reference'],
        candidate_sha256=results['mini']['candidate_sha256'],
        metrics={j:r['metrics'] for j,r in results.items()},
        raw_delta_pp=100*mean(deltas),
        discrete_delta_pp=results['luna']['metrics']['discrete_win_rate']-results['mini']['metrics']['discrete_win_rate'],
        transitions=dict(transition),groups=groups,
        lengths={k:mean([r[k] for r in entries]) for k in ('reference_chars','candidate_chars','reference_words','candidate_words')},
        length_ratio_delta_spearman=float(corr.statistic),
        raw_diagnostics={j:raw_diagnostics(v) for j,v in data.items()},
        annotation_sha256={j:sha(p/'annotations.json') for j,p in paths.items()},
        cache_replay={})
    if benchmark=='alpacaeval2':
        mini,luna=results['mini']['metrics'],results['luna']['metrics']
        summary['lc_delta_pp']=luna['length_controlled_winrate']-mini['length_controlled_winrate']
        summary['lc_minus_raw_delta_pp']=summary['lc_delta_pp']-summary['raw_delta_pp']
    asset=ROOT/'data/eval_references/alpacaeval_df_gamed.csv'
    for judge in ('mini','luna'):
        recomputed=ref.score_annotations('sep' if benchmark=='sep_utility' else benchmark,data[judge],
            metric_asset=asset,metric_asset_sha256=ref.ALPACA_METRIC_SHA256)
        fields=['win_rate','discrete_win_rate']+(['length_controlled_winrate'] if benchmark=='alpacaeval2' else [])
        errors={key:recomputed[key]-results[judge]['metrics'][key] for key in fields}
        assert all(abs(e)<1e-8 for e in errors.values()),errors
        summary.setdefault('recomputed_metric_error',{})[judge]=errors
    if replay_cache:
        prompt_hashes={}
        for judge in ('mini','luna'):
            summary['cache_replay'][judge],prompt_hashes[judge]=replay(paths[judge],judge,benchmark,data[judge])
        assert Counter(prompt_hashes['mini'])==Counter(prompt_hashes['luna'])
        summary['identical_rendered_judge_requests']=True
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('variant','benchmark','n','raw_delta_pp','transitions','groups')}),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('/data2/jaewonc/secpo_analysis/judge_shift_20260923'))
    parser.add_argument('--skip-cache-replay',action='store_true')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries=[]
    for variant in VARIANTS:
        for benchmark in BENCHMARKS:
            print(f'Auditing {variant}/{benchmark}',flush=True)
            summaries.append(audit_pair(variant,benchmark,args.output_dir,not args.skip_cache_replay))
    (args.output_dir/'summary.json').write_text(json.dumps(dict(api_requests=0,
        script_sha256=sha(Path(__file__)),comparisons=summaries),indent=2)+'\n')
    print('Offline audit complete:',args.output_dir,flush=True)


if __name__=='__main__':
    main()
