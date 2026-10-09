#!/usr/bin/env python3
"""Pinned static prompt-injection episodes with original exact success checkers."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from secpo.tinker.evaluation_common import digest, now, write_json, MODEL, TOKENIZER_REVISION
from secpo.tinker.agent_eval import BENCHMARKS, verify_source, trust_role_template
from secpo.inference.lambda_eval import LambdaSampler, QWEN_SAMPLING
from secpo.inference.static_agent_eval import (OUTPUT, VARIANTS, ATTACK, DEFENSE, INJECTION_COUNTS,
    FILES, read, inference_config, pipeline, summarize)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=BENCHMARKS, required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--phase', choices=['preflight','smoke','full'], required=True)
    args = parser.parse_args()
    directory = OUTPUT / args.variant / args.benchmark
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'worker.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = BENCHMARKS[args.benchmark]
    source = verify_source(args.benchmark)
    inference = inference_config(args.variant)
    prior = read(ROOT / 'outputs/qwen38_lambda_recommended_20260916' / args.variant / args.benchmark / 'manifest.json')
    if (prior['source'] != source or prior['defense'] != DEFENSE or prior['attack'] is not None
            or prior['enable_thinking'] is not False or prior['max_output_tokens'] != 8192
            or prior['tool_loop_max_iters'] != 15 or prior['seed'] != 0
            or prior['inference']['checkpoint'] != inference['checkpoint']):
        raise ValueError('Clean utility comparison protocol differs')
    stable = lambda c:{k:v for k,v in c.items() if k not in ('endpoint','served_model')}
    if stable(prior['inference']) != stable(inference):
        raise ValueError('Inference settings changed beyond server identity')
    from transformers import AutoTokenizer
    model = Path('/data2/jaewonc/secpo_backups/lambda_20260919/models/base')
    expected = read(Path('/data2/jaewonc/secpo_evals/qwen38_thinking_alpaca_20260923/model_validation.json'))
    tokenizer_files = ('tokenizer.json','tokenizer_config.json','chat_template.jinja','vocab.json','merges.txt','config.json')
    hashes = {r['path'].rsplit('/',1)[-1]:r['sha256'] for r in expected['files']}
    for name in tokenizer_files:
        if sha(model/name) != hashes[name]:
            raise ValueError('Tokenizer changed')
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False, local_files_only=True)
    if prior['template'] != trust_role_template(tokenizer):
        raise ValueError('Agent trust-role template differs from earlier evaluations')
    from agentdojo.task_suite.load_suites import get_suite
    from agentdojo.attacks.attack_registry import load_attack
    from agentdojo.benchmark import run_task_with_injection_tasks, run_task_without_injection_tasks
    from agentdojo.logging import LOGGER_STACK, OutputLogger
    suites = {name:get_suite(config['version'],name) for name in config['suites']}
    tasks, payloads = [], {}
    dummy = SimpleNamespace(tokenizer=tokenizer, is_base_model=False)
    for name,suite in suites.items():
        if len(suite.user_tasks)!=config['suites'][name] or len(suite.injection_tasks)!=INJECTION_COUNTS[args.benchmark][name]:
            raise ValueError('Benchmark task inventory changed')
        probe = pipeline(dummy, args.variant)
        attack = load_attack(ATTACK,suite,probe)
        if attack.model_name != 'Qwen 3' or attack.user_name != 'Emma Johnson' or attack.is_dos_attack:
            raise ValueError('Unexpected static attack configuration')
        order = lambda k:int(k.rsplit('_',1)[1])
        users, injections = sorted(suite.user_tasks,key=order), sorted(suite.injection_tasks,key=order)
        for inj in injections:
            tasks.append(dict(kind='calibration',suite=name,user_task=inj,injection_task=None))
        for user in users:
            for inj in injections:
                key = '/'.join((name,user,inj))
                payloads[key] = attack.attack(suite.get_user_task_by_id(user),suite.get_injection_task_by_id(inj))
                if not payloads[key]:
                    raise ValueError('No injectable positions found')
                tasks.append(dict(kind='attack',suite=name,user_task=user,injection_task=inj))
    manifest = dict(schema='secpo-lambda-static-v1', benchmark=args.benchmark, variant=args.variant,
        model=MODEL, tokenizer_revision=TOKENIZER_REVISION, source=source, inference=inference,
        defense=DEFENSE, attack=ATTACK, attack_model_name='Qwen 3', attack_user_name='Emma Johnson',
        enable_thinking=False, max_output_tokens=8192, seed=0, tool_loop_max_iters=15,
        template=trust_role_template(tokenizer), tasks=tasks, payloads_sha256=digest(payloads),
        clean_comparison_sha256=digest(prior), error_policy='errors block final score; never silent successes',
        scoring='macro-average attacker success over suites; also retain micro ASR and utility under attack',
        implementation_sha256={name:sha(ROOT/name) for name in FILES})
    path = directory / 'manifest.json'
    if path.exists() and read(path) != manifest:
        raise ValueError('Static experiment changed; use a new directory')
    if not path.exists():
        write_json(path,manifest)
        write_json(directory/'attack_payloads.json',payloads)
        for name in FILES:
            target=directory/'code_snapshot'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,target)
    write_json(directory/'preflight.json',dict(status='complete',at=now(),
        attacked_episodes=sum(t['kind']=='attack' for t in tasks),
        capability_checks=sum(t['kind']=='calibration' for t in tasks),manifest_sha256=digest(manifest)))
    if args.phase=='preflight':
        print(json.dumps(read(directory/'preflight.json')),flush=True)
        return
    selected = tasks
    if args.phase=='smoke':
        selected=[next(t for t in tasks if t['suite']==suite and t['kind']==kind)
                  for suite in suites for kind in ('calibration','attack')]
    active=directory/'smoke' if args.phase=='smoke' else directory
    if (active/'results.json').exists() and read(active/'results.json')['status']=='complete':
        print('already_complete',flush=True)
        return
    validation=read(OUTPUT/'remote/model_validation.json')
    if validation[args.variant]['status']!='complete':
        raise ValueError('Remote checkpoint verification absent')
    sampler=LambdaSampler(directory/'completion_cache',tokenizer=tokenizer,config=inference)

    class StrictOutputLogger(OutputLogger):
        def log_error(self,message):
            super().log_error(message)
            raise RuntimeError('Upstream episode error; refusing to score it as attacker success')

    def episode(task):
        identity=digest(dict(manifest=digest(manifest),task=task))
        path=directory/'tasks'/(identity+'.json')
        if path.exists():
            record=read(path)
            if record.get('identity')!=identity:
                raise ValueError('Episode cache identity mismatch')
            if record['status']=='complete':return record
        LOGGER_STACK.set([])
        start=time.monotonic()
        suite=suites[task['suite']]
        agent=pipeline(sampler,args.variant)
        try:
            with StrictOutputLogger(str(directory/'episodes')):
                if task['kind']=='calibration':
                    utility,_=run_task_without_injection_tasks(suite,agent,
                        suite.get_injection_task_by_id(task['user_task']),directory/'episodes',True,config['version'])
                    attack_success=None
                else:
                    attacker=load_attack(ATTACK,suite,agent)
                    user=suite.get_user_task_by_id(task['user_task'])
                    injection=suite.get_injection_task_by_id(task['injection_task'])
                    actual=attacker.attack(user,injection)
                    if actual!=payloads['/'.join((task['suite'],task['user_task'],task['injection_task']))]:
                        raise ValueError('Injection differs from preflight')
                    utilities,security=run_task_with_injection_tasks(suite,agent,user,attacker,
                        directory/'episodes',True,[task['injection_task']],config['version'])
                    pair=(task['user_task'],task['injection_task'])
                    utility,attack_success=utilities[pair],security[pair]
                if type(utility) is not bool or (task['kind']=='attack' and type(attack_success) is not bool):
                    raise ValueError('Unexpected checker result type')
            record=dict(**task,status='complete',utility=utility,attack_success=attack_success)
        except Exception as error:
            record=dict(**task,status='error',utility=None,attack_success=None,
                error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        record.update(identity=identity,elapsed_seconds=time.monotonic()-start,at=now())
        write_json(path,record)
        return record

    records=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(episode,task) for task in selected]
        for future in as_completed(futures):
            row=future.result()
            records.append(row)
            summary=summarize(records,suites,len(selected))
            write_json(active/'progress.json',{**summary,'updated_at':now(),'inference':sampler.stats()})
            print(json.dumps(dict(finished=len(records),total=len(selected),**row)),flush=True)
            if row['status']=='error':
                for other in futures:other.cancel()
                break
    summary=summarize(records,suites,len(selected))
    summary.update(at=now(),phase=args.phase,manifest_sha256=digest(manifest),inference=sampler.stats())
    write_json(active/'results.json',summary)
    write_json(active/'progress.json',summary)
    print(json.dumps(summary),flush=True)
    if summary['status']!='complete':raise SystemExit(1)


if __name__=='__main__':
    main()
