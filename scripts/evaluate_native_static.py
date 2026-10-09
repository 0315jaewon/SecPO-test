#!/usr/bin/env python3
"""Separate Qwen3.8 launch wrapper; original SecPO evaluation modules unchanged."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
from pathlib import Path
from queue import Queue
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from secpo.inference.native_static import (OUTPUT, VARIANTS, ENDPOINT, FILES, BENCHMARKS,
    INJECTION_COUNTS, read, inference_config, template, verify_source, verify_originals,
    configure_environment, pipeline, ChatCache, sha, digest, summarize, now, write_json, served)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=BENCHMARKS, required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--phase', choices=('preflight', 'smoke', 'full'), required=True)
    args = parser.parse_args()
    directory = OUTPUT / args.variant / args.benchmark
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'worker.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    originals = verify_originals()
    source = verify_source(args.benchmark)
    config = BENCHMARKS[args.benchmark]
    inference = {**inference_config(args.variant), 'endpoint':ENDPOINT, 'served_model':served(args.variant)}
    configure_environment(args.benchmark, args.variant)
    from agentdojo.task_suite.load_suites import get_suite
    from agentdojo.attacks.attack_registry import load_attack
    from agentdojo.benchmark import run_task_with_injection_tasks, run_task_without_injection_tasks
    from agentdojo.logging import LOGGER_STACK, OutputLogger
    suites = {name:get_suite(config['version'], name) for name in config['suites']}
    probe = pipeline(args.benchmark, args.variant)
    tasks, payloads = [], {}
    for name, suite in suites.items():
        if len(suite.user_tasks) != config['suites'][name] or len(suite.injection_tasks) != INJECTION_COUNTS[args.benchmark][name]:
            raise ValueError('Task inventory changed')
        attack = load_attack('important_instructions', suite, probe)
        if attack.model_name != 'Qwen 3' or attack.user_name != 'Emma Johnson':
            raise ValueError('Static attack identity differs')
        order = lambda x:int(x.rsplit('_', 1)[1])
        users, injections = sorted(suite.user_tasks, key=order), sorted(suite.injection_tasks, key=order)
        for injection in injections:
            tasks.append(dict(kind='calibration', suite=name, user_task=injection, injection_task=None))
        for user in users:
            for injection in injections:
                task = dict(kind='attack', suite=name, user_task=user, injection_task=injection)
                payload = attack.attack(suite.get_user_task_by_id(user), suite.get_injection_task_by_id(injection))
                if not payload:
                    raise ValueError('No injection candidates')
                tasks.append(task)
                payloads['/'.join((name, user, injection))] = payload
    manifest = dict(schema='secpo-native-static-v1', benchmark=args.benchmark, variant=args.variant,
        source=source, inference=inference, original_evaluators=originals,
        adapter='secpo.evaluation.' + args.benchmark + '_qwen36 (unchanged)',
        launch_overrides='Qwen3.8 checkpoint, thinking-off, 8192 cap, recommended sampling',
        defense='repeat_user_prompt', attack='important_instructions', attack_model_name='Qwen 3',
        attack_user_name='Emma Johnson', enable_thinking=False, max_output_tokens=8192, seed=0,
        tool_loop_max_iters=15, server_tool_parser='qwen3_coder', server_reasoning_parser='qwen3',
        template=template(), tasks=tasks, payloads_sha256=digest(payloads),
        result_policy='original upstream checker booleans; exceptions recorded without inventing scores',
        implementation_sha256={name:sha(ROOT / name) for name in FILES})
    if (directory / 'manifest.json').exists() and read(directory / 'manifest.json') != manifest:
        raise ValueError('Native experiment identity changed')
    if not (directory / 'manifest.json').exists():
        write_json(directory / 'manifest.json', manifest)
        write_json(directory / 'attack_payloads.json', payloads)
        for name in FILES:
            target = directory / 'code_snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
    write_json(directory / 'preflight.json', dict(status='complete', at=now(),
        attacked_episodes=sum(t['kind']=='attack' for t in tasks),
        capability_checks=sum(t['kind']=='calibration' for t in tasks), manifest_sha256=digest(manifest)))
    if args.phase == 'preflight':
        print(json.dumps(read(directory / 'preflight.json')), flush=True)
        return
    selected = tasks
    if args.phase == 'smoke':
        selected = [next(t for t in tasks if t['suite']==suite and t['kind']==kind)
                    for suite in suites for kind in ('calibration', 'attack')]
        if args.benchmark == 'agentdojo':
            selected += [t for t in tasks if t['suite']=='slack' and t['user_task']=='user_task_7'
                         and t['injection_task'] in ('injection_task_2','injection_task_3','injection_task_4')]
    active = directory / 'smoke' if args.phase=='smoke' else directory
    if (active / 'results.json').exists() and read(active / 'results.json')['status']=='complete':
        print('already_complete', flush=True)
        return
    import requests
    with requests.Session() as session:
        session.trust_env = False
        response = session.get(ENDPOINT + '/models', timeout=15)
        response.raise_for_status()
        if [m['id'] for m in response.json()['data']] != [served(args.variant)]:
            raise ValueError('Wrong server model')
    remote = read(OUTPUT / 'remote/server.json')
    if remote['variant'] != args.variant or remote['template_sha256'] != hashlib_sha_template():
        raise ValueError('Server template/model identity differs')
    cache = ChatCache(directory / 'completion_cache', args.variant, digest(manifest))
    agents = Queue()
    for _ in range(3):
        agent = pipeline(args.benchmark, args.variant)
        cache.attach(agent)
        agents.put(agent)

    def episode(task):
        identity = digest(dict(manifest=digest(manifest), task=task))
        path = directory / 'tasks' / (identity + '.json')
        if path.exists():
            old = read(path)
            if old.get('identity') != identity:
                raise ValueError('Episode identity mismatch')
            if old['status']=='complete':
                return old
        start = time.monotonic()
        LOGGER_STACK.set([])
        agent = agents.get()
        try:
            suite = suites[task['suite']]
            with OutputLogger(str(directory / 'episodes')):
                if task['kind']=='calibration':
                    utility, _ = run_task_without_injection_tasks(suite, agent,
                        suite.get_injection_task_by_id(task['user_task']), directory / 'episodes', True, config['version'])
                    success = None
                else:
                    attack = load_attack('important_instructions', suite, agent)
                    user = suite.get_user_task_by_id(task['user_task'])
                    injection = suite.get_injection_task_by_id(task['injection_task'])
                    if attack.attack(user, injection) != payloads['/'.join((task['suite'], task['user_task'], task['injection_task']))]:
                        raise ValueError('Attack payload differs')
                    utilities, securities = run_task_with_injection_tasks(suite, agent, user, attack,
                        directory / 'episodes', True, [task['injection_task']], config['version'])
                    pair = (task['user_task'], task['injection_task'])
                    utility, success = utilities[pair], securities[pair]
                if type(utility) is not bool or (task['kind']=='attack' and type(success) is not bool):
                    raise ValueError('Unexpected checker type')
            record = dict(**task, status='complete', utility=utility, attack_success=success)
        except Exception as error:
            record = dict(**task, status='error', utility=None, attack_success=None,
                error_type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        finally:
            agents.put(agent)
        record.update(identity=identity, elapsed_seconds=time.monotonic()-start, at=now())
        write_json(path, record)
        return record

    records = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(episode, task) for task in selected]
        for future in as_completed(futures):
            if future.cancelled():
                continue
            row = future.result()
            records.append(row)
            summary = summarize(records, suites, len(selected))
            write_json(active / 'progress.json', {**summary, 'updated_at':now(), 'inference':cache.stats()})
            print(json.dumps(dict(finished=len(records), total=len(selected), **row)), flush=True)
            if row['status']=='error':
                for other in futures:
                    other.cancel()
    summary = summarize(records, suites, len(selected))
    summary.update(at=now(), phase=args.phase, manifest_sha256=digest(manifest), inference=cache.stats())
    write_json(active / 'results.json', summary)
    write_json(active / 'progress.json', summary)
    print(json.dumps(summary), flush=True)
    if summary['status']!='complete':
        raise SystemExit(1)


def hashlib_sha_template():
    import hashlib
    return hashlib.sha256(template().encode()).hexdigest()


if __name__ == '__main__':
    main()
