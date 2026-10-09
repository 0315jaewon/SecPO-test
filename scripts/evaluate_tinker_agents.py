#!/usr/bin/env python3
"""Run clean AgentDojo/AgentDyn utility against a saved Tinker checkpoint."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from secpo.tinker.evaluation_common import (
    CheckpointSampler, MODEL, TOKENIZER_REVISION, digest, evaluation_model_id,
    load_credentials, load_tokenizer, now, write_json, validate_base_comparison,
)
from secpo.tinker.agent_eval import BENCHMARKS, make_pipeline, trust_role_template, verify_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=list(BENCHMARKS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, default=ROOT / "secpo_eval_keys.env")
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--model-variant", choices=["secpo", "base"], default="secpo")
    parser.add_argument("--inference-config", type=Path, help="Pinned Lambda/vLLM backend configuration")
    args = parser.parse_args()
    load_credentials(args.credentials)
    config = BENCHMARKS[args.benchmark]
    source = verify_source(args.benchmark)
    from agentdojo.task_suite.load_suites import get_suite
    from agentdojo.benchmark import run_task_without_injection_tasks
    from agentdojo.logging import LOGGER_STACK, OutputLogger

    tokenizer = load_tokenizer()
    suites = {name:get_suite(config["version"],name) for name in config["suites"]}
    tasks = []
    for name, suite in suites.items():
        if len(suite.user_tasks) != config["suites"][name]:
            raise ValueError(f"Unexpected task count in {name}")
        for task_id in sorted(suite.user_tasks, key=lambda x:int(x.rsplit('_',1)[1])):
            tasks.append((name,task_id))
    manifest = dict(benchmark=args.benchmark, benchmark_version=config["version"], source=source,
                    checkpoint=evaluation_model_id(args.model_variant), model=MODEL, tokenizer_revision=TOKENIZER_REVISION,
                    enable_thinking=False, defense="repeat_user_prompt", attack=None,
                    temperature=1.0, top_p=0.95, top_k=20, seed=0, max_output_tokens=8192,
                    tool_loop_max_iters=15, template=trust_role_template(tokenizer), tasks=[list(t) for t in tasks])
    inference = None
    if args.inference_config:
        from secpo.inference.lambda_eval import configure_protocol
        inference = configure_protocol(manifest, args.inference_config, args.model_variant)
    elif args.model_variant == "base":
        validate_base_comparison(manifest)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Evaluation manifest changed; use a new output directory")
    write_json(manifest_path,manifest)
    print(json.dumps({"benchmark":args.benchmark,"tasks":len(tasks),"source":source}),flush=True)
    if args.preflight_only:
        return
    if args.smoke:
        tasks = tasks[:args.smoke]
    active_dir = args.output_dir / "smoke" if args.smoke else args.output_dir
    active_dir.mkdir(parents=True,exist_ok=True)
    from secpo.inference.lambda_eval import make_sampler
    sampler = make_sampler(args.output_dir / "completion_cache", tokenizer=tokenizer,
                           variant=args.model_variant, config_path=args.inference_config)
    # Every thread gets a separate logger stack and each episode gets a fresh
    # pipeline. The upstream runner copies the task environment for each task.
    def episode(pair):
        name,task_id = pair
        path=active_dir / "tasks" / f"{name}-{task_id}.json"
        if path.exists():
            cached=json.loads(path.read_text())
            if cached.get("status")=="complete": return cached
        LOGGER_STACK.set([])
        pipeline=make_pipeline(sampler, sampling_params=inference["sampling"] if inference else None)
        suite=suites[name]
        task=suite.get_user_task_by_id(task_id)
        try:
            with OutputLogger(str(active_dir / "episodes")):
                utility,_=run_task_without_injection_tasks(suite,pipeline,task,
                             active_dir / "episodes",True,config["version"])
            result=dict(suite=name,task_id=task_id,status="complete",utility=bool(utility))
        except Exception as error:
            trace = traceback.format_exc()
            for key in ("TINKER_API_KEY", "HF_TOKEN"):
                secret = os.environ.get(key)
                if secret:
                    trace = trace.replace(secret, "[REDACTED]")
            result=dict(suite=name,task_id=task_id,status="error",utility=None,
                        error_type=type(error).__name__,traceback=trace)
        write_json(path,result)
        return result
    results=[]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures=[pool.submit(episode,pair) for pair in tasks]
        for future in as_completed(futures):
            result=future.result()
            results.append(result)
            errors=sum(r['status']=='error' for r in results)
            write_json(active_dir / "progress.json",dict(status="running",updated_at=now(),
                completed=len(results)-errors,errors=errors,total=len(tasks),**sampler.stats()))
            print(json.dumps({"finished":len(results),"total":len(tasks),**result}),flush=True)
    errors=[r for r in results if r['status']=='error']
    suite_scores={name:{"completed":sum(r['status']=='complete' for r in results if r['suite']==name),
                        "successes":sum(r['utility'] is True for r in results if r['suite']==name),
                        "total":sum(r['suite']==name for r in results)} for name in suites}
    for value in suite_scores.values():
        value['utility_rate']=value['successes']/value['total'] if value['total'] and value['completed']==value['total'] else None
    scores=[v['utility_rate'] for v in suite_scores.values() if v['total']]
    summary=dict(status="failed" if errors else "complete",updated_at=now(),total=len(tasks),
                 completed=len(results)-len(errors),errors=len(errors),suites=suite_scores,
                 macro_average=sum(scores)/len(scores) if scores and all(s is not None for s in scores) else None,
                 **sampler.stats())
    write_json(active_dir / "results.json",summary)
    write_json(active_dir / "progress.json",summary)
    print(json.dumps(summary),flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
