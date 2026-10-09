#!/usr/bin/env python3
"""Evaluate the approved question subsets using lm-eval and Tinker sampling."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from secpo.tinker.evaluation_common import (
    CheckpointSampler, MODEL, TOKENIZER_REVISION, digest,
    load_credentials, load_tokenizer, now, read_indices, write_json, validate_base_comparison,
    validate_output_cap_comparison,
)
from secpo.tinker.evaluation_variants import (
    MODEL_VARIANTS, evaluation_model_id, sampler_model_kwargs, validate_checkpoint_comparison,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["mmlu_pro", "gpqa_diamond", "minerva_math"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, default=ROOT / "secpo_eval_keys.env")
    parser.add_argument("--indices", type=Path)
    parser.add_argument("--smoke", type=int, default=0, help="Evaluate this many selected items in total")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--model-variant", choices=MODEL_VARIANTS, default="secpo")
    parser.add_argument("--inference-config", type=Path, help="Pinned Lambda/vLLM backend configuration")
    parser.add_argument("--max-output-tokens", type=int,
                        help="Override the output cap for every task, including MMLU-Pro")
    args = parser.parse_args()
    if args.max_output_tokens is not None and not 0 < args.max_output_tokens < 65536:
        parser.error("--max-output-tokens must be between 1 and 65535")
    if args.model_variant == "secpo_lr6e5" and args.max_output_tokens is None:
        parser.error("Checkpoint comparisons require an explicit --max-output-tokens")
    load_credentials(args.credentials)
    random.seed(42)
    import numpy as np
    import torch
    np.random.seed(42)
    torch.manual_seed(42)
    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager, get_task_dict
    from lm_eval.evaluator_utils import get_task_list
    from secpo.tinker.lm_eval_backend import TinkerEvaluationLM, set_task_output_cap

    if version("lm_eval") != "0.4.11":
        raise ValueError("Use the isolated lm-eval==0.4.11 environment")
    task_name = "gpqa_diamond_cot_zeroshot" if args.benchmark == "gpqa_diamond" else args.benchmark
    task_dict = get_task_dict([task_name], TaskManager())
    task_list = get_task_list(task_dict)
    supplied = read_indices(args.indices, args.benchmark) if args.indices else None
    if supplied is not None and set(supplied) != {t.task_name for t in task_list}:
        raise ValueError("Index file categories differ from the loaded lm-eval tasks")
    samples = {}
    fingerprints = {}
    for item in task_list:
        task = item.task
        if args.max_output_tokens is not None:
            set_task_output_cap(task, args.max_output_tokens)
        docs = task.eval_docs
        count = len(docs)
        if supplied is not None:
            declared = supplied[item.task_name]
            if count != declared["num_docs"]:
                raise ValueError(f"{item.task_name}: dataset has {count} rows, expected {declared['num_docs']}")
            samples[item.task_name] = declared["indices"]
        else:
            if args.benchmark != "gpqa_diamond" or count != 198:
                raise ValueError("Unexpected full-dataset evaluation requested")
            samples[item.task_name] = list(range(count))
        fingerprints[item.task_name] = {
            "num_docs": count, "dataset_fingerprint": getattr(docs, "_fingerprint", None),
            "selected_documents_sha256": digest([docs[i] for i in samples[item.task_name]]),
            "indices": samples[item.task_name], "generation_kwargs": task.get_config("generation_kwargs"),
            "num_fewshot": task.get_config("num_fewshot"),
        }
    manifest = dict(benchmark=args.benchmark, checkpoint=evaluation_model_id(args.model_variant), model=MODEL,
                    tokenizer_revision=TOKENIZER_REVISION, enable_thinking=False,
                    lm_eval_version=version("lm_eval"), math_verify_version=version("math-verify"),
                    random_seed=42, numpy_seed=42, torch_seed=42, fewshot_seed=42,
                    default_max_gen_toks=args.max_output_tokens or 2048, tasks=fingerprints)
    if args.inference_config:
        from secpo.inference.lambda_eval import configure_protocol
        configure_protocol(manifest, args.inference_config, args.model_variant)
        for item in task_list:
            item.task.set_config("generation_kwargs", manifest["tasks"][item.task_name]["generation_kwargs"])
    elif args.model_variant == "secpo_lr6e5":
        validate_checkpoint_comparison(manifest, args.model_variant, args.output_dir)
    elif args.max_output_tokens is not None:
        validate_output_cap_comparison(manifest, args.model_variant)
    elif args.model_variant == "base":
        validate_base_comparison(manifest)
    fingerprint = digest(manifest)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing evaluation manifest differs; use a new output directory")
    write_json(manifest_path, manifest)
    print(json.dumps({"benchmark":args.benchmark,"selected":sum(map(len,samples.values())),
                      "manifest_sha256":fingerprint}),flush=True)
    tokenizer = load_tokenizer()
    rendered = tokenizer.apply_chat_template([{"role":"user","content":"Reply with OK."}],
                                            tokenize=False, add_generation_prompt=True, enable_thinking=False)
    from secpo.tinker.data import QWEN38_NON_THINKING_SUFFIX
    if not rendered.endswith(QWEN38_NON_THINKING_SUFFIX):
        raise ValueError("Tokenizer thinking-off prefix differs from training")
    if args.preflight_only:
        return
    if args.smoke:
        remaining = args.smoke
        picked = {}
        keep = {}
        for item in task_list:
            selected = samples[item.task_name][:remaining]
            if selected:
                picked[item.task_name] = selected
                keep[item.task_name] = item.task
                remaining -= len(selected)
        samples = picked
        task_dict = keep
    # Use preloaded task instances so the row identities just checked are exactly
    # the ones passed to the evaluator. Set the same four seeds as the old CLI.
    for item in get_task_list(task_dict):
        item.task.set_fewshot_seed(42)
    # lm-eval iterates selected rows in dataset order, but labels them using
    # samples[ordinal]. Sort to keep logged doc_id equal to the original row ID.
    samples = {name: sorted(indices) for name, indices in samples.items()}
    active_dir = args.output_dir / "smoke" if args.smoke else args.output_dir
    active_dir.mkdir(parents=True, exist_ok=True)
    from secpo.inference.lambda_eval import make_sampler
    sampler = make_sampler(args.output_dir / "completion_cache", tokenizer=tokenizer,
                           variant=args.model_variant, config_path=args.inference_config)
    lm = TinkerEvaluationLM(sampler, active_dir, concurrency=args.concurrency,
                            default_max_tokens=args.max_output_tokens or 2048)
    write_json(active_dir / "progress.json", {"status":"starting", "started_at":now(),
                                              "total":sum(map(len,samples.values()))})
    result = evaluator.evaluate(lm=lm, task_dict=task_dict, samples=samples,
                                log_samples=True, apply_chat_template=True, bootstrap_iters=1000)
    for item in get_task_list(task_dict):
        observed = result["samples"][item.task_name]
        expected = [item.task.eval_docs[i] for i in sorted(samples[item.task_name])]
        # One sample record per scoring filter (e.g. strict and flexible GPQA
        # extraction), not one record per generation.
        for filter_name in {row["filter"] for row in observed}:
            rows = [row for row in observed if row["filter"] == filter_name]
            if ([row["doc_id"] for row in rows] != samples[item.task_name]
                    or digest([row["doc"] for row in rows]) != digest(expected)):
                raise RuntimeError(f"Scored document identities differ from the supplied subset: {item.task_name}")
        if not observed:
            raise RuntimeError(f"Missing scored samples: {item.task_name}")
    write_json(active_dir / "results.json", result)
    write_json(active_dir / "progress.json", {"status":"complete", "completed_at":now(),
               "completed":sum(map(len,samples.values())), "total":sum(map(len,samples.values())),
               "length_limited":lm.length_limited,
               **sampler.stats()})
    print(json.dumps({"status":"complete","results":result["results"]},default=str),flush=True)


if __name__ == "__main__":
    main()
