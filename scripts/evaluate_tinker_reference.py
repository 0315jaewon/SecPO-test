#!/usr/bin/env python3
"""Generate full Tinker utility answers and score with AlpacaEval/OpenRouter."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from secpo.evaluation.reference_utility import preflight_reference_utility, sha256_file
from secpo.tinker.evaluation_common import (
    CheckpointSampler, MODEL, TOKENIZER_REVISION, digest,
    load_credentials, load_tokenizer, now, write_json,
)
from secpo.tinker.evaluation_variants import (
    MODEL_VARIANTS, evaluation_model_id, sampler_model_kwargs, validate_checkpoint_comparison,
)
from secpo.tinker.reference_eval import (
    ALPACA_METRIC_SHA256, ANNOTATOR_NAME, JUDGE_BASE_URL, JUDGE_MODEL, JUDGE_PARAMS, OpenRouterJudge,
    evaluator_sources, make_annotator, read_openrouter_credentials, reference_inputs,
    render_clean_prompt, score_annotations, validate_annotations,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["alpacaeval2", "sep_utility"], required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sha256", help="Explicitly approved alternate SEP reference hash")
    parser.add_argument("--sep-test-dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-variant", choices=MODEL_VARIANTS, required=True)
    parser.add_argument("--inference-config", type=Path, help="Pinned Lambda/vLLM backend configuration")
    parser.add_argument("--credentials", type=Path, default=ROOT / "secpo_eval_keys.env")
    parser.add_argument("--judge-credentials", type=Path, default=ROOT / "secpo_openrouter.env")
    parser.add_argument("--metric-asset", type=Path,
                        default=ROOT / "data/eval_references/alpacaeval_df_gamed.csv")
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--judge-concurrency", type=int, default=8)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not 0 < args.max_output_tokens < 65536 or args.concurrency < 1 or args.judge_concurrency < 1:
        parser.error("Invalid output budget or concurrency")
    if args.smoke < 0:
        parser.error("Smoke count cannot be negative")
    suite = "sep" if args.benchmark == "sep_utility" else "alpacaeval2"
    config, references, reference_info = reference_inputs(
        suite, args.reference, approved_sha256=args.reference_sha256,
        sep_test_path=args.sep_test_dataset,
    )
    if args.smoke > len(references):
        parser.error("Smoke count exceeds the benchmark size")
    load_credentials(args.credentials)
    read_openrouter_credentials(args.judge_credentials)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("httpcore").setLevel(logging.ERROR)
    tokenizer = load_tokenizer()
    prompts = [render_clean_prompt(tokenizer, suite, row) for row in references]
    prompt_lengths = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
    if any(length + args.max_output_tokens > 65536 for length in prompt_lengths):
        raise ValueError("A prompt exceeds the Tinker context budget; no generation started")
    asset_hash = sha256_file(args.metric_asset) if suite == "alpacaeval2" else None
    if suite == "alpacaeval2" and asset_hash != ALPACA_METRIC_SHA256:
        raise ValueError("AlpacaEval metric asset differs from the verified copy")
    model_id = evaluation_model_id(args.model_variant)
    manifest = {
        "schema_version": "secpo-tinker-reference-eval-v1", "benchmark": args.benchmark,
        "model": MODEL, "checkpoint": model_id, "tokenizer_revision": TOKENIZER_REVISION,
        "reference": reference_info, "prompt_sha256": digest(prompts),
        "enable_thinking": False, "default_max_gen_toks": args.max_output_tokens,
        "generation": {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "seed": 42},
        "prompt_policy": "trusted-user; SEP clean data in input role; no system or repeated user prompt",
        "judge": {"model": JUDGE_MODEL, "endpoint": JUDGE_BASE_URL,
                  "params": JUDGE_PARAMS, "provider": {"require_parameters": True},
                  "source_annotator": ANNOTATOR_NAME, "annotator_seed": 0,
                  "is_randomize_output_order": True},
        "evaluator_sources": evaluator_sources(), "metric": asdict(config.metric),
        "metric_asset_sha256": asset_hash,
        "implementation_sha256": {name: sha256_file(ROOT / name) for name in (
            "scripts/evaluate_tinker_reference.py", "src/secpo/tinker/reference_eval.py",
            "src/secpo/tinker/evaluation_common.py", "src/secpo/tinker/agent_eval.py",
            "src/secpo/evaluation/reference_utility.py")},
    }
    if args.inference_config:
        from secpo.inference.lambda_eval import configure_protocol
        configure_protocol(manifest, args.inference_config, args.model_variant)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing manifest differs; use a new output directory")
    # Both variants must share a protocol whenever the peer manifest exists.
    peer_name = ("qwen38_27b_tinker_utility_max" if args.model_variant == "base"
                 else "qwen38_27b_base_tinker_utility_max") + str(args.max_output_tokens)
    peer_path = ROOT / "outputs" / peer_name / args.benchmark / "manifest.json"
    if args.inference_config:
        pass  # configure_protocol validated the old reference, judge and metric settings.
    elif args.model_variant == "secpo_lr6e5":
        validate_checkpoint_comparison(manifest, args.model_variant, args.output_dir)
    elif peer_path.exists():
        peer = json.loads(peer_path.read_text())
        settings = lambda value: {k: v for k, v in value.items() if k != "checkpoint"}
        if settings(peer) != settings(manifest):
            raise ValueError("Base and SecPO reference-evaluation protocols differ")
    write_json(manifest_path, manifest)
    print(json.dumps({"status": "preflight_passed", "benchmark": args.benchmark,
                      "rows": len(references), "max_prompt_tokens": max(prompt_lengths),
                      "manifest_sha256": digest(manifest), "reference": reference_info}), flush=True)
    if args.preflight_only:
        return
    active_dir = args.output_dir / "smoke" if args.smoke else args.output_dir
    active_dir.mkdir(parents=True, exist_ok=True)
    if (active_dir / "results.json").exists():
        previous = json.loads((active_dir / "results.json").read_text())
        if (previous.get("manifest_sha256") != digest(manifest)
                or previous.get("n") != (args.smoke or len(references))
                or previous.get("smoke") != bool(args.smoke)):
            raise ValueError("Completed result belongs to another configuration")
        print("already_complete", flush=True)
        return
    count = args.smoke or len(references)
    selected_references = references[:count]
    selected_prompts = prompts[:count]
    generation_name = "Qwen3.8-27B-base" if args.model_variant == "base" else "Qwen3.8-27B-SecPO"
    if args.model_variant == "secpo_lr6e5":
        generation_name = "Qwen3.8-27B-SecPO-lr6e-5"
    from secpo.inference.lambda_eval import make_sampler
    sampler = make_sampler(args.output_dir / "completion_cache", tokenizer=tokenizer,
                           variant=args.model_variant, config_path=args.inference_config)
    candidates = [None] * count
    generation_rows = [None] * count
    progress = {"status": "generating", "started_at": now(), "total": count,
                "completed": 0, "judged": 0, "length_limited": 0}
    write_json(active_dir / "progress.json", progress)
    prompt_indices = {}
    for i, prompt in enumerate(selected_prompts):
        prompt_indices.setdefault(prompt, []).append(i)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        jobs = {pool.submit(sampler.generate, prompt, max_tokens=args.max_output_tokens,
                            **manifest["generation"]): indices for prompt, indices in prompt_indices.items()}
        try:
            for job in as_completed(jobs):
                generation = job.result()
                for i in jobs[job]:
                    candidates[i] = {**selected_references[i], "output": generation.text,
                                     "generator": generation_name}
                    generation_rows[i] = {"index": i, **asdict(generation)}
                    progress["completed"] += 1
                    progress["length_limited"] += int(generation.stop_reason == "length")
                write_json(active_dir / "progress.json", {**progress, "updated_at": now(),
                                                           "tinker": sampler.stats()})
        except BaseException:
            pool.shutdown(wait=True, cancel_futures=True)
            raise
    candidate_path = active_dir / "candidate_outputs.json"
    write_json(candidate_path, candidates)
    write_json(active_dir / "generations.json", generation_rows)
    if not args.smoke:
        report = preflight_reference_utility(config, candidate_path=candidate_path,
                                              reference_path=args.reference)
        write_json(active_dir / "reference_preflight.json", asdict(report))
    progress["status"] = "judging"
    def judge_progress(stats):
        write_json(active_dir / "progress.json", {**progress, "updated_at": now(),
                    "tinker": sampler.stats(), "openrouter": stats})
    judge = OpenRouterJudge(args.output_dir / "judge_cache", args.judge_credentials,
                            concurrency=args.judge_concurrency, progress=judge_progress)
    judge_progress(judge.stats)
    annotator = make_annotator(judge, config)
    annotations = annotator.annotate_head2head(outputs_1=selected_references,
                                               outputs_2=candidates, is_ordered=True)
    validate_annotations(annotations, candidates, selected_references, config.identity_keys)
    write_json(active_dir / "annotations.json", annotations)
    progress.update(status="scoring", judged=count)
    judge_progress(judge.stats)
    # A smoke only tests raw preferences; never fit Alpaca's 805-row LC model
    # to a two-example subset and report it as the benchmark metric.
    metrics = score_annotations("sep" if args.smoke else suite, annotations,
                                metric_asset=args.metric_asset, metric_asset_sha256=asset_hash)
    import math
    primary_metric = "win_rate" if args.smoke else config.metric.primary
    if not math.isfinite(float(metrics[primary_metric])):
        raise ValueError("Primary utility metric is not finite; refusing a completed result")
    result = {"status": "complete", "benchmark": args.benchmark,
              "model_variant": args.model_variant, "n": count, "smoke": bool(args.smoke),
              "metrics": metrics, "manifest_sha256": digest(manifest),
              "candidate_sha256": sha256_file(candidate_path),
              "reference": reference_info, "judge_model": JUDGE_MODEL,
              "completed_at": now(), "length_limited": progress["length_limited"],
              "tinker_this_process": sampler.stats(), "openrouter_this_process": judge.stats}
    write_json(active_dir / "results.json", result)
    progress.update(status="complete", completed_at=now())
    judge_progress(judge.stats)
    print(json.dumps({"status": "complete", "metrics": metrics}, default=float), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # External errors may contain headers or private service details. The
        # supervisor records a nonzero exit; never dump unsanitized exceptions.
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}), flush=True)
        raise SystemExit(1) from None
