#!/usr/bin/env python3
"""Frozen-prompt GPQA replay: prepare/score locally, generate on a Lambda GPU.

No Tinker SDK or API is used. Only prompts, not reference answers or previous
model outputs, are sent to the GPU worker. Existing evaluation files are read-only.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import tempfile
import time

MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TASK = "gpqa_diamond_cot_zeroshot"
PARAMS = dict(max_tokens=16384, stop=["</s>", "<|im_end|>", "<|endoftext|>"],
              temperature=0.0, top_p=1.0, top_k=-1, seed=42)
TASK_PARAMS = dict(until=["</s>"], do_sample=False, temperature=0.0, max_gen_toks=16384)
SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        try:
            json.dump(value, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def read_reference(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    result = json.loads((directory / "results.json").read_text())
    worker = json.loads((directory / "worker.json").read_text())
    if worker["status"] != "complete" or worker["exit_code"] != 0:
        raise ValueError("Reference evaluation must have completed successfully")
    if (manifest["checkpoint"] != f"base:{MODEL}" or manifest["model"] != MODEL
            or manifest["tokenizer_revision"] != REVISION or manifest["enable_thinking"] is not False
            or manifest["lm_eval_version"] != "0.4.11" or manifest["default_max_gen_toks"] != 16384
            or any(manifest[k] != 42 for k in ("random_seed", "numpy_seed", "torch_seed", "fewshot_seed"))):
        raise ValueError("Unexpected reference protocol")
    task = manifest["tasks"][TASK]
    if task["indices"] != list(range(198)) or task["generation_kwargs"] != TASK_PARAMS or task["num_fewshot"] != 0:
        raise ValueError("Unexpected reference task selection/settings")
    rows = sorted((r for r in result["samples"][TASK] if r["filter"] == "flexible-extract"), key=lambda r: r["doc_id"])
    if [r["doc_id"] for r in rows] != list(range(198)):
        raise ValueError("Reference must contain each GPQA question exactly once per filter")
    if digest([r["doc"] for r in rows]) != task["selected_documents_sha256"]:
        raise ValueError("Reference question identity mismatch")
    return manifest, result, rows


def check_inputs(bundle):
    if bundle.get("schema") != "secpo-lambda-gpqa-v1" or bundle["model"] != MODEL or bundle["revision"] != REVISION:
        raise ValueError("Wrong model or input schema")
    if bundle["sampling_params"] != PARAMS or bundle["enable_thinking"] is not False:
        raise ValueError("Sampling protocol changed")
    requests = bundle["requests"]
    if [r["doc_id"] for r in requests] != list(range(198)) or digest(requests) != bundle["requests_sha256"]:
        raise ValueError("Missing, reordered, duplicated, or changed questions")
    for row in requests:
        if not row["prompt"].endswith(SUFFIX) or digest(row["prompt"]) != row["prompt_sha256"]:
            raise ValueError("Prompt changed or thinking mode enabled")
        ids = row["prompt_token_ids"]
        if not ids or any(type(i) is not int or i < 0 for i in ids) or len(ids) + PARAMS["max_tokens"] > 32768:
            raise ValueError("Invalid prompt tokens or context budget")
    return requests


def prepare(args):
    from transformers import AutoTokenizer
    if version("transformers") != "4.57.1":
        raise ValueError("Prepare with the original Transformers 4.57.1 environment")
    manifest, result, rows = read_reference(args.reference)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    requests = []
    for row in rows:
        if len(row["arguments"]) != 1 or row["arguments"][0][1] != TASK_PARAMS:
            raise ValueError("Saved request settings differ")
        prompt = row["arguments"][0][0]
        key = digest(dict(checkpoint=f"base:{MODEL}", tokenizer_revision=REVISION,
                          render_version="qwen38-secpo-eval-thinking-off-v1", prompt=prompt, params=PARAMS))
        cached = json.loads((args.reference / "completion_cache" / f"{key}.json").read_text())
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if (cached["request_hash"] != key or cached["generation"]["text"] != row["resps"][0][0]
                or cached["generation"]["input_tokens"] != len(ids)):
            raise ValueError("Saved prompt/response does not match original Tinker cache")
        requests.append(dict(doc_id=row["doc_id"], prompt=prompt, prompt_sha256=digest(prompt),
                             prompt_token_ids=ids, tinker_request_hash=key))
    bundle = dict(schema="secpo-lambda-gpqa-v1", model=MODEL, revision=REVISION,
                  enable_thinking=False, sampling_params=PARAMS, requests=requests,
                  requests_sha256=digest(requests), reference_manifest=manifest,
                  reference_manifest_sha256=digest(manifest), reference_results_sha256=file_hash(args.reference / "results.json"),
                  tokenizer_files_sha256={p.name:file_hash(p) for p in Path(args.tokenizer).iterdir() if p.is_file()},
                  prepared_with=dict(transformers=version("transformers"), tokenizers=version("tokenizers")))
    check_inputs(bundle)
    if args.output.exists() and json.loads(args.output.read_text()) != bundle:
        raise ValueError("Existing input bundle differs")
    write_json(args.output, bundle)
    print(json.dumps(dict(prepared=198, max_prompt_tokens=max(len(r["prompt_token_ids"]) for r in requests),
                          inputs_sha256=file_hash(args.output), output=str(args.output))))


def generate(args):
    import fcntl
    import signal
    import platform
    import importlib.metadata as metadata
    bundle = json.loads(args.inputs.read_text())
    requests = check_inputs(bundle)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "worker.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    if version("vllm") != "0.29.0" or version("transformers") != "5.10.4":
        raise ValueError("Use the pinned Lambda inference environment")
    runtime = dict(model=MODEL, revision=REVISION, tokenizer_revision=REVISION, dtype="bfloat16",
                   tensor_parallel_size=1, max_model_len=32768, gpu_memory_utilization=0.90,
                   max_num_seqs=args.batch_size, max_num_batched_tokens=4096, enforce_eager=True,
                   enable_prefix_caching=False, generation_config="vllm", seed=42,
                   language_model_only=True, trust_remote_code=False)
    identity = dict(inputs_sha256=file_hash(args.inputs), runtime=runtime,
                    versions={n:version(n) for n in ("vllm", "transformers", "torch", "tokenizers")},
                    source_sha256=file_hash(__file__), sampling_params=PARAMS)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != identity:
        raise ValueError("Generation identity changed; use a fresh output directory")
    write_json(manifest_path, identity)
    write_json(output / "environment.json", dict(python=sys.version, platform=platform.platform(),
               packages=sorted((d.metadata["Name"],d.version) for d in metadata.distributions())))
    started = now()
    progress = dict(status="validating_tokenizer", total=198, completed=0, length_limited=0,
                    started_at=started, pid=os.getpid(), command=sys.argv)
    def save_progress():
        write_json(output / "progress.json", {**progress, "updated_at":now()})
    def interrupted(signum, frame):
        progress.update(status="paused", signal=signum)
        save_progress()
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    save_progress()
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
        for row in requests:
            if tokenizer.encode(row["prompt"], add_special_tokens=False) != row["prompt_token_ids"]:
                raise ValueError(f"Tokenizer IDs differ for doc_id={row['doc_id']}")
        cache_dir = output / "completion_cache"
        cache_dir.mkdir(exist_ok=True)
        records = {}
        pending = []
        for row in requests:
            path = cache_dir / f"{row['doc_id']:03d}.json"
            if path.exists():
                record = json.loads(path.read_text())
                if record["request_sha256"] != digest(row) or record["manifest_sha256"] != digest(identity):
                    raise ValueError("Cached completion identity differs")
                records[row["doc_id"]] = record
            else:
                pending.append(row)
        progress.update(status="loading_model", completed=len(records),
                        length_limited=sum(r["finish_reason"] == "length" for r in records.values()))
        save_progress()
        if pending:
            llm = LLM(**runtime)
            # Explicit settings override HF generation defaults. Passing token IDs
            # avoids re-rendering or prepending BOS. No reasoning parser is used.
            sampling = SamplingParams(**PARAMS, n=1, presence_penalty=0.0, frequency_penalty=0.0,
                                      repetition_penalty=1.0, min_p=0.0, skip_special_tokens=False,
                                      spaces_between_special_tokens=False)
            # Start with two full-budget smoke questions; retain them in the final set.
            batches = [pending[:2]] + [pending[i:i+args.batch_size] for i in range(2,len(pending),args.batch_size)]
            for ordinal, batch in enumerate(batches):
                progress.update(status="smoke_generating" if ordinal == 0 else "generating")
                save_progress()
                begin = time.monotonic()
                outputs = llm.generate([dict(prompt_token_ids=r["prompt_token_ids"]) for r in batch],
                                       sampling, use_tqdm=False)
                if len(outputs) != len(batch):
                    raise RuntimeError("Missing generation outputs")
                for row, generated in zip(batch,outputs):
                    if list(generated.prompt_token_ids) != row["prompt_token_ids"] or len(generated.outputs) != 1:
                        raise RuntimeError("Generation request ordering mismatch")
                    completion = generated.outputs[0]
                    ids = list(completion.token_ids)
                    if not ids or len(ids) > 16384 or completion.finish_reason not in {"stop", "length"}:
                        raise RuntimeError("Invalid completion or finish reason")
                    record = dict(doc_id=row["doc_id"], request_sha256=digest(row), manifest_sha256=digest(identity),
                                  output_token_ids=ids, output_tokens=len(ids), input_tokens=len(row["prompt_token_ids"]),
                                  text=completion.text, finish_reason=completion.finish_reason,
                                  stop_reason=completion.stop_reason, batch_elapsed_seconds=time.monotonic()-begin,
                                  completed_at=now())
                    write_json(cache_dir / f"{row['doc_id']:03d}.json", record)
                    records[row["doc_id"]] = record
                progress.update(completed=len(records),
                                length_limited=sum(r["finish_reason"] == "length" for r in records.values()))
                save_progress()
                print(json.dumps(progress),flush=True)
                if ordinal == 0:
                    write_json(output / "smoke.json", dict(status="complete", doc_ids=[r["doc_id"] for r in batch],
                               manifest_sha256=digest(identity), completed_at=now()))
        write_json(output / "generations.json", [records[i] for i in range(198)])
        progress.update(status="generation_complete", completed_at=now(), scoring="pending_local_lm_eval")
        save_progress()
    except Exception as error:
        progress.update(status="failed", error_type=type(error).__name__, error=str(error))
        save_progress()
        raise


def score_responses(result, rows, texts):
    from types import SimpleNamespace
    from lm_eval.filters import build_filter_ensemble
    from lm_eval.api.metrics import exact_match_fn
    instances = [SimpleNamespace(doc=r["doc"], resps=[text], filtered_resps={}) for r,text in zip(rows,texts)]
    scored = {}
    for spec in result["configs"][TASK]["filter_list"]:
        components = [(s["function"], {k:v for k,v in s.items() if k != "function"}) for s in spec["filter"]]
        build_filter_ensemble(spec["name"], components).apply(instances)
        scored[spec["name"]] = [dict(extracted=inst.filtered_resps[spec["name"]],
            correct=float(exact_match_fn(references=[row["target"]], predictions=[inst.filtered_resps[spec["name"]]],
                                         ignore_case=True, ignore_punctuation=True)["exact_match"]))
            for inst,row in zip(instances,rows)]
    return scored


def saved_extraction(row):
    value = row["filtered_resps"]
    # lm-eval's saved records wrap the single extracted answer in a list.
    if isinstance(value, list):
        if len(value) != 1 or not isinstance(value[0], str):
            raise ValueError("Unexpected saved extraction shape")
        return value[0]
    if not isinstance(value, str):
        raise ValueError("Unexpected saved extraction type")
    return value


def score(args):
    from transformers import AutoTokenizer
    if version("lm-eval") != "0.4.11" or version("transformers") != "4.57.1":
        raise ValueError("Score using the original lm-eval/Transformers environment")
    manifest, reference, rows = read_reference(args.reference)
    bundle = json.loads(args.inputs.read_text())
    requests = check_inputs(bundle)
    if digest(manifest) != bundle["reference_manifest_sha256"] or file_hash(args.reference / "results.json") != bundle["reference_results_sha256"]:
        raise ValueError("Reference artifacts changed")
    prior_scores = score_responses(reference, rows, [r["resps"][0][0] for r in rows])
    for name, scores in prior_scores.items():
        old = sorted((r for r in reference["samples"][TASK] if r["filter"] == name), key=lambda r:r["doc_id"])
        if len(old) != 198 or any(s["correct"] != r["exact_match"] or s["extracted"] != saved_extraction(r) for s,r in zip(scores,old)):
            raise ValueError("Scoring replay does not reproduce the original per-question results")
    if args.self_test:
        print(json.dumps(dict(status="scoring_self_test_passed", correct=sum(r["correct"] for r in prior_scores["flexible-extract"]), n=198)))
        return
    generated = json.loads((args.generations / "generations.json").read_text())
    identity = json.loads((args.generations / "manifest.json").read_text())
    if identity["inputs_sha256"] != file_hash(args.inputs) or [r["doc_id"] for r in generated] != list(range(198)):
        raise ValueError("Generation input identity/coverage differs")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    texts = []
    for row, record in zip(requests,generated):
        if record["request_sha256"] != digest(row) or record["manifest_sha256"] != digest(identity):
            raise ValueError("Output identity mismatch")
        text = tokenizer.decode(record["output_token_ids"], skip_special_tokens=False)
        ends = [text.find(s) for s in PARAMS["stop"] if s in text]
        texts.append(text[:min(ends)] if ends else text)
    scores = score_responses(reference, rows, texts)
    questions = [dict(doc_id=r["doc_id"], doc=r["doc"], target=r["target"], prompt=q["prompt"],
                      response=text, output_tokens=g["output_tokens"], finish_reason=g["finish_reason"],
                      scores={name:values[i] for name,values in scores.items()},
                      tinker_correct=prior_scores["flexible-extract"][i]["correct"],
                      tinker_response=r["resps"][0][0])
                 for i,(r,q,g,text) in enumerate(zip(rows,requests,generated,texts))]
    result = dict(status="complete", model=MODEL, benchmark=TASK, n=198, max_output_tokens=16384,
                  results={f"exact_match,{name}":sum(r["correct"] for r in values)/198 for name,values in scores.items()},
                  tinker_reference_accuracy=reference["results"][TASK]["exact_match,flexible-extract"],
                  length_limited=sum(r["finish_reason"] == "length" for r in generated),
                  input_sha256=file_hash(args.inputs), generation_manifest_sha256=digest(identity),
                  scoring_source_sha256=file_hash(__file__), completed_at=now(), samples=questions)
    write_json(args.output / "results.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "samples"},indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action",required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--reference",type=Path,required=True)
    prep.add_argument("--tokenizer",type=Path,required=True)
    prep.add_argument("--output",type=Path,required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--inputs",type=Path,required=True)
    gen.add_argument("--output",type=Path,required=True)
    gen.add_argument("--batch-size",type=int,choices=range(1,9),default=4)
    scorer = sub.add_parser("score")
    scorer.add_argument("--reference",type=Path,required=True)
    scorer.add_argument("--inputs",type=Path,required=True)
    scorer.add_argument("--tokenizer",type=Path,required=True)
    scorer.add_argument("--generations",type=Path)
    scorer.add_argument("--output",type=Path)
    scorer.add_argument("--self-test",action="store_true")
    args = p.parse_args()
    if args.action == "score" and not args.self_test and (args.generations is None or args.output is None):
        p.error("Scoring requires --generations and --output")
    globals()[args.action](args)


if __name__ == "__main__":
    main()
