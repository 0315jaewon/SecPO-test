"""Explicit trained-model variants without changing the original default model."""

import json
from pathlib import Path

from secpo.evaluation.reference_utility import sha256_file
from . import evaluation_common as common

ROOT = common.ROOT
MODEL_VARIANTS = ("secpo", "base", "secpo_lr6e5")
CHECKPOINT_BENCHMARKS = ("gpqa_diamond", "mmlu_pro", "minerva_math", "sep_utility", "alpacaeval2")
COMPARISON_SOURCES = (
    "src/secpo/tinker/evaluation_variants.py", "src/secpo/tinker/evaluation_common.py",
    "src/secpo/tinker/lm_eval_backend.py", "src/secpo/tinker/reference_eval.py",
    "scripts/evaluate_tinker_lm.py", "scripts/evaluate_tinker_reference.py",
)


def evaluation_model_id(variant):
    if variant in {"base", "secpo"}:
        return common.evaluation_model_id(variant)
    if variant != "secpo_lr6e5":
        raise ValueError(f"Unknown model variant: {variant}")
    result = json.loads((ROOT / "outputs/qwen38_27b_tinker_lr6e5/result.json").read_text())
    path = result["final_sampler_path"]
    if (result.get("steps_completed") != 297 or not isinstance(path, str)
            or not path.startswith("tinker://") or "/sampler_weights/" not in path):
        raise ValueError("The lr6e5 variant requires its completed final sampler")
    if path == common.final_checkpoint():
        raise ValueError("The lr6e5 variant cannot use the previous SecPO checkpoint")
    return path


def evaluation_output_root(variant, max_output_tokens=None):
    if variant in {"base", "secpo"}:
        return common.evaluation_output_root(variant, max_output_tokens)
    if variant != "secpo_lr6e5":
        raise ValueError(f"Unknown model variant: {variant}")
    original = common.evaluation_output_root("secpo", max_output_tokens)
    return original.with_name(original.name.replace("qwen38_27b_tinker_utility",
                                                    "qwen38_27b_tinker_lr6e5_utility"))


def sampler_model_kwargs(variant):
    if variant == "base":
        return {"base_model": common.MODEL}
    return {"checkpoint": evaluation_model_id(variant)}


def validate_checkpoint_comparison(manifest, variant, output_dir):
    """Require the same existing cap-specific protocol, changing only weights.

    Reference manifests also hash their entry-point script. Its reviewed variant
    routing change is recorded separately; judge, parser, renderer and metric
    source hashes must remain identical. A sidecar pins all current routing code
    so a later resume cannot silently change it.
    """
    if variant != "secpo_lr6e5" or manifest["benchmark"] not in CHECKPOINT_BENCHMARKS:
        raise ValueError("Unsupported checkpoint comparison")
    cap = manifest["default_max_gen_toks"]
    expected_output = evaluation_output_root(variant, cap) / manifest["benchmark"]
    if Path(output_dir).resolve() != expected_output.resolve():
        raise ValueError("Use the checkpoint variant's separate output directory")
    prior_path = common.evaluation_output_root("secpo", cap) / manifest["benchmark"] / "manifest.json"
    prior = json.loads(prior_path.read_text())
    if prior["checkpoint"] != common.final_checkpoint():
        raise ValueError("Prior protocol does not belong to the original SecPO model")
    if manifest["checkpoint"] != evaluation_model_id(variant):
        raise ValueError("Manifest does not select the requested checkpoint")
    settings = lambda value: {k: v for k, v in value.items()
                              if k not in {"checkpoint", "implementation_sha256"}}
    if settings(prior) != settings(manifest):
        raise ValueError("Checkpoint comparison changes evaluation protocol settings")
    old_sources, new_sources = prior.get("implementation_sha256", {}), manifest.get("implementation_sha256", {})
    if set(old_sources) != set(new_sources):
        raise ValueError("Implementation source inventory changed")
    changes = sorted(k for k in old_sources if old_sources[k] != new_sources[k])
    if set(changes) - {"scripts/evaluate_tinker_reference.py"}:
        raise ValueError("Judge, scoring, or sampling implementation changed")
    record = {
        "baseline_manifest": str(prior_path), "baseline_manifest_sha256": common.digest(prior),
        "current_manifest_sha256": common.digest(manifest), "model_variant": variant,
        "protocol_changes": ["checkpoint"], "changed_source_hashes": changes,
        "source_change_reason": "Explicit checkpoint selection and variant output labeling only",
        "implementation_sha256": {name: sha256_file(ROOT / name) for name in COMPARISON_SOURCES},
    }
    sidecar = Path(output_dir) / "checkpoint_comparison.json"
    if sidecar.exists() and json.loads(sidecar.read_text()) != record:
        raise ValueError("Checkpoint comparison provenance changed; use a new run")
    common.write_json(sidecar, record)
