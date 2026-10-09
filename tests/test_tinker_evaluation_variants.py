"""Checkpoint selection and strict protocol parity without remote API calls."""

from copy import deepcopy
import json

import pytest

from secpo.tinker import evaluation_common as common
from secpo.tinker import evaluation_variants as variants


@pytest.fixture
def variant_files(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(variants, "ROOT", tmp_path)
    common.write_json(tmp_path / "outputs/qwen38_27b_tinker/result.json", {
        "final_sampler_path": "tinker://old/sampler_weights/final", "steps_completed": 297,
    })
    common.write_json(tmp_path / "outputs/qwen38_27b_tinker_lr6e5/result.json", {
        "final_sampler_path": "tinker://new/sampler_weights/final", "steps_completed": 297,
    })
    for name in variants.COMPARISON_SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("unchanged test source")
    return tmp_path


def test_variant_selects_new_weights_without_changing_old_defaults(variant_files):
    assert common.final_checkpoint() == "tinker://old/sampler_weights/final"
    assert variants.evaluation_model_id("secpo") == common.final_checkpoint()
    assert variants.sampler_model_kwargs("secpo_lr6e5") == {"checkpoint": "tinker://new/sampler_weights/final"}
    assert variants.sampler_model_kwargs("base") == {"base_model": common.MODEL}
    assert len({variants.evaluation_output_root(v, 16384) for v in variants.MODEL_VARIANTS}) == 3
    assert variants.evaluation_output_root("secpo", 16384) == common.evaluation_output_root("secpo", 16384)
    for invalid in (0, -1, 65536, True):
        with pytest.raises(ValueError):
            variants.evaluation_output_root("secpo_lr6e5", invalid)
    with pytest.raises(ValueError):
        variants.evaluation_model_id("typo")


def test_variant_rejects_incomplete_or_original_checkpoint(variant_files):
    path = variant_files / "outputs/qwen38_27b_tinker_lr6e5/result.json"
    for value in (
        {"final_sampler_path": "tinker://new/sampler_weights/final", "steps_completed": 2},
        {"final_sampler_path": common.final_checkpoint(), "steps_completed": 297},
        {"final_sampler_path": "tinker://new/weights/final", "steps_completed": 297},
    ):
        common.write_json(path, value)
        with pytest.raises(ValueError):
            variants.evaluation_model_id("secpo_lr6e5")


@pytest.mark.parametrize("benchmark", ["mmlu_pro", "sep_utility"])
def test_checkpoint_comparison_only_allows_weights_and_routing_source(variant_files, benchmark):
    prior = {"benchmark": benchmark, "checkpoint": common.final_checkpoint(),
             "default_max_gen_toks": 16384, "enable_thinking": False,
             "tasks": {"task": {"indices": [3, 5], "max_gen_toks": 16384}},
             "judge": {"model": "openai/gpt-4o-mini"}, "reference": {"sha256": "frozen"}}
    if benchmark == "sep_utility":
        prior["implementation_sha256"] = {"scripts/evaluate_tinker_reference.py": "old-entry",
                                          "src/secpo/tinker/reference_eval.py": "same-judge"}
    common.write_json(common.evaluation_output_root("secpo", 16384) / benchmark / "manifest.json", prior)
    current = deepcopy(prior)
    current["checkpoint"] = variants.evaluation_model_id("secpo_lr6e5")
    if benchmark == "sep_utility":
        current["implementation_sha256"]["scripts/evaluate_tinker_reference.py"] = "new-entry"
    output = variants.evaluation_output_root("secpo_lr6e5", 16384) / benchmark
    variants.validate_checkpoint_comparison(current, "secpo_lr6e5", output)
    variants.validate_checkpoint_comparison(current, "secpo_lr6e5", output)  # Safe identical resume.
    saved = json.loads((output / "checkpoint_comparison.json").read_text())
    assert saved["protocol_changes"] == ["checkpoint"]
    for key, value in (("checkpoint", common.final_checkpoint()), ("enable_thinking", True),
                       ("default_max_gen_toks", 8192), ("judge", {"model": "other"}),
                       ("reference", {"sha256": "changed"}),
                       ("tasks", {"task": {"indices": [3, 6], "max_gen_toks": 16384}})):
        with pytest.raises((ValueError, FileNotFoundError)):
            variants.validate_checkpoint_comparison({**current, key: value}, "secpo_lr6e5", output)
    if benchmark == "sep_utility":
        bad = deepcopy(current)
        bad["implementation_sha256"]["src/secpo/tinker/reference_eval.py"] = "changed-judge"
        with pytest.raises(ValueError, match="implementation changed"):
            variants.validate_checkpoint_comparison(bad, "secpo_lr6e5", output)
    with pytest.raises(ValueError, match="separate output"):
        variants.validate_checkpoint_comparison(current, "secpo_lr6e5",
                                                 common.evaluation_output_root("secpo", 16384) / benchmark)
    (variant_files / variants.COMPARISON_SOURCES[0]).write_text("changed after launch")
    with pytest.raises(ValueError, match="provenance changed"):
        variants.validate_checkpoint_comparison(current, "secpo_lr6e5", output)
