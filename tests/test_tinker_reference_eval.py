"""Offline checks of the actual installed AlpacaEval pipeline and API bridge."""

from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

pytest.importorskip("alpaca_eval")

from secpo.tinker import reference_eval as bridge
from secpo.tinker.evaluation_common import ROOT, write_json
from secpo.evaluation.reference_utility import ReferenceUtilityError, load_reference_utility_config, sha256_file


def choice(p=0.8):
    return {"message": {"role": "assistant", "content": "m"}, "index": 0,
            "finish_reason": "length", "logprobs": {"content": [{"token": "m",
            "logprob": math.log(p), "top_logprobs": [
                {"token": "m", "logprob": math.log(p)},
                {"token": "M", "logprob": math.log(1-p)}]}]}}


@pytest.fixture
def credentials(tmp_path):
    path = tmp_path / "judge.env"
    path.write_text('OPENROUTER_API_KEY="fake-test-key"\nOPENROUTER_MODEL="openai/gpt-4o-mini"\n')
    return path


def test_credentials_are_literal_and_private(credentials, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert bridge.read_openrouter_credentials(credentials) == "fake-test-key"
    import os
    assert "OPENROUTER_API_KEY" not in os.environ
    for line in ('OPENROUTER_BASE_URL="https://wrong.example"',
                 'OPENROUTER_MODEL="another-model"', 'OPENROUTER_API_KEY="duplicate"'):
        credentials.write_text('OPENROUTER_API_KEY="private-value"\n' + line + '\n')
        with pytest.raises(ValueError) as exc:
            bridge.read_openrouter_credentials(credentials)
        assert "private-value" not in str(exc.value)


def test_clean_sep_never_includes_injection():
    row = {"instruction": "merged", "instruction_only": "Summarize", "input": "Clean data",
           "injection": "MALICIOUS_SENTINEL"}
    assert bridge.clean_messages("sep", row) == [
        {"role": "user", "content": "Summarize"}, {"role": "input", "content": "Clean data"}]
    assert len(bridge.clean_messages("sep", {**row, "input": ""})) == 1
    assert bridge.clean_messages("alpacaeval2", row) == [{"role": "user", "content": "merged"}]


def test_actual_tokenizer_uses_trust_roles_and_thinking_off():
    from transformers import AutoTokenizer
    from secpo.tinker.evaluation_common import MODEL, TOKENIZER_REVISION
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=TOKENIZER_REVISION, local_files_only=True)
    except OSError:
        pytest.skip("Pinned tokenizer not cached")
    row = {"instruction_only": "Summarize", "input": "Clean data", "injection": "MALICIOUS_SENTINEL"}
    prompt = bridge.render_clean_prompt(tokenizer, "sep", row)
    assert "<|im_start|>user\nSummarize" in prompt
    assert "<|im_start|>input\nClean data" in prompt
    assert "MALICIOUS_SENTINEL" not in prompt
    assert "<|im_start|>system" not in prompt
    assert prompt.endswith(bridge.QWEN38_NON_THINKING_SUFFIX)


def test_sep_alternate_is_explicit_and_checks_test_alignment(tmp_path, monkeypatch):
    config = load_reference_utility_config(ROOT / "configs/evaluate_sep_utility.json")
    config = replace(config, expected_rows=1)
    monkeypatch.setattr(bridge, "load_reference_utility_config", lambda path: config)
    row = {"instruction": "Summarize\n\nClean", "instruction_only": "Summarize", "input": "Clean",
           "output": "Reference", "injection": "attack", "witness": "witness",
           "generator": config.reference.generator}
    reference = tmp_path / "reference.json"
    raw = tmp_path / "test.json"
    write_json(reference, [row])
    write_json(raw, [{**row, "instruction": row["instruction_only"]}])
    monkeypatch.setattr(bridge, "SEP_TEST_SHA256", sha256_file(raw))
    with pytest.raises(ReferenceUtilityError):
        bridge.reference_inputs("sep", reference)
    args = dict(approved_sha256=sha256_file(reference), sep_test_path=raw)
    _, rows, info = bridge.reference_inputs("sep", reference, **args)
    assert rows == [row] and info["is_alternate_reference"]
    assert info["release_sha256"] == config.reference.sha256
    write_json(reference, [{**row, "witness": "different"}])
    with pytest.raises(ValueError, match="align"):
        bridge.reference_inputs("sep", reference, approved_sha256=sha256_file(reference), sep_test_path=raw)


def test_logprobs_require_valid_comparison():
    assert bridge.validate_judge_choice(choice()) == pytest.approx(1.2)
    for invalid in ({}, {"logprobs": None}, {"logprobs": {"content": []}},
                    {"logprobs": {"content": [{"top_logprobs": [{"token": "?", "logprob": -1}]}]}}):
        with pytest.raises(ValueError):
            bridge.validate_judge_choice(invalid)


def test_judge_caches_deduplicates_and_never_saves_key(credentials, tmp_path):
    calls = []
    def post(payload):
        calls.append(payload)
        return {"model": bridge.JUDGE_MODEL, "choices": [choice()], "provider": "OpenAI",
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "cost": 0.001}}
    judge = bridge.OpenRouterJudge(tmp_path / "cache", credentials, post=post)
    prompt = "<|im_start|>user\nCompare answers<|im_end|>"
    result = judge([prompt, prompt], **bridge.JUDGE_PARAMS)
    assert len(calls) == 1 and len(result["completions_all"]) == 2
    judge([prompt], **bridge.JUDGE_PARAMS)
    assert len(calls) == 1 and judge.stats["cached_requests"] == 1
    assert calls[0]["provider"] == {"require_parameters": True}
    assert calls[0]["max_tokens"] == 1 and calls[0]["logprobs"] is True
    for path in (tmp_path / "cache").glob("*.json"):
        assert "fake-test-key" not in path.read_text()
    with pytest.raises(ValueError):
        judge([prompt], **{**bridge.JUDGE_PARAMS, "max_tokens": 2})


@pytest.mark.parametrize("suite", ["alpacaeval2", "sep"])
def test_real_annotator_preserves_alignment_and_undoes_randomization(credentials, tmp_path, suite):
    config = load_reference_utility_config(ROOT / f"configs/evaluate_{suite}_utility.json")
    references = [{"instruction": f"Question {i}", "output": "bad-answer", "generator": "reference"}
                  for i in range(10)]
    if suite == "sep":
        references = [{**r, "instruction_only": r["instruction"], "input": "", "injection": "ATTACK_SENTINEL"}
                      for r in references]
    candidates = [{**r, "output": "good-answer", "generator": "candidate"} for r in references]
    seen = []
    def post(payload):
        content = "\n".join(m["content"] for m in payload["messages"])
        assert "ATTACK_SENTINEL" not in content
        good_first = content.index("good-answer") < content.index("bad-answer")
        seen.append(good_first)
        return {"model": bridge.JUDGE_MODEL, "choices": [choice(0.9 if good_first else 0.1)], "usage": {}}
    judge = bridge.OpenRouterJudge(tmp_path / "cache", credentials, post=post)
    annotator = bridge.make_annotator(judge, config)
    annotations = annotator.annotate_head2head(outputs_1=references, outputs_2=candidates, is_ordered=True)
    bridge.validate_annotations(annotations, candidates, references, config.identity_keys)
    assert set(seen) == {True, False}
    assert all(row["preference"] == pytest.approx(1.9) for row in annotations)
    assert bridge.score_annotations("sep", annotations)["win_rate"] == pytest.approx(90)
    with pytest.raises(ValueError, match="Missing"):
        bridge.validate_annotations(annotations[:-1], candidates, references, config.identity_keys)
    with pytest.raises(ValueError, match="805"):
        bridge.score_annotations("alpacaeval2", annotations)


def test_installed_evaluator_sources_are_available():
    source = bridge.evaluator_sources()
    assert source["version"] == "0.6.6"
    assert len(source["sha256"]) == 9


def test_full_length_control_fit_works_offline():
    asset = ROOT / "data/eval_references/alpacaeval_df_gamed.csv"
    if not asset.exists():
        pytest.skip("Optional metric asset not downloaded")
    from alpaca_eval.metrics import glm_winrate
    original = glm_winrate.hf_hub_download
    rows = [{"instruction": f"Question {i}", "output_1": "x" * (20 + i % 17),
             "output_2": "y" * (15 + i % 13), "generator_1": "reference", "generator_2": "candidate",
             "preference": 1.2 + (i % 7) / 10, "annotator": "test"} for i in range(805)]
    metrics = bridge.score_annotations("alpacaeval2", rows, metric_asset=asset,
                                      metric_asset_sha256=bridge.ALPACA_METRIC_SHA256)
    assert glm_winrate.hf_hub_download is original
    assert math.isfinite(metrics["length_controlled_winrate"])
    assert metrics["n_total"] == 805
