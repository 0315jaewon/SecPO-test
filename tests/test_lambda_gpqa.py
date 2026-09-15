import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("lambda_gpqa", Path(__file__).parents[1] / "scripts/lambda_gpqa.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def inputs():
    rows = [dict(doc_id=i, prompt=f"question {i}" + runner.SUFFIX,
                 prompt_sha256=runner.digest(f"question {i}" + runner.SUFFIX),
                 prompt_token_ids=[1, 2, i]) for i in range(198)]
    return dict(schema="secpo-lambda-gpqa-v1", model=runner.MODEL, revision=runner.REVISION,
                enable_thinking=False, sampling_params=copy.deepcopy(runner.PARAMS),
                requests=rows, requests_sha256=runner.digest(rows))


def test_frozen_inputs_have_all_questions(inputs):
    assert len(runner.check_inputs(inputs)) == 198


@pytest.mark.parametrize("field,value", [("model", "other"), ("revision", "main"), ("enable_thinking", True)])
def test_model_revision_thinking_are_locked(inputs, field, value):
    inputs[field] = value
    with pytest.raises(ValueError):
        runner.check_inputs(inputs)


@pytest.mark.parametrize("field,value", [("max_tokens", 8192), ("temperature", 1.0), ("seed", 0), ("top_k", 20)])
def test_generation_settings_are_locked(inputs, field, value):
    inputs["sampling_params"][field] = value
    with pytest.raises(ValueError):
        runner.check_inputs(inputs)


def test_reordered_questions_rejected_even_with_new_hash(inputs):
    inputs["requests"].reverse()
    inputs["requests_sha256"] = runner.digest(inputs["requests"])
    with pytest.raises(ValueError):
        runner.check_inputs(inputs)


def test_mutated_prompt_or_tokens_rejected(inputs):
    inputs["requests"][0]["prompt_token_ids"].append(99)
    with pytest.raises(ValueError):
        runner.check_inputs(inputs)


def test_thinking_prefix_and_context_checked(inputs):
    row = inputs["requests"][0]
    row["prompt"] = "a different template"
    row["prompt_sha256"] = runner.digest(row["prompt"])
    inputs["requests_sha256"] = runner.digest(inputs["requests"])
    with pytest.raises(ValueError):
        runner.check_inputs(inputs)


def test_atomic_json_roundtrip(tmp_path):
    path = tmp_path / "nested/data.json"
    runner.write_json(path, {"text": "π"})
    assert runner.json.loads(path.read_text()) == {"text": "π"}
    runner.write_json(path, {"text": "second"})
    assert runner.json.loads(path.read_text()) == {"text": "second"}


def test_original_lm_eval_filter_selects_final_answer():
    config = dict(configs={runner.TASK:dict(filter_list=[dict(name="flexible-extract", filter=[
        dict(function="multi_choice_regex", regex_pattern=r"(\([A-Z]\))", group_select=-1,
             ignore_case=True, ignore_punctuation=True), dict(function="take_first")])])})
    rows = [dict(doc={"choices":["first","second","third","fourth"]}, target="(B)")]
    scores = runner.score_responses(config, rows, ["Initially (A), but the correct answer is (B)."])
    assert scores["flexible-extract"] == [dict(extracted="(B)", correct=1.0)]


def test_missing_answer_is_incorrect():
    config = dict(configs={runner.TASK:dict(filter_list=[dict(name="strict-match", filter=[
        dict(function="regex", regex_pattern="(?<=The answer is )(.*)(?=.)"), dict(function="take_first")])])})
    scores = runner.score_responses(config, [dict(doc={}, target="(B)")], ["I do not know."])
    assert scores["strict-match"][0]["correct"] == 0.0


def test_saved_extractions_accept_singleton_list_or_string():
    assert runner.saved_extraction({"filtered_resps":["(B)"]}) == "(B)"
    assert runner.saved_extraction({"filtered_resps":"(B)"}) == "(B)"
    with pytest.raises(ValueError):
        runner.saved_extraction({"filtered_resps":["(B)","(C)"]})
