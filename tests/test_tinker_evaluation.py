from functools import partial
import json
from types import SimpleNamespace

import pytest

from secpo.tinker.evaluation_common import Generation, cut_at_stop, read_indices, write_json
from secpo.tinker.agent_eval import parse_tool_response, render_agent_prompt


def test_indices_validate_exact_selection(tmp_path):
    path = tmp_path / "indices.json"
    record = {"num_docs": 10, "num_sampled": 2, "indices": [7, 3]}
    write_json(path, {"task": "mmlu_pro", "leaf_tasks": {"category": record}})
    assert read_indices(path, "mmlu_pro")["category"]["indices"] == [7, 3]
    for indices in ([3, 3], [3, 10], [3, True], []):
        write_json(path, {"task": "mmlu_pro", "leaf_tasks": {"category": {**record, "indices": indices}}})
        with pytest.raises(ValueError):
            read_indices(path, "mmlu_pro")


def test_atomic_json_handles_evaluator_callables_and_preserves_on_error(tmp_path):
    path = tmp_path / "result.json"
    write_json(path, {"metric": partial(sorted, reverse=True)})
    assert json.loads(path.read_text())["metric"]["kwargs"] == {"reverse": True}
    before = path.read_text()
    with pytest.raises(TypeError):
        write_json(path, object())
    assert path.read_text() == before
    assert list(tmp_path.iterdir()) == [path]


def test_stops_choose_earliest_not_list_order():
    assert cut_at_stop("answer END later STOP", ["STOP", "", "END"]) == "answer "


TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {
    "type": "object", "properties": {"account": {"type": "string"}, "count": {"type": "integer"}}}}}]


def test_native_xml_calls_preserve_string_types():
    visible, calls = parse_tool_response("Checking.<tool_call><function=lookup>\n"
        "<parameter=account>00123</parameter>\n<parameter=count>2</parameter>"
        "</function></tool_call>", TOOLS)
    assert visible == "Checking."
    assert calls[0].function == "lookup"
    assert calls[0].args == {"account": "00123", "count": 2}
    with pytest.raises(ValueError):
        parse_tool_response("<tool_call><function=lookup>", TOOLS)


def test_tool_result_dates_use_iso_without_changing_normal_json():
    from datetime import datetime
    from secpo.tinker.agent_eval import tool_json_dumps
    assert tool_json_dumps({"amount": 4, "items": ["x"]}) == json.dumps({"amount": 4, "items": ["x"]})
    assert json.loads(tool_json_dumps({"created": datetime(2025, 1, 2, 3, 4)})) == {
        "created": "2025-01-02T03:04:00"}


def test_generation_order_survives_concurrency(tmp_path):
    pytest.importorskip("lm_eval")
    from secpo.tinker.lm_eval_backend import TinkerEvaluationLM
    class Sampler:
        tokenizer = SimpleNamespace(chat_template="template")
        def generate(self, prompt, **kwargs):
            assert kwargs["temperature"] == 0
            return Generation(prompt + " response", 1, 2, "stop", 0.1)
        def stats(self):
            return {}
    lm = TinkerEvaluationLM(Sampler(), tmp_path, concurrency=3)
    requests = [SimpleNamespace(args=(str(i), {"until": "STOP"})) for i in range(20)]
    assert lm.generate_until(requests) == [f"{i} response" for i in range(20)]


def test_pinned_native_template_roundtrips_tool_history():
    # Offline integration check: no HF/Tinker calls or credential access.
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer
    from secpo.tinker.evaluation_common import MODEL, TOKENIZER_REVISION
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=TOKENIZER_REVISION,
                                                  local_files_only=True)
    except OSError:
        pytest.skip("Pinned tokenizer not cached")
    messages = [{"role": "system", "content": "System instruction"},
                {"role": "user", "content": "User request"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"function": "lookup", "args": {"account": "00123", "count": 2}, "id": "call_1"}]},
                {"role": "tool", "content": "Untrusted result", "tool_call_id": "call_1"}]
    prompt = render_agent_prompt(tokenizer, messages, TOOLS)
    assert "<|im_start|>input\n" in prompt
    assert "<|im_start|>system\n" not in prompt
    assert "<parameter=account>\n00123\n</parameter>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_base_sampler_has_no_adapter_and_cannot_reuse_secpo_responses(tmp_path, monkeypatch):
    tinker = pytest.importorskip("tinker")
    from secpo.tinker.evaluation_common import CheckpointSampler, MODEL
    created = []
    sampled = []
    def future(value):
        return SimpleNamespace(result=lambda **kwargs: value)
    class Client:
        def sample(self, **kwargs):
            sampled.append(kwargs)
            return future(SimpleNamespace(sequences=[SimpleNamespace(tokens=[1], stop_reason="stop")]))
    class Service:
        def __init__(self, **kwargs):
            pass
        def create_sampling_client(self, **kwargs):
            created.append(kwargs)
            return Client()
        def create_rest_client(self):
            return SimpleNamespace(get_weights_info_by_tinker_path=lambda path: future(SimpleNamespace(base_model=MODEL)))
    monkeypatch.setattr(tinker, "ServiceClient", Service)
    tokenizer = SimpleNamespace(encode=lambda prompt, **kwargs: [1], decode=lambda tokens, **kwargs: "answer")
    base = CheckpointSampler(tmp_path, base_model=MODEL, tokenizer=tokenizer)
    secpo = CheckpointSampler(tmp_path, checkpoint="tinker://test/sampler", tokenizer=tokenizer)
    assert created == [{"base_model":MODEL}, {"model_path":"tinker://test/sampler"}]
    assert base.is_base_model and not secpo.is_base_model
    base.generate("same prompt", max_tokens=2048)
    secpo.generate("same prompt", max_tokens=2048)
    base.generate("same prompt", max_tokens=2048)
    assert len(sampled) == 2
    assert base.stats()["cached_requests"] == 1
    assert len(list(tmp_path.glob("*.json"))) == 2
    base.generate("same prompt", max_tokens=4096)
    assert len(sampled) == 3
    assert len(list(tmp_path.glob("*.json"))) == 3
    with pytest.raises(ValueError):
        CheckpointSampler(tmp_path, checkpoint="tinker://test/sampler", base_model=MODEL, tokenizer=tokenizer)


def test_base_manifest_must_match_existing_protocol(tmp_path, monkeypatch):
    import secpo.tinker.evaluation_common as common
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(common, "final_checkpoint", lambda: "tinker://final/sampler")
    prior = {"benchmark":"gpqa_diamond", "checkpoint":"tinker://final/sampler",
             "enable_thinking":False, "default_max_gen_toks":2048, "indices":[1, 2]}
    write_json(tmp_path / "outputs/qwen38_27b_tinker_utility/gpqa_diamond/manifest.json", prior)
    base = {**prior, "checkpoint":f"base:{common.MODEL}"}
    common.validate_base_comparison(base)
    for overrides in ({"default_max_gen_toks":8192}, {"enable_thinking":True}, {"indices":[1, 3]}):
        with pytest.raises(ValueError, match="protocol differs"):
            common.validate_base_comparison({**base, **overrides})


def test_output_cap_overrides_explicit_task_limit_and_reaches_sampler(tmp_path):
    pytest.importorskip("lm_eval")
    from secpo.tinker.lm_eval_backend import TinkerEvaluationLM, set_task_output_cap
    config = {"generation_kwargs":{"until":["Question:"], "max_gen_toks":2048, "temperature":0.0}}
    task = SimpleNamespace(get_config=config.get, set_config=lambda name, value: config.update({name:value}))
    set_task_output_cap(task, 4096)
    assert config["generation_kwargs"] == {"until":["Question:"], "max_gen_toks":4096, "temperature":0.0}
    class Sampler:
        tokenizer = SimpleNamespace(chat_template="template")
        def generate(self, prompt, **kwargs):
            assert kwargs["max_tokens"] == 4096
            assert kwargs["stops"] == ["Question:"]
            return Generation("answer", 1, 4096, "length", 0.1)
        def stats(self):
            return {}
    lm = TinkerEvaluationLM(Sampler(), tmp_path)
    assert lm.generate_until([SimpleNamespace(args=("prompt",config["generation_kwargs"]))]) == ["answer"]
    assert lm.length_limited == 1
    assert json.loads((tmp_path / "progress.json").read_text())["length_limited"] == 1
    for invalid in (0, -1, 65536, True):
        with pytest.raises(ValueError):
            set_task_output_cap(task, invalid)


def test_cap_variant_requires_same_samples_prompts_and_weights(tmp_path, monkeypatch):
    from copy import deepcopy
    import secpo.tinker.evaluation_common as common
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(common, "final_checkpoint", lambda: "tinker://final/sampler")
    prior = {"benchmark":"mmlu_pro", "checkpoint":"tinker://final/sampler",
             "enable_thinking":False, "default_max_gen_toks":2048,
             "tasks":{"mmlu_pro_biology":{"indices":[3, 5], "num_fewshot":5,
                 "generation_kwargs":{"until":["Question:"], "max_gen_toks":2048}}}}
    write_json(common.evaluation_output_root("secpo") / "mmlu_pro/manifest.json", prior)
    new = deepcopy(prior)
    new["default_max_gen_toks"] = 4096
    new["tasks"]["mmlu_pro_biology"]["generation_kwargs"]["max_gen_toks"] = 4096
    common.validate_output_cap_comparison(new, "secpo")
    for key, value in (("enable_thinking",True), ("checkpoint","tinker://other/sampler")):
        with pytest.raises(ValueError, match="beyond the token budget"):
            common.validate_output_cap_comparison({**new, key:value}, "secpo")
    bad = deepcopy(new)
    bad["tasks"]["mmlu_pro_biology"]["indices"] = [3, 6]
    with pytest.raises(ValueError, match="beyond the token budget"):
        common.validate_output_cap_comparison(bad, "secpo")
    assert common.evaluation_output_root("secpo",4096) != common.evaluation_output_root("secpo")
    assert common.evaluation_output_root("secpo",4096) != common.evaluation_output_root("base",4096)
