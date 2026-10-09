from dataclasses import replace
import json

import pytest

from secpo.evaluation.agentdojo_models import AgentDojoProviderError
from secpo.evaluation.agentdojo_openrouter import (
    BASE_URL, ENDPOINT, MODELS, RecordedTransport, load_key, make_backend,
)
from secpo.attacks.agentdojo_genetic import AgentDojoGeneticConfig
from secpo.evaluation.agentdojo_qwen36 import AgentDojoEpisode, paper_genetic_tasks, run_genetic
from scripts.probe_agentdojo_openrouter import exercise, main


def response(payload, text):
    return {"model": payload["model"],
            "choices": [{"finish_reason": "stop", "message": {"content": text}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "cost": .001}}


def test_load_key_is_data_not_executable(tmp_path):
    path = tmp_path / "credentials.env"
    path.write_text("export OPENROUTER_API_KEY='fixture-key'\nOPENROUTER_MODEL=unused\n")
    assert load_key(path) == "fixture-key"
    path.write_text("OPENROUTER_API_KEY='$(run-something)'\n")
    with pytest.raises(ValueError):
        load_key(path)


def test_key_rejects_non_openrouter_host(tmp_path):
    path = tmp_path / "credentials.env"
    path.write_text("OPENROUTER_API_KEY=fixture\nOPENROUTER_BASE_URL=https://example.invalid/v1\n")
    with pytest.raises(ValueError, match="only be sent to OpenRouter"):
        load_key(path)


@pytest.mark.parametrize("model", MODELS)
def test_existing_backend_and_feedback_roundtrip(tmp_path, monkeypatch, model):
    seen = []
    def post(url, payload, headers, timeout):
        seen.append(payload)
        assert url == ENDPOINT
        assert headers["Authorization"] == "Bearer fixture-secret"
        if len(seen) == 1:
            assert "reasoning_effort" not in payload
            return response(payload, json.dumps({"score": 10, "explanation": "EXPLANATION_MARKER",
                                                 "suggestion": "SUGGESTION_MARKER"}))
        prompt = payload["messages"][1]["content"]
        assert '"score": 9.0' in prompt  # critic cannot declare success
        assert "EXPLANATION_MARKER" in prompt and "SUGGESTION_MARKER" in prompt
        assert payload["reasoning_effort"] == "high"
        return response(payload, '{"improvement":"fixture change","trigger":"a new counter description"}')
    transport = RecordedTransport(tmp_path / "receipts", post=post)
    result = exercise(model, "fixture-secret", transport)
    assert result["status"] == "passed"
    assert result["critic_feedback"]["score"] == 9
    assert transport.summary()["reported_cost_usd"] == .002
    for path in transport.directory.glob("*.json"):
        assert "fixture-secret" not in path.read_text()
        assert "Authorization" not in path.read_text()


def test_call_budget_is_enforced_before_network(tmp_path):
    transport = RecordedTransport(tmp_path, max_requests=1,
        post=lambda url, payload, headers, timeout: response(payload, "ok"))
    payload = {"model": MODELS[0], "max_tokens": 16}
    transport(ENDPOINT, payload, {}, 1)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        transport(ENDPOINT, payload, {}, 1)
    assert transport.requests == 1


@pytest.mark.parametrize("url,model,tokens", [
    ("https://example.invalid/v1/chat/completions", MODELS[0], 16),
    (ENDPOINT, "unapproved/model", 16), (ENDPOINT, MODELS[0], 4097),
])
def test_fail_closed_before_network(tmp_path, url, model, tokens):
    transport = RecordedTransport(tmp_path, post=lambda *a: pytest.fail("No call allowed"))
    with pytest.raises(ValueError):
        transport(url, {"model": model, "max_tokens": tokens}, {}, 1)
    assert transport.requests == 0


def test_truncated_response_is_saved_but_rejected(tmp_path):
    def post(url, payload, headers, timeout):
        result = response(payload, "partial")
        result["choices"][0]["finish_reason"] = "length"
        return result
    transport = RecordedTransport(tmp_path, post=post)
    with pytest.raises(ValueError, match="Incomplete"):
        transport(ENDPOINT, {"model": MODELS[0], "max_tokens": 16}, {}, 1)
    assert transport.summary()["reported_cost_usd"] == .001
    assert json.loads((tmp_path / "001.json").read_text())["status"] == "failed"


def test_default_plan_does_not_read_credentials(monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert main([]) == 0
    assert json.loads(capsys.readouterr().out)["maximum_requests"] == 4


def test_exact_success_finishes_seed_batch_then_replays():
    calls = []
    def episode(task, injection):
        calls.append(injection)
        return AgentDojoEpisode("trace", injection == "exact", True, 9.0)
    result = run_genetic(task=paper_genetic_tasks(goal="fixture")[0],
        proposer=lambda request: pytest.fail("No mutation after exact success"),
        episode=episode, initial_injections=("high-heuristic", "exact"),
        config=AgentDojoGeneticConfig(steps=2, candidates_per_step=2, query_budget=4))
    assert result["pass_at_n"] is True
    assert result["pass_at_n_denominator"] == 2
    assert result["final_attack_success"] is True
    assert result["final_replay_queries"] == 1
    assert calls == ["high-heuristic", "exact", "exact"]
