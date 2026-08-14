from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import evaluate_agentdyn_qwen36_27b as agentdyn_cli
from secpo.evaluation.agentdyn_qwen36 import (
    DEFAULT_ATTACKS,
    DEFAULT_SUITES,
    InvalidModelOutputError,
    _TreeEntry,
    _logical_tree_sha256,
    bounded_retry,
    build_benchmark_commands,
    load_config,
    messages_to_qwen_input_roles,
    normalize_assistant_response,
    required_vllm_server_arguments,
    runtime_environment,
)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "evaluate_agentdyn_qwen36_27b.json"
TEMPLATE = ROOT / "templates" / "qwen36_agentdyn_input_role.jinja"


def test_release_config_is_pinned_and_endpoint_free() -> None:
    config = load_config(CONFIG)
    assert config.suites == DEFAULT_SUITES
    assert config.attacks == DEFAULT_ATTACKS
    assert config.base_url_env == "AGENTDYN_BASE_URL"
    assert config.request_max_attempts == 3
    assert config.request_retry_max_wait_seconds == 40
    assert config.tool_call_parser == "qwen3_coder"
    assert config.reasoning_parser == "qwen3"
    text = CONFIG.read_text(encoding="utf-8")
    assert "127" + ".0.0.1" not in text
    assert "local" + "host" not in text.casefold()


def test_runtime_environment_passes_only_configured_model_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(CONFIG)
    monkeypatch.setenv(config.base_url_env, "https://model.example.invalid/v1")
    monkeypatch.setenv(config.api_key_env, "configured-agentdyn-key")
    monkeypatch.setenv("NVIDIA_API_KEY", "unrelated-nvidia-key")
    monkeypatch.setenv("UNRELATED_DATABASE_PASSWORD", "unrelated-password")

    environment = runtime_environment(config)

    assert environment[config.api_key_env] == "configured-agentdyn-key"
    assert environment["AGENTDYN_LOCAL_BASE_URL"] == (
        "https://model.example.invalid/v1"
    )
    assert config.base_url_env not in environment
    assert "NVIDIA_API_KEY" not in environment
    assert "UNRELATED_DATABASE_PASSWORD" not in environment


def test_release_config_rejects_boolean_coercion_and_extra_fields(
    tmp_path: Path,
) -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["generation"]["enable_thinking"] = "true"
    changed = tmp_path / "coerced.json"
    changed.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON boolean"):
        load_config(changed)

    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["server"]["debug"] = True
    changed.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="fields differ"):
        load_config(changed)


def test_trusted_untrusted_and_replayed_reasoning_are_separated() -> None:
    converted = messages_to_qwen_input_roles(
        [
            {"role": "system", "content": "trusted policy"},
            {"role": "user", "content": "trusted task"},
            {
                "role": "assistant",
                "content": [{"type": "text", "content": "calling"}],
                "reasoning_content": "private reasoning",
                "tool_calls": [
                    SimpleNamespace(
                        function="lookup",
                        args={"query": "term"},
                        id="c1",
                    )
                ],
            },
            {
                "role": "tool",
                "content": "untrusted injection",
                "tool_call_id": "c1",
            },
        ]
    )
    assert converted[0] == {
        "role": "user",
        "content": "trusted policy\n\ntrusted task",
    }
    assert converted[1]["content"].startswith("<think>\nprivate reasoning")
    assert converted[1]["tool_calls"][0]["function"]["name"] == "lookup"
    assert converted[-1]["role"] == "input"
    assert "untrusted injection" in converted[-1]["content"]


def test_one_scalar_command_is_built_for_each_attack(tmp_path: Path) -> None:
    config = load_config(CONFIG)
    commands = build_benchmark_commands(config=config, log_dir=tmp_path)
    assert [attack for attack, _command in commands] == list(DEFAULT_ATTACKS)
    clean = commands[0][1]
    attacked = commands[1][1]
    assert "--attack" not in clean
    assert attacked.count("--attack") == 1
    assert attacked[attacked.index("--attack") + 1] == "important_instructions"
    for _attack, command in commands:
        assert command[command.index("--tool-delimiter") + 1] == "input"
        assert command[command.index("--module-to-load") + 1] == (
            "secpo.evaluation.agentdyn_qwen36"
        )


def test_cli_executes_each_attack_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        agentdyn_cli,
        "verify_agentdyn_checkout",
        lambda *_args, **_kwargs: {"commit": "pinned"},
    )
    monkeypatch.setattr(
        agentdyn_cli,
        "required_vllm_server_arguments",
        lambda *_args, **_kwargs: ["--chat-template", "reviewed-template"],
    )
    monkeypatch.setattr(agentdyn_cli, "runtime_environment", lambda _config: {})
    observed: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> None:
        observed.append(command)

    monkeypatch.setattr(agentdyn_cli.subprocess, "run", fake_run)
    assert (
        agentdyn_cli.main(
            [
                "--checkout",
                str(tmp_path),
                "run",
                "--log-dir",
                str(tmp_path / "logs"),
                "--execute",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert len(observed) == 2
    assert "--attack" not in observed[0]
    assert observed[1].count("--attack") == 1


def test_required_server_arguments_pin_template_and_parsers() -> None:
    config = load_config(CONFIG)
    arguments = required_vllm_server_arguments(config, template=TEMPLATE)
    assert "--language-model-only" in arguments
    assert "--enable-auto-tool-choice" in arguments
    assert arguments[arguments.index("--chat-template") + 1] == (
        "templates/qwen36_agentdyn_input_role.jinja"
    )
    assert arguments[arguments.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert arguments[arguments.index("--reasoning-parser") + 1] == "qwen3"


def test_required_server_arguments_reject_changed_template(tmp_path: Path) -> None:
    changed = tmp_path / "template.jinja"
    changed.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="template checksum"):
        required_vllm_server_arguments(load_config(CONFIG), template=changed)


@pytest.mark.parametrize("reasoning_key", ["reasoning", "reasoning_content"])
def test_response_extras_never_leak_reasoning_into_visible_content(
    reasoning_key: str,
) -> None:
    normalized = normalize_assistant_response(
        {
            "content": "visible answer",
            "model_extra": {reasoning_key: "private chain"},
        },
        enable_thinking=True,
    )
    assert normalized.reasoning == "private chain"
    assert normalized.visible_content == "visible answer"
    assert "private chain" not in normalized.visible_content


def test_content_tags_are_split_and_xml_tool_call_is_recovered() -> None:
    normalized = normalize_assistant_response(
        {
            "content": (
                "private chain</think>\n"
                "<tool_call>\n"
                "<function=lookup>\n"
                "<parameter=query>\n{\"term\": \"value\"}\n</parameter>\n"
                "</function>\n"
                "</tool_call>"
            )
        },
        enable_thinking=True,
    )
    assert normalized.reasoning == "private chain"
    assert normalized.visible_content is None
    assert len(normalized.tool_calls) == 1
    assert normalized.tool_calls[0].function == "lookup"
    assert normalized.tool_calls[0].args == {"query": {"term": "value"}}


def test_open_think_prefix_is_not_exposed_as_visible_content() -> None:
    normalized = normalize_assistant_response(
        {"content": "private prefix<think>private body</think>visible answer"},
        enable_thinking=True,
    )
    assert normalized.visible_content == "visible answer"
    assert normalized.reasoning == "private prefix\nprivate body"


def test_unclosed_thinking_without_tool_or_visible_answer_fails() -> None:
    with pytest.raises(InvalidModelOutputError):
        normalize_assistant_response(
            {"content": "unfinished private chain"},
            enable_thinking=True,
        )


def test_bounded_retry_stops_after_success_and_uses_finite_waits() -> None:
    attempts = 0
    waits: list[float] = []

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("transient")
        return "ok"

    assert (
        bounded_retry(
            operation,
            max_attempts=3,
            max_wait_seconds=40,
            sleep=waits.append,
        )
        == "ok"
    )
    assert attempts == 3
    assert waits == [1.0, 2.0]


def test_bounded_retry_does_not_retry_nonretryable_error() -> None:
    class InvalidRequest(RuntimeError):
        pass

    attempts = 0

    def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise InvalidRequest("invalid")

    with pytest.raises(InvalidRequest):
        bounded_retry(
            operation,
            max_attempts=3,
            max_wait_seconds=40,
            non_retryable_exceptions=(InvalidRequest,),
            sleep=lambda _seconds: pytest.fail("nonretryable request slept"),
        )
    assert attempts == 1


def test_logical_whole_tree_hash_rejects_extra_touched_file_edit() -> None:
    entries = (
        _TreeEntry(path="a.py", mode="100644", object_id="base-a"),
        _TreeEntry(path="b.py", mode="100644", object_id="base-b"),
    )
    reviewed = _logical_tree_sha256(
        entries,
        {"a.py": ("100644", "reviewed-content")},
    )
    extra_edit = _logical_tree_sha256(
        entries,
        {"a.py": ("100644", "reviewed-content-plus-extra")},
    )
    assert reviewed != extra_edit
