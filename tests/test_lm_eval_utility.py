from __future__ import annotations

import json

import pytest

import secpo.evaluation.lm_eval_utility as utility
from secpo.evaluation.lm_eval_utility import (
    build_lm_eval_commands,
    load_utility_config,
    parse_utility_config,
)


def _config():
    return {
        "model_name_or_path": "public/model",
        "model_revision": "0123456789abcdef0123456789abcdef01234567",
        "tasks": [
            {"task": "mmlu_pro", "num_fewshot": 5, "metric": "exact_match"},
            {"task": "minerva_math", "num_fewshot": 4, "metric": "math_verify"},
        ],
        "batch_size": "auto",
        "dtype": "bfloat16",
        "seed": 42,
        "trust_remote_code": False,
        "apply_chat_template": True,
    }


def test_builds_one_resumable_command_per_task(tmp_path) -> None:
    config = parse_utility_config(_config())
    commands = build_lm_eval_commands(
        config,
        output_dir=tmp_path / "results",
        adapter_path=tmp_path / "adapter",
        python_executable="python",
    )

    assert len(commands) == 2
    assert commands[0][commands[0].index("--tasks") + 1] == "mmlu_pro"
    assert commands[0][commands[0].index("--num_fewshot") + 1] == "5"
    assert "--apply_chat_template" in commands[0]
    model_args = commands[0][commands[0].index("--model_args") + 1]
    assert "pretrained=public/model" in model_args
    assert "revision=0123456789abcdef0123456789abcdef01234567" in model_args
    assert f"peft={tmp_path / 'adapter'}" in model_args


def test_config_rejects_duplicate_tasks() -> None:
    value = _config()
    value["tasks"].append(dict(value["tasks"][0]))

    with pytest.raises(ValueError, match="duplicate task"):
        parse_utility_config(value)


def test_model_id_must_be_public() -> None:
    value = _config()
    value["model_name_or_path"] = "/private/model"

    with pytest.raises(ValueError, match="organization/model"):
        parse_utility_config(value)


def test_model_revision_must_be_immutable_commit() -> None:
    value = _config()
    value["model_revision"] = "main"
    with pytest.raises(ValueError, match="40-character"):
        parse_utility_config(value)


def test_released_utility_configs_pin_exact_model_revisions() -> None:
    llama, _ = load_utility_config(
        "configs/evaluate_lm_eval_llama31_8b.json"
    )
    qwen, _ = load_utility_config("configs/evaluate_lm_eval_qwen36_27b.json")
    assert llama.model_revision == "0e9e39f249a16976918f6564b8830bc894c89659"
    assert qwen.model_revision == "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"


def test_execute_passes_only_repeatable_explicit_environment_names(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "utility.json"
    config.write_text(json.dumps(_config()), encoding="utf-8")
    monkeypatch.setenv("APPROVED_MODEL_TOKEN", "approved-value")
    monkeypatch.setenv("NVIDIA_API_KEY", "unrelated-nvidia-key")
    monkeypatch.setenv("UNRELATED_DATABASE_PASSWORD", "unrelated-password")
    observed = []

    def fake_run(command, **kwargs):
        observed.append((command, kwargs["env"]))

    monkeypatch.setattr(utility.subprocess, "run", fake_run)
    assert utility.main(
        [
            "--config",
            str(config),
            "--output-dir",
            str(tmp_path / "results"),
            "--execute",
            "--pass-env",
            "APPROVED_MODEL_TOKEN",
        ]
    ) == 0
    assert observed
    for _command, environment in observed:
        assert environment["APPROVED_MODEL_TOKEN"] == "approved-value"
        assert "NVIDIA_API_KEY" not in environment
        assert "UNRELATED_DATABASE_PASSWORD" not in environment

    observed.clear()
    assert utility.main(
        [
            "--config",
            str(config),
            "--output-dir",
            str(tmp_path / "results-default"),
            "--execute",
        ]
    ) == 0
    assert "APPROVED_MODEL_TOKEN" not in observed[0][1]
