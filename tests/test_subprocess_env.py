from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from secpo.subprocess_env import (
    build_subprocess_environment,
    isolated_environment,
)


def test_actual_child_receives_only_operational_and_explicit_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("NVIDIA_API_KEY", "must-not-reach-child")
    monkeypatch.setenv("UNRELATED_DATABASE_PASSWORD", "also-must-not-reach-child")
    monkeypatch.setenv("APPROVED_MODEL_TOKEN", "approved-test-value")

    environment = build_subprocess_environment(
        pass_env=("APPROVED_MODEL_TOKEN",),
        require_pass_env=True,
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,os; print(json.dumps(dict(os.environ), sort_keys=True))",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    child = json.loads(completed.stdout)

    assert child["CUDA_VISIBLE_DEVICES"] == "2"
    assert child["APPROVED_MODEL_TOKEN"] == "approved-test-value"
    assert "NVIDIA_API_KEY" not in child
    assert "UNRELATED_DATABASE_PASSWORD" not in child


def test_sensitive_override_requires_specific_approval() -> None:
    with pytest.raises(ValueError, match="explicit approval"):
        build_subprocess_environment(
            source={},
            overrides={"SERVICE_API_KEY": "test-value"},
        )
    assert build_subprocess_environment(
        source={},
        overrides={"SERVICE_API_KEY": "test-value"},
        approved_sensitive_names=("SERVICE_API_KEY",),
    ) == {"SERVICE_API_KEY": "test-value"}


def test_isolated_environment_hides_secrets_and_restores_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "parent-nvidia-value")
    monkeypatch.setenv("UNRELATED_DATABASE_PASSWORD", "parent-password")
    monkeypatch.setenv("APPROVED_ENDPOINT", "https://model.example.invalid/v1")
    selected = build_subprocess_environment(pass_env=("APPROVED_ENDPOINT",))

    with isolated_environment(selected):
        assert os.environ["APPROVED_ENDPOINT"].startswith("https://")
        assert "NVIDIA_API_KEY" not in os.environ
        assert "UNRELATED_DATABASE_PASSWORD" not in os.environ

    assert os.environ["NVIDIA_API_KEY"] == "parent-nvidia-value"
    assert os.environ["UNRELATED_DATABASE_PASSWORD"] == "parent-password"
