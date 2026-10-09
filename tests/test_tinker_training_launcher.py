"""Offline duplicate-launch guards; no training clients are constructed."""

import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "tinker_training_launcher", Path(__file__).resolve().parents[1] / "scripts/launch_tinker_training.py"
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.mark.parametrize("artifact", ["config.json", "metrics.jsonl", "result.json", "worker.json", "launch.json"])
def test_existing_artifact_prevents_fresh_training(tmp_path, artifact):
    (tmp_path / artifact).touch()
    with pytest.raises(FileExistsError):
        launcher.require_fresh_output(tmp_path)


def test_lock_file_alone_does_not_block_fresh_output(tmp_path):
    (tmp_path / "launch.lock").touch()
    launcher.require_fresh_output(tmp_path)
