import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def collector(tmp_path, monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("collector_test", scripts / "collect_lambda_gpqa.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "OUTPUT", tmp_path)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    return module


@pytest.mark.parametrize("status", ["paused", "failed"])
def test_remote_stop_never_restarts_or_scores(collector, monkeypatch, status):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps(dict(status=status, completed=2, total=198)))
    monkeypatch.setattr(collector.subprocess, "run", run)
    assert collector.worker() == 1
    assert len(calls) == 1 and calls[0][0] == "ssh"
    assert json.loads((collector.OUTPUT / "collector.json").read_text())["status"] == "remote_" + status


def test_incomplete_generation_cannot_score(collector, monkeypatch):
    monkeypatch.setattr(collector.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0,
        stdout=json.dumps(dict(status="generation_complete", completed=197, total=198))))
    with pytest.raises(ValueError, match="Incomplete"):
        collector.worker()


def test_connection_failures_stop_after_ten_attempts(collector, monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=255, stdout="", stderr="Connection refused")
    monkeypatch.setattr(collector.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="ten attempts"):
        collector.worker()
    assert len(calls) == 10


def test_complete_run_collected_and_scored_offline(collector, monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "ssh":
            return SimpleNamespace(returncode=0, stdout=json.dumps(dict(status="generation_complete", completed=198, total=198)))
        if command[0] == "scp":
            assert command[-1].startswith(str(collector.OUTPUT / "remote"))
            return SimpleNamespace(returncode=0)
        assert "score" in command and "--generations" in command
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        collector.write_json(collector.OUTPUT / "results.json", dict(status="complete", n=198, results={"accuracy":0.8}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(collector.subprocess, "run", run)
    assert collector.worker() == 0
    assert len([c for c in calls if c[0] == "scp"]) == 6
    assert json.loads((collector.OUTPUT / "collector.json").read_text())["status"] == "complete"
