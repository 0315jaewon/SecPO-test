from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
from types import ModuleType

import pytest

import secpo.evaluation.dtap_qwen_agent_backend as backend
from secpo.evaluation.dtap_qwen36 import (
    backend_fingerprint,
    canonical_bytes,
    fingerprint,
    load_config,
)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "evaluate_dtap_qwen36_27b.json"


def _set_judge_runtime(monkeypatch: pytest.MonkeyPatch) -> str:
    base_url = "https://judge.example.invalid/v1"
    model = "public-fixture-judge"
    timeout = 60
    identity = hashlib.sha256(
        canonical_bytes(
            {
                "base_url": base_url,
                "model": model,
                "timeout_seconds": timeout,
            }
        )
    ).hexdigest()
    monkeypatch.setenv("SECPODTAP_JUDGE_BASE_URL", base_url)
    monkeypatch.setenv("SECPODTAP_JUDGE_MODEL", model)
    monkeypatch.setenv("SECPODTAP_JUDGE_API_KEY", "test-only-secret")
    monkeypatch.setenv("SECPODTAP_JUDGE_TIMEOUT_SECONDS", str(timeout))
    monkeypatch.setenv("SECPODTAP_JUDGE_IDENTITY_SHA256", identity)
    return identity


def _request() -> dict:
    task = {
        "domain": "browser",
        "type": "malicious",
        "task_id": "1",
        "threat_model": "indirect",
        "risk_category": "browser-integrity",
    }
    return {
        "schema_version": "secpo-dtap-qwen36-v1",
        "task": task,
        "task_fingerprint": fingerprint(task),
        "protocol": load_config(CONFIG).protocol(),
    }


def _runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    checkout = tmp_path / "dtap"
    task_dir = (
        checkout
        / "dataset"
        / "browser"
        / "malicious"
        / "indirect"
        / "browser-integrity"
        / "1"
    )
    task_dir.mkdir(parents=True)
    (task_dir / "config.yaml").write_text("task_id: 1\n", encoding="utf-8")
    results = tmp_path / "dtap-results"
    runtime_root = tmp_path / "runtime"
    shard_root = runtime_root / "shard_00"
    lock_root = shard_root / "port-locks"
    lock_root.mkdir(parents=True)
    playwright_root = tmp_path / "playwright-browsers"
    playwright_root.mkdir()
    monkeypatch.setenv("SECPODTAP_CHECKOUT", str(checkout))
    monkeypatch.setenv("SECPODTAP_RESULTS_ROOT", str(results))
    monkeypatch.setenv("SECPODTAP_MODEL", "checkpoint/public-fixture")
    monkeypatch.setenv("SECPODTAP_MODEL_IDENTITY_SHA256", "a" * 64)
    _set_judge_runtime(monkeypatch)
    monkeypatch.setenv(
        "SECPODTAP_PLAYWRIGHT_BROWSERS_PATH", str(playwright_root)
    )
    monkeypatch.setenv("SECPODTAP_RUNTIME_ROOT", str(runtime_root))
    monkeypatch.setenv("SECPODTAP_SHARD_ID", "0")
    monkeypatch.setenv("SECPODTAP_SHARD_RUNTIME_ROOT", str(shard_root))
    monkeypatch.setenv("DT_PORT_DEFAULT_OFFSET", "0")
    monkeypatch.setenv("DT_PORT_RANGE", "30000-30999")
    monkeypatch.setenv("DT_PORT_LOCK_DIR", str(lock_root))
    monkeypatch.setenv("DTAP_QWEN_BASE_URL", "https://victim.example.invalid/v1")
    monkeypatch.delenv(backend.DTAP_PASS_ENV_SELECTION, raising=False)
    monkeypatch.setattr(backend, "_verify_checkout", lambda path: None)
    monkeypatch.setattr(backend, "_verify_release_adapter", lambda: None)
    monkeypatch.setattr(backend, "_verify_task_payload", lambda *args: None)
    return checkout, results


def _judge_payload(*, task_success: object, attack_success: object) -> dict:
    return {
        "task_success": task_success,
        "task_message": "utility outcome",
        "task_metadata": {},
        "attack_success": attack_success,
        "attack_message": "attack outcome",
        "attack_metadata": {},
        "error": None,
    }


def test_concrete_backend_runs_task_lifecycle_and_reads_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout, results = _runtime(tmp_path, monkeypatch)
    monkeypatch.setenv("DTAP_QWEN_API_KEY", "unselected-victim-key")
    monkeypatch.setenv("NVIDIA_API_KEY", "unrelated-nvidia-key")
    monkeypatch.setenv("UNRELATED_DATABASE_PASSWORD", "unrelated-password")
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        judge = (
            results
            / "benchmark"
            / "qwen_agent_input"
            / "checkpoint_public-fixture"
            / "browser"
            / "malicious"
            / "indirect"
            / "browser-integrity"
            / "1"
            / "judge_result.json"
        )
        judge.parent.mkdir(parents=True)
        judge.write_text(
            json.dumps(_judge_payload(task_success=False, attack_success=True)),
            encoding="utf-8",
        )
        # DTap returns nonzero when task_success is false; a valid judge result
        # remains the authoritative outcome.
        return 1

    monkeypatch.setattr(backend, "_run_task_process", fake_run)
    assert backend.run_task(_request()) == {
        "task_success": False,
        "attack_success": True,
    }
    command = observed["command"]
    assert command[1:3] == ["-m", "eval.evaluation"]
    task_list = Path(command[command.index("--task-list") + 1])
    assert json.loads(task_list.read_text(encoding="utf-8")) == _request()["task"]
    assert command[command.index("--agent-type") + 1] == "qwen_agent_input"
    assert command[command.index("--max-parallel") + 1] == "1"
    assert command[command.index("--temperature") + 1] == "1.0"
    assert command[command.index("--port-range") + 1] == "30000-30999"
    assert "--keep-envs" not in command
    assert observed["kwargs"]["checkout"] == checkout.resolve()
    environment = observed["kwargs"]["environment"]
    assert environment["DTAP_QWEN_ENABLE_THINKING"] == "0"
    assert environment["DTAP_QWEN_MAX_OUTPUT_TOKENS"] == "32768"
    assert environment["DTAP_QWEN_BASE_URL"] == (
        "https://victim.example.invalid/v1"
    )
    assert "DTAP_QWEN_API_KEY" not in environment
    assert "NVIDIA_API_KEY" not in environment
    assert "UNRELATED_DATABASE_PASSWORD" not in environment
    assert environment["DTAP_QWEN_TEMPERATURE"] == "1.0"
    assert environment["DTAP_QWEN_TOP_P"] == "0.95"
    assert environment["DTAP_QWEN_TOP_K"] == "20"
    assert environment["DTAP_QWEN_SEED"] == "0"
    assert environment["DTAP_QWEN_PRESENCE_PENALTY"] == "0"
    assert environment["DTAP_QWEN_TOOL_TIMEOUT"] == "30"
    assert environment["DTAP_QWEN_REQUEST_TIMEOUT"] == "1800"
    assert environment["OPENAI_BASE_URL"] == "https://judge.example.invalid/v1"
    assert environment["OPENAI_MODEL"] == "public-fixture-judge"
    assert environment["OPENAI_API_KEY"] == "test-only-secret"
    assert environment["OPENAI_TIMEOUT_SEC"] == "60"
    assert environment["EVAL_TASK_TIMEOUT"] == "12960"
    assert environment["DTAP_SKIP_DATASET_DOWNLOAD"] == "1"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["UV_PYTHON_DOWNLOADS"] == "never"
    assert environment["UV_NO_SYNC"] == "1"
    assert environment["PLAYWRIGHT_BROWSERS_PATH"].endswith(
        "playwright-browsers"
    )
    assert environment["TOKENIZERS_PARALLELISM"] == "false"
    assert environment["OMP_NUM_THREADS"] == "1"
    task_root = task_list.parent
    assert Path(environment["TMPDIR"]).parent == task_root
    shard_shared = task_root.parents[1] / "shared"
    assert Path(environment["HOME"]).parent == task_root
    assert Path(environment["XDG_CACHE_HOME"]) == shard_shared / "cache"
    assert Path(environment["UV_CACHE_DIR"]) == shard_shared / "cache" / "uv"
    assert Path(environment["HF_HOME"]) == shard_shared / "cache" / "huggingface"
    assert Path(environment["PYTHONPYCACHEPREFIX"]) == shard_shared / "pycache"
    assert Path(environment["XDG_RUNTIME_DIR"]).parent == task_root
    assert Path(environment["XDG_CONFIG_HOME"]).parent == task_root
    assert Path(environment["XDG_DATA_HOME"]).parent == task_root
    assert Path(environment["XDG_STATE_HOME"]).parent == task_root
    assert Path(environment["DTAP_MCP_LOG_DIR"]).parent == task_root
    assert Path(environment[backend.TASK_SESSION_FILE_ENVIRONMENT]).parent == task_root
    assert (
        Path(environment[backend.ENVIRONMENT_CLEANUP_FILE_ENVIRONMENT]).parent
        == task_root
    )
    assert environment["DT_POOL_PREFIX"] == environment["COMPOSE_PROJECT_NAME"]
    assert environment["DT_POOL_PREFIX"].startswith("secpo-")
    assert environment["DTAP_MCP_LOG_SCOPE"] == fingerprint(_request()["task"])


def test_victim_credential_requires_explicit_dtap_pass_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, results = _runtime(tmp_path, monkeypatch)
    monkeypatch.setenv("DTAP_QWEN_API_KEY", "explicit-victim-key")
    monkeypatch.setenv(
        backend.DTAP_PASS_ENV_SELECTION,
        "DTAP_QWEN_API_KEY",
    )
    monkeypatch.setenv("NVIDIA_API_KEY", "unrelated-nvidia-key")
    monkeypatch.setenv("UNRELATED_DATABASE_PASSWORD", "unrelated-password")
    observed: dict[str, str] = {}

    def fake_run(command, **kwargs):
        del command
        observed.update(kwargs["environment"])
        judge = (
            results
            / "benchmark/qwen_agent_input/checkpoint_public-fixture/browser/malicious"
            / "indirect/browser-integrity/1/judge_result.json"
        )
        judge.parent.mkdir(parents=True)
        judge.write_text(
            json.dumps(_judge_payload(task_success=True, attack_success=False)),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(backend, "_run_task_process", fake_run)
    assert backend.run_task(_request()) == {
        "task_success": True,
        "attack_success": False,
    }
    assert observed["DTAP_QWEN_API_KEY"] == "explicit-victim-key"
    assert "NVIDIA_API_KEY" not in observed
    assert "UNRELATED_DATABASE_PASSWORD" not in observed


@pytest.mark.parametrize(
    "value",
    [
        "relative-endpoint",
        "ftp://victim.example.invalid/v1",
        "https://user:password" + "@" + "victim.example.invalid/v1",
        "https://victim.example.invalid/v1?tenant=private",
        "https://victim.example.invalid/v1#fragment",
        "https://victim.example.invalid:invalid/v1",
    ],
)
def test_victim_endpoint_rejects_non_absolute_or_secret_bearing_urls(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DTAP_QWEN_BASE_URL", value)
    with pytest.raises(ValueError, match="DTAP_QWEN_BASE_URL"):
        backend._validated_model_base_url()


def test_task_attempts_share_shard_caches_but_isolate_mutable_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _runtime(tmp_path, monkeypatch)
    shard_id, shard_root = backend._validated_shard_runtime()
    first_environment: dict[str, str] = {}
    second_environment: dict[str, str] = {}
    first_task = _request()["task"]
    second_task = {**first_task, "task_id": "2"}
    first_list = backend._task_runtime(
        first_environment,
        shard_id=shard_id,
        shard_root=shard_root,
        task=first_task,
    )
    second_list = backend._task_runtime(
        second_environment,
        shard_id=shard_id,
        shard_root=shard_root,
        task=second_task,
    )

    for name in (
        "XDG_CACHE_HOME",
        "UV_CACHE_DIR",
        "HF_HOME",
        "PYTHONPYCACHEPREFIX",
    ):
        assert first_environment[name] == second_environment[name]
    for name in (
        "HOME",
        "TMPDIR",
        "XDG_RUNTIME_DIR",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
        "DT_POOL_PREFIX",
        "COMPOSE_PROJECT_NAME",
        "DTAP_MCP_LOG_DIR",
        "DTAP_MCP_LOG_SCOPE",
        backend.TASK_SESSION_FILE_ENVIRONMENT,
        backend.ENVIRONMENT_CLEANUP_FILE_ENVIRONMENT,
    ):
        assert first_environment[name] != second_environment[name]
    assert first_list.parent != second_list.parent


def test_concrete_backend_fails_closed_without_boolean_judge_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, results = _runtime(tmp_path, monkeypatch)

    def fake_run(command, **kwargs):
        del command, kwargs
        judge = (
            results
            / "benchmark"
            / "qwen_agent_input"
            / "checkpoint_public-fixture"
            / "browser"
            / "malicious"
            / "indirect"
            / "browser-integrity"
            / "1"
            / "judge_result.json"
        )
        judge.parent.mkdir(parents=True)
        judge.write_text(
            json.dumps(_judge_payload(task_success=1, attack_success=False)),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(backend, "_run_task_process", fake_run)
    # The public boundary intentionally collapses every malformed/missing
    # native judge artifact into one fail-closed backend error.
    with pytest.raises(RuntimeError, match="no valid judge result"):
        backend.run_task(_request())


def test_concrete_backend_rejects_stale_judge_after_failed_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, results = _runtime(tmp_path, monkeypatch)
    judge = (
        results
        / "benchmark/qwen_agent_input/checkpoint_public-fixture/browser/malicious"
        / "indirect/browser-integrity/1/judge_result.json"
    )
    judge.parent.mkdir(parents=True)
    judge.write_text(
        json.dumps(_judge_payload(task_success=False, attack_success=True)),
        encoding="utf-8",
    )
    monkeypatch.setattr(backend, "_run_task_process", lambda *args, **kwargs: 1)
    with pytest.raises(RuntimeError, match="fresh judge"):
        backend.run_task(_request())


def test_backend_fingerprint_binds_public_checkpoint_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_judge = _set_judge_runtime(monkeypatch)
    monkeypatch.setenv("SECPODTAP_MODEL_IDENTITY_SHA256", "a" * 64)
    first = backend_fingerprint(backend.run_task)
    monkeypatch.setenv("SECPODTAP_MODEL_IDENTITY_SHA256", "b" * 64)
    assert backend_fingerprint(backend.run_task) != first
    monkeypatch.setenv("SECPODTAP_MODEL_IDENTITY_SHA256", "a" * 64)
    monkeypatch.setenv("SECPODTAP_JUDGE_MODEL", "different-judge")
    different_judge = hashlib.sha256(
        canonical_bytes(
            {
                "base_url": "https://judge.example.invalid/v1",
                "model": "different-judge",
                "timeout_seconds": 60,
            }
        )
    ).hexdigest()
    monkeypatch.setenv("SECPODTAP_JUDGE_IDENTITY_SHA256", different_judge)
    assert different_judge != first_judge
    assert backend_fingerprint(backend.run_task) != first


def test_release_adapter_checksum_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    backend._verify_release_adapter()
    monkeypatch.setattr(backend, "file_sha256", lambda path: "0" * 64)
    with pytest.raises(RuntimeError, match="adapter checksum"):
        backend._verify_release_adapter()


def test_task_timeout_terminates_group_and_reaps_leader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []
    alive = {"value": True}

    class FakeProcess:
        pid = 4321

        def __init__(self):
            self.calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("dtap", timeout)
            return -signal.SIGTERM

    process = FakeProcess()
    monkeypatch.setattr(backend.subprocess, "Popen", lambda *a, **k: process)
    def fake_killpg(pid, selected_signal):
        if selected_signal == 0:
            if not alive["value"]:
                raise ProcessLookupError
            return
        observed.append((pid, selected_signal))
        alive["value"] = False

    monkeypatch.setattr(backend.os, "killpg", fake_killpg)
    with pytest.raises(subprocess.TimeoutExpired):
        backend._run_task_process(
            ["python", "-m", "eval.task_runner"],
            checkout=Path("/tmp"),
            environment={},
            timeout=1,
        )
    assert observed == [(4321, signal.SIGTERM)]


def test_normal_task_exit_cleans_any_remaining_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []
    alive = {"value": True}

    class FakeProcess:
        pid = 7654

        def wait(self, timeout=None):
            del timeout
            return 0

    monkeypatch.setattr(backend.subprocess, "Popen", lambda *a, **k: FakeProcess())
    def fake_killpg(pid, selected_signal):
        if selected_signal == 0:
            if not alive["value"]:
                raise ProcessLookupError
            return
        observed.append((pid, selected_signal))
        alive["value"] = False

    monkeypatch.setattr(backend.os, "killpg", fake_killpg)
    assert (
        backend._run_task_process(
            ["python", "-m", "eval.task_runner"],
            checkout=Path("/tmp"),
            environment={},
            timeout=1,
        )
        == 0
    )
    assert observed == [(7654, signal.SIGTERM)]


def test_stubborn_descendant_group_is_escalated_to_sigkill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []

    class FakeProcess:
        pid = 8765

        def wait(self, timeout=None):
            del timeout
            return 0

    monkeypatch.setattr(backend.subprocess, "Popen", lambda *a, **k: FakeProcess())
    monkeypatch.setattr(
        backend.os, "killpg", lambda pid, sig: observed.append((pid, sig))
    )
    monkeypatch.setattr(
        backend, "_wait_for_process_group_exit", lambda *args, **kwargs: False
    )
    assert (
        backend._run_task_process(
            ["python", "-m", "eval.task_runner"],
            checkout=Path("/tmp"),
            environment={},
            timeout=1,
        )
        == 0
    )
    assert observed == [
        (8765, signal.SIGTERM),
        (8765, signal.SIGKILL),
    ]


def test_outer_timeout_kills_every_group_in_registered_task_session(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "task-session.json"
    session_id_file = tmp_path / "task-session-id"
    evaluation_session_file = tmp_path / "evaluation-session-id"
    child_code = (
        "import subprocess,sys,time;"
        "subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],"
        "process_group=0);"
        "time.sleep(60)"
    )
    evaluation_code = (
        "import json,os,subprocess,sys,time;"
        "compose=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],"
        "preexec_fn=os.setpgrp);"
        f"child=subprocess.Popen([sys.executable,'-c',{child_code!r}],"
        "start_new_session=True);"
        f"open({str(evaluation_session_file)!r},'w').write(str(os.getpid()));"
        f"open({str(session_id_file)!r},'w').write(str(child.pid));"
        "payload={'schema_version':'dtap-task-session-v1',"
        "'evaluation_pid':os.getpid(),'task_session':child.pid};"
        f"open({str(registry)!r},'w').write(json.dumps(payload));"
        "time.sleep(60)"
    )
    environment = {
        **backend.os.environ,
        backend.TASK_SESSION_FILE_ENVIRONMENT: str(registry),
    }
    with pytest.raises(subprocess.TimeoutExpired):
        backend._run_task_process(
            [sys.executable, "-c", evaluation_code],
            checkout=tmp_path,
            environment=environment,
            timeout=1,
        )
    task_session = int(session_id_file.read_text(encoding="utf-8"))
    evaluation_session = int(
        evaluation_session_file.read_text(encoding="utf-8")
    )
    assert backend._task_session_members(evaluation_session) == []
    assert backend._task_session_members(task_session) == []
    assert not registry.exists()


def test_compose_project_discovery_is_exactly_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = "secpo-run-s00-t123-a456"
    labels = [
        f"com.docker.compose.project={prefix}",
        f"other=value,com.docker.compose.project={prefix}_gmail_1_2",
        f"com.docker.compose.project={prefix}evil",
        "com.docker.compose.project=unrelated",
    ]
    monkeypatch.setattr(backend, "_docker_output", lambda base, command: labels)
    assert backend._compose_projects(["docker"], prefix) == {
        prefix,
        prefix + "_gmail_1_2",
    }


def test_compose_cleanup_removes_exact_projects_and_fails_on_residuals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = "secpo-run-s00-t123-a456"
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    environment = {
        "DT_POOL_PREFIX": prefix,
        "SECPODTAP_RUNTIME_ROOT": str(runtime_root),
    }
    projects = {prefix, prefix + "_gmail_1_2"}
    snapshots = iter((projects, set()))
    calls: list[list[str]] = []
    monkeypatch.setattr(backend, "_docker_base_command", lambda: ["docker"])
    monkeypatch.setattr(
        backend,
        "_compose_projects",
        lambda base, observed_prefix: next(snapshots),
    )

    def fake_output(base, arguments):
        del base
        calls.append(arguments)
        if "--filter" in arguments:
            return ["resource-id"]
        return []

    monkeypatch.setattr(backend, "_docker_output", fake_output)
    backend._cleanup_compose_resources(environment)
    removals = [
        arguments[0]
        for arguments in calls
        if arguments[0] in {"rm", "network", "volume"}
        and "--filter" not in arguments
    ]
    assert removals == ["rm", "network", "volume"] * 2

    monkeypatch.setattr(
        backend,
        "_compose_projects",
        lambda base, observed_prefix: {prefix},
    )
    with pytest.raises(RuntimeError, match="remained after task cleanup"):
        backend._cleanup_compose_resources(environment)


def test_verified_cleanup_marker_skips_fallback_docker_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation_pid = 43210
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    marker = tmp_path / "environment-cleanup.json"
    marker.write_text(
        json.dumps(
            {
                "schema_version": backend.ENVIRONMENT_CLEANUP_SCHEMA,
                "evaluation_pid": evaluation_pid,
            }
        ),
        encoding="utf-8",
    )
    environment = {
        "DT_POOL_PREFIX": "secpo-run-s00-t123-a456",
        "SECPODTAP_RUNTIME_ROOT": str(runtime_root),
        backend.ENVIRONMENT_CLEANUP_FILE_ENVIRONMENT: str(marker),
    }
    monkeypatch.setattr(
        backend,
        "_docker_base_command",
        lambda: pytest.fail("verified cleanup must not scan Docker"),
    )
    backend._cleanup_compose_resources(environment, evaluation_pid)
    assert not marker.exists()


def test_backend_rejects_model_alias_that_escapes_result_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _runtime(tmp_path, monkeypatch)
    monkeypatch.setenv("SECPODTAP_MODEL", "..")
    with pytest.raises(ValueError, match="safe result component"):
        backend.run_task(_request())


def test_release_adapter_has_no_baked_endpoint_and_freezes_thinking_off() -> None:
    source = (ROOT / "src/secpo/evaluation/dtap_qwen_agent_adapter.py").read_text(
        encoding="utf-8"
    )
    assert "http://" not in source and "https://" not in source
    assert 'os.environ["DTAP_QWEN_BASE_URL"]' in source
    assert '_env_bool("DTAP_QWEN_ENABLE_THINKING", False)' in source
    assert "self._enable_thinking = False" in source
    assert "Qwen-Agent bounded output-budget recovery exhausted" in source
    assert "raise last_error" not in source


def test_patch_registers_qwen_agent_and_separates_indirect_input() -> None:
    source = (ROOT / "third_party/patches/dtap.patch").read_text(encoding="utf-8")
    assert '"qwen_agent_input"' in source
    assert '"secpo.evaluation.dtap_qwen_agent_adapter"' in source
    assert 'agent_type in {"qwen_input", "qwen_agent_input"}' in source
    for lifecycle_contract in (
        "DTAP_TASK_SESSION_FILE",
        "DTAP_ENVIRONMENT_CLEANUP_FILE",
        "process_group=0",
        "_ordered_task_environments",
        "_merge_launch_ports",
        "inherited_ports",
        "_portable_setup_command",
        "crm_quote_seed_overlay.sql",
        "_stop_managers_concurrently",
        "_bind_research_judge_arxiv_endpoint",
        "DTAP_PYTHON",
        "_decode_runtime_ps",
        "_communicate_bounded",
        '"--all"',
        "_openai_chat_completions_url",
    ):
        assert lifecycle_contract in source
    assert '+    base_url = os.getenv("OPENAI_BASE_URL", "").strip()' in source
    assert '+    model = os.getenv("OPENAI_MODEL", "").strip()' in source


def _load_adapter_with_dependency_stubs(monkeypatch: pytest.MonkeyPatch):
    class Message:
        def __init__(
            self,
            *,
            role,
            content=None,
            extra=None,
            reasoning_content=None,
            function_call=None,
            **kwargs,
        ):
            del kwargs
            self.role = role
            self.content = content
            self.extra = extra
            self.reasoning_content = reasoning_content
            self.function_call = function_call

    class ContentItem:
        def __init__(self, *, text=None, image=None):
            self.text = text
            self.image = image

        def model_dump(self):
            return {"text": self.text, "image": self.image}

    class TextChatAtOAI:
        def __init__(self, cfg):
            self.generate_cfg = cfg.get("generate_cfg", {})

    class BaseTool:
        def __init__(self):
            pass

    modules = {
        "openai": ModuleType("openai"),
        "qwen_agent": ModuleType("qwen_agent"),
        "qwen_agent.agents": ModuleType("qwen_agent.agents"),
        "qwen_agent.llm": ModuleType("qwen_agent.llm"),
        "qwen_agent.llm.oai": ModuleType("qwen_agent.llm.oai"),
        "qwen_agent.llm.schema": ModuleType("qwen_agent.llm.schema"),
        "qwen_agent.tools": ModuleType("qwen_agent.tools"),
        "qwen_agent.tools.base": ModuleType("qwen_agent.tools.base"),
        "agent": ModuleType("agent"),
        "agent.qwen_input": ModuleType("agent.qwen_input"),
        "agent.qwen_input.src": ModuleType("agent.qwen_input.src"),
        "agent.qwen_input.src.agent": ModuleType("agent.qwen_input.src.agent"),
        "dt_arena": ModuleType("dt_arena"),
        "dt_arena.src": ModuleType("dt_arena.src"),
        "dt_arena.src.types": ModuleType("dt_arena.src.types"),
        "dt_arena.src.types.agent": ModuleType("dt_arena.src.types.agent"),
        "dt_arena.src.types.trajectory": ModuleType("dt_arena.src.types.trajectory"),
    }
    modules["openai"].BadRequestError = Exception
    modules["openai"].OpenAI = object
    modules["qwen_agent.agents"].FnCallAgent = object
    modules["qwen_agent.llm.oai"].TextChatAtOAI = TextChatAtOAI
    schema = modules["qwen_agent.llm.schema"]
    schema.ASSISTANT = "assistant"
    schema.USER = "user"
    schema.ContentItem = ContentItem
    schema.Message = Message
    modules["qwen_agent.tools.base"].BaseTool = BaseTool
    lower = modules["agent.qwen_input.src.agent"]
    lower.MIN_OUTPUT_TOKENS = 32_768
    lower.TOOL_INJECTION_BOUNDARY = "boundary"
    lower.TRUSTED_TOOL_RESULT_NAME = "dtap_trusted_tool_result"
    lower.UNTRUSTED_MESSAGE_NAME = "dtap_untrusted"
    lower.UNTRUSTED_TOOL_RESULT_NAME = "dtap_untrusted_tool_result"
    lower.QwenInputRoleAgent = object
    lower._env_bool = lambda name, default: default
    lower._fallback_tool_calls = lambda content: []
    agent_types = modules["dt_arena.src.types.agent"]
    agent_types.Agent = object
    agent_types.AgentConfig = object
    agent_types.AgentResult = object
    agent_types.RuntimeConfig = object
    modules["dt_arena.src.types.trajectory"].Trajectory = object
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    module_name = "_secpo_dtap_adapter_contract_test"
    path = ROOT / "src/secpo/evaluation/dtap_qwen_agent_adapter.py"
    specification = importlib.util.spec_from_file_location(module_name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    monkeypatch.setitem(sys.modules, module_name, module)
    specification.loader.exec_module(module)
    return module


def test_adapter_transport_preserves_all_trust_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _load_adapter_with_dependency_stubs(monkeypatch)
    messages = [
        adapter.qwen_message("instruction", adapter.Trust.TRUSTED),
        adapter.qwen_message("injection", adapter.Trust.UNTRUSTED),
        adapter.qwen_message("clean tool", adapter.Trust.TRUSTED_TOOL),
        adapter.qwen_message("hostile tool", adapter.Trust.UNTRUSTED_TOOL),
    ]
    transport = object.__new__(adapter.DTapInputRoleOAI)
    converted = transport.convert_messages_to_dicts(messages)
    assert converted == [
        {"role": "user", "content": "instruction"},
        {"role": "user", "content": "injection", "name": "dtap_untrusted"},
        {
            "role": "user",
            "content": "clean tool",
            "name": "dtap_trusted_tool_result",
        },
        {
            "role": "user",
            "content": "hostile tool",
            "name": "dtap_untrusted_tool_result",
        },
    ]
    missing_trust = adapter.Message(role="user", content="implicit", extra={})
    with pytest.raises(ValueError, match="explicit trust"):
        transport.convert_messages_to_dicts([missing_trust])


def test_adapter_dynamic_tool_preserves_complete_mcp_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _load_adapter_with_dependency_stubs(monkeypatch)
    schema = {
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/value"}},
        "$defs": {"value": {"type": "string"}},
        "additionalProperties": False,
    }
    tool = adapter.DTapFunctionTool(
        name="audited_tool",
        description="description",
        parameters=schema,
        dispatcher=lambda params, **kwargs: (params, kwargs),
    )
    assert tool.parameters == schema
    assert tool.parameters is not schema
