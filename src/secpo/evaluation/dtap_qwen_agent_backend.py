"""Concrete task backend for the pinned DTap Qwen-Agent experiment.

The portable 33-by-6 launcher calls :func:`run_task` once per selected task.
This module resolves the task in an already verified, patched DTap checkout,
runs DTap's complete setup/MCP/agent/judge lifecycle in a child process, and
returns only the two boolean paper metrics. Runtime paths and the model-service
address are required environment configuration and are never written into the
SecPO result manifests.
"""

from __future__ import annotations

from collections.abc import Mapping
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib.parse import urlsplit

from secpo.subprocess_env import build_subprocess_environment
from secpo.evaluation.dtap_qwen36 import (
    DTAP_COMMIT,
    DTAP_PATCH_SHA256,
    DTAP_QWEN_AGENT_ADAPTER_SHA256,
    MAX_CONTEXT_TOKENS,
    MAX_OUTPUT_TOKENS,
    ROLE_MAP,
    RUNTIME_ROOT_ENVIRONMENT,
    SCHEMA_VERSION,
    SHARD_COUNT,
    SHARD_ID_ENVIRONMENT,
    SHARD_RUNTIME_ROOT_ENVIRONMENT,
    canonical_bytes,
    file_sha256,
    fingerprint,
    port_namespace_contract,
    shard_port_namespace,
)


AGENT_TYPE = "qwen_agent_input"
MAX_JUDGE_RESULT_BYTES = 16 * 1024 * 1024
MAX_TASK_SESSION_FILE_BYTES = 4 * 1024
MAX_ENVIRONMENT_CLEANUP_FILE_BYTES = 4 * 1024
TASK_SESSION_SCHEMA = "dtap-task-session-v1"
TASK_SESSION_FILE_ENVIRONMENT = "DTAP_TASK_SESSION_FILE"
ENVIRONMENT_CLEANUP_SCHEMA = "dtap-environment-cleanup-v1"
ENVIRONMENT_CLEANUP_FILE_ENVIRONMENT = "DTAP_ENVIRONMENT_CLEANUP_FILE"
PROCESS_GROUP_TERMINATION_GRACE_SECONDS = 90.0
PROCESS_GROUP_KILL_GRACE_SECONDS = 1.0
JUDGE_TIMEOUT_SECONDS = 60
DTAP_PASS_ENV_SELECTION = "SECPODTAP_PASS_ENV"
DTAP_ALLOWED_PASS_ENV = ("DTAP_QWEN_API_KEY",)

_REQUIRED_ENVIRONMENT = (
    "SECPODTAP_CHECKOUT",
    "SECPODTAP_RESULTS_ROOT",
    "SECPODTAP_MODEL",
    "SECPODTAP_MODEL_IDENTITY_SHA256",
    "SECPODTAP_PLAYWRIGHT_BROWSERS_PATH",
    "SECPODTAP_JUDGE_BASE_URL",
    "SECPODTAP_JUDGE_MODEL",
    "SECPODTAP_JUDGE_API_KEY",
    "SECPODTAP_JUDGE_TIMEOUT_SECONDS",
    "SECPODTAP_JUDGE_IDENTITY_SHA256",
    RUNTIME_ROOT_ENVIRONMENT,
    SHARD_ID_ENVIRONMENT,
    SHARD_RUNTIME_ROOT_ENVIRONMENT,
    "DT_PORT_DEFAULT_OFFSET",
    "DT_PORT_RANGE",
    "DT_PORT_LOCK_DIR",
    "DTAP_QWEN_BASE_URL",
)

# These are the model-facing and lifecycle files produced by the reviewed
# patch. The full-tree verifier remains the mandatory preflight; checking the
# critical subset here prevents a worker from silently running an unpatched or
# partially patched checkout after that preflight.
_CRITICAL_CHECKOUT_FILES = {
    "agent/__init__.py": (
        "320b30f12ab956c36903c842dc919d2abb314e3316eb91b9993a7e56fccabfc6"
    ),
    "agent/qwen_input/__init__.py": (
        "3d52897fe4d171855d8303800e65460f437f6ad7da4a6424ccb29c9fcb5c95cd"
    ),
    "agent/qwen_input/src/__init__.py": (
        "f4c48764355bddb223581311ab2333d64863036beed1e0e026efa543798a5362"
    ),
    "agent/qwen_input/src/agent.py": (
        "108c72777d0bc3a6e356b916d1b96f96a2fd5a7574d38f8e3f3b316eee3a2c74"
    ),
    "dt_arena/config/env.yaml": (
        "324aac1ac173979a64908801965dd295fd922a6b8feddc9774df91d8ecec6dab"
    ),
    "dt_arena/config/injection_mcp.yaml": (
        "6d7e2f70fd1df721199567a274ba1d7e54471be3c2a4ffa39b98c56b8aebbbed"
    ),
    "dt_arena/config/mcp.yaml": (
        "3d998dafe767dbd222c4e188b49a569e9782b88ebc1d3df21e9fbf3c586c2f5f"
    ),
    "dt_arena/mcp_server/finance/main.py": (
        "537e14943fe8faaff9bd70ab8142a06c12a9f6f490789c60a25c20c6950b3be1"
    ),
    "dt_arena/utils/research/helpers.py": (
        "b75976b89ee69936830b084fcb5a6806141a02b7b3098b56cf626125952f909a"
    ),
    "eval/evaluation.py": (
        "5f48646f855686bf051120e5ce5358575f4ae55d9faa873b0e483c54f753e61c"
    ),
    "eval/task_runner.py": (
        "d52af887de249985460e1e3dc5ff011d288030c4217a40cad32a3b2f86450c1f"
    ),
    "utils/a2a_helpers.py": (
        "ca931ca28f5a76f2e31541d604da7fa1aa4c43b8d442253ec73f74333c599c09"
    ),
    "utils/agent_helpers.py": (
        "154777dcdd223fcfdd00b7649fe4052f877e4d8fc90b0e161841667745b33505"
    ),
    "utils/crm_quote_seed_overlay.sql": (
        "9c1e89b5474b9c607b56a43dc64f5cc2b2f91111c71fd18b283538ab88586423"
    ),
    "utils/env_helpers.py": (
        "7c3f98d877877e7f9d8e0644a5f8641dd14470b201a561a9005977ca900792b1"
    ),
    "utils/injection_helpers.py": (
        "be9e7d3c65f73c148aff828de5ebfddf53176008648f081694a64e6734fc804f"
    ),
    "utils/injection_mcp_helpers.py": (
        "396b73b3ebf23f60c4bdd504c88f6cc47991d8f0f236df318e08c30db71dd38b"
    ),
    "utils/judge_helpers.py": (
        "aa943f44cfd8ed8a5613ac4aed16b27fa91bf2da423fd486e489230c956ca9bb"
    ),
    "utils/mcp_helpers.py": (
        "9eee452288a1cab49cf4ecad464b856fbf2c36ee296555a12b77af04a05b128a"
    ),
    "utils/mcp_manager.py": (
        "055b63f04eafe525ad904a964713d0a857fd6c40c6d213deab3d0f6871d03de5"
    ),
    "utils/resource_manager.py": (
        "f332c74c5d54c3309195eed5b48371fc2008d78e69b7d0115fad271fb9b12e05"
    ),
    "utils/task_executor.py": (
        "fa5ea9a1d5d9739189f2a44366203e84d1a4ebb19e29f3cafae9bd0d4594c1cb"
    ),
    "utils/template_helpers.py": (
        "a588749058b819e6bf967f6f9a4e8bd140e9a7298204cd31aa9339758e932109"
    ),
}


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required runtime environment variable is missing: {name}")
    return value


def _explicit_child_pass_env() -> tuple[str, ...]:
    """Resolve the narrowly supported, opt-in victim credential passthrough."""

    raw = os.getenv(DTAP_PASS_ENV_SELECTION, "")
    if not raw:
        return ()
    names = tuple(raw.split(","))
    if any(not name or name.strip() != name for name in names):
        raise ValueError(f"{DTAP_PASS_ENV_SELECTION} is malformed")
    if len(set(names)) != len(names):
        raise ValueError(f"{DTAP_PASS_ENV_SELECTION} contains duplicates")
    unsupported = sorted(set(names) - set(DTAP_ALLOWED_PASS_ENV))
    if unsupported:
        raise ValueError(
            f"{DTAP_PASS_ENV_SELECTION} contains unsupported names: {unsupported}"
        )
    for name in names:
        _required_environment(name)
    return names


def _validated_model_base_url() -> str:
    """Return the explicit victim endpoint after rejecting URL-side secrets."""

    value = _required_environment("DTAP_QWEN_BASE_URL")
    try:
        parsed = urlsplit(value)
        # Accessing ``port`` also rejects malformed or out-of-range ports.
        parsed.port
    except ValueError as error:
        raise ValueError("DTAP_QWEN_BASE_URL is not a valid URL") from error
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("DTAP_QWEN_BASE_URL must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "DTAP_QWEN_BASE_URL must not contain credentials, query, or fragment"
        )
    if any(character.isspace() or ord(character) < 32 for character in value):
        raise ValueError("DTAP_QWEN_BASE_URL must not contain whitespace or controls")
    return value


def _safe_component(value: Any, label: str) -> str:
    text = str(value)
    if not text or text in {".", ".."} or Path(text).name != text:
        raise ValueError(f"unsafe DTap task {label}")
    return text


def _validated_task(request: Mapping[str, Any]) -> dict[str, str]:
    expected_request = {"schema_version", "task", "task_fingerprint", "protocol"}
    if set(request) != expected_request or request.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported DTap backend request")
    task = request.get("task")
    if not isinstance(task, Mapping):
        raise ValueError("DTap backend task must be an object")
    expected_task = {"domain", "type", "task_id", "threat_model", "risk_category"}
    if set(task) != expected_task:
        raise ValueError("DTap backend task fields differ")
    normalized = {key: _safe_component(task[key], key) for key in expected_task}
    if normalized["type"] != "malicious" or normalized["threat_model"] != "indirect":
        raise ValueError("DTap backend accepts only malicious indirect tasks")
    if request.get("task_fingerprint") != fingerprint(normalized):
        raise ValueError("DTap backend task fingerprint differs")

    protocol = request.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("DTap backend protocol must be an object")
    external = protocol.get("external_sources")
    benchmark = protocol.get("benchmark")
    generation = protocol.get("generation")
    if not all(isinstance(value, Mapping) for value in (external, benchmark, generation)):
        raise ValueError("DTap backend protocol sections are missing")
    if external.get("dtap_commit") != DTAP_COMMIT:
        raise ValueError("DTap backend commit contract differs")
    if external.get("dtap_patch_sha256") != DTAP_PATCH_SHA256:
        raise ValueError("DTap backend patch contract differs")
    if benchmark.get("roles") != ROLE_MAP:
        raise ValueError("DTap backend trust-role contract differs")
    if benchmark.get("port_namespace") != port_namespace_contract():
        raise ValueError("DTap backend port-namespace contract differs")
    if generation.get("enable_thinking") is not False:
        raise ValueError("DTap backend requires thinking off")
    if generation.get("max_context_tokens") != MAX_CONTEXT_TOKENS:
        raise ValueError("DTap backend context contract differs")
    if generation.get("max_output_tokens") != MAX_OUTPUT_TOKENS:
        raise ValueError("DTap backend output contract differs")
    return normalized


def _verify_checkout(checkout: Path) -> None:
    if not checkout.is_dir() or not (checkout / ".git").exists():
        raise RuntimeError("SECPODTAP_CHECKOUT is not a Git checkout")
    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        env=build_subprocess_environment(),
    ).stdout.strip()
    if head != DTAP_COMMIT:
        raise RuntimeError("DTap runtime checkout commit differs")
    for relative, expected in _CRITICAL_CHECKOUT_FILES.items():
        path = checkout / relative
        if not path.is_file() or path.is_symlink() or file_sha256(path) != expected:
            raise RuntimeError(f"DTap critical runtime file differs: {relative}")


def _verify_release_adapter() -> None:
    path = Path(__file__).with_name("dtap_qwen_agent_adapter.py")
    if file_sha256(path) != DTAP_QWEN_AGENT_ADAPTER_SHA256:
        raise RuntimeError("released DTap Qwen-Agent adapter checksum differs")


def _verify_task_payload(checkout: Path, dataset_path: Path) -> None:
    """Require the selected task payload to remain pristine at the pinned commit."""
    relative = Path("dataset") / dataset_path
    tracked_config = subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "ls-files",
            "--error-unmatch",
            "--",
            str(relative / "config.yaml"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=build_subprocess_environment(),
    )
    if tracked_config.returncode != 0:
        raise RuntimeError("selected DTap task config is not tracked at the pinned commit")
    modified = subprocess.run(
        ["git", "-C", str(checkout), "diff", "--quiet", DTAP_COMMIT, "--", str(relative)],
        check=False,
        env=build_subprocess_environment(),
    )
    if modified.returncode != 0:
        raise RuntimeError("selected DTap task payload differs from the pinned commit")
    untracked = subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            str(relative),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=build_subprocess_environment(),
    )
    if untracked.stdout.strip():
        raise RuntimeError("selected DTap task payload contains untracked files")


def _dataset_path(task: Mapping[str, str]) -> Path:
    return (
        Path(task["domain"])
        / task["type"]
        / task["threat_model"]
        / task["risk_category"]
        / task["task_id"]
    )


def _safe_model_name(model: str) -> str:
    safe = model.replace("/", "_").replace(":", "_")
    if not safe or safe in {".", ".."} or Path(safe).name != safe:
        raise ValueError("SECPODTAP_MODEL does not map to a safe result component")
    return safe


def _validated_shard_runtime() -> tuple[int, Path]:
    raw_shard = _required_environment(SHARD_ID_ENVIRONMENT)
    if not raw_shard.isdecimal() or str(int(raw_shard)) != raw_shard:
        raise ValueError(f"{SHARD_ID_ENVIRONMENT} must be a canonical integer")
    shard_id = int(raw_shard)
    if not 0 <= shard_id < SHARD_COUNT:
        raise ValueError(f"{SHARD_ID_ENVIRONMENT} is outside the pinned topology")
    namespace = shard_port_namespace(shard_id)
    expected_port_environment = {
        "DT_PORT_DEFAULT_OFFSET": str(namespace["default_offset"]),
        "DT_PORT_RANGE": (
            f"{namespace['fallback_start']}-{namespace['fallback_end']}"
        ),
    }
    for name, expected in expected_port_environment.items():
        if _required_environment(name) != expected:
            raise RuntimeError(f"DTap shard port namespace differs: {name}")

    root = Path(_required_environment(RUNTIME_ROOT_ENVIRONMENT))
    shard_root = Path(_required_environment(SHARD_RUNTIME_ROOT_ENVIRONMENT))
    lock_root = Path(_required_environment("DT_PORT_LOCK_DIR"))
    for label, path in (
        (RUNTIME_ROOT_ENVIRONMENT, root),
        (SHARD_RUNTIME_ROOT_ENVIRONMENT, shard_root),
        ("DT_PORT_LOCK_DIR", lock_root),
    ):
        if not path.is_absolute() or path.is_symlink() or not path.is_dir():
            raise RuntimeError(f"{label} is not a direct absolute directory")
    root = root.resolve(strict=True)
    shard_root = shard_root.resolve(strict=True)
    lock_root = lock_root.resolve(strict=True)
    if shard_root != root / f"shard_{shard_id:02d}":
        raise RuntimeError("DTap shard runtime root differs")
    if lock_root != shard_root / "port-locks":
        raise RuntimeError("DTap shard port-lock directory differs")
    return shard_id, shard_root


def _private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"DTap task runtime path is not a direct directory: {path}")
    return path.resolve(strict=True)


def _atomic_runtime_write(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _task_runtime(
    environment: dict[str, str],
    *,
    shard_id: int,
    shard_root: Path,
    task: Mapping[str, str],
) -> Path:
    tasks_root = _private_directory(shard_root / "tasks")
    shared_root = _private_directory(shard_root / "shared")
    shared = {
        "cache": _private_directory(shared_root / "cache"),
        "pycache": _private_directory(shared_root / "pycache"),
    }
    task_hash = fingerprint(task)
    task_root = Path(
        tempfile.mkdtemp(prefix=f"task-{task_hash[:12]}-", dir=tasks_root)
    ).resolve(strict=True)
    directories = {
        "home": _private_directory(task_root / "home"),
        "tmp": _private_directory(task_root / "tmp"),
        "xdg_runtime": _private_directory(task_root / "xdg-runtime"),
        "xdg_config": _private_directory(task_root / "xdg-config"),
        "xdg_data": _private_directory(task_root / "xdg-data"),
        "xdg_state": _private_directory(task_root / "xdg-state"),
        "mcp_logs": _private_directory(task_root / "mcp-logs"),
    }
    runtime_tag = hashlib.sha256(str(shard_root.parent).encode()).hexdigest()[:12]
    attempt_tag = hashlib.sha256(str(task_root).encode()).hexdigest()[:10]
    prefix = (
        f"secpo-{runtime_tag}-s{shard_id:02d}-"
        f"t{task_hash[:10]}-a{attempt_tag}"
    )
    environment.update(
        {
            "TMPDIR": str(directories["tmp"]),
            "HOME": str(directories["home"]),
            "XDG_CACHE_HOME": str(shared["cache"]),
            "UV_CACHE_DIR": str(shared["cache"] / "uv"),
            "HF_HOME": str(shared["cache"] / "huggingface"),
            "PYTHONPYCACHEPREFIX": str(shared["pycache"]),
            "XDG_RUNTIME_DIR": str(directories["xdg_runtime"]),
            "XDG_CONFIG_HOME": str(directories["xdg_config"]),
            "XDG_DATA_HOME": str(directories["xdg_data"]),
            "XDG_STATE_HOME": str(directories["xdg_state"]),
            "DT_POOL_PREFIX": prefix,
            "COMPOSE_PROJECT_NAME": prefix,
            "DTAP_MCP_LOG_DIR": str(directories["mcp_logs"]),
            "DTAP_MCP_LOG_SCOPE": task_hash,
            TASK_SESSION_FILE_ENVIRONMENT: str(task_root / "task-session.json"),
            ENVIRONMENT_CLEANUP_FILE_ENVIRONMENT: str(
                task_root / "environment-cleanup.json"
            ),
        }
    )
    task_list = task_root / "task.jsonl"
    _atomic_runtime_write(task_list, canonical_bytes(task) + b"\n")
    return task_list


def _load_judge_result(path: Path) -> dict[str, bool]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_JUDGE_RESULT_BYTES:
            raise RuntimeError("DTap judge result is not a bounded regular file")
        encoded = os.read(descriptor, MAX_JUDGE_RESULT_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(encoded) > MAX_JUDGE_RESULT_BYTES:
        raise RuntimeError("DTap judge result exceeds the size limit")
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("DTap judge result is malformed") from exc
    expected_fields = {
        "task_success",
        "task_message",
        "task_metadata",
        "attack_success",
        "attack_message",
        "attack_metadata",
        "error",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise RuntimeError("DTap judge result schema differs")
    if value.get("error") is not None:
        raise RuntimeError("DTap judge reported an evaluation error")
    task_success = value.get("task_success")
    attack_success = value.get("attack_success")
    if type(task_success) is not bool or type(attack_success) is not bool:
        raise RuntimeError("DTap judge result lacks explicit boolean scores")
    if not isinstance(value.get("task_message"), str) or not isinstance(
        value.get("attack_message"), str
    ):
        raise RuntimeError("DTap judge result messages have invalid types")
    if not isinstance(value.get("task_metadata"), Mapping) or not isinstance(
        value.get("attack_metadata"), Mapping
    ):
        raise RuntimeError("DTap judge result metadata has invalid types")
    return {"task_success": task_success, "attack_success": attack_success}


def _judge_file_identity(path: Path) -> tuple[int, int, int, str] | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise RuntimeError("DTap judge result path is not a regular file")
    return (info.st_ino, info.st_mtime_ns, info.st_size, file_sha256(path))


def backend_runtime_identity() -> Mapping[str, str]:
    """Return the checkpoint identity bound into outer resume state."""

    model_digest = _required_environment("SECPODTAP_MODEL_IDENTITY_SHA256").lower()
    if len(model_digest) != 64 or any(
        character not in "0123456789abcdef" for character in model_digest
    ):
        raise ValueError("SECPODTAP_MODEL_IDENTITY_SHA256 must be lowercase SHA-256")
    judge = _validated_judge_runtime()
    return {
        "model_identity_sha256": model_digest,
        "judge_identity_sha256": judge["identity_sha256"],
    }


def _validated_judge_runtime() -> dict[str, str]:
    """Validate the provider-neutral external judge contract without persisting it."""

    base_url = _required_environment("SECPODTAP_JUDGE_BASE_URL").rstrip("/")
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("SECPODTAP_JUDGE_BASE_URL must be a credential-free HTTP URL")
    model = _required_environment("SECPODTAP_JUDGE_MODEL")
    if len(model) > 512 or any(character in model for character in "\r\n"):
        raise ValueError("SECPODTAP_JUDGE_MODEL is invalid")
    _required_environment("SECPODTAP_JUDGE_API_KEY")
    timeout = _required_environment("SECPODTAP_JUDGE_TIMEOUT_SECONDS")
    if not timeout.isdecimal() or str(int(timeout)) != timeout:
        raise ValueError("SECPODTAP_JUDGE_TIMEOUT_SECONDS must be a canonical integer")
    if not 0 < int(timeout) <= 3600:
        raise ValueError("SECPODTAP_JUDGE_TIMEOUT_SECONDS must be in (0, 3600]")
    identity = hashlib.sha256(
        canonical_bytes(
            {
                "base_url": base_url,
                "model": model,
                "timeout_seconds": int(timeout),
            }
        )
    ).hexdigest()
    configured_identity = _required_environment(
        "SECPODTAP_JUDGE_IDENTITY_SHA256"
    ).lower()
    if configured_identity != identity:
        raise ValueError(
            "SECPODTAP_JUDGE_IDENTITY_SHA256 does not match the judge runtime"
        )
    return {
        "base_url": base_url,
        "model": model,
        "timeout_seconds": timeout,
        "identity_sha256": identity,
    }


def _run_task_process(
    command: list[str],
    *,
    checkout: Path,
    environment: Mapping[str, str],
    timeout: float,
) -> int:
    """Run DTap in its own process group and clean up that group on exit."""

    process = subprocess.Popen(
        command,
        cwd=checkout,
        env=dict(environment),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            _terminate_task_process(
                process,
                initial_signal=signal.SIGTERM,
                grace=PROCESS_GROUP_TERMINATION_GRACE_SECONDS,
            )
        finally:
            try:
                _terminate_evaluation_session(process.pid)
            finally:
                try:
                    _terminate_registered_task_session(environment, process.pid)
                finally:
                    _cleanup_compose_resources(environment, process.pid)
        raise
    except BaseException:
        try:
            _terminate_task_process(
                process,
                initial_signal=signal.SIGKILL,
                grace=PROCESS_GROUP_KILL_GRACE_SECONDS,
            )
        finally:
            try:
                _terminate_evaluation_session(process.pid)
            finally:
                try:
                    _terminate_registered_task_session(environment, process.pid)
                finally:
                    _cleanup_compose_resources(environment, process.pid)
        raise
    try:
        _terminate_task_process(
            process,
            initial_signal=signal.SIGTERM,
            grace=PROCESS_GROUP_TERMINATION_GRACE_SECONDS,
        )
    finally:
        try:
            _terminate_evaluation_session(process.pid)
        finally:
            try:
                _terminate_registered_task_session(environment, process.pid)
            finally:
                _cleanup_compose_resources(environment, process.pid)
    return returncode


def _docker_base_command() -> list[str]:
    candidates = (["docker"], ["sudo", "-n", "docker"])
    for candidate in candidates:
        try:
            completed = subprocess.run(
                [*candidate, "info", "--format", "{{.ServerVersion}}"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
                env=build_subprocess_environment(),
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if completed.returncode == 0:
            return list(candidate)
    raise RuntimeError("cannot access Docker for timed-out DTap cleanup")


def _docker_output(base: list[str], arguments: list[str]) -> list[str]:
    completed = subprocess.run(
        [*base, *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env=build_subprocess_environment(),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Docker cleanup command failed: " + " ".join(arguments[:3])
        )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _compose_projects(base: list[str], prefix: str) -> set[str]:
    label_format = "{{.Labels}}"
    commands = (
        ["ps", "-a", "--filter", "label=com.docker.compose.project", "--format", label_format],
        ["network", "ls", "--filter", "label=com.docker.compose.project", "--format", label_format],
        ["volume", "ls", "--filter", "label=com.docker.compose.project", "--format", label_format],
    )
    project_label = "com.docker.compose.project="
    projects = {
        assignment.removeprefix(project_label)
        for command in commands
        for labels in _docker_output(base, command)
        for assignment in labels.split(",")
        if assignment == project_label + prefix
        or assignment.startswith(project_label + prefix + "_")
    }
    return projects


def _read_environment_cleanup_marker(path: Path, evaluation_pid: int) -> bool:
    """Trust only the exact marker written after TaskExecutor shutdown."""

    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
    except (FileNotFoundError, OSError):
        return False
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_size > MAX_ENVIRONMENT_CLEANUP_FILE_BYTES
        ):
            return False
        encoded = os.read(descriptor, MAX_ENVIRONMENT_CLEANUP_FILE_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(encoded) > MAX_ENVIRONMENT_CLEANUP_FILE_BYTES:
        return False
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(value, Mapping)
        and set(value) == {"schema_version", "evaluation_pid"}
        and value.get("schema_version") == ENVIRONMENT_CLEANUP_SCHEMA
        and type(value.get("evaluation_pid")) is int
        and value.get("evaluation_pid") == evaluation_pid
    )


def _docker_cleanup_lock(environment: Mapping[str, str]) -> int:
    """Serialize fallback Docker scans across every shard in one run."""

    raw_root = environment.get(RUNTIME_ROOT_ENVIRONMENT, "").strip()
    root = Path(raw_root)
    if not raw_root or not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise RuntimeError("DTap runtime root differs during Docker cleanup")
    lock_path = root / "docker-cleanup.lock"
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError("DTap Docker-cleanup lock is not a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _cleanup_compose_resources(
    environment: Mapping[str, str], evaluation_pid: int | None = None
) -> None:
    """Remove only Compose resources bearing this task attempt's prefix."""
    prefix = environment.get("DT_POOL_PREFIX", "").strip()
    if not prefix:
        return
    if (
        len(prefix) > 96
        or not prefix.startswith("secpo-")
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in prefix)
    ):
        raise RuntimeError("DTap Compose cleanup prefix differs")
    cleanup_path_raw = environment.get(
        ENVIRONMENT_CLEANUP_FILE_ENVIRONMENT, ""
    ).strip()
    if evaluation_pid is not None and cleanup_path_raw:
        cleanup_path = Path(cleanup_path_raw)
        if (
            cleanup_path.is_absolute()
            and not cleanup_path.is_symlink()
            and _read_environment_cleanup_marker(cleanup_path, evaluation_pid)
        ):
            cleanup_path.unlink(missing_ok=True)
            return

    lock_descriptor = _docker_cleanup_lock(environment)
    try:
        base = _docker_base_command()
        projects = _compose_projects(base, prefix)
        if not projects:
            return
        resource_commands = (
            (["ps", "-aq"], ["rm", "-f", "-v"]),
            (["network", "ls", "-q"], ["network", "rm"]),
            (["volume", "ls", "-q"], ["volume", "rm", "-f"]),
        )
        for project in sorted(projects):
            label = f"label=com.docker.compose.project={project}"
            for query, remove in resource_commands:
                identifiers = _docker_output(base, [*query, "--filter", label])
                if identifiers:
                    for attempt in range(3):
                        try:
                            _docker_output(base, [*remove, *identifiers])
                            break
                        except RuntimeError:
                            if attempt == 2:
                                raise
                            time.sleep(0.2)
        remaining = _compose_projects(base, prefix)
        if remaining:
            raise RuntimeError(
                "DTap Compose resources remained after task cleanup: "
                + ", ".join(sorted(remaining))
            )
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)


def _task_session_members(task_session: int) -> list[tuple[int, int]]:
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        raise RuntimeError("DTap task-session cleanup requires Linux /proc")
    members: list[tuple[int, int]] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal():
            continue
        pid = int(entry.name)
        try:
            if os.getsid(pid) == task_session:
                members.append((pid, os.getpgid(pid)))
        except (ProcessLookupError, PermissionError):
            continue
    return members


def _read_task_session(path: Path, evaluation_pid: int) -> int | None:
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_TASK_SESSION_FILE_BYTES:
            raise RuntimeError("DTap task-session registry is not a bounded file")
        encoded = os.read(descriptor, MAX_TASK_SESSION_FILE_BYTES + 1)
    finally:
        os.close(descriptor)
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("DTap task-session registry is malformed") from exc
    expected_keys = {"schema_version", "evaluation_pid", "task_session"}
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise RuntimeError("DTap task-session registry schema differs")
    task_session = value.get("task_session")
    if (
        value.get("schema_version") != TASK_SESSION_SCHEMA
        or value.get("evaluation_pid") != evaluation_pid
        or type(task_session) is not int
        or task_session <= 1
        or task_session == os.getsid(0)
    ):
        raise RuntimeError("DTap task-session registry identity differs")
    return task_session


def _signal_task_session(members: list[tuple[int, int]], signum: int) -> None:
    for process_group in sorted({group for _pid, group in members}):
        try:
            os.killpg(process_group, signum)
        except ProcessLookupError:
            pass
    for pid, _process_group in members:
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass


def _wait_for_task_session_exit(task_session: int, timeout: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout)
    while _task_session_members(task_session):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.05, remaining))
    return True


def _terminate_evaluation_session(evaluation_pid: int) -> None:
    """Terminate private Compose CLI groups that remain in evaluation's SID."""

    if evaluation_pid <= 1 or evaluation_pid == os.getsid(0):
        raise RuntimeError("refusing unsafe DTap evaluation session")
    members = _task_session_members(evaluation_pid)
    _signal_task_session(members, signal.SIGTERM)
    if not _wait_for_task_session_exit(
        evaluation_pid, PROCESS_GROUP_KILL_GRACE_SECONDS
    ):
        _signal_task_session(
            _task_session_members(evaluation_pid), signal.SIGKILL
        )
    if not _wait_for_task_session_exit(
        evaluation_pid, PROCESS_GROUP_KILL_GRACE_SECONDS
    ):
        raise RuntimeError("DTap evaluation session remained live after SIGKILL")


def _terminate_registered_task_session(
    environment: Mapping[str, str], evaluation_pid: int
) -> None:
    raw_path = environment.get(TASK_SESSION_FILE_ENVIRONMENT, "").strip()
    if not raw_path:
        return
    path = Path(raw_path)
    if not path.is_absolute() or path.is_symlink():
        raise RuntimeError("DTap task-session registry path differs")
    task_session = _read_task_session(path, evaluation_pid)
    if task_session is None:
        return
    members = _task_session_members(task_session)
    _signal_task_session(members, signal.SIGTERM)
    if not _wait_for_task_session_exit(
        task_session, PROCESS_GROUP_KILL_GRACE_SECONDS
    ):
        _signal_task_session(_task_session_members(task_session), signal.SIGKILL)
    if not _wait_for_task_session_exit(
        task_session, PROCESS_GROUP_KILL_GRACE_SECONDS
    ):
        raise RuntimeError("DTap task session remained live after SIGKILL")
    path.unlink(missing_ok=True)


def _signal_process_group(process_group: int, selected_signal: signal.Signals) -> bool:
    try:
        os.killpg(process_group, selected_signal)
    except ProcessLookupError:
        return False
    return True


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(process_group: int, timeout: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout)
    while _process_group_exists(process_group):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.05, remaining))
    return True


def _terminate_task_process(
    process: subprocess.Popen[bytes],
    *,
    initial_signal: signal.Signals,
    grace: float,
) -> None:
    """Reap the task-runner leader and terminate every same-group descendant."""

    deadline = time.monotonic() + grace
    _signal_process_group(process.pid, initial_signal)
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        _signal_process_group(process.pid, signal.SIGKILL)
        process.wait()
        _wait_for_process_group_exit(
            process.pid, PROCESS_GROUP_KILL_GRACE_SECONDS
        )
        return
    remaining = max(0.0, deadline - time.monotonic())
    if not _wait_for_process_group_exit(process.pid, remaining):
        _signal_process_group(process.pid, signal.SIGKILL)
        _wait_for_process_group_exit(
            process.pid, PROCESS_GROUP_KILL_GRACE_SECONDS
        )


def run_task(request: Mapping[str, Any]) -> Mapping[str, bool]:
    """Run one complete DTap task and return its fail-closed judge scores."""

    task = _validated_task(request)
    for name in _REQUIRED_ENVIRONMENT:
        _required_environment(name)
    checkout = Path(_required_environment("SECPODTAP_CHECKOUT")).resolve()
    results_root = Path(_required_environment("SECPODTAP_RESULTS_ROOT"))
    if not results_root.is_absolute():
        raise RuntimeError("SECPODTAP_RESULTS_ROOT must be absolute")
    model = _required_environment("SECPODTAP_MODEL")
    _verify_checkout(checkout)
    _verify_release_adapter()
    backend_runtime_identity()
    judge_runtime = _validated_judge_runtime()
    shard_id, shard_root = _validated_shard_runtime()
    model_base_url = _validated_model_base_url()
    child_pass_env = _explicit_child_pass_env()

    dataset_path = _dataset_path(task)
    task_dir = (checkout / "dataset" / dataset_path).resolve()
    dataset_root = (checkout / "dataset").resolve()
    if not task_dir.is_relative_to(dataset_root) or not (task_dir / "config.yaml").is_file():
        raise RuntimeError("selected DTap task directory is missing")
    _verify_task_payload(checkout, dataset_path)

    timeout = float(os.getenv("SECPODTAP_TASK_TIMEOUT_SECONDS", "14400"))
    if not 0 < timeout <= 86_400:
        raise ValueError("SECPODTAP_TASK_TIMEOUT_SECONDS must be in (0, 86400]")
    playwright_path = Path(
        _required_environment("SECPODTAP_PLAYWRIGHT_BROWSERS_PATH")
    )
    if (
        not playwright_path.is_absolute()
        or playwright_path.is_symlink()
        or not playwright_path.is_dir()
    ):
        raise RuntimeError(
            "SECPODTAP_PLAYWRIGHT_BROWSERS_PATH must be a direct absolute directory"
        )
    # Translate the reviewed parent configuration into exactly the names used
    # by DTap.  In particular, ambient provider credentials are never copied.
    environment = build_subprocess_environment(
        pass_env=child_pass_env,
        require_pass_env=True,
        overrides={
            RUNTIME_ROOT_ENVIRONMENT: _required_environment(
                RUNTIME_ROOT_ENVIRONMENT
            ),
            SHARD_ID_ENVIRONMENT: _required_environment(SHARD_ID_ENVIRONMENT),
            SHARD_RUNTIME_ROOT_ENVIRONMENT: _required_environment(
                SHARD_RUNTIME_ROOT_ENVIRONMENT
            ),
            "DT_PORT_DEFAULT_OFFSET": _required_environment(
                "DT_PORT_DEFAULT_OFFSET"
            ),
            "DT_PORT_RANGE": _required_environment("DT_PORT_RANGE"),
            "DT_PORT_LOCK_DIR": _required_environment("DT_PORT_LOCK_DIR"),
            "DTAP_QWEN_BASE_URL": model_base_url,
            "EVAL_RESULTS_ROOT": str(results_root),
            "DTAP_QWEN_ENABLE_THINKING": "0",
            "DTAP_QWEN_MAX_OUTPUT_TOKENS": str(MAX_OUTPUT_TOKENS),
            "DTAP_QWEN_TEMPERATURE": "1.0",
            "DTAP_QWEN_TOP_P": "0.95",
            "DTAP_QWEN_TOP_K": "20",
            "DTAP_QWEN_SEED": "0",
            "DTAP_QWEN_PRESENCE_PENALTY": "0",
            "DTAP_QWEN_TOOL_TIMEOUT": "30",
            "DTAP_QWEN_REQUEST_TIMEOUT": "1800",
            "OPENAI_API_KEY": _required_environment(
                "SECPODTAP_JUDGE_API_KEY"
            ),
            "OPENAI_BASE_URL": judge_runtime["base_url"],
            "OPENAI_MODEL": judge_runtime["model"],
            "OPENAI_TIMEOUT_SEC": judge_runtime["timeout_seconds"],
            "EVAL_TASK_TIMEOUT": f"{timeout * 0.9:.12g}",
            "DTAP_SKIP_DATASET_DOWNLOAD": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "UV_PYTHON_DOWNLOADS": "never",
            "UV_NO_SYNC": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "PLAYWRIGHT_BROWSERS_PATH": str(
                playwright_path.resolve(strict=True)
            ),
        },
        approved_sensitive_names=("OPENAI_API_KEY",),
    )
    task_list = _task_runtime(
        environment,
        shard_id=shard_id,
        shard_root=shard_root,
        task=task,
    )
    judge_path = (
        results_root
        / "benchmark"
        / AGENT_TYPE
        / _safe_model_name(model)
        / dataset_path
        / "judge_result.json"
    )
    previous_judge = _judge_file_identity(judge_path)
    returncode = _run_task_process(
        [
            sys.executable,
            "-m",
            "eval.evaluation",
            "--task-list",
            str(task_list),
            "--agent-type",
            AGENT_TYPE,
            "--model",
            model,
            "--max-parallel",
            "1",
            "--temperature",
            "1.0",
            "--port-range",
            environment["DT_PORT_RANGE"],
        ],
        checkout=checkout,
        environment=environment,
        timeout=timeout,
    )
    try:
        result = _load_judge_result(judge_path)
        current_judge = _judge_file_identity(judge_path)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError("DTap task process produced no valid judge result") from exc
    if current_judge == previous_judge:
        raise RuntimeError("DTap task process did not produce a fresh judge result")
    expected_returncode = 0 if result["task_success"] else 1
    if returncode != expected_returncode:
        raise RuntimeError("DTap task process return code disagrees with its judge result")
    return result


__all__ = [
    "AGENT_TYPE",
    "backend_runtime_identity",
    "run_task",
]
