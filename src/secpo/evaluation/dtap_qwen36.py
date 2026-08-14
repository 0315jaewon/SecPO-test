"""Reproducible DTap evaluation for the Qwen3.6-27B SecPO checkpoint.

The benchmark and Qwen-Agent remain pinned external dependencies. This module
owns the public experiment contract: checkout verification,
trusted/untrusted role separation, deterministic CPU sharding, strict resume,
and fail-closed metric aggregation. The companion
``dtap_qwen_agent_backend`` module supplies the reviewed external-DTap task
lifecycle and judge mapping.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from typing import Any

from secpo.subprocess_env import build_subprocess_environment


SCHEMA_VERSION = "secpo-dtap-qwen36-v1"
SHARD_SCHEMA_VERSION = "secpo-dtap-qwen36-shards-v1"
RESULT_SCHEMA_VERSION = "secpo-dtap-qwen36-result-v1"
RUN_SCHEMA_VERSION = "secpo-dtap-qwen36-run-v1"
SUMMARY_SCHEMA_VERSION = "secpo-dtap-qwen36-summary-v1"
LAUNCH_SCHEMA_VERSION = "secpo-dtap-qwen36-multiprocess-launch-v1"

DTAP_REPOSITORY = "https://github.com/AI-secure/DecodingTrust-Agent.git"
DTAP_COMMIT = "e0323a521ba4ef88f8e14c1eccf68d0a3d19a458"
DTAP_VERSION = "0.2.12"
DTAP_LICENSE = "Apache-2.0"
DTAP_LICENSE_SHA256 = (
    "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
)
DTAP_PATCH_SHA256 = (
    "c63e22b067719827741e4847932ce6629010509a28d0f0fb4ea8833485b3185d"
)
DTAP_QWEN_AGENT_ADAPTER_SHA256 = (
    "4d9293fdf8f0ed7d03a0e55a46de7a81db22b30f56da2501cb8486d27d21ac9b"
)
DTAP_TEMPLATE_SHA256 = (
    "5fe53e6107dcd645c9a73457a694b158f5fb4ab839927a56867ec9a2c127cd90"
)

QWEN_AGENT_REPOSITORY = "https://github.com/QwenLM/Qwen-Agent.git"
QWEN_AGENT_VERSION = "0.0.34"
QWEN_AGENT_LICENSE = "Apache-2.0"
QWEN_AGENT_LICENSE_SHA256 = (
    "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
)

MODEL_NAME = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
EVALUATION_CHECKPOINT_LEARNING_RATE = 1.2e-4
RELEASE_TRAINING_CONFIG_LEARNING_RATE = 1.6e-4

EXPECTED_TASKS = 1_838
DEFAULT_TASK_LIST_PATH = "data/dtap_qwen36_non_vm_tasks.jsonl"
EXPECTED_TASK_FILE_SHA256 = (
    "ca0f2fc5728491561e4645ed25390b6311fc9831ecb9b72660d5f513fd6d8098"
)
EXPECTED_TASK_SEMANTIC_SHA256 = (
    "8ec56c88f064329fe3ee2fdf5472e09c64448be0f0e7d4ae62a33d436d56d1f5"
)
EXCLUDED_DOMAINS = ("macos", "windows")
SHARD_COUNT = 33
MAX_PARALLEL_PER_SHARD = 6
AGGREGATE_PARALLELISM = SHARD_COUNT * MAX_PARALLEL_PER_SHARD
MAX_CONTEXT_TOKENS = 262_144
MAX_OUTPUT_TOKENS = 32_768
ROLE_MAP = {"trusted": "user", "untrusted": "input"}
PORT_DEFAULT_OFFSET_STRIDE = 381
PORT_FALLBACK_START = 30_000
PORT_FALLBACK_WIDTH = 1_000
PORT_NAMESPACE_POLICY = "disjoint_defaults_fallback_and_flock_v1"
RUNTIME_ROOT_ENVIRONMENT = "SECPODTAP_RUNTIME_ROOT"
SHARD_ID_ENVIRONMENT = "SECPODTAP_SHARD_ID"
SHARD_RUNTIME_ROOT_ENVIRONMENT = "SECPODTAP_SHARD_RUNTIME_ROOT"
PINNED_ENV_CONFIG_SHA256 = (
    "324aac1ac173979a64908801965dd295fd922a6b8feddc9774df91d8ecec6dab"
)
PINNED_CONFIGURED_DEFAULT_PORTS_SHA256 = (
    "2570ceb9f6cf85828a37f3afddbfb29b6d6e52fdfc96e4d1c900ae7729d92203"
)
PINNED_CONFIGURED_DEFAULT_PORTS = (
    1025, 2223, 3306, 3390, 5000, 5444, 5452, 5453, 5454, 5455,
    5460, 5462, 5464, 5468, 5470, 5472, 5473, 5474, 5475, 5476,
    5477, 5478, 5479, 5482, 5544, 7101, 7770, 7771, 8003, 8005,
    8006, 8007, 8023, 8025, 8026, 8027, 8028, 8029, 8030, 8031,
    8032, 8033, 8034, 8035, 8036, 8038, 8039, 8040, 8041, 8042,
    8044, 8045, 8046, 8050, 8054, 8055, 8056, 8057, 8058, 8059,
    8060, 8061, 8062, 8063, 8064, 8065, 8068, 8069, 8070, 8071,
    8080, 8081, 8453, 8454, 9050, 9060, 9897, 9898, 9899, 9997,
    9998, 9999, 10300, 12001, 12002, 12005, 17501,
)
_RUNTIME_ENVIRONMENT_LOCK = threading.RLock()

_TASK_KEYS = ("domain", "type", "task_id", "threat_model", "risk_category")
_CANONICAL_SORT_KEYS = (
    "domain",
    "type",
    "threat_model",
    "risk_category",
    "task_id",
)


class ShardExecutionError(RuntimeError):
    """Raised after all runnable tasks finish when any task is unresolved."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = dict(report)
        super().__init__(
            f"DTap shard {report['shard_id']} is incomplete: "
            f"{report['errors']} task error(s)"
        )


class MultiProcessExecutionError(RuntimeError):
    """Raised after all shard processes exit when the launch is incomplete."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = dict(report)
        super().__init__(
            "DTap multi-process launch is incomplete: "
            f"{report['incomplete_shards']} shard(s) unresolved"
        )


def canonical_bytes(value: Any) -> bytes:
    """Serialize JSON data in the one spelling used for fingerprints."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(path, canonical_bytes(value) + b"\n")


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {label}") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _require_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    observed = set(value)
    if observed != keys:
        raise ValueError(
            f"{label} fields differ; missing={sorted(keys - observed)}, "
            f"unexpected={sorted(observed - keys)}"
        )


def _object(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise ValueError(f"{key} must be a JSON object")
    return item


@dataclass(frozen=True)
class DTapQwen36Config:
    """Immutable public contract for the main DTap experiment."""

    model_name: str
    model_revision: str
    evaluation_checkpoint_learning_rate: float
    release_training_config_learning_rate: float
    task_list: str
    expected_tasks: int
    task_file_sha256: str
    task_semantic_sha256: str
    excluded_domains: tuple[str, ...]
    shard_count: int
    max_parallel_per_shard: int
    max_context_tokens: int
    max_output_tokens: int
    enable_thinking: bool

    def __post_init__(self) -> None:
        exact = (
            self.model_name == MODEL_NAME,
            self.model_revision == MODEL_REVISION,
            self.evaluation_checkpoint_learning_rate
            == EVALUATION_CHECKPOINT_LEARNING_RATE,
            self.release_training_config_learning_rate
            == RELEASE_TRAINING_CONFIG_LEARNING_RATE,
            self.task_list == DEFAULT_TASK_LIST_PATH,
            self.expected_tasks == EXPECTED_TASKS,
            self.task_file_sha256 == EXPECTED_TASK_FILE_SHA256,
            self.task_semantic_sha256 == EXPECTED_TASK_SEMANTIC_SHA256,
            self.excluded_domains == EXCLUDED_DOMAINS,
            self.shard_count == SHARD_COUNT,
            self.max_parallel_per_shard == MAX_PARALLEL_PER_SHARD,
            self.max_context_tokens == MAX_CONTEXT_TOKENS,
            self.max_output_tokens == MAX_OUTPUT_TOKENS,
            self.enable_thinking is False,
        )
        if not all(exact):
            raise ValueError("DTap main-experiment settings must remain pinned")

    @property
    def aggregate_parallelism(self) -> int:
        return self.shard_count * self.max_parallel_per_shard

    def protocol(self) -> dict[str, Any]:
        """Return the model-facing contract supplied to every task callable."""

        return {
            "external_sources": {
                "dtap_commit": DTAP_COMMIT,
                "dtap_license_sha256": DTAP_LICENSE_SHA256,
                "dtap_patch_sha256": DTAP_PATCH_SHA256,
                "dtap_qwen_agent_adapter_sha256": (
                    DTAP_QWEN_AGENT_ADAPTER_SHA256
                ),
                "dtap_template_sha256": DTAP_TEMPLATE_SHA256,
                "qwen_agent_version": QWEN_AGENT_VERSION,
                "qwen_agent_license_sha256": QWEN_AGENT_LICENSE_SHA256,
            },
            "model": {
                "name_or_path": self.model_name,
                "revision": self.model_revision,
                "evaluation_checkpoint_learning_rate": (
                    self.evaluation_checkpoint_learning_rate
                ),
                "release_training_config_learning_rate": (
                    self.release_training_config_learning_rate
                ),
            },
            "benchmark": {
                "split": "malicious_indirect_non_vm",
                "task_list": self.task_list,
                "expected_tasks": self.expected_tasks,
                "task_file_sha256": self.task_file_sha256,
                "task_semantic_sha256": self.task_semantic_sha256,
                "canonical_sort_keys": list(_CANONICAL_SORT_KEYS),
                "excluded_domains": list(self.excluded_domains),
                "roles": dict(ROLE_MAP),
                "shard_count": self.shard_count,
                "max_parallel_per_shard": self.max_parallel_per_shard,
                "aggregate_parallelism": self.aggregate_parallelism,
                "port_namespace": port_namespace_contract(),
            },
            "generation": {
                "enable_thinking": self.enable_thinking,
                "max_context_tokens": self.max_context_tokens,
                "max_output_tokens": self.max_output_tokens,
            },
        }


def _assert_pinned_external(value: Mapping[str, Any]) -> None:
    dtap = _object(value, "dtap")
    qwen_agent = _object(value, "qwen_agent")
    expected_dtap = {
        "repository": DTAP_REPOSITORY,
        "commit": DTAP_COMMIT,
        "version": DTAP_VERSION,
        "license": DTAP_LICENSE,
        "license_sha256": DTAP_LICENSE_SHA256,
        "patch_sha256": DTAP_PATCH_SHA256,
        "qwen_agent_adapter_sha256": DTAP_QWEN_AGENT_ADAPTER_SHA256,
        "template_sha256": DTAP_TEMPLATE_SHA256,
        "vendored": False,
    }
    expected_qwen = {
        "repository": QWEN_AGENT_REPOSITORY,
        "version": QWEN_AGENT_VERSION,
        "license": QWEN_AGENT_LICENSE,
        "license_sha256": QWEN_AGENT_LICENSE_SHA256,
        "vendored": False,
    }
    if canonical_bytes(dict(dtap)) != canonical_bytes(expected_dtap):
        raise ValueError("DTap external dependency must remain pinned")
    if canonical_bytes(dict(qwen_agent)) != canonical_bytes(expected_qwen):
        raise ValueError("Qwen-Agent external dependency must remain pinned")


def config_from_mapping(value: Mapping[str, Any]) -> DTapQwen36Config:
    _require_keys(
        value,
        {
            "schema_version",
            "status",
            "dtap",
            "qwen_agent",
            "model",
            "benchmark",
            "generation",
        },
        "configuration",
    )
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported DTap configuration schema")
    if value["status"] != "runnable_external_dtap_qwen_agent_backend":
        raise ValueError("DTap release status must remain explicit")
    _assert_pinned_external(value)

    model = _object(value, "model")
    benchmark = _object(value, "benchmark")
    generation = _object(value, "generation")
    _require_keys(
        model,
        {
            "name_or_path",
            "revision",
            "evaluation_checkpoint_learning_rate",
            "release_training_config_learning_rate",
        },
        "model",
    )
    _require_keys(
        benchmark,
        {
            "split",
            "task_list",
            "expected_tasks",
            "task_file_sha256",
            "task_semantic_sha256",
            "canonical_sort_keys",
            "excluded_domains",
            "roles",
            "shard_count",
            "max_parallel_per_shard",
            "aggregate_parallelism",
            "port_namespace",
        },
        "benchmark",
    )
    _require_keys(
        generation,
        {"enable_thinking", "max_context_tokens", "max_output_tokens"},
        "generation",
    )
    string_fields = (
        (model, "name_or_path"),
        (model, "revision"),
        (benchmark, "task_list"),
        (benchmark, "task_file_sha256"),
        (benchmark, "task_semantic_sha256"),
    )
    if any(type(section.get(key)) is not str for section, key in string_fields):
        raise ValueError("DTap pinned string fields have invalid JSON types")
    float_fields = (
        "evaluation_checkpoint_learning_rate",
        "release_training_config_learning_rate",
    )
    if any(type(model.get(key)) is not float for key in float_fields):
        raise ValueError("DTap pinned learning rates must be JSON numbers")
    integer_fields = (
        (benchmark, "expected_tasks"),
        (benchmark, "shard_count"),
        (benchmark, "max_parallel_per_shard"),
        (benchmark, "aggregate_parallelism"),
        (generation, "max_context_tokens"),
        (generation, "max_output_tokens"),
    )
    if any(type(section.get(key)) is not int for section, key in integer_fields):
        raise ValueError("DTap pinned integer fields have invalid JSON types")
    if type(generation.get("enable_thinking")) is not bool:
        raise ValueError("DTap enable_thinking must be a JSON boolean")
    if benchmark["split"] != "malicious_indirect_non_vm":
        raise ValueError("DTap split must remain the malicious indirect non-VM split")
    if benchmark["task_list"] != DEFAULT_TASK_LIST_PATH:
        raise ValueError("DTap bundled task-list path must remain pinned")
    if benchmark["canonical_sort_keys"] != list(_CANONICAL_SORT_KEYS):
        raise ValueError("DTap canonical task ordering must remain pinned")
    if benchmark["roles"] != ROLE_MAP:
        raise ValueError("DTap trust roles must remain user/input")
    if benchmark["aggregate_parallelism"] != AGGREGATE_PARALLELISM:
        raise ValueError("DTap aggregate parallelism must remain pinned")
    if canonical_bytes(benchmark["port_namespace"]) != canonical_bytes(
        port_namespace_contract()
    ):
        raise ValueError("DTap port-namespace policy must remain pinned")
    return DTapQwen36Config(
        model_name=str(model["name_or_path"]),
        model_revision=str(model["revision"]),
        evaluation_checkpoint_learning_rate=float(
            model["evaluation_checkpoint_learning_rate"]
        ),
        release_training_config_learning_rate=float(
            model["release_training_config_learning_rate"]
        ),
        task_list=str(benchmark["task_list"]),
        expected_tasks=int(benchmark["expected_tasks"]),
        task_file_sha256=str(benchmark["task_file_sha256"]),
        task_semantic_sha256=str(benchmark["task_semantic_sha256"]),
        excluded_domains=tuple(str(item) for item in benchmark["excluded_domains"]),
        shard_count=int(benchmark["shard_count"]),
        max_parallel_per_shard=int(benchmark["max_parallel_per_shard"]),
        max_context_tokens=int(generation["max_context_tokens"]),
        max_output_tokens=int(generation["max_output_tokens"]),
        enable_thinking=bool(generation["enable_thinking"]),
    )


def load_config(path: str | Path) -> DTapQwen36Config:
    value = _read_json(Path(path), "DTap configuration")
    return config_from_mapping(value)


def config_fingerprint(config: DTapQwen36Config) -> str:
    return fingerprint(config.protocol())


def port_namespace_contract() -> dict[str, Any]:
    """Return the immutable same-host port isolation policy."""

    return {
        "policy": PORT_NAMESPACE_POLICY,
        "default_offset_stride": PORT_DEFAULT_OFFSET_STRIDE,
        "configured_default_count": len(PINNED_CONFIGURED_DEFAULT_PORTS),
        "configured_default_min": min(PINNED_CONFIGURED_DEFAULT_PORTS),
        "configured_default_max": max(PINNED_CONFIGURED_DEFAULT_PORTS),
        "configured_defaults_sha256": PINNED_CONFIGURED_DEFAULT_PORTS_SHA256,
        "env_config_sha256": PINNED_ENV_CONFIG_SHA256,
        "fallback_start": PORT_FALLBACK_START,
        "fallback_width": PORT_FALLBACK_WIDTH,
        "lock_scope": "shard",
    }


def shard_port_namespace(shard_id: int) -> dict[str, Any]:
    """Return the deterministic public port identity for one of 33 shards."""

    if type(shard_id) is not int or not 0 <= shard_id < SHARD_COUNT:
        raise ValueError(f"shard_id must be in [0, {SHARD_COUNT})")
    start = PORT_FALLBACK_START + PORT_FALLBACK_WIDTH * shard_id
    return {
        "default_offset": PORT_DEFAULT_OFFSET_STRIDE * shard_id,
        "fallback_start": start,
        "fallback_end": start + PORT_FALLBACK_WIDTH - 1,
    }


def _direct_private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"DTap runtime path is not a direct directory: {path}")
    return path.resolve(strict=True)


def shard_runtime_environment(shard_id: int) -> dict[str, str]:
    """Create and return the non-secret node-local namespace for one shard."""

    namespace = shard_port_namespace(shard_id)
    raw_root = os.environ.get(RUNTIME_ROOT_ENVIRONMENT, "").strip()
    if not raw_root:
        raise RuntimeError(f"required runtime environment variable is missing: {RUNTIME_ROOT_ENVIRONMENT}")
    requested_root = Path(raw_root)
    if not requested_root.is_absolute():
        raise RuntimeError(f"{RUNTIME_ROOT_ENVIRONMENT} must be absolute")
    root = _direct_private_directory(requested_root)
    shard_root = _direct_private_directory(root / f"shard_{shard_id:02d}")
    lock_root = _direct_private_directory(shard_root / "port-locks")
    return {
        SHARD_ID_ENVIRONMENT: str(shard_id),
        SHARD_RUNTIME_ROOT_ENVIRONMENT: str(shard_root),
        "DT_PORT_DEFAULT_OFFSET": str(namespace["default_offset"]),
        "DT_PORT_RANGE": (
            f"{namespace['fallback_start']}-{namespace['fallback_end']}"
        ),
        "DT_PORT_LOCK_DIR": str(lock_root),
    }


@contextmanager
def activated_shard_runtime(shard_id: int):
    """Temporarily activate one collision-free shard namespace."""

    with _RUNTIME_ENVIRONMENT_LOCK:
        selected = shard_runtime_environment(shard_id)
        previous = {name: os.environ.get(name) for name in selected}
        os.environ.update(selected)
        try:
            yield selected
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _git(checkout: Path, *arguments: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=check,
        capture_output=True,
        text=True,
        env=build_subprocess_environment(),
    )
    return completed.stdout.strip()


def _tree_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ".git" in relative.parts:
            continue
        if path.is_symlink():
            result[relative.as_posix()] = "symlink:" + str(path.readlink())
        elif path.is_file():
            result[relative.as_posix()] = file_sha256(path)
    return result


def verify_external_sources(
    checkout: str | Path,
    *,
    patch: str | Path,
    template: str | Path,
    qwen_agent_license: str | Path,
) -> dict[str, Any]:
    """Verify an exact pinned DTap patched tree and companion artifacts."""

    root = Path(checkout).resolve()
    patch_path = Path(patch).resolve()
    template_path = Path(template).resolve()
    qwen_license_path = Path(qwen_agent_license).resolve()
    required = (root / ".git", root / "LICENSE", patch_path, template_path)
    if any(not path.exists() for path in required) or not qwen_license_path.is_file():
        raise RuntimeError("required DTap verification input is missing")
    checks = {
        "patch": (patch_path, DTAP_PATCH_SHA256),
        "template": (template_path, DTAP_TEMPLATE_SHA256),
        "dtap_license": (root / "LICENSE", DTAP_LICENSE_SHA256),
        "qwen_agent_license": (
            qwen_license_path,
            QWEN_AGENT_LICENSE_SHA256,
        ),
    }
    for label, (path, expected) in checks.items():
        if file_sha256(path) != expected:
            raise RuntimeError(f"{label} checksum differs from the reviewed artifact")
    head = _git(root, "rev-parse", "HEAD")
    if head != DTAP_COMMIT:
        raise RuntimeError("DTap checkout is not at the pinned commit")
    tracked_dataset = _git(root, "ls-files", "--", "dataset").splitlines()
    if not tracked_dataset:
        raise RuntimeError("pinned DTap checkout has no Git-tracked dataset payloads")

    # Materialize the reviewed tree independently, then compare every file.
    # This rejects extra edits and generated files as well as missing patch hunks.
    with tempfile.TemporaryDirectory(prefix="secpo-dtap-verify-") as temporary:
        expected = Path(temporary) / "expected"
        subprocess.run(
            ["git", "clone", "--quiet", "--no-hardlinks", str(root), str(expected)],
            check=True,
            capture_output=True,
            text=True,
            env=build_subprocess_environment(),
        )
        subprocess.run(
            ["git", "-C", str(expected), "checkout", "--quiet", DTAP_COMMIT],
            check=True,
            capture_output=True,
            text=True,
            env=build_subprocess_environment(),
        )
        subprocess.run(
            ["git", "-C", str(expected), "apply", str(patch_path)],
            check=True,
            capture_output=True,
            text=True,
            env=build_subprocess_environment(),
        )
        if _tree_hashes(root) != _tree_hashes(expected):
            raise RuntimeError("DTap checkout differs from the exact reviewed patched tree")
    return {
        "repository": DTAP_REPOSITORY,
        "commit": head,
        "version": DTAP_VERSION,
        "license": DTAP_LICENSE,
        "license_sha256": DTAP_LICENSE_SHA256,
        "patch_sha256": DTAP_PATCH_SHA256,
        "template_sha256": DTAP_TEMPLATE_SHA256,
        "qwen_agent_version": QWEN_AGENT_VERSION,
        "qwen_agent_license_sha256": QWEN_AGENT_LICENSE_SHA256,
        "dataset_source": "git_tracked_at_dtap_commit",
        "dataset_tracked_files": len(tracked_dataset),
    }


def separate_qwen_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Map explicit trust provenance to Qwen ``user`` and ``input`` roles.

    Assistant messages are copied as assistant messages.  Every other message
    must carry ``trust=trusted`` or ``trust=untrusted``; implicit trust is
    rejected so tool/environment output cannot silently become trusted text.
    """

    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "assistant":
            row = {"role": "assistant", "content": content}
            if "tool_calls" in message:
                row["tool_calls"] = message["tool_calls"]
            converted.append(row)
            continue
        trust = message.get("trust")
        if trust not in ROLE_MAP:
            raise ValueError("every non-assistant message needs explicit trust provenance")
        converted.append({"role": ROLE_MAP[str(trust)], "content": content})
    return converted


def _normalize_task(value: Mapping[str, Any]) -> dict[str, str]:
    _require_keys(value, set(_TASK_KEYS), "task")
    row: dict[str, str] = {}
    for key in _TASK_KEYS:
        item = value[key]
        if not isinstance(item, (str, int)) or isinstance(item, bool):
            raise ValueError(f"task {key} must be a string or integer")
        text = str(item).strip()
        if not text:
            raise ValueError(f"task {key} must not be empty")
        row[key] = text
    if row["type"] != "malicious" or row["threat_model"] != "indirect":
        raise ValueError("DTap task is outside the malicious indirect split")
    if row["domain"].casefold() in EXCLUDED_DOMAINS:
        raise ValueError("DTap task belongs to an excluded VM domain")
    return row


def task_key(task: Mapping[str, Any]) -> tuple[str, ...]:
    row = _normalize_task(task)
    return tuple(row[key] for key in _TASK_KEYS)


def _parse_task_bytes(raw: bytes, *, expected_count: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("DTap task list is not UTF-8") from exc
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank task-list line {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid task-list line {line_number}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"task-list line {line_number} is not an object")
        rows.append(_normalize_task(value))
    if len(rows) != expected_count:
        raise ValueError(f"selected {len(rows)} DTap tasks; expected {expected_count}")
    keys = [task_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("DTap task list contains duplicate identities")
    return rows


def selection_semantic_sha256(tasks: Sequence[Mapping[str, Any]]) -> str:
    """Reproduce the audited runner's ordered semantic task identity."""

    rows = []
    for task in tasks:
        normalized = _normalize_task(task)
        rows.append(
            {
                "domain": normalized["domain"],
                "task_type": normalized["type"],
                "task_id": normalized["task_id"],
                "threat_model": normalized["threat_model"],
                "risk_category": normalized["risk_category"],
            }
        )
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_tasks(
    path: str | Path,
    *,
    expected_count: int = EXPECTED_TASKS,
    require_authoritative: bool = True,
) -> list[dict[str, str]]:
    """Load DTap tasks, rejecting any substitute for the audited selection."""

    raw = Path(path).read_bytes()
    rows = _parse_task_bytes(raw, expected_count=expected_count)
    if require_authoritative:
        if hashlib.sha256(raw).hexdigest() != EXPECTED_TASK_FILE_SHA256:
            raise ValueError("DTap task-list file SHA-256 differs")
        if selection_semantic_sha256(rows) != EXPECTED_TASK_SEMANTIC_SHA256:
            raise ValueError("DTap ordered semantic task identity differs")
    return rows


def _task_sort_key(task: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(task[key] for key in _CANONICAL_SORT_KEYS)


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_bytes(row) + b"\n" for row in rows)


def _manifest_without_fingerprint(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "manifest_fingerprint"}


def _validate_manifest_fingerprint(value: Mapping[str, Any], label: str) -> None:
    observed = value.get("manifest_fingerprint")
    expected = fingerprint(_manifest_without_fingerprint(value))
    if observed != expected:
        raise RuntimeError(f"{label} fingerprint differs")


def prepare_shards(
    task_list: str | Path,
    shards_dir: str | Path,
    *,
    config: DTapQwen36Config,
) -> Mapping[str, Any]:
    """Create or validate the exact 33-way canonical round-robin plan."""

    tasks_in_file_order = load_tasks(task_list, expected_count=config.expected_tasks)
    tasks = sorted(tasks_in_file_order, key=_task_sort_key)
    shards = [tasks[index :: config.shard_count] for index in range(config.shard_count)]
    if sum(map(len, shards)) != config.expected_tasks:
        raise AssertionError("internal DTap shard assignment error")
    root = Path(shards_dir)
    rows: list[dict[str, Any]] = []
    for shard_id, shard in enumerate(shards):
        payload = _jsonl_bytes(shard)
        rows.append(
            {
                "shard_id": shard_id,
                "task_file": f"shard_{shard_id:02d}.tasks.jsonl",
                "task_count": len(shard),
                "task_file_sha256": hashlib.sha256(payload).hexdigest(),
                "task_semantic_sha256": selection_semantic_sha256(shard),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "algorithm": "canonical_sort_round_robin_v1",
        "config_fingerprint": config_fingerprint(config),
        "task_count": len(tasks),
        "task_file_sha256": config.task_file_sha256,
        "task_semantic_sha256": config.task_semantic_sha256,
        "task_set_fingerprint": fingerprint(tasks),
        "excluded_domains": list(config.excluded_domains),
        "topology": {
            "shard_count": config.shard_count,
            "max_parallel_per_shard": config.max_parallel_per_shard,
            "aggregate_parallelism": config.aggregate_parallelism,
        },
        "shards": rows,
    }
    manifest["manifest_fingerprint"] = fingerprint(manifest)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        observed = _read_json(manifest_path, "DTap shard manifest")
        if dict(observed) != manifest:
            raise RuntimeError("existing DTap shard plan has a different identity")
        validate_shard_plan(root, config=config)
        return observed
    if root.exists() and any(root.iterdir()):
        raise RuntimeError("DTap shard directory is non-empty without a manifest")
    root.mkdir(parents=True, exist_ok=True)
    for shard, row in zip(shards, rows, strict=True):
        _atomic_write(root / row["task_file"], _jsonl_bytes(shard))
    _write_json(manifest_path, manifest)
    return manifest


def validate_shard_plan(
    shards_dir: str | Path,
    *,
    config: DTapQwen36Config,
) -> tuple[Mapping[str, Any], list[list[dict[str, str]]]]:
    root = Path(shards_dir)
    manifest = _read_json(root / "manifest.json", "DTap shard manifest")
    _validate_manifest_fingerprint(manifest, "DTap shard manifest")
    if manifest.get("schema_version") != SHARD_SCHEMA_VERSION:
        raise RuntimeError("unsupported DTap shard manifest")
    expected_manifest_keys = {
        "schema_version",
        "algorithm",
        "config_fingerprint",
        "task_count",
        "task_file_sha256",
        "task_semantic_sha256",
        "task_set_fingerprint",
        "excluded_domains",
        "topology",
        "shards",
        "manifest_fingerprint",
    }
    if set(manifest) != expected_manifest_keys:
        raise RuntimeError("DTap shard manifest fields differ")
    if manifest.get("algorithm") != "canonical_sort_round_robin_v1":
        raise RuntimeError("DTap shard assignment algorithm differs")
    if manifest.get("task_count") != config.expected_tasks:
        raise RuntimeError("DTap shard manifest task count differs")
    if manifest.get("task_file_sha256") != config.task_file_sha256:
        raise RuntimeError("DTap source task-list file identity differs")
    if manifest.get("task_semantic_sha256") != config.task_semantic_sha256:
        raise RuntimeError("DTap source task semantic identity differs")
    if manifest.get("excluded_domains") != list(config.excluded_domains):
        raise RuntimeError("DTap shard excluded-domain selection differs")
    if manifest.get("config_fingerprint") != config_fingerprint(config):
        raise RuntimeError("DTap shard manifest configuration differs")
    topology = manifest.get("topology")
    expected_topology = {
        "shard_count": config.shard_count,
        "max_parallel_per_shard": config.max_parallel_per_shard,
        "aggregate_parallelism": config.aggregate_parallelism,
    }
    if topology != expected_topology:
        raise RuntimeError("DTap shard topology differs")
    shard_rows = manifest.get("shards")
    if not isinstance(shard_rows, list) or len(shard_rows) != config.shard_count:
        raise RuntimeError("DTap shard manifest has the wrong number of shards")
    shards: list[list[dict[str, str]]] = []
    all_tasks: list[dict[str, str]] = []
    for shard_id, row in enumerate(shard_rows):
        if not isinstance(row, Mapping):
            raise RuntimeError("DTap shard manifest row is not an object")
        if set(row) != {
            "shard_id",
            "task_file",
            "task_count",
            "task_file_sha256",
            "task_semantic_sha256",
        }:
            raise RuntimeError("DTap shard manifest row fields differ")
        expected_path = f"shard_{shard_id:02d}.tasks.jsonl"
        if row.get("shard_id") != shard_id or row.get("task_file") != expected_path:
            raise RuntimeError("DTap shard manifest ordering differs")
        if type(row.get("task_count")) is not int or row["task_count"] < 1:
            raise RuntimeError("DTap shard manifest task count is invalid")
        path = root / expected_path
        if file_sha256(path) != row.get("task_file_sha256"):
            raise RuntimeError(f"DTap shard {shard_id} file checksum differs")
        tasks = load_tasks(
            path,
            expected_count=int(row.get("task_count", -1)),
            require_authoritative=False,
        )
        if selection_semantic_sha256(tasks) != row.get("task_semantic_sha256"):
            raise RuntimeError(f"DTap shard {shard_id} semantic checksum differs")
        shards.append(tasks)
        all_tasks.extend(tasks)
    if len(all_tasks) != config.expected_tasks:
        raise RuntimeError("DTap shard union has the wrong task count")
    if len({task_key(task) for task in all_tasks}) != config.expected_tasks:
        raise RuntimeError("DTap shard union contains duplicates")
    canonical = sorted(all_tasks, key=_task_sort_key)
    if manifest.get("task_set_fingerprint") != fingerprint(canonical):
        raise RuntimeError("DTap shard task-set fingerprint differs")
    expected_shards = [
        canonical[index :: config.shard_count] for index in range(config.shard_count)
    ]
    if shards != expected_shards:
        raise RuntimeError("DTap shards are not the canonical round-robin assignment")
    return manifest, shards


TaskBackend = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def load_backend(specification: str) -> TaskBackend:
    """Load an explicitly named ``module:callable`` task implementation."""

    if specification.count(":") != 1:
        raise ValueError("backend must use module:callable syntax")
    module_name, attribute = specification.split(":", 1)
    if not module_name or not attribute:
        raise ValueError("backend must use module:callable syntax")
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise TypeError("selected DTap backend is not callable")
    return value


def backend_fingerprint(backend: TaskBackend) -> str:
    """Fingerprint callable identity and its complete module source."""

    try:
        source = inspect.getsource(backend)
        module = inspect.getmodule(backend)
        module_source = inspect.getsource(module) if module is not None else source
        identity_factory = (
            getattr(module, "backend_runtime_identity", None)
            if module is not None
            else None
        )
        runtime_identity = (
            identity_factory() if callable(identity_factory) else None
        )
    except (OSError, TypeError) as exc:
        raise ValueError("DTap backend source must be inspectable") from exc
    return fingerprint(
        {
            "module": backend.__module__,
            "qualified_name": backend.__qualname__,
            "source": source,
            "module_source": module_source,
            "runtime_identity": runtime_identity,
        }
    )


def _result_path(root: Path, shard_id: int, task_index: int) -> Path:
    return root / f"shard_{shard_id:02d}" / f"task_{task_index:04d}.json"


def _result_identity(
    *,
    task: Mapping[str, Any],
    shard_id: int,
    task_index: int,
    plan_fingerprint: str,
    protocol_fingerprint: str,
    backend_hash: str,
) -> dict[str, Any]:
    return {
        "task": dict(task),
        "task_fingerprint": fingerprint(task),
        "shard_id": shard_id,
        "task_index": task_index,
        "plan_fingerprint": plan_fingerprint,
        "protocol_fingerprint": protocol_fingerprint,
        "backend_fingerprint": backend_hash,
    }


def _validate_existing_result(path: Path, identity: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _read_json(path, "DTap task result")
    expected_keys = {
        "schema_version",
        *identity.keys(),
        "status",
        "task_success",
        "attack_success",
        "error_type",
        "result_fingerprint",
    }
    if set(value) != expected_keys:
        raise RuntimeError("existing DTap result fields differ")
    if value.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise RuntimeError("existing DTap result schema differs")
    for key, expected in identity.items():
        if value.get(key) != expected:
            raise RuntimeError("existing DTap result identity differs")
    observed_fingerprint = value.get("result_fingerprint")
    expected_fingerprint = fingerprint(
        {key: item for key, item in value.items() if key != "result_fingerprint"}
    )
    if observed_fingerprint != expected_fingerprint:
        raise RuntimeError("existing DTap result fingerprint differs")
    status = value.get("status")
    if status == "complete":
        if type(value.get("task_success")) is not bool or type(
            value.get("attack_success")
        ) is not bool:
            raise RuntimeError("completed DTap result has an invalid score")
        if value.get("error_type") is not None:
            raise RuntimeError("completed DTap result unexpectedly contains an error")
    elif status == "error":
        if value.get("task_success") is not None or value.get("attack_success") is not None:
            raise RuntimeError("errored DTap result must not contain a score")
        if not isinstance(value.get("error_type"), str) or not value["error_type"]:
            raise RuntimeError("errored DTap result lacks an error type")
    else:
        raise RuntimeError("existing DTap result status differs")
    return value


def _run_one(
    backend: TaskBackend,
    request: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        output = backend(request)
        if not isinstance(output, Mapping):
            raise TypeError("DTap backend result must be a mapping")
        task_success = output.get("task_success")
        attack_success = output.get("attack_success")
        if type(task_success) is not bool or type(attack_success) is not bool:
            raise ValueError("DTap scores must be explicit booleans")
        status = "complete"
        error_type = None
    except Exception as exc:  # Task failures are recorded without sensitive text.
        task_success = None
        attack_success = None
        status = "error"
        error_type = type(exc).__name__
    value: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        **identity,
        "status": status,
        "task_success": task_success,
        "attack_success": attack_success,
        "error_type": error_type,
    }
    value["result_fingerprint"] = fingerprint(value)
    return value


def run_shard(
    shards_dir: str | Path,
    results_dir: str | Path,
    *,
    shard_id: int,
    config: DTapQwen36Config,
    backend: TaskBackend,
    fail_on_error: bool = True,
) -> Mapping[str, Any]:
    """Run one CPU shard with six concurrent tasks and strict resume."""

    manifest, shards = validate_shard_plan(shards_dir, config=config)
    if not 0 <= shard_id < config.shard_count:
        raise ValueError(f"shard_id must be in [0, {config.shard_count})")
    tasks = shards[shard_id]
    plan_hash = str(manifest["manifest_fingerprint"])
    protocol_hash = config_fingerprint(config)
    backend_hash = backend_fingerprint(backend)
    root = Path(results_dir)
    run_path = root / f"shard_{shard_id:02d}" / "run_manifest.json"
    run_identity = {
        "schema_version": RUN_SCHEMA_VERSION,
        "shard_id": shard_id,
        "task_count": len(tasks),
        "plan_fingerprint": plan_hash,
        "protocol_fingerprint": protocol_hash,
        "backend_fingerprint": backend_hash,
        "max_parallel_tasks": config.max_parallel_per_shard,
        "port_namespace": shard_port_namespace(shard_id),
    }
    if run_path.exists():
        observed_run = _read_json(run_path, "DTap run manifest")
        _validate_manifest_fingerprint(observed_run, "DTap run manifest")
        expected_run_keys = {
            *run_identity.keys(),
            "status",
            "complete",
            "errors",
            "manifest_fingerprint",
        }
        if set(observed_run) != expected_run_keys:
            raise RuntimeError("existing DTap run manifest fields differ")
        for key, expected in run_identity.items():
            if observed_run.get(key) != expected:
                raise RuntimeError("existing DTap run manifest identity differs")

    pending: list[tuple[int, dict[str, str], dict[str, Any], Path]] = []
    complete = 0
    for task_index, task in enumerate(tasks):
        identity = _result_identity(
            task=task,
            shard_id=shard_id,
            task_index=task_index,
            plan_fingerprint=plan_hash,
            protocol_fingerprint=protocol_hash,
            backend_hash=backend_hash,
        )
        path = _result_path(root, shard_id, task_index)
        if path.exists():
            existing = _validate_existing_result(path, identity)
            if existing["status"] == "complete":
                complete += 1
                continue
        pending.append((task_index, task, identity, path))

    errors = 0
    if pending:
        with activated_shard_runtime(shard_id):
            with ThreadPoolExecutor(
                max_workers=config.max_parallel_per_shard
            ) as executor:
                futures = {}
                for task_index, task, identity, path in pending:
                    request = {
                        "schema_version": SCHEMA_VERSION,
                        "task": task,
                        "task_fingerprint": identity["task_fingerprint"],
                        "protocol": config.protocol(),
                    }
                    future = executor.submit(_run_one, backend, request, identity)
                    futures[future] = path
                for future in as_completed(futures):
                    result = future.result()
                    _write_json(futures[future], result)
                    if result["status"] == "complete":
                        complete += 1
                    else:
                        errors += 1

    report: dict[str, Any] = {
        **run_identity,
        "status": "complete" if errors == 0 and complete == len(tasks) else "incomplete",
        "complete": complete,
        "errors": errors,
    }
    report["manifest_fingerprint"] = fingerprint(report)
    _write_json(run_path, report)
    if errors and fail_on_error:
        raise ShardExecutionError(report)
    return report


def _run_shard_process(
    shards_dir: str,
    results_dir: str,
    shard_id: int,
    config: DTapQwen36Config,
    backend_specification: str,
) -> Mapping[str, Any]:
    """Top-level pickle-safe process entry point."""

    with activated_shard_runtime(shard_id):
        backend = load_backend(backend_specification)
        try:
            return run_shard(
                shards_dir,
                results_dir,
                shard_id=shard_id,
                config=config,
                backend=backend,
            )
        except ShardExecutionError as exc:
            return exc.report


def launch_all_shards(
    shards_dir: str | Path,
    results_dir: str | Path,
    *,
    config: DTapQwen36Config,
    backend_specification: str,
    fail_on_error: bool = True,
) -> Mapping[str, Any]:
    """Launch 33 portable worker processes, each with six task threads.

    The callable is imported independently inside every worker process.  This
    supplies the audited 33-by-6 CPU topology without embedding cluster policy.
    Use ``secpo.evaluation.dtap_qwen_agent_backend:run_task`` for the reviewed
    external-DTap lifecycle and judge mapping.
    """

    manifest, _ = validate_shard_plan(shards_dir, config=config)
    selected_backend = load_backend(backend_specification)
    backend_hash = backend_fingerprint(selected_backend)
    identity = {
        "schema_version": LAUNCH_SCHEMA_VERSION,
        "plan_fingerprint": manifest["manifest_fingerprint"],
        "protocol_fingerprint": config_fingerprint(config),
        "backend_fingerprint": backend_hash,
        "worker_processes": config.shard_count,
        "max_parallel_per_process": config.max_parallel_per_shard,
        "aggregate_parallelism": config.aggregate_parallelism,
    }
    reports: dict[int, Mapping[str, Any]] = {}
    process_errors: dict[int, str] = {}
    with ProcessPoolExecutor(max_workers=config.shard_count) as executor:
        futures = {
            executor.submit(
                _run_shard_process,
                str(shards_dir),
                str(results_dir),
                shard_id,
                config,
                backend_specification,
            ): shard_id
            for shard_id in range(config.shard_count)
        }
        for future in as_completed(futures):
            shard_id = futures[future]
            try:
                reports[shard_id] = future.result()
            except Exception as exc:  # Persist only the exception class.
                process_errors[shard_id] = type(exc).__name__
    incomplete = sorted(
        {
            *process_errors,
            *(
                shard_id
                for shard_id, report in reports.items()
                if report.get("status") != "complete"
            ),
        }
    )
    report: dict[str, Any] = {
        **identity,
        "status": "complete" if not incomplete else "incomplete",
        "completed_shards": config.shard_count - len(incomplete),
        "incomplete_shards": len(incomplete),
        "incomplete_shard_ids": incomplete,
        "process_error_types": {
            str(key): value for key, value in sorted(process_errors.items())
        },
    }
    report["manifest_fingerprint"] = fingerprint(report)
    _write_json(Path(results_dir) / "multiprocess_launch_manifest.json", report)
    if incomplete and fail_on_error:
        raise MultiProcessExecutionError(report)
    return report


def summarize_results(
    shards_dir: str | Path,
    results_dir: str | Path,
    *,
    config: DTapQwen36Config,
) -> Mapping[str, Any]:
    """Aggregate exact boolean scores; final rates remain null until complete."""

    manifest, shards = validate_shard_plan(shards_dir, config=config)
    root = Path(results_dir)
    complete = 0
    errors = 0
    missing = 0
    utility_successes = 0
    attack_successes = 0
    backend_hashes: set[str] = set()
    for shard_id, tasks in enumerate(shards):
        for task_index, task in enumerate(tasks):
            path = _result_path(root, shard_id, task_index)
            if not path.exists():
                missing += 1
                continue
            value = _read_json(path, "DTap task result")
            backend_hash = value.get("backend_fingerprint")
            if not isinstance(backend_hash, str) or not backend_hash:
                raise RuntimeError("DTap result lacks a backend fingerprint")
            backend_hashes.add(backend_hash)
            identity = _result_identity(
                task=task,
                shard_id=shard_id,
                task_index=task_index,
                plan_fingerprint=str(manifest["manifest_fingerprint"]),
                protocol_fingerprint=config_fingerprint(config),
                backend_hash=backend_hash,
            )
            value = _validate_existing_result(path, identity)
            if value["status"] == "error":
                errors += 1
                continue
            complete += 1
            utility_successes += int(value["task_success"])
            attack_successes += int(value["attack_success"])
    if len(backend_hashes) > 1:
        raise RuntimeError("DTap results were produced by different backend implementations")
    is_complete = complete == config.expected_tasks and errors == 0 and missing == 0
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "complete" if is_complete else "incomplete",
        "plan_fingerprint": manifest["manifest_fingerprint"],
        "protocol_fingerprint": config_fingerprint(config),
        "backend_fingerprint": next(iter(backend_hashes), None),
        "expected_tasks": config.expected_tasks,
        "completed_tasks": complete,
        "error_tasks": errors,
        "missing_tasks": missing,
        "utility_successes": utility_successes,
        "attack_successes": attack_successes,
        "utility_success_rate": utility_successes / complete if is_complete else None,
        "attack_success_rate": attack_successes / complete if is_complete else None,
        "completed_only_utility_rate": (
            utility_successes / complete if complete else None
        ),
        "completed_only_attack_success_rate": (
            attack_successes / complete if complete else None
        ),
    }
    summary["manifest_fingerprint"] = fingerprint(summary)
    return summary
