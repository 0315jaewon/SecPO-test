"""Pinned AgentDyn adapter for Qwen trusted-user/untrusted-input messages.

AgentDyn remains an external MIT-licensed checkout. This module verifies the
immutable checkout and reviewed patch, normalizes the message trust boundary,
and builds the upstream benchmark commands. Runtime endpoints and optional
credentials are resolved exclusively from environment variables.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib.parse import urlsplit
import uuid

from secpo.subprocess_env import build_subprocess_environment


SCHEMA_VERSION = "secpo-agentdyn-qwen36-v1"
UPSTREAM_REPOSITORY = "https://github.com/SaFo-Lab/AgentDyn.git"
UPSTREAM_COMMIT = "5353cf7615b135cace8d07c8f12dac53a16b6db3"
UPSTREAM_VERSION = "0.1.35"
UPSTREAM_LICENSE = "MIT"
UPSTREAM_LICENSE_SHA256 = (
    "4285a071f2d382338e52b4fb0a186d952984a34d43a33d8872e1a1d8cb43401e"
)
PATCH_SHA256 = "43a8e090daa08854d93cc8ea613d203354b004da645253b90380b3e59a7075f6"
TEMPLATE_SHA256 = "0ae8886ac5955f0bbb9b67da4065b701902891ff67249baf8e4cfd4c01d55aa7"
# Filled from the pinned base tree plus the reviewed patch using
# ``_derive_expected_patched_tree_sha256``. It is not a hash of a local path.
PATCHED_TREE_SHA256 = "c0cc7b493d3239c57e7de622f398a8e04f8b6f102b690ca31d99f43695fb353f"
PATCH_RELEASE_PATH = "third_party/patches/agentdyn.patch"
TEMPLATE_RELEASE_PATH = "templates/qwen36_agentdyn_input_role.jinja"
QWEN36_MODEL = "Qwen/Qwen3.6-27B"
QWEN36_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
DEFAULT_SUITES = ("shopping", "github", "dailylife")
DEFAULT_ATTACKS = ("none", "important_instructions")
MIN_OUTPUT_TOKENS = 32_768
DEFAULT_REQUEST_MAX_ATTEMPTS = 3
DEFAULT_REQUEST_RETRY_MAX_WAIT_SECONDS = 40.0
SERVER_TOOL_CALL_PARSER = "qwen3_coder"
SERVER_REASONING_PARSER = "qwen3"
SERVER_GENERATION_CONFIG = "vllm"
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_INSTALLED_MARKER = "_secpo_agentdyn_qwen36_input_adapter"


class InvalidModelOutputError(RuntimeError):
    """Raised when the local server returns neither visible text nor tools."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    item = value.get(name)
    if not isinstance(item, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return item


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], name: str
) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{name} fields differ; missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _exact_bool(value: Mapping[str, Any], key: str) -> bool:
    item = value.get(key)
    if type(item) is not bool:
        raise ValueError(f"{key} must be a JSON boolean")
    return item


def _exact_int(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if type(item) is not int:
        raise ValueError(f"{key} must be a JSON integer")
    return item


def _exact_number(value: Mapping[str, Any], key: str) -> float:
    item = value.get(key)
    if type(item) not in {int, float}:
        raise ValueError(f"{key} must be a JSON number")
    return float(item)


def _validate_endpoint(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("AgentDyn endpoint must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("AgentDyn endpoint must not embed credentials or query data")
    return value.rstrip("/")


@dataclass(frozen=True)
class AgentDynConfig:
    """Strict public configuration for the pinned AgentDyn integration."""

    model_name: str = QWEN36_MODEL
    model_revision: str = QWEN36_REVISION
    benchmark_version: str = "v1.2.2"
    suites: tuple[str, ...] = DEFAULT_SUITES
    attacks: tuple[str, ...] = DEFAULT_ATTACKS
    defense: str = "repeat_user_prompt"
    max_workers: int = 3
    max_output_tokens: int = MIN_OUTPUT_TOKENS
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 0.0
    seed: int = 0
    enable_thinking: bool = True
    preserve_thinking: bool = True
    base_url_env: str = "AGENTDYN_BASE_URL"
    api_key_env: str = "AGENTDYN_API_KEY"
    request_timeout_seconds: float = 1800.0
    request_max_attempts: int = DEFAULT_REQUEST_MAX_ATTEMPTS
    request_retry_max_wait_seconds: float = DEFAULT_REQUEST_RETRY_MAX_WAIT_SECONDS
    chat_template: str = TEMPLATE_RELEASE_PATH
    language_model_only: bool = True
    enable_auto_tool_choice: bool = True
    tool_call_parser: str = SERVER_TOOL_CALL_PARSER
    reasoning_parser: str = SERVER_REASONING_PARSER
    generation_config: str = SERVER_GENERATION_CONFIG

    def __post_init__(self) -> None:
        if self.model_name != QWEN36_MODEL or self.model_revision != QWEN36_REVISION:
            raise ValueError("AgentDyn model name and revision must remain pinned")
        if self.benchmark_version != "v1.2.2":
            raise ValueError("AgentDyn benchmark version must remain v1.2.2")
        frozen = (
            self.suites == DEFAULT_SUITES,
            self.attacks == DEFAULT_ATTACKS,
            self.defense == "repeat_user_prompt",
            self.max_workers == 3,
            self.max_output_tokens == MIN_OUTPUT_TOKENS,
            self.temperature == 1.0,
            self.top_p == 0.95,
            self.top_k == 20,
            self.presence_penalty == 0.0,
            self.seed == 0,
            self.enable_thinking is True,
            self.preserve_thinking is True,
            self.base_url_env == "AGENTDYN_BASE_URL",
            self.api_key_env == "AGENTDYN_API_KEY",
            self.request_timeout_seconds == 1800.0,
            self.request_max_attempts == DEFAULT_REQUEST_MAX_ATTEMPTS,
            self.request_retry_max_wait_seconds
            == DEFAULT_REQUEST_RETRY_MAX_WAIT_SECONDS,
        )
        if not all(frozen):
            raise ValueError("AgentDyn main-experiment settings must remain pinned")
        for name in (self.base_url_env, self.api_key_env):
            if not _ENV_NAME.fullmatch(name):
                raise ValueError("AgentDyn environment-variable name is invalid")
        if self.chat_template != TEMPLATE_RELEASE_PATH:
            raise ValueError("AgentDyn chat template path must remain pinned")
        if not self.language_model_only or not self.enable_auto_tool_choice:
            raise ValueError("AgentDyn requires language-only native tool serving")
        if self.tool_call_parser != SERVER_TOOL_CALL_PARSER:
            raise ValueError("AgentDyn vLLM tool-call parser must remain pinned")
        if self.reasoning_parser != SERVER_REASONING_PARSER:
            raise ValueError("AgentDyn vLLM reasoning parser must remain pinned")
        if self.generation_config != SERVER_GENERATION_CONFIG:
            raise ValueError("AgentDyn vLLM generation config must remain pinned")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AgentDynConfig":
        _require_exact_keys(
            value,
            {
                "schema_version",
                "status",
                "upstream",
                "model",
                "benchmark",
                "generation",
                "server",
            },
            "configuration",
        )
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported AgentDyn release configuration")
        if value.get("status") != "external_checkout_required":
            raise ValueError("AgentDyn release status must remain explicit")
        upstream = _mapping(value, "upstream")
        expected_upstream = {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "version": UPSTREAM_VERSION,
            "license": UPSTREAM_LICENSE,
            "license_sha256": UPSTREAM_LICENSE_SHA256,
            "patch": PATCH_RELEASE_PATH,
            "patch_sha256": PATCH_SHA256,
            "patched_tree_sha256": PATCHED_TREE_SHA256,
            "template": TEMPLATE_RELEASE_PATH,
            "template_sha256": TEMPLATE_SHA256,
            "vendored": False,
        }
        if dict(upstream) != expected_upstream:
            raise ValueError("AgentDyn upstream identity must remain pinned")
        model = _mapping(value, "model")
        benchmark = _mapping(value, "benchmark")
        generation = _mapping(value, "generation")
        server = _mapping(value, "server")
        _require_exact_keys(
            model,
            {
                "name_or_path",
                "revision",
                "base_url_env",
                "api_key_env",
                "request_timeout_seconds",
                "request_max_attempts",
                "request_retry_max_wait_seconds",
            },
            "model",
        )
        _require_exact_keys(
            benchmark,
            {"version", "suites", "attacks", "defense", "max_workers", "roles"},
            "benchmark",
        )
        _require_exact_keys(
            generation,
            {
                "enable_thinking",
                "preserve_thinking",
                "max_output_tokens",
                "temperature",
                "top_p",
                "top_k",
                "presence_penalty",
                "seed",
            },
            "generation",
        )
        _require_exact_keys(
            server,
            {
                "chat_template",
                "language_model_only",
                "enable_auto_tool_choice",
                "tool_call_parser",
                "reasoning_parser",
                "generation_config",
            },
            "server",
        )
        forbidden = {
            key
            for key in model
            if key.casefold() in {"api_key", "token", "secret", "password"}
        }
        if forbidden:
            raise ValueError(
                "AgentDyn config may reference credentials only by environment name"
            )
        if benchmark.get("roles") != {"trusted": "user", "untrusted": "input"}:
            raise ValueError("AgentDyn trust roles must remain user/input")
        return cls(
            model_name=str(model["name_or_path"]),
            model_revision=str(model["revision"]),
            base_url_env=str(model["base_url_env"]),
            api_key_env=str(model["api_key_env"]),
            request_timeout_seconds=_exact_number(model, "request_timeout_seconds"),
            request_max_attempts=_exact_int(model, "request_max_attempts"),
            request_retry_max_wait_seconds=_exact_number(
                model, "request_retry_max_wait_seconds"
            ),
            benchmark_version=str(benchmark["version"]),
            suites=tuple(str(item) for item in benchmark["suites"]),
            attacks=tuple(str(item) for item in benchmark["attacks"]),
            defense=str(benchmark["defense"]),
            max_workers=_exact_int(benchmark, "max_workers"),
            max_output_tokens=_exact_int(generation, "max_output_tokens"),
            temperature=_exact_number(generation, "temperature"),
            top_p=_exact_number(generation, "top_p"),
            top_k=_exact_int(generation, "top_k"),
            presence_penalty=_exact_number(generation, "presence_penalty"),
            seed=_exact_int(generation, "seed"),
            enable_thinking=_exact_bool(generation, "enable_thinking"),
            preserve_thinking=_exact_bool(generation, "preserve_thinking"),
            chat_template=str(server["chat_template"]),
            language_model_only=_exact_bool(server, "language_model_only"),
            enable_auto_tool_choice=_exact_bool(server, "enable_auto_tool_choice"),
            tool_call_parser=str(server["tool_call_parser"]),
            reasoning_parser=str(server["reasoning_parser"]),
            generation_config=str(server["generation_config"]),
        )

    def endpoint(self) -> str:
        value = os.getenv(self.base_url_env, "").strip()
        if not value:
            raise RuntimeError(f"{self.base_url_env} is required at runtime")
        return _validate_endpoint(value)


def load_config(path: str | Path) -> AgentDynConfig:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("AgentDyn configuration must be a JSON object")
    return AgentDynConfig.from_mapping(value)


def _git(
    checkout: Path, *arguments: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=check,
        capture_output=True,
        text=True,
        env=build_subprocess_environment(),
    )


def _git_bytes(
    checkout: Path, *arguments: str, check: bool = True
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=check,
        capture_output=True,
        env=build_subprocess_environment(),
    )


@dataclass(frozen=True)
class _TreeEntry:
    path: str
    mode: str
    object_id: str


def _base_tree_entries(checkout: Path) -> tuple[_TreeEntry, ...]:
    raw = _git_bytes(checkout, "ls-tree", "-r", "-z", UPSTREAM_COMMIT).stdout
    entries: list[_TreeEntry] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        header, separator, path_bytes = record.partition(b"\t")
        if not separator:
            raise RuntimeError("AgentDyn base tree contains an invalid entry")
        fields = header.split()
        if len(fields) != 3 or fields[1] != b"blob":
            raise RuntimeError("AgentDyn base tree contains an unsupported entry")
        entries.append(
            _TreeEntry(
                path=os.fsdecode(path_bytes),
                mode=fields[0].decode("ascii"),
                object_id=fields[2].decode("ascii"),
            )
        )
    if not entries:
        raise RuntimeError("AgentDyn base tree is empty")
    return tuple(entries)


def _patch_paths(path: Path) -> tuple[str, ...]:
    old_paths: list[str] = []
    new_paths: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("--- "):
            raw = line[4:].split("\t", 1)[0]
            old_paths.append(raw.removeprefix("a/"))
        elif line.startswith("+++ "):
            raw = line[4:].split("\t", 1)[0]
            new_paths.append(raw.removeprefix("b/"))
    if not old_paths or old_paths != new_paths:
        raise ValueError(
            "AgentDyn patch must modify existing paths without additions or renames"
        )
    if any(path_name == "/dev/null" for path_name in old_paths):
        raise ValueError("AgentDyn patch must not add or delete files")
    if len(old_paths) != len(set(old_paths)):
        raise ValueError("AgentDyn patch repeats a modified path")
    return tuple(old_paths)


def _file_identity(path: Path) -> tuple[str, str]:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        mode = "120000"
        content = os.fsencode(os.readlink(path))
    elif stat.S_ISREG(metadata.st_mode):
        mode = "100755" if metadata.st_mode & 0o111 else "100644"
        content = path.read_bytes()
    else:
        raise RuntimeError(f"unsupported AgentDyn tree entry at {path}")
    return mode, hashlib.sha256(content).hexdigest()


def _logical_tree_sha256(
    entries: Sequence[_TreeEntry],
    overrides: Mapping[str, tuple[str, str]],
) -> str:
    """Hash the complete logical Git tree, substituting reviewed worktree files."""

    known = {entry.path for entry in entries}
    unexpected = set(overrides) - known
    if unexpected:
        raise RuntimeError(f"unexpected AgentDyn tree paths: {sorted(unexpected)!r}")
    digest = hashlib.sha256()
    for entry in entries:
        if entry.path in overrides:
            mode, content_identity = overrides[entry.path]
            identity = f"sha256:{content_identity}"
        else:
            mode = entry.mode
            identity = f"git:{entry.object_id}"
        for field in (mode, entry.path, identity):
            encoded = field.encode("utf-8", errors="surrogateescape")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _derive_expected_patched_tree_sha256(
    checkout: Path,
    patch_path: Path,
    entries: Sequence[_TreeEntry],
) -> str:
    """Apply the reviewed patch to independent base blobs and hash the full tree."""

    by_path = {entry.path: entry for entry in entries}
    patch_paths = _patch_paths(patch_path)
    if any(path not in by_path for path in patch_paths):
        raise RuntimeError("AgentDyn patch references a path outside the pinned tree")
    with tempfile.TemporaryDirectory(prefix="secpo-agentdyn-verify-") as temporary:
        expected_root = Path(temporary)
        for path_name in patch_paths:
            entry = by_path[path_name]
            destination = expected_root / path_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(
                _git_bytes(
                    checkout,
                    "show",
                    f"{UPSTREAM_COMMIT}:{path_name}",
                ).stdout
            )
            destination.chmod(0o755 if entry.mode == "100755" else 0o644)
        applied = subprocess.run(
            [
                "git",
                "apply",
                "--unsafe-paths",
                f"--directory={expected_root}",
                str(patch_path),
            ],
            check=False,
            capture_output=True,
            env=build_subprocess_environment(),
        )
        if applied.returncode != 0:
            raise RuntimeError("AgentDyn patch does not apply to independent base files")
        overrides = {
            path_name: _file_identity(expected_root / path_name)
            for path_name in patch_paths
        }
        return _logical_tree_sha256(entries, overrides)


def verify_agentdyn_checkout(
    checkout: str | Path,
    *,
    patch: str | Path,
    template: str | Path,
) -> dict[str, Any]:
    """Require the pinned AgentDyn checkout with exactly the reviewed patch."""

    root = Path(checkout).expanduser().resolve()
    patch_path = Path(patch).expanduser().resolve()
    template_path = Path(template).expanduser().resolve()
    if not patch_path.is_file() or _sha256(patch_path) != PATCH_SHA256:
        raise RuntimeError("AgentDyn patch checksum differs from the reviewed patch")
    if not template_path.is_file() or _sha256(template_path) != TEMPLATE_SHA256:
        raise RuntimeError("AgentDyn template checksum differs from the reviewed template")
    required = (
        root / "LICENSE",
        root / "pyproject.toml",
        root / "src" / "agentdojo" / "benchmark.py",
    )
    if any(not path.is_file() for path in required):
        raise RuntimeError("AgentDyn checkout is incomplete")
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    if head != UPSTREAM_COMMIT:
        raise RuntimeError("AgentDyn checkout is not at the pinned commit")
    if _sha256(root / "LICENSE") != UPSTREAM_LICENSE_SHA256:
        raise RuntimeError("AgentDyn license checksum differs")

    patch_paths = set(_patch_paths(patch_path))
    changed = {
        os.fsdecode(path)
        for path in _git_bytes(
            root, "diff", "--name-only", "-z", "HEAD", "--"
        ).stdout.split(b"\0")
        if path
    }
    if changed != patch_paths:
        raise RuntimeError("AgentDyn tracked changes do not match the reviewed patch")
    untracked = [
        os.fsdecode(path)
        for path in _git_bytes(
            root, "ls-files", "--others", "--exclude-standard", "-z"
        ).stdout.split(b"\0")
        if path
    ]
    if untracked:
        raise RuntimeError("AgentDyn checkout contains untracked files")

    entries = _base_tree_entries(root)
    expected_tree = _derive_expected_patched_tree_sha256(
        root, patch_path, entries
    )
    if expected_tree != PATCHED_TREE_SHA256:
        raise RuntimeError("AgentDyn pinned patched-tree checksum is inconsistent")
    observed_overrides: dict[str, tuple[str, str]] = {}
    for path_name in patch_paths:
        observed_path = root / path_name
        if not observed_path.exists():
            raise RuntimeError("AgentDyn reviewed patched tree is incomplete")
        observed_overrides[path_name] = _file_identity(observed_path)
    observed_tree = _logical_tree_sha256(entries, observed_overrides)
    if observed_tree != expected_tree:
        raise RuntimeError("AgentDyn checkout is not exactly the reviewed patched tree")
    return {
        "repository": UPSTREAM_REPOSITORY,
        "commit": head,
        "version": UPSTREAM_VERSION,
        "license": UPSTREAM_LICENSE,
        "license_sha256": UPSTREAM_LICENSE_SHA256,
        "patch_sha256": PATCH_SHA256,
        "template_sha256": TEMPLATE_SHA256,
        "patched_tree_sha256": expected_tree,
    }


def required_vllm_server_arguments(
    config: AgentDynConfig,
    *,
    template: str | Path,
) -> list[str]:
    """Return the required endpoint-free vLLM serving argument fragment."""

    template_path = Path(template).expanduser().resolve()
    if not template_path.is_file() or _sha256(template_path) != TEMPLATE_SHA256:
        raise RuntimeError("AgentDyn template checksum differs from the reviewed template")
    arguments: list[str] = []
    if config.language_model_only:
        arguments.append("--language-model-only")
    if config.enable_auto_tool_choice:
        arguments.append("--enable-auto-tool-choice")
    arguments.extend(
        [
            "--tool-call-parser",
            config.tool_call_parser,
            "--reasoning-parser",
            config.reasoning_parser,
            "--chat-template",
            config.chat_template,
            "--default-chat-template-kwargs",
            json.dumps(
                {
                    "enable_thinking": config.enable_thinking,
                    "preserve_thinking": config.preserve_thinking,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            "--generation-config",
            config.generation_config,
        ]
    )
    return arguments


def _content_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                content = item.get("content") or item.get("text")
                if content is not None:
                    parts.append(str(content))
        return "\n".join(parts)
    return str(value)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _assistant_tool_calls(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for call in message.get("tool_calls") or ():
        function_value = _field(call, "function")
        if isinstance(function_value, Mapping) or hasattr(function_value, "name"):
            name = _field(function_value, "name")
            arguments = _field(function_value, "arguments", {})
        else:
            name = function_value
            arguments = _field(call, "args", {})
        if not isinstance(name, str) or not name:
            raise ValueError("AgentDyn assistant tool call has no function name")
        if isinstance(arguments, str):
            try:
                arguments_value = json.loads(arguments)
            except json.JSONDecodeError as error:
                raise ValueError("AgentDyn assistant tool arguments are invalid") from error
        else:
            arguments_value = arguments
        if not isinstance(arguments_value, Mapping):
            raise ValueError("AgentDyn assistant tool arguments must be an object")
        call_id = _field(call, "id") or f"call_{uuid.uuid4().hex[:24]}"
        result.append(
            {
                "id": str(call_id),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(dict(arguments_value)),
                },
            }
        )
    return result


def messages_to_qwen_input_roles(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Map trusted system/user text to ``user`` and tool results to ``input``."""

    converted: list[dict[str, Any]] = []
    trusted_prefix: list[str] = []
    for message in messages:
        role = str(message.get("role", ""))
        content = _content_text(message.get("content"))
        if role == "system":
            if content:
                trusted_prefix.append(content)
            continue
        if role == "user":
            joined = "\n\n".join([*trusted_prefix, content]).strip()
            trusted_prefix.clear()
            converted.append({"role": "user", "content": joined})
            continue
        if trusted_prefix:
            converted.append({"role": "user", "content": "\n\n".join(trusted_prefix)})
            trusted_prefix.clear()
        if role == "tool":
            if message.get("error") is not None:
                content = json.dumps({"error": str(message["error"])})
            wrapped = f"<tool_response>\n{content or 'Success'}\n</tool_response>"
            result = {"role": "input", "content": wrapped}
            tool_call = message.get("tool_call")
            call_id = message.get("tool_call_id") or _field(tool_call, "id")
            if call_id:
                result["tool_call_id"] = str(call_id)
            converted.append(result)
        elif role == "assistant":
            reasoning = message.get("reasoning") or message.get("reasoning_content")
            if reasoning:
                content = f"<think>\n{reasoning}\n</think>\n\n{content}"
            result = {"role": "assistant", "content": content or None}
            calls = _assistant_tool_calls(message)
            if calls:
                result["tool_calls"] = calls
            converted.append(result)
        else:
            raise ValueError(f"unsupported AgentDyn message role: {role!r}")
    if trusted_prefix:
        converted.append({"role": "user", "content": "\n\n".join(trusted_prefix)})
    return converted


@dataclass(frozen=True)
class ParsedToolCall:
    function: str
    args: Mapping[str, Any]
    id: str


@dataclass(frozen=True)
class NormalizedAssistantResponse:
    visible_content: str | None
    reasoning: str | None
    tool_calls: tuple[ParsedToolCall, ...]


def _parse_parameter_value(raw_value: str) -> Any:
    value = raw_value.strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_qwen_xml_tool_calls(completion: str) -> tuple[ParsedToolCall, ...]:
    """Parse Qwen XML calls when a server returns them as ordinary text."""

    calls: list[ParsedToolCall] = []
    for tool_match in re.finditer(
        r"<tool_call>\s*(.*?)\s*</tool_call>", completion, re.DOTALL
    ):
        function_match = re.search(
            r"<function\s*=\s*([^>]+)>\s*(.*?)\s*</function>",
            tool_match.group(1),
            re.DOTALL,
        )
        if function_match is None:
            continue
        arguments = {
            match.group(1).strip(): _parse_parameter_value(match.group(2))
            for match in re.finditer(
                r"<parameter\s*=\s*([^>]+)>\s*(.*?)\s*</parameter>",
                function_match.group(2),
                re.DOTALL,
            )
        }
        calls.append(
            ParsedToolCall(
                function=function_match.group(1).strip(),
                args=arguments,
                id=f"call_{uuid.uuid4().hex[:24]}",
            )
        )
    return tuple(calls)


def _response_field(message: Any, name: str) -> Any:
    direct = _field(message, name)
    if direct is not None:
        return direct
    extras = _field(message, "model_extra")
    if isinstance(extras, Mapping):
        return extras.get(name)
    return None


def _separate_reasoning(
    content: str | None,
    *,
    explicit_reasoning: str | None,
    enable_thinking: bool,
) -> tuple[str | None, str | None]:
    text = content or ""
    inline_reasoning: str | None = None
    visible = text

    if "<think>" in text:
        prefix, _, after_open = text.partition("<think>")
        if "</think>" in after_open:
            tagged_reasoning, _, after_close = after_open.partition("</think>")
            inline_reasoning = "\n".join(
                part for part in (prefix, tagged_reasoning) if part
            )
            visible = after_close.strip()
        else:
            inline_reasoning = after_open
            visible = prefix.strip()
    elif "</think>" in text:
        inline_reasoning, _, visible = text.partition("</think>")
        visible = visible.strip()
    elif enable_thinking and explicit_reasoning is None:
        tool_index = text.find("<tool_call>")
        if tool_index >= 0:
            inline_reasoning = text[:tool_index]
            visible = text[tool_index:]
        else:
            inline_reasoning = text
            visible = ""

    reasoning = explicit_reasoning if explicit_reasoning else inline_reasoning
    normalized_reasoning = reasoning.strip() if reasoning and reasoning.strip() else None
    normalized_visible = visible.strip() if visible and visible.strip() else None
    return normalized_reasoning, normalized_visible


def _native_tool_calls(message: Any) -> tuple[ParsedToolCall, ...]:
    calls: list[ParsedToolCall] = []
    for call in _response_field(message, "tool_calls") or ():
        function_value = _field(call, "function")
        name = _field(function_value, "name")
        raw_arguments = _field(function_value, "arguments")
        if not isinstance(name, str) or not name:
            continue
        try:
            arguments = (
                json.loads(raw_arguments)
                if isinstance(raw_arguments, str)
                else raw_arguments
            )
        except json.JSONDecodeError:
            continue
        if not isinstance(arguments, Mapping):
            continue
        calls.append(
            ParsedToolCall(
                function=name,
                args=dict(arguments),
                id=str(_field(call, "id") or f"call_{uuid.uuid4().hex[:24]}"),
            )
        )
    return tuple(calls)


def normalize_assistant_response(
    message: Any,
    *,
    enable_thinking: bool,
) -> NormalizedAssistantResponse:
    """Separate reasoning and recover native or XML tool calls fail-closed."""

    explicit_text: str | None = None
    for field_name in ("reasoning", "reasoning_content"):
        explicit = _response_field(message, field_name)
        candidate = _content_text(explicit) if explicit is not None else ""
        if candidate.strip():
            explicit_text = candidate
            break
    content_value = _response_field(message, "content")
    content = _content_text(content_value) if content_value is not None else None
    reasoning, visible = _separate_reasoning(
        content,
        explicit_reasoning=explicit_text,
        enable_thinking=enable_thinking,
    )

    calls = _native_tool_calls(message)
    if not calls and visible and "<tool_call>" in visible:
        calls = parse_qwen_xml_tool_calls(visible)
        if calls:
            prefix = visible.split("<tool_call>", 1)[0].strip()
            visible = prefix or None

    if visible is None and not calls:
        raise InvalidModelOutputError("AgentDyn model returned no visible response")
    return NormalizedAssistantResponse(
        visible_content=visible,
        reasoning=reasoning,
        tool_calls=calls,
    )


def bounded_retry(
    operation: Callable[[], Any],
    *,
    max_attempts: int,
    max_wait_seconds: float,
    non_retryable_exceptions: tuple[type[BaseException], ...] = (),
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Retry transient request failures with a finite exponential schedule."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if max_wait_seconds < 0:
        raise ValueError("max_wait_seconds must be nonnegative")
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as error:
            if isinstance(error, non_retryable_exceptions) or attempt == max_attempts:
                raise
            sleep(min(max_wait_seconds, float(2 ** (attempt - 1))))
    raise AssertionError("unreachable retry state")


def _selected_attacks(
    config: AgentDynConfig,
    attacks: Sequence[str] | None,
) -> tuple[str, ...]:
    selected = tuple(attacks) if attacks is not None else config.attacks
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("selected AgentDyn attacks must be non-empty and unique")
    unsupported = set(selected) - set(config.attacks)
    if unsupported:
        raise ValueError(f"AgentDyn attacks are not configured: {sorted(unsupported)!r}")
    return selected


def build_benchmark_command(
    *,
    config: AgentDynConfig,
    log_dir: Path,
    attack: str,
    suites: Sequence[str] | None = None,
) -> list[str]:
    """Build one patched upstream command for one scalar attack condition."""

    if attack not in config.attacks:
        raise ValueError(f"AgentDyn attack is not configured: {attack!r}")
    selected_suites = tuple(suites) if suites is not None else config.suites
    if not selected_suites or len(selected_suites) != len(set(selected_suites)):
        raise ValueError("selected AgentDyn suites must be non-empty and unique")
    unsupported = set(selected_suites) - set(config.suites)
    if unsupported:
        raise ValueError(f"AgentDyn suites are not configured: {sorted(unsupported)!r}")
    command = [
        sys.executable,
        "-m",
        "agentdojo.scripts.benchmark",
        "--model",
        "local",
        "--model-id",
        config.model_name,
        "--logdir",
        str(log_dir),
        "--benchmark-version",
        config.benchmark_version,
        "--tool-delimiter",
        "input",
        "--max-workers",
        str(config.max_workers),
        "--module-to-load",
        "secpo.evaluation.agentdyn_qwen36",
    ]
    if config.defense != "none":
        command.extend(("--defense", config.defense))
    if attack != "none":
        command.extend(("--attack", attack))
    for suite in selected_suites:
        command.extend(("--suite", suite))
    return command


def build_benchmark_commands(
    *,
    config: AgentDynConfig,
    log_dir: Path,
    suites: Sequence[str] | None = None,
    attacks: Sequence[str] | None = None,
) -> list[tuple[str, list[str]]]:
    """Build one upstream command per selected scalar attack."""

    return [
        (
            attack,
            build_benchmark_command(
                config=config,
                log_dir=log_dir,
                attack=attack,
                suites=suites,
            ),
        )
        for attack in _selected_attacks(config, attacks)
    ]


def runtime_environment(config: AgentDynConfig) -> dict[str, str]:
    """Build a child environment with only the configured model credential."""

    return build_subprocess_environment(
        pass_env=(config.api_key_env,),
        overrides={
            "AGENTDYN_ENABLE_AUTO_INSTALL": "true",
            "AGENTDYN_LOCAL_BASE_URL": config.endpoint(),
            "AGENTDYN_LOCAL_CREDENTIAL_ENV": config.api_key_env,
            "AGENTDYN_LOCAL_ENABLE_THINKING": str(
                config.enable_thinking
            ).lower(),
            "AGENTDYN_LOCAL_PRESERVE_THINKING": str(
                config.preserve_thinking
            ).lower(),
            "AGENTDYN_LOCAL_MAX_TOKENS": str(config.max_output_tokens),
            "AGENTDYN_LOCAL_TEMPERATURE": str(config.temperature),
            "AGENTDYN_LOCAL_TOP_P": str(config.top_p),
            "AGENTDYN_LOCAL_TOP_K": str(config.top_k),
            "AGENTDYN_LOCAL_PRESENCE_PENALTY": str(config.presence_penalty),
            "AGENTDYN_LOCAL_SEED": str(config.seed),
            "AGENTDYN_LOCAL_REQUEST_TIMEOUT": str(
                config.request_timeout_seconds
            ),
            "AGENTDYN_LOCAL_REQUEST_MAX_ATTEMPTS": str(
                config.request_max_attempts
            ),
            "AGENTDYN_LOCAL_RETRY_MAX_WAIT_SECONDS": str(
                config.request_retry_max_wait_seconds
            ),
        },
        approved_sensitive_names=("AGENTDYN_LOCAL_CREDENTIAL_ENV",),
    )


def install() -> None:
    """Install a native-tool local-provider adapter into the patched AgentDyn."""

    import openai
    from agentdojo.agent_pipeline import agent_pipeline
    from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
    from agentdojo.functions_runtime import EmptyEnv, FunctionCall
    from agentdojo.types import ChatAssistantMessage, text_content_block_from_string

    current = agent_pipeline.get_llm
    if getattr(current, _INSTALLED_MARKER, False):
        return
    original = current

    class Qwen36AgentDynLLM(BasePipelineElement):
        def __init__(self, client: Any, model: str) -> None:
            self.client = client
            self.model = model

        def query(
            self,
            query: str,
            runtime: Any,
            env: Any = EmptyEnv(),
            messages: Sequence[Any] = (),
            extra_args: Mapping[str, Any] | None = None,
        ) -> tuple[Any, ...]:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters.model_json_schema(),
                    },
                }
                for tool in runtime.functions.values()
            ]
            request: dict[str, Any] = {
                "model": self.model,
                "messages": messages_to_qwen_input_roles(messages),
                "max_tokens": int(os.environ["AGENTDYN_LOCAL_MAX_TOKENS"]),
                "temperature": float(os.environ["AGENTDYN_LOCAL_TEMPERATURE"]),
                "top_p": float(os.environ["AGENTDYN_LOCAL_TOP_P"]),
                "presence_penalty": float(
                    os.environ["AGENTDYN_LOCAL_PRESENCE_PENALTY"]
                ),
                "seed": int(os.environ["AGENTDYN_LOCAL_SEED"]),
                "extra_body": {
                    "top_k": int(os.environ["AGENTDYN_LOCAL_TOP_K"]),
                    "chat_template_kwargs": {
                        "enable_thinking": os.environ[
                            "AGENTDYN_LOCAL_ENABLE_THINKING"
                        ]
                        == "true",
                        "preserve_thinking": os.environ[
                            "AGENTDYN_LOCAL_PRESERVE_THINKING"
                        ]
                        == "true",
                    },
                },
            }
            if tools:
                request.update(tools=tools, tool_choice="auto")

            def create_message() -> Any:
                completion = self.client.chat.completions.create(**request)
                choices = _field(completion, "choices")
                if not choices:
                    raise InvalidModelOutputError(
                        "AgentDyn completion contained no choices"
                    )
                message = _field(choices[0], "message")
                if message is None:
                    raise InvalidModelOutputError(
                        "AgentDyn completion contained no message"
                    )
                return message

            response = bounded_retry(
                create_message,
                max_attempts=int(
                    os.environ["AGENTDYN_LOCAL_REQUEST_MAX_ATTEMPTS"]
                ),
                max_wait_seconds=float(
                    os.environ["AGENTDYN_LOCAL_RETRY_MAX_WAIT_SECONDS"]
                ),
                non_retryable_exceptions=(openai.BadRequestError,),
            )
            normalized = normalize_assistant_response(
                response,
                enable_thinking=(
                    os.environ["AGENTDYN_LOCAL_ENABLE_THINKING"] == "true"
                ),
            )
            calls = [
                FunctionCall(
                    function=call.function,
                    args=dict(call.args),
                    id=call.id,
                )
                for call in normalized.tool_calls
            ]
            output = ChatAssistantMessage(
                role="assistant",
                content=(
                    [text_content_block_from_string(normalized.visible_content)]
                    if normalized.visible_content
                    else None
                ),
                tool_calls=calls,
            )
            if normalized.reasoning:
                output["reasoning"] = normalized.reasoning
            return query, runtime, env, [*messages, output], dict(extra_args or {})

    def get_llm(
        provider: str,
        model: str,
        model_id: str | None,
        tool_delimiter: str,
    ) -> Any:
        if provider != "local":
            return original(provider, model, model_id, tool_delimiter)
        base_url = _validate_endpoint(os.environ["AGENTDYN_LOCAL_BASE_URL"])
        credential_name = os.environ["AGENTDYN_LOCAL_CREDENTIAL_ENV"]
        credential = os.getenv(credential_name, "") or ("local" + "-only")
        client = openai.OpenAI(
            api_key=credential,
            base_url=base_url,
            timeout=float(os.environ["AGENTDYN_LOCAL_REQUEST_TIMEOUT"]),
            max_retries=0,
        )
        return Qwen36AgentDynLLM(client, model_id or QWEN36_MODEL)

    setattr(get_llm, _INSTALLED_MARKER, True)
    agent_pipeline.get_llm = get_llm


if os.getenv("AGENTDYN_ENABLE_AUTO_INSTALL", "").casefold() in {
    "1",
    "true",
    "yes",
}:
    install()
