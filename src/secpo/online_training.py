"""Portable file protocol for Llama online SecPO attack generation.

The original online experiment ran the defender and attacker on separate GPU
groups and exchanged per-rank JSON files.  This module keeps that useful
separation while making the contract explicit, validated, atomic, and bounded
by a timeout.  It deliberately contains no model-provider or scheduler code.

An attack worker reads a request, generates one attacked prompt per example,
and writes the matching response.  :class:`OnlineAttackCollator` then replaces
only the policy prompt before delegating to the normal benign-reference
collator; the reference prompt and preference responses remain unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import importlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any
from uuid import uuid4

PROTOCOL_VERSION = 1
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class OnlineAttackError(RuntimeError):
    """Raised when the online attack exchange fails or violates its schema."""


@dataclass(frozen=True)
class AttackExample:
    """One example sent to the training-time attack worker."""

    id: str
    seed_prompt: str
    clean_prompt: str
    chosen: str
    rejected: str
    attack_goal: str | None = None
    witness: str | None = None


@dataclass(frozen=True)
class AttackRequest:
    """Versioned request written by one defender data-loader rank."""

    protocol_version: int
    request_id: str
    rank: int
    batch_index: int
    examples: tuple[AttackExample, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["examples"] = [asdict(example) for example in self.examples]
        return value


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OnlineAttackError(f"{name} must be a non-empty string")
    if "\x00" in value:
        raise OnlineAttackError(f"{name} must not contain a NUL byte")
    return value


def _require_identifier(value: Any, name: str) -> str:
    value = _require_text(value, name)
    if not _SAFE_IDENTIFIER.fullmatch(value):
        raise OnlineAttackError(
            f"{name} must match {_SAFE_IDENTIFIER.pattern!r}"
        )
    return value


def _load_json_object(path: Path) -> dict[str, Any]:
    def no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise OnlineAttackError(f"duplicate JSON key {key!r} in {path.name}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=no_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OnlineAttackError(f"could not decode {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise OnlineAttackError(f"{path.name} must contain one JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _regular_directory(path.parent)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"
    descriptor: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _regular_directory(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise OnlineAttackError(f"exchange path must be a regular directory: {path}")
    # Prompts and preference responses are sensitive experimental data. Do not
    # rely on a shared-cluster umask, which is commonly world-readable.
    try:
        os.chmod(path, 0o700, follow_symlinks=False)
    except (NotImplementedError, TypeError):  # pragma: no cover - platform fallback
        path.chmod(0o700)
    if path.stat().st_mode & 0o077:
        raise OnlineAttackError(
            f"exchange directory must not be accessible to group/other users: {path}"
        )
    return path


def validate_attacked_prompt(
    attacked_prompt: str,
    clean_prompt: str,
    *,
    example_id: str,
) -> str:
    """Require an attacked prompt to add exactly one contiguous span.

    This matches the released benign-reference data contract and prevents a
    worker from silently changing trusted prompt content.
    """

    attacked_prompt = _require_text(attacked_prompt, f"attack for {example_id}")
    clean_prompt = _require_text(clean_prompt, f"clean prompt for {example_id}")
    prefix_length = 0
    shared_limit = min(len(attacked_prompt), len(clean_prompt))
    while (
        prefix_length < shared_limit
        and attacked_prompt[prefix_length] == clean_prompt[prefix_length]
    ):
        prefix_length += 1
    suffix_length = 0
    clean_remaining = len(clean_prompt) - prefix_length
    while (
        suffix_length < clean_remaining
        and attacked_prompt[len(attacked_prompt) - 1 - suffix_length]
        == clean_prompt[len(clean_prompt) - 1 - suffix_length]
    ):
        suffix_length += 1
    deleted_end = len(attacked_prompt) - suffix_length
    injection = attacked_prompt[prefix_length:deleted_end]
    reconstructed = (
        attacked_prompt[:prefix_length] + attacked_prompt[deleted_end:]
    )
    if not injection or reconstructed != clean_prompt:
        raise OnlineAttackError(
            f"attack for {example_id!r} must preserve the clean prompt and add "
            "one contiguous injection"
        )
    return attacked_prompt


class FileAttackExchange:
    """Atomic request/response exchange rooted in a caller-selected directory."""

    def __init__(
        self,
        root: str | Path,
        *,
        timeout_seconds: float = 1800.0,
        poll_seconds: float = 1.0,
        retain_files: bool = False,
    ) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if not math.isfinite(poll_seconds) or poll_seconds <= 0:
            raise ValueError("poll_seconds must be finite and positive")
        self.root = _regular_directory(Path(root).expanduser())
        self.requests = _regular_directory(self.root / "requests")
        self.responses = _regular_directory(self.root / "responses")
        self.failures = _regular_directory(self.root / "failures")
        self.timeout_seconds = float(timeout_seconds)
        self.poll_seconds = float(poll_seconds)
        self.retain_files = bool(retain_files)

    def submit(self, request: AttackRequest) -> Path:
        """Validate and atomically publish a request."""

        if request.protocol_version != PROTOCOL_VERSION:
            raise OnlineAttackError("unsupported online-attack protocol version")
        _require_identifier(request.request_id, "request_id")
        if request.rank < 0 or request.batch_index < 0:
            raise OnlineAttackError("rank and batch_index must be non-negative")
        if not request.examples:
            raise OnlineAttackError("an attack request must contain examples")
        seen: set[str] = set()
        for example in request.examples:
            _require_identifier(example.id, "example id")
            if example.id in seen:
                raise OnlineAttackError(f"duplicate example id {example.id!r}")
            seen.add(example.id)
            _require_text(example.seed_prompt, f"seed prompt for {example.id}")
            _require_text(example.clean_prompt, f"clean prompt for {example.id}")
            _require_text(example.chosen, f"chosen response for {example.id}")
            _require_text(example.rejected, f"rejected response for {example.id}")
            if example.attack_goal is not None:
                _require_text(example.attack_goal, f"attack goal for {example.id}")
            if example.witness is not None:
                _require_text(example.witness, f"witness for {example.id}")
        path = self.requests / f"{request.request_id}.json"
        if path.exists():
            raise OnlineAttackError(f"request already exists: {request.request_id}")
        _atomic_json(path, request.to_dict())
        return path

    def wait(self, request: AttackRequest) -> dict[str, str]:
        """Wait for and validate the response matching ``request``."""

        response_path = self.responses / f"{request.request_id}.json"
        failure_path = self.failures / f"{request.request_id}.json"
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            if failure_path.is_file():
                failure = _load_json_object(failure_path)
                message = failure.get("error", "attack worker reported a failure")
                raise OnlineAttackError(
                    f"attack request {request.request_id} failed: {message}"
                )
            if response_path.is_file():
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"timed out after {self.timeout_seconds:g}s waiting for "
                    f"attack request {request.request_id}"
                )
            time.sleep(min(self.poll_seconds, remaining))

        value = _load_json_object(response_path)
        if set(value) != {"protocol_version", "request_id", "attacked_prompts"}:
            raise OnlineAttackError("attack response has missing or unknown fields")
        if value["protocol_version"] != PROTOCOL_VERSION:
            raise OnlineAttackError("attack response protocol version does not match")
        if value["request_id"] != request.request_id:
            raise OnlineAttackError("attack response request_id does not match")
        rows = value["attacked_prompts"]
        if not isinstance(rows, list):
            raise OnlineAttackError("attacked_prompts must be a list")

        expected = {example.id: example for example in request.examples}
        attacked: dict[str, str] = {}
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or set(row) != {"id", "prompt"}:
                raise OnlineAttackError(
                    f"attacked_prompts[{index}] must contain only id and prompt"
                )
            example_id = _require_identifier(row["id"], "response example id")
            if example_id not in expected:
                raise OnlineAttackError(f"unexpected response id {example_id!r}")
            if example_id in attacked:
                raise OnlineAttackError(f"duplicate response id {example_id!r}")
            attacked[example_id] = validate_attacked_prompt(
                row["prompt"],
                expected[example_id].clean_prompt,
                example_id=example_id,
            )
        missing = set(expected) - set(attacked)
        if missing:
            raise OnlineAttackError(f"attack response is missing ids: {sorted(missing)}")

        if not self.retain_files:
            for path in (
                self.requests / f"{request.request_id}.json",
                response_path,
                failure_path,
            ):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
        return attacked

    def attack(self, request: AttackRequest) -> dict[str, str]:
        """Submit ``request`` and return its validated attacked prompts."""

        self.submit(request)
        return self.wait(request)


class OnlineAttackCollator:
    """Replace policy prompts through an online worker before tokenization."""

    def __init__(
        self,
        base_collator: Callable[[Sequence[Mapping[str, Any]]], Any],
        exchange: FileAttackExchange,
        *,
        rank: int = 0,
    ) -> None:
        if rank < 0:
            raise ValueError("rank must be non-negative")
        self.base_collator = base_collator
        self.exchange = exchange
        self.rank = rank
        self.batch_index = 0

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> Any:
        examples: list[AttackExample] = []
        for index, feature in enumerate(features):
            example_id = feature.get("id")
            if example_id is None:
                raise OnlineAttackError(
                    "online SecPO requires a stable id on every preference record"
                )
            examples.append(
                AttackExample(
                    id=_require_identifier(example_id, f"features[{index}].id"),
                    seed_prompt=_require_text(
                        feature.get("prompt"), f"features[{index}].prompt"
                    ),
                    clean_prompt=_require_text(
                        feature.get("clean_prompt"),
                        f"features[{index}].clean_prompt",
                    ),
                    chosen=_require_text(
                        feature.get("chosen"), f"features[{index}].chosen"
                    ),
                    rejected=_require_text(
                        feature.get("rejected"), f"features[{index}].rejected"
                    ),
                    attack_goal=feature.get("attack_goal"),
                    witness=feature.get("witness"),
                )
            )

        request_id = (
            f"rank{self.rank:04d}-batch{self.batch_index:08d}-{uuid4().hex}"
        )
        request = AttackRequest(
            protocol_version=PROTOCOL_VERSION,
            request_id=request_id,
            rank=self.rank,
            batch_index=self.batch_index,
            examples=tuple(examples),
        )
        self.batch_index += 1
        attacked = self.exchange.attack(request)

        replaced: list[dict[str, Any]] = []
        for feature, example in zip(features, examples, strict=True):
            row = dict(feature)
            row["prompt"] = attacked[example.id]
            replaced.append(row)
        return self.base_collator(replaced)


def load_backend(specification: str) -> Callable[[Mapping[str, Any]], Mapping[str, str]]:
    """Load a user-controlled ``module:callable`` attack backend."""

    if not isinstance(specification, str) or specification.count(":") != 1:
        raise ValueError("backend must use the form 'module:callable'")
    module_name, attribute_name = specification.split(":", 1)
    if not module_name or not attribute_name:
        raise ValueError("backend must use the form 'module:callable'")
    module = importlib.import_module(module_name)
    backend = getattr(module, attribute_name, None)
    if not callable(backend):
        raise TypeError(f"backend is not callable: {specification}")
    return backend


def parse_request(value: Mapping[str, Any]) -> AttackRequest:
    """Strictly parse a decoded worker request."""

    if set(value) != {
        "protocol_version",
        "request_id",
        "rank",
        "batch_index",
        "examples",
    }:
        raise OnlineAttackError("attack request has missing or unknown fields")
    if value["protocol_version"] != PROTOCOL_VERSION:
        raise OnlineAttackError("unsupported online-attack protocol version")
    request_id = _require_identifier(value["request_id"], "request_id")
    rank = value["rank"]
    batch_index = value["batch_index"]
    if type(rank) is not int or rank < 0:
        raise OnlineAttackError("rank must be a non-negative integer")
    if type(batch_index) is not int or batch_index < 0:
        raise OnlineAttackError("batch_index must be a non-negative integer")
    rows = value["examples"]
    if not isinstance(rows, list) or not rows:
        raise OnlineAttackError("examples must be a non-empty list")
    examples: list[AttackExample] = []
    allowed = {
        "id",
        "seed_prompt",
        "clean_prompt",
        "chosen",
        "rejected",
        "attack_goal",
        "witness",
    }
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != allowed:
            raise OnlineAttackError(
                f"examples[{index}] has missing or unknown fields"
            )
        examples.append(
            AttackExample(
                id=_require_identifier(row["id"], f"examples[{index}].id"),
                seed_prompt=_require_text(
                    row["seed_prompt"], f"examples[{index}].seed_prompt"
                ),
                clean_prompt=_require_text(
                    row["clean_prompt"], f"examples[{index}].clean_prompt"
                ),
                chosen=_require_text(
                    row["chosen"], f"examples[{index}].chosen"
                ),
                rejected=_require_text(
                    row["rejected"], f"examples[{index}].rejected"
                ),
                attack_goal=(
                    None
                    if row["attack_goal"] is None
                    else _require_text(
                        row["attack_goal"], f"examples[{index}].attack_goal"
                    )
                ),
                witness=(
                    None
                    if row["witness"] is None
                    else _require_text(
                        row["witness"], f"examples[{index}].witness"
                    )
                ),
            )
        )
    return AttackRequest(
        protocol_version=PROTOCOL_VERSION,
        request_id=request_id,
        rank=rank,
        batch_index=batch_index,
        examples=tuple(examples),
    )


def write_response(
    exchange: FileAttackExchange,
    request: AttackRequest,
    attacked_prompts: Mapping[str, str],
) -> None:
    """Validate and atomically write a worker response."""

    expected = {example.id: example for example in request.examples}
    if set(attacked_prompts) != set(expected):
        raise OnlineAttackError(
            "backend response ids do not match request ids; "
            f"missing={sorted(set(expected) - set(attacked_prompts))}, "
            f"unexpected={sorted(set(attacked_prompts) - set(expected))}"
        )
    rows = []
    for example in request.examples:
        prompt = validate_attacked_prompt(
            attacked_prompts[example.id],
            example.clean_prompt,
            example_id=example.id,
        )
        rows.append({"id": example.id, "prompt": prompt})
    _atomic_json(
        exchange.responses / f"{request.request_id}.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request.request_id,
            "attacked_prompts": rows,
        },
    )


def process_request_file(
    exchange: FileAttackExchange,
    path: Path,
    backend: Callable[[Mapping[str, Any]], Mapping[str, str]],
) -> None:
    """Process one request file, reporting bounded error text on failure."""

    request_id = path.stem
    if (
        exchange.responses / f"{request_id}.json"
    ).exists() or (
        exchange.failures / f"{request_id}.json"
    ).exists():
        return
    try:
        request = parse_request(_load_json_object(path))
        result = backend(request.to_dict())
        if not isinstance(result, Mapping):
            raise OnlineAttackError("backend must return a mapping of id to prompt")
        write_response(exchange, request, result)
    except Exception as error:
        # Provider exceptions can embed endpoints or credential fragments.
        # Persist only the exception type in the shared runtime protocol.
        message = type(error).__name__
        _atomic_json(
            exchange.failures / f"{request_id}.json",
            {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": request_id,
                "error": message,
            },
        )


def serve_requests(
    exchange: FileAttackExchange,
    backend: Callable[[Mapping[str, Any]], Mapping[str, str]],
    *,
    rank: int | None = None,
    once: bool = False,
    idle_timeout_seconds: float | None = None,
) -> int:
    """Serve request files until interrupted or an optional idle timeout."""

    if rank is not None and rank < 0:
        raise ValueError("rank must be non-negative")
    if idle_timeout_seconds is not None and (
        not math.isfinite(idle_timeout_seconds) or idle_timeout_seconds <= 0
    ):
        raise ValueError("idle_timeout_seconds must be finite and positive")
    pattern = "*.json" if rank is None else f"rank{rank:04d}-*.json"
    processed = 0
    last_activity = time.monotonic()
    while True:
        paths = sorted(exchange.requests.glob(pattern))
        for path in paths:
            if path.is_symlink() or not path.is_file():
                continue
            process_request_file(exchange, path, backend)
            processed += 1
            last_activity = time.monotonic()
            if once:
                return processed
        if idle_timeout_seconds is not None and (
            time.monotonic() - last_activity >= idle_timeout_seconds
        ):
            return processed
        time.sleep(exchange.poll_seconds)
