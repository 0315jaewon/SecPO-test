"""Strict lm-eval command builder for the released utility suites.

The actual benchmark datasets and ``lm-evaluation-harness`` remain external
dependencies with their own licenses.  This wrapper freezes the task IDs,
few-shot counts, seeds, model arguments, and output layout used by a run.  It
prints commands by default and executes them only with an explicit flag.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any, Sequence

from secpo.subprocess_env import build_subprocess_environment


@dataclass(frozen=True)
class UtilityTask:
    """One independently resumable lm-eval task invocation."""

    task: str
    num_fewshot: int
    metric: str


@dataclass(frozen=True)
class UtilityConfig:
    """Validated utility-evaluation configuration."""

    model_name_or_path: str
    model_revision: str
    tasks: tuple[UtilityTask, ...]
    batch_size: str
    dtype: str
    seed: int
    trust_remote_code: bool
    apply_chat_template: bool


_CONFIG_FIELDS = {
    "model_name_or_path",
    "model_revision",
    "tasks",
    "batch_size",
    "dtype",
    "seed",
    "trust_remote_code",
    "apply_chat_template",
}
_TASK_FIELDS = {"task", "num_fewshot", "metric"}
_MODEL_REVISION = re.compile(r"^[0-9a-f]{40}$")


def _public_model_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("model_name_or_path must be a public model id")
    value = value.strip()
    if len(value.split("/")) != 2 or Path(value).is_absolute():
        raise ValueError("model_name_or_path must use organization/model form")
    if any(character in value for character in ",="):
        raise ValueError("model_name_or_path contains unsupported punctuation")
    return value


def parse_utility_config(value: Any) -> UtilityConfig:
    """Strictly validate a decoded utility JSON object."""

    if not isinstance(value, dict) or set(value) != _CONFIG_FIELDS:
        raise ValueError("utility config has missing or unknown fields")
    model_id = _public_model_id(value["model_name_or_path"])
    model_revision = value["model_revision"]
    if not isinstance(model_revision, str) or not _MODEL_REVISION.fullmatch(
        model_revision
    ):
        raise ValueError("model_revision must be an immutable 40-character commit")
    rows = value["tasks"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("tasks must be a non-empty list")
    tasks: list[UtilityTask] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != _TASK_FIELDS:
            raise ValueError(f"tasks[{index}] has missing or unknown fields")
        task = row["task"]
        metric = row["metric"]
        fewshot = row["num_fewshot"]
        if (
            not isinstance(task, str)
            or not task.strip()
            or any(character in task for character in ",= /\\")
        ):
            raise ValueError(f"tasks[{index}].task is invalid")
        if task in seen:
            raise ValueError(f"duplicate task {task!r}")
        seen.add(task)
        if type(fewshot) is not int or fewshot < 0:
            raise ValueError(f"tasks[{index}].num_fewshot must be non-negative")
        if not isinstance(metric, str) or not metric.strip():
            raise ValueError(f"tasks[{index}].metric must be non-empty")
        tasks.append(UtilityTask(task=task, num_fewshot=fewshot, metric=metric))
    batch_size = value["batch_size"]
    if not isinstance(batch_size, str) or not batch_size.strip():
        raise ValueError("batch_size must be a non-empty lm-eval value")
    if any(character in batch_size for character in " \t\n"):
        raise ValueError("batch_size must not contain whitespace")
    if value["dtype"] not in {"bfloat16", "float16", "float32"}:
        raise ValueError("dtype must be bfloat16, float16, or float32")
    if type(value["seed"]) is not int or value["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    for field in ("trust_remote_code", "apply_chat_template"):
        if type(value[field]) is not bool:
            raise ValueError(f"{field} must be a boolean")
    return UtilityConfig(
        model_name_or_path=model_id,
        model_revision=model_revision,
        tasks=tuple(tasks),
        batch_size=batch_size,
        dtype=value["dtype"],
        seed=value["seed"],
        trust_remote_code=value["trust_remote_code"],
        apply_chat_template=value["apply_chat_template"],
    )


def load_utility_config(path: str | Path) -> tuple[UtilityConfig, str]:
    """Load a config and return it with a stable SHA-256 fingerprint."""

    source = Path(path)
    raw = source.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid utility config JSON: {error.msg}") from error
    return parse_utility_config(value), hashlib.sha256(raw).hexdigest()


def build_lm_eval_commands(
    config: UtilityConfig,
    *,
    output_dir: str | Path,
    adapter_path: str | Path | None = None,
    python_executable: str = sys.executable,
) -> list[list[str]]:
    """Return one explicit, independently resumable command per task."""

    output = Path(output_dir)
    model_arguments = [
        f"pretrained={config.model_name_or_path}",
        f"revision={config.model_revision}",
        f"dtype={config.dtype}",
        f"trust_remote_code={str(config.trust_remote_code).lower()}",
    ]
    if adapter_path is not None:
        adapter = str(adapter_path)
        if not adapter or any(character in adapter for character in ",\n\r"):
            raise ValueError("adapter_path must not be empty or contain commas/newlines")
        model_arguments.append(f"peft={adapter}")

    commands: list[list[str]] = []
    for task in config.tasks:
        command = [
            python_executable,
            "-m",
            "lm_eval",
            "--model",
            "hf",
            "--model_args",
            ",".join(model_arguments),
            "--tasks",
            task.task,
            "--num_fewshot",
            str(task.num_fewshot),
            "--batch_size",
            config.batch_size,
            "--seed",
            str(config.seed),
            "--output_path",
            str(output / task.task),
            "--log_samples",
        ]
        if config.apply_chat_template:
            command.append("--apply_chat_template")
        commands.append(command)
    return commands


def _parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--adapter-path", type=Path)
    result.add_argument(
        "--execute",
        action="store_true",
        help="Run commands. Without this flag they are printed only.",
    )
    result.add_argument(
        "--pass-env",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Pass exactly this parent environment variable to lm-eval; "
            "repeat for additional names. No credentials are passed by default."
        ),
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config, fingerprint = load_utility_config(args.config)
    commands = build_lm_eval_commands(
        config,
        output_dir=args.output_dir,
        adapter_path=args.adapter_path,
    )
    print(f"utility_config_sha256={fingerprint}")
    for command in commands:
        print(shlex.join(command))
    if not args.execute:
        return 0

    environment = build_subprocess_environment(
        pass_env=tuple(args.pass_env),
        require_pass_env=True,
        overrides={
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DISABLE_TELEMETRY": "1",
        },
    )
    for command in commands:
        subprocess.run(command, check=True, env=environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
