#!/usr/bin/env python3
"""Verify, shard, run, or summarize the pinned Qwen3.6-27B DTap protocol."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
from typing import Sequence

from secpo.evaluation.dtap_qwen_agent_backend import (
    DTAP_ALLOWED_PASS_ENV,
    DTAP_PASS_ENV_SELECTION,
)
from secpo.evaluation.dtap_qwen36 import (
    DEFAULT_TASK_LIST_PATH,
    MultiProcessExecutionError,
    ShardExecutionError,
    activated_shard_runtime,
    launch_all_shards,
    load_backend,
    load_config,
    prepare_shards,
    run_shard,
    summarize_results,
    verify_external_sources,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "evaluate_dtap_qwen36_27b.json"
TASK_LIST = ROOT / DEFAULT_TASK_LIST_PATH


@contextmanager
def _selected_pass_environment(names: Sequence[str]):
    """Activate only explicitly requested DTap victim-service credentials."""

    selected = tuple(names)
    if len(set(selected)) != len(selected):
        raise ValueError("--pass-env must not repeat a name")
    unsupported = sorted(set(selected) - set(DTAP_ALLOWED_PASS_ENV))
    if unsupported:
        raise ValueError(f"unsupported --pass-env names: {unsupported}")
    for name in selected:
        if not os.getenv(name, "").strip():
            raise RuntimeError(f"--pass-env requested an unset variable: {name}")
    previous = os.environ.get(DTAP_PASS_ENV_SELECTION)
    if selected:
        os.environ[DTAP_PASS_ENV_SELECTION] = ",".join(selected)
    else:
        os.environ.pop(DTAP_PASS_ENV_SELECTION, None)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(DTAP_PASS_ENV_SELECTION, None)
        else:
            os.environ[DTAP_PASS_ENV_SELECTION] = previous


def _add_pass_env(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pass-env",
        action="append",
        default=[],
        choices=DTAP_ALLOWED_PASS_ENV,
        metavar="NAME",
        help=(
            "Pass the optional victim-service credential explicitly. "
            "No victim credential is inherited by default."
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("verify", help="verify exact external source inputs")
    verify.add_argument("--checkout", type=Path, required=True)
    verify.add_argument("--patch", type=Path, required=True)
    verify.add_argument("--template", type=Path, required=True)
    verify.add_argument("--qwen-agent-license", type=Path, required=True)

    prepare = commands.add_parser(
        "prepare-shards", help="create the deterministic 33-shard task plan"
    )
    prepare.add_argument(
        "--task-list",
        type=Path,
        default=TASK_LIST,
        help="exact task-selection JSONL (default: bundled audited manifest)",
    )
    prepare.add_argument("--shards-dir", type=Path, required=True)

    run = commands.add_parser("run-shard", help="run one six-task-parallel CPU shard")
    run.add_argument("--shards-dir", type=Path, required=True)
    run.add_argument("--results-dir", type=Path, required=True)
    run.add_argument("--shard-id", type=int, required=True)
    run.add_argument("--backend", required=True, help="local module:callable task backend")
    _add_pass_env(run)

    run_all = commands.add_parser(
        "run-all-shards",
        help="launch all 33 CPU shard processes with six task threads each",
    )
    run_all.add_argument("--shards-dir", type=Path, required=True)
    run_all.add_argument("--results-dir", type=Path, required=True)
    run_all.add_argument("--backend", required=True, help="local module:callable task backend")
    _add_pass_env(run_all)

    summarize = commands.add_parser("summarize", help="aggregate fail-closed scores")
    summarize.add_argument("--shards-dir", type=Path, required=True)
    summarize.add_argument("--results-dir", type=Path, required=True)
    summarize.add_argument("--output", type=Path)
    return parser


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "verify":
        _print(
            verify_external_sources(
                args.checkout,
                patch=args.patch,
                template=args.template,
                qwen_agent_license=args.qwen_agent_license,
            )
        )
        return 0
    if args.command == "prepare-shards":
        _print(prepare_shards(args.task_list, args.shards_dir, config=config))
        return 0
    if args.command == "run-shard":
        with _selected_pass_environment(args.pass_env):
            with activated_shard_runtime(args.shard_id):
                backend = load_backend(args.backend)
                try:
                    report = run_shard(
                        args.shards_dir,
                        args.results_dir,
                        shard_id=args.shard_id,
                        config=config,
                        backend=backend,
                    )
                except ShardExecutionError as exc:
                    _print(exc.report)
                    return 2
        _print(report)
        return 0
    if args.command == "run-all-shards":
        with _selected_pass_environment(args.pass_env):
            try:
                report = launch_all_shards(
                    args.shards_dir,
                    args.results_dir,
                    config=config,
                    backend_specification=args.backend,
                )
            except MultiProcessExecutionError as exc:
                _print(exc.report)
                return 2
        _print(report)
        return 0
    summary = summarize_results(args.shards_dir, args.results_dir, config=config)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    _print(summary)
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
