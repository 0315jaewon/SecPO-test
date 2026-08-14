#!/usr/bin/env python3
"""Prepare, run, or score the paper-main static SEP attacks.

The optional generator and judge are ordinary local Python callables named as
``package.module:attribute``.  A generator accepts a batch of message lists and
returns strings.  A judge accepts witness-positive result rows and returns
booleans.  This script never configures a hosted provider.

Examples::

    # Portable JSONL requests for a separate inference harness.
    python scripts/evaluate_sep_static.py prepare \
      --input SEP_dataset_test.json --output requests.jsonl \
      --model-family llama31_8b

    # Local callable inference; witness hits remain pending without --judge.
    python scripts/evaluate_sep_static.py run \
      --input SEP_dataset_test.json --output predictions.jsonl \
      --model-family qwen36_27b --generator local_model:generate

    # Score JSONL after an external harness has added an `output` field.
    python scripts/evaluate_sep_static.py score \
      --input predictions.jsonl --output scored.jsonl \
      --judge local_judge:judge
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any

from secpo.evaluation.sep_static import (
    ATTACK_REGISTRY,
    build_static_sep_requests,
    evaluate_static_sep,
    load_json_records,
    score_static_sep_rows,
    summarize_static_sep,
    write_jsonl,
)


def _load_callable(specification: str, label: str) -> Any:
    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError(f"{label} must use package.module:attribute syntax")
    module = importlib.import_module(module_name)
    value: Any = module
    for component in attribute_name.split("."):
        if not component or component.startswith("_"):
            raise ValueError(f"invalid {label} attribute {attribute_name!r}")
        value = getattr(value, component)
    if not callable(value):
        raise TypeError(f"{specification!r} does not resolve to a callable")
    return value


def _add_attack_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-family",
        required=True,
        choices=("llama31_8b", "qwen36_27b"),
        help="victim family; selects its two basic-adaptive delimiter attacks",
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        choices=tuple(ATTACK_REGISTRY),
        default=None,
        help="explicit subset; default is six static plus two model-specific attacks",
    )
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument(
        "--flat-user-message",
        action="store_true",
        help="merge trusted instruction and untrusted input into one user message",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="write attack requests as JSONL")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    _add_attack_arguments(prepare)

    run = subparsers.add_parser("run", help="run a local victim callable")
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--generator", required=True)
    run.add_argument("--judge")
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--judge-batch-size", type=int, default=128)
    _add_attack_arguments(run)

    score = subparsers.add_parser("score", help="gate and optionally judge JSONL")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--judge")
    score.add_argument("--judge-batch-size", type=int, default=128)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    records = load_json_records(args.input)

    if args.command == "prepare":
        rows = build_static_sep_requests(
            records,
            model_family=args.model_family,
            attack_names=args.attacks,
            seed=args.seed,
            instruction_hierarchy=not args.flat_user_message,
        )
    elif args.command == "run":
        generator = _load_callable(args.generator, "generator")
        judge = _load_callable(args.judge, "judge") if args.judge else None
        rows = evaluate_static_sep(
            records,
            generator,
            model_family=args.model_family,
            attack_names=args.attacks,
            seed=args.seed,
            instruction_hierarchy=not args.flat_user_message,
            batch_size=args.batch_size,
            judge=judge,
            judge_batch_size=args.judge_batch_size,
        )
    else:
        judge = _load_callable(args.judge, "judge") if args.judge else None
        rows = score_static_sep_rows(
            records,
            judge=judge,
            judge_batch_size=args.judge_batch_size,
        )

    write_jsonl(rows, args.output)
    summary = summarize_static_sep(rows)
    print(json.dumps(summary, indent=2, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
