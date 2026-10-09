"""Command-line entry point for model-agnostic SecPO training on Tinker."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

from .config import load_tinker_config, parse_tinker_config
from .runner import run_secpo_training


DEFAULT_CONFIG = Path("configs/train_qwen38_27b_tinker.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train benign-reference SecPO with the Tinker backend."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Strict JSON run configuration.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the full dataset and batch plan without contacting Tinker.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Override max_steps for a small smoke run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_tinker_config(args.config)
    if args.max_steps is not None:
        raw = asdict(config)
        raw["max_steps"] = args.max_steps
        config = parse_tinker_config(raw)
    result = run_secpo_training(config, preflight_only=args.preflight_only)
    print(
        json.dumps(
            {
                "preflight_only": args.preflight_only,
                "records": result.plan.record_count,
                "total_steps": result.plan.total_steps,
                "steps_completed": result.steps_completed,
                "final_state_path": result.final_state_path,
                "final_sampler_path": result.final_sampler_path,
                "metrics_path": result.metrics_path,
                "token_preflight": (
                    asdict(result.token_preflight)
                    if result.token_preflight is not None
                    else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
