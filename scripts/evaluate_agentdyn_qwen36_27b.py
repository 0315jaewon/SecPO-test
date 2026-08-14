#!/usr/bin/env python3
"""Verify or run the pinned Qwen3.6-27B AgentDyn integration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Sequence

from secpo.evaluation.agentdyn_qwen36 import (
    build_benchmark_commands,
    load_config,
    required_vllm_server_arguments,
    runtime_environment,
    verify_agentdyn_checkout,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "evaluate_agentdyn_qwen36_27b.json"
DEFAULT_PATCH = ROOT / "third_party" / "patches" / "agentdyn.patch"
DEFAULT_TEMPLATE = ROOT / "templates" / "qwen36_agentdyn_input_role.jinja"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--patch", type=Path, default=DEFAULT_PATCH)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify")
    run = subparsers.add_parser("run")
    run.add_argument("--log-dir", type=Path, required=True)
    run.add_argument("--suite", action="append")
    run.add_argument("--attack", action="append")
    run.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    verified = verify_agentdyn_checkout(
        args.checkout,
        patch=args.patch,
        template=args.template,
    )
    server_arguments = required_vllm_server_arguments(
        config,
        template=args.template,
    )
    if args.command == "verify":
        print(
            json.dumps(
                {
                    "required_vllm_server_arguments": server_arguments,
                    "upstream": verified,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    commands = build_benchmark_commands(
        config=config,
        log_dir=args.log_dir,
        suites=args.suite,
        attacks=args.attack,
    )
    print(
        json.dumps(
            {
                "commands": [
                    {"attack": attack, "command": command}
                    for attack, command in commands
                ],
                "required_vllm_server_arguments": server_arguments,
                "upstream": verified,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.execute:
        environment = runtime_environment(config)
        python_paths = [
            str(ROOT / "src"),
            str(args.checkout.expanduser().resolve() / "src"),
        ]
        if environment.get("PYTHONPATH"):
            python_paths.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(python_paths)
        for _attack, command in commands:
            subprocess.run(
                command,
                check=True,
                cwd=args.checkout,
                env=environment,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
