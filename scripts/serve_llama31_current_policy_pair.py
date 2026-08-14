#!/usr/bin/env python3
"""Serve current-policy Llama-3.1-8B PAIR requests.

The attacker, iteration judge, and final judge are caller-owned batch
callables.  They receive the typed request objects documented in
``docs/LLAMA31_CURRENT_POLICY_PAIR.md``.  Provider credentials and endpoints
therefore remain outside this repository.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
from typing import Sequence

from secpo.backends.llama31_current_policy_pair import (
    CurrentPolicyPairBackend,
    PairBackendConfig,
    VllmCurrentPolicy,
    load_callable,
)
from secpo.online_training import FileAttackExchange, serve_requests


LLAMA31_8B_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA31_8B_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"


def _environment_rank() -> int:
    raw = os.environ.get("RANK", "0")
    try:
        rank = int(raw)
    except ValueError as error:
        raise ValueError("RANK must be a non-negative integer") from error
    if rank < 0:
        raise ValueError("RANK must be a non-negative integer")
    return rank


def parser(*, force_exact_batch: bool = False) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--exchange-dir", type=Path, required=True)
    result.add_argument("--policy-root", type=Path, required=True)
    result.add_argument("--base-model", default=LLAMA31_8B_MODEL)
    result.add_argument("--base-model-revision", default=LLAMA31_8B_REVISION)
    result.add_argument(
        "--proposer",
        required=True,
        help="PAIR attacker batch callable in module:attribute form.",
    )
    result.add_argument(
        "--iteration-judge",
        required=True,
        help="1--10 iteration-judge batch callable in module:attribute form.",
    )
    result.add_argument(
        "--final-judge",
        required=True,
        help="YES/NO final-judge batch callable in module:attribute form.",
    )
    result.add_argument("--rank", type=int, default=None)
    result.add_argument("--minimum-snapshot-age-seconds", type=float, default=20.0)
    result.add_argument("--snapshot-settle-timeout-seconds", type=float, default=60.0)
    result.add_argument("--max-lora-rank", type=int, default=64)
    result.add_argument("--poll-seconds", type=float, default=1.0)
    result.add_argument("--idle-timeout-seconds", type=float)
    result.add_argument("--once", action="store_true")
    result.add_argument("--retain-files", action="store_true")
    result.add_argument("--trust-remote-code", action="store_true")
    if not force_exact_batch:
        result.add_argument(
            "--exact-batch",
            action="store_true",
            help=(
                "Require 32 examples in each rank request, as in the archived "
                "experiment. The released trainer emits microbatches; use this "
                "only with an accumulation-group request producer."
            ),
        )
    return result


def build_backend(args: argparse.Namespace, *, exact_batch: bool) -> CurrentPolicyPairBackend:
    config = PairBackendConfig.from_path(args.config)
    if exact_batch and not config.strict_per_rank_batch:
        raise ValueError("exact-batch mode requires strict_per_rank_batch=true")
    if not exact_batch:
        # The public OnlineAttackCollator emits the trainer's two-example
        # microbatches. Its blocking request/response schedule is intentionally
        # documented as trajectory-different from the archived asynchronous
        # 32-example/rank producer.
        config = replace(config, strict_per_rank_batch=False)

    policy = VllmCurrentPolicy(
        base_model=args.base_model,
        revision=args.base_model_revision,
        policy_root=args.policy_root,
        minimum_snapshot_age_seconds=args.minimum_snapshot_age_seconds,
        snapshot_settle_timeout_seconds=args.snapshot_settle_timeout_seconds,
        max_lora_rank=args.max_lora_rank,
        trust_remote_code=args.trust_remote_code,
    )
    return CurrentPolicyPairBackend(
        config=config,
        policy=policy,
        proposer=load_callable(args.proposer),
        iteration_judge=load_callable(args.iteration_judge),
        final_judge=load_callable(args.final_judge),
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    force_exact_batch: bool = False,
) -> int:
    args = parser(force_exact_batch=force_exact_batch).parse_args(argv)
    rank = _environment_rank() if args.rank is None else args.rank
    if rank < 0:
        raise ValueError("rank must be non-negative")
    exact_batch = force_exact_batch or getattr(args, "exact_batch", False)
    backend = build_backend(args, exact_batch=exact_batch)
    if rank >= backend.config.num_attacker_ranks:
        raise ValueError(
            f"rank {rank} is outside configured attacker ranks "
            f"[0, {backend.config.num_attacker_ranks})"
        )

    mode = "exact-32-per-rank" if exact_batch else "microbatch-compatible"
    print(f"Starting Llama current-policy PAIR worker rank={rank} mode={mode}")
    exchange = FileAttackExchange(
        args.exchange_dir,
        poll_seconds=args.poll_seconds,
        retain_files=args.retain_files,
    )
    processed = serve_requests(
        exchange,
        backend,
        rank=rank,
        once=args.once,
        idle_timeout_seconds=args.idle_timeout_seconds,
    )
    print(f"Processed {processed} online PAIR request(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
