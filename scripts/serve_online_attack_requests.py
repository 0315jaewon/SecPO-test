#!/usr/bin/env python3
"""Serve versioned online SecPO attack requests.

The worker backend is intentionally supplied by the user as an importable
``module:callable``.  That callable receives one decoded request dictionary and
must return ``{example_id: attacked_prompt}``.  This keeps credentials and
provider-specific inference code outside the release artifact while preserving
the complete attacker/defender data contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from secpo.online_training import FileAttackExchange, load_backend, serve_requests


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--exchange-dir",
        type=Path,
        required=True,
        help="Runtime request/response directory shared with the trainer.",
    )
    result.add_argument(
        "--backend",
        required=True,
        help="Import path of an attack backend, in module:callable form.",
    )
    result.add_argument(
        "--rank",
        type=int,
        help="Serve only requests from this defender rank.",
    )
    result.add_argument("--poll-seconds", type=float, default=1.0)
    result.add_argument("--once", action="store_true")
    result.add_argument(
        "--idle-timeout-seconds",
        type=float,
        help="Exit successfully after this many seconds without a request.",
    )
    result.add_argument(
        "--retain-files",
        action="store_true",
        help="Retain runtime requests/responses for debugging; never publish them.",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    backend = load_backend(args.backend)
    exchange = FileAttackExchange(
        args.exchange_dir,
        poll_seconds=args.poll_seconds,
        retain_files=args.retain_files,
    )
    processed = serve_requests(
        exchange,
        backend,
        rank=args.rank,
        once=args.once,
        idle_timeout_seconds=args.idle_timeout_seconds,
    )
    print(f"Processed {processed} online attack request(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
