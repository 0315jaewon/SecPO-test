#!/usr/bin/env python3
"""Plot a Tinker SecPO loss curve using linear axes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, help="Tinker metrics.jsonl file")
    parser.add_argument("output", type=Path, help="Output PNG path")
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=10,
        help="Trailing-average window (default: 10)",
    )
    return parser.parse_args()


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("rolling window must be positive")
    result = np.full(values.shape, np.nan, dtype=float)
    if len(values) >= window:
        result[window - 1 :] = np.convolve(
            values, np.ones(window, dtype=float) / window, mode="valid"
        )
    return result


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.metrics.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no metric rows found in {args.metrics}")

    steps = np.asarray([row["step"] for row in rows], dtype=int)
    losses = np.asarray([row["secpo_loss"] for row in rows], dtype=float)
    smoothed = trailing_mean(losses, args.rolling_window)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 4.8),
        gridspec_kw={"width_ratios": [1.8, 1]},
        constrained_layout=True,
    )
    for axis in axes:
        axis.plot(
            steps,
            losses,
            color="#4C78A8",
            linewidth=1,
            alpha=0.42,
            label="Per-step loss",
        )
        axis.plot(
            steps,
            smoothed,
            color="#E45756",
            linewidth=2.2,
            label=f"{args.rolling_window}-step trailing mean",
        )
        axis.set_yscale("linear")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.22)
        axis.set_xlabel("Training step")

    axes[0].set_title("Full training run")
    axes[0].set_ylabel("SecPO loss (linear scale)")
    axes[0].legend(frameon=False)

    early_last = min(80, int(steps[-1]))
    axes[1].set_xlim(0, early_last)
    axes[1].set_title(f"First {early_last} steps")

    fig.suptitle("Qwen3.8-27B SecPO training loss", fontsize=15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
