#!/usr/bin/env python3
"""Run the Llama current-policy PAIR worker in exact 32-example/rank mode."""

from __future__ import annotations

from serve_llama31_current_policy_pair import main


if __name__ == "__main__":
    raise SystemExit(main(force_exact_batch=True))
