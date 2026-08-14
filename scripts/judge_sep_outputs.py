#!/usr/bin/env python3
"""Run the provider-neutral, witness-first SecPO paper judge protocol.

See ``python scripts/judge_sep_outputs.py --help`` and
``secpo.evaluation.sep_judge`` for the JSONL schemas and backend contract.
"""

from secpo.evaluation.sep_judge import main


if __name__ == "__main__":
    raise SystemExit(main())
