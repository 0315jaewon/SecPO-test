"""Non-adaptive smoke-test backend for the online exchange protocol.

This example intentionally returns each record's existing seed attack.  It is
useful for validating orchestration but does not reproduce PAIR or any reported
adaptive result.  A real backend should query the current policy and use the
released search primitives under :mod:`secpo.attacks`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def reuse_seed_attacks(request: Mapping[str, Any]) -> dict[str, str]:
    """Return the already validated seed prompt for every request example."""

    return {
        example["id"]: example["seed_prompt"]
        for example in request["examples"]
    }

