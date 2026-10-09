"""Narrow conversion boundary between SecPO values and the Tinker SDK."""

from __future__ import annotations

from types import ModuleType
from typing import Any

from .data import ShiftedSequence, TokenizedSecPOPair


def _sdk(tinker_module: ModuleType | Any | None) -> Any:
    if tinker_module is not None:
        return tinker_module
    try:
        import tinker
    except ImportError as error:  # pragma: no cover - installation error path
        raise RuntimeError(
            "the Tinker runtime requires the optional 'tinker' dependency"
        ) from error
    return tinker


def sequence_to_datum(
    sequence: ShiftedSequence,
    *,
    tinker_module: ModuleType | Any | None = None,
) -> Any:
    """Convert one shifted policy sequence into a ``tinker.Datum``."""

    tinker = _sdk(tinker_module)
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(list(sequence.input_tokens)),
        loss_fn_inputs={
            "target_tokens": list(sequence.target_tokens),
            "weights": list(sequence.weights),
        },
    )


def policy_datums(
    pair: TokenizedSecPOPair,
    *,
    tinker_module: ModuleType | Any | None = None,
) -> tuple[Any, Any]:
    """Return the attacked chosen/rejected datums in loss-callback order."""

    return (
        sequence_to_datum(pair.policy_chosen, tinker_module=tinker_module),
        sequence_to_datum(pair.policy_rejected, tinker_module=tinker_module),
    )


def reference_inputs(
    pair: TokenizedSecPOPair,
    *,
    tinker_module: ModuleType | Any | None = None,
) -> tuple[Any, Any]:
    """Return clean chosen/rejected full sequences for frozen scoring."""

    tinker = _sdk(tinker_module)
    return (
        tinker.ModelInput.from_ints(list(pair.reference_chosen.full_tokens)),
        tinker.ModelInput.from_ints(list(pair.reference_rejected.full_tokens)),
    )
