"""Reference implementation of benign-reference SecPO.

Top-level exports are loaded lazily so the dependency-light evaluation and
release-audit tools can run without importing the training stack or CUDA.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


__version__ = "0.3.0"

_EXPORTS = {
    "IGNORE_INDEX": ("secpo.constants", "IGNORE_INDEX"),
    "BenignReferenceCollator": ("secpo.data", "BenignReferenceCollator"),
    "PreferenceExample": ("secpo.data", "PreferenceExample"),
    "PreferenceSchemaError": ("secpo.data", "PreferenceSchemaError"),
    "SecPOLossOutput": ("secpo.loss", "SecPOLossOutput"),
    "SecPOTrainer": ("secpo.trainer", "SecPOTrainer"),
    "build_input_and_labels": ("secpo.data", "build_input_and_labels"),
    "derive_clean_prompt": ("secpo.data", "derive_clean_prompt"),
    "load_preference_dataset": ("secpo.data", "load_preference_dataset"),
    "model_sequence_logps": ("secpo.trainer", "model_sequence_logps"),
    "secpo_loss": ("secpo.loss", "secpo_loss"),
    "sequence_logps": ("secpo.loss", "sequence_logps"),
    "validate_preference_record": ("secpo.data", "validate_preference_record"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve public training exports only when requested."""

    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
