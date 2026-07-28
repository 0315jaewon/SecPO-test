"""Anonymous reference implementation of benign-reference SecPO."""

from .data import (
    BenignReferenceCollator,
    PreferenceExample,
    PreferenceSchemaError,
    build_input_and_labels,
    derive_clean_prompt,
    load_preference_dataset,
    validate_preference_record,
)
from .loss import IGNORE_INDEX, SecPOLossOutput, secpo_loss, sequence_logps
from .trainer import SecPOTrainer, model_sequence_logps


__all__ = [
    "IGNORE_INDEX",
    "BenignReferenceCollator",
    "PreferenceExample",
    "PreferenceSchemaError",
    "SecPOLossOutput",
    "SecPOTrainer",
    "build_input_and_labels",
    "derive_clean_prompt",
    "load_preference_dataset",
    "model_sequence_logps",
    "secpo_loss",
    "sequence_logps",
    "validate_preference_record",
]

__version__ = "0.1.0"
