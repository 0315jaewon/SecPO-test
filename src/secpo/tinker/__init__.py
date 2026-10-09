"""Optional Tinker backend for Security Preference Optimization.

The modules in this package keep their lightweight validation and loss helpers
importable without the Tinker SDK.  Runtime modules import Tinker lazily so the
existing Transformers backend remains an independent installation path.
"""

from .data import (
    QWEN_END_TOKEN,
    QWEN38_SOURCE_FIELDS,
    GENERIC_PRE_RENDERED_FIELDS,
    PreparedSecPOExample,
    ShiftedSequence,
    TokenizedSecPOPair,
    prepare_qwen38_record,
    prepare_pre_rendered_record,
    tokenize_pre_rendered_example,
)
from .loss import compute_tinker_secpo_loss
from .config import TinkerSecPOConfig, load_tinker_config, parse_tinker_config
from .runtime import (
    SecPOStepResult,
    TinkerClients,
    create_tinker_clients,
    scheduled_learning_rate,
    score_reference_pairs,
    train_secpo_step,
)
from .runner import (
    BatchPlan,
    SecPORunResult,
    TokenPreflightStats,
    TrainingPlan,
    build_training_plan,
    run_secpo_training,
    validate_all_records,
)

__all__ = [
    "QWEN_END_TOKEN",
    "QWEN38_SOURCE_FIELDS",
    "GENERIC_PRE_RENDERED_FIELDS",
    "PreparedSecPOExample",
    "ShiftedSequence",
    "TokenizedSecPOPair",
    "prepare_qwen38_record",
    "prepare_pre_rendered_record",
    "tokenize_pre_rendered_example",
    "compute_tinker_secpo_loss",
    "TinkerSecPOConfig",
    "load_tinker_config",
    "parse_tinker_config",
    "SecPOStepResult",
    "TinkerClients",
    "create_tinker_clients",
    "scheduled_learning_rate",
    "score_reference_pairs",
    "train_secpo_step",
    "BatchPlan",
    "SecPORunResult",
    "TokenPreflightStats",
    "TrainingPlan",
    "build_training_plan",
    "run_secpo_training",
    "validate_all_records",
]
