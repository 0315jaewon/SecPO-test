"""Runtime backends for the public SecPO release.

Backends keep model serving and provider calls outside the core training and
evaluation modules.  Importing this package performs no network or GPU work.
"""

from .llama31_current_policy_pair import (
    CurrentPolicyPairBackend,
    FinalJudgeRequest,
    IterationJudgeRequest,
    PairBackendConfig,
    PairExampleResult,
    PolicySnapshot,
    PolicySnapshotLocator,
    ProposalRequest,
    VllmCurrentPolicy,
    extract_llama31_instruction,
)

__all__ = [
    "CurrentPolicyPairBackend",
    "FinalJudgeRequest",
    "IterationJudgeRequest",
    "PairBackendConfig",
    "PairExampleResult",
    "PolicySnapshot",
    "PolicySnapshotLocator",
    "ProposalRequest",
    "VllmCurrentPolicy",
    "extract_llama31_instruction",
]
