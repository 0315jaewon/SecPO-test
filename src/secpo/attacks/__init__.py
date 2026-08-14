"""Adaptive prompt-injection search algorithms.

All algorithms are pure orchestration: model and judge behavior is supplied by
callables.  This keeps the released implementation suitable for local models,
approved hosted services, or deterministic tests without embedding endpoints
or credentials.

These modules are clean, behavioral reimplementations of the research
prototype's search flow.  They preserve the algorithmic interfaces and query
accounting, but are not bit-for-bit copies of its provider- and cluster-bound
implementation.
"""

from .base import (
    AttackRequest,
    CandidateEvaluation,
    EvaluationSummary,
    SearchResult,
)
from .genetic import GeneticConfig, run_genetic_search
from .pair import PairConfig, run_pair_search
from .tap import TapConfig, run_tap_search


__all__ = [
    "AttackRequest",
    "CandidateEvaluation",
    "EvaluationSummary",
    "GeneticConfig",
    "PairConfig",
    "SearchResult",
    "TapConfig",
    "run_genetic_search",
    "run_pair_search",
    "run_tap_search",
]
