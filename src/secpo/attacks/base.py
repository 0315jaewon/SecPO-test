"""Provider-neutral interfaces shared by adaptive prompt-injection searches.

The original research code coupled search logic to hosted model clients,
GPU workers, and cluster checkpoints.  This module keeps the algorithmic
boundary explicit: callers inject a proposer, a victim, and a scorer.

No network, model, or filesystem operation happens in this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class EvaluationSummary:
    """Compact feedback exposed to an attacker proposal callable."""

    injection: str
    response: str
    score: float
    step: int


@dataclass(frozen=True)
class AttackRequest:
    """One deterministic request for new candidate injections.

    ``seed`` is supplied instead of a mutable random-number generator so a
    remote or local proposer can reproduce a request independently.
    """

    algorithm: str
    goal: str
    step: int
    count: int
    parents: tuple[str, ...] = ()
    history: tuple[EvaluationSummary, ...] = ()
    seed: int = 0


@dataclass(frozen=True)
class CandidateEvaluation:
    """A victim response and attack score for one candidate injection."""

    candidate_id: str
    injection: str
    response: str
    score: float
    step: int
    parents: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["parents"] = list(self.parents)
        result["metadata"] = dict(self.metadata)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateEvaluation":
        return cls(
            candidate_id=str(value["candidate_id"]),
            injection=str(value["injection"]),
            response=str(value["response"]),
            score=float(value["score"]),
            step=int(value["step"]),
            parents=tuple(str(item) for item in value.get("parents", ())),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class SearchResult:
    """Serializable result returned by every adaptive search."""

    algorithm: str
    best: CandidateEvaluation
    evaluations: tuple[CandidateEvaluation, ...]
    completed_steps: int
    stopped_early: bool
    state: Mapping[str, Any]

    def to_dict(self, *, include_state: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "algorithm": self.algorithm,
            "best": self.best.to_dict(),
            "evaluations": [item.to_dict() for item in self.evaluations],
            "victim_queries": len(self.evaluations),
            "completed_steps": self.completed_steps,
            "stopped_early": self.stopped_early,
            "query_budget_exhausted": bool(
                self.state.get("query_budget_exhausted", False)
            ),
        }
        if include_state:
            result["state"] = dict(self.state)
        return result


ProposalFn = Callable[[AttackRequest], Sequence[str]]
VictimFn = Callable[[str], str]
ScoreFn = Callable[[str, str], float]
CheckpointFn = Callable[[Mapping[str, Any]], None]
SuccessPredicate = Callable[[CandidateEvaluation], bool]


def proposal_candidates(proposer: ProposalFn, request: AttackRequest) -> list[str]:
    """Call ``proposer`` and validate/de-duplicate its returned candidates."""

    proposed = proposer(request)
    if isinstance(proposed, (str, bytes)) or not isinstance(proposed, Sequence):
        raise TypeError("the proposal callable must return a sequence of strings")

    unique: list[str] = []
    seen: set[str] = set()
    for value in proposed:
        if not isinstance(value, str):
            raise TypeError("every proposed injection must be a string")
        if not value:
            raise ValueError("candidate injections must not be empty")
        if value not in seen:
            seen.add(value)
            unique.append(value)
        if len(unique) == request.count:
            break
    return unique


def evaluate_candidates(
    *,
    algorithm: str,
    step: int,
    injections: Sequence[str],
    parents: Sequence[str],
    victim: VictimFn,
    scorer: ScoreFn,
    ordinal_start: int = 0,
) -> list[CandidateEvaluation]:
    """Evaluate candidates synchronously in input order.

    Synchronous evaluation is deliberate: the reference implementation is
    deterministic and portable.  Applications may batch or parallelize inside
    their injected victim callable.
    """

    evaluations: list[CandidateEvaluation] = []
    for offset, injection in enumerate(injections):
        response = victim(injection)
        if not isinstance(response, str):
            raise TypeError("the victim callable must return a string")
        score = float(scorer(injection, response))
        if not math.isfinite(score):
            raise ValueError(f"the scorer returned a non-finite value: {score!r}")
        digest = hashlib.sha256(injection.encode("utf-8")).hexdigest()[:10]
        candidate_id = (
            f"{algorithm}-s{step:04d}-c{ordinal_start + offset:04d}-{digest}"
        )
        evaluations.append(
            CandidateEvaluation(
                candidate_id=candidate_id,
                injection=injection,
                response=response,
                score=score,
                step=step,
                parents=tuple(parents),
            )
        )
    return evaluations


def evaluation_history(
    evaluations: Sequence[CandidateEvaluation],
) -> tuple[EvaluationSummary, ...]:
    return tuple(
        EvaluationSummary(
            injection=item.injection,
            response=item.response,
            score=item.score,
            step=item.step,
        )
        for item in evaluations
    )


def ranked(
    evaluations: Sequence[CandidateEvaluation],
) -> list[CandidateEvaluation]:
    """Return a deterministic descending ranking (score, then stable id)."""

    return sorted(evaluations, key=lambda item: (-item.score, item.candidate_id))


def validate_resume_state(
    state: Mapping[str, Any], *, algorithm: str, version: int = 1
) -> None:
    if state.get("algorithm") != algorithm:
        raise ValueError(
            f"resume state is for {state.get('algorithm')!r}, expected {algorithm!r}"
        )
    if state.get("version") != version:
        raise ValueError(
            f"unsupported {algorithm} state version {state.get('version')!r}"
        )
