"""Portable PAIR-style iterative prompt-injection search.

This clean behavioral reimplementation preserves iterative feedback and beam
selection without copying the prototype's provider conversation objects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .base import (
    AttackRequest,
    CandidateEvaluation,
    CheckpointFn,
    ProposalFn,
    ScoreFn,
    SearchResult,
    SuccessPredicate,
    VictimFn,
    evaluate_candidates,
    evaluation_history,
    proposal_candidates,
    ranked,
    validate_resume_state,
)


@dataclass(frozen=True)
class PairConfig:
    steps: int = 100
    candidates_per_step: int = 8
    beam_width: int = 8
    seed: int = 1234
    early_stop_score: float | None = None

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.candidates_per_step <= 0 or self.beam_width <= 0:
            raise ValueError("steps, candidates_per_step, and beam_width must be positive")


def _fingerprint(config: PairConfig) -> dict[str, Any]:
    value = asdict(config)
    value.pop("steps")
    return value


def _state(
    config: PairConfig,
    evaluations: Sequence[CandidateEvaluation],
    frontier: Sequence[str],
    completed_steps: int,
    stopped_early: bool,
) -> dict[str, Any]:
    return {
        "version": 1,
        "algorithm": "pair",
        "config": _fingerprint(config),
        "evaluations": [item.to_dict() for item in evaluations],
        "frontier": list(frontier),
        "completed_steps": completed_steps,
        "stopped_early": stopped_early,
    }


def run_pair_search(
    *,
    goal: str,
    initial_injections: Sequence[str],
    proposer: ProposalFn,
    victim: VictimFn,
    scorer: ScoreFn,
    config: PairConfig = PairConfig(),
    resume_state: Mapping[str, Any] | None = None,
    checkpoint: CheckpointFn | None = None,
    success_predicate: SuccessPredicate | None = None,
) -> SearchResult:
    """Iteratively revise a beam of attacks using victim/scorer feedback."""

    if not goal:
        raise ValueError("goal must not be empty")
    if resume_state is None:
        seeds = list(dict.fromkeys(initial_injections))
        if not seeds or any(not isinstance(item, str) or not item for item in seeds):
            raise ValueError("initial_injections must contain non-empty strings")
        evaluations = evaluate_candidates(
            algorithm="pair",
            step=0,
            injections=seeds,
            parents=(),
            victim=victim,
            scorer=scorer,
        )
        frontier = [item.candidate_id for item in ranked(evaluations)[: config.beam_width]]
        completed_steps = 0
        stopped_early = False
    else:
        validate_resume_state(resume_state, algorithm="pair")
        if resume_state.get("config") != _fingerprint(config):
            raise ValueError("resume state does not match the current PAIR configuration")
        evaluations = [
            CandidateEvaluation.from_dict(item)
            for item in resume_state.get("evaluations", [])
        ]
        frontier = [str(item) for item in resume_state.get("frontier", [])]
        completed_steps = int(resume_state.get("completed_steps", 0))
        stopped_early = bool(resume_state.get("stopped_early", False))
        if not evaluations or not frontier:
            raise ValueError("PAIR resume state is incomplete")

    by_id = {item.candidate_id: item for item in evaluations}
    seen = {item.injection for item in evaluations}
    if success_predicate is not None and any(
        success_predicate(item) for item in evaluations
    ):
        stopped_early = True
    elif (
        config.early_stop_score is not None
        and ranked(evaluations)[0].score >= config.early_stop_score
    ):
        stopped_early = True
    state = _state(config, evaluations, frontier, completed_steps, stopped_early)
    if checkpoint is not None and resume_state is None:
        checkpoint(state)

    for step in range(completed_steps + 1, config.steps + 1):
        if stopped_early:
            break
        parents = tuple(by_id[item].injection for item in frontier)
        request = AttackRequest(
            algorithm="pair",
            goal=goal,
            step=step,
            count=config.candidates_per_step,
            parents=parents,
            history=evaluation_history(evaluations),
            seed=config.seed + step,
        )
        proposed = [
            item for item in proposal_candidates(proposer, request) if item not in seen
        ]
        new_items = evaluate_candidates(
            algorithm="pair",
            step=step,
            injections=proposed,
            parents=parents,
            victim=victim,
            scorer=scorer,
            ordinal_start=len(evaluations),
        )
        for item in new_items:
            seen.add(item.injection)
            evaluations.append(item)
            by_id[item.candidate_id] = item
        if new_items:
            frontier = [item.candidate_id for item in ranked(new_items)[: config.beam_width]]
        completed_steps = step
        if success_predicate is not None and any(
            success_predicate(item) for item in new_items
        ):
            stopped_early = True
        elif config.early_stop_score is not None:
            stopped_early = ranked(evaluations)[0].score >= config.early_stop_score
        state = _state(config, evaluations, frontier, completed_steps, stopped_early)
        if checkpoint is not None:
            checkpoint(state)

    best = ranked(evaluations)[0]
    return SearchResult(
        algorithm="pair",
        best=best,
        evaluations=tuple(evaluations),
        completed_steps=completed_steps,
        stopped_early=stopped_early,
        state=state,
    )
