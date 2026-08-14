"""Portable tree-of-attacks-with-pruning (TAP) search.

This clean behavioral reimplementation preserves tree expansion and pruning
without copying the prototype's hosted-model and GPU execution code.
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
class TapConfig:
    max_depth: int = 100
    branching_factor: int = 8
    beam_width: int = 8
    query_budget: int = 800
    seed: int = 1234
    early_stop_score: float | None = None

    def __post_init__(self) -> None:
        if (
            self.max_depth <= 0
            or self.branching_factor <= 0
            or self.beam_width <= 0
            or self.query_budget <= 0
        ):
            raise ValueError(
                "max_depth, branching_factor, beam_width, and query_budget "
                "must be positive"
            )


def _fingerprint(config: TapConfig) -> dict[str, Any]:
    value = asdict(config)
    value.pop("max_depth")
    return value


def _state(
    config: TapConfig,
    evaluations: Sequence[CandidateEvaluation],
    frontier: Sequence[str],
    completed_depth: int,
    stopped_early: bool,
) -> dict[str, Any]:
    return {
        "version": 1,
        "algorithm": "tap",
        "config": _fingerprint(config),
        "evaluations": [item.to_dict() for item in evaluations],
        "frontier": list(frontier),
        "completed_steps": completed_depth,
        "stopped_early": stopped_early,
        "query_budget_exhausted": len(evaluations) >= config.query_budget,
    }


def run_tap_search(
    *,
    goal: str,
    initial_injections: Sequence[str],
    proposer: ProposalFn,
    victim: VictimFn,
    scorer: ScoreFn,
    config: TapConfig = TapConfig(),
    resume_state: Mapping[str, Any] | None = None,
    checkpoint: CheckpointFn | None = None,
    success_predicate: SuccessPredicate | None = None,
) -> SearchResult:
    """Expand each frontier node and retain the best-scoring next layer."""

    if not goal:
        raise ValueError("goal must not be empty")
    if resume_state is None:
        seeds = list(dict.fromkeys(initial_injections))
        if not seeds or any(not isinstance(item, str) or not item for item in seeds):
            raise ValueError("initial_injections must contain non-empty strings")
        evaluations = evaluate_candidates(
            algorithm="tap",
            step=0,
            # Seed scoring counts toward Pass@N query accounting.
            injections=seeds[: config.query_budget],
            parents=(),
            victim=victim,
            scorer=scorer,
        )
        frontier = [item.candidate_id for item in ranked(evaluations)[: config.beam_width]]
        completed_depth = 0
        stopped_early = False
    else:
        validate_resume_state(resume_state, algorithm="tap")
        if resume_state.get("config") != _fingerprint(config):
            raise ValueError("resume state does not match the current TAP configuration")
        evaluations = [
            CandidateEvaluation.from_dict(item)
            for item in resume_state.get("evaluations", [])
        ]
        frontier = [str(item) for item in resume_state.get("frontier", [])]
        completed_depth = int(resume_state.get("completed_steps", 0))
        stopped_early = bool(resume_state.get("stopped_early", False))
        if not evaluations or not frontier:
            raise ValueError("TAP resume state is incomplete")

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
    state = _state(config, evaluations, frontier, completed_depth, stopped_early)
    if checkpoint is not None and resume_state is None:
        checkpoint(state)

    for depth in range(completed_depth + 1, config.max_depth + 1):
        if stopped_early:
            break
        if len(evaluations) >= config.query_budget:
            break
        children: list[CandidateEvaluation] = []
        for parent_index, parent_id in enumerate(frontier):
            remaining_queries = config.query_budget - len(evaluations) - len(children)
            if remaining_queries <= 0:
                break
            parent = by_id[parent_id]
            request = AttackRequest(
                algorithm="tap",
                goal=goal,
                step=depth,
                count=min(config.branching_factor, remaining_queries),
                parents=(parent.injection,),
                history=evaluation_history(evaluations),
                seed=config.seed + depth * 10_007 + parent_index,
            )
            proposed = [
                item
                for item in proposal_candidates(proposer, request)
                if item not in seen
            ]
            branch = evaluate_candidates(
                algorithm="tap",
                step=depth,
                injections=proposed,
                parents=(parent.injection,),
                victim=victim,
                scorer=scorer,
                ordinal_start=len(evaluations) + len(children),
            )
            for item in branch:
                seen.add(item.injection)
                by_id[item.candidate_id] = item
            children.extend(branch)
        evaluations.extend(children)
        if children:
            frontier = [item.candidate_id for item in ranked(children)[: config.beam_width]]
        completed_depth = depth
        if success_predicate is not None and any(
            success_predicate(item) for item in children
        ):
            stopped_early = True
        elif config.early_stop_score is not None:
            stopped_early = ranked(evaluations)[0].score >= config.early_stop_score
        state = _state(config, evaluations, frontier, completed_depth, stopped_early)
        if checkpoint is not None:
            checkpoint(state)

    best = ranked(evaluations)[0]
    return SearchResult(
        algorithm="tap",
        best=best,
        evaluations=tuple(evaluations),
        completed_steps=completed_depth,
        stopped_early=stopped_early,
        state=state,
    )
