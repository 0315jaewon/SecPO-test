"""Deterministic island genetic search with a small MAP-Elites archive.

This is a provider-neutral reimplementation of the search structure used in
the research prototype.  It intentionally does not copy the prototype's model
clients, GPU orchestration, logging stack, or environment-specific machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
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
class GeneticConfig:
    """Configuration for island evolution and MAP-Elites selection."""

    steps: int = 100
    candidates_per_step: int = 8
    num_islands: int = 5
    island_capacity: int = 100
    map_bins: int = 10
    length_bin_width: int = 64
    migration_interval: int = 10
    migration_size: int = 1
    inspiration_count: int = 8
    query_budget: int = 800
    seed: int = 1234
    early_stop_score: float | None = None

    def __post_init__(self) -> None:
        positive = {
            "steps": self.steps,
            "candidates_per_step": self.candidates_per_step,
            "num_islands": self.num_islands,
            "island_capacity": self.island_capacity,
            "map_bins": self.map_bins,
            "length_bin_width": self.length_bin_width,
            "migration_interval": self.migration_interval,
            "migration_size": self.migration_size,
            "inspiration_count": self.inspiration_count,
            "query_budget": self.query_budget,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")


def _feature_cell(text: str, config: GeneticConfig) -> str:
    length_bin = min(len(text) // config.length_bin_width, config.map_bins - 1)
    tokens = text.casefold().split()
    diversity = len(set(tokens)) / max(1, len(tokens))
    diversity_bin = min(int(diversity * config.map_bins), config.map_bins - 1)
    return f"{length_bin}:{diversity_bin}"


def _decode_state(
    state: Mapping[str, Any], config: GeneticConfig
) -> tuple[list[CandidateEvaluation], list[list[str]], dict[str, str], int, bool]:
    validate_resume_state(state, algorithm="genetic")
    fingerprint = state.get("config")
    expected = _config_fingerprint(config)
    if fingerprint != expected:
        raise ValueError("resume state does not match the current genetic configuration")
    evaluations = [
        CandidateEvaluation.from_dict(item) for item in state.get("evaluations", [])
    ]
    if not evaluations:
        raise ValueError("genetic resume state contains no evaluated candidates")
    islands = [[str(item) for item in island] for island in state.get("islands", [])]
    if len(islands) != config.num_islands:
        raise ValueError("genetic resume state has the wrong number of islands")
    archive = {str(key): str(value) for key, value in state.get("archive", {}).items()}
    return (
        evaluations,
        islands,
        archive,
        int(state.get("completed_steps", 0)),
        bool(state.get("stopped_early", False)),
    )


def _config_fingerprint(config: GeneticConfig) -> dict[str, Any]:
    return {
        "candidates_per_step": config.candidates_per_step,
        "num_islands": config.num_islands,
        "island_capacity": config.island_capacity,
        "map_bins": config.map_bins,
        "length_bin_width": config.length_bin_width,
        "migration_interval": config.migration_interval,
        "migration_size": config.migration_size,
        "inspiration_count": config.inspiration_count,
        "query_budget": config.query_budget,
        "seed": config.seed,
        "early_stop_score": config.early_stop_score,
    }


def _encode_state(
    *,
    config: GeneticConfig,
    evaluations: Sequence[CandidateEvaluation],
    islands: Sequence[Sequence[str]],
    archive: Mapping[str, str],
    completed_steps: int,
    stopped_early: bool,
) -> dict[str, Any]:
    return {
        "version": 1,
        "algorithm": "genetic",
        "config": _config_fingerprint(config),
        "completed_steps": completed_steps,
        "stopped_early": stopped_early,
        "query_budget_exhausted": len(evaluations) >= config.query_budget,
        "evaluations": [item.to_dict() for item in evaluations],
        "islands": [list(island) for island in islands],
        "archive": dict(sorted(archive.items())),
    }


def _update_archive(
    archive: dict[str, str],
    evaluation: CandidateEvaluation,
    by_id: Mapping[str, CandidateEvaluation],
    config: GeneticConfig,
) -> None:
    cell = _feature_cell(evaluation.injection, config)
    incumbent_id = archive.get(cell)
    if incumbent_id is None:
        archive[cell] = evaluation.candidate_id
        return
    incumbent = by_id[incumbent_id]
    if evaluation.score > incumbent.score or (
        evaluation.score == incumbent.score
        and evaluation.candidate_id < incumbent.candidate_id
    ):
        archive[cell] = evaluation.candidate_id


def _trim_island(
    candidate_ids: Sequence[str],
    by_id: Mapping[str, CandidateEvaluation],
    capacity: int,
) -> list[str]:
    unique = list(dict.fromkeys(candidate_ids))
    return [item.candidate_id for item in ranked([by_id[item] for item in unique])][
        :capacity
    ]


def _select_inspirations(
    *,
    step: int,
    island_index: int,
    islands: Sequence[Sequence[str]],
    archive: Mapping[str, str],
    by_id: Mapping[str, CandidateEvaluation],
    config: GeneticConfig,
) -> list[CandidateEvaluation]:
    local = list(islands[island_index])
    # Sorting makes a state round-trip insensitive to JSON object key order.
    elite = sorted(set(archive.values()))
    pool = list(dict.fromkeys(local + elite))
    local_rng = random.Random(config.seed + step * 1_000_003 + island_index)
    local_rng.shuffle(pool)
    chosen = pool[: config.inspiration_count]
    return [by_id[item] for item in chosen]


def _result(
    state: Mapping[str, Any], evaluations: Sequence[CandidateEvaluation]
) -> SearchResult:
    best = ranked(evaluations)[0]
    return SearchResult(
        algorithm="genetic",
        best=best,
        evaluations=tuple(evaluations),
        completed_steps=int(state["completed_steps"]),
        stopped_early=bool(state["stopped_early"]),
        state=state,
    )


def run_genetic_search(
    *,
    goal: str,
    initial_injections: Sequence[str],
    proposer: ProposalFn,
    victim: VictimFn,
    scorer: ScoreFn,
    config: GeneticConfig = GeneticConfig(),
    resume_state: Mapping[str, Any] | None = None,
    checkpoint: CheckpointFn | None = None,
    success_predicate: SuccessPredicate | None = None,
) -> SearchResult:
    """Run or resume island evolution.

    The archive cells use two inexpensive behavioral descriptors: character
    length and whitespace-token diversity.  Each cell keeps its highest-scored
    candidate, while bounded island populations provide local selection and
    periodic ring migration.
    """

    if not goal:
        raise ValueError("goal must not be empty")

    if resume_state is None:
        seeds = list(dict.fromkeys(initial_injections))
        if not seeds or any(not isinstance(item, str) or not item for item in seeds):
            raise ValueError("initial_injections must contain non-empty strings")
        evaluations = evaluate_candidates(
            algorithm="genetic",
            step=0,
            # Seed scoring counts toward Pass@N query accounting.
            injections=seeds[: config.query_budget],
            parents=(),
            victim=victim,
            scorer=scorer,
        )
        islands: list[list[str]] = [[] for _ in range(config.num_islands)]
        archive: dict[str, str] = {}
        by_id = {item.candidate_id: item for item in evaluations}
        for index, item in enumerate(evaluations):
            islands[index % config.num_islands].append(item.candidate_id)
            _update_archive(archive, item, by_id, config)
        completed_steps = 0
        stopped_early = False
    else:
        evaluations, islands, archive, completed_steps, stopped_early = _decode_state(
            resume_state, config
        )

    by_id = {item.candidate_id: item for item in evaluations}
    seen = {item.injection for item in evaluations}
    initial_best = ranked(evaluations)[0]
    if success_predicate is not None and any(
        success_predicate(item) for item in evaluations
    ):
        stopped_early = True
    elif config.early_stop_score is not None and initial_best.score >= config.early_stop_score:
        stopped_early = True

    state = _encode_state(
        config=config,
        evaluations=evaluations,
        islands=islands,
        archive=archive,
        completed_steps=completed_steps,
        stopped_early=stopped_early,
    )
    if checkpoint is not None and resume_state is None:
        checkpoint(state)

    for step in range(completed_steps + 1, config.steps + 1):
        if stopped_early:
            break
        remaining_queries = config.query_budget - len(evaluations)
        if remaining_queries <= 0:
            break
        island_index = (step - 1) % config.num_islands
        inspirations = _select_inspirations(
            step=step,
            island_index=island_index,
            islands=islands,
            archive=archive,
            by_id=by_id,
            config=config,
        )
        request = AttackRequest(
            algorithm="genetic",
            goal=goal,
            step=step,
            count=min(config.candidates_per_step, remaining_queries),
            parents=tuple(item.injection for item in inspirations),
            history=evaluation_history(evaluations),
            seed=config.seed + step,
        )
        proposed = [
            item for item in proposal_candidates(proposer, request) if item not in seen
        ]
        new_items = evaluate_candidates(
            algorithm="genetic",
            step=step,
            injections=proposed,
            parents=request.parents,
            victim=victim,
            scorer=scorer,
            ordinal_start=len(evaluations),
        )
        for item in new_items:
            seen.add(item.injection)
            evaluations.append(item)
            by_id[item.candidate_id] = item
            islands[island_index].append(item.candidate_id)
            _update_archive(archive, item, by_id, config)
        islands[island_index] = _trim_island(
            islands[island_index], by_id, config.island_capacity
        )

        if step % config.migration_interval == 0 and config.num_islands > 1:
            migrants = [
                _trim_island(island, by_id, config.migration_size)
                for island in islands
            ]
            for source, migrant_ids in enumerate(migrants):
                destination = (source + 1) % config.num_islands
                islands[destination] = _trim_island(
                    islands[destination] + migrant_ids,
                    by_id,
                    config.island_capacity,
                )

        completed_steps = step
        if success_predicate is not None and any(
            success_predicate(item) for item in new_items
        ):
            stopped_early = True
        elif config.early_stop_score is not None:
            stopped_early = ranked(evaluations)[0].score >= config.early_stop_score
        state = _encode_state(
            config=config,
            evaluations=evaluations,
            islands=islands,
            archive=archive,
            completed_steps=completed_steps,
            stopped_early=stopped_early,
        )
        if checkpoint is not None:
            checkpoint(state)

    return _result(state, evaluations)
