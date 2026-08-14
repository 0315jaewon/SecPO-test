from __future__ import annotations

import math

import pytest

from secpo.attacks import (
    GeneticConfig,
    PairConfig,
    TapConfig,
    run_genetic_search,
    run_pair_search,
    run_tap_search,
)


def echo_victim(injection: str) -> str:
    return f"victim: {injection}"


def marker_score(injection: str, response: str) -> float:
    assert injection in response
    return float(injection.count("+"))


def revising_proposer(request):
    parent = request.parents[0] if request.parents else "seed"
    return [f"{parent}+{request.step}.{index}" for index in range(request.count)]


def test_genetic_is_deterministic_and_json_serializable():
    config = GeneticConfig(
        steps=4,
        candidates_per_step=3,
        num_islands=2,
        island_capacity=3,
        map_bins=4,
        migration_interval=2,
        seed=17,
        early_stop_score=None,
    )
    first = run_genetic_search(
        goal="emit witness",
        initial_injections=("seed-a", "seed-b"),
        proposer=revising_proposer,
        victim=echo_victim,
        scorer=marker_score,
        config=config,
    )
    second = run_genetic_search(
        goal="emit witness",
        initial_injections=("seed-a", "seed-b"),
        proposer=revising_proposer,
        victim=echo_victim,
        scorer=marker_score,
        config=config,
    )

    assert first.to_dict(include_state=True) == second.to_dict(include_state=True)
    assert first.completed_steps == 4
    assert first.best.score >= 2
    assert first.state["archive"]
    assert all(len(island) <= config.island_capacity for island in first.state["islands"])


def test_genetic_resume_matches_uninterrupted_run():
    checkpoints = []
    partial_config = GeneticConfig(
        steps=2,
        candidates_per_step=2,
        num_islands=2,
        migration_interval=2,
        seed=9,
        early_stop_score=None,
    )
    run_genetic_search(
        goal="emit witness",
        initial_injections=("seed-a", "seed-b"),
        proposer=revising_proposer,
        victim=echo_victim,
        scorer=marker_score,
        config=partial_config,
        checkpoint=checkpoints.append,
    )
    full_config = GeneticConfig(
        steps=5,
        candidates_per_step=2,
        num_islands=2,
        migration_interval=2,
        seed=9,
        early_stop_score=None,
    )
    resumed = run_genetic_search(
        goal="emit witness",
        initial_injections=("ignored-on-resume",),
        proposer=revising_proposer,
        victim=echo_victim,
        scorer=marker_score,
        config=full_config,
        resume_state=checkpoints[-1],
    )
    uninterrupted = run_genetic_search(
        goal="emit witness",
        initial_injections=("seed-a", "seed-b"),
        proposer=revising_proposer,
        victim=echo_victim,
        scorer=marker_score,
        config=full_config,
    )

    assert resumed.to_dict(include_state=True) == uninterrupted.to_dict(
        include_state=True
    )


def test_pair_uses_feedback_and_tracks_a_bounded_frontier():
    result = run_pair_search(
        goal="emit witness",
        initial_injections=("seed",),
        proposer=revising_proposer,
        victim=echo_victim,
        scorer=marker_score,
        config=PairConfig(
            steps=3,
            candidates_per_step=3,
            beam_width=2,
            early_stop_score=None,
        ),
    )

    assert result.completed_steps == 3
    assert result.best.step == 3
    assert result.best.score == 3
    assert len(result.state["frontier"]) == 2
    assert result.evaluations[-1].parents


def test_tap_expands_each_parent_then_prunes_the_layer():
    result = run_tap_search(
        goal="emit witness",
        initial_injections=("seed",),
        proposer=revising_proposer,
        victim=echo_victim,
        scorer=marker_score,
        config=TapConfig(
            max_depth=3,
            branching_factor=2,
            beam_width=2,
            early_stop_score=None,
        ),
    )

    assert result.completed_steps == 3
    assert result.best.score == 3
    assert len(result.state["frontier"]) == 2
    for item in result.evaluations:
        if item.step:
            assert len(item.parents) == 1


def test_search_rejects_non_finite_scores():
    with pytest.raises(ValueError, match="non-finite"):
        run_pair_search(
            goal="goal",
            initial_injections=("seed",),
            proposer=revising_proposer,
            victim=echo_victim,
            scorer=lambda injection, response: math.nan,
            config=PairConfig(steps=1),
        )


@pytest.mark.parametrize(
    ("runner", "config"),
    [
        (
            run_genetic_search,
            GeneticConfig(
                steps=100,
                candidates_per_step=8,
                num_islands=2,
                query_budget=17,
                early_stop_score=None,
            ),
        ),
        (
            run_tap_search,
            TapConfig(
                max_depth=100,
                branching_factor=8,
                beam_width=8,
                query_budget=17,
                early_stop_score=None,
            ),
        ),
    ],
)
def test_pass_at_n_budget_includes_seed_queries(runner, config):
    result = runner(
        goal="goal",
        initial_injections=("seed-a", "seed-b"),
        proposer=revising_proposer,
        victim=echo_victim,
        scorer=marker_score,
        config=config,
    )

    assert len(result.evaluations) == 17
    assert result.to_dict()["victim_queries"] == 17
    assert result.to_dict()["query_budget_exhausted"] is True
