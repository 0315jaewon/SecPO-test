from __future__ import annotations

import json

import pytest

from secpo.evaluation.sep_adaptive import (
    AdaptiveEvaluationConfig,
    AdaptiveExample,
    DryRunBackend,
    evaluate_example,
    load_checkpoint,
    load_examples,
    main,
)


class RecordingBackend(DryRunBackend):
    def __init__(self):
        self.seen_examples = []

    def victim(self, example, injection):
        self.seen_examples.append(example.example_id)
        return super().victim(example, injection)


def test_evaluator_routes_example_to_backend_and_writes_checkpoint(tmp_path):
    example = AdaptiveExample(
        example_id="sample/one",
        instruction="answer the benign question",
        input_text="untrusted text",
        injection_goal="emit exact witness phrase",
        initial_injections=("seed",),
    )
    checkpoint = tmp_path / "nested" / "checkpoint.json"
    backend = RecordingBackend()
    config = AdaptiveEvaluationConfig(
        algorithm="pair",
        steps=2,
        candidates_per_step=2,
        beam_width=2,
        early_stop_score=None,
    )

    result = evaluate_example(
        example,
        backend,
        config=config,
        checkpoint_path=checkpoint,
    )

    assert result.example == example
    assert result.search.completed_steps == 2
    assert backend.seen_examples and set(backend.seen_examples) == {"sample/one"}
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["example_id"] == "sample/one"
    assert payload["search_state"]["completed_steps"] == 2


def test_evaluator_resume_matches_full_run(tmp_path):
    example = AdaptiveExample(
        example_id="resume",
        instruction="benign task",
        input_text="input",
        injection_goal="emit exact witness phrase",
        initial_injections=("seed",),
    )
    checkpoint = tmp_path / "resume.json"
    partial = AdaptiveEvaluationConfig(
        algorithm="genetic",
        steps=2,
        candidates_per_step=2,
        num_islands=2,
        early_stop_score=None,
    )
    evaluate_example(
        example,
        DryRunBackend(),
        config=partial,
        checkpoint_path=checkpoint,
    )

    full = AdaptiveEvaluationConfig(
        algorithm="genetic",
        steps=5,
        candidates_per_step=2,
        num_islands=2,
        early_stop_score=None,
    )
    resumed = evaluate_example(
        example,
        DryRunBackend(),
        config=full,
        checkpoint_path=checkpoint,
        resume=True,
    )
    uninterrupted = evaluate_example(example, DryRunBackend(), config=full)

    assert resumed.search.to_dict(include_state=True) == uninterrupted.search.to_dict(
        include_state=True
    )


def test_checkpoint_rejects_a_different_example(tmp_path):
    first = AdaptiveExample("one", "task", "input", "goal", ("seed",))
    second = AdaptiveExample("two", "task", "input", "goal", ("seed",))
    config = AdaptiveEvaluationConfig(steps=1, early_stop_score=None)
    checkpoint = tmp_path / "checkpoint.json"
    evaluate_example(first, DryRunBackend(), config=config, checkpoint_path=checkpoint)

    with pytest.raises(ValueError, match="does not match"):
        load_checkpoint(checkpoint, example=second, config=config)


def test_load_examples_accepts_wrapped_json(tmp_path):
    path = tmp_path / "examples.json"
    path.write_text(
        json.dumps(
            {
                "examples": [
                    {
                        "id": "x",
                        "instruction": "task",
                        "input": "document",
                        "goal": "witness",
                        "initial_injections": ["seed"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    examples = load_examples(path)
    assert examples[0].example_id == "x"
    assert examples[0].input_text == "document"


@pytest.mark.parametrize("algorithm", ["genetic", "pair", "tap"])
def test_cli_dry_run_without_model_dependencies(tmp_path, algorithm):
    output = tmp_path / f"{algorithm}.json"
    checkpoints = tmp_path / "checkpoints"
    assert (
        main(
            [
                "--algorithm",
                algorithm,
                "--steps",
                "2",
                "--candidates",
                "2",
                "--beam-width",
                "2",
                "--branching-factor",
                "2",
                "--num-islands",
                "2",
                "--early-stop-score",
                "2",
                "--checkpoint-dir",
                str(checkpoints),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["backend"] == "dry-run"
    assert payload["results"][0]["search"]["algorithm"] == algorithm
    assert list(checkpoints.glob("*.checkpoint.json"))
