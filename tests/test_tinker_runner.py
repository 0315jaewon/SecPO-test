from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from secpo.tinker.config import load_tinker_config
from secpo.tinker.runner import build_training_plan, run_secpo_training
from secpo.tinker.runtime import SecPOStepResult, TinkerClients


class CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]


class Immediate:
    def __init__(self, value: object):
        self.value = value

    def result(self) -> object:
        return self.value


class FakeTrainingClient:
    def __init__(self) -> None:
        self.states: list[tuple[str, int | None]] = []
        self.samplers: list[tuple[str, int | None]] = []

    def save_state(self, name: str, ttl_seconds: int | None = None) -> Immediate:
        self.states.append((name, ttl_seconds))
        return Immediate(SimpleNamespace(path=f"tinker://state/{name}"))

    def save_weights_for_sampler(
        self, name: str, ttl_seconds: int | None = None
    ) -> Immediate:
        self.samplers.append((name, ttl_seconds))
        return Immediate(SimpleNamespace(path=f"tinker://sampler/{name}"))


def _records(count: int = 3) -> list[dict[str, str]]:
    return [
        {
            "prompt": f"attacked-{index}",
            "clean_prompt": f"clean-{index}",
            "chosen": f"chosen-{index}",
            "rejected": f"rejected-{index}",
        }
        for index in range(count)
    ]


def _config(tmp_path, **changes):
    config = load_tinker_config("configs/train_qwen38_27b_tinker.json")
    defaults = {
        "dataset_format": "secpo_pre_rendered",
        "expected_records": 3,
        "batch_size": 2,
        "num_epochs": 1,
        "max_steps": None,
        "save_every": 1,
        "max_prompt_length": 100,
        "max_length": 100,
        "output_dir": str(tmp_path / "run"),
    }
    defaults.update(changes)
    return replace(config, **defaults)


def test_training_plan_is_deterministic_and_covers_each_epoch() -> None:
    first = build_training_plan(
        record_count=5,
        batch_size=2,
        num_epochs=2,
        seed=17,
        max_steps=None,
    )
    second = build_training_plan(
        record_count=5,
        batch_size=2,
        num_epochs=2,
        seed=17,
        max_steps=None,
    )

    assert first == second
    assert first.batches_per_epoch == 3
    assert first.total_steps == 6
    for epoch in range(2):
        epoch_indices = [
            index
            for batch in first.batches
            if batch.epoch == epoch
            for index in batch.indices
        ]
        assert sorted(epoch_indices) == list(range(5))


def test_training_plan_honors_max_steps() -> None:
    plan = build_training_plan(
        record_count=10,
        batch_size=3,
        num_epochs=2,
        seed=0,
        max_steps=3,
    )
    assert plan.total_steps == 3
    assert [batch.step for batch in plan.batches] == [0, 1, 2]


def test_preflight_validates_every_row_without_creating_outputs(tmp_path) -> None:
    config = _config(tmp_path)

    result = run_secpo_training(
        config,
        preflight_only=True,
        dataset=_records(),
        tokenizer=CharacterTokenizer(),
    )

    assert result.plan.total_steps == 2
    assert result.steps_completed == 0
    assert result.token_preflight is not None
    assert result.token_preflight.max_policy_prompt == len("attacked-0")
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    ("save_every", "intermediate_states"),
    [(1, ["secpo-step-000001"]), (0, [])],
)
@pytest.mark.parametrize("ttl_seconds", [604800, None])
def test_full_runner_trains_batches_logs_and_exports(
    tmp_path, save_every, intermediate_states, ttl_seconds
) -> None:
    config = _config(
        tmp_path, save_every=save_every, checkpoint_ttl_seconds=ttl_seconds
    )
    training = FakeTrainingClient()
    seen_sources: list[list[int | None]] = []

    def fake_step(pairs, **kwargs):
        seen_sources.append([pair.source_index for pair in pairs])
        step = kwargs["step"]
        assert kwargs["total_steps"] == 2
        return SecPOStepResult(
            metrics={"step": step, "secpo_loss": 0.5},
            backward_result=object(),
            optimizer_result=object(),
        )

    result = run_secpo_training(
        config,
        dataset=_records(),
        clients=TinkerClients(training=training, reference=object()),
        tokenizer=CharacterTokenizer(),
        tinker_module=SimpleNamespace(),
        train_step_fn=fake_step,
    )

    assert sorted(index for batch in seen_sources for index in batch) == [0, 1, 2]
    assert training.states == [
        (name, ttl_seconds) for name in [*intermediate_states, "secpo-final-state"]
    ]
    assert training.samplers == [("secpo-final-sampler", ttl_seconds)]
    assert result.final_state_path == "tinker://state/secpo-final-state"
    assert result.final_sampler_path == "tinker://sampler/secpo-final-sampler"
    assert result.token_preflight is not None
    metrics = [
        json.loads(line)
        for line in (tmp_path / "run" / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [metric["step"] for metric in metrics] == [0, 1]
    assert sum("state_path" in metric for metric in metrics) == len(intermediate_states)
    assert json.loads(
        (tmp_path / "run" / "result.json").read_text(encoding="utf-8")
    )["steps_completed"] == 2


def test_runner_refuses_to_overwrite_prior_run_before_training(tmp_path) -> None:
    config = _config(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    (output / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_secpo_training(
            config,
            dataset=_records(),
            tokenizer=CharacterTokenizer(),
            tinker_module=SimpleNamespace(),
        )
