from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from secpo.tinker.config import load_tinker_config
from secpo.tinker.data import ShiftedSequence, TokenizedSecPOPair
from secpo.tinker.runtime import (
    create_tinker_clients,
    make_secpo_loss_callback,
    scheduled_learning_rate,
    score_reference_pairs,
    train_secpo_step,
)


@dataclass
class FakeDatum:
    model_input: tuple[int, ...]
    loss_fn_inputs: dict[str, object]


class FakeModelInput:
    @staticmethod
    def from_ints(tokens: list[int]) -> tuple[int, ...]:
        return tuple(tokens)


@dataclass
class FakeAdamParams:
    learning_rate: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    grad_clip_norm: float


FAKE_TINKER = SimpleNamespace(
    Datum=FakeDatum,
    ModelInput=FakeModelInput,
    AdamParams=FakeAdamParams,
)


class Immediate:
    def __init__(self, value: object):
        self.value = value

    def result(self) -> object:
        return self.value


def _sequence(tokens: tuple[int, ...], prompt_length: int) -> ShiftedSequence:
    full = tokens
    response_length = len(full) - prompt_length
    return ShiftedSequence(
        input_tokens=full[:-1],
        target_tokens=full[1:],
        weights=(0.0,) * (prompt_length - 1) + (1.0,) * response_length,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def _pair() -> TokenizedSecPOPair:
    return TokenizedSecPOPair(
        policy_chosen=_sequence((1, 2, 3, 4), 2),
        policy_rejected=_sequence((1, 2, 5, 6), 2),
        reference_chosen=_sequence((7, 3, 4), 1),
        reference_rejected=_sequence((7, 5, 6), 1),
        source_index=9,
    )


class FakeReferenceClient:
    def __init__(self) -> None:
        self.submitted: list[tuple[int, ...]] = []

    def compute_logprobs(self, value: tuple[int, ...]) -> Immediate:
        self.submitted.append(value)
        return Immediate([None] + [-0.5] * (len(value) - 1))


class FakeTrainingClient:
    def __init__(self) -> None:
        self.datums: list[FakeDatum] | None = None
        self.params: FakeAdamParams | None = None
        self.saved_reference = object()

    def save_weights_and_get_sampling_client(self) -> object:
        return self.saved_reference

    def forward_backward_custom(self, datums, callback) -> Immediate:
        self.datums = datums
        policy_logprobs = [
            torch.full((len(datum.model_input),), -0.4, requires_grad=True)
            for datum in datums
        ]
        loss, metrics = callback(datums, policy_logprobs)
        loss.backward()
        return Immediate(SimpleNamespace(metrics=metrics))

    def optim_step(self, params: FakeAdamParams) -> Immediate:
        self.params = params
        return Immediate("optimized")


def test_create_clients_freezes_reference_before_training() -> None:
    config = load_tinker_config("configs/train_qwen38_27b_tinker.json")
    training = FakeTrainingClient()

    class Service:
        def create_lora_training_client(self, **kwargs):
            assert kwargs == {
                "base_model": "Qwen/Qwen3.8-27B",
                "rank": 64,
                "seed": 42,
                "train_mlp": True,
                "train_attn": True,
                "train_unembed": False,
                "user_metadata": {"objective": "benign-reference-secpo"},
            }
            return training

    clients = create_tinker_clients(
        config,
        service_client=Service(),
        tinker_module=FAKE_TINKER,
    )

    assert clients.training is training
    assert clients.reference is training.saved_reference


@pytest.mark.parametrize(
    ("schedule", "expected"),
    [
        ("constant", [0.0, 0.5, 1.0, 1.0, 1.0]),
        ("linear", [0.0, 0.5, 1.0, 2 / 3, 1 / 3]),
        ("cosine", [0.0, 0.5, 1.0, 0.75, 0.25]),
    ],
)
def test_scheduled_learning_rate_has_warmup_and_requested_decay(
    schedule: str, expected: list[float]
) -> None:
    actual = [
        scheduled_learning_rate(
            1.0,
            step=step,
            total_steps=5,
            warmup_ratio=0.4,
            schedule=schedule,
        )
        for step in range(5)
    ]
    assert actual == pytest.approx(expected)


def test_reference_scoring_uses_clean_full_sequences_and_shifts_first_token() -> None:
    client = FakeReferenceClient()

    values = score_reference_pairs(
        [_pair()], client, tinker_module=FAKE_TINKER
    )

    assert client.submitted == [(7, 3, 4), (7, 5, 6)]
    assert [value.tolist() for value in values] == [
        [-0.5, -0.5],
        [-0.5, -0.5],
    ]


@pytest.mark.parametrize(
    "values",
    [
        [-0.1, -0.2, -0.3],
        [None, -0.2],
        [None, None, -0.3],
        [None, float("nan"), -0.3],
    ],
)
def test_reference_scoring_fails_closed_on_misaligned_values(values) -> None:
    class BadReference:
        def compute_logprobs(self, unused):
            return Immediate(values)

    with pytest.raises((ValueError, TypeError)):
        score_reference_pairs(
            [_pair()], BadReference(), tinker_module=FAKE_TINKER
        )


def test_callback_allows_clean_and_attacked_prompts_to_have_different_lengths() -> None:
    pair = _pair()
    reference = [torch.tensor([-0.5, -0.5]), torch.tensor([-0.6, -0.6])]
    callback = make_secpo_loss_callback([pair], reference, beta=0.1)
    policy = [
        torch.tensor([-9.0, -0.4, -0.4], requires_grad=True),
        torch.tensor([-9.0, -0.8, -0.8], requires_grad=True),
    ]

    loss, metrics = callback([object(), object()], policy)

    assert loss.ndim == 0
    assert metrics["margin"] > 0
    loss.backward()
    assert policy[0].grad[0].item() == 0
    assert policy[1].grad[0].item() == 0


def test_train_step_runs_custom_backward_then_adam_update() -> None:
    config = load_tinker_config("configs/train_qwen38_27b_tinker.json")
    training = FakeTrainingClient()
    reference = FakeReferenceClient()

    result = train_secpo_step(
        [_pair()],
        training_client=training,
        reference_client=reference,
        config=config,
        step=1,
        total_steps=10,
        tinker_module=FAKE_TINKER,
    )

    assert training.datums is not None and len(training.datums) == 2
    assert training.params is not None
    # ceil(10 * 0.03) is one warmup step, so step 1 reaches the peak LR.
    assert training.params.learning_rate == pytest.approx(8e-5)
    assert training.params.grad_clip_norm == 1.0
    assert result.optimizer_result == "optimized"
    assert result.metrics["step"] == 1
    assert result.metrics["num_pairs"] == 1
    assert result.metrics["policy_tokens"] == 6
    assert "secpo_loss" in result.metrics
