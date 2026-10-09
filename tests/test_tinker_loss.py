from __future__ import annotations

import pytest
import torch

from secpo.loss import secpo_loss
from secpo.tinker.loss import compute_tinker_secpo_loss


def test_tinker_loss_matches_shared_secpo_equation_with_different_prompt_lengths() -> None:
    policy = [
        torch.tensor([-9.0, -1.0, -2.0], dtype=torch.float64, requires_grad=True),
        torch.tensor([-8.0, -3.0, -4.0], dtype=torch.float64, requires_grad=True),
        torch.tensor([-7.0, -0.5], dtype=torch.float64, requires_grad=True),
        torch.tensor([-6.0, -2.5], dtype=torch.float64, requires_grad=True),
    ]
    reference = [
        torch.tensor([-5.0, -6.0, -0.8, -1.2], dtype=torch.float64),
        torch.tensor([-4.0, -5.0, -1.8, -2.2], dtype=torch.float64),
        torch.tensor([-3.0, -0.4], dtype=torch.float64),
        torch.tensor([-2.0, -1.4], dtype=torch.float64),
    ]
    policy_weights = [
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ]
    reference_weights = [
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ]

    actual, metrics = compute_tinker_secpo_loss(
        policy,
        reference,
        policy_weights,
        reference_weights,
        beta=0.2,
    )
    expected = secpo_loss(
        torch.tensor([-3.0, -0.5], dtype=torch.float64),
        torch.tensor([-7.0, -2.5], dtype=torch.float64),
        torch.tensor([-2.0, -0.4], dtype=torch.float64),
        torch.tensor([-4.0, -1.4], dtype=torch.float64),
        beta=0.2,
    ).losses.mean()

    torch.testing.assert_close(actual, expected)
    assert metrics["secpo_loss"] == pytest.approx(expected.item())
    assert set(metrics) == {
        "secpo_loss",
        "accuracy",
        "margin",
        "chosen_reward",
        "rejected_reward",
    }

    actual.backward()
    assert torch.all(policy[0].grad[1:] < 0)
    assert torch.all(policy[1].grad[1:] > 0)
    assert policy[0].grad[0].item() == 0.0
    assert policy[1].grad[0].item() == 0.0


@pytest.mark.parametrize(
    ("policy", "reference", "policy_weights", "reference_weights", "message"),
    [
        ([], [], [], [], "non-empty even"),
        (
            [torch.zeros(2)],
            [torch.zeros(2)],
            [[0.0, 1.0]],
            [[0.0, 1.0]],
            "non-empty even",
        ),
        (
            [torch.zeros(2), torch.zeros(2)],
            [torch.zeros(2)],
            [[0.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 1.0]],
            "equal lengths",
        ),
        (
            [torch.zeros(2), torch.zeros(2)],
            [torch.zeros(2), torch.zeros(2)],
            [[1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 1.0]],
            "must match",
        ),
    ],
)
def test_tinker_loss_rejects_misaligned_batches(
    policy: list[torch.Tensor],
    reference: list[torch.Tensor],
    policy_weights: list[list[float]],
    reference_weights: list[list[float]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_tinker_secpo_loss(
            policy,
            reference,
            policy_weights,
            reference_weights,
        )
