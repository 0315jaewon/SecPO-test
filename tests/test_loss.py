from __future__ import annotations

import torch
import torch.nn.functional as F

from secpo.loss import IGNORE_INDEX, secpo_loss, sequence_logps


def test_secpo_loss_matches_benign_reference_equation() -> None:
    policy_chosen = torch.tensor([-1.2, -2.0], dtype=torch.float64)
    policy_rejected = torch.tensor([-2.7, -1.5], dtype=torch.float64)
    reference_chosen = torch.tensor([-0.8, -2.4], dtype=torch.float64)
    reference_rejected = torch.tensor([-1.1, -2.1], dtype=torch.float64)
    beta = 0.1

    output = secpo_loss(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
        beta=beta,
    )

    # Equation (5): the policy margin is evaluated on the attacked prompt,
    # while the reference margin is evaluated on the corresponding clean
    # prompt.
    expected_logits = (policy_chosen - policy_rejected) - (
        reference_chosen - reference_rejected
    )
    expected_losses = -F.logsigmoid(beta * expected_logits)

    torch.testing.assert_close(output.logits, expected_logits)
    torch.testing.assert_close(output.losses, expected_losses)
    torch.testing.assert_close(
        output.chosen_rewards, beta * (policy_chosen - reference_chosen)
    )
    torch.testing.assert_close(
        output.rejected_rewards, beta * (policy_rejected - reference_rejected)
    )


def test_secpo_loss_has_preference_improving_policy_gradients() -> None:
    policy_chosen = torch.tensor([-1.4, -0.9], dtype=torch.float64, requires_grad=True)
    policy_rejected = torch.tensor([-1.0, -1.7], dtype=torch.float64, requires_grad=True)
    reference_chosen = torch.tensor([-1.1, -1.3], dtype=torch.float64)
    reference_rejected = torch.tensor([-1.6, -1.2], dtype=torch.float64)
    beta = 0.2

    output = secpo_loss(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
        beta=beta,
    )
    output.losses.mean().backward()

    # Gradient descent must raise log pi(chosen|x_adv) and lower
    # log pi(rejected|x_adv). Check the exact derivative as well as its sign.
    expected_factor = beta * torch.sigmoid(-beta * output.logits.detach()) / 2
    torch.testing.assert_close(policy_chosen.grad, -expected_factor)
    torch.testing.assert_close(policy_rejected.grad, expected_factor)
    assert torch.all(policy_chosen.grad < 0)
    assert torch.all(policy_rejected.grad > 0)


def test_sequence_logps_uses_only_labeled_response_tokens() -> None:
    logits = torch.tensor(
        [
            [
                [2.0, 0.0, -1.0],
                [-0.5, 1.5, 0.0],
                [0.0, 0.0, 0.0],
                [5.0, -5.0, 0.0],
            ]
        ],
        dtype=torch.float64,
    )
    labels = torch.tensor([[IGNORE_INDEX, 1, 2, IGNORE_INDEX]])

    actual = sequence_logps(logits, labels)
    expected = (
        F.log_softmax(logits[0, 0], dim=-1)[1]
        + F.log_softmax(logits[0, 1], dim=-1)[2]
    ).unsqueeze(0)

    torch.testing.assert_close(actual, expected)
