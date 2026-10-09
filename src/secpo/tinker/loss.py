"""Token-logprob reduction for Tinker's custom SecPO backward call."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch

from secpo.loss import secpo_loss


def _sequence_sum(
    logprobs: torch.Tensor,
    weights: Sequence[float] | torch.Tensor,
    *,
    label: str,
) -> torch.Tensor:
    if not isinstance(logprobs, torch.Tensor):
        raise TypeError(f"{label} logprobs must be a torch tensor")
    if logprobs.ndim != 1:
        raise ValueError(f"{label} logprobs must have shape [tokens]")
    weight_tensor = torch.as_tensor(
        weights,
        dtype=logprobs.dtype,
        device=logprobs.device,
    )
    if weight_tensor.ndim != 1 or weight_tensor.shape != logprobs.shape:
        raise ValueError(
            f"{label} weights must match logprobs shape {tuple(logprobs.shape)}"
        )
    if not torch.all((weight_tensor == 0) | (weight_tensor == 1)):
        raise ValueError(f"{label} weights must contain only 0 or 1")
    if not torch.any(weight_tensor == 1):
        raise ValueError(f"{label} weights must score at least one response token")
    return torch.dot(logprobs, weight_tensor)


def compute_tinker_secpo_loss(
    policy_token_logprobs: Sequence[torch.Tensor],
    reference_token_logprobs: Sequence[torch.Tensor],
    policy_weights: Sequence[Sequence[float] | torch.Tensor],
    reference_weights: Sequence[Sequence[float] | torch.Tensor],
    *,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Reduce interleaved token logprobs and compute the SecPO batch loss.

    Every sequence is ordered ``chosen, rejected``.  Policy sequences contain
    attacked prompts while reference sequences contain their clean prompts, so
    their masks and token lengths are intentionally allowed to differ.
    """

    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    sizes = {
        len(policy_token_logprobs),
        len(reference_token_logprobs),
        len(policy_weights),
        len(reference_weights),
    }
    if len(sizes) != 1:
        raise ValueError("logprob and weight collections must have equal lengths")
    sequence_count = len(policy_token_logprobs)
    if sequence_count == 0 or sequence_count % 2:
        raise ValueError(
            "SecPO requires a non-empty even number of chosen/rejected sequences"
        )

    policy_sums = [
        _sequence_sum(logprobs, weights, label=f"policy[{index}]")
        for index, (logprobs, weights) in enumerate(
            zip(policy_token_logprobs, policy_weights, strict=True)
        )
    ]
    reference_sums = [
        _sequence_sum(logprobs, weights, label=f"reference[{index}]")
        for index, (logprobs, weights) in enumerate(
            zip(reference_token_logprobs, reference_weights, strict=True)
        )
    ]

    policy_chosen = torch.stack(policy_sums[0::2])
    policy_rejected = torch.stack(policy_sums[1::2])
    reference_chosen = torch.stack(reference_sums[0::2])
    reference_rejected = torch.stack(reference_sums[1::2])
    output = secpo_loss(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
        beta=beta,
    )
    loss = output.losses.mean()
    accuracy = (output.logits > 0).float().mean()
    metrics = {
        "secpo_loss": float(loss.detach().item()),
        "accuracy": float(accuracy.detach().item()),
        "margin": float(
            (output.chosen_rewards - output.rejected_rewards).mean().item()
        ),
        "chosen_reward": float(output.chosen_rewards.mean().item()),
        "rejected_reward": float(output.rejected_rewards.mean().item()),
    }
    return loss, metrics
