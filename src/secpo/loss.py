"""The benign-reference Security Preference Optimization objective.

The policy log-probabilities are computed on an attacked prompt, while the
reference log-probabilities are computed on its injection-free counterpart.
This module intentionally contains no model-specific code.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .constants import IGNORE_INDEX



@dataclass(frozen=True)
class SecPOLossOutput:
    """Per-example values produced by :func:`secpo_loss`."""

    losses: torch.Tensor
    chosen_rewards: torch.Tensor
    rejected_rewards: torch.Tensor
    logits: torch.Tensor


def sequence_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Return each sequence's summed causal log-probability.

    ``labels`` must have the same batch and sequence dimensions as ``logits``.
    Positions equal to ``ignore_index`` are excluded. As usual for a causal
    language model, token ``t`` is scored by the logits at position ``t - 1``.

    Summed, rather than length-normalized, log-probabilities are used by the
    main SecPO experiments.
    """

    if logits.ndim != 3:
        raise ValueError(
            f"logits must have shape [batch, sequence, vocabulary], got "
            f"{tuple(logits.shape)}"
        )
    if labels.ndim != 2:
        raise ValueError(
            f"labels must have shape [batch, sequence], got {tuple(labels.shape)}"
        )
    if tuple(logits.shape[:2]) != tuple(labels.shape):
        raise ValueError(
            "logits and labels must have identical batch/sequence dimensions; "
            f"got {tuple(logits.shape[:2])} and {tuple(labels.shape)}"
        )
    if logits.shape[1] < 2:
        raise ValueError("sequences must contain at least two tokens")
    if not logits.is_floating_point():
        raise TypeError("logits must be a floating-point tensor")

    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:].to(device=logits.device)
    response_mask = shifted_labels.ne(ignore_index)
    response_lengths = response_mask.sum(dim=-1)
    if torch.any(response_lengths == 0):
        bad_rows = torch.nonzero(response_lengths == 0, as_tuple=False)
        bad_rows = bad_rows.flatten().tolist()
        raise ValueError(f"each sequence must contain a scored token; empty rows: {bad_rows}")

    safe_labels = shifted_labels.masked_fill(~response_mask, 0)
    if torch.any(safe_labels[response_mask] < 0) or torch.any(
        safe_labels[response_mask] >= shifted_logits.shape[-1]
    ):
        raise ValueError("labels contain a token id outside the logits vocabulary")

    token_logps = F.log_softmax(shifted_logits, dim=-1).gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    )
    token_logps = token_logps.squeeze(-1)
    return token_logps.masked_fill(~response_mask, 0.0).sum(dim=-1)


def secpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    *,
    beta: float = 0.1,
) -> SecPOLossOutput:
    """Compute the sigmoid SecPO loss and implicit rewards.

    The four arguments must contain one scalar log-probability per example.
    Reference tensors are moved to the policy tensor's device, but all shapes
    and dtypes otherwise remain under the caller's control.
    """

    tensors = (
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    )
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("all log-probability arguments must be torch tensors")
    shape = policy_chosen_logps.shape
    if any(value.shape != shape for value in tensors[1:]):
        raise ValueError(
            "all log-probability tensors must have the same shape; got "
            + ", ".join(str(tuple(value.shape)) for value in tensors)
        )
    if policy_chosen_logps.ndim != 1:
        raise ValueError(
            "log-probability tensors must have shape [batch], got "
            f"{tuple(policy_chosen_logps.shape)}"
        )
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError(f"beta must be finite and positive, got {beta!r}")

    device = policy_chosen_logps.device
    policy_rejected_logps = policy_rejected_logps.to(device)
    reference_chosen_logps = reference_chosen_logps.to(device)
    reference_rejected_logps = reference_rejected_logps.to(device)

    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps
    preference_logits = policy_logratios - reference_logratios
    losses = -F.logsigmoid(beta * preference_logits)

    chosen_rewards = beta * (
        policy_chosen_logps - reference_chosen_logps
    ).detach()
    rejected_rewards = beta * (
        policy_rejected_logps - reference_rejected_logps
    ).detach()
    return SecPOLossOutput(
        losses=losses,
        chosen_rewards=chosen_rewards,
        rejected_rewards=rejected_rewards,
        logits=preference_logits,
    )
