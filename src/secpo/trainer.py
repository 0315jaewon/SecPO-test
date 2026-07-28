"""Hugging Face Trainer implementation of benign-reference SecPO."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

import torch
from transformers import Trainer

from .loss import SecPOLossOutput, secpo_loss, sequence_logps


_FORWARD_NAMES = (
    "policy_chosen",
    "policy_rejected",
    "reference_chosen",
    "reference_rejected",
)


def model_sequence_logps(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Run a causal-LM forward pass and sum response-token log-probabilities."""

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    if not hasattr(outputs, "logits"):
        raise TypeError("the causal language model output must expose a logits tensor")
    return sequence_logps(outputs.logits, labels)


def _adapter_disabled_context(model: torch.nn.Module) -> AbstractContextManager[Any]:
    """Find PEFT's adapter context through common distributed wrappers."""

    candidate: Any = model
    visited: set[int] = set()
    while candidate is not None and id(candidate) not in visited:
        visited.add(id(candidate))
        disable_adapter = getattr(candidate, "disable_adapter", None)
        if callable(disable_adapter):
            return disable_adapter()
        candidate = getattr(candidate, "module", None)
    raise TypeError(
        "SecPO requires a PEFT model exposing the disable_adapter() context manager"
    )


class SecPOTrainer(Trainer):
    """Train a PEFT policy against its clean-prompt, adapter-free reference.

    Policy forwards use the attacked prompt tensors. Reference forwards use
    separate clean-prompt tensors while the policy adapter is disabled. The
    same base weights therefore serve as the frozen reference without a second
    model copy.
    """

    def __init__(self, *args: Any, beta: float = 0.1, **kwargs: Any) -> None:
        if not isinstance(beta, (int, float)) or isinstance(beta, bool) or beta <= 0:
            raise ValueError(f"beta must be positive, got {beta!r}")
        self.beta = float(beta)
        super().__init__(*args, **kwargs)

    @staticmethod
    def _validate_inputs(inputs: dict[str, Any]) -> None:
        expected = {
            f"{name}_{suffix}"
            for name in _FORWARD_NAMES
            for suffix in ("input_ids", "attention_mask", "labels")
        }
        missing = expected - set(inputs)
        if missing:
            raise ValueError(f"SecPO batch is missing tensors: {sorted(missing)}")
        unexpected = set(inputs) - expected
        if unexpected:
            raise ValueError(f"SecPO batch has unexpected entries: {sorted(unexpected)}")
        for name in expected:
            if not isinstance(inputs[name], torch.Tensor):
                raise TypeError(f"SecPO batch entry {name!r} must be a torch tensor")

    @staticmethod
    def _forward_block(
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        name: str,
    ) -> torch.Tensor:
        return model_sequence_logps(
            model,
            input_ids=inputs[f"{name}_input_ids"],
            attention_mask=inputs[f"{name}_attention_mask"],
            labels=inputs[f"{name}_labels"],
        )

    def compute_secpo_output(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
    ) -> SecPOLossOutput:
        """Compute unreduced loss and rewards for one already-collated batch."""

        self._validate_inputs(inputs)

        policy_chosen_logps = self._forward_block(model, inputs, "policy_chosen")
        policy_rejected_logps = self._forward_block(model, inputs, "policy_rejected")

        # Disabling the adapter changes the same wrapped model used for policy
        # forwards. This is compatible with sharded base weights and avoids a
        # second full reference-model allocation.
        with torch.no_grad():
            with _adapter_disabled_context(model):
                reference_chosen_logps = self._forward_block(
                    model,
                    inputs,
                    "reference_chosen",
                )
                reference_rejected_logps = self._forward_block(
                    model,
                    inputs,
                    "reference_rejected",
                )

        return secpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            beta=self.beta,
        )

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the mean sigmoid SecPO loss for ``transformers.Trainer``."""

        del num_items_in_batch  # SecPO is defined as a mean over preference pairs.
        output = self.compute_secpo_output(model, inputs)
        loss = output.losses.mean()
        if not return_outputs:
            return loss
        return loss, {
            "logits": output.logits.detach(),
            "chosen_rewards": output.chosen_rewards,
            "rejected_rewards": output.rejected_rewards,
        }
