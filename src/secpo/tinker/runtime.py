"""Small, testable execution units for SecPO training on Tinker."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch

from .bridge import _sdk, policy_datums, reference_inputs
from .config import TinkerSecPOConfig
from .data import ShiftedSequence, TokenizedSecPOPair
from .loss import compute_tinker_secpo_loss


@dataclass(frozen=True)
class TinkerClients:
    """The trainable policy and its frozen initial reference snapshot."""

    training: Any
    reference: Any


@dataclass(frozen=True)
class SecPOStepResult:
    """Observable result of one complete remote optimization step."""

    metrics: dict[str, float | int]
    backward_result: Any
    optimizer_result: Any


def create_tinker_clients(
    config: TinkerSecPOConfig,
    *,
    service_client: Any | None = None,
    tinker_module: Any | None = None,
) -> TinkerClients:
    """Create a LoRA policy and freeze its initial weights as the reference."""

    tinker = _sdk(tinker_module)
    if service_client is None:
        service_client = tinker.ServiceClient(
            user_metadata={"recipe": "secpo", "backend": "secpo-tinker"}
        )
    training = service_client.create_lora_training_client(
        base_model=config.model_name,
        rank=config.lora_rank,
        seed=config.seed,
        train_mlp=config.train_mlp,
        train_attn=config.train_attn,
        train_unembed=config.train_unembed,
        user_metadata={"objective": "benign-reference-secpo"},
    )
    # This call happens before any optimization.  The returned sampling client
    # therefore remains the frozen base-plus-initial-LoRA reference throughout
    # the run, matching the existing Transformers trainer's reference model.
    reference = training.save_weights_and_get_sampling_client()
    return TinkerClients(training=training, reference=reference)


def scheduled_learning_rate(
    peak_learning_rate: float,
    *,
    step: int,
    total_steps: int,
    warmup_ratio: float,
    schedule: str,
) -> float:
    """Return a Transformers-compatible warmup/decay learning rate."""

    if not math.isfinite(peak_learning_rate) or peak_learning_rate <= 0:
        raise ValueError("peak_learning_rate must be finite and positive")
    if type(step) is not int or type(total_steps) is not int:
        raise TypeError("step and total_steps must be integers")
    if total_steps <= 0 or step < 0 or step >= total_steps:
        raise ValueError("step must satisfy 0 <= step < total_steps")
    if (
        not math.isfinite(warmup_ratio)
        or warmup_ratio < 0
        or warmup_ratio > 1
    ):
        raise ValueError("warmup_ratio must be between 0 and 1")
    if schedule not in {"linear", "cosine", "constant"}:
        raise ValueError("schedule must be linear, cosine, or constant")

    warmup_steps = math.ceil(total_steps * warmup_ratio)
    if step < warmup_steps:
        multiplier = step / max(1, warmup_steps)
    elif schedule == "constant":
        multiplier = 1.0
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        if schedule == "linear":
            multiplier = max(0.0, 1.0 - progress)
        else:
            multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_learning_rate * multiplier


def _ordered_sequences(
    pairs: Sequence[TokenizedSecPOPair],
    *,
    reference: bool,
) -> list[ShiftedSequence]:
    ordered: list[ShiftedSequence] = []
    for pair in pairs:
        if reference:
            ordered.extend((pair.reference_chosen, pair.reference_rejected))
        else:
            ordered.extend((pair.policy_chosen, pair.policy_rejected))
    return ordered


def score_reference_pairs(
    pairs: Sequence[TokenizedSecPOPair],
    reference_client: Any,
    *,
    tinker_module: Any | None = None,
) -> list[torch.Tensor]:
    """Score clean-prompt sequences with the frozen reference client.

    All requests are submitted before any result is awaited, allowing the SDK
    to overlap the remote calls while keeping this public API synchronous.
    """

    if not pairs:
        raise ValueError("a SecPO batch must contain at least one pair")
    model_inputs: list[Any] = []
    sequences = _ordered_sequences(pairs, reference=True)
    for pair in pairs:
        model_inputs.extend(
            reference_inputs(pair, tinker_module=tinker_module)
        )
    futures = [reference_client.compute_logprobs(value) for value in model_inputs]

    scored: list[torch.Tensor] = []
    for index, (future, sequence) in enumerate(
        zip(futures, sequences, strict=True)
    ):
        values = future.result()
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise TypeError(f"reference[{index}] logprobs must be a sequence")
        if len(values) != sequence.full_length:
            raise ValueError(
                f"reference[{index}] returned {len(values)} logprobs for "
                f"{sequence.full_length} tokens"
            )
        if values[0] is not None:
            raise ValueError(
                f"reference[{index}] first-token logprob must be None"
            )
        shifted = values[1:]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in shifted
        ):
            raise ValueError(
                f"reference[{index}] contains a missing or non-finite logprob"
            )
        scored.append(torch.tensor(shifted, dtype=torch.float32))
    return scored


def make_secpo_loss_callback(
    pairs: Sequence[TokenizedSecPOPair],
    reference_logprobs: Sequence[torch.Tensor],
    *,
    beta: float,
) -> Callable[
    [list[Any], list[torch.Tensor]],
    tuple[torch.Tensor, dict[str, float]],
]:
    """Bind clean reference scores and response masks for Tinker's callback."""

    policy_sequences = _ordered_sequences(pairs, reference=False)
    reference_sequences = _ordered_sequences(pairs, reference=True)
    if not policy_sequences:
        raise ValueError("a SecPO batch must contain at least one pair")
    if len(reference_logprobs) != len(reference_sequences):
        raise ValueError(
            "reference scores must contain chosen/rejected values per pair"
        )
    policy_weights = [sequence.weights for sequence in policy_sequences]
    reference_weights = [sequence.weights for sequence in reference_sequences]

    def callback(
        data: list[Any],
        policy_logprobs: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if len(data) != len(policy_sequences):
            raise ValueError(
                "Tinker callback data no longer matches the SecPO batch"
            )
        return compute_tinker_secpo_loss(
            policy_logprobs,
            reference_logprobs,
            policy_weights,
            reference_weights,
            beta=beta,
        )

    return callback


def train_secpo_step(
    pairs: Sequence[TokenizedSecPOPair],
    *,
    training_client: Any,
    reference_client: Any,
    config: TinkerSecPOConfig,
    step: int,
    total_steps: int,
    tinker_module: Any | None = None,
) -> SecPOStepResult:
    """Run frozen scoring, custom SecPO backward, and one AdamW update."""

    if not pairs:
        raise ValueError("a SecPO batch must contain at least one pair")
    tinker = _sdk(tinker_module)
    datums: list[Any] = []
    for pair in pairs:
        datums.extend(policy_datums(pair, tinker_module=tinker))
    reference_logprobs = score_reference_pairs(
        pairs,
        reference_client,
        tinker_module=tinker,
    )
    callback = make_secpo_loss_callback(
        pairs,
        reference_logprobs,
        beta=config.beta,
    )
    learning_rate = scheduled_learning_rate(
        config.learning_rate,
        step=step,
        total_steps=total_steps,
        warmup_ratio=config.warmup_ratio,
        schedule=config.lr_scheduler_type,
    )
    backward_result = training_client.forward_backward_custom(
        datums,
        callback,
    ).result()
    optimizer_result = training_client.optim_step(
        tinker.AdamParams(
            learning_rate=learning_rate,
            beta1=config.adam_beta1,
            beta2=config.adam_beta2,
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
            grad_clip_norm=config.grad_clip_norm,
        )
    ).result()

    remote_metrics = getattr(backward_result, "metrics", None)
    if not isinstance(remote_metrics, dict):
        raise TypeError("Tinker backward result must expose a metrics dictionary")
    metrics: dict[str, float | int] = {
        key: float(value) for key, value in remote_metrics.items()
    }
    metrics.update(
        {
            "step": step,
            "learning_rate": learning_rate,
            "num_pairs": len(pairs),
            "policy_tokens": sum(
                len(datum.model_input) if hasattr(datum.model_input, "__len__")
                else datum.model_input.length
                for datum in datums
            ),
        }
    )
    return SecPOStepResult(
        metrics=metrics,
        backward_result=backward_result,
        optimizer_result=optimizer_result,
    )
