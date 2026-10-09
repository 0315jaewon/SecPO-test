"""Guarded end-to-end orchestration for the Tinker SecPO backend."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Any

from .bridge import _sdk
from .config import TinkerSecPOConfig
from .data import TokenizedSecPOPair, tokenize_pre_rendered_example
from .dataset import load_huggingface_split, prepare_record, validate_dataset_size
from .runtime import (
    SecPOStepResult,
    TinkerClients,
    create_tinker_clients,
    train_secpo_step,
)


@dataclass(frozen=True)
class BatchPlan:
    """Stable record indices for one update."""

    step: int
    epoch: int
    batch: int
    indices: tuple[int, ...]


@dataclass(frozen=True)
class TrainingPlan:
    """Deterministic update schedule computed before remote training."""

    record_count: int
    batches_per_epoch: int
    total_steps: int
    batches: tuple[BatchPlan, ...]


@dataclass(frozen=True)
class SecPORunResult:
    """Paths and progress returned by a preflight or completed run."""

    plan: TrainingPlan
    steps_completed: int
    final_state_path: str | None = None
    final_sampler_path: str | None = None
    metrics_path: str | None = None
    token_preflight: TokenPreflightStats | None = None


@dataclass(frozen=True)
class TokenPreflightStats:
    """Maximum lengths observed while tokenizing the complete dataset."""

    max_policy_prompt: int
    max_reference_prompt: int
    max_sequence: int


def build_training_plan(
    *,
    record_count: int,
    batch_size: int,
    num_epochs: int,
    seed: int,
    max_steps: int | None,
) -> TrainingPlan:
    """Shuffle once per epoch and split every record into stable batches."""

    for name, value in {
        "record_count": record_count,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
    }.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if max_steps is not None and (type(max_steps) is not int or max_steps <= 0):
        raise ValueError("max_steps must be null or a positive integer")

    batches_per_epoch = math.ceil(record_count / batch_size)
    limit = batches_per_epoch * num_epochs
    if max_steps is not None:
        limit = min(limit, max_steps)
    plans: list[BatchPlan] = []
    step = 0
    for epoch in range(num_epochs):
        indices = list(range(record_count))
        random.Random(seed + epoch).shuffle(indices)
        for batch in range(batches_per_epoch):
            if step >= limit:
                break
            start = batch * batch_size
            plans.append(
                BatchPlan(
                    step=step,
                    epoch=epoch,
                    batch=batch,
                    indices=tuple(indices[start : start + batch_size]),
                )
            )
            step += 1
        if step >= limit:
            break
    return TrainingPlan(
        record_count=record_count,
        batches_per_epoch=batches_per_epoch,
        total_steps=len(plans),
        batches=tuple(plans),
    )


def load_local_tokenizer(
    model_name: str,
    *,
    revision: str | None,
) -> Any:
    """Load only the Hugging Face tokenizer for preflight and training."""

    try:
        from transformers import AutoTokenizer
    except ImportError as error:  # pragma: no cover - installation error path
        raise RuntimeError(
            "token preflight requires the 'transformers' package"
        ) from error
    # Extended-context Tinker IDs append a deployment suffix to the ordinary
    # Hugging Face repository name while retaining the same tokenizer.
    huggingface_name = model_name.partition(":peft:")[0]
    kwargs: dict[str, Any] = {"token": True}
    if revision is not None:
        kwargs["revision"] = revision
    return AutoTokenizer.from_pretrained(huggingface_name, **kwargs)


def validate_all_records(
    dataset: Any,
    config: TinkerSecPOConfig,
    tokenizer: Any,
) -> TokenPreflightStats:
    """Validate and tokenize all rows before allocating a remote model."""

    max_policy_prompt = 0
    max_reference_prompt = 0
    max_sequence = 0
    for index in range(len(dataset)):
        example = prepare_record(
            dataset[index],
            dataset_format=config.dataset_format,
            source_index=index,
            require_exact_fields=config.require_exact_source_fields,
        )
        pair = tokenize_pre_rendered_example(
            example,
            tokenizer,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
        )
        max_policy_prompt = max(
            max_policy_prompt,
            pair.policy_chosen.prompt_length,
            pair.policy_rejected.prompt_length,
        )
        max_reference_prompt = max(
            max_reference_prompt,
            pair.reference_chosen.prompt_length,
            pair.reference_rejected.prompt_length,
        )
        max_sequence = max(
            max_sequence,
            pair.policy_chosen.full_length,
            pair.policy_rejected.full_length,
            pair.reference_chosen.full_length,
            pair.reference_rejected.full_length,
        )
    return TokenPreflightStats(
        max_policy_prompt=max_policy_prompt,
        max_reference_prompt=max_reference_prompt,
        max_sequence=max_sequence,
    )


def _tokenize_batch(
    dataset: Any,
    indices: Sequence[int],
    tokenizer: Any,
    config: TinkerSecPOConfig,
) -> list[TokenizedSecPOPair]:
    pairs: list[TokenizedSecPOPair] = []
    for index in indices:
        example = prepare_record(
            dataset[index],
            dataset_format=config.dataset_format,
            source_index=index,
            require_exact_fields=config.require_exact_source_fields,
        )
        pairs.append(
            tokenize_pre_rendered_example(
                example,
                tokenizer,
                max_length=config.max_length,
                max_prompt_length=config.max_prompt_length,
            )
        )
    return pairs


def _saved_path(result: Any, *, label: str) -> str:
    path = getattr(result, "path", None)
    if not isinstance(path, str) or not path:
        raise TypeError(f"Tinker {label} result must expose a non-empty path")
    return path


def _check_output_available(config: TinkerSecPOConfig) -> None:
    output_dir = Path(config.output_dir).expanduser()
    config_path = output_dir / "config.json"
    metrics_path = output_dir / "metrics.jsonl"
    if config_path.exists() or metrics_path.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing Tinker run in {output_dir}"
        )


def _prepare_output(config: TinkerSecPOConfig) -> tuple[Path, Path]:
    _check_output_available(config)
    output_dir = Path(config.output_dir).expanduser()
    config_path = output_dir / "config.json"
    metrics_path = output_dir / "metrics.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_dir, metrics_path


def _write_metric(path: Path, metric: dict[str, float | int | str]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(metric, sort_keys=True) + "\n")


def run_secpo_training(
    config: TinkerSecPOConfig,
    *,
    preflight_only: bool = False,
    dataset: Any | None = None,
    clients: TinkerClients | None = None,
    tokenizer: Any | None = None,
    tinker_module: Any | None = None,
    train_step_fn: Callable[..., SecPOStepResult] = train_secpo_step,
) -> SecPORunResult:
    """Validate, train, checkpoint, and export one SecPO run.

    ``preflight_only`` performs authenticated dataset loading, exact size and
    schema validation, full tokenizer-bound checks, and deterministic batch
    planning.  It creates no Tinker client and writes no local run directory.
    """

    if dataset is None:
        dataset = load_huggingface_split(
            config.dataset_name,
            split=config.dataset_split,
            revision=config.dataset_revision,
        )
    validate_dataset_size(dataset, expected_records=config.expected_records)
    if tokenizer is None:
        tokenizer = load_local_tokenizer(
            config.model_name,
            revision=config.tokenizer_revision,
        )
    token_preflight = validate_all_records(dataset, config, tokenizer)
    plan = build_training_plan(
        record_count=len(dataset),
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
        seed=config.seed,
        max_steps=config.max_steps,
    )
    if preflight_only:
        return SecPORunResult(
            plan=plan,
            steps_completed=0,
            token_preflight=token_preflight,
        )

    _check_output_available(config)
    tinker = _sdk(tinker_module)
    if clients is None:
        clients = create_tinker_clients(config, tinker_module=tinker)
    output_dir, metrics_path = _prepare_output(config)

    for batch_plan in plan.batches:
        pairs = _tokenize_batch(
            dataset,
            batch_plan.indices,
            tokenizer,
            config,
        )
        step_result = train_step_fn(
            pairs,
            training_client=clients.training,
            reference_client=clients.reference,
            config=config,
            step=batch_plan.step,
            total_steps=plan.total_steps,
            tinker_module=tinker,
        )
        metric: dict[str, float | int | str] = dict(step_result.metrics)
        metric.update(epoch=batch_plan.epoch, batch=batch_plan.batch)
        completed = batch_plan.step + 1
        if (
            config.save_every > 0
            and completed % config.save_every == 0
            and completed < plan.total_steps
        ):
            saved = clients.training.save_state(
                f"secpo-step-{completed:06d}",
                ttl_seconds=config.checkpoint_ttl_seconds,
            ).result()
            metric["state_path"] = _saved_path(saved, label="checkpoint")
        _write_metric(metrics_path, metric)

    state = clients.training.save_state(
        "secpo-final-state",
        ttl_seconds=config.checkpoint_ttl_seconds,
    ).result()
    sampler = clients.training.save_weights_for_sampler(
        "secpo-final-sampler",
        ttl_seconds=config.checkpoint_ttl_seconds,
    ).result()
    final_state_path = _saved_path(state, label="final state")
    final_sampler_path = _saved_path(sampler, label="final sampler")
    summary = {
        "steps_completed": plan.total_steps,
        "final_state_path": final_state_path,
        "final_sampler_path": final_sampler_path,
    }
    (output_dir / "result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return SecPORunResult(
        plan=plan,
        steps_completed=plan.total_steps,
        final_state_path=final_state_path,
        final_sampler_path=final_sampler_path,
        metrics_path=str(metrics_path),
        token_preflight=token_preflight,
    )
