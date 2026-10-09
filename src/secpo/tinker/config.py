"""Strict, dependency-light configuration for the Tinker SecPO backend."""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
import math
from pathlib import Path
import re
from typing import Any


_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


@dataclass(frozen=True)
class TinkerSecPOConfig:
    """One pre-rendered offline SecPO run on Tinker."""

    model_name: str
    tokenizer_revision: str | None
    dataset_name: str
    dataset_split: str
    dataset_revision: str | None
    dataset_format: str
    output_dir: str
    expected_records: int
    beta: float
    max_length: int
    max_prompt_length: int
    learning_rate: float
    num_epochs: int
    batch_size: int
    warmup_ratio: float
    lr_scheduler_type: str
    lora_rank: int
    train_mlp: bool
    train_attn: bool
    train_unembed: bool
    seed: int
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    weight_decay: float
    grad_clip_norm: float
    save_every: int
    checkpoint_ttl_seconds: int | None
    max_steps: int | None
    require_exact_source_fields: bool


_CONFIG_FIELDS = {field.name for field in fields(TinkerSecPOConfig)}


def _identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    value = value.strip()
    if len(value.split("/")) != 2 or value.startswith((".", "~", "/")):
        raise ValueError(f"{name} must use the 'organization/name' form")
    return value


def _positive_int(name: str, value: Any, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _number(
    name: str,
    value: Any,
    *,
    lower: float,
    upper: float | None = None,
    inclusive_lower: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    lower_ok = result >= lower if inclusive_lower else result > lower
    if not math.isfinite(result) or not lower_ok or (
        upper is not None and result > upper
    ):
        raise ValueError(f"{name} is outside its allowed range")
    return result


def parse_tinker_config(raw: Any) -> TinkerSecPOConfig:
    """Validate a decoded Tinker configuration without importing its SDK."""

    if not isinstance(raw, dict):
        raise ValueError("Tinker SecPO config must be a JSON object")
    missing = _CONFIG_FIELDS - set(raw)
    unexpected = set(raw) - _CONFIG_FIELDS
    if missing or unexpected:
        raise ValueError(
            f"Tinker SecPO config fields differ: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    values = dict(raw)
    values["model_name"] = _identifier("model_name", values["model_name"])
    values["dataset_name"] = _identifier("dataset_name", values["dataset_name"])

    for name in ("dataset_split", "output_dir"):
        if not isinstance(values[name], str) or not values[name].strip():
            raise ValueError(f"{name} must be a non-empty string")
        values[name] = values[name].strip()
    for name in ("tokenizer_revision", "dataset_revision"):
        revision = values[name]
        if revision is not None and (
            not isinstance(revision, str) or _COMMIT.fullmatch(revision) is None
        ):
            raise ValueError(
                f"{name} must be null or an immutable 40-character commit"
            )
    if values["dataset_format"] not in {
        "qwen38_pre_rendered",
        "secpo_pre_rendered",
    }:
        raise ValueError(
            "dataset_format must be qwen38_pre_rendered or secpo_pre_rendered"
        )

    for name in (
        "expected_records",
        "max_length",
        "max_prompt_length",
        "num_epochs",
        "batch_size",
        "lora_rank",
    ):
        _positive_int(name, values[name])
    _positive_int("seed", values["seed"], allow_zero=True)
    _positive_int("save_every", values["save_every"], allow_zero=True)
    if values["checkpoint_ttl_seconds"] is not None:
        _positive_int(
            "checkpoint_ttl_seconds", values["checkpoint_ttl_seconds"]
        )
    if values["max_steps"] is not None:
        _positive_int("max_steps", values["max_steps"])
    if values["max_prompt_length"] > values["max_length"]:
        raise ValueError("max_prompt_length must not exceed max_length")

    _number("beta", values["beta"], lower=0.0)
    _number("learning_rate", values["learning_rate"], lower=0.0)
    _number(
        "warmup_ratio",
        values["warmup_ratio"],
        lower=0.0,
        upper=1.0,
        inclusive_lower=True,
    )
    _number(
        "adam_beta1",
        values["adam_beta1"],
        lower=0.0,
        upper=1.0,
        inclusive_lower=True,
    )
    _number(
        "adam_beta2",
        values["adam_beta2"],
        lower=0.0,
        upper=1.0,
        inclusive_lower=True,
    )
    if values["adam_beta1"] >= 1 or values["adam_beta2"] >= 1:
        raise ValueError("Adam beta values must be less than 1")
    _number("adam_eps", values["adam_eps"], lower=0.0)
    _number(
        "weight_decay",
        values["weight_decay"],
        lower=0.0,
        inclusive_lower=True,
    )
    _number(
        "grad_clip_norm",
        values["grad_clip_norm"],
        lower=0.0,
        inclusive_lower=True,
    )
    if values["lr_scheduler_type"] not in {"linear", "constant", "cosine"}:
        raise ValueError("lr_scheduler_type must be linear, constant, or cosine")
    for name in (
        "train_mlp",
        "train_attn",
        "train_unembed",
        "require_exact_source_fields",
    ):
        if type(values[name]) is not bool:
            raise ValueError(f"{name} must be a boolean")
    if (
        not values["train_mlp"]
        and not values["train_attn"]
        and not values["train_unembed"]
    ):
        raise ValueError("at least one LoRA layer group must be trainable")
    return TinkerSecPOConfig(**values)


def load_tinker_config(path: str | Path) -> TinkerSecPOConfig:
    """Load a strict JSON configuration from disk."""

    source = Path(path).expanduser()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid Tinker SecPO JSON: {error.msg}") from error
    return parse_tinker_config(raw)
