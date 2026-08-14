"""Shared JSON-configured CLI for the released Llama and Qwen SecPO runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Sequence

# Keep the release self-contained and local-only with respect to experiment
# reporting. Model files may still be downloaded from the model hub.
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DISABLE_TELEMETRY"] = "1"

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from .data import BenignReferenceCollator, load_preference_dataset
from .online_training import FileAttackExchange, OnlineAttackCollator
from .trainer import SecPOTrainer


@dataclass(frozen=True)
class SecPOTrainingConfig:
    """Strict schema for a released main-experiment JSON configuration."""

    model_name_or_path: str
    model_revision: str
    data_file: str
    output_dir: str
    beta: float
    max_length: int
    learning_rate: float
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    bf16: bool
    gradient_checkpointing: bool
    gradient_checkpointing_use_reentrant: bool
    lora_r: int
    lora_alpha: float
    lora_dropout: float
    lora_target_modules: tuple[str, ...]
    logging_steps: int
    save_strategy: str
    report_to: str
    save_steps: int = 500
    max_prompt_length: int | None = None
    seed: int = 42
    data_seed: int | None = None
    dataloader_drop_last: bool = False
    max_steps: int = -1
    attn_implementation: str | None = None
    torch_dtype: str = "bfloat16"
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "linear"
    max_grad_norm: float = 1.0
    ddp_find_unused_parameters: bool = False
    append_eos: bool = True
    trust_remote_code: bool = False
    online_exchange_dir: str | None = None
    online_timeout_seconds: float = 1800.0
    online_poll_seconds: float = 1.0
    online_retain_files: bool = False


_REQUIRED_CONFIG_FIELDS = frozenset(
    {
        "model_name_or_path",
        "model_revision",
        "data_file",
        "output_dir",
        "beta",
        "max_length",
        "learning_rate",
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "warmup_ratio",
        "bf16",
        "gradient_checkpointing",
        "gradient_checkpointing_use_reentrant",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
        "logging_steps",
        "save_strategy",
        "report_to",
    }
)


def _positive_int(name: str, value: Any, *, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def _finite_number(
    name: str,
    value: Any,
    *,
    lower: float,
    upper: float | None = None,
    lower_inclusive: bool = False,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    number = float(value)
    lower_ok = number >= lower if lower_inclusive else number > lower
    upper_ok = upper is None or number <= upper
    if not math.isfinite(number) or not lower_ok or not upper_ok:
        interval = f"{'[' if lower_inclusive else '('}{lower}, "
        interval += f"{upper if upper is not None else 'infinity'}]"
        raise ValueError(f"{name} must be in {interval}, got {value!r}")


def _public_model_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("model_name_or_path must be a non-empty public model id")
    value = value.strip()
    path = Path(value)
    if path.is_absolute() or value.startswith((".", "~")) or path.exists():
        raise ValueError(
            "model_name_or_path must be a public model id, not a local path; "
            "PEFT stores this value in adapter metadata"
        )
    pieces = value.split("/")
    if len(pieces) != 2 or any(not piece for piece in pieces):
        raise ValueError(
            "model_name_or_path must use the public 'organization/model' form"
        )
    return value


def _immutable_model_revision(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(
            "model_revision must be a lowercase 40-character commit digest"
        )
    return value


def _validate_online_checkpoint_publication(config: SecPOTrainingConfig) -> None:
    """Require a loadable current-policy adapter after every optimizer step."""

    if config.online_exchange_dir is None:
        return
    if config.save_strategy != "steps" or config.save_steps != 1:
        raise ValueError(
            "online current-policy training requires save_strategy='steps' "
            "and save_steps=1 so every optimizer step publishes a PEFT "
            "checkpoint for the attack worker"
        )


def parse_training_config(raw: dict[str, Any]) -> SecPOTrainingConfig:
    """Validate a decoded configuration object and return its typed form."""

    if not isinstance(raw, dict):
        raise ValueError("the training configuration must be a JSON object")
    allowed = {field.name for field in fields(SecPOTrainingConfig)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown training configuration keys: {sorted(unknown)}")
    missing = _REQUIRED_CONFIG_FIELDS - set(raw)
    if missing:
        raise ValueError(
            f"missing required training configuration keys: {sorted(missing)}"
        )

    values = dict(raw)
    values["model_name_or_path"] = _public_model_id(values["model_name_or_path"])
    values["model_revision"] = _immutable_model_revision(values["model_revision"])
    for name in ("data_file", "output_dir"):
        if not isinstance(values[name], str) or not values[name].strip():
            raise ValueError(f"{name} must be a non-empty string")
        values[name] = values[name].strip()

    _finite_number("beta", values["beta"], lower=0.0)
    _positive_int("max_length", values["max_length"], minimum=2)
    _finite_number("learning_rate", values["learning_rate"], lower=0.0)
    _finite_number("num_train_epochs", values["num_train_epochs"], lower=0.0)
    _positive_int(
        "per_device_train_batch_size",
        values["per_device_train_batch_size"],
    )
    _positive_int(
        "gradient_accumulation_steps",
        values["gradient_accumulation_steps"],
    )
    _finite_number(
        "warmup_ratio",
        values["warmup_ratio"],
        lower=0.0,
        upper=1.0,
        lower_inclusive=True,
    )

    for name in (
        "bf16",
        "gradient_checkpointing",
        "gradient_checkpointing_use_reentrant",
        "ddp_find_unused_parameters",
        "append_eos",
        "trust_remote_code",
        "online_retain_files",
        "dataloader_drop_last",
    ):
        if name in values and type(values[name]) is not bool:
            raise ValueError(f"{name} must be a boolean")

    _positive_int("lora_r", values["lora_r"])
    _finite_number("lora_alpha", values["lora_alpha"], lower=0.0)
    _finite_number(
        "lora_dropout",
        values["lora_dropout"],
        lower=0.0,
        upper=1.0,
        lower_inclusive=True,
    )
    modules = values["lora_target_modules"]
    if (
        not isinstance(modules, list)
        or not modules
        or any(not isinstance(name, str) or not name.strip() for name in modules)
    ):
        raise ValueError("lora_target_modules must be a non-empty list of strings")
    normalized_modules = tuple(name.strip() for name in modules)
    if len(set(normalized_modules)) != len(normalized_modules):
        raise ValueError("lora_target_modules must not contain duplicates")
    values["lora_target_modules"] = normalized_modules

    _positive_int("logging_steps", values["logging_steps"])
    if values["save_strategy"] not in {"no", "steps", "epoch"}:
        raise ValueError("save_strategy must be one of: no, steps, epoch")
    if "save_steps" in values:
        _positive_int("save_steps", values["save_steps"])
    if values["report_to"] != "none":
        raise ValueError(
            "report_to must be 'none' in the clean release; external "
            "experiment telemetry is disabled"
        )

    if "max_prompt_length" in values and values["max_prompt_length"] is not None:
        _positive_int("max_prompt_length", values["max_prompt_length"])
    if "seed" in values:
        _positive_int("seed", values["seed"], minimum=0)
    if "data_seed" in values and values["data_seed"] is not None:
        _positive_int("data_seed", values["data_seed"], minimum=0)
    if "max_steps" in values:
        max_steps = values["max_steps"]
        if type(max_steps) is not int or max_steps == 0 or max_steps < -1:
            raise ValueError("max_steps must be -1 or a positive integer")
    if "attn_implementation" in values and values["attn_implementation"] is not None:
        if (
            not isinstance(values["attn_implementation"], str)
            or not values["attn_implementation"].strip()
        ):
            raise ValueError("attn_implementation must be a non-empty string or null")
    if values.get("torch_dtype", "bfloat16") not in {
        "bfloat16",
        "float16",
        "float32",
    }:
        raise ValueError("torch_dtype must be bfloat16, float16, or float32")
    if not isinstance(values.get("optim", "adamw_torch"), str) or not values.get(
        "optim",
        "adamw_torch",
    ).strip():
        raise ValueError("optim must be a non-empty string")
    if values.get("lr_scheduler_type", "linear") not in {
        "linear",
        "cosine",
        "constant",
        "constant_with_warmup",
    }:
        raise ValueError(
            "lr_scheduler_type must be linear, cosine, constant, or "
            "constant_with_warmup"
        )
    if "max_grad_norm" in values:
        _finite_number(
            "max_grad_norm",
            values["max_grad_norm"],
            lower=0.0,
            lower_inclusive=True,
        )
    if values.get("online_exchange_dir") is not None:
        exchange_dir = values["online_exchange_dir"]
        if not isinstance(exchange_dir, str) or not exchange_dir.strip():
            raise ValueError("online_exchange_dir must be a non-empty path or null")
        values["online_exchange_dir"] = exchange_dir.strip()
    for name in ("online_timeout_seconds", "online_poll_seconds"):
        if name in values:
            _finite_number(name, values[name], lower=0.0)

    config = SecPOTrainingConfig(**values)
    if config.bf16 and config.torch_dtype != "bfloat16":
        raise ValueError("bf16=true requires torch_dtype='bfloat16'")
    _validate_online_checkpoint_publication(config)
    return config


def load_training_config(path: str | Path) -> SecPOTrainingConfig:
    """Read and strictly validate a SecPO model/training JSON file."""

    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"training config does not exist: {config_path}")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid JSON in training config {config_path}: {error.msg}"
        ) from error
    return parse_training_config(raw)


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _load_tokenizer(config: SecPOTrainingConfig) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        revision=config.model_revision,
        use_fast=True,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("the selected tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _load_policy(config: SecPOTrainingConfig) -> torch.nn.Module:
    model_kwargs: dict[str, Any] = {
        "dtype": _torch_dtype(config.torch_dtype),
        "trust_remote_code": config.trust_remote_code,
    }
    if config.attn_implementation is not None:
        model_kwargs["attn_implementation"] = config.attn_implementation

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        revision=config.model_revision,
        **model_kwargs,
    )
    base_model.config.use_cache = False
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(config.lora_target_modules),
    )
    policy = get_peft_model(base_model, lora_config)
    if config.gradient_checkpointing:
        # The frozen embedding output must require gradients for checkpointed
        # LoRA layers downstream. This hook does not unfreeze the embeddings.
        policy.enable_input_require_grads()
    return policy


def _filter_prompt_budget(
    dataset: Any,
    tokenizer: Any,
    max_prompt_length: int | None,
) -> Any:
    if max_prompt_length is None:
        return dataset

    def within_budget(example: dict[str, Any]) -> bool:
        attacked_length = len(
            tokenizer(
                example["prompt"],
                add_special_tokens=False,
            )["input_ids"]
        )
        clean_length = len(
            tokenizer(
                example["clean_prompt"],
                add_special_tokens=False,
            )["input_ids"]
        )
        return max(attacked_length, clean_length) <= max_prompt_length

    original_size = len(dataset)
    dataset = dataset.filter(
        within_budget,
        desc=f"Filtering prompts over {max_prompt_length} tokens",
    )
    if len(dataset) == 0:
        raise ValueError(
            f"all {original_size} records exceeded max_prompt_length="
            f"{max_prompt_length}"
        )
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            f"Prompt-length filter kept {len(dataset)}/{original_size} records.",
            flush=True,
        )
    return dataset


def _build_training_arguments(config: SecPOTrainingConfig) -> TrainingArguments:
    """Build Trainer arguments, keeping model-specific data controls explicit."""

    return TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        bf16=config.bf16,
        fp16=False,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={
            "use_reentrant": config.gradient_checkpointing_use_reentrant
        },
        optim=config.optim,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=config.ddp_find_unused_parameters,
        seed=config.seed,
        data_seed=config.seed if config.data_seed is None else config.data_seed,
        dataloader_drop_last=config.dataloader_drop_last,
        max_steps=config.max_steps,
    )


def build_trainer(config: SecPOTrainingConfig) -> tuple[SecPOTrainer, Any]:
    """Construct the policy, validated dataset, collator, and Trainer."""

    # ``dataclasses.replace``-based CLI overrides bypass JSON parsing, so keep
    # the online current-policy publication invariant at construction too.
    _validate_online_checkpoint_publication(config)

    tokenizer = _load_tokenizer(config)
    policy = _load_policy(config)
    dataset = load_preference_dataset(config.data_file)
    dataset = _filter_prompt_budget(
        dataset,
        tokenizer,
        config.max_prompt_length,
    )

    training_args = _build_training_arguments(config)
    collator: Any = BenignReferenceCollator(
        tokenizer=tokenizer,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        append_eos=config.append_eos,
    )
    if config.online_exchange_dir is not None:
        # Keep the exchange in the rank's main process. Multiple DataLoader
        # workers would each own an independent request sequence.
        exchange = FileAttackExchange(
            config.online_exchange_dir,
            timeout_seconds=config.online_timeout_seconds,
            poll_seconds=config.online_poll_seconds,
            retain_files=config.online_retain_files,
        )
        collator = OnlineAttackCollator(
            collator,
            exchange,
            rank=int(os.environ.get("RANK", "0")),
        )
    trainer = SecPOTrainer(
        model=policy,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
        beta=config.beta,
    )
    return trainer, tokenizer


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the benign-reference SecPO objective."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a strict SecPO model/training JSON configuration.",
    )
    parser.add_argument(
        "--data-file",
        help="Override the preference-data path from the JSON configuration.",
    )
    parser.add_argument(
        "--model-name-or-path",
        help="Override the public model id from the JSON configuration.",
    )
    parser.add_argument(
        "--model-revision",
        help="Override the model with an immutable 40-character commit digest.",
    )
    parser.add_argument(
        "--output-dir",
        help="Override the local output directory from the JSON configuration.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Resume from a local Transformers checkpoint directory.",
    )
    parser.add_argument(
        "--online-exchange-dir",
        help=(
            "Enable batch-time attack generation through this shared runtime "
            "directory. Every record must have a stable id."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _argument_parser().parse_args(argv)
    config = load_training_config(args.config)
    overrides: dict[str, Any] = {}
    if args.data_file is not None:
        overrides["data_file"] = args.data_file
    if args.model_name_or_path is not None and args.model_revision is None:
        raise SystemExit(
            "--model-name-or-path requires the matching --model-revision"
        )
    if args.model_name_or_path is not None:
        overrides["model_name_or_path"] = _public_model_id(args.model_name_or_path)
    if args.model_revision is not None:
        overrides["model_revision"] = _immutable_model_revision(args.model_revision)
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.online_exchange_dir is not None:
        overrides["online_exchange_dir"] = args.online_exchange_dir
    config = replace(config, **overrides)

    trainer, tokenizer = build_trainer(config)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(config.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(config.output_dir)


if __name__ == "__main__":
    main()
