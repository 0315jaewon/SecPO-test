from __future__ import annotations

import json

import pytest

from secpo.tinker.config import load_tinker_config, parse_tinker_config


def _config() -> dict[str, object]:
    return {
        "model_name": "Qwen/Qwen3.8-27B",
        "tokenizer_revision": None,
        "dataset_name": "Sizhe-Chen/Qwen3.8-27B-Instruct-SecPO-trainset",
        "dataset_split": "train",
        "dataset_revision": None,
        "dataset_format": "qwen38_pre_rendered",
        "output_dir": "outputs/test",
        "expected_records": 18_951,
        "beta": 0.1,
        "max_length": 6_144,
        "max_prompt_length": 512,
        "learning_rate": 8e-5,
        "num_epochs": 1,
        "batch_size": 64,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "linear",
        "lora_rank": 64,
        "train_mlp": True,
        "train_attn": True,
        "train_unembed": False,
        "seed": 42,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_eps": 1e-8,
        "weight_decay": 0.0,
        "grad_clip_norm": 1.0,
        "save_every": 20,
        "checkpoint_ttl_seconds": 604800,
        "max_steps": None,
        "require_exact_source_fields": True,
    }


def test_repository_qwen38_config_loads() -> None:
    config = load_tinker_config("configs/train_qwen38_27b_tinker.json")

    assert config.model_name == "Qwen/Qwen3.8-27B"
    assert config.tokenizer_revision == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    assert config.expected_records == 18_951
    assert config.dataset_format == "qwen38_pre_rendered"
    assert config.dataset_revision == "7fd85bf8b670370ad396a049ad11f0dad36364af"
    assert config.learning_rate == 8e-5
    assert config.train_unembed is False
    assert config.save_every == 0
    assert config.checkpoint_ttl_seconds is None


def test_repository_generic_example_loads() -> None:
    config = load_tinker_config("configs/train_generic_tinker.example.json")

    assert config.dataset_format == "secpo_pre_rendered"
    assert config.model_name == "Qwen/Qwen3.5-9B"
    assert config.save_every == 0
    assert config.checkpoint_ttl_seconds is None


def test_lr6e5_variant_changes_only_learning_rate_and_output() -> None:
    from dataclasses import asdict

    baseline = asdict(load_tinker_config("configs/train_qwen38_27b_tinker.json"))
    variant = asdict(load_tinker_config("configs/train_qwen38_27b_tinker_lr6e5.json"))
    assert variant["learning_rate"] == 6e-5
    assert variant["output_dir"] == "outputs/qwen38_27b_tinker_lr6e5"
    assert {k for k in baseline if baseline[k] != variant[k]} == {"learning_rate", "output_dir"}


def test_lr4e5_variant_changes_only_learning_rate_and_output() -> None:
    from dataclasses import asdict

    previous = asdict(load_tinker_config("configs/train_qwen38_27b_tinker_lr6e5.json"))
    variant = asdict(load_tinker_config("configs/train_qwen38_27b_tinker_lr4e5.json"))
    assert variant["learning_rate"] == 4e-5
    assert variant["output_dir"] == "outputs/qwen38_27b_tinker_lr4e5"
    assert variant["save_every"] == 0
    assert variant["checkpoint_ttl_seconds"] is None
    assert {k for k in previous if previous[k] != variant[k]} == {"learning_rate", "output_dir"}


def test_config_accepts_only_an_immutable_optional_dataset_revision(tmp_path) -> None:
    raw = _config()
    raw["dataset_revision"] = "a" * 40
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    assert load_tinker_config(path).dataset_revision == "a" * 40
    raw["dataset_revision"] = "main"
    with pytest.raises(ValueError, match="immutable"):
        parse_tinker_config(raw)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("learning_rate", 0, "allowed range"),
        ("batch_size", True, "positive integer"),
        ("max_prompt_length", 7_000, "must not exceed"),
        ("adam_beta2", 1.0, "less than 1"),
        ("train_mlp", "true", "must be a boolean"),
        ("dataset_format", "autodetect", "dataset_format must be"),
    ],
)
def test_config_rejects_invalid_training_values(
    field: str, value: object, message: str
) -> None:
    raw = _config()
    raw[field] = value
    with pytest.raises(ValueError, match=message):
        parse_tinker_config(raw)


def test_config_requires_a_trainable_lora_group() -> None:
    raw = _config()
    raw.update(train_mlp=False, train_attn=False, train_unembed=False)
    with pytest.raises(ValueError, match="at least one"):
        parse_tinker_config(raw)
