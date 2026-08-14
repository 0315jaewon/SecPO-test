from __future__ import annotations

import json

import pytest

from secpo.model_merge_qwen36 import (
    MERGE_SCHEMA_VERSION,
    MULTIMODAL_CAUSAL_LM_KEY_MAPPING,
    TOKENIZER_FILES,
    _require_tokenizer_assets,
    _tensor_keys,
    _write_manifest,
    copy_tokenizer,
)


def test_multimodal_lora_prefix_mapping_is_preserved() -> None:
    assert MULTIMODAL_CAUSAL_LM_KEY_MAPPING == {
        r"^base_model\.model\.model\.layers\.": (
            "base_model.model.model.language_model.layers."
        ),
        r"^model\.model\.layers\.": "model.language_model.layers.",
        r"^model\.layers\.": "model.language_model.layers.",
    }


def test_tokenizer_copy_is_allowlisted(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    for name in TOKENIZER_FILES:
        (source / name).write_text(name, encoding="utf-8")
    (source / "private_notes.txt").write_text("not copied", encoding="utf-8")

    copied = copy_tokenizer(source, output)

    assert copied == list(TOKENIZER_FILES)
    assert not (output / "private_notes.txt").exists()


def test_merge_manifest_omits_source_paths(tmp_path) -> None:
    _write_manifest(
        tmp_path,
        loader_name="AutoModelForCausalLM",
        multimodal=False,
        lora_count=2,
        lora_norm=1.5,
        copied_tokenizer_files=["tokenizer.json"],
        template=None,
        tensor_keys=["model.layers.0.weight"],
    )
    value = json.loads((tmp_path / "merge_manifest.json").read_text())
    assert value["schema_version"] == MERGE_SCHEMA_VERSION
    assert "base" not in value and "lora" not in value


def test_tokenizer_assets_are_required() -> None:
    _require_tokenizer_assets(["tokenizer_config.json", "tokenizer.json"])
    with pytest.raises(RuntimeError, match="needs tokenizer"):
        _require_tokenizer_assets(["tokenizer_config.json"])


def test_index_rejects_missing_weight_shard(tmp_path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.weight": "missing.safetensors"}}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="shard is missing"):
        _tensor_keys(tmp_path)
