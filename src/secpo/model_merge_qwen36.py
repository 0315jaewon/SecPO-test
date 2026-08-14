"""Fail-closed Qwen3.6 LoRA merge with multimodal-weight preservation.

Heavy model dependencies are imported only by :func:`merge_qwen36_lora`, so
release validation and ``--help`` remain dependency-light.  The implementation
preserves the useful PEFT key remapping from the research utility while
removing machine paths and scheduler assumptions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Sequence
from uuid import uuid4


MERGE_SCHEMA_VERSION = "secpo-qwen36-lora-merge-v1"
TOKENIZER_FILES = (
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
)
SANITY_CHECK_KEYS = (
    "model.language_model.layers.11.self_attn.q_proj.weight",
    "language_model.model.layers.11.self_attn.q_proj.weight",
    "model.layers.11.self_attn.q_proj.weight",
)

# Text-only Qwen LoRA checkpoints and multimodal Qwen base checkpoints use
# different module prefixes.  PEFT applies these anchored substitutions while
# loading the adapter so visual tensors can remain in the merged checkpoint.
MULTIMODAL_CAUSAL_LM_KEY_MAPPING = {
    r"^base_model\.model\.model\.layers\.": (
        "base_model.model.model.language_model.layers."
    ),
    r"^model\.model\.layers\.": "model.language_model.layers.",
    r"^model\.layers\.": "model.language_model.layers.",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_tokenizer(source: Path, destination: Path) -> list[str]:
    """Copy only the documented tokenizer assets and return their names."""

    copied: list[str] = []
    for name in TOKENIZER_FILES:
        candidate = source / name
        if candidate.is_file() and not candidate.is_symlink():
            shutil.copy2(candidate, destination / name)
            copied.append(name)
    return copied


def lora_b_weight_norm(model: Any) -> tuple[int, float, list[tuple[str, float]]]:
    """Return count, combined norm, and a small diagnostic LoRA-B sample."""

    total_squared = 0.0
    count = 0
    samples: list[tuple[str, float]] = []
    for module_name, module in model.named_modules():
        lora_b = getattr(module, "lora_B", None)
        if lora_b is None or not hasattr(lora_b, "items"):
            continue
        for adapter_name, layer in lora_b.items():
            weight = getattr(layer, "weight", None)
            if weight is None:
                continue
            norm = weight.detach().float().norm().item()
            total_squared += norm * norm
            count += 1
            if len(samples) < 5:
                samples.append((f"{module_name}.lora_B.{adapter_name}", norm))
    return count, total_squared**0.5, samples


def _safe_input_directory(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link: {expanded}")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{label} must be a direct directory: {resolved}")
    return resolved


def _tensor_keys(output: Path) -> list[str]:
    index_path = output / "model.safetensors.index.json"
    if index_path.is_file():
        value = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = value.get("weight_map", {}) if isinstance(value, dict) else {}
        if (
            not isinstance(weight_map, dict)
            or not weight_map
            or any(
                not isinstance(key, str)
                or not key
                or not isinstance(shard, str)
                or not shard
                for key, shard in weight_map.items()
            )
        ):
            raise RuntimeError("saved safetensors index has an invalid weight map")
        observed: set[str] = set()
        for shard_name in sorted(set(weight_map.values())):
            if Path(shard_name).name != shard_name:
                raise RuntimeError("saved safetensors index contains an unsafe shard path")
            shard_path = output / shard_name
            if not shard_path.is_file() or shard_path.is_symlink():
                raise RuntimeError(f"saved safetensors shard is missing: {shard_name}")
            from safetensors import safe_open

            with safe_open(str(shard_path), framework="pt") as handle:
                shard_keys = set(handle.keys())
            expected = {
                key for key, mapped_shard in weight_map.items() if mapped_shard == shard_name
            }
            if shard_keys != expected:
                raise RuntimeError(
                    f"saved safetensors shard keys differ from index: {shard_name}"
                )
            observed.update(shard_keys)
        if observed != set(weight_map):
            raise RuntimeError("saved safetensors keys differ from the index")
        return sorted(observed)
    single = output / "model.safetensors"
    if single.is_file():
        from safetensors import safe_open

        with safe_open(str(single), framework="pt") as handle:
            return list(handle.keys())
    raise RuntimeError("merged output contains no safetensors weights")


def _require_tokenizer_assets(copied: Sequence[str]) -> None:
    names = set(copied)
    if "tokenizer_config.json" not in names or not names.intersection(
        {"tokenizer.json", "vocab.json"}
    ):
        raise RuntimeError(
            "merged output needs tokenizer_config.json and tokenizer.json or vocab.json"
        )


def _write_manifest(
    output: Path,
    *,
    loader_name: str,
    multimodal: bool,
    lora_count: int,
    lora_norm: float,
    copied_tokenizer_files: Sequence[str],
    template: Path | None,
    tensor_keys: Sequence[str],
) -> None:
    manifest = {
        "schema_version": MERGE_SCHEMA_VERSION,
        "loader": loader_name,
        "multimodal": multimodal,
        "lora_b_tensor_count": lora_count,
        "lora_b_combined_norm": lora_norm,
        "tokenizer_files": list(copied_tokenizer_files),
        "template_sha256": None if template is None else sha256_file(template),
        "saved_tensor_count": len(tensor_keys),
        "saved_visual_tensor_count": sum("visual." in key for key in tensor_keys),
    }
    temporary = output / ".merge_manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output / "merge_manifest.json")


def merge_qwen36_lora(
    *,
    base: Path,
    lora: Path,
    output: Path,
    tokenizer_fallback: Path | None = None,
    template: Path | None = None,
) -> None:
    """Merge one Qwen LoRA adapter without overwriting an existing output."""

    base = _safe_input_directory(base, "base")
    lora = _safe_input_directory(lora, "lora")
    fallback = (
        base
        if tokenizer_fallback is None
        else _safe_input_directory(tokenizer_fallback, "tokenizer fallback")
    )
    if template is not None:
        expanded_template = template.expanduser()
        if expanded_template.is_symlink():
            raise ValueError("template must not be a symbolic link")
        template = expanded_template.resolve(strict=True)
        if not template.is_file():
            raise ValueError("template must be a direct regular file")
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {output.parent}")

    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM

    architecture = AutoConfig.from_pretrained(
        str(base), trust_remote_code=True
    ).architectures or []
    multimodal = any(
        "ConditionalGeneration" in name or "ImageTextToText" in name
        for name in architecture
    )
    if multimodal:
        from transformers import AutoModelForImageTextToText

        loader = AutoModelForImageTextToText
    else:
        loader = AutoModelForCausalLM
    base_model = loader.from_pretrained(
        str(base),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    base_state = base_model.state_dict()
    sanity_key = next((key for key in SANITY_CHECK_KEYS if key in base_state), None)
    sanity_before = (
        None
        if sanity_key is None
        else base_state[sanity_key].detach().float().clone()
    )

    key_mapping = None
    if multimodal:
        key_mapping = dict(
            getattr(base_model, "_checkpoint_conversion_mapping", {}) or {}
        )
        key_mapping.update(MULTIMODAL_CAUSAL_LM_KEY_MAPPING)
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(lora),
        key_mapping=key_mapping,
    )
    lora_count, lora_norm, samples = lora_b_weight_norm(peft_model)
    if lora_count == 0 or lora_norm == 0.0:
        raise RuntimeError(
            "LoRA tensors were not loaded; check adapter prefixes and key mapping"
        )
    print(
        f"Loaded {lora_count} LoRA-B tensors (combined norm={lora_norm:.6g}); "
        f"samples={samples}",
        flush=True,
    )
    merged = peft_model.merge_and_unload()
    if sanity_before is not None and sanity_key is not None:
        delta = merged.state_dict()[sanity_key].detach().float() - sanity_before
        if delta.norm().item() == 0.0:
            print(
                "Warning: the diagnostic bf16 weight did not change; the loaded "
                "LoRA norm was nevertheless nonzero.",
                flush=True,
            )

    staging = output.parent / f".{output.name}.merge-{uuid4().hex}"
    staging.mkdir()
    try:
        merged.save_pretrained(
            str(staging), safe_serialization=True, max_shard_size="5GB"
        )
        keys = _tensor_keys(staging)
        if multimodal and not any("visual." in key for key in keys):
            raise RuntimeError(
                "multimodal base was selected but saved output has no visual tensors"
            )
        tokenizer_source = (
            lora
            if (lora / "tokenizer_config.json").is_file()
            and (
                (lora / "tokenizer.json").is_file()
                or (lora / "vocab.json").is_file()
            )
            else fallback
        )
        copied = copy_tokenizer(tokenizer_source, staging)
        _require_tokenizer_assets(copied)
        if template is not None:
            shutil.copy2(template, staging / "chat_template.jinja")
            if "chat_template.jinja" not in copied:
                copied.append("chat_template.jinja")
        _write_manifest(
            staging,
            loader_name=loader.__name__,
            multimodal=multimodal,
            lora_count=lora_count,
            lora_norm=lora_norm,
            copied_tokenizer_files=copied,
            template=template,
            tensor_keys=keys,
        )
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output}")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--lora", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer-fallback", type=Path)
    parser.add_argument("--template", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    merge_qwen36_lora(
        base=args.base,
        lora=args.lora,
        output=args.output,
        tokenizer_fallback=args.tokenizer_fallback,
        template=args.template,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
