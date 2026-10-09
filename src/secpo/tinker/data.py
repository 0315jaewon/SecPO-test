"""Validation and token preparation for the pre-rendered Qwen3.8 dataset.

The mentor-provided Qwen3.8 artifact already contains its complete chat
template prefix and response terminators.  Applying another renderer would
silently change the experiment.  This module therefore validates that exact
contract and exposes Tinker-independent shifted sequences that a runtime
adapter can later convert to ``tinker.Datum`` values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from secpo.data import PreferenceSchemaError, validate_preference_record


QWEN_END_TOKEN = "<|im_end|>"
QWEN38_NON_THINKING_SUFFIX = (
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
QWEN38_SOURCE_FIELDS = frozenset(
    {
        "prompt",
        "chosen_input",
        "rejected_input",
        "chosen_data",
        "inject_at_the_end",
        "rejected_input_whole",
        "chosen",
        "rejected",
    }
)
GENERIC_PRE_RENDERED_FIELDS = frozenset(
    {"prompt", "clean_prompt", "chosen", "rejected"}
)


@dataclass(frozen=True)
class PreparedSecPOExample:
    """The four strings consumed by pre-rendered SecPO training."""

    prompt: str
    clean_prompt: str
    chosen: str
    rejected: str
    source_index: int | None = None


@dataclass(frozen=True)
class ShiftedSequence:
    """One causal-LM sequence in Tinker's shift-by-one layout."""

    input_tokens: tuple[int, ...]
    target_tokens: tuple[int, ...]
    weights: tuple[float, ...]
    prompt_length: int
    response_length: int

    @property
    def full_length(self) -> int:
        """Return the unshifted prompt-plus-response length."""

        return len(self.input_tokens) + 1

    @property
    def full_tokens(self) -> tuple[int, ...]:
        """Reconstruct the unshifted sequence used for reference scoring."""

        return self.input_tokens + (self.target_tokens[-1],)


@dataclass(frozen=True)
class TokenizedSecPOPair:
    """The attacked-policy and clean-reference sequences for one pair."""

    policy_chosen: ShiftedSequence
    policy_rejected: ShiftedSequence
    reference_chosen: ShiftedSequence
    reference_rejected: ShiftedSequence
    source_index: int | None = None


def _where(source_index: int | None) -> str:
    return "record" if source_index is None else f"record {source_index}"


def _require_source_text(
    record: Mapping[str, Any],
    field: str,
    *,
    source_index: int | None,
) -> str:
    value = record[field]
    if not isinstance(value, str) or not value.strip():
        raise PreferenceSchemaError(
            f"{_where(source_index)} field {field!r} must be a non-empty string"
        )
    if "\x00" in value:
        raise PreferenceSchemaError(
            f"{_where(source_index)} field {field!r} contains a NUL byte"
        )
    return value


def _validate_non_thinking_completion(
    value: str,
    field: str,
    *,
    source_index: int | None,
) -> None:
    if not value.endswith(QWEN_END_TOKEN) or value.count(QWEN_END_TOKEN) != 1:
        raise PreferenceSchemaError(
            f"{_where(source_index)} field {field!r} must end in exactly one "
            f"{QWEN_END_TOKEN!r} token"
        )
    if "<think>" in value or "</think>" in value:
        raise PreferenceSchemaError(
            f"{_where(source_index)} field {field!r} contains a thinking tag in "
            "the thinking-off dataset"
        )


def prepare_qwen38_record(
    record: Mapping[str, Any],
    *,
    source_index: int | None = None,
    require_exact_fields: bool = True,
) -> PreparedSecPOExample:
    """Validate and minimize one Qwen3.8-Instruct SecPO source record.

    The published dataset currently promises one homogeneous eight-field
    schema.  Exact field validation detects upstream schema drift before a
    paid remote run.  Callers adapting an authorized derivative may opt out of
    exactness while all fields needed by this contract remain required.
    """

    if not isinstance(record, Mapping):
        raise PreferenceSchemaError(f"{_where(source_index)} must be a mapping")
    fields = set(record)
    missing = QWEN38_SOURCE_FIELDS - fields
    if missing:
        raise PreferenceSchemaError(
            f"{_where(source_index)} is missing Qwen3.8 source fields: "
            f"{sorted(missing)}"
        )
    if require_exact_fields:
        unexpected = fields - QWEN38_SOURCE_FIELDS
        if unexpected:
            raise PreferenceSchemaError(
                f"{_where(source_index)} has unexpected Qwen3.8 source fields: "
                f"{sorted(unexpected)}"
            )

    for field in QWEN38_SOURCE_FIELDS - {"inject_at_the_end"}:
        _require_source_text(record, field, source_index=source_index)
    if type(record["inject_at_the_end"]) is not bool:
        raise PreferenceSchemaError(
            f"{_where(source_index)} field 'inject_at_the_end' must be a boolean"
        )

    prompt = record["prompt"]
    chosen = record["chosen"]
    rejected = record["rejected"]
    injection = record["rejected_input_whole"]
    assert isinstance(prompt, str)
    assert isinstance(chosen, str)
    assert isinstance(rejected, str)
    assert isinstance(injection, str)

    if not prompt.endswith(QWEN38_NON_THINKING_SUFFIX):
        raise PreferenceSchemaError(
            f"{_where(source_index)} prompt does not end with the canonical "
            "Qwen3.8 thinking-off assistant prefix"
        )
    _validate_non_thinking_completion(
        chosen, "chosen", source_index=source_index
    )
    _validate_non_thinking_completion(
        rejected, "rejected", source_index=source_index
    )

    normalized = validate_preference_record(
        {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "rejected_input_whole": injection,
        },
        row_index=source_index,
    )
    return PreparedSecPOExample(
        prompt=normalized.prompt,
        clean_prompt=normalized.clean_prompt,
        chosen=normalized.chosen,
        rejected=normalized.rejected,
        source_index=source_index,
    )


def prepare_pre_rendered_record(
    record: Mapping[str, Any],
    *,
    source_index: int | None = None,
    require_exact_fields: bool = True,
) -> PreparedSecPOExample:
    """Validate the portable four-string SecPO dataset contract.

    Text must already be rendered for the selected model.  In particular,
    ``prompt`` and ``clean_prompt`` include the assistant-generation prefix,
    while ``chosen`` and ``rejected`` include any model-specific terminator.
    This deliberately avoids guessing a model's evolving chat template.
    """

    if not isinstance(record, Mapping):
        raise PreferenceSchemaError(f"{_where(source_index)} must be a mapping")
    fields = set(record)
    missing = GENERIC_PRE_RENDERED_FIELDS - fields
    if missing:
        raise PreferenceSchemaError(
            f"{_where(source_index)} is missing pre-rendered fields: "
            f"{sorted(missing)}"
        )
    if require_exact_fields:
        unexpected = fields - GENERIC_PRE_RENDERED_FIELDS
        if unexpected:
            raise PreferenceSchemaError(
                f"{_where(source_index)} has unexpected pre-rendered fields: "
                f"{sorted(unexpected)}"
            )
    values = {
        field: _require_source_text(record, field, source_index=source_index)
        for field in GENERIC_PRE_RENDERED_FIELDS
    }
    if values["prompt"] == values["clean_prompt"]:
        raise PreferenceSchemaError(
            f"{_where(source_index)} prompt and clean_prompt must differ"
        )
    if values["chosen"] == values["rejected"]:
        raise PreferenceSchemaError(
            f"{_where(source_index)} chosen and rejected must differ"
        )
    return PreparedSecPOExample(
        prompt=values["prompt"],
        clean_prompt=values["clean_prompt"],
        chosen=values["chosen"],
        rejected=values["rejected"],
        source_index=source_index,
    )


def _encode(tokenizer: Any, text: str, *, label: str) -> tuple[int, ...]:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise TypeError("tokenizer must expose an encode() method")
    encoded = encode(text, add_special_tokens=False)
    if (
        not isinstance(encoded, Sequence)
        or isinstance(encoded, (str, bytes))
        or not encoded
        or any(type(token) is not int or token < 0 for token in encoded)
    ):
        raise TypeError(
            f"tokenizer.encode() must return non-empty non-negative integers for {label}"
        )
    return tuple(encoded)


def _shifted_sequence(
    prompt_tokens: tuple[int, ...],
    response_tokens: tuple[int, ...],
) -> ShiftedSequence:
    full = prompt_tokens + response_tokens
    # At shifted position len(prompt)-1 the final prompt token predicts the
    # first response token, so exactly len(prompt)-1 weights are zero.
    weights = (0.0,) * (len(prompt_tokens) - 1) + (1.0,) * len(response_tokens)
    return ShiftedSequence(
        input_tokens=full[:-1],
        target_tokens=full[1:],
        weights=weights,
        prompt_length=len(prompt_tokens),
        response_length=len(response_tokens),
    )


def tokenize_pre_rendered_example(
    example: PreparedSecPOExample,
    tokenizer: Any,
    *,
    max_length: int = 6_144,
    max_prompt_length: int = 512,
) -> TokenizedSecPOPair:
    """Tokenize the four SecPO paths without rerendering or truncation."""

    if type(max_length) is not int or max_length < 2:
        raise ValueError("max_length must be an integer >= 2")
    if type(max_prompt_length) is not int or max_prompt_length < 1:
        raise ValueError("max_prompt_length must be a positive integer")

    attacked = _encode(tokenizer, example.prompt, label="prompt")
    clean = _encode(tokenizer, example.clean_prompt, label="clean_prompt")
    chosen = _encode(tokenizer, example.chosen, label="chosen")
    rejected = _encode(tokenizer, example.rejected, label="rejected")

    prompt_lengths = {"prompt": len(attacked), "clean_prompt": len(clean)}
    too_long_prompts = {
        name: length
        for name, length in prompt_lengths.items()
        if length > max_prompt_length
    }
    if too_long_prompts:
        raise ValueError(
            f"{_where(example.source_index)} exceeds max_prompt_length="
            f"{max_prompt_length}: {too_long_prompts}"
        )

    raw_sequences = {
        "policy_chosen": (attacked, chosen),
        "policy_rejected": (attacked, rejected),
        "reference_chosen": (clean, chosen),
        "reference_rejected": (clean, rejected),
    }
    lengths = {
        name: len(prompt_tokens) + len(response_tokens)
        for name, (prompt_tokens, response_tokens) in raw_sequences.items()
    }
    too_long_sequences = {
        name: length for name, length in lengths.items() if length > max_length
    }
    if too_long_sequences:
        raise ValueError(
            f"{_where(example.source_index)} exceeds max_length={max_length}: "
            f"{too_long_sequences}; pre-rendered SecPO sequences are never truncated"
        )

    shifted = {
        name: _shifted_sequence(prompt_tokens, response_tokens)
        for name, (prompt_tokens, response_tokens) in raw_sequences.items()
    }
    return TokenizedSecPOPair(
        policy_chosen=shifted["policy_chosen"],
        policy_rejected=shifted["policy_rejected"],
        reference_chosen=shifted["reference_chosen"],
        reference_rejected=shifted["reference_rejected"],
        source_index=example.source_index,
    )
