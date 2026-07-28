"""Strict preference-data validation and collation for SecPO."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import gzip
import json
from pathlib import Path
from typing import Any

import torch

from .loss import IGNORE_INDEX


_REQUIRED_FIELDS = frozenset({"prompt", "chosen", "rejected"})
_OPTIONAL_FIELDS = frozenset({"id", "clean_prompt", "rejected_input_whole"})
_ALLOWED_FIELDS = _REQUIRED_FIELDS | _OPTIONAL_FIELDS


class PreferenceSchemaError(ValueError):
    """Raised when a preference record violates the released data contract."""


@dataclass(frozen=True)
class PreferenceExample:
    """One normalized benign-reference preference pair."""

    prompt: str
    clean_prompt: str
    chosen: str
    rejected: str
    id: str | None = None

    def to_dict(self) -> dict[str, str]:
        """Return a JSON- and ``datasets``-compatible representation."""

        record = asdict(self)
        if record["id"] is None:
            del record["id"]
        return record


def _where(row_index: int | None) -> str:
    return "record" if row_index is None else f"record {row_index}"


def _require_nonempty_string(
    record: Mapping[str, Any],
    field: str,
    *,
    row_index: int | None,
) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        raise PreferenceSchemaError(
            f"{_where(row_index)} field {field!r} must be a string"
        )
    if not value.strip():
        raise PreferenceSchemaError(
            f"{_where(row_index)} field {field!r} must not be empty"
        )
    return value


def derive_clean_prompt(
    prompt: str,
    rejected_input_whole: str,
    *,
    row_index: int | None = None,
) -> str:
    """Remove exactly one injection occurrence from an attacked prompt.

    Failing when the occurrence is absent or ambiguous avoids silently
    constructing the wrong reference prompt.
    """

    if not isinstance(prompt, str) or not prompt.strip():
        raise PreferenceSchemaError(
            f"{_where(row_index)} field 'prompt' must be a non-empty string"
        )
    if not isinstance(rejected_input_whole, str) or not rejected_input_whole:
        raise PreferenceSchemaError(
            f"{_where(row_index)} field 'rejected_input_whole' must be a "
            "non-empty string"
        )
    occurrences = prompt.count(rejected_input_whole)
    if occurrences != 1:
        raise PreferenceSchemaError(
            f"{_where(row_index)} injection must occur exactly once in 'prompt'; "
            f"found {occurrences}"
        )
    clean_prompt = prompt.replace(rejected_input_whole, "", 1)
    if not clean_prompt.strip():
        raise PreferenceSchemaError(
            f"{_where(row_index)} injection removal produced an empty clean prompt"
        )
    return clean_prompt


def _validate_single_deletion(
    prompt: str,
    clean_prompt: str,
    *,
    row_index: int | None = None,
) -> None:
    """Require ``clean_prompt`` to remove one contiguous span from ``prompt``."""

    if len(clean_prompt) >= len(prompt):
        raise PreferenceSchemaError(
            f"{_where(row_index)} clean prompt must be shorter than the attacked "
            "prompt by one removed injection"
        )

    prefix_length = 0
    shared_limit = min(len(prompt), len(clean_prompt))
    while (
        prefix_length < shared_limit
        and prompt[prefix_length] == clean_prompt[prefix_length]
    ):
        prefix_length += 1

    suffix_length = 0
    clean_remaining = len(clean_prompt) - prefix_length
    while (
        suffix_length < clean_remaining
        and prompt[len(prompt) - 1 - suffix_length]
        == clean_prompt[len(clean_prompt) - 1 - suffix_length]
    ):
        suffix_length += 1

    deleted_end = len(prompt) - suffix_length
    deleted = prompt[prefix_length:deleted_end]
    reconstructed = prompt[:prefix_length] + prompt[deleted_end:]
    if not deleted or reconstructed != clean_prompt:
        raise PreferenceSchemaError(
            f"{_where(row_index)} clean prompt must equal the attacked prompt "
            "with one contiguous injection removed"
        )


def validate_preference_record(
    record: Mapping[str, Any],
    *,
    row_index: int | None = None,
) -> PreferenceExample:
    """Validate and normalize one input record.

    A record must provide ``prompt``, ``chosen``, and ``rejected``, plus either
    an explicit ``clean_prompt`` or the exact injected substring in
    ``rejected_input_whole``. If both clean representations are present, they
    must agree byte-for-byte. Unknown fields are rejected.
    """

    if not isinstance(record, Mapping):
        raise PreferenceSchemaError(f"{_where(row_index)} must be a JSON object")

    field_names = set(record)
    unknown = field_names - _ALLOWED_FIELDS
    if unknown:
        raise PreferenceSchemaError(
            f"{_where(row_index)} has unknown fields: {sorted(unknown)}"
        )
    missing = _REQUIRED_FIELDS - field_names
    if missing:
        raise PreferenceSchemaError(
            f"{_where(row_index)} is missing required fields: {sorted(missing)}"
        )

    prompt = _require_nonempty_string(record, "prompt", row_index=row_index)
    chosen = _require_nonempty_string(record, "chosen", row_index=row_index)
    rejected = _require_nonempty_string(record, "rejected", row_index=row_index)
    if chosen == rejected:
        raise PreferenceSchemaError(
            f"{_where(row_index)} chosen and rejected responses must differ"
        )

    clean_value = record.get("clean_prompt")
    injection_value = record.get("rejected_input_whole")
    has_clean = clean_value is not None
    has_injection = injection_value is not None
    if not has_clean and not has_injection:
        raise PreferenceSchemaError(
            f"{_where(row_index)} must contain 'clean_prompt' or "
            "'rejected_input_whole'"
        )

    derived_clean: str | None = None
    if has_injection:
        derived_clean = derive_clean_prompt(
            prompt,
            injection_value,
            row_index=row_index,
        )

    if has_clean:
        clean_prompt = _require_nonempty_string(
            record,
            "clean_prompt",
            row_index=row_index,
        )
        if derived_clean is not None and clean_prompt != derived_clean:
            raise PreferenceSchemaError(
                f"{_where(row_index)} explicit clean prompt does not match exact "
                "injection removal"
            )
    else:
        assert derived_clean is not None
        clean_prompt = derived_clean

    _validate_single_deletion(
        prompt,
        clean_prompt,
        row_index=row_index,
    )

    example_id = record.get("id")
    # ``datasets.Dataset`` represents a missing value as None when only some
    # source rows provide an id, so None is treated exactly like omission.
    if example_id is not None:
        if not isinstance(example_id, str) or not example_id.strip():
            raise PreferenceSchemaError(
                f"{_where(row_index)} field 'id' must be a non-empty string"
            )

    return PreferenceExample(
        id=example_id,
        prompt=prompt,
        clean_prompt=clean_prompt,
        chosen=chosen,
        rejected=rejected,
    )


def _read_json_records(path: Path) -> list[Any]:
    try:
        if path.name.lower().endswith((".jsonl.gz", ".json.gz")):
            with gzip.open(path, mode="rt", encoding="utf-8") as handle:
                text = handle.read()
        else:
            text = path.read_text(encoding="utf-8")
    except (EOFError, gzip.BadGzipFile, OSError, UnicodeDecodeError) as error:
        raise PreferenceSchemaError(
            f"could not read preference file {path}: {error}"
        ) from error

    if not text.strip():
        raise PreferenceSchemaError(f"preference file is empty: {path}")

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        records: list[Any] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise PreferenceSchemaError(
                    f"invalid JSON on line {line_number} of {path}: {error.msg}"
                ) from error
        return records

    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    raise PreferenceSchemaError(
        f"top-level JSON in {path} must be an object or an array of objects"
    )


def load_preference_dataset(path: str | Path) -> Any:
    """Load a validated JSON/JSONL file as a Hugging Face ``Dataset``."""

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"preference data file does not exist: {source}")

    raw_records = _read_json_records(source)
    if not raw_records:
        raise PreferenceSchemaError(f"preference file contains no records: {source}")

    normalized = [
        validate_preference_record(record, row_index=index).to_dict()
        for index, record in enumerate(raw_records)
    ]
    ids = [record.get("id") for record in normalized if record.get("id") is not None]
    if len(ids) != len(set(ids)):
        raise PreferenceSchemaError("preference record ids must be unique")

    try:
        from datasets import Dataset
    except ImportError as error:  # pragma: no cover - installation error path
        raise RuntimeError(
            "load_preference_dataset requires the 'datasets' package"
        ) from error
    return Dataset.from_list(normalized)


def build_input_and_labels(
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    max_length: int,
) -> tuple[list[int], list[int]]:
    """Join a prompt and response, left-truncating only the prompt.

    At least one prompt token is retained so that the first response token has
    causal context. Responses are never silently truncated.
    """

    if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 2:
        raise ValueError(f"max_length must be an integer >= 2, got {max_length!r}")
    prompt = list(prompt_ids)
    response = list(response_ids)
    if not prompt:
        raise ValueError("the tokenized prompt must not be empty")
    if not response:
        raise ValueError("the tokenized response must not be empty")
    if len(response) >= max_length:
        raise ValueError(
            f"response has {len(response)} tokens but max_length={max_length}; "
            "responses are not truncated"
        )

    prompt_budget = max_length - len(response)
    prompt = prompt[-prompt_budget:]
    input_ids = prompt + response
    labels = [IGNORE_INDEX] * len(prompt) + response
    return input_ids, labels


@dataclass
class BenignReferenceCollator:
    """Tokenize four SecPO forward paths and right-pad each independently."""

    tokenizer: Any
    max_length: int
    max_prompt_length: int | None = None
    append_eos: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_length, int)
            or isinstance(self.max_length, bool)
            or self.max_length < 2
        ):
            raise ValueError("max_length must be an integer >= 2")
        if self.max_prompt_length is not None and (
            not isinstance(self.max_prompt_length, int)
            or isinstance(self.max_prompt_length, bool)
            or self.max_prompt_length < 1
        ):
            raise ValueError("max_prompt_length must be a positive integer or None")
        self.pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if self.pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id must be configured")
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if self.append_eos and self.eos_token_id is None:
            raise ValueError(
                "tokenizer.eos_token_id must be configured when append_eos=True"
            )

    def _tokenize(self, text: str) -> list[int]:
        encoded = self.tokenizer(text, add_special_tokens=False)
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise TypeError("tokenizer output must be a mapping containing input_ids")
        input_ids = encoded["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        if (
            not isinstance(input_ids, Sequence)
            or isinstance(input_ids, (str, bytes))
            or any(not isinstance(token, int) for token in input_ids)
        ):
            raise TypeError("tokenizer input_ids must be a flat sequence of integers")
        if not input_ids:
            raise ValueError("tokenizer produced an empty token sequence")
        return list(input_ids)

    def _tokenize_response(self, text: str) -> list[int]:
        input_ids = self._tokenize(text)
        if self.append_eos and input_ids[-1] != self.eos_token_id:
            input_ids.append(self.eos_token_id)
        return input_ids

    @staticmethod
    def _pad(
        sequences: list[list[int]],
        *,
        pad_value: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_length = max(len(sequence) for sequence in sequences)
        values = torch.full(
            (len(sequences), max_length),
            fill_value=pad_value,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(values)
        for row, sequence in enumerate(sequences):
            length = len(sequence)
            values[row, :length] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[row, :length] = 1
        return values, attention_mask

    def __call__(
        self,
        features: list[Mapping[str, Any]],
    ) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("cannot collate an empty batch")

        blocks: dict[str, list[list[int]]] = {
            "policy_chosen": [],
            "policy_rejected": [],
            "reference_chosen": [],
            "reference_rejected": [],
        }
        label_blocks: dict[str, list[list[int]]] = {
            name: [] for name in blocks
        }

        for row_index, raw_feature in enumerate(features):
            example = validate_preference_record(raw_feature, row_index=row_index)
            attacked_prompt = self._tokenize(example.prompt)
            clean_prompt = self._tokenize(example.clean_prompt)
            chosen = self._tokenize_response(example.chosen)
            rejected = self._tokenize_response(example.rejected)

            if self.max_prompt_length is not None:
                prompt_lengths = {
                    "prompt": len(attacked_prompt),
                    "clean_prompt": len(clean_prompt),
                }
                too_long = {
                    name: length
                    for name, length in prompt_lengths.items()
                    if length > self.max_prompt_length
                }
                if too_long:
                    raise ValueError(
                        f"record {row_index} exceeds max_prompt_length="
                        f"{self.max_prompt_length}: {too_long}"
                    )

            pairs = {
                "policy_chosen": (attacked_prompt, chosen),
                "policy_rejected": (attacked_prompt, rejected),
                "reference_chosen": (clean_prompt, chosen),
                "reference_rejected": (clean_prompt, rejected),
            }
            for name, (prompt_ids, response_ids) in pairs.items():
                input_ids, labels = build_input_and_labels(
                    prompt_ids,
                    response_ids,
                    self.max_length,
                )
                blocks[name].append(input_ids)
                label_blocks[name].append(labels)

        batch: dict[str, torch.Tensor] = {}
        for name, sequences in blocks.items():
            input_ids, attention_mask = self._pad(
                sequences,
                pad_value=self.pad_token_id,
            )
            labels, _ = self._pad(
                label_blocks[name],
                pad_value=IGNORE_INDEX,
            )
            batch[f"{name}_input_ids"] = input_ids
            # This mask comes from pre-padding lengths. Comparing token ids
            # against pad_token_id is incorrect when pad_token_id == eos_token_id.
            batch[f"{name}_attention_mask"] = attention_mask
            batch[f"{name}_labels"] = labels
        return batch
