#!/usr/bin/env python3
"""Validate and minimize SecPO preference records before training.

The converter accepts a JSON array, a single JSON object, or JSON Lines.  It
writes only the documented public schema fields, verifies that attacked and clean
prompts differ by exactly one contiguous deletion, and rejects ambiguous
legacy injection removal.  Output is written atomically so a bad row cannot
leave a partially sanitized dataset behind.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, NoReturn


PUBLIC_FIELDS = (
    "id",
    "prompt",
    "clean_prompt",
    "chosen",
    "rejected",
    "attack_goal",
    "witness",
)
REQUIRED_FIELDS = ("prompt", "chosen", "rejected")
LEGACY_INJECTION_FIELD = "rejected_input_whole"
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


class PreferenceDataError(ValueError):
    """Raised when input cannot be sanitized without guessing."""


def _fail(message: str) -> NoReturn:
    raise PreferenceDataError(message)


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _decode_json(text: str, source: Path) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_object_without_duplicate_keys)
    except PreferenceDataError:
        raise
    except json.JSONDecodeError as error:
        _fail(
            f"{source}:{error.lineno}:{error.colno}: invalid JSON: {error.msg}"
        )


def _load_records(source: Path) -> list[dict[str, Any]]:
    try:
        text = source.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        _fail(f"{source}: input is not valid UTF-8: {error}")
    except OSError as error:
        _fail(f"{source}: could not read input: {error}")

    if not text.strip():
        _fail(f"{source}: input is empty")

    try:
        decoded = _decode_json(text, source)
    except PreferenceDataError as whole_file_error:
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                _fail(f"{source}:{line_number}: blank JSONL records are not allowed")
            try:
                record = _decode_json(line, source)
            except PreferenceDataError as line_error:
                _fail(
                    f"{source}:{line_number}: invalid JSONL record "
                    f"({line_error}); whole-file parse also failed "
                    f"({whole_file_error})"
                )
            if not isinstance(record, dict):
                _fail(f"{source}:{line_number}: each JSONL record must be an object")
            records.append(record)
        return records

    if isinstance(decoded, dict):
        return [decoded]
    if not isinstance(decoded, list):
        _fail(f"{source}: top-level JSON must be an object or an array of objects")
    for index, record in enumerate(decoded):
        if not isinstance(record, dict):
            _fail(f"{source}: record {index} is not an object")
    return decoded


def _validated_text(record: dict[str, Any], field: str, row: str) -> str:
    if field not in record:
        _fail(f"{row}: missing required field {field!r}")
    value = record[field]
    if not isinstance(value, str):
        _fail(f"{row}: field {field!r} must be a string")
    if not value.strip():
        _fail(f"{row}: field {field!r} must not be empty or whitespace-only")
    if "\x00" in value:
        _fail(f"{row}: field {field!r} contains a NUL byte")
    return value


def _single_deleted_span(attacked: str, clean: str, row: str) -> str:
    if attacked == clean:
        _fail(f"{row}: clean_prompt is identical to prompt")
    if len(clean) >= len(attacked):
        _fail(
            f"{row}: clean_prompt must be shorter than prompt by one removed injection"
        )

    prefix_length = 0
    shared_limit = min(len(attacked), len(clean))
    while (
        prefix_length < shared_limit
        and attacked[prefix_length] == clean[prefix_length]
    ):
        prefix_length += 1

    suffix_length = 0
    clean_remaining = len(clean) - prefix_length
    while (
        suffix_length < clean_remaining
        and attacked[len(attacked) - 1 - suffix_length]
        == clean[len(clean) - 1 - suffix_length]
    ):
        suffix_length += 1

    deleted_end = len(attacked) - suffix_length
    deleted = attacked[prefix_length:deleted_end]
    reconstructed = attacked[:prefix_length] + attacked[deleted_end:]
    if not deleted or reconstructed != clean:
        _fail(
            f"{row}: clean_prompt must equal prompt with one contiguous "
            "injection removed"
        )
    return deleted


def _clean_prompt(record: dict[str, Any], prompt: str, row: str) -> str:
    supplied_clean = record.get("clean_prompt")
    legacy_injection = record.get(LEGACY_INJECTION_FIELD)

    if supplied_clean is None and legacy_injection is None:
        _fail(
            f"{row}: provide 'clean_prompt' or {LEGACY_INJECTION_FIELD!r}"
        )
    if supplied_clean is not None and not isinstance(supplied_clean, str):
        _fail(f"{row}: field 'clean_prompt' must be a string")
    if legacy_injection is not None and not isinstance(legacy_injection, str):
        _fail(f"{row}: field {LEGACY_INJECTION_FIELD!r} must be a string")

    derived_clean: str | None = None
    if legacy_injection is not None:
        if not legacy_injection:
            _fail(f"{row}: legacy injection must not be empty")
        occurrences = prompt.count(legacy_injection)
        if occurrences != 1:
            _fail(
                f"{row}: legacy injection must occur exactly once in prompt; "
                f"found {occurrences}"
            )
        derived_clean = prompt.replace(legacy_injection, "", 1)

    if supplied_clean is None:
        clean = derived_clean
    else:
        clean = supplied_clean
        if derived_clean is not None and clean != derived_clean:
            _fail(
                f"{row}: clean_prompt disagrees with exact legacy injection removal"
            )

    assert clean is not None
    if not clean.strip():
        _fail(f"{row}: clean_prompt must not be empty or whitespace-only")
    if "\x00" in clean:
        _fail(f"{row}: clean_prompt contains a NUL byte")
    _single_deleted_span(prompt, clean, row)
    return clean


def _online_pair_id(prompt: str, chosen: str, rejected: str) -> str:
    hasher = hashlib.sha256(b"secpo-online-pair-id-v1\0")
    for value in (prompt, chosen, rejected):
        encoded = value.encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)
    digest = hasher.hexdigest()[:24]
    return f"pair-{digest}"


def _online_pair_goal(record: dict[str, Any], row: str) -> str:
    explicit = record.get("attack_goal")
    legacy = record.get("rejected_input")
    if explicit is None and legacy is None:
        _fail(
            f"{row}: online PAIR mode requires 'attack_goal' or legacy "
            "'rejected_input'"
        )
    values = []
    for field, value in (("attack_goal", explicit), ("rejected_input", legacy)):
        if value is None:
            continue
        values.append(_validated_text(record, field, row).strip())
    if len(values) == 2 and values[0] != values[1]:
        _fail(f"{row}: attack_goal disagrees with legacy rejected_input")
    return values[0]


def _sanitize_record(
    record: dict[str, Any],
    index: int,
    seen_ids: set[str],
    *,
    online_pair: bool,
) -> dict[str, str]:
    row = f"record {index}"
    prompt = _validated_text(record, "prompt", row)
    chosen = _validated_text(record, "chosen", row)
    rejected = _validated_text(record, "rejected", row)
    if chosen == rejected:
        _fail(f"{row}: chosen and rejected responses must differ")

    clean = _clean_prompt(record, prompt, row)
    sanitized: dict[str, str] = {
        "prompt": prompt,
        "clean_prompt": clean,
        "chosen": chosen,
        "rejected": rejected,
    }

    if "id" in record:
        record_id = record["id"]
        if not isinstance(record_id, str) or not SAFE_ID.fullmatch(record_id):
            _fail(
                f"{row}: id must be 1-128 characters from "
                "[A-Za-z0-9._:-], beginning with an alphanumeric character"
            )
        if record_id in seen_ids:
            _fail(f"{row}: duplicate id {record_id!r}")
        seen_ids.add(record_id)
        sanitized = {"id": record_id, **sanitized}
    elif online_pair:
        record_id = _online_pair_id(prompt, chosen, rejected)
        if record_id in seen_ids:
            _fail(
                f"{row}: duplicate content-derived online id {record_id!r}"
            )
        seen_ids.add(record_id)
        sanitized = {"id": record_id, **sanitized}

    for field in ("attack_goal", "witness"):
        if field not in record:
            continue
        sanitized[field] = _validated_text(record, field, row)
    if online_pair:
        sanitized["attack_goal"] = _online_pair_goal(record, row)

    return sanitized


def _sanitize(
    records: Iterable[dict[str, Any]],
    *,
    online_pair: bool = False,
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        result.append(
            _sanitize_record(
                record,
                index,
                seen_ids,
                online_pair=online_pair,
            )
        )
    if not result:
        _fail("input contains no preference records")
    return result


def _write_jsonl(records: Iterable[dict[str, str]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        if destination.suffix.lower() == ".gz":
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as raw_handle:
                temporary_name = raw_handle.name
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    fileobj=raw_handle,
                    mtime=0,
                ) as gzip_handle:
                    with io.TextIOWrapper(
                        gzip_handle,
                        encoding="utf-8",
                        newline="\n",
                    ) as handle:
                        for record in records:
                            handle.write(
                                json.dumps(
                                    record,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                )
                            )
                            handle.write("\n")
                raw_handle.flush()
                os.fsync(raw_handle.fileno())
        else:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as handle:
                temporary_name = handle.name
                for record in records:
                    handle.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    )
                    handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except OSError as error:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
        _fail(f"{destination}: could not write sanitized output: {error}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate SecPO preference JSON/JSONL and write only the public "
            "training schema."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--expected-records",
        type=int,
        help="Fail unless this many records survive validation.",
    )
    parser.add_argument(
        "--online-pair",
        action="store_true",
        help=(
            "Require/derive the online PAIR fields: map legacy rejected_input "
            "to attack_goal and derive a deterministic id when absent."
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.expected_records is not None and args.expected_records < 1:
        _fail("--expected-records must be positive")

    records = _sanitize(
        _load_records(args.input),
        online_pair=args.online_pair,
    )
    if (
        args.expected_records is not None
        and len(records) != args.expected_records
    ):
        _fail(
            f"expected {args.expected_records} records, validated {len(records)}"
        )
    _write_jsonl(records, args.output)
    print(f"Wrote {len(records)} validated records to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreferenceDataError as error:
        raise SystemExit(f"error: {error}") from None
