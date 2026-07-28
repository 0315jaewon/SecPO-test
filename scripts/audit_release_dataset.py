#!/usr/bin/env python3
"""Stream-validate the public SecPO preference artifact without model imports."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any, NoReturn, TextIO


PREPARED_REQUIRED_FIELDS = frozenset(
    {"prompt", "clean_prompt", "chosen", "rejected"}
)
PREPARED_ALLOWED_FIELDS = PREPARED_REQUIRED_FIELDS | {"id"}
RAW_REQUIRED_FIELDS = frozenset(
    {
        "prompt",
        "chosen",
        "rejected",
        "rejected_input_whole",
        "chosen_input",
        "rejected_input",
        "chosen_data",
        "inject_at_the_end",
    }
)
RAW_ALLOWED_FIELDS = RAW_REQUIRED_FIELDS
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
EMAIL_LIKE = re.compile(
    r"(?<![A-Za-z0-9._%+-])"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r"(?![A-Za-z0-9.-])"
)
LOCAL_PATH_LIKE = re.compile(
    r"/(?:home|Users|lustre|checkpoint|scratch|gpfs|nfs|fsx)/",
    re.IGNORECASE,
)
SECRET_LIKE = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    r"(?:hf|ghp|glpat)_[A-Za-z0-9_-]{16,}"
    r"|sk[-_][A-Za-z0-9_-]{16,}"
    r"|AKIA[0-9A-Z]{16}"
    r")"
)
PRIVATE_KEY = re.compile(
    r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"
)


class DatasetAuditError(ValueError):
    """Raised when the released dataset fails a publication invariant."""


def fail(message: str) -> NoReturn:
    raise DatasetAuditError(message)


def object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def open_text(path: Path) -> TextIO:
    if path.name.endswith(".gz"):
        with path.open("rb") as raw:
            header = raw.read(10)
        if len(header) != 10 or header[:2] != b"\x1f\x8b":
            fail(f"{path} does not have a valid gzip header")
        if int.from_bytes(header[4:8], "little") != 0:
            fail(f"{path} gzip header stores a nonzero timestamp")
        if header[3] & 0x08:
            fail(f"{path} gzip header stores an original filename")
        return gzip.open(path, mode="rt", encoding="utf-8", newline="")
    return path.open(mode="rt", encoding="utf-8", newline="")


def validate_single_deletion(
    attacked: str,
    clean: str,
    *,
    line_number: int,
) -> None:
    if len(clean) >= len(attacked):
        fail(
            f"line {line_number}: clean_prompt must be shorter than prompt"
        )

    prefix_length = 0
    while (
        prefix_length < len(clean)
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
        fail(
            f"line {line_number}: clean_prompt is not one contiguous "
            "deletion from prompt"
        )


def require_text(record: dict[str, Any], field: str, line_number: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        fail(f"line {line_number}: {field!r} must be a non-empty string")
    if "\x00" in value:
        fail(f"line {line_number}: {field!r} contains a NUL byte")
    return value


def iter_records(path: Path) -> Any:
    """Yield ``(ordinal, record)`` from a JSON array/object or JSON Lines."""

    with open_text(path) as handle:
        first_nonspace = ""
        while True:
            character = handle.read(1)
            if not character or not character.isspace():
                first_nonspace = character
                break
        handle.seek(0)

        if first_nonspace == "[":
            try:
                decoded = json.load(
                    handle,
                    object_pairs_hook=object_without_duplicate_keys,
                )
            except json.JSONDecodeError as error:
                fail(
                    f"invalid JSON at line {error.lineno}, column "
                    f"{error.colno}: {error.msg}"
                )
            if not isinstance(decoded, list):
                fail("top-level JSON beginning with '[' must be an array")
            for ordinal, record in enumerate(decoded, start=1):
                yield ordinal, record
            return

        if first_nonspace == "{":
            try:
                decoded = json.load(
                    handle,
                    object_pairs_hook=object_without_duplicate_keys,
                )
            except json.JSONDecodeError as error:
                # Strict JSON Lines begins with the same byte as a single
                # object. "Extra data" is the unambiguous signal that another
                # record follows; all other whole-object errors remain errors.
                if error.msg != "Extra data":
                    fail(
                        f"invalid JSON at line {error.lineno}, column "
                        f"{error.colno}: {error.msg}"
                    )
                handle.seek(0)
            else:
                if not isinstance(decoded, dict):
                    fail("top-level JSON object did not decode to a mapping")
                yield 1, decoded
                return

        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                fail(f"line {line_number}: blank JSONL record")
            try:
                record = json.loads(
                    line,
                    object_pairs_hook=object_without_duplicate_keys,
                )
            except json.JSONDecodeError as error:
                fail(
                    f"line {line_number}: invalid JSON at column "
                    f"{error.colno}: {error.msg}"
                )
            yield line_number, record


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Audit a raw or prepared SecPO JSON/JSONL dataset."
    )
    result.add_argument("dataset", type=Path)
    result.add_argument("--expected-records", type=int)
    result.add_argument(
        "--schema",
        choices=("auto", "raw", "prepared"),
        default="auto",
        help="Expected record schema; default detects it from public fields.",
    )
    result.add_argument(
        "--deny-term",
        action="append",
        default=[],
        help=(
            "Case-insensitive literal identity term that must not occur. "
            "Repeat for multiple author, username, or institution terms."
        ),
    )
    return result


def main() -> int:
    args = parser().parse_args()
    path = args.dataset
    if not path.is_file():
        fail(f"dataset does not exist: {path}")
    if args.expected_records is not None and args.expected_records < 1:
        fail("--expected-records must be positive")

    deny_terms = [term.casefold() for term in args.deny_term if term]
    if len(deny_terms) != len(args.deny_term):
        fail("--deny-term values must not be empty")

    record_count = 0
    ids: set[str] = set()
    row_hashes: Counter[str] = Counter()
    email_matches = 0
    local_path_matches = 0
    observed_schema: str | None = None

    for line_number, record in iter_records(path):
        if not isinstance(record, dict):
            fail(f"record {line_number}: record must be a JSON object")

        fields = set(record)
        if args.schema == "auto":
            if "clean_prompt" in fields and "rejected_input_whole" not in fields:
                record_schema = "prepared"
            elif "rejected_input_whole" in fields and "clean_prompt" not in fields:
                record_schema = "raw"
            else:
                fail(
                    f"record {line_number}: cannot determine raw/prepared schema"
                )
        else:
            record_schema = args.schema
        if observed_schema is None:
            observed_schema = record_schema
        elif observed_schema != record_schema:
            fail(f"record {line_number}: mixed dataset schemas are not allowed")

        if record_schema == "prepared":
            required_fields = PREPARED_REQUIRED_FIELDS
            allowed_fields = PREPARED_ALLOWED_FIELDS
        else:
            required_fields = RAW_REQUIRED_FIELDS
            allowed_fields = RAW_ALLOWED_FIELDS
        missing = required_fields - fields
        unknown = fields - allowed_fields
        if missing or unknown:
            fail(
                f"record {line_number}: schema mismatch; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )

        prompt = require_text(record, "prompt", line_number)
        chosen = require_text(record, "chosen", line_number)
        rejected = require_text(record, "rejected", line_number)
        if chosen == rejected:
            fail(f"record {line_number}: chosen and rejected must differ")

        if record_schema == "prepared":
            clean_prompt = require_text(record, "clean_prompt", line_number)
            validate_single_deletion(
                prompt,
                clean_prompt,
                line_number=line_number,
            )
            if "id" in record:
                record_id = require_text(record, "id", line_number)
                if not SAFE_ID.fullmatch(record_id):
                    fail(f"record {line_number}: invalid public record id")
                if record_id in ids:
                    fail(f"record {line_number}: duplicate record id")
                ids.add(record_id)
        else:
            injection = require_text(
                record,
                "rejected_input_whole",
                line_number,
            )
            occurrences = prompt.count(injection)
            if occurrences != 1:
                fail(
                    f"record {line_number}: injection must occur exactly once "
                    f"in prompt; found {occurrences}"
                )
            clean_prompt = prompt.replace(injection, "", 1)
            if not clean_prompt.strip():
                fail(f"record {line_number}: empty derived clean prompt")
            validate_single_deletion(
                prompt,
                clean_prompt,
                line_number=line_number,
            )
            require_text(record, "chosen_input", line_number)
            require_text(record, "rejected_input", line_number)
            require_text(record, "chosen_data", line_number)
            if type(record["inject_at_the_end"]) is not bool:
                fail(
                    f"record {line_number}: 'inject_at_the_end' must be boolean"
                )

        for field, value in record.items():
            if isinstance(value, str):
                if SECRET_LIKE.search(value) or PRIVATE_KEY.search(value):
                    fail(
                        f"record {line_number}: credential-like material in "
                        f"{field!r}"
                    )
                folded = value.casefold()
                for term in deny_terms:
                    if term in folded:
                        fail(
                            f"record {line_number}: denied identity term in "
                            f"{field!r}"
                        )
                email_matches += len(EMAIL_LIKE.findall(value))
                local_path_matches += len(LOCAL_PATH_LIKE.findall(value))

        canonical = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        row_hashes[hashlib.sha256(canonical).hexdigest()] += 1
        record_count += 1

    if record_count == 0:
        fail("dataset contains no records")
    if (
        args.expected_records is not None
        and record_count != args.expected_records
    ):
        fail(
            f"expected {args.expected_records} records, found {record_count}"
        )

    duplicate_rows = sum(count - 1 for count in row_hashes.values() if count > 1)
    print(
        "Dataset audit passed: "
        f"schema={observed_schema}, "
        f"records={record_count}, "
        f"duplicate_rows={duplicate_rows}, "
        f"email_like_matches={email_matches}, "
        f"local_path_like_matches={local_path_matches}."
    )
    if email_matches or local_path_matches:
        print(
            "Note: email/path-like strings are retained experimental text, "
            "not release metadata; review the dataset card before sharing."
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DatasetAuditError, OSError, UnicodeError) as error:
        raise SystemExit(f"error: {error}") from None
