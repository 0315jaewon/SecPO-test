#!/usr/bin/env python3
"""Verify and reconstruct the split, deterministic Qwen dataset archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, BinaryIO, NoReturn
import zlib


MANIFEST_FIELDS = frozenset(
    {
        "format_version",
        "dataset",
        "records",
        "raw_filename",
        "raw_size_bytes",
        "raw_sha256",
        "compression",
        "compressed_size_bytes",
        "compressed_sha256",
        "part_size_bytes",
        "parts",
    }
)
PART_FIELDS = frozenset({"file", "size_bytes", "sha256"})
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
PART_NAME = re.compile(r"dataset\.json\.gz\.part-[0-9]{2}\Z")


class ReconstructionError(ValueError):
    """Raised when the manifest, parts, or reconstructed bytes do not match."""


def fail(message: str) -> NoReturn:
    raise ReconstructionError(message)


def object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def positive_int(name: str, value: Any) -> int:
    if type(value) is not int or value < 1:
        fail(f"manifest {name!r} must be a positive integer")
    return value


def digest_value(name: str, value: Any) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        fail(f"manifest {name!r} must be a lowercase SHA-256 digest")
    return value


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        fail(f"manifest must be a regular, non-symlink file: {path}")
    try:
        manifest = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=object_without_duplicate_keys,
        )
    except json.JSONDecodeError as error:
        fail(
            f"invalid manifest JSON at line {error.lineno}, "
            f"column {error.colno}: {error.msg}"
        )
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_FIELDS:
        fail("manifest has missing or unknown top-level fields")
    if manifest["format_version"] != 1:
        fail("unsupported manifest format_version")
    if not isinstance(manifest["dataset"], str) or not manifest["dataset"]:
        fail("manifest dataset must be a non-empty string")
    if (
        not isinstance(manifest["raw_filename"], str)
        or not manifest["raw_filename"]
        or Path(manifest["raw_filename"]).name != manifest["raw_filename"]
    ):
        fail("manifest raw_filename must be a plain filename")
    if manifest["compression"] != "gzip --no-name --best":
        fail("manifest compression is not the supported deterministic gzip mode")

    for name in (
        "records",
        "raw_size_bytes",
        "compressed_size_bytes",
        "part_size_bytes",
    ):
        positive_int(name, manifest[name])
    digest_value("raw_sha256", manifest["raw_sha256"])
    digest_value("compressed_sha256", manifest["compressed_sha256"])

    parts = manifest["parts"]
    if not isinstance(parts, list) or not parts:
        fail("manifest parts must be a non-empty list")
    seen: set[str] = set()
    for index, part in enumerate(parts):
        if not isinstance(part, dict) or set(part) != PART_FIELDS:
            fail(f"manifest part {index} has missing or unknown fields")
        name = part["file"]
        if (
            not isinstance(name, str)
            or not PART_NAME.fullmatch(name)
            or name in seen
        ):
            fail(f"manifest part {index} has an invalid or duplicate filename")
        seen.add(name)
        positive_int(f"parts[{index}].size_bytes", part["size_bytes"])
        if part["size_bytes"] > manifest["part_size_bytes"]:
            fail(f"manifest part {index} exceeds part_size_bytes")
        digest_value(f"parts[{index}].sha256", part["sha256"])
    expected_entries = {path.name, *seen}
    actual_entries = {entry.name for entry in path.parent.iterdir()}
    if actual_entries != expected_entries:
        fail(
            "dataset-parts directory differs from manifest; "
            f"missing={sorted(expected_entries - actual_entries)}, "
            f"unexpected={sorted(actual_entries - expected_entries)}"
        )
    return manifest


def write_and_hash(
    output: BinaryIO | None,
    digest: Any,
    value: bytes,
) -> int:
    digest.update(value)
    if output is not None:
        output.write(value)
    return len(value)


def reconstruct(
    manifest_path: Path,
    manifest: dict[str, Any],
    output: BinaryIO | None,
) -> None:
    compressed_digest = hashlib.sha256()
    raw_digest = hashlib.sha256()
    decompressor = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
    compressed_size = 0
    raw_size = 0

    for index, part in enumerate(manifest["parts"]):
        part_path = manifest_path.parent / part["file"]
        if not part_path.is_file() or part_path.is_symlink():
            fail(f"missing regular dataset part: {part_path}")
        part_digest = hashlib.sha256()
        part_size = 0
        with part_path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                part_digest.update(chunk)
                compressed_digest.update(chunk)
                part_size += len(chunk)
                compressed_size += len(chunk)
                try:
                    decoded = decompressor.decompress(chunk)
                except zlib.error as error:
                    fail(f"invalid gzip stream in part {index}: {error}")
                raw_size += write_and_hash(output, raw_digest, decoded)
        if part_size != part["size_bytes"]:
            fail(f"part {index} size does not match manifest")
        if part_digest.hexdigest() != part["sha256"]:
            fail(f"part {index} SHA-256 does not match manifest")

    try:
        tail = decompressor.flush()
    except zlib.error as error:
        fail(f"invalid gzip stream trailer: {error}")
    raw_size += write_and_hash(output, raw_digest, tail)

    if not decompressor.eof or decompressor.unused_data:
        fail("split archive is truncated or has trailing compressed data")
    if compressed_size != manifest["compressed_size_bytes"]:
        fail("compressed size does not match manifest")
    if compressed_digest.hexdigest() != manifest["compressed_sha256"]:
        fail("compressed SHA-256 does not match manifest")
    if raw_size != manifest["raw_size_bytes"]:
        fail("reconstructed size does not match manifest")
    if raw_digest.hexdigest() != manifest["raw_sha256"]:
        fail("reconstructed SHA-256 does not match manifest")

    print(
        "Dataset parts verified: "
        f"parts={len(manifest['parts'])}, "
        f"compressed_bytes={compressed_size}, "
        f"raw_bytes={raw_size}, "
        f"raw_sha256={raw_digest.hexdigest()}."
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Verify or reconstruct the split Qwen preference dataset."
    )
    result.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/qwen36_offline_dataset/manifest.json"),
    )
    action = result.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path)
    action.add_argument("--verify-only", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    manifest_path = args.manifest
    manifest = load_manifest(manifest_path)
    if args.verify_only:
        reconstruct(manifest_path, manifest, None)
        return 0

    output_path = args.output
    assert output_path is not None
    if output_path.exists():
        fail(f"refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            reconstruct(manifest_path, manifest, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
    except Exception:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
        raise
    print(f"Wrote verified dataset to {output_path}.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReconstructionError, OSError) as error:
        raise SystemExit(f"error: {error}") from None
