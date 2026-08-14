from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys


RELEASE_ROOT = Path(__file__).parents[1]
RECONSTRUCT = RELEASE_ROOT / "scripts" / "reconstruct_dataset.py"
AUDIT = RELEASE_ROOT / "scripts" / "audit_release_dataset.py"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _deterministic_gzip(value: bytes) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=output,
        compresslevel=9,
        mtime=0,
    ) as handle:
        handle.write(value)
    return output.getvalue()


def _write_split_package(directory: Path, raw: bytes) -> None:
    directory.mkdir()
    compressed = _deterministic_gzip(raw)
    split_at = max(1, len(compressed) // 2)
    values = (compressed[:split_at], compressed[split_at:])
    parts = []
    for index, value in enumerate(values):
        name = f"dataset.json.gz.part-{index:02d}"
        (directory / name).write_bytes(value)
        parts.append(
            {
                "file": name,
                "size_bytes": len(value),
                "sha256": _sha256(value),
            }
        )
    manifest = {
        "format_version": 1,
        "dataset": "test preferences",
        "records": 1,
        "raw_filename": "preferences.json",
        "raw_size_bytes": len(raw),
        "raw_sha256": _sha256(raw),
        "compression": "gzip --no-name --best",
        "compressed_size_bytes": len(compressed),
        "compressed_sha256": _sha256(compressed),
        "part_size_bytes": max(len(value) for value in values),
        "parts": parts,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def test_reconstructor_verifies_and_recovers_exact_bytes(tmp_path) -> None:
    raw = b'[{"prompt":"attacked","chosen":"yes","rejected":"no"}]\n'
    package = tmp_path / "package"
    _write_split_package(package, raw)
    output = tmp_path / "reconstructed.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(RECONSTRUCT),
            "--manifest",
            str(package / "manifest.json"),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output.read_bytes() == raw
    assert _sha256(raw) in completed.stdout

    subprocess.run(
        [
            sys.executable,
            str(RECONSTRUCT),
            "--manifest",
            str(package / "manifest.json"),
            "--verify-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_reconstructor_rejects_corrupt_part_without_output(tmp_path) -> None:
    package = tmp_path / "package"
    _write_split_package(package, b'[{"record":1}]\n')
    corrupt_part = package / "dataset.json.gz.part-00"
    value = bytearray(corrupt_part.read_bytes())
    value[0] ^= 0x01
    corrupt_part.write_bytes(value)
    output = tmp_path / "must-not-exist.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(RECONSTRUCT),
            "--manifest",
            str(package / "manifest.json"),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert not output.exists()


def test_raw_dataset_auditor_accepts_contract_and_rejects_secret(tmp_path) -> None:
    record = {
        "prompt": "trusted INJECTION data",
        "chosen": "trusted answer",
        "rejected": "injected answer",
        "rejected_input_whole": "INJECTION ",
        "chosen_input": "trusted",
        "rejected_input": "injected",
        "chosen_data": "data",
        "inject_at_the_end": False,
    }
    source = tmp_path / "raw.json"
    source.write_text(json.dumps([record]), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(AUDIT),
            str(source),
            "--schema",
            "raw",
            "--expected-records",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    record["chosen"] = "credential-like " + "hf_" + ("a" * 16)
    source.write_text(json.dumps([record]), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(AUDIT),
            str(source),
            "--schema",
            "raw",
            "--expected-records",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "credential-like material" in completed.stderr


def test_raw_dataset_auditor_rejects_extended_provider_credentials(tmp_path) -> None:
    record = {
        "prompt": "trusted INJECTION data",
        "chosen": "trusted answer",
        "rejected": "injected answer",
        "rejected_input_whole": "INJECTION ",
        "chosen_input": "trusted",
        "rejected_input": "injected",
        "chosen_data": "data",
        "inject_at_the_end": False,
    }
    tokens = (
        "nvapi" + "-" + ("a" * 20),
        "github" + "_pat_" + ("b" * 24),
        "AI" + "za" + ("c" * 35),
        "xox" + "b-" + ("d" * 16),
        "LLM" + "|" + "12345678" + "|" + ("e" * 20),
    )

    for index, token in enumerate(tokens):
        record["chosen"] = "credential-like " + token
        source = tmp_path / f"raw-{index}.json"
        source.write_text(json.dumps([record]), encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                str(AUDIT),
                str(source),
                "--schema",
                "raw",
                "--expected-records",
                "1",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode != 0
        assert "credential-like material" in completed.stderr
        assert token not in completed.stderr


def test_dataset_auditor_reads_deterministic_gzip_json_lines(tmp_path) -> None:
    records = [
        {
            "prompt": f"trusted INJECTION-{index} data",
            "clean_prompt": "trusted data",
            "chosen": f"trusted answer {index}",
            "rejected": f"injected answer {index}",
        }
        for index in range(2)
    ]
    encoded = b"".join(
        json.dumps(record, separators=(",", ":")).encode("utf-8") + b"\n"
        for record in records
    )
    source = tmp_path / "prepared.jsonl.gz"
    source.write_bytes(_deterministic_gzip(encoded))

    subprocess.run(
        [
            sys.executable,
            str(AUDIT),
            str(source),
            "--schema",
            "prepared",
            "--expected-records",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
