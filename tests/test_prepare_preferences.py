from __future__ import annotations

import gzip
import json
from pathlib import Path
import subprocess
import sys

import pytest

from secpo.data import PreferenceSchemaError, load_preference_dataset


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_preferences.py"


def _source_record() -> dict[str, str]:
    return {
        "id": "gzip-example",
        "prompt": "Summarize: REMOVE-ME trusted text.",
        "rejected_input_whole": "REMOVE-ME",
        "chosen": "Trusted text.",
        "rejected": "Injected answer.",
        "private_metadata": "must not survive",
    }


def _run_prepare(source: Path, destination: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(source),
            "--output",
            str(destination),
            "--expected-records",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_gzip_output_is_deterministic_anonymous_and_loadable(tmp_path) -> None:
    source = tmp_path / "private-source-name.json"
    source.write_text(json.dumps([_source_record()]), encoding="utf-8")
    first = tmp_path / "first-public-name.jsonl.gz"
    second = tmp_path / "different-public-name.jsonl.gz"

    _run_prepare(source, first)
    _run_prepare(source, second)

    first_bytes = first.read_bytes()
    assert first_bytes == second.read_bytes()
    assert first_bytes[:3] == b"\x1f\x8b\x08"
    assert first_bytes[3] & 0x08 == 0  # Gzip FNAME header flag is absent.
    assert first_bytes[4:8] == b"\x00\x00\x00\x00"

    with gzip.open(first, mode="rt", encoding="utf-8") as handle:
        public_records = [json.loads(line) for line in handle]
    assert public_records == [
        {
            "id": "gzip-example",
            "prompt": "Summarize: REMOVE-ME trusted text.",
            "clean_prompt": "Summarize:  trusted text.",
            "chosen": "Trusted text.",
            "rejected": "Injected answer.",
        }
    ]

    dataset = load_preference_dataset(first)
    assert [dict(record) for record in dataset] == public_records


def test_loader_reads_gzipped_json_array(tmp_path) -> None:
    source = tmp_path / "preferences.json.gz"
    public_record = {
        key: value
        for key, value in _source_record().items()
        if key != "private_metadata"
    }
    with gzip.open(source, mode="wt", encoding="utf-8") as handle:
        json.dump([public_record], handle)

    dataset = load_preference_dataset(source)

    assert dataset[0]["clean_prompt"] == "Summarize:  trusted text."


def test_loader_rejects_malformed_gzip_as_schema_error(tmp_path) -> None:
    source = tmp_path / "preferences.jsonl.gz"
    source.write_bytes(b"not a gzip stream")

    with pytest.raises(PreferenceSchemaError, match="could not read preference file"):
        load_preference_dataset(source)
