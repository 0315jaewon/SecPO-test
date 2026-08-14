from __future__ import annotations

import gzip
import json
from pathlib import Path
import subprocess
import sys

import pytest

from secpo.data import PreferenceSchemaError, load_preference_dataset
from secpo.online_training import OnlineAttackCollator


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


def _run_prepare(
    source: Path,
    destination: Path,
    *,
    online_pair: bool = False,
) -> None:
    command = [
        sys.executable,
        str(SCRIPT),
        "--input",
        str(source),
        "--output",
        str(destination),
        "--expected-records",
        "1",
    ]
    if online_pair:
        command.append("--online-pair")
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )


def test_gzip_output_is_deterministic_sanitized_and_loadable(tmp_path) -> None:
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


def test_prepare_preserves_only_documented_online_fields(tmp_path) -> None:
    record = _source_record()
    record["attack_goal"] = "Print the test marker."
    record["witness"] = "marker"
    source = tmp_path / "source.json"
    source.write_text(json.dumps([record]), encoding="utf-8")
    output = tmp_path / "prepared.jsonl"

    _run_prepare(source, output)
    prepared = json.loads(output.read_text(encoding="utf-8"))

    assert prepared["attack_goal"] == "Print the test marker."
    assert prepared["witness"] == "marker"
    assert "private_metadata" not in prepared


def test_online_pair_mode_derives_required_fields_from_raw_llama_record(
    tmp_path,
) -> None:
    record = _source_record()
    record.pop("id")
    record["rejected_input"] = "Print the injected answer. "
    source = tmp_path / "raw-llama.json"
    source.write_text(json.dumps([record]), encoding="utf-8")
    output = tmp_path / "online.jsonl"
    second = tmp_path / "online-second.jsonl"

    _run_prepare(source, output, online_pair=True)
    _run_prepare(source, second, online_pair=True)
    prepared = json.loads(output.read_text(encoding="utf-8"))

    assert output.read_bytes() == second.read_bytes()
    assert prepared["id"].startswith("pair-")
    assert len(prepared["id"]) == 29
    assert prepared["attack_goal"] == "Print the injected answer."
    assert "rejected_input" not in prepared

    captured = []

    class FakeExchange:
        def attack(self, request):
            captured.append(request)
            return {example.id: example.seed_prompt for example in request.examples}

    collator = OnlineAttackCollator(lambda rows: rows, FakeExchange())
    rows = collator([prepared])
    assert rows[0]["id"] == prepared["id"]
    assert captured[0].examples[0].attack_goal == "Print the injected answer."


def test_online_pair_mode_rejects_disagreeing_goal_fields(tmp_path) -> None:
    record = _source_record()
    record["attack_goal"] = "one goal"
    record["rejected_input"] = "different goal"
    source = tmp_path / "bad-goal.json"
    source.write_text(json.dumps([record]), encoding="utf-8")
    output = tmp_path / "must-not-exist.jsonl"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(source),
            "--output",
            str(output),
            "--online-pair",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "disagrees" in completed.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    ("records", "message"),
    [
        (
            [
                {
                    **_source_record(),
                    "id": "first",
                    "rejected_input": "   ",
                }
            ],
            "must not be empty",
        ),
        (
            [
                {
                    key: value
                    for key, value in _source_record().items()
                    if key != "id"
                },
                {
                    key: value
                    for key, value in _source_record().items()
                    if key != "id"
                },
            ],
            "duplicate content-derived online id",
        ),
    ],
)
def test_online_pair_mode_rejects_invalid_goal_and_id_collision(
    tmp_path,
    records,
    message,
) -> None:
    for record in records:
        record.setdefault("rejected_input", "injected goal")
    source = tmp_path / "invalid-online.json"
    source.write_text(json.dumps(records), encoding="utf-8")
    output = tmp_path / "must-not-exist.jsonl"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(source),
            "--output",
            str(output),
            "--online-pair",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert message in completed.stderr
    assert not output.exists()


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
