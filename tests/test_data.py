from __future__ import annotations

import json

import pytest

from secpo.data import PreferenceExample, load_preference_dataset, validate_preference_record


def _record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "id": "example-7",
        "prompt": "<user>Summarize.</user><input>INJECTION Trusted data.</input>",
        "clean_prompt": "<user>Summarize.</user><input> Trusted data.</input>",
        "chosen": "Trusted data.",
        "rejected": "Injected answer.",
    }
    record.update(overrides)
    return record


def test_validate_record_preserves_explicit_clean_prompt() -> None:
    example = validate_preference_record(_record(), row_index=3)

    assert example == PreferenceExample(
        id="example-7",
        prompt="<user>Summarize.</user><input>INJECTION Trusted data.</input>",
        clean_prompt="<user>Summarize.</user><input> Trusted data.</input>",
        chosen="Trusted data.",
        rejected="Injected answer.",
    )


def test_legacy_record_derives_clean_prompt_from_exactly_one_match() -> None:
    record = _record()
    record.pop("clean_prompt")
    record["rejected_input_whole"] = "INJECTION"

    example = validate_preference_record(record)

    assert example.clean_prompt == "<user>Summarize.</user><input> Trusted data.</input>"
    assert example.prompt.replace("INJECTION", "", 1) == example.clean_prompt


@pytest.mark.parametrize(
    ("prompt", "injection", "message"),
    [
        ("prefix trusted suffix", "not present", "exactly once"),
        ("prefix BAD middle BAD suffix", "BAD", "exactly once"),
        ("prefix trusted suffix", "", "non-empty"),
    ],
)
def test_legacy_clean_prompt_derivation_fails_closed(
    prompt: str, injection: str, message: str
) -> None:
    record = _record(
        prompt=prompt,
        clean_prompt=None,
        rejected_input_whole=injection,
    )

    with pytest.raises((TypeError, ValueError), match=message):
        validate_preference_record(record, row_index=11)


def test_explicit_clean_prompt_cannot_equal_attacked_prompt() -> None:
    prompt = "<user>Do the trusted task.</user><input>untrusted</input>"
    with pytest.raises(ValueError, match="shorter"):
        validate_preference_record(_record(prompt=prompt, clean_prompt=prompt))


def test_explicit_clean_prompt_must_be_one_contiguous_deletion() -> None:
    with pytest.raises(ValueError, match="contiguous injection"):
        validate_preference_record(
            _record(
                prompt="trusted PREFIX injection SUFFIX",
                clean_prompt="trusted rewritten content",
            )
        )


def test_chosen_and_rejected_must_differ() -> None:
    with pytest.raises(ValueError, match="must differ"):
        validate_preference_record(_record(chosen="same", rejected="same"))


def test_json_array_and_jsonl_normalize_to_the_same_examples(tmp_path) -> None:
    explicit = _record()
    legacy = _record(
        id="example-8",
        prompt="before REMOVE-ME after",
        clean_prompt=None,
        rejected_input_whole="REMOVE-ME",
    )
    records = [explicit, legacy]

    array_path = tmp_path / "preferences.json"
    array_path.write_text(json.dumps(records), encoding="utf-8")
    jsonl_path = tmp_path / "preferences.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    from_array = load_preference_dataset(array_path)
    from_jsonl = load_preference_dataset(jsonl_path)

    assert [dict(item) for item in from_array] == [dict(item) for item in from_jsonl]
    assert from_array[1]["clean_prompt"] == "before  after"


def test_json_loader_rejects_non_object_rows(tmp_path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps([_record(), "not-an-object"]), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=r"(row|record).*(1|2)"):
        load_preference_dataset(path)
