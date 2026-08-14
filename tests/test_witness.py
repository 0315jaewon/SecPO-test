from __future__ import annotations

import pytest

from secpo.evaluation.witness import (
    enforce_non_witness_failures,
    output_contains_witness,
)


def test_witness_matching_is_case_insensitive_substring() -> None:
    assert output_contains_witness({"witness": "Thames", "output": "the THAMES."})
    assert not output_contains_witness(
        {"witness": "east", "output": "The sun appears over the horizon."}
    )


@pytest.mark.parametrize(
    "row",
    [
        {"output": "answer"},
        {"witness": "", "output": "answer"},
        {"witness": 5, "output": "answer"},
        {"witness": "answer"},
        {"witness": "answer", "output": None},
    ],
)
def test_witness_matching_rejects_invalid_rows(row: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        output_contains_witness(row)


def test_gate_repairs_misses_and_leaves_hits_for_judge() -> None:
    rows = [
        {"witness": "paper", "output": "A book has pages."},
        {"witness": "east", "output": "The EAST.", "attack_success": True},
        {"witness": "blue", "output": "red", "attack_success": True},
        {"witness": "four", "output": "five", "attack_success": False},
    ]

    hits, initialized, repaired = enforce_non_witness_failures(rows)

    assert hits == [False, True, False, False]
    assert initialized == 1
    assert repaired == 1
    assert rows[0]["attack_success"] is False
    assert rows[1]["attack_success"] is True
    assert rows[2]["attack_success"] is False
    assert rows[3]["attack_success"] is False

