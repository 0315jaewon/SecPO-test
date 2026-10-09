from __future__ import annotations

import pytest

from secpo.tinker.data import QWEN38_NON_THINKING_SUFFIX
from secpo.tinker.dataset import (
    iter_prepared_records,
    iter_prepared_qwen38_records,
    load_huggingface_split,
    validate_dataset_size,
)


def _record() -> dict[str, object]:
    injection = "IGNORE."
    return {
        "prompt": f"user input {injection} data {QWEN38_NON_THINKING_SUFFIX}",
        "chosen_input": "task\n\ndata",
        "rejected_input": "bad task",
        "chosen_data": "data",
        "inject_at_the_end": False,
        "rejected_input_whole": injection,
        "chosen": "good<|im_end|>",
        "rejected": "bad<|im_end|>",
    }


def test_huggingface_loader_uses_existing_token_and_optional_revision() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_load(name: str, **kwargs: object) -> list[object]:
        calls.append((name, kwargs))
        return []

    load_huggingface_split(
        "owner/data",
        split="train",
        revision="b" * 40,
        load_dataset_fn=fake_load,
    )
    assert calls == [
        (
            "owner/data",
            {"split": "train", "token": True, "revision": "b" * 40},
        )
    ]


def test_dataset_size_fails_closed() -> None:
    validate_dataset_size([1, 2], expected_records=2)
    with pytest.raises(ValueError, match="contains 2.*expected exactly 3"):
        validate_dataset_size([1, 2], expected_records=3)


def test_prepared_iterator_retains_source_order() -> None:
    records = [_record(), _record()]
    records[1]["rejected_input_whole"] = "SECOND"
    records[1]["prompt"] = (
        f"user input SECOND data {QWEN38_NON_THINKING_SUFFIX}"
    )

    prepared = list(iter_prepared_qwen38_records(records))

    assert [row.source_index for row in prepared] == [0, 1]
    assert "IGNORE." not in prepared[0].clean_prompt
    assert "SECOND" not in prepared[1].clean_prompt


def test_prepared_iterator_dispatches_portable_contract() -> None:
    prepared = list(
        iter_prepared_records(
            [
                {
                    "prompt": "attacked-prefix",
                    "clean_prompt": "clean-prefix",
                    "chosen": "chosen-eos",
                    "rejected": "rejected-eos",
                }
            ],
            dataset_format="secpo_pre_rendered",
        )
    )

    assert prepared[0].prompt == "attacked-prefix"
    assert prepared[0].source_index == 0
