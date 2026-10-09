"""Authenticated Hugging Face loading for Qwen3.8 SecPO preferences."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

from .data import (
    PreparedSecPOExample,
    prepare_pre_rendered_record,
    prepare_qwen38_record,
)


QWEN38_DATASET_NAME = "Sizhe-Chen/Qwen3.8-27B-Instruct-SecPO-trainset"
QWEN38_DATASET_SPLIT = "train"
QWEN38_EXPECTED_RECORDS = 18_951


def load_huggingface_split(
    dataset_name: str,
    *,
    split: str,
    revision: str | None,
    load_dataset_fn: Callable[..., Any] | None = None,
) -> Any:
    """Load one private-capable dataset split using existing HF credentials."""

    if load_dataset_fn is None:
        try:
            from datasets import load_dataset as load_dataset_fn
        except ImportError as error:  # pragma: no cover - installation error path
            raise RuntimeError(
                "loading a Hugging Face dataset requires the 'datasets' package"
            ) from error
    kwargs: dict[str, Any] = {"split": split, "token": True}
    if revision is not None:
        kwargs["revision"] = revision
    return load_dataset_fn(dataset_name, **kwargs)


def validate_dataset_size(dataset: Any, *, expected_records: int) -> None:
    """Fail before paid training when a dataset revision changes size."""

    try:
        actual = len(dataset)
    except TypeError as error:
        raise TypeError("the loaded dataset split must have a finite length") from error
    if actual != expected_records:
        raise ValueError(
            f"dataset contains {actual} records; expected exactly {expected_records}"
        )


def iter_prepared_qwen38_records(
    records: Iterable[Mapping[str, Any]],
    *,
    require_exact_fields: bool = True,
) -> Iterator[PreparedSecPOExample]:
    """Validate Qwen3.8 rows lazily while retaining stable source indices."""

    for index, record in enumerate(records):
        yield prepare_qwen38_record(
            record,
            source_index=index,
            require_exact_fields=require_exact_fields,
        )


def iter_prepared_records(
    records: Iterable[Mapping[str, Any]],
    *,
    dataset_format: str,
    require_exact_fields: bool = True,
) -> Iterator[PreparedSecPOExample]:
    """Dispatch rows to a named, explicit dataset contract."""

    for index, record in enumerate(records):
        yield prepare_record(
            record,
            dataset_format=dataset_format,
            source_index=index,
            require_exact_fields=require_exact_fields,
        )


def prepare_record(
    record: Mapping[str, Any],
    *,
    dataset_format: str,
    source_index: int | None = None,
    require_exact_fields: bool = True,
) -> PreparedSecPOExample:
    """Validate one row using the explicitly configured dataset contract."""

    preparers = {
        "qwen38_pre_rendered": prepare_qwen38_record,
        "secpo_pre_rendered": prepare_pre_rendered_record,
    }
    try:
        prepare = preparers[dataset_format]
    except KeyError as error:
        raise ValueError(
            f"unsupported SecPO dataset format: {dataset_format!r}"
        ) from error
    return prepare(
        record,
        source_index=source_index,
        require_exact_fields=require_exact_fields,
    )
