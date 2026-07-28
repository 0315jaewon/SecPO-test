from __future__ import annotations

from typing import Any

import torch

from secpo.data import BenignReferenceCollator, validate_preference_record
from secpo.loss import IGNORE_INDEX


class EosIsPadTokenizer:
    """Small tokenizer double for the Qwen/Llama EOS-as-padding edge case."""

    pad_token_id = 0
    eos_token_id = 0

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        **_: Any,
    ) -> list[int]:
        assert not add_special_tokens
        # Text tokens are nonzero and every encoded fragment ends in EOS. Its
        # ID deliberately equals the padding ID, so token-id-based attention
        # masks would incorrectly hide real tokens.
        return [1 + (ord(character) % 47) for character in text] + [
            self.eos_token_id
        ]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        **kwargs: Any,
    ) -> dict[str, list[int]]:
        return {
            "input_ids": self.encode(
                text, add_special_tokens=add_special_tokens, **kwargs
            )
        }


def _examples() -> list[dict[str, str]]:
    return [
        validate_preference_record(
            {
                "id": "short",
                "prompt": "short INJECTION prompt",
                "clean_prompt": "short  prompt",
                "chosen": "ok",
                "rejected": "bad",
            }
        ).to_dict(),
        validate_preference_record(
            {
                "id": "long",
                "prompt": "a substantially longer INJECTION attacked prompt",
                "clean_prompt": "a substantially longer  attacked prompt",
                "chosen": "a longer preferred response",
                "rejected": "a much longer dispreferred response",
            }
        ).to_dict(),
    ]


def test_collator_masks_by_true_length_when_eos_is_padding() -> None:
    tokenizer = EosIsPadTokenizer()
    batch = BenignReferenceCollator(tokenizer, max_length=128)(_examples())

    prefixes = (
        "policy_chosen",
        "policy_rejected",
        "reference_chosen",
        "reference_rejected",
    )
    saw_real_eos_and_padded_eos = False

    for prefix in prefixes:
        input_ids = batch[f"{prefix}_input_ids"]
        attention_mask = batch[f"{prefix}_attention_mask"]
        labels = batch[f"{prefix}_labels"]

        assert input_ids.shape == attention_mask.shape == labels.shape
        assert attention_mask.dtype in (torch.bool, torch.int32, torch.int64)

        for row_ids, row_mask in zip(input_ids, attention_mask, strict=True):
            true_length = int(row_mask.sum().item())
            torch.testing.assert_close(
                row_mask,
                torch.tensor(
                    [1] * true_length + [0] * (row_mask.numel() - true_length),
                    dtype=row_mask.dtype,
                ),
            )
            # The final real token is EOS even though EOS shares its ID with
            # padding. It must remain attended.
            assert row_ids[true_length - 1].item() == tokenizer.eos_token_id
            assert row_mask[true_length - 1].item() == 1

            padded_eos = (row_ids == tokenizer.eos_token_id) & (row_mask == 0)
            if padded_eos.any():
                saw_real_eos_and_padded_eos = True

        # Padding labels must never contribute to response log-probabilities.
        assert torch.all(labels[attention_mask == 0] == IGNORE_INDEX)

    assert saw_real_eos_and_padded_eos


def test_policy_and_reference_batches_use_attacked_and_clean_prompts() -> None:
    tokenizer = EosIsPadTokenizer()
    example = _examples()[0]
    batch = BenignReferenceCollator(tokenizer, max_length=128)([example])

    policy_labels = batch["policy_chosen_labels"][0]
    reference_labels = batch["reference_chosen_labels"][0]
    policy_prompt_length = int((policy_labels == IGNORE_INDEX).sum().item())
    reference_prompt_length = int((reference_labels == IGNORE_INDEX).sum().item())

    # The attacked prompt in this fixture is longer than the clean prompt. If
    # the reference accidentally used the attacked prompt, these would match.
    assert policy_prompt_length > reference_prompt_length
