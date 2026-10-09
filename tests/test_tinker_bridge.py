from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from secpo.tinker.bridge import policy_datums, reference_inputs, sequence_to_datum
from secpo.tinker.data import ShiftedSequence, TokenizedSecPOPair


@dataclass
class FakeDatum:
    model_input: object
    loss_fn_inputs: dict[str, object]


class FakeModelInput:
    @staticmethod
    def from_ints(tokens: list[int]) -> tuple[int, ...]:
        return tuple(tokens)


FAKE_TINKER = SimpleNamespace(Datum=FakeDatum, ModelInput=FakeModelInput)


def _sequence(start: int) -> ShiftedSequence:
    return ShiftedSequence(
        input_tokens=(start, start + 1, start + 2),
        target_tokens=(start + 1, start + 2, start + 3),
        weights=(0.0, 1.0, 1.0),
        prompt_length=2,
        response_length=2,
    )


def _pair() -> TokenizedSecPOPair:
    return TokenizedSecPOPair(
        policy_chosen=_sequence(10),
        policy_rejected=_sequence(20),
        reference_chosen=_sequence(30),
        reference_rejected=_sequence(40),
        source_index=1,
    )


def test_sequence_to_datum_preserves_shifted_values() -> None:
    sequence = _sequence(10)
    datum = sequence_to_datum(sequence, tinker_module=FAKE_TINKER)

    assert datum.model_input == sequence.input_tokens
    assert datum.loss_fn_inputs == {
        "target_tokens": list(sequence.target_tokens),
        "weights": list(sequence.weights),
    }


def test_pair_conversion_keeps_chosen_rejected_order_and_clean_full_inputs() -> None:
    pair = _pair()

    chosen, rejected = policy_datums(pair, tinker_module=FAKE_TINKER)
    clean_chosen, clean_rejected = reference_inputs(
        pair, tinker_module=FAKE_TINKER
    )

    assert chosen.model_input == pair.policy_chosen.input_tokens
    assert rejected.model_input == pair.policy_rejected.input_tokens
    assert clean_chosen == pair.reference_chosen.full_tokens
    assert clean_rejected == pair.reference_rejected.full_tokens
