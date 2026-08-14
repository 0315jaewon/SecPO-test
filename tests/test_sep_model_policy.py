from __future__ import annotations

import json
from pathlib import Path

import pytest

from secpo.evaluation.sep_adaptive import (
    AdaptiveEvaluationConfig,
    AdaptiveExample,
    DryRunBackend,
    evaluate_example,
    main,
)
from secpo.evaluation.sep_model_policy import (
    LLAMA31_EOT_TOKEN_ID,
    model_family_policy,
    resolve_model_source,
)


RELEASE_ROOT = Path(__file__).parents[1]


class FakeTokenizer:
    eos_token_id = 17
    unk_token_id = 0

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 23, "<|eot_id|>": LLAMA31_EOT_TOKEN_ID}.get(
            token, self.unk_token_id
        )


def test_model_source_never_guesses_adapter_from_underscores():
    source = resolve_model_source("/models/merged_qwen_checkpoint_final")
    assert source.model_name_or_path == "/models/merged_qwen_checkpoint_final"
    assert source.adapter_name_or_path is None


def test_qwen_stop_ids_remove_llama_eot_and_use_tokenizer_end_ids():
    policy = model_family_policy("qwen36", enable_thinking=True)
    assert policy.stop_token_ids(
        FakeTokenizer(), requested=(5, LLAMA31_EOT_TOKEN_ID)
    ) == (5, 17, 23)
    assert policy.max_output_tokens == 32768


def test_llama_stop_ids_preserve_native_eot():
    policy = model_family_policy("llama31")
    assert policy.stop_token_ids(FakeTokenizer(), requested=(5,)) == (
        5,
        17,
        LLAMA31_EOT_TOKEN_ID,
    )


def test_qwen_input_role_and_visible_response_normalization():
    policy = model_family_policy("qwen36", enable_thinking=True)
    messages = policy.render_messages(
        instruction="Answer the task",
        input_text="document: {injection}",
        injection="candidate",
    )
    assert [message["role"] for message in messages] == ["user", "input"]
    assert messages[1]["content"] == "document: candidate"
    assert policy.normalize_response("<think>private</think>visible") == "visible"
    assert policy.normalize_response("<think>unfinished private") == ""


@pytest.mark.parametrize("algorithm", ["genetic", "pair", "tap"])
def test_every_attack_scores_only_qwen_visible_output(algorithm):
    seen = []

    class Backend(DryRunBackend):
        def victim(self, example, injection):
            return "<think>hidden witness</think>visible answer"

        def score(self, example, injection, response):
            seen.append(response)
            return float("hidden witness" in response)

    result = evaluate_example(
        AdaptiveExample("x", "task", "input", "goal", ("seed",)),
        Backend(),
        config=AdaptiveEvaluationConfig(
            algorithm=algorithm,
            steps=1,
            candidates_per_step=1,
            beam_width=1,
            branching_factor=1,
            num_islands=1,
            query_budget=2,
        ),
        model_policy=model_family_policy("qwen36", enable_thinking=True),
    )
    assert seen
    assert set(seen) == {"visible answer"}
    assert result.search.best.score == 0.0


def test_public_qwen_config_selects_paper_tap_and_nine_judges(tmp_path):
    output = tmp_path / "tap.json"
    config = RELEASE_ROOT / "configs" / "evaluate_sep_adaptive_qwen36_27b.json"
    assert main(
        [
            "--config",
            str(config),
            "--algorithm",
            "tap",
            "--steps",
            "1",
            "--branching-factor",
            "1",
            "--beam-width",
            "1",
            "--query-budget",
            "2",
            "--output",
            str(output),
        ]
    ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["model_policy"]["family"] == "qwen36"
    assert payload["model_policy"]["untrusted_role"] == "input"
    assert payload["judging"]["required_votes"] == 9


def test_checkpoint_fingerprint_rejects_model_family_change(tmp_path):
    checkpoint = tmp_path / "state.json"
    example = AdaptiveExample("x", "task", "input", "goal", ("seed",))
    config = AdaptiveEvaluationConfig(
        algorithm="genetic",
        steps=1,
        candidates_per_step=1,
        num_islands=1,
        query_budget=2,
    )
    evaluate_example(
        example,
        DryRunBackend(),
        config=config,
        checkpoint_path=checkpoint,
        model_policy=model_family_policy("llama31"),
    )
    with pytest.raises(ValueError, match="does not match"):
        evaluate_example(
            example,
            DryRunBackend(),
            config=config,
            checkpoint_path=checkpoint,
            resume=True,
            model_policy=model_family_policy("qwen36"),
        )
