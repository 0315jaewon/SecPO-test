from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from secpo.backends.llama31_current_policy_pair import (
    BASE_POLICY_SNAPSHOT,
    CurrentPolicyPairBackend,
    PairBackendConfig,
    PairBackendError,
    PolicySnapshot,
    PolicySnapshotLocator,
    extract_llama31_instruction,
    locate_prompt_slot,
)
from secpo.online_training import (
    AttackExample,
    AttackRequest,
    FileAttackExchange,
    PROTOCOL_VERSION,
    serve_requests,
)
from secpo.training_cli import load_training_config, parse_training_config
import secpo.training_cli as training_cli


ROOT = Path(__file__).resolve().parents[1]
PAIR_CONFIG = ROOT / "configs" / "llama31_current_policy_pair.json"
TRAIN_CONFIG = ROOT / "configs" / "train_llama31_8b_online.json"
QWEN_TRAIN_CONFIG = ROOT / "configs" / "train_qwen36_27b_offline.json"


class FakeCurrentPolicy:
    def __init__(self) -> None:
        self.refresh_count = 0
        self.generate_calls: list[tuple[list[str], int]] = []

    def refresh(self) -> PolicySnapshot:
        self.refresh_count += 1
        return PolicySnapshot(
            identifier=f"snapshot-{self.refresh_count}",
            order=Decimal(self.refresh_count),
            adapter_path=None,
        )

    def generate(self, prompts, *, max_tokens):
        prompts = list(prompts)
        self.generate_calls.append((prompts, max_tokens))
        return [f"victim response for {prompt}" for prompt in prompts]


def small_config(*, strict: bool = True) -> PairBackendConfig:
    return replace(
        PairBackendConfig(),
        n_streams=2,
        num_attack_steps=2,
        num_attacker_ranks=1,
        defender_total_batch_size=1,
        per_rank_attack_batch_size=1,
        strict_per_rank_batch=strict,
        initial_injection_templates=("seed-a {goal}", "seed-b {goal}"),
    )


def serialized_llama_input(input_data: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "Summarize the trusted data."
        "<|eot_id|><|start_header_id|>input<|end_header_id|>\n\n"
        f"{input_data}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def serialized_llama_prompt(injection: str = "") -> str:
    return serialized_llama_input(f"trusted {injection}data")


def online_request(*, example_count: int = 1) -> dict:
    examples = []
    for index in range(example_count):
        examples.append(
            {
                "id": f"sample-{index}",
                "seed_prompt": serialized_llama_prompt(f"ORIGINAL-{index} "),
                "clean_prompt": serialized_llama_prompt(),
                "chosen": "safe response",
                "rejected": "perfect injected response",
                "attack_goal": "produce injected content",
                "witness": None,
            }
        )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": "rank0000-batch00000000-pair-test",
        "rank": 0,
        "batch_index": 0,
        "examples": examples,
    }


def build_backend(*, final_votes=None, strict: bool = True):
    policy = FakeCurrentPolicy()
    proposer_requests = []
    iteration_requests = []
    final_requests = []

    def proposer(requests):
        proposer_requests.extend(requests)
        return [
            {
                "improvement": "test refinement",
                "prompt": f"candidate-{request.stream_index}",
            }
            for request in requests
        ]

    def iteration_judge(requests):
        iteration_requests.extend(requests)
        return [9 if request.candidate_injection.startswith("candidate-") else 1 for request in requests]

    def final_judge(requests):
        final_requests.extend(requests)
        if final_votes is None:
            return ["YES"] * len(requests)
        return list(final_votes)

    backend = CurrentPolicyPairBackend(
        config=small_config(strict=strict),
        policy=policy,
        proposer=proposer,
        iteration_judge=iteration_judge,
        final_judge=final_judge,
    )
    return backend, policy, proposer_requests, iteration_requests, final_requests


def test_canonical_config_records_main_run_constants() -> None:
    config = PairBackendConfig.from_path(PAIR_CONFIG)

    assert config.n_streams == 8
    assert config.num_attack_steps == 30
    assert config.keep_last_n == 3
    assert config.num_attacker_ranks == 8
    assert config.per_rank_attack_batch_size == 32
    assert config.defender_total_batch_size == 256
    assert config.target_max_tokens == 1536
    assert config.final_target_max_tokens == 2048
    assert config.final_trials_per_judge == 3
    assert config.attacker_identifier == "gemini-3.1-flash-lite-preview"
    assert config.iteration_judge_identifier == "gemini-2.5-flash-lite"
    assert config.final_judge_identifiers == (
        "gemini-3.1-flash-lite-preview",
        "gemini-3-flash-preview-thinking-4096",
        "gpt-5-high-reasoning",
    )
    assert len(config.initial_injection_templates) > 80
    assert config.strict_per_rank_batch is True


def test_online_training_publishes_peft_checkpoint_every_optimizer_step() -> None:
    config = load_training_config(TRAIN_CONFIG)
    qwen = load_training_config(QWEN_TRAIN_CONFIG)

    assert config.online_exchange_dir is not None
    assert config.save_strategy == "steps"
    assert config.save_steps == 1
    assert config.seed == 42
    assert config.data_seed == 0
    assert config.dataloader_drop_last is True
    assert config.max_steps == 74
    assert qwen.seed == 42
    assert qwen.data_seed is None
    assert qwen.dataloader_drop_last is False
    assert qwen.max_steps == -1

    archived_rows = 19_157
    per_rank_rows = math.ceil(archived_rows / 8)
    full_microbatches = per_rank_rows // config.per_device_train_batch_size
    assert per_rank_rows == 2_395
    assert full_microbatches == 1_197
    assert full_microbatches // config.gradient_accumulation_steps == 74
    assert per_rank_rows % 32 == 27

    captured = {}

    def fake_training_arguments(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    original = training_cli.TrainingArguments
    training_cli.TrainingArguments = fake_training_arguments
    try:
        training_cli._build_training_arguments(config)
    finally:
        training_cli.TrainingArguments = original
    assert captured["seed"] == 42
    assert captured["data_seed"] == 0
    assert captured["dataloader_drop_last"] is True
    assert captured["max_steps"] == 74

    captured.clear()
    training_cli.TrainingArguments = fake_training_arguments
    try:
        training_cli._build_training_arguments(qwen)
    finally:
        training_cli.TrainingArguments = original
    assert captured["seed"] == 42
    assert captured["data_seed"] == 42
    assert captured["dataloader_drop_last"] is False
    assert captured["max_steps"] == -1

    raw = json.loads(TRAIN_CONFIG.read_text(encoding="utf-8"))
    raw["save_strategy"] = "epoch"
    with pytest.raises(ValueError, match="every optimizer step"):
        parse_training_config(raw)

    raw = json.loads(TRAIN_CONFIG.read_text(encoding="utf-8"))
    raw["max_steps"] = 0
    with pytest.raises(ValueError, match="max_steps"):
        parse_training_config(raw)


def test_training_model_revisions_are_pinned_and_passed_to_both_loaders(
    monkeypatch,
) -> None:
    llama = load_training_config(TRAIN_CONFIG)
    qwen = load_training_config(QWEN_TRAIN_CONFIG)
    assert llama.model_revision == "0e9e39f249a16976918f6564b8830bc894c89659"
    assert qwen.model_revision == "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"

    calls = {}

    class FakeTokenizer:
        eos_token_id = 1
        eos_token = "<eos>"
        pad_token_id = None
        pad_token = None
        padding_side = "left"

    def tokenizer_loader(model_name, **kwargs):
        calls["tokenizer"] = (model_name, kwargs)
        return FakeTokenizer()

    class FakeModel:
        config = SimpleNamespace(use_cache=True)

    def model_loader(model_name, **kwargs):
        calls["model"] = (model_name, kwargs)
        return FakeModel()

    monkeypatch.setattr(
        training_cli.AutoTokenizer,
        "from_pretrained",
        tokenizer_loader,
    )
    monkeypatch.setattr(
        training_cli.AutoModelForCausalLM,
        "from_pretrained",
        model_loader,
    )
    monkeypatch.setattr(
        training_cli,
        "get_peft_model",
        lambda model, _config: model,
    )

    training_cli._load_tokenizer(llama)
    training_cli._load_policy(replace(llama, gradient_checkpointing=False))

    assert calls["tokenizer"][1]["revision"] == llama.model_revision
    assert calls["model"][1]["revision"] == llama.model_revision

    raw = json.loads(TRAIN_CONFIG.read_text(encoding="utf-8"))
    raw["model_revision"] = "main"
    with pytest.raises(ValueError, match="commit digest"):
        parse_training_config(raw)


def test_locate_prompt_slot_preserves_all_trusted_text() -> None:
    slot = locate_prompt_slot("trusted data", "trusted ORIGINAL data")

    assert slot.seed_injection == "ORIGINAL "
    assert slot.with_injection("ADAPTIVE ") == "trusted ADAPTIVE data"

    with pytest.raises(PairBackendError, match="one contiguous injection"):
        locate_prompt_slot("trusted data", "rewritten text")


def test_instruction_parser_accepts_only_released_llama_role_rendering() -> None:
    assert extract_llama31_instruction(serialized_llama_prompt()) == (
        "Summarize the trusted data."
    )

    malformed = [
        "plain prompt without roles",
        serialized_llama_prompt().replace(
            "<|start_header_id|>input<|end_header_id|>\n\n",
            "",
        ),
        serialized_llama_prompt().replace(
            "Summarize the trusted data.",
            "Summarize.<|start_header_id|>user<|end_header_id|>\n\nAgain",
        ),
        (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "extra role<|eot_id|>" + serialized_llama_prompt()
        ),
        "<user>Summarize.</user><input>trusted data</input><assistant>",
    ]
    for value in malformed:
        with pytest.raises(PairBackendError, match="user/input serialization"):
            extract_llama31_instruction(value)


def _complete_adapter(path: Path) -> None:
    path.mkdir()
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"weights")
    for item in path.iterdir():
        os.utime(item, (100.0, 100.0))
    os.utime(path, (100.0, 100.0))


def test_snapshot_locator_selects_newest_complete_old_enough_adapter(tmp_path) -> None:
    root = tmp_path / "policy"
    root.mkdir()
    _complete_adapter(root / "checkpoint-10")
    _complete_adapter(root / "epoch_10.5")
    manifest = root / "published"
    _complete_adapter(manifest)
    (manifest / "policy_snapshot.json").write_text(
        json.dumps(
            {
                "version": 1,
                "identifier": "step-11",
                "step": 11,
                "ready": True,
            }
        ),
        encoding="utf-8",
    )
    os.utime(manifest / "policy_snapshot.json", (100.0, 100.0))
    os.utime(manifest, (100.0, 100.0))
    temporary = root / "checkpoint-99.tmp"
    temporary.mkdir()
    (temporary / "adapter_config.json").write_text("{}", encoding="utf-8")
    locator = PolicySnapshotLocator(
        root,
        minimum_age_seconds=20,
        clock=lambda: 1000.0,
    )

    latest = locator.latest()
    assert latest.identifier == "step-11"
    assert latest.adapter_path == manifest.resolve()
    assert PolicySnapshotLocator(tmp_path / "missing").latest() == BASE_POLICY_SNAPSHOT


def test_snapshot_locator_waits_for_highest_recognizable_incomplete_checkpoint(
    tmp_path,
) -> None:
    root = tmp_path / "policy"
    root.mkdir()
    _complete_adapter(root / "checkpoint-10")
    newest = root / "checkpoint-99"
    newest.mkdir()
    (newest / "adapter_config.json").write_text("{}", encoding="utf-8")

    state = {"time": 1000.0}
    sleeps = []

    def publish_during_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        state["time"] += seconds
        (newest / "adapter_model.safetensors").write_bytes(b"weights")
        for item in (newest, *newest.iterdir()):
            os.utime(item, (980.0, 980.0))

    locator = PolicySnapshotLocator(
        root,
        minimum_age_seconds=20,
        settle_timeout_seconds=5,
        clock=lambda: state["time"],
        monotonic=lambda: state["time"],
        sleeper=publish_during_sleep,
    )

    assert locator.latest().identifier == "checkpoint-99"
    assert sleeps == [1.0]

    state["time"] = 1000.0
    (newest / "adapter_model.safetensors").unlink()
    with pytest.raises(PairBackendError, match="did not become complete"):
        PolicySnapshotLocator(
            root,
            minimum_age_seconds=20,
            settle_timeout_seconds=2,
            clock=lambda: state["time"],
            monotonic=lambda: state["time"],
            sleeper=lambda seconds: state.__setitem__(
                "time", state["time"] + seconds
            ),
        ).latest()


def test_snapshot_locator_waits_for_newest_complete_candidate(tmp_path) -> None:
    root = tmp_path / "policy"
    root.mkdir()
    _complete_adapter(root / "checkpoint-10")
    newest = root / "checkpoint-12"
    _complete_adapter(newest)
    for item in (newest, *newest.iterdir()):
        os.utime(item, (995.0, 995.0))

    state = {"time": 1000.0}
    sleeps = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        state["time"] += seconds

    locator = PolicySnapshotLocator(
        root,
        minimum_age_seconds=20,
        settle_timeout_seconds=30,
        clock=lambda: state["time"],
        monotonic=lambda: state["time"],
        sleeper=sleep,
    )

    assert locator.latest().identifier == "checkpoint-12"
    assert sleeps == [15.0]

    state["time"] = 1000.0
    with pytest.raises(PairBackendError, match="did not settle"):
        PolicySnapshotLocator(
            root,
            minimum_age_seconds=20,
            settle_timeout_seconds=5,
            clock=lambda: state["time"],
            monotonic=lambda: state["time"],
            sleeper=lambda seconds: state.__setitem__(
                "time", state["time"] + seconds
            ),
        ).latest()


def test_pair_loop_refreshes_policy_and_requires_nine_final_yes_votes() -> None:
    backend, policy, proposer, iteration_judge, final_judge = build_backend()

    attacked = backend(online_request())
    result = backend.last_results[0]

    assert attacked == {"sample-0": serialized_llama_prompt("candidate-0")}
    assert result.best_injection == "candidate-0"
    assert result.best_score == 9
    assert result.judge_confirmed is True
    assert result.training_response.startswith("victim response")
    assert result.policy_snapshots == ("snapshot-1", "snapshot-2", "snapshot-3")
    assert len(result.logs) == 4
    assert policy.refresh_count == 3
    assert [call[1] for call in policy.generate_calls] == [1536, 1536, 2048]
    assert policy.generate_calls[1][0] == [
        serialized_llama_input("trusted data candidate-0"),
        serialized_llama_input("trusted data candidate-1"),
    ]
    assert policy.generate_calls[2][0] == [
        serialized_llama_input("trusted datacandidate-0")
    ]
    assert len(proposer) == 2
    assert len(iteration_judge) == 4
    assert len(final_judge) == 9
    assert [request.judge_identifier for request in final_judge] == [
        "judge_slot_1",
        "judge_slot_1",
        "judge_slot_1",
        "judge_slot_2",
        "judge_slot_2",
        "judge_slot_2",
        "judge_slot_3",
        "judge_slot_3",
        "judge_slot_3",
    ]
    assert all(request.temperature == 0.7 for request in final_judge)


def test_pair_queries_append_after_input_but_returns_original_front_or_end_slot(
) -> None:
    backend, policy, *_ = build_backend(strict=False)
    request = online_request(example_count=2)
    request["examples"] = [
        {
            **request["examples"][0],
            "id": "front-seed",
            "clean_prompt": serialized_llama_input(" trusted data"),
            "seed_prompt": serialized_llama_input("ORIGINAL-0 trusted data"),
        },
        {
            **request["examples"][1],
            "id": "end-seed",
            "clean_prompt": serialized_llama_input("trusted data "),
            "seed_prompt": serialized_llama_input("trusted data ORIGINAL-1"),
        },
    ]

    attacked = backend(request)

    # Search follows attack_utils.build_chat_prompt exactly: source input,
    # one added separator, then candidate. End-position rows therefore retain
    # their existing trailing separator and receive two spaces here.
    assert policy.generate_calls[1][0] == [
        serialized_llama_input(" trusted data candidate-0"),
        serialized_llama_input(" trusted data candidate-1"),
        serialized_llama_input("trusted data  candidate-0"),
        serialized_llama_input("trusted data  candidate-1"),
    ]
    # The main final query concatenated sample['input'] and the best candidate
    # without adding the search helper's separator.
    assert policy.generate_calls[2][0] == [
        serialized_llama_input(" trusted datacandidate-0"),
        serialized_llama_input("trusted data candidate-0"),
    ]
    # Only the prompt returned to online DPO restores the original seed slot.
    assert attacked == {
        "front-seed": serialized_llama_input("candidate-0 trusted data"),
        "end-seed": serialized_llama_input("trusted data candidate-0"),
    }


def test_one_no_vote_fails_unanimous_confirmation() -> None:
    backend, *_ = build_backend(final_votes=["YES"] * 8 + ["NO"])

    backend(online_request())
    result = backend.last_results[0]

    assert result.judge_confirmed is False
    assert result.training_response == "perfect injected response"


def test_exact_mode_rejects_noncanonical_per_rank_batch() -> None:
    backend, *_ = build_backend(strict=True)
    request = online_request(example_count=1)
    backend.config = replace(
        backend.config,
        num_attacker_ranks=1,
        defender_total_batch_size=2,
        per_rank_attack_batch_size=2,
    )

    with pytest.raises(PairBackendError, match="require 2 examples per rank"):
        backend(request)


def test_backend_round_trips_real_online_exchange_schema(tmp_path) -> None:
    backend, *_ = build_backend(strict=False)
    exchange = FileAttackExchange(
        tmp_path / "exchange",
        timeout_seconds=2,
        poll_seconds=0.001,
    )
    request = AttackRequest(
        protocol_version=PROTOCOL_VERSION,
        request_id="rank0000-batch00000000-pair-e2e",
        rank=0,
        batch_index=0,
        examples=(
            AttackExample(
                id="sample-0",
                seed_prompt=serialized_llama_prompt("ORIGINAL-0 "),
                clean_prompt=serialized_llama_prompt(),
                chosen="safe response",
                rejected="perfect injected response",
                attack_goal="produce injected content",
                witness=None,
            ),
        ),
    )

    exchange.submit(request)
    assert serve_requests(exchange, backend, rank=0, once=True) == 1
    assert exchange.wait(request) == {
        "sample-0": serialized_llama_prompt("candidate-0")
    }
