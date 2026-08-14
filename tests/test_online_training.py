from __future__ import annotations

import stat
from threading import Thread

import pytest

from secpo.online_training import (
    AttackExample,
    AttackRequest,
    FileAttackExchange,
    OnlineAttackCollator,
    OnlineAttackError,
    PROTOCOL_VERSION,
    serve_requests,
    validate_attacked_prompt,
)


def _request() -> AttackRequest:
    return AttackRequest(
        protocol_version=PROTOCOL_VERSION,
        request_id="rank0000-batch00000000-test",
        rank=0,
        batch_index=0,
        examples=(
            AttackExample(
                id="sample-1",
                seed_prompt="trusted SEED data",
                clean_prompt="trusted data",
                chosen="trusted answer",
                rejected="injected answer",
                attack_goal="produce the injected answer",
                witness="marker",
            ),
        ),
    )


def test_exchange_round_trip_is_atomic_and_validated(tmp_path) -> None:
    exchange = FileAttackExchange(
        tmp_path / "exchange",
        timeout_seconds=2,
        poll_seconds=0.005,
    )

    def backend(request):
        return {request["examples"][0]["id"]: "trusted ADAPTIVE data"}

    worker = Thread(
        target=serve_requests,
        args=(exchange, backend),
        kwargs={"once": True},
        daemon=True,
    )
    worker.start()
    attacked = exchange.attack(_request())
    worker.join(timeout=2)

    assert attacked == {"sample-1": "trusted ADAPTIVE data"}
    assert not worker.is_alive()
    assert not list(exchange.requests.iterdir())
    assert not list(exchange.responses.iterdir())


def test_exchange_uses_private_directory_and_file_permissions(tmp_path) -> None:
    exchange = FileAttackExchange(tmp_path / "exchange", retain_files=True)

    request_path = exchange.submit(_request())

    for directory in (
        exchange.root,
        exchange.requests,
        exchange.responses,
        exchange.failures,
    ):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(request_path.stat().st_mode) == 0o600


def test_collator_changes_only_policy_prompt(tmp_path) -> None:
    exchange = FileAttackExchange(
        tmp_path / "exchange",
        timeout_seconds=2,
        poll_seconds=0.005,
    )

    def backend(request):
        return {request["examples"][0]["id"]: "trusted NEW data"}

    worker = Thread(
        target=serve_requests,
        args=(exchange, backend),
        kwargs={"once": True},
        daemon=True,
    )
    worker.start()
    collator = OnlineAttackCollator(lambda rows: rows, exchange)
    output = collator(
        [
            {
                "id": "sample-1",
                "prompt": "trusted OLD data",
                "clean_prompt": "trusted data",
                "chosen": "trusted answer",
                "rejected": "injected answer",
            }
        ]
    )
    worker.join(timeout=2)

    assert output[0]["prompt"] == "trusted NEW data"
    assert output[0]["clean_prompt"] == "trusted data"
    assert output[0]["chosen"] == "trusted answer"
    assert output[0]["rejected"] == "injected answer"


def test_attacked_prompt_cannot_rewrite_trusted_content() -> None:
    with pytest.raises(OnlineAttackError, match="preserve the clean prompt"):
        validate_attacked_prompt(
            "rewritten content",
            "trusted data",
            example_id="sample-1",
        )


def test_worker_failure_does_not_persist_backend_exception_text(tmp_path) -> None:
    exchange = FileAttackExchange(
        tmp_path / "exchange",
        timeout_seconds=2,
        poll_seconds=0.005,
        retain_files=True,
    )

    def backend(request):
        raise RuntimeError("sensitive provider diagnostics")

    worker = Thread(
        target=serve_requests,
        args=(exchange, backend),
        kwargs={"once": True},
        daemon=True,
    )
    worker.start()
    with pytest.raises(OnlineAttackError, match="RuntimeError"):
        exchange.attack(_request())
    worker.join(timeout=2)

    failure = next(exchange.failures.glob("*.json")).read_text(encoding="utf-8")
    assert "sensitive provider diagnostics" not in failure


def test_online_collator_requires_stable_ids(tmp_path) -> None:
    exchange = FileAttackExchange(tmp_path / "exchange")
    collator = OnlineAttackCollator(lambda rows: rows, exchange)

    with pytest.raises(OnlineAttackError, match="stable id"):
        collator(
            [
                {
                    "prompt": "trusted ATTACK data",
                    "clean_prompt": "trusted data",
                    "chosen": "trusted answer",
                    "rejected": "injected answer",
                }
            ]
        )
