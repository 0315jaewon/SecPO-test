from __future__ import annotations

import json
from concurrent.futures import Future
from pathlib import Path

import pytest
import secpo.evaluation.dtap_qwen36 as dtap

from secpo.evaluation.dtap_qwen36 import (
    AGGREGATE_PARALLELISM,
    DEFAULT_TASK_LIST_PATH,
    DTAP_COMMIT,
    DTAP_LICENSE_SHA256,
    DTAP_PATCH_SHA256,
    DTAP_TEMPLATE_SHA256,
    EXPECTED_TASK_FILE_SHA256,
    EXPECTED_TASK_SEMANTIC_SHA256,
    EXPECTED_TASKS,
    MAX_CONTEXT_TOKENS,
    MAX_OUTPUT_TOKENS,
    QWEN_AGENT_LICENSE_SHA256,
    QWEN_AGENT_VERSION,
    RELEASE_TRAINING_CONFIG_LEARNING_RATE,
    SHARD_COUNT,
    config_fingerprint,
    config_from_mapping,
    file_sha256,
    launch_all_shards,
    load_config,
    load_tasks,
    prepare_shards,
    run_shard,
    selection_semantic_sha256,
    separate_qwen_messages,
    summarize_results,
    validate_shard_plan,
)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "evaluate_dtap_qwen36_27b.json"
TASK_LIST = ROOT / DEFAULT_TASK_LIST_PATH
CALLS: list[str] = []


@pytest.fixture(autouse=True)
def _node_local_runtime_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SECPODTAP_RUNTIME_ROOT", str(tmp_path / "dtap-runtime"))


def successful_backend(request):
    CALLS.append(request["task_fingerprint"])
    assert request["protocol"]["benchmark"]["roles"] == {
        "trusted": "user",
        "untrusted": "input",
    }
    assert request["protocol"]["generation"]["enable_thinking"] is False
    return {"task_success": True, "attack_success": False}


def invalid_score_backend(request):
    del request
    return {"task_success": 1, "attack_success": False}


def _tasks() -> list[dict[str, str]]:
    domains = (
        "browser",
        "code",
        "crm",
        "customer-service",
        "finance",
        "legal",
        "medical",
        "os-filesystem",
        "research",
        "telecom",
        "travel",
        "workflow",
    )
    return [
        {
            "domain": domains[index % len(domains)],
            "type": "malicious",
            "task_id": str(index + 1),
            "threat_model": "indirect",
            "risk_category": f"risk-{index % 7}",
        }
        for index in range(EXPECTED_TASKS)
    ]


def _write_tasks(path: Path, rows=None) -> None:
    selected = _tasks() if rows is None else rows
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )


def _allow_synthetic_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests small without weakening the production entry point."""

    def load_synthetic(path, *, expected_count=EXPECTED_TASKS, require_authoritative=True):
        del require_authoritative
        return dtap._parse_task_bytes(Path(path).read_bytes(), expected_count=expected_count)

    monkeypatch.setattr(dtap, "load_tasks", load_synthetic)


def test_release_configuration_pins_the_exact_protocol() -> None:
    config = load_config(CONFIG)
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert raw["status"] == "runnable_external_dtap_qwen_agent_backend"
    assert config.task_list == DEFAULT_TASK_LIST_PATH
    assert config.expected_tasks == 1_838
    assert config.task_file_sha256 == EXPECTED_TASK_FILE_SHA256
    assert config.task_semantic_sha256 == EXPECTED_TASK_SEMANTIC_SHA256
    assert config.shard_count == 33
    assert config.max_parallel_per_shard == 6
    assert config.aggregate_parallelism == AGGREGATE_PARALLELISM == 198
    assert config.max_context_tokens == MAX_CONTEXT_TOKENS == 262_144
    assert config.max_output_tokens == MAX_OUTPUT_TOKENS == 32_768
    assert config.enable_thinking is False
    assert config.evaluation_checkpoint_learning_rate == 1.2e-4
    assert (
        config.release_training_config_learning_rate
        == RELEASE_TRAINING_CONFIG_LEARNING_RATE
        == 1.6e-4
    )
    assert len(config_fingerprint(config)) == 64


def test_release_configuration_rejects_type_coercion() -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["benchmark"]["expected_tasks"] = "1838"
    with pytest.raises(ValueError, match="integer fields"):
        config_from_mapping(raw)

    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["generation"]["enable_thinking"] = 0
    with pytest.raises(ValueError, match="JSON boolean"):
        config_from_mapping(raw)

    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["dtap"]["vendored"] = 0
    with pytest.raises(ValueError, match="external dependency"):
        config_from_mapping(raw)

    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["benchmark"]["port_namespace"]["default_offset_stride"] = 382
    with pytest.raises(ValueError, match="port-namespace"):
        config_from_mapping(raw)


def test_all_33_shards_have_disjoint_port_namespaces() -> None:
    configured_owners: dict[int, int] = {}
    fallback_ports: set[int] = set()
    for shard_id in range(SHARD_COUNT):
        namespace = dtap.shard_port_namespace(shard_id)
        shifted = {
            port + namespace["default_offset"]
            for port in dtap.PINNED_CONFIGURED_DEFAULT_PORTS
        }
        assert shifted.isdisjoint(configured_owners)
        configured_owners.update({port: shard_id for port in shifted})

        fallback = set(
            range(namespace["fallback_start"], namespace["fallback_end"] + 1)
        )
        assert fallback.isdisjoint(fallback_ports)
        assert fallback.isdisjoint(shifted)
        fallback_ports.update(fallback)

    assert min(configured_owners) == 1025
    assert max(configured_owners) == 29693
    assert min(fallback_ports) == 30000
    assert max(fallback_ports) == 62999
    assert len(configured_owners) == 87 * SHARD_COUNT
    assert len(fallback_ports) == 1000 * SHARD_COUNT


def test_worker_activates_namespace_before_loading_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, dict[str, str]] = {}

    def snapshot() -> dict[str, str]:
        names = (
            "SECPODTAP_SHARD_ID",
            "SECPODTAP_SHARD_RUNTIME_ROOT",
            "DT_PORT_DEFAULT_OFFSET",
            "DT_PORT_RANGE",
            "DT_PORT_LOCK_DIR",
        )
        return {name: dtap.os.environ[name] for name in names}

    def fake_load_backend(specification):
        assert specification == "release:backend"
        observed["load"] = snapshot()
        return successful_backend

    def fake_run_shard(*args, **kwargs):
        del args
        observed["run"] = snapshot()
        assert kwargs["shard_id"] == 32
        assert kwargs["backend"] is successful_backend
        return {"status": "complete", "shard_id": 32}

    monkeypatch.setattr(dtap, "load_backend", fake_load_backend)
    monkeypatch.setattr(dtap, "run_shard", fake_run_shard)
    report = dtap._run_shard_process(
        "shards",
        "results",
        32,
        load_config(CONFIG),
        "release:backend",
    )
    assert report == {"status": "complete", "shard_id": 32}
    assert observed["load"] == observed["run"]
    assert observed["run"]["SECPODTAP_SHARD_ID"] == "32"
    assert observed["run"]["DT_PORT_DEFAULT_OFFSET"] == "12192"
    assert observed["run"]["DT_PORT_RANGE"] == "62000-62999"


def test_portable_launcher_submits_exactly_33_process_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(CONFIG)
    task_list = tmp_path / "tasks.jsonl"
    shards_dir = tmp_path / "shards"
    results_dir = tmp_path / "results"
    _write_tasks(task_list)
    _allow_synthetic_selection(monkeypatch)
    prepare_shards(task_list, shards_dir, config=config)
    observed: dict[str, object] = {"shards": []}

    class RecordingPool:
        def __init__(self, *, max_workers):
            observed["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, *args):
            del function
            shard_id = args[2]
            observed["shards"].append(shard_id)
            future = Future()
            future.set_result({"status": "complete", "shard_id": shard_id})
            return future

    monkeypatch.setattr(dtap, "ProcessPoolExecutor", RecordingPool)
    report = launch_all_shards(
        shards_dir,
        results_dir,
        config=config,
        backend_specification=f"{__name__}:successful_backend",
    )
    assert observed["max_workers"] == 33
    assert observed["shards"] == list(range(33))
    assert report["status"] == "complete"
    assert report["aggregate_parallelism"] == 198


def test_external_artifact_identities_are_frozen() -> None:
    assert DTAP_COMMIT == "e0323a521ba4ef88f8e14c1eccf68d0a3d19a458"
    assert DTAP_LICENSE_SHA256.startswith("c71d239d")
    assert DTAP_PATCH_SHA256.startswith("c63e22b0")
    assert DTAP_TEMPLATE_SHA256.startswith("5fe53e61")
    assert QWEN_AGENT_VERSION == "0.0.34"
    assert QWEN_AGENT_LICENSE_SHA256.startswith("cfc7749b")


def test_trust_provenance_maps_to_user_and_input() -> None:
    converted = separate_qwen_messages(
        [
            {"role": "system", "content": "policy", "trust": "trusted"},
            {"role": "user", "content": "task", "trust": "trusted"},
            {"role": "tool", "content": "injected result", "trust": "untrusted"},
            {"role": "assistant", "content": "done"},
        ]
    )
    assert [row["role"] for row in converted] == [
        "user",
        "user",
        "input",
        "assistant",
    ]
    with pytest.raises(ValueError, match="explicit trust"):
        separate_qwen_messages([{"role": "tool", "content": "unmarked"}])


def test_canonical_sort_places_threat_and_risk_before_task_id() -> None:
    common = {
        "domain": "workflow",
        "type": "malicious",
        "threat_model": "indirect",
    }
    task_id_first_under_old_order = {
        **common,
        "risk_category": "z-risk",
        "task_id": "001",
    }
    risk_first_under_audited_order = {
        **common,
        "risk_category": "a-risk",
        "task_id": "999",
    }
    observed = sorted(
        [task_id_first_under_old_order, risk_first_under_audited_order],
        key=dtap._task_sort_key,
    )
    assert observed == [risk_first_under_audited_order, task_id_first_under_old_order]


def test_arbitrary_1838_task_substitute_is_rejected(tmp_path: Path) -> None:
    task_list = tmp_path / "substitute.jsonl"
    _write_tasks(task_list)
    with pytest.raises(ValueError, match="file SHA-256 differs"):
        load_tasks(task_list)


def test_bundled_selection_topology_and_fingerprints(tmp_path: Path) -> None:
    task_list = TASK_LIST
    config = load_config(CONFIG)
    assert task_list.stat().st_size == 225_328
    assert file_sha256(task_list) == EXPECTED_TASK_FILE_SHA256
    tasks = load_tasks(task_list)
    assert selection_semantic_sha256(tasks) == EXPECTED_TASK_SEMANTIC_SHA256
    manifest = prepare_shards(task_list, tmp_path / "real-shards", config=config)
    assert manifest["task_file_sha256"] == EXPECTED_TASK_FILE_SHA256
    assert manifest["task_semantic_sha256"] == EXPECTED_TASK_SEMANTIC_SHA256
    assert manifest["shards"][0] == {
        "shard_id": 0,
        "task_file": "shard_00.tasks.jsonl",
        "task_count": 56,
        "task_file_sha256": "3680ae62cf0af86a4f2a84181ab77607a126914fe0a7e768f025086ae23f7757",
        "task_semantic_sha256": "a1dad07e72659b54240163dfe59388102fbdddbbb4e625483eae9bfa571cc5cc",
    }
    _, shards = validate_shard_plan(tmp_path / "real-shards", config=config)
    assert shards[0][0] == {
        "domain": "browser",
        "type": "malicious",
        "task_id": "1",
        "threat_model": "indirect",
        "risk_category": "browser-integrity",
    }
    assert shards[0][-1]["task_id"] == "005"
    assert shards[0][-1]["risk_category"] == "sensitive-file-deletion"


def test_prepare_shards_is_exact_deterministic_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(CONFIG)
    task_list = tmp_path / "tasks.jsonl"
    shards_dir = tmp_path / "shards"
    _write_tasks(task_list)
    _allow_synthetic_selection(monkeypatch)
    first = prepare_shards(task_list, shards_dir, config=config)
    second = prepare_shards(task_list, shards_dir, config=config)
    assert first == second
    assert first["task_count"] == EXPECTED_TASKS
    assert len(first["shards"]) == SHARD_COUNT
    assert sorted(row["task_count"] for row in first["shards"]) == [55] * 10 + [56] * 23
    _, shards = validate_shard_plan(shards_dir, config=config)
    assert len({tuple(task.values()) for shard in shards for task in shard}) == EXPECTED_TASKS


def test_prepare_shards_rejects_vm_domain_and_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(CONFIG)
    rows = _tasks()
    rows[0] = {**rows[0], "domain": "windows"}
    task_list = tmp_path / "tasks.jsonl"
    _write_tasks(task_list, rows)
    with pytest.raises(ValueError, match="excluded VM domain"):
        prepare_shards(task_list, tmp_path / "shards", config=config)

    _write_tasks(task_list)
    _allow_synthetic_selection(monkeypatch)
    shards_dir = tmp_path / "clean-shards"
    prepare_shards(task_list, shards_dir, config=config)
    with (shards_dir / "shard_00.tasks.jsonl").open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(RuntimeError, match="file checksum differs"):
        validate_shard_plan(shards_dir, config=config)


def test_run_shard_resumes_only_exact_completed_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(CONFIG)
    task_list = tmp_path / "tasks.jsonl"
    shards_dir = tmp_path / "shards"
    results_dir = tmp_path / "results"
    _write_tasks(task_list)
    _allow_synthetic_selection(monkeypatch)
    prepare_shards(task_list, shards_dir, config=config)
    CALLS.clear()
    first = run_shard(
        shards_dir,
        results_dir,
        shard_id=0,
        config=config,
        backend=successful_backend,
    )
    called = len(CALLS)
    assert first["status"] == "complete"
    assert called == 56
    second = run_shard(
        shards_dir,
        results_dir,
        shard_id=0,
        config=config,
        backend=successful_backend,
    )
    assert second["complete"] == 56
    assert len(CALLS) == called

    result = results_dir / "shard_00" / "task_0000.json"
    value = json.loads(result.read_text(encoding="utf-8"))
    value["task_success"] = False
    result.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="fingerprint differs"):
        run_shard(
            shards_dir,
            results_dir,
            shard_id=0,
            config=config,
            backend=successful_backend,
        )


def test_invalid_scores_are_errors_and_final_rates_stay_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(CONFIG)
    task_list = tmp_path / "tasks.jsonl"
    shards_dir = tmp_path / "shards"
    results_dir = tmp_path / "results"
    _write_tasks(task_list)
    _allow_synthetic_selection(monkeypatch)
    prepare_shards(task_list, shards_dir, config=config)
    report = run_shard(
        shards_dir,
        results_dir,
        shard_id=0,
        config=config,
        backend=invalid_score_backend,
        fail_on_error=False,
    )
    assert report["status"] == "incomplete"
    assert report["errors"] == 56
    result = json.loads(
        (results_dir / "shard_00" / "task_0000.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "error"
    assert result["task_success"] is None
    assert result["attack_success"] is None
    assert "message" not in result

    summary = summarize_results(shards_dir, results_dir, config=config)
    assert summary["status"] == "incomplete"
    assert summary["error_tasks"] == 56
    assert summary["missing_tasks"] == EXPECTED_TASKS - 56
    assert summary["utility_success_rate"] is None
    assert summary["attack_success_rate"] is None
