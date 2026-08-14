from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path


RELEASE_ROOT = Path(__file__).parents[1]
MANIFEST_PATH = RELEASE_ROOT / "release_manifest.json"
VALIDATOR_PATH = RELEASE_ROOT / "scripts" / "validate_release_manifest.py"

SPEC = importlib.util.spec_from_file_location("release_manifest_validator", VALIDATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def _minimal_manifest() -> dict:
    return {
        "schema_version": 1,
        "active_profile": "public_release",
        "policy": {
            "default_action": "deny",
            "ignored_vcs_globs": [".git", ".git/**"],
            "always_deny_globs": [
                "**/.env",
                "**/*.key",
                "**/logs/**",
                "**/*.safetensors",
            ],
            "always_deny_path_components": ["logs", "cache"],
            "loopback_endpoint_exceptions": [],
            "content_match_exceptions": [],
            "content_scan_extensions": [".md", ".py"],
            "maximum_file_bytes": 100_000,
            "large_file_exceptions": [],
        },
        "profiles": {
            "public_release": {
                "allow_globs": ["README.md", "src/secpo/*.py"],
                "required_paths": ["README.md"],
            },
            "maintainer_source_backup": {
                "extends": "public_release",
                "allow_globs": ["docs/*.md"],
                "required_paths": [],
            },
        },
        "components": [
            {"id": "core", "classification": "include", "scope": "core scope"}
        ],
    }


def test_committed_manifest_has_complete_decision_inventory() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert validator.validate_manifest_schema(manifest) == []
    by_id = {item["id"]: item for item in manifest["components"]}
    assert by_id["agentdojo"]["classification"] == "external"
    assert by_id["agentdojo"]["upstream_revision"] == (
        "d3640b5b03a88eb44dad96852e1de1ef437d836e"
    )
    assert manifest["profiles"]["maintainer_source_backup"]["extends"] == (
        "public_release"
    )
    assert manifest["profiles"]["public_release"]["allow_mode"] == "exact"
    public_profile = manifest["profiles"]["public_release"]
    assert "CITATION.bib" in public_profile["allow_globs"]
    assert "CITATION.bib" in public_profile["required_paths"]
    assert ".bib" in manifest["policy"]["content_scan_extensions"]
    assert not any(
        any(character in path for character in "*?[")
        for path in public_profile["allow_globs"]
    )
    denied = manifest["policy"]["always_deny_globs"]
    assert "**/.env" in denied
    assert "**/*.safetensors" in denied
    assert "**/logs/**" in denied
    assert "**/findclose_*" in denied
    assert "instruction_hierarchy_experiments" in by_id
    assert "agentdyn" in by_id
    assert "dtap_qwen_agent" in by_id
    assert by_id["dtap_qwen_agent"]["classification"] == "external"


def test_manifest_does_not_require_unused_classifications() -> None:
    manifest = _minimal_manifest()

    assert validator.validate_manifest_schema(manifest) == []


def test_exact_allowlist_rejects_wildcards() -> None:
    manifest = _minimal_manifest()
    manifest["profiles"]["public_release"]["allow_mode"] = "exact"

    errors = validator.validate_manifest_schema(manifest)

    assert any("exact allowlist contains wildcard" in error for error in errors)


def test_exact_allowlist_rejects_missing_listed_file(tmp_path: Path) -> None:
    manifest = _minimal_manifest()
    manifest["profiles"]["public_release"]["allow_mode"] = "exact"
    manifest["profiles"]["public_release"]["allow_globs"] = [
        "README.md",
        "missing.py",
    ]
    (tmp_path / "README.md").write_text("Public release\n", encoding="utf-8")

    errors = validator.validate_release_tree(tmp_path, manifest, "public_release")

    assert any("exactly allowlisted release file is missing" in error for error in errors)


def test_validator_is_read_only_and_accepts_allowlisted_tree(tmp_path: Path) -> None:
    manifest = _minimal_manifest()
    (tmp_path / "README.md").write_text("Public release\n", encoding="utf-8")
    source = tmp_path / "src" / "secpo"
    source.mkdir(parents=True)
    (source / "loss.py").write_text("BETA = 0.1\n", encoding="utf-8")
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    assert validator.validate_release_tree(
        tmp_path, manifest, "public_release"
    ) == []

    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_deny_rules_override_allow_rules(tmp_path: Path) -> None:
    manifest = _minimal_manifest()
    manifest["profiles"]["public_release"]["allow_globs"].append("logs/**")
    (tmp_path / "README.md").write_text("Public release\n", encoding="utf-8")
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "run.txt").write_text("generated state\n", encoding="utf-8")

    errors = validator.validate_release_tree(tmp_path, manifest, "public_release")

    assert any("denied path" in error and "logs/run.txt" in error for error in errors)


def test_double_star_deny_rule_also_applies_at_release_root(tmp_path: Path) -> None:
    manifest = _minimal_manifest()
    manifest["profiles"]["public_release"]["allow_globs"].append(".env")
    (tmp_path / "README.md").write_text("Public release\n", encoding="utf-8")
    (tmp_path / ".env").write_text("placeholder\n", encoding="utf-8")

    errors = validator.validate_release_tree(tmp_path, manifest, "public_release")

    assert any("denied path" in error and ".env" in error for error in errors)


def test_validator_rejects_unlisted_file_and_sensitive_content(tmp_path: Path) -> None:
    manifest = _minimal_manifest()
    (tmp_path / "README.md").write_text("Public release\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("not selected\n", encoding="utf-8")
    source = tmp_path / "src" / "secpo"
    source.mkdir(parents=True)
    sensitive_value = "/" + "home" + "/private-user/model"
    (source / "backend.py").write_text(
        f'MODEL_PATH = "{sensitive_value}"\n', encoding="utf-8"
    )

    errors = validator.validate_release_tree(tmp_path, manifest, "public_release")
    rendered = "\n".join(errors)

    assert "notes.md" in rendered and "not allowlisted" in rendered
    assert "machine-local absolute path" in rendered
    assert sensitive_value not in rendered


def test_validator_detects_nvidia_style_token_without_echoing_it(tmp_path: Path) -> None:
    manifest = _minimal_manifest()
    (tmp_path / "README.md").write_text("Public release\n", encoding="utf-8")
    source = tmp_path / "src" / "secpo"
    source.mkdir(parents=True)
    token = "LLM" + "|" + "12345678" + "|" + ("A" * 20)
    (source / "backend.py").write_text(
        f'DEFAULT_PROVIDER_VALUE = "{token}"\n', encoding="utf-8"
    )

    errors = validator.validate_release_tree(tmp_path, manifest, "public_release")
    rendered = "\n".join(errors)

    assert "credential-like token" in rendered
    assert token not in rendered


def test_loopback_exception_is_exact_and_keeps_private_network_scan(
    tmp_path: Path,
) -> None:
    manifest = _minimal_manifest()
    patch_path = "third_party/patches/dtap.patch"
    manifest["profiles"]["public_release"]["allow_globs"].append(patch_path)
    manifest["policy"]["content_scan_extensions"].append(".patch")
    manifest["policy"]["loopback_endpoint_exceptions"] = [patch_path]
    (tmp_path / "README.md").write_text("Public release\n", encoding="utf-8")
    target = tmp_path / patch_path
    target.parent.mkdir(parents=True)
    loopback = "http://127" + ".0.0.1:8123/service\n"
    target.write_text(loopback, encoding="utf-8")

    assert validator.validate_release_tree(
        tmp_path, manifest, "public_release"
    ) == []

    private_network = "http://10" + ".12.34.56:8123/service\n"
    target.write_text(private_network, encoding="utf-8")
    errors = validator.validate_release_tree(
        tmp_path, manifest, "public_release"
    )

    assert any("content rule matched (private endpoint)" in error for error in errors)


def test_loopback_exception_rejects_wildcard_and_missing_path(tmp_path: Path) -> None:
    manifest = _minimal_manifest()
    manifest["policy"]["loopback_endpoint_exceptions"] = ["third_party/*.patch"]

    errors = validator.validate_manifest_schema(manifest)

    assert any("must contain exact paths only" in error for error in errors)

    manifest["policy"]["loopback_endpoint_exceptions"] = ["missing.patch"]
    (tmp_path / "README.md").write_text("Public release\n", encoding="utf-8")
    errors = validator.validate_release_tree(
        tmp_path, manifest, "public_release"
    )

    assert any("loopback endpoint exception path is missing" in error for error in errors)


def test_content_match_exception_is_digest_pinned_and_must_be_used(
    tmp_path: Path,
) -> None:
    manifest = _minimal_manifest()
    patch_path = "third_party/patches/fixture.patch"
    manifest["profiles"]["public_release"]["allow_globs"].append(patch_path)
    manifest["policy"]["content_scan_extensions"].append(".patch")
    (tmp_path / "README.md").write_text("Public release\n", encoding="utf-8")
    target = tmp_path / patch_path
    target.parent.mkdir(parents=True)
    fixture_email = "benchmark-user" + "@example.test"
    target.write_text(fixture_email + "\n", encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(fixture_email.encode("utf-8")).hexdigest()
    manifest["policy"]["content_match_exceptions"] = [
        {
            "path": patch_path,
            "rule": "email address",
            "match_sha256": digest,
            "reason": "Synthetic benchmark fixture",
        }
    ]

    assert validator.validate_release_tree(
        tmp_path, manifest, "public_release"
    ) == []

    target.write_text("changed-" + fixture_email + "\n", encoding="utf-8")
    errors = validator.validate_release_tree(
        tmp_path, manifest, "public_release"
    )
    rendered = "\n".join(errors)
    assert "content rule matched (email address)" in rendered
    assert "content-match exception is stale or unused" in rendered


def test_profile_inheritance_and_cycle_detection() -> None:
    manifest = _minimal_manifest()

    allowed, required = validator.resolve_profile(
        manifest, "maintainer_source_backup"
    )

    assert "README.md" in allowed
    assert "docs/*.md" in allowed
    assert required == ["README.md"]

    cyclic = copy.deepcopy(manifest)
    cyclic["profiles"]["public_release"]["extends"] = (
        "maintainer_source_backup"
    )
    try:
        validator.resolve_profile(cyclic, "public_release")
    except validator.ManifestError as error:
        assert "cycle" in str(error)
    else:
        raise AssertionError("expected profile inheritance cycle to fail")
