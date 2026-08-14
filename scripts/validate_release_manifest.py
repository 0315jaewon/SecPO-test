#!/usr/bin/env python3
"""Validate the release tree against its deny-first manifest.

This tool only reads files. It never copies, deletes, archives, or modifies the
release tree. Deny rules take precedence over every profile allow rule.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable


CLASSIFICATIONS = {"include", "refactor", "external", "exclude", "missing"}
ALLOW_MODES = {"glob", "exact"}
TEXT_READ_LIMIT = 4_000_000


class ManifestError(ValueError):
    """Raised when the manifest itself is malformed."""


def _matches(path: str, patterns: Iterable[str]) -> bool:
    return any(
        fnmatch.fnmatchcase(path, pattern)
        or (
            pattern.startswith("**/")
            and fnmatch.fnmatchcase(path, pattern.removeprefix("**/"))
        )
        for pattern in patterns
    )


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read JSON manifest: {error}") from error
    if not isinstance(value, dict):
        raise ManifestError("manifest root must be an object")
    return value


def validate_manifest_schema(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != 1:
        errors.append("schema_version must be 1")

    policy = manifest.get("policy")
    if not isinstance(policy, dict):
        errors.append("policy must be an object")
    elif policy.get("default_action") != "deny":
        errors.append("policy.default_action must be 'deny'")
    else:
        loopback_exceptions = policy.get("loopback_endpoint_exceptions", [])
        if not isinstance(loopback_exceptions, list) or not all(
            isinstance(item, str) and item for item in loopback_exceptions
        ):
            errors.append(
                "policy.loopback_endpoint_exceptions must be a list of strings"
            )
        elif any(
            any(character in path for character in "*?[")
            for path in loopback_exceptions
        ):
            errors.append(
                "policy.loopback_endpoint_exceptions must contain exact paths only"
            )
        content_exceptions = policy.get("content_match_exceptions", [])
        if not isinstance(content_exceptions, list):
            errors.append("policy.content_match_exceptions must be a list")
        else:
            expected_keys = {"path", "rule", "match_sha256", "reason"}
            allowed_rules = {label for label, _ in _content_rules()}
            seen_exceptions: set[tuple[str, str, str]] = set()
            for index, exception in enumerate(content_exceptions):
                where = f"policy.content_match_exceptions[{index}]"
                if not isinstance(exception, dict):
                    errors.append(f"{where} must be an object")
                    continue
                if set(exception) != expected_keys:
                    errors.append(f"{where} must have exactly {sorted(expected_keys)}")
                    continue
                path = exception["path"]
                rule = exception["rule"]
                digest = exception["match_sha256"]
                reason = exception["reason"]
                if not isinstance(path, str) or not path:
                    errors.append(f"{where}.path must be a non-empty string")
                elif any(character in path for character in "*?["):
                    errors.append(f"{where}.path must be exact")
                if rule not in allowed_rules:
                    errors.append(f"{where}.rule must name a content rule")
                if not isinstance(digest, str) or re.fullmatch(
                    r"[0-9a-f]{64}", digest
                ) is None:
                    errors.append(f"{where}.match_sha256 must be lowercase SHA-256")
                if not isinstance(reason, str) or not reason.strip():
                    errors.append(f"{where}.reason must be a non-empty string")
                if (
                    isinstance(path, str)
                    and isinstance(rule, str)
                    and isinstance(digest, str)
                ):
                    identity = (path, rule, digest)
                    if identity in seen_exceptions:
                        errors.append(f"duplicate content-match exception: {identity}")
                    seen_exceptions.add(identity)

    profiles = manifest.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        errors.append("profiles must be a non-empty object")
        profiles = {}
    active_profile = manifest.get("active_profile")
    if not isinstance(active_profile, str) or active_profile not in profiles:
        errors.append("active_profile must name a declared profile")

    for name, profile in profiles.items():
        if not isinstance(name, str) or not name:
            errors.append("profile names must be non-empty strings")
            continue
        if not isinstance(profile, dict):
            errors.append(f"profile {name!r} must be an object")
            continue
        for key in ("allow_globs", "required_paths"):
            values = profile.get(key, [])
            if not isinstance(values, list) or not all(
                isinstance(item, str) and item for item in values
            ):
                errors.append(f"profile {name!r} {key} must be a list of strings")
        allow_mode = profile.get("allow_mode", "glob")
        if allow_mode not in ALLOW_MODES:
            errors.append(
                f"profile {name!r} allow_mode must be one of "
                f"{sorted(ALLOW_MODES)}"
            )
        elif allow_mode == "exact":
            for path in profile.get("allow_globs", []):
                if any(character in path for character in "*?["):
                    errors.append(
                        f"profile {name!r} exact allowlist contains wildcard: {path}"
                    )
        parent = profile.get("extends")
        if parent is not None and parent not in profiles:
            errors.append(f"profile {name!r} extends unknown profile {parent!r}")

    components = manifest.get("components")
    if not isinstance(components, list) or not components:
        errors.append("components must be a non-empty list")
        components = []
    seen_ids: set[str] = set()
    for index, component in enumerate(components):
        where = f"components[{index}]"
        if not isinstance(component, dict):
            errors.append(f"{where} must be an object")
            continue
        component_id = component.get("id")
        if not isinstance(component_id, str) or not component_id:
            errors.append(f"{where}.id must be a non-empty string")
        elif component_id in seen_ids:
            errors.append(f"duplicate component id: {component_id}")
        else:
            seen_ids.add(component_id)
        classification = component.get("classification")
        if classification not in CLASSIFICATIONS:
            errors.append(
                f"{where}.classification must be one of "
                f"{sorted(CLASSIFICATIONS)}"
            )
        if not isinstance(component.get("scope"), str) or not component["scope"]:
            errors.append(f"{where}.scope must be a non-empty string")
    return errors


def resolve_profile(
    manifest: dict[str, Any], profile_name: str
) -> tuple[list[str], list[str]]:
    profiles = manifest["profiles"]
    visiting: set[str] = set()
    resolved: set[str] = set()
    allow_globs: list[str] = []
    required_paths: list[str] = []

    def visit(name: str) -> None:
        if name in resolved:
            return
        if name in visiting:
            raise ManifestError(f"profile inheritance cycle involving {name!r}")
        if name not in profiles:
            raise ManifestError(f"unknown profile {name!r}")
        visiting.add(name)
        profile = profiles[name]
        parent = profile.get("extends")
        if parent is not None:
            visit(parent)
        allow_globs.extend(profile.get("allow_globs", []))
        required_paths.extend(profile.get("required_paths", []))
        visiting.remove(name)
        resolved.add(name)

    visit(profile_name)
    return list(dict.fromkeys(allow_globs)), list(dict.fromkeys(required_paths))


def _iter_tree(root: Path, ignored_globs: list[str]) -> tuple[list[str], list[str]]:
    files: list[str] = []
    symlinks: list[str] = []
    for directory, names, filenames in os.walk(root, topdown=True, followlinks=False):
        directory_path = Path(directory)
        relative_directory = directory_path.relative_to(root)
        kept_names: list[str] = []
        for name in names:
            candidate = directory_path / name
            relative = candidate.relative_to(root).as_posix()
            if _matches(relative, ignored_globs):
                continue
            if candidate.is_symlink():
                symlinks.append(relative)
                continue
            kept_names.append(name)
        names[:] = kept_names
        for name in filenames:
            candidate = directory_path / name
            relative = candidate.relative_to(root).as_posix()
            if _matches(relative, ignored_globs):
                continue
            if candidate.is_symlink():
                symlinks.append(relative)
            else:
                files.append(relative)
    return sorted(files), sorted(symlinks)


def _deny_reason(path: str, policy: dict[str, Any]) -> str | None:
    if _matches(path, policy.get("always_deny_globs", [])):
        return "always-deny glob"
    denied_components = set(policy.get("always_deny_path_components", []))
    if denied_components.intersection(Path(path).parts):
        return "always-deny path component"
    return None


def _size_limit(path: str, policy: dict[str, Any]) -> int:
    for exception in policy.get("large_file_exceptions", []):
        if _matches(path, [exception["glob"]]):
            return int(exception["maximum_file_bytes"])
    return int(policy.get("maximum_file_bytes", 8_000_000))


def _content_rules() -> list[tuple[str, re.Pattern[str]]]:
    private_roots = (
        "home",
        "Users",
        "lustre",
        "scratch",
        "checkpoint",
        "gpfs",
        "nfs",
        "fsx",
    )
    private_path = re.compile("/(?:" + "|".join(private_roots) + r")(?:/|\\b)")
    email = re.compile(
        r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
        r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
        r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+"
    )
    token_prefixes = ("hf_", "sk-", "ghp_", "glpat-", "nvapi-", "github_pat_")
    token = re.compile(
        r"(?:"
        + "|".join(re.escape(value) for value in token_prefixes)
        + r")[A-Za-z0-9_-]{16,}"
        + r"|AIza[A-Za-z0-9_-]{35}"
        + r"|xox[baprs]-[A-Za-z0-9-]{10,}"
        + r"|LLM\|[0-9]{8,}\|[A-Za-z0-9_-]{16,}"
    )
    credential_name = r"(?:api[_-]?key|access[_-]?token|secret[_-]?key)"
    literal_assignment = re.compile(
        credential_name + r"\s*[:=]\s*['\"][^'\"${}<>]{8,}['\"]",
        re.IGNORECASE,
    )
    loopback_name = "local" + "host"
    loopback_address = "127" + r"\.0\.0\.1"
    private_suffixes = "inter" + "nal|lo" + "cal"
    loopback_endpoint = re.compile(
        r"(?:https?://)?(?:"
        + loopback_name
        + "|"
        + loopback_address
        + r")(?:[:/]|\b)",
        re.IGNORECASE,
    )
    private_endpoint = re.compile(
        r"(?:https?://)?(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|"
        + r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}|"
        + r"[A-Za-z0-9.-]+\.(?:"
        + private_suffixes
        + r"))(?:[:/]|\b)",
        re.IGNORECASE,
    )
    scheduler = re.compile(r"(?:^|\n)\s*#SBATCH\b|(?:^|\s)srun\s")
    private_key = re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----")
    return [
        ("machine-local absolute path", private_path),
        ("email address", email),
        ("credential-like token", token),
        ("embedded credential assignment", literal_assignment),
        ("loopback endpoint", loopback_endpoint),
        ("private endpoint", private_endpoint),
        ("cluster scheduler directive", scheduler),
        ("private-key material", private_key),
    ]


def validate_release_tree(
    root: Path, manifest: dict[str, Any], profile_name: str
) -> list[str]:
    errors = validate_manifest_schema(manifest)
    if errors:
        return errors
    try:
        allow_globs, required_paths = resolve_profile(manifest, profile_name)
    except ManifestError as error:
        return [str(error)]
    policy = manifest["policy"]
    ignored = policy.get("ignored_vcs_globs", [])
    files, symlinks = _iter_tree(root, ignored)
    errors.extend(f"symbolic link is not releasable: {path}" for path in symlinks)

    loopback_exceptions = set(policy.get("loopback_endpoint_exceptions", []))
    for path in sorted(loopback_exceptions):
        target = root / path
        if not target.is_file() or target.is_symlink():
            errors.append(f"loopback endpoint exception path is missing: {path}")
    content_exceptions = {
        (item["path"], item["rule"], item["match_sha256"])
        for item in policy.get("content_match_exceptions", [])
    }
    used_content_exceptions: set[tuple[str, str, str]] = set()

    profile = manifest["profiles"][profile_name]
    if profile.get("allow_mode", "glob") == "exact":
        for path in allow_globs:
            if any(character in path for character in "*?["):
                errors.append(f"effective exact allowlist contains wildcard: {path}")
                continue
            target = root / path
            if not target.is_file() or target.is_symlink():
                errors.append(f"exactly allowlisted release file is missing: {path}")

    for required in required_paths:
        target = root / required
        if not target.is_file() or target.is_symlink():
            errors.append(f"required release file is missing: {required}")

    scan_extensions = set(policy.get("content_scan_extensions", []))
    rules = _content_rules()
    for path in files:
        reason = _deny_reason(path, policy)
        if reason is not None:
            errors.append(f"denied path ({reason}): {path}")
            continue
        if not _matches(path, allow_globs):
            errors.append(f"path is not allowlisted for profile {profile_name!r}: {path}")
            continue
        target = root / path
        size = target.stat().st_size
        maximum = _size_limit(path, policy)
        if size > maximum:
            errors.append(
                f"file exceeds {maximum}-byte release limit ({size} bytes): {path}"
            )
        if target.suffix.lower() not in scan_extensions or size > TEXT_READ_LIMIT:
            continue
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeError:
            errors.append(f"declared text file is not valid UTF-8: {path}")
            continue
        for label, pattern in rules:
            if label == "loopback endpoint" and path in loopback_exceptions:
                continue
            unapproved_match = False
            for match in pattern.finditer(text):
                digest = hashlib.sha256(match.group(0).encode("utf-8")).hexdigest()
                identity = (path, label, digest)
                if identity in content_exceptions:
                    used_content_exceptions.add(identity)
                else:
                    unapproved_match = True
            if unapproved_match:
                errors.append(f"content rule matched ({label}): {path}")
    for path, label, digest in sorted(
        content_exceptions - used_content_exceptions
    ):
        errors.append(
            "content-match exception is stale or unused: "
            f"{path} ({label}; sha256={digest})"
        )
    return errors


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="manifest path (default: <root>/release_manifest.json)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="release root to inspect",
    )
    parser.add_argument(
        "--profile",
        help="profile to validate (default: manifest active_profile)",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="validate manifest structure without walking the release tree",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    manifest_path = (args.manifest or root / "release_manifest.json").resolve()
    try:
        manifest = load_manifest(manifest_path)
    except ManifestError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2
    if args.schema_only:
        errors = validate_manifest_schema(manifest)
        profile_name = manifest.get("active_profile", "<invalid>")
    else:
        profile_name = args.profile or manifest.get("active_profile")
        errors = validate_release_tree(root, manifest, profile_name)
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        print(
            f"Release-manifest validation failed with {len(errors)} issue(s).",
            file=sys.stderr,
        )
        return 1
    counts: dict[str, int] = {value: 0 for value in sorted(CLASSIFICATIONS)}
    for component in manifest["components"]:
        counts[component["classification"]] += 1
    summary = ", ".join(f"{key}={value}" for key, value in counts.items())
    print(f"Release manifest valid (profile={profile_name}; {summary}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
