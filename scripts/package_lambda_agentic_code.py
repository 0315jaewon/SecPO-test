#!/usr/bin/env python3
"""Build a code-review ZIP for the completed Lambda agentic utility experiment.

Explicitly allowlisted source files only: no environments, symlink traversal,
credentials, model weights, datasets or generated evaluation output.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/qwen38_lambda_recommended_20260916"
PACKAGE = "secpo_lambda_agentic_code"


def digest(content):
    return hashlib.sha256(content).hexdigest()


def credential_values():
    """Load known secret literals only for exclusion checks, never for output."""
    values = []
    for name in ("secpo_eval_keys.env", "secpo_openrouter.env"):
        path = ROOT / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            parts = shlex.split(line, comments=True)
            if parts and parts[0] == "export":
                parts = parts[1:]
            if len(parts) == 1 and "=" in parts[0]:
                key, value = parts[0].split("=", 1)
                if (key.endswith("_API_KEY") or key == "HF_TOKEN") and len(value) >= 12:
                    values.append(value.encode())
    return values


def validate_content(name, content, secrets):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe archive filename")
    if path.suffix not in {".py", ".json", ".md", ".patch"} and "LICENSE" not in path.name:
        raise ValueError("Unexpected archive file type")
    if any(secret in content for secret in secrets):
        raise ValueError("Known credential detected; values withheld")
    if re.search(rb"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----", content):
        raise ValueError("Private key detected; values withheld")
    if re.search(rb"(?:sk-or-v1-[A-Za-z0-9]{24,}|hf_[A-Za-z0-9]{24,}|sk-proj-[A-Za-z0-9_-]{24,})", content):
        raise ValueError("Potential API credential detected; values withheld")


def build(destination):
    if destination.exists():
        raise FileExistsError("Refusing to overwrite an existing sharing archive")
    secrets = credential_values()
    frozen = json.loads((RUN / "source_identity.json").read_text())
    lr6 = RUN / "secpo_lr6e5/code_snapshot"
    provenance = json.loads((lr6 / "provenance.json").read_text())["files"]
    sources = {str(p.relative_to(ROOT)): p for p in sorted((ROOT / "src/secpo").rglob("*.py"))}
    for name in ("scripts/evaluate_tinker_agents.py", "LICENSE", "third_party/patches/agentdyn.patch"):
        sources[name] = ROOT / name
    for variant in ("base", "secpo", "secpo_lr6e5"):
        name = f"configs/lambda_qwen38_{variant}_recommended.json"
        sources[name] = ROOT / name
    sources["README.md"] = ROOT / "docs/sharing/LAMBDA_AGENTIC_CODE_README.md"
    sources["third_party/licenses/AgentDyn-LICENSE"] = ROOT / "runtime/benchmarks/agentdyn/LICENSE"
    for name in ("server_remote.py", "server_lr6e5_remote.py"):
        sources["server/" + name] = ROOT / "runtime/lambda_suite_20260916" / name
    for name in ("scripts/evaluate_tinker_agents.py", "src/secpo/inference/lambda_eval.py"):
        sources["lr6e5_overrides/" + name] = lr6 / name
    content, inventory = {}, {}
    for name, source in sources.items():
        if source.is_symlink() or not source.is_file():
            raise ValueError("Expected a regular source file, not a link")
        data = source.read_bytes()
        sha = digest(data)
        relative = str(source.relative_to(ROOT))
        if relative in frozen and frozen[relative] != sha:
            raise ValueError("Source differs from the original experiment snapshot")
        if relative in provenance and provenance[relative]["source_sha256"] != sha:
            raise ValueError("Shared source differs from lr6e5 experiment provenance")
        if name.startswith("lr6e5_overrides/"):
            original = name.removeprefix("lr6e5_overrides/")
            if provenance[original]["sha256"] != sha:
                raise ValueError("lr6e5 override differs from its experiment snapshot")
        validate_content(name, data, secrets)
        content[name] = data
        inventory[name] = {"sha256": sha, "bytes": len(data), "source": relative}
    commands = []
    for variant in ("base", "secpo", "secpo_lr6e5"):
        for benchmark in ("agentdojo", "agentdyn"):
            worker = json.loads((RUN / variant / benchmark / "full_worker.json").read_text())
            if worker["status"] != "complete" or worker["exit_code"] != 0:
                raise ValueError("Expected a completed historical evaluation")
            commands.append({"model_variant": variant, "benchmark": benchmark,
                             "command": worker["command"]})
    content["RUN_COMMANDS.json"] = (json.dumps({"note": "Historical commands, not portable launchers; see README.md",
                                              "runs": commands}, indent=2) + "\n").encode()
    validate_content("RUN_COMMANDS.json", content["RUN_COMMANDS.json"], secrets)
    inventory["RUN_COMMANDS.json"] = {"sha256": digest(content["RUN_COMMANDS.json"]),
                                     "bytes": len(content["RUN_COMMANDS.json"]),
                                     "source": "command arrays from six completed full_worker.json records"}
    manifest = {"purpose": "Code review of clean agentic utility; not a standalone reproduction environment",
                "experiment": RUN.name, "defense": "repeat_user_prompt", "attack": None,
                "benchmark_commits": {"agentdojo": "d3640b5b03a88eb44dad96852e1de1ef437d836e",
                                      "agentdyn": "5353cf7615b135cace8d07c8f12dac53a16b6db3"},
                "files": inventory, "excludes": ["credentials", "SSH keys", "model weights", "datasets",
                                                  "evaluation outputs", "caches", "virtual environments", "symlinks"]}
    content["ARCHIVE_MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    validate_content("ARCHIVE_MANIFEST.json", content["ARCHIVE_MANIFEST.json"], secrets)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for name, data in sorted(content.items()):
            info = zipfile.ZipInfo(f"{PACKAGE}/{name}", date_time=(2026, 9, 18, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            bundle.writestr(info, data)
    with zipfile.ZipFile(destination) as bundle:
        if bundle.testzip() is not None or len(bundle.namelist()) != len(content):
            raise ValueError("Archive integrity check failed")
        for name, data in content.items():
            if bundle.read(f"{PACKAGE}/{name}") != data:
                raise ValueError("Archive content differs from source")
    return {"archive": str(destination), "files": len(content), "bytes": destination.stat().st_size,
            "sha256": digest(destination.read_bytes()), "integrity": "verified", "credential_scan": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/share/secpo_lambda_agentic_code_20260918.zip")
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
