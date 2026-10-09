#!/usr/bin/env python3
"""Preserve the two completed SecPO runs, without exposing download credentials.

Remove expiration only on their final checkpoints. Copy and verify the existing
inference exports. Tinker's archive endpoint supports sampler weights only, so
training states remain remote with no expiry. No remote checkpoint is published
or deleted. Existing verified backups are reused.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from secpo.tinker.evaluation_common import load_credentials, now, write_json


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def verify_archive(path, expected_files=None):
    """Read every member, reject unsafe entries, and optionally verify file hashes."""
    found = {}
    with tarfile.open(path, "r|*") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or not (member.isdir() or member.isfile()):
                raise ValueError("Unsafe checkpoint archive member")
            if member.isdir():
                continue
            normalized = str(name)
            if normalized in found:
                raise ValueError("Duplicate checkpoint archive member")
            result = hashlib.sha256()
            with archive.extractfile(member) as stream:
                while chunk := stream.read(8 * 1024 * 1024):
                    result.update(chunk)
            found[normalized] = result.hexdigest()
    if not found:
        raise ValueError("Empty checkpoint archive")
    if expected_files is not None and found != expected_files:
        raise ValueError("Archive differs from the saved export hashes")
    return found


def backup_one(rest, checkpoint, target, existing_export=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = target.with_suffix(".json")
    if target.exists():
        if not metadata_path.exists():
            raise ValueError("Existing backup lacks its identity manifest; not overwriting")
        metadata = json.loads(metadata_path.read_text())
        if metadata["checkpoint"] != checkpoint or metadata["sha256"] != sha256(target):
            raise ValueError("Existing backup identity or hash differs")
        return metadata
    expected = None
    partial = target.with_suffix(".partial")
    if partial.exists():
        raise ValueError("Incomplete backup exists; inspect before retrying")
    if existing_export is not None:
        manifest = json.loads((existing_export / "export_manifest.json").read_text())
        if manifest["checkpoint"] != checkpoint:
            raise ValueError("Export belongs to a different checkpoint")
        expected = manifest["files"]
        with (existing_export / "checkpoint.tar").open("rb") as source, partial.open("xb") as dest:
            shutil.copyfileobj(source, dest, length=8 * 1024 * 1024)
            dest.flush()
            os.fsync(dest.fileno())
        origin = "verified_existing_sampler_export"
    else:
        response = rest.get_checkpoint_archive_url_from_tinker_path(checkpoint).result(timeout=900)
        if not response.url.startswith("https://"):
            raise ValueError("Checkpoint download requires HTTPS")
        # Neither signed URLs nor raw exceptions are written to logs or metadata.
        with urllib.request.urlopen(response.url, timeout=120) as source, partial.open("xb") as dest:
            shutil.copyfileobj(source, dest, length=8 * 1024 * 1024)
            dest.flush()
            os.fsync(dest.fileno())
        origin = "tinker_checkpoint_archive"
    files = verify_archive(partial, expected)
    metadata = {"checkpoint": checkpoint, "archive": str(target), "sha256": sha256(partial),
                "bytes": partial.stat().st_size, "files": files, "origin": origin,
                "verified_at": now()}
    # Manifest first: interruption before the rename leaves a visible partial,
    # never an apparently complete archive with no provenance.
    write_json(metadata_path, metadata)
    partial.rename(target)
    print(json.dumps({"status": "backup_verified", "checkpoint": checkpoint,
                      "archive": str(target), "bytes": metadata["bytes"]}), flush=True)
    return metadata


def preserve(directory):
    import tinker
    load_credentials(ROOT / "secpo_eval_keys.env")
    rest = tinker.ServiceClient(timeout=600, max_retries=2).create_rest_client()
    jobs, records = [], []
    for label, output, export in [
        ("lr8e5", "qwen38_27b_tinker", "secpo_export"),
        ("lr6e5", "qwen38_27b_tinker_lr6e5", "secpo_lr6e5_export"),
    ]:
        result = json.loads((ROOT / "outputs" / output / "result.json").read_text())
        if result["steps_completed"] != 297:
            raise ValueError("Expected completed 297-step training run")
        for field, kind in [("final_state_path", "training_state"), ("final_sampler_path", "sampler")]:
            path = result[field]
            run_id = path.removeprefix("tinker://").split("/", 1)[0]
            checkpoints = rest.list_checkpoints(run_id).result(timeout=120).checkpoints
            found = [c for c in checkpoints if c.tinker_path == path]
            if len(found) != 1:
                raise ValueError("Final checkpoint missing or ambiguous")
            if found[0].expires_at is not None:
                rest.set_checkpoint_ttl_from_tinker_path(path, ttl_seconds=None).result(timeout=120)
            checked = rest.list_checkpoints(run_id).result(timeout=120).checkpoints
            checkpoint = next(c for c in checked if c.tinker_path == path)
            if checkpoint.expires_at is not None or checkpoint.public:
                raise ValueError("Expected private final checkpoint with no expiry")
            records.append({"model": label, **checkpoint.model_dump(mode="json")})
            if kind == "sampler":
                source = ROOT / "runtime/lambda_suite_20260916" / export
                jobs.append((path, directory / label / (kind + ".tar"), source))
    write_json(directory / "retention.json", {"verified_at": now(), "checkpoints": records,
               "automatic_expiration_disabled": True, "deleted": [], "published": []})
    print(json.dumps({"status": "remote_retention_verified", "checkpoints": len(records)}), flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(backup_one, rest, *job) for job in jobs]
        backups = [f.result() for f in futures]
    write_json(directory / "manifest.json", {"status": "complete", "verified_at": now(),
               "checkpoints": records, "backups": backups,
               "local_backup_scope": "Inference adapters only; training-state archives are not supported by Tinker.",
               "note": "All four remote finals have no expiry; local copies still require normal disk backups."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/checkpoint_backups")
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "backup.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(directory / "status.json", {"status": "running", "started_at": now(), "pid": os.getpid()})
        try:
            preserve(directory)
        except Exception as exc:
            write_json(directory / "status.json", {"status": "failed", "ended_at": now(),
                       "error_type": type(exc).__name__})
            print(json.dumps({"status": "failed", "error_type": type(exc).__name__}), flush=True)
            return 1
        write_json(directory / "status.json", {"status": "complete", "ended_at": now()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
