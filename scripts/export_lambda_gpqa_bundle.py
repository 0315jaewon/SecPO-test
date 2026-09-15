#!/usr/bin/env python3
"""Package private frozen inputs for the original Lambda GPQA runner.

This packaging helper was added after the run. It does not generate or score
answers. The resulting archive contains benchmark data and must not be committed.
Only an explicit allowlist is copied; no model weights or credentials are needed.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

import lambda_gpqa as runner


TOKENIZER_FILES = {
    "chat_template.jinja", "merges.txt", "tokenizer.json",
    "tokenizer_config.json", "vocab.json",
}


def export_bundle(inputs, reference, tokenizer, output):
    inputs, reference, tokenizer, output = map(Path, (inputs, reference, tokenizer, output))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    bundle = json.loads(inputs.read_text())
    runner.check_inputs(bundle)
    manifest, _, _ = runner.read_reference(reference)
    if (runner.digest(manifest) != bundle["reference_manifest_sha256"]
            or runner.file_hash(reference / "results.json") != bundle["reference_results_sha256"]):
        raise ValueError("Reference does not match the frozen input bundle")
    expected = bundle["tokenizer_files_sha256"]
    if set(expected) != TOKENIZER_FILES:
        raise ValueError("Unexpected tokenizer file list; refusing to copy arbitrary files")
    paths = {"inputs.json": inputs}
    for name in ("manifest.json", "results.json", "worker.json"):
        paths[f"reference/{name}"] = reference / name
    for name in sorted(TOKENIZER_FILES):
        source = tokenizer / name
        if not source.is_file() or runner.file_hash(source) != expected[name]:
            raise ValueError(f"Missing or changed tokenizer file: {name}")
        paths[f"tokenizer/{name}"] = source
    # Read and validate everything before creating the destination. Reading
    # tokenizer symlinks copies their contents, not dangling HF-cache symlinks.
    contents = {name: path.read_bytes() for name, path in sorted(paths.items())}
    checksums = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in contents.items()
    )
    contents["SHA256SUMS"] = checksums.encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally replacing an existing archive.
    with tarfile.open(output, "x:gz") as archive:
        for name, data in contents.items():
            info = tarfile.TarInfo("lambda_gpqa/" + name)
            info.size = len(data)
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(data))
    return {"archive": str(output), "files": len(contents),
            "sha256": runner.file_hash(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_bundle(args.inputs, args.reference, args.tokenizer, args.output), indent=2))


if __name__ == "__main__":
    main()
