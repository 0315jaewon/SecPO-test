import hashlib
import json
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import export_lambda_gpqa_bundle as exporter

runner = exporter.runner


@pytest.fixture
def bundle_paths(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    requests = [dict(doc_id=i, prompt=f"Question {i}" + runner.SUFFIX,
                     prompt_sha256=runner.digest(f"Question {i}" + runner.SUFFIX),
                     prompt_token_ids=[1, 2, i]) for i in range(198)]
    rows = [dict(doc_id=i, doc={"question": str(i)}, filter="flexible-extract") for i in range(198)]
    manifest = dict(checkpoint=f"base:{runner.MODEL}", model=runner.MODEL,
                    tokenizer_revision=runner.REVISION, enable_thinking=False,
                    lm_eval_version="0.4.11", default_max_gen_toks=16384,
                    random_seed=42, numpy_seed=42, torch_seed=42, fewshot_seed=42,
                    tasks={runner.TASK: dict(indices=list(range(198)),
                        generation_kwargs=runner.TASK_PARAMS, num_fewshot=0,
                        selected_documents_sha256=runner.digest([r["doc"] for r in rows]))})
    runner.write_json(reference / "manifest.json", manifest)
    runner.write_json(reference / "results.json", {"samples": {runner.TASK: rows}})
    runner.write_json(reference / "worker.json", {"status": "complete", "exit_code": 0})
    hashes = {}
    for name in exporter.TOKENIZER_FILES:
        blob = tmp_path / (name + ".blob")
        blob.write_bytes(("test contents for " + name).encode())
        (tokenizer / name).symlink_to(blob)
        hashes[name] = runner.file_hash(blob)
    bundle = dict(schema="secpo-lambda-gpqa-v1", model=runner.MODEL,
                  revision=runner.REVISION, enable_thinking=False,
                  sampling_params=runner.PARAMS, requests=requests,
                  requests_sha256=runner.digest(requests),
                  reference_manifest_sha256=runner.digest(manifest),
                  reference_results_sha256=runner.file_hash(reference / "results.json"),
                  tokenizer_files_sha256=hashes)
    inputs = tmp_path / "inputs.json"
    runner.write_json(inputs, bundle)
    return dict(inputs=inputs, reference=reference, tokenizer=tokenizer,
                output=tmp_path / "private.tar.gz")


def test_exports_only_required_regular_files_and_valid_checksums(bundle_paths):
    (bundle_paths["reference"] / ".env").write_text("NOT_FOR_EXPORT=test-only")
    (bundle_paths["tokenizer"] / "model.safetensors").write_text("not a model")
    info = exporter.export_bundle(**bundle_paths)
    assert info["files"] == 10
    assert info["sha256"] == runner.file_hash(bundle_paths["output"])
    with tarfile.open(bundle_paths["output"]) as archive:
        members = archive.getmembers()
        assert len(members) == 10
        assert all(m.isfile() and m.mode == 0o600 for m in members)
        names = {m.name for m in members}
        assert names == {"lambda_gpqa/inputs.json", "lambda_gpqa/SHA256SUMS",
                         *("lambda_gpqa/reference/" + n for n in ("manifest.json", "results.json", "worker.json")),
                         *("lambda_gpqa/tokenizer/" + n for n in exporter.TOKENIZER_FILES)}
        checksum_text = archive.extractfile("lambda_gpqa/SHA256SUMS").read().decode()
        for line in checksum_text.splitlines():
            checksum, name = line.split("  ", 1)
            assert hashlib.sha256(archive.extractfile("lambda_gpqa/" + name).read()).hexdigest() == checksum
        assert archive.extractfile("lambda_gpqa/inputs.json").read() == bundle_paths["inputs"].read_bytes()


def test_refuses_to_overwrite_existing_archive(bundle_paths):
    bundle_paths["output"].write_bytes(b"preserve me")
    with pytest.raises(FileExistsError):
        exporter.export_bundle(**bundle_paths)
    assert bundle_paths["output"].read_bytes() == b"preserve me"


@pytest.mark.parametrize("file", ["manifest.json", "results.json", "worker.json"])
def test_changed_reference_is_rejected_before_writing(bundle_paths, file):
    path = bundle_paths["reference"] / file
    data = json.loads(path.read_text())
    if file == "manifest.json":
        data["extra_metadata"] = "changed"
    elif file == "results.json":
        data["extra_metadata"] = "changed"
    else:
        data["status"] = "failed"
    runner.write_json(path, data)
    with pytest.raises(ValueError):
        exporter.export_bundle(**bundle_paths)
    assert not bundle_paths["output"].exists()


def test_wrong_tokenizer_rejected(bundle_paths):
    (bundle_paths["tokenizer"] / "tokenizer.json").write_text("changed")
    with pytest.raises(ValueError, match="tokenizer"):
        exporter.export_bundle(**bundle_paths)
    assert not bundle_paths["output"].exists()


def test_unexpected_tokenizer_path_rejected(bundle_paths):
    data = json.loads(bundle_paths["inputs"].read_text())
    data["tokenizer_files_sha256"]["../../.env"] = "not-a-real-hash"
    runner.write_json(bundle_paths["inputs"], data)
    with pytest.raises(ValueError, match="file list"):
        exporter.export_bundle(**bundle_paths)
    assert not bundle_paths["output"].exists()


def test_mutated_input_rejected(bundle_paths):
    data = json.loads(bundle_paths["inputs"].read_text())
    data["requests"][0]["prompt"] = "changed"
    runner.write_json(bundle_paths["inputs"], data)
    with pytest.raises(ValueError):
        exporter.export_bundle(**bundle_paths)
    assert not bundle_paths["output"].exists()
