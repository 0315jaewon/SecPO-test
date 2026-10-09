"""Offline integrity checks; no Hub writes or credentials needed."""

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    'hf_lr8e5_upload', Path(__file__).resolve().parents[1] / 'scripts/upload_hf_lr8e5.py')
UPLOAD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UPLOAD)


@pytest.fixture
def remote(tmp_path, monkeypatch):
    monkeypatch.setattr(UPLOAD, 'STAGE', tmp_path)
    data = b'example weights'
    (tmp_path / 'model.safetensors').write_bytes(data)
    (tmp_path / 'README.md').write_bytes(b'model card\n')
    hashes = {p.name: UPLOAD.sha256(p) for p in tmp_path.iterdir()}
    entries = []
    for name, digest in hashes.items():
        payload = (tmp_path / name).read_bytes()
        entries.append(SimpleNamespace(
            path=name, size=len(payload),
            blob_id=hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest(),
            lfs=SimpleNamespace(sha256=digest) if name.endswith('.safetensors') else None))
    info = SimpleNamespace(private=True, sha='pinned-commit')
    def tree(repo_id, *, revision):
        assert repo_id == UPLOAD.REPO
        assert revision == 'pinned-commit'
        return entries
    api = SimpleNamespace(model_info=lambda _: info, list_repo_tree=tree)
    return api, hashes, entries, info


def test_remote_verification_checks_lfs_and_git_blobs(remote):
    api, hashes, _, _ = remote
    assert UPLOAD.verify_remote(api, hashes, complete=True) == 'pinned-commit'


def test_public_repository_is_rejected(remote):
    api, hashes, _, info = remote
    info.private = False
    with pytest.raises(ValueError, match='public'):
        UPLOAD.verify_remote(api, hashes, complete=True)


def test_lfs_hash_mismatch_is_rejected(remote):
    api, hashes, entries, _ = remote
    next(e for e in entries if e.lfs).lfs.sha256 = 'wrong'
    with pytest.raises(ValueError, match='LFS checksum'):
        UPLOAD.verify_remote(api, hashes, complete=True)


def test_git_blob_mismatch_is_rejected(remote):
    api, hashes, entries, _ = remote
    next(e for e in entries if not e.lfs).blob_id = 'wrong'
    with pytest.raises(ValueError, match='Git blob checksum'):
        UPLOAD.verify_remote(api, hashes, complete=True)


def test_size_mismatch_is_rejected(remote):
    api, hashes, entries, _ = remote
    entries[0].size += 1
    with pytest.raises(ValueError, match='size mismatch'):
        UPLOAD.verify_remote(api, hashes, complete=True)


def test_partial_upload_only_allowed_before_completion(remote):
    api, hashes, entries, _ = remote
    entries.pop()
    UPLOAD.verify_remote(api, hashes, complete=False)
    with pytest.raises(ValueError, match='incomplete'):
        UPLOAD.verify_remote(api, hashes, complete=True)


def test_unknown_existing_files_are_rejected(remote):
    api, hashes, entries, _ = remote
    entries.append(SimpleNamespace(path='unrelated.txt', blob_id='anything'))
    with pytest.raises(ValueError, match='Unexpected existing'):
        UPLOAD.verify_remote(api, hashes, complete=False)


def test_fixed_destination_and_source_are_lr8e5():
    assert UPLOAD.REPO == 'promptinjections/Qwen3.8-27B-SecPO-lr8e-5'
    assert UPLOAD.SOURCE.name == 'secpo_lr8e5'
    assert str(UPLOAD.ROOT).startswith('/data2/jaewonc/')
