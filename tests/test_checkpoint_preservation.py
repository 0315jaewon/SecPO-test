"""Offline checks for checkpoint backup validation; no API calls."""

import importlib.util
import io
from pathlib import Path
import tarfile

import pytest

SPEC = importlib.util.spec_from_file_location(
    "preserve_checkpoints", Path(__file__).resolve().parents[1] / "scripts/preserve_tinker_checkpoints.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def archive(path, name="adapter.bin", content=b"weights", link=False):
    with tarfile.open(path, "w") as bundle:
        member = tarfile.TarInfo(name)
        if link:
            member.type = tarfile.SYMTYPE
            member.linkname = "outside"
            bundle.addfile(member)
        else:
            member.size = len(content)
            bundle.addfile(member, io.BytesIO(content))


def test_archive_hash_check(tmp_path):
    import hashlib
    path = tmp_path / "model.tar"
    archive(path)
    expected = {"adapter.bin": hashlib.sha256(b"weights").hexdigest()}
    assert MODULE.verify_archive(path, expected) == expected
    with pytest.raises(ValueError, match="differs"):
        MODULE.verify_archive(path, {"adapter.bin": "incorrect"})


@pytest.mark.parametrize("name,link", [("../outside", False), ("/absolute", False), ("adapter.bin", True)])
def test_reject_unsafe_archives(tmp_path, name, link):
    path = tmp_path / "model.tar"
    archive(path, name, link=link)
    with pytest.raises(ValueError, match="Unsafe"):
        MODULE.verify_archive(path)


def test_never_overwrite_unknown_backup(tmp_path):
    path = tmp_path / "model.tar"
    path.write_bytes(b"existing")
    with pytest.raises(ValueError, match="identity manifest"):
        MODULE.backup_one(None, "tinker://example/weights/final", path)
    assert path.read_bytes() == b"existing"
