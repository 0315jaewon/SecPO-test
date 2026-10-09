import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('shutdown_archive', ROOT / 'scripts/archive_lambda_shutdown.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


def manifest_for(source, path, mode='links'):
    with path.open('w') as output:
        subprocess.run([sys.executable, '-c', archive.HASH_CODE, str(source), mode,
                        json.dumps(archive.EXCLUDED_NAMES), '[]'], stdout=output, check=True)


def test_manifest_verifies_contents_and_links(tmp_path):
    source, copied = tmp_path / 'source', tmp_path / 'copied'
    source.mkdir()
    copied.mkdir()
    for directory in (source, copied):
        (directory / 'weights.bin').write_bytes(b'checkpoint bytes')
        (directory / 'alias').symlink_to('weights.bin')
    manifest = tmp_path / 'manifest.jsonl'
    manifest_for(source, manifest)
    verified = archive.verify_manifest(manifest, copied)
    assert verified['files'] == 2
    assert verified['bytes'] == len(b'checkpoint bytes')
    (copied / 'weights.bin').write_bytes(b'corrupted bytes!')
    with pytest.raises(ValueError, match='checksum differs'):
        archive.verify_manifest(manifest, copied)


def test_model_symlinks_are_dereferenced(tmp_path):
    source, copied = tmp_path / 'source', tmp_path / 'copied'
    source.mkdir()
    copied.mkdir()
    blob = tmp_path / 'blob'
    blob.write_bytes(b'actual model data')
    (source / 'model.safetensors').symlink_to(blob)
    (copied / 'model.safetensors').write_bytes(blob.read_bytes())
    manifest = tmp_path / 'manifest.jsonl'
    manifest_for(source, manifest, 'dereference')
    assert archive.verify_manifest(manifest, copied)['files'] == 1


def test_secrets_and_git_are_excluded(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'keep.json').write_text('{}')
    (source / 'secpo_openai.env').write_text('FAKE_SECRET')
    (source / 'id_ed25519').write_text('FAKE_SECRET')
    (source / '.git').mkdir()
    (source / '.git/config').write_text('FAKE_SECRET')
    manifest = tmp_path / 'manifest.jsonl'
    manifest_for(source, manifest)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert [r['path'] for r in rows] == ['keep.json']


def test_reject_unsafe_inventory_path(tmp_path):
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(json.dumps({'path': '../escape', 'kind': 'file', 'bytes': 1, 'sha256': 'x'}) + '\n')
    with pytest.raises(ValueError, match='Unsafe'):
        archive.verify_manifest(manifest, tmp_path)


def test_all_bulk_destinations_are_on_data2():
    assert str(archive.DEST).startswith('/data2/jaewonc/')
    assert set(archive.MODELS) == {'base', 'secpo_lr8e5', 'secpo_lr6e5', 'secpo_lr4e5'}


def test_resumed_task_does_not_keep_stale_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, 'DEST', tmp_path)
    archive.task_status('base', status='failed', error_type='RuntimeError')
    archive.task_status('base', status='verifying_local', current_file='shard.bin', verified_bytes=10)
    progress = archive.read(tmp_path / 'audit/tasks/base.json')
    assert 'error_type' not in progress
    archive.task_status('base', status='complete', bytes=20)
    completed = archive.read(tmp_path / 'audit/tasks/base.json')
    assert completed['status'] == 'complete'
    assert completed['bytes'] == 20
    assert 'current_file' not in completed
    assert 'verified_bytes' not in completed


def populate_finished_results(destination):
    counts = dict(gpqa_diamond=198, mmlu_pro=1400, minerva_math=700,
                  agentdojo=97, agentdyn=60, alpacaeval2=805, sep_utility=9160)
    for model in ('base', 'secpo', 'secpo_lr6e5', 'secpo_lr4e5'):
        for benchmark, count in counts.items():
            run = ('qwen38_lambda_luna_openai_20260918'
                   if benchmark in {'alpacaeval2', 'sep_utility'}
                   else 'qwen38_lambda_recommended_20260916')
            directory = destination / 'workspace/outputs' / run / model / benchmark
            archive.write(directory / 'progress.json', dict(status='complete', completed=count))
            archive.write(directory / 'results.json', dict(score=0.5))


def test_shutdown_readiness_requires_all_full_results(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, 'DEST', tmp_path)
    populate_finished_results(tmp_path)
    missing = tmp_path / 'workspace/outputs/qwen38_lambda_luna_openai_20260918/secpo_lr4e5/sep_utility/progress.json'
    archive.write(missing, dict(status='complete', completed=9159))
    with pytest.raises(ValueError, match='full evaluation result is missing'):
        archive.finalize_backup()
    assert not (tmp_path / 'audit/shutdown_readiness.json').exists()


def test_shutdown_readiness_records_all_models_and_results(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, 'DEST', tmp_path)
    populate_finished_results(tmp_path)
    for name in archive.MODELS:
        path = tmp_path / 'audit' / (name + '.remote_sha256.jsonl')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(path='model.safetensors', kind='file', sha256='a' * 64)) + '\n')
    archive.finalize_backup()
    audit = archive.read(tmp_path / 'audit/shutdown_readiness.json')
    assert audit['full_benchmark_results'] == 28
    assert len(audit['results']) == 28
    assert set(audit['models']) == set(archive.MODELS)
    assert len((tmp_path / 'SHA256SUMS.models').read_text().splitlines()) == 4
    assert (tmp_path / 'README.md').is_file()
