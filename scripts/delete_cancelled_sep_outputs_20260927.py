#!/usr/bin/env python3
"""Delete only the explicitly cancelled September 27 SEP generation artifacts."""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from secpo.tinker.evaluation_common import now, write_json

RUN = Path('/data2/jaewonc/secpo_evals/qwen38_full_sep_20260927')
TARGET = RUN/'secpo/sep_static'
NAMES = ('rows','completion_cache','judgments','judge_cache','smoke','progress.json',
         'generation_summary.json','judge_progress.json','results.json','full.log','smoke.log')


def main():
    if TARGET.is_symlink() or TARGET.resolve() != TARGET:
        raise ValueError('Unexpected SEP output target')
    record = json.loads((TARGET/'full_worker.json').read_text())
    if record.get('exit_code') != -15 or record['status'] != 'failed':
        raise ValueError('SEP worker has not been confirmed stopped')
    if Path(f'/proc/{record["pid"]}').exists():
        raise ValueError('Recorded SEP process still exists')
    targets = [TARGET/name for name in NAMES if (TARGET/name).exists()]
    for path in targets:
        if path.is_symlink() or path.resolve().parent != TARGET:
            raise ValueError('Refusing unexpected artifact path')
        if path.is_dir() and any(child.is_symlink() for child in path.rglob('*')):
            raise ValueError('Unexpected symlink inside cancelled outputs')
    files = [child for path in targets for child in (path.rglob('*') if path.is_dir() else [path]) if child.is_file()]
    audit = dict(status='deleting', at=now(), worker_pid=record['pid'],
        reason='User cancelled SEP Luna variant and requested original three-judge protocol.',
        paths=[str(p) for p in targets], files=len(files), bytes=sum(p.stat().st_size for p in files),
        generated_rows=sum(1 for p in files if p.parent.parent == TARGET/'rows'),
        recoverable=False, preserved='launch metadata and manifests only; no response backups retained')
    write_json(RUN/'sep_cancellation.json', audit)
    for path in targets:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    audit.update(status='deleted', completed_at=now())
    write_json(RUN/'sep_cancellation.json', audit)
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
