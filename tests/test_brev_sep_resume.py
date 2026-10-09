import ast
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import launch_brev_sep_resume as resume


def test_sep_only_gpu_allocation():
    replicas = resume.SETTINGS['replicas']
    assert sorted(j['gpu_id'] for j in replicas.values()) == [0,2,3,4]
    assert len({j['gpu_uuid'] for j in replicas.values()}) == 4
    assert {j['benchmark'] for j in replicas.values()} == {'sep_static'}
    assert resume.SETTINGS['max_transport_attempts'] == 3


def test_remote_reuses_verified_recipe_and_launches_correct_wrapper():
    code = '''
import sys
sys.path.insert(0,'scripts')
import brev_base_remote as base
before=base.server_command('sep_static')
import brev_sep_resume_remote as remote
remote.configure()
assert sorted(j['gpu_id'] for j in base.SETTINGS['jobs'].values())==[0,2,3,4]
assert remote.previous.launch_command('oct08_sep_0')[1].endswith('brev_sep_resume_remote.py')
after=base.server_command('oct08_sep_0')
for command in (before,after): command[command.index('--port')+1]='PORT'
assert before==after
assert base.env()['CUDA_VISIBLE_DEVICES']==''
'''
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)


def test_only_sep_pipeline_is_invoked():
    tree = ast.parse((ROOT/'scripts/launch_brev_sep_resume.py').read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call) and
             isinstance(n.func,ast.Attribute) and n.func.attr == 'pipeline']
    assert len(calls) == 1
    assert ast.literal_eval(calls[0].args[0]) == 'sep_static'


def test_only_recent_transport_failures_are_retried(monkeypatch,tmp_path):
    run = tmp_path/'run'
    execution = run/'resume'
    sep = run/'base/sep_static'
    sep.mkdir(parents=True)
    errors = execution/'transport_errors'; errors.mkdir(parents=True)
    monkeypatch.setattr(resume,'RUN',run)
    monkeypatch.setattr(resume,'EXECUTION',execution)
    started = '2026-10-08T17:00:00+00:00'
    (sep/'full_worker.json').write_text(json.dumps(dict(status='failed',started_at=started)))
    (sep/'full.log').write_text('RuntimeError: Lambda inference failed with HTTP 502\n')
    (errors/'one.json').write_text(json.dumps(dict(at=started,problem='transport_error')))
    assert resume.recoverable_transport_failure(started)
    assert not resume.recoverable_transport_failure('2026-10-08T18:00:00+00:00')
    (sep/'full.log').write_text('ValueError: Unexpected inference token accounting or finish reason\n')
    assert not resume.recoverable_transport_failure(started)
    (sep/'full.log').write_text('RuntimeError: Lambda inference failed with HTTP 502\n')
    (errors/'one.json').write_text(json.dumps(dict(at=started,problem='token_accounting')))
    assert not resume.recoverable_transport_failure(started)


def test_original_evaluator_sources_still_frozen():
    original=resume.previous.original
    assert original.identity() == original.read(resume.RUN/'source_identity.json')
