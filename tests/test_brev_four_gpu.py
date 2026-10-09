from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import brev_inference_proxy as proxy
import launch_brev_four_gpu as queue


def payload_and_result():
    payload = dict(model='secpo-suite-base',prompt=[1,2],max_tokens=16384,seed=42,
                   temperature=0.7,top_p=0.8,top_k=20,presence_penalty=1.5)
    result = dict(model='secpo-suite-base',choices=[dict(token_ids=[3],finish_reason='stop')],
                  usage=dict(prompt_tokens=2,completion_tokens=1))
    return payload,result


def test_response_checks_reject_aborts_and_misaccounting():
    payload,result = payload_and_result()
    assert proxy.response_problem(payload,result) is None
    bad = deepcopy(result); bad['choices'][0]['finish_reason'] = 'abort'
    assert proxy.response_problem(payload,bad) == 'aborted_or_unexpected_finish_reason'
    bad = deepcopy(result); bad['usage']['completion_tokens'] = 2
    assert proxy.response_problem(payload,bad) == 'token_accounting'
    assert proxy.response_problem(payload,{}) == 'malformed_response'


def test_replica_routing_is_balanced_without_changing_concurrency():
    router = object.__new__(proxy.ReplicaProxy)
    router.replicas = ['a','b']; router.cursor = 0; router.routing_lock = threading.Lock()
    assert [router.choose(True) for _ in range(8)] == ['a','b']*4
    assert router.choose(False) == 'a'
    assert queue.original.worker_command('sep_static','full')[-1] == '8'
    assert queue.original.worker_command('mmlu_pro','full')[-1] == '12'


def test_forwarding_preserves_body_and_aborted_response(monkeypatch):
    payload,result = payload_and_result()
    result['choices'][0]['finish_reason'] = 'abort'
    sent = json.dumps(payload,indent=2).encode()
    returned = json.dumps(result,indent=3).encode()
    calls = []
    audits = []
    class Connection:
        def __init__(self,*args,**kwargs): pass
        def request(self,*args): calls.append(args)
        def getresponse(self): return SimpleNamespace(status=200,read=lambda:returned)
        def close(self): pass
    monkeypatch.setattr(proxy.http.client,'HTTPConnection',Connection)
    handler = object.__new__(proxy.Handler)
    handler.path = '/v1/completions'
    handler.headers = {'Content-Length':str(len(sent))}
    handler.rfile = io.BytesIO(sent); handler.wfile = io.BytesIO()
    handler.send_response = lambda status: calls.append(status)
    handler.send_header = lambda *a: None
    handler.end_headers = lambda: None
    handler.server = SimpleNamespace(
        choose=lambda completion:dict(name='replica',gpu_id=1,local_port=19540),
        audit=lambda *args:audits.append(args))
    handler.forward(True)
    assert calls[0][2] == sent
    assert handler.wfile.getvalue() == returned
    assert audits[0][0]['problem'] == 'aborted_or_unexpected_finish_reason'
    assert audits[0][1:] == (sent,returned)


def test_execution_uses_four_unique_gpus_and_original_cache_endpoints():
    replicas = queue.JOBS
    assert sorted(j['gpu_id'] for j in replicas.values()) == [1,2,5,6]
    assert len({j['gpu_uuid'] for j in replicas.values()}) == 4
    assert [j['benchmark'] for j in replicas.values()].count('sep_static') == 2
    assert queue.RUN.name == 'qwen38_brev_base_full_20261003_ready'
    assert queue.original.SETTINGS['jobs']['sep_static']['local_port'] == 19440


def test_remote_override_keeps_server_recipe_identical():
    # Isolate the execution-only module overrides from other tests' imports.
    code = '''
import sys
sys.path.insert(0,'scripts')
import brev_base_remote as base
before=base.server_command('sep_static')
import brev_four_gpu_remote as remote
remote.validate_allocation()
assert sorted(j['gpu_id'] for j in base.SETTINGS['jobs'].values())==[1,2,5,6]
after=base.server_command('resume_sep_a')
for values in (before,after): values[values.index('--port')+1]='PORT'
assert before==after
assert base.env()['CUDA_VISIBLE_DEVICES']==''
'''
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)


def test_frozen_evaluation_sources_are_unchanged():
    assert queue.original.identity() == queue.original.read(queue.RUN/'source_identity.json')
