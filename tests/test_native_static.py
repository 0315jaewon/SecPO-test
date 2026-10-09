"""Offline integration checks, run in each benchmark's pinned environment."""
import json
from types import SimpleNamespace
import pytest

from secpo.inference.native_static import (
    verify_originals, served, template, configure_environment, pipeline, elements, ChatCache,
)


def test_original_evaluators_are_unmodified():
    assert len(verify_originals()) == 2


def test_template_preserves_trust_roles_and_string_argument_whitespace():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        '/data2/jaewonc/secpo_backups/lambda_20260919/models/base',local_files_only=True)
    name='External_\n\nINFORMATION\n\n'
    messages=[{'role':'system','content':'System'}, {'role':'user','content':'User'},
        {'role':'assistant','content':None,'tool_calls':[{'type':'function','id':'c',
            'function':{'name':'lookup','arguments':{'channel':name}}}]},
        {'role':'tool','content':'Native untrusted result','tool_call_id':'c'},
        {'role':'input','content':'Converted untrusted result'}]
    text=tokenizer.apply_chat_template(messages,chat_template=template(),tokenize=False,
        enable_thinking=False,add_generation_prompt=True)
    assert '<|im_start|>system' not in text
    assert '<|im_start|>input\n<tool_response>\nNative untrusted result' in text
    assert '<|im_start|>input\nConverted untrusted result' in text
    assert '<parameter=channel>\n'+name+'\n</parameter>' in text
    assert text.endswith('<think>\n\n</think>\n\n')


@pytest.mark.parametrize('valid_call',[False,True])
def test_original_native_adapter_request_response_and_cache(monkeypatch,tmp_path,valid_call):
    pytest.importorskip('agentdojo')
    from importlib.metadata import version
    from openai.types.chat import ChatCompletion
    from agentdojo.task_suite.load_suites import get_suite
    from agentdojo.functions_runtime import FunctionsRuntime
    benchmark='agentdojo' if version('agentdojo')=='0.1.30' else 'agentdyn'
    configure_environment(benchmark,'secpo_lr4e5')
    agent=pipeline(benchmark,'secpo_lr4e5')
    llm=next(e for e in elements(agent) if getattr(e,'client',None) is not None)
    assert llm.__class__.__module__=='secpo.evaluation.'+benchmark+'_qwen36'
    name='External_\n\nINFORMATION\n\n'
    incomplete='<tool_call>\n<function=lookup>\n<parameter=channel>\nExternal_'
    message={'role':'assistant','content':None if valid_call else incomplete,
        'tool_calls':[{'id':'native-call','type':'function','function':{
            'name':'lookup','arguments':json.dumps({'channel':name})}}] if valid_call else None}
    response=ChatCompletion.model_validate(dict(id='local-test',created=0,model=served('secpo_lr4e5'),
        object='chat.completion',choices=[dict(index=0,message=message,finish_reason='stop')],
        usage=dict(prompt_tokens=10,completion_tokens=20,total_tokens=30)))
    requests=[]
    def create(**request):
        requests.append(request)
        return response
    llm.client.chat.completions.create=create
    def forbidden(*args,**kwargs):
        raise AssertionError('Custom XML parser must never run')
    monkeypatch.setattr('secpo.tinker.agent_eval.parse_tool_response',forbidden)
    cache=ChatCache(tmp_path,'secpo_lr4e5','test-manifest')
    cache.attach(agent)
    suite=get_suite('v1.2.1' if benchmark=='agentdojo' else 'v1.2.2',
                    'slack' if benchmark=='agentdojo' else 'shopping')
    runtime=FunctionsRuntime(suite.tools)
    messages=[{'role':'user','content':[{'type':'text','content':'Do the task'}]}]
    for _ in range(2):
        result=llm.query('Do the task',runtime,suite.load_and_inject_default_environment({}),messages)
        output=result[3][-1]
        if valid_call:
            assert output['tool_calls'][0].args['channel']==name
        else:
            assert not output['tool_calls']
            assert output['content'][0]['content']==incomplete
    assert len(requests)==1
    assert cache.stats()['cached_requests']==1
    assert requests[0]['max_tokens']==8192
    assert requests[0]['extra_body']['chat_template_kwargs']['enable_thinking'] is False
    assert len(verify_originals())==2
