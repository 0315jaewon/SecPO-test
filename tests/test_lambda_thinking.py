import json
from types import SimpleNamespace
import pytest
from secpo.inference.thinking_eval import split_answer, split_token_output, render_prompt, ThinkingSampler, SAMPLING, MODEL_NAME


def test_answer_only_and_stop_trimming():
    result=split_answer('private reasoning\n</think>\n\nThe answer.\n<|im_end|>ignored','stop')
    assert result['text']=='The answer.\n'
    assert result['reasoning']=='private reasoning\n'
    assert result['reasoning_complete'] and not result['empty_final']


def test_incomplete_thinking_never_leaks_to_judge():
    result=split_answer('unfinished reasoning','length')
    assert result['text']=='' and result['empty_final'] and not result['reasoning_complete']
    result=split_answer('stopped during reasoning','stop')
    assert result['text']=='' and result['empty_final'] and not result['reasoning_complete']


def test_length_limited_final_is_retained():
    assert split_answer('reason</think>\npartial answer','length')['text']=='partial answer'
    assert split_answer('reason</think>\n','length')['empty_final']


def test_literal_tag_in_final_is_not_removed():
    assert split_answer('reason</think>\nUse </think> as a tag.','stop')['text']=='Use </think> as a tag.'


def test_rendering_uses_official_thinking_settings():
    def template(messages,**kwargs):
        assert messages==[dict(role='user',content='Question')]
        assert kwargs==dict(tokenize=False,add_generation_prompt=True,enable_thinking=True,
                           preserve_thinking=True,reasoning_effort='xhigh')
        return 'Reasoning effort is set to xhigh.\n<|im_start|>assistant\n<think>\n'
    assert render_prompt(SimpleNamespace(apply_chat_template=template),'Question').endswith('<think>\n')


def test_reject_public_endpoint_and_wrong_cap(tmp_path):
    for endpoint in ['https://example.com/v1','http://127.0.0.1.evil/v1','http://user@127.0.0.1/v1']:
        with pytest.raises(ValueError):
            ThinkingSampler(tmp_path,None,endpoint,32768,{})
    with pytest.raises(ValueError):
        ThinkingSampler(tmp_path,None,'http://127.0.0.1:18680/v1',16384,{})


def test_generation_cache_and_token_counts(tmp_path,monkeypatch):
    import requests
    posts=[]
    class Session:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def post(self,url,*,json,timeout):
            assert self.trust_env is False
            assert {k:json[k] for k in SAMPLING}==SAMPLING
            assert json['max_tokens']==32768 and json['prompt']==[10,11]
            posts.append(json)
            return SimpleNamespace(status_code=200,json=lambda:dict(model=MODEL_NAME,
                usage=dict(prompt_tokens=2,completion_tokens=3),
                choices=[dict(token_ids=[1,2,3],finish_reason='stop',text='reason</think>answer')]))
    monkeypatch.setattr(requests,'Session',Session)
    tokenizer=SimpleNamespace(encode=lambda s,**kw:[2] if s=='</think>' else [10,11],
                              decode=lambda ids,**kw:''.join({1:'reason',2:'</think>',3:'answer'}[i] for i in ids))
    sampler=ThinkingSampler(tmp_path,tokenizer,'http://127.0.0.1:18680/v1',32768,{'id':'base'})
    first=sampler.generate('prompt'); second=sampler.generate('prompt')
    assert first==second and first['text']=='answer'
    assert first['reasoning_tokens']==1 and first['after_reasoning_tokens']==1
    assert len(posts)==1 and sampler.stats()==dict(completed_api_requests=1,cached_requests=1,recovered_raw_requests=0)
    saved=json.loads(next(tmp_path.glob('*.json')).read_text())
    assert saved['output_token_ids']==[1,2,3]


def test_quoted_boundary_in_reasoning_is_not_the_special_token():
    tokenizer=SimpleNamespace(encode=lambda *args,**kwargs:[2],
        decode=lambda ids,**kwargs:''.join({1:'Discuss the literal </think> tag.',2:'</think>',3:'Final.'}[i] for i in ids))
    result,boundary=split_token_output(tokenizer,[1,2,3],'stop')
    assert boundary==1 and result['text']=='Final.'
    assert '</think>' in result['reasoning']
    result,boundary=split_token_output(tokenizer,[1],'length')
    assert boundary is None and result['text']==''


def test_overlength_prompt_rejected_without_inference(tmp_path):
    tokenizer=SimpleNamespace(encode=lambda *args,**kwargs:[1]*32769)
    sampler=ThinkingSampler(tmp_path,tokenizer,'http://127.0.0.1:18680/v1',32768,{})
    with pytest.raises(ValueError,match='context'):
        sampler.generate('prompt')


@pytest.mark.parametrize('bad_count',[False,True])
def test_early_stop_and_raw_capture_before_validation(tmp_path,monkeypatch,bad_count):
    import requests
    calls=[]
    class Session:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def post(self,*args,**kwargs):
            calls.append(kwargs['json'])
            return SimpleNamespace(status_code=200,json=lambda:dict(model=MODEL_NAME,
                usage=dict(prompt_tokens=2,completion_tokens=2 if bad_count else 1),
                choices=[dict(token_ids=[1],finish_reason='stop',text='unfinished reasoning')]))
    monkeypatch.setattr(requests,'Session',Session)
    tokenizer=SimpleNamespace(encode=lambda s,**kw:[2] if s=='</think>' else [10,11],
                              decode=lambda ids,**kw:'unfinished reasoning')
    sampler=ThinkingSampler(tmp_path,tokenizer,'http://127.0.0.1:18680/v1',32768,{'id':'base'})
    if bad_count:
        for attempt in range(2):
            with pytest.raises(ValueError,match='Token counts'):
                sampler.generate('prompt')
        assert len(calls)==1  # Invalid raw response remains available without another request.
        assert not list(tmp_path.glob('*.json'))
    else:
        result=sampler.generate('prompt')
        assert result['text']=='' and result['empty_final']
        assert result['stopped_without_final_boundary']
        assert not result['reasoning_complete']
    assert len(list((tmp_path/'raw_responses').glob('*.json')))==1


def test_raw_probe_response_reused_without_network(tmp_path,monkeypatch):
    from secpo.tinker.evaluation_common import digest,write_json
    import requests
    monkeypatch.setattr(requests,'Session',lambda:pytest.fail('Should reuse raw response'))
    tokenizer=SimpleNamespace(encode=lambda s,**kw:[2] if s=='</think>' else [10,11],
        decode=lambda ids,**kw:''.join({1:'reason',2:'</think>',3:'answer'}[i] for i in ids))
    sampler=ThinkingSampler(tmp_path,tokenizer,'http://127.0.0.1:18680/v1',32768,{'id':'base'})
    _,_,identity=sampler.request_identity('prompt'); key=digest(identity)
    write_json(tmp_path/'raw_responses'/(key+'.json'),dict(request_hash=key,request_identity=identity,
        elapsed_seconds=4,response=dict(model=MODEL_NAME,usage=dict(prompt_tokens=2,completion_tokens=3),
            choices=[dict(token_ids=[1,2,3],finish_reason='stop',text='reason</think>answer')])))
    assert sampler.generate('prompt')['text']=='answer'
    assert sampler.stats()['recovered_raw_requests']==1
