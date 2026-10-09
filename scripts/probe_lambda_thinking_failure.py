#!/usr/bin/env python3
"""Retry just the missing failed example, preserving its raw response first."""
import fcntl
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from secpo.tinker.evaluation_common import digest, write_json, now

OUT = Path('/data2/jaewonc/secpo_evals/qwen38_thinking_alpaca_20260923/base/alpacaeval2')


def main():
    from transformers import AutoTokenizer
    import requests
    spec = importlib.util.spec_from_file_location('original_thinking', OUT/'code_snapshot/src/secpo/inference/thinking_eval.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    tokenizer = AutoTokenizer.from_pretrained('/data2/jaewonc/secpo_backups/lambda_20260919/models/base', local_files_only=True)
    manifest = json.loads((OUT/'manifest.json').read_text())
    refs = json.loads((ROOT/'data/eval_references/alpaca_eval2_gpt-4o-2024-05-13_official_reference_outputs.json').read_text())
    index = 466
    ids = tokenizer.encode(original.render_prompt(tokenizer, refs[index]['instruction']), add_special_tokens=False)
    params = dict(max_tokens=32768, seed=42, stop=original.STOPS, **original.SAMPLING)
    identity = dict(model=manifest['model_identity'], render_version=original.RENDER_VERSION,
                    prompt_token_ids=ids, params=params)
    key = digest(identity)
    cache = OUT/'completion_cache'
    locks = cache/'request_locks'
    locks.mkdir(exist_ok=True)
    with (locks/(key+'.lock')).open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        raw_path = cache/'raw_responses'/(key+'.json')
        if raw_path.exists():
            raw = json.loads(raw_path.read_text())
            assert raw['request_identity']==identity and raw['request_hash']==key
            result = raw['response']
        else:
            if (cache/(key+'.json')).exists():
                raise ValueError('Example is already cached; no new inference needed')
            start = time.monotonic()
            write_json(OUT/'failure_probe.json', dict(status='running', index=index, request_hash=key, at=now()))
            payload = dict(model=original.MODEL_NAME, prompt=ids, n=1, return_token_ids=True,
                           skip_special_tokens=False, **params)
            with requests.Session() as session:
                session.trust_env=False
                response=session.post(manifest['endpoint']+'/completions',json=payload,timeout=(30,7200))
                if response.status_code != 200:
                    raise RuntimeError('Inference HTTP '+str(response.status_code))
                result=response.json()
            write_json(raw_path,dict(request_hash=key,request_identity=identity,response=result,
                                    elapsed_seconds=time.monotonic()-start,created_at=now(),source='failed-example-probe'))
        choice=result['choices'][0]
        output_ids=choice.get('token_ids')
        diagnostic=dict(index=index, request_hash=key, at=now(), finish_reason=choice.get('finish_reason'),
            expected_prompt_tokens=len(ids), reported_usage=result.get('usage'),
            returned_token_ids=len(output_ids) if isinstance(output_ids,list) else None,
            closing_think_present=248069 in output_ids if isinstance(output_ids,list) else None)
        try:
            if result.get('model') != original.MODEL_NAME or len(result['choices']) != 1:
                raise ValueError('Unexpected inference model or choice count')
            if not isinstance(output_ids,list) or any(type(i) is not int for i in output_ids):
                raise ValueError('Raw output token IDs missing')
            if (result['usage']['prompt_tokens']!=len(ids) or result['usage']['completion_tokens']!=len(output_ids)
                    or len(output_ids)>32768 or choice['finish_reason'] not in {'length','stop'}):
                raise ValueError('Token counts or termination reason differ')
            original.split_token_output(tokenizer,output_ids,choice['finish_reason'])
        except ValueError as error:
            diagnostic.update(status='original_validator_failed', reason=str(error))
        else:
            diagnostic['status']='original_validator_passed'
        write_json(OUT/'failure_probe.json',diagnostic)
        print(json.dumps(diagnostic),flush=True)


if __name__=='__main__':
    main()
