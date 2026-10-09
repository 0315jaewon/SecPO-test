"""Isolated thinking-on protocol; never change historical non-thinking runs."""
from copy import deepcopy
import fcntl
from pathlib import Path
import json
import threading
import time
from urllib.parse import urlparse

from secpo.tinker.evaluation_common import cut_at_stop, digest, write_json

MODEL_NAME = 'secpo-thinking-base-20260923'
SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
                presence_penalty=0.0, repetition_penalty=1.0, frequency_penalty=0.0)
STOPS = ['<|im_end|>', '<|endoftext|>']
RENDER_VERSION = 'qwen38-alpaca-thinking-xhigh-v1'


def render_prompt(tokenizer, instruction):
    prompt = tokenizer.apply_chat_template([dict(role='user', content=instruction)],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
        preserve_thinking=True, reasoning_effort='xhigh')
    if not prompt.endswith('<|im_start|>assistant\n<think>\n'):
        raise ValueError('Unexpected thinking-on template suffix')
    if 'Reasoning effort is set to xhigh.' not in prompt:
        raise ValueError('Official xhigh template instruction missing')
    return prompt


def split_answer(raw_text, finish_reason):
    """The opening think tag is in the prompt. Never judge unfinished reasoning."""
    raw_text = cut_at_stop(raw_text, STOPS)
    if '</think>' not in raw_text:
        if finish_reason not in {'length', 'stop'}:
            raise ValueError('Unexpected termination reason')
        return dict(reasoning=raw_text, text='', reasoning_complete=False, empty_final=True)
    reasoning, answer = raw_text.split('</think>', 1)
    # Match the usual reasoning parser: remove only boundary whitespace, not
    # words or code fences in the actual answer.
    answer = answer.lstrip('\r\n')
    return dict(reasoning=reasoning, text=answer, reasoning_complete=True,
                empty_final=not bool(answer.strip()))


def split_token_output(tokenizer, output_ids, finish_reason):
    """Use the special boundary token, not a quoted literal in reasoning text."""
    close_ids = tokenizer.encode('</think>', add_special_tokens=False)
    if len(close_ids) != 1:
        raise ValueError('Expected a single closing-think token')
    boundary = output_ids.index(close_ids[0]) if close_ids[0] in output_ids else None
    if boundary is None:
        if finish_reason not in {'length', 'stop'}:
            raise ValueError('Unexpected termination reason')
        parsed = dict(reasoning=tokenizer.decode(output_ids, skip_special_tokens=False),
                      text='', reasoning_complete=False, empty_final=True)
    else:
        answer = cut_at_stop(tokenizer.decode(output_ids[boundary+1:], skip_special_tokens=False), STOPS).lstrip('\r\n')
        parsed = dict(reasoning=tokenizer.decode(output_ids[:boundary], skip_special_tokens=False),
                      text=answer, reasoning_complete=True, empty_final=not bool(answer.strip()))
    return parsed, boundary


class ThinkingSampler:
    def __init__(self, cache_dir, tokenizer, endpoint, max_tokens, model_identity):
        url = urlparse(endpoint)
        if (url.scheme != 'http' or url.hostname != '127.0.0.1' or url.path != '/v1'
                or url.username or url.password or url.query or url.fragment):
            raise ValueError('Inference endpoint must be a private loopback tunnel')
        if max_tokens != 32768:
            raise ValueError('This experiment is approved for a 32768-token output cap')
        self.cache_dir, self.tokenizer = Path(cache_dir), tokenizer
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.endpoint, self.max_tokens = endpoint, max_tokens
        self.model_identity = deepcopy(model_identity)
        self.lock = threading.Lock()
        self.completed = self.cached = 0
        self.recovered_raw = 0

    def request_identity(self, prompt):
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) + self.max_tokens > 65536:
            raise ValueError('Prompt plus output exceeds context budget')
        params = dict(max_tokens=self.max_tokens, seed=42, stop=STOPS, **SAMPLING)
        identity = dict(model=self.model_identity, render_version=RENDER_VERSION,
                        prompt_token_ids=ids, params=params)
        return ids, params, identity

    def generate(self, prompt):
        ids, params, identity = self.request_identity(prompt)
        key = digest(identity)
        locks = self.cache_dir / 'request_locks'
        locks.mkdir(exist_ok=True)
        with (locks / (key + '.lock')).open('a+') as lock:
            # Coordinates retries, including a standalone diagnostic request.
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._generate_locked(ids, params, identity, key)

    def _generate_locked(self, ids, params, identity, key):
        import requests
        path = self.cache_dir / (key + '.json')
        if path.exists():
            record = json.loads(path.read_text())
            if record['request_hash'] != key or record['request_identity'] != identity:
                raise ValueError('Generation cache identity differs')
            parsed, _ = split_token_output(self.tokenizer, record['output_token_ids'], record['finish_reason'])
            if any(record[k] != v for k, v in parsed.items()):
                raise ValueError('Cached reasoning/answer parsing differs')
            with self.lock:
                self.cached += 1
            return record
        raw_path = self.cache_dir / 'raw_responses' / (key + '.json')
        recovered = raw_path.exists()
        if recovered:
            transport = json.loads(raw_path.read_text())
            if transport['request_hash'] != key or transport['request_identity'] != identity:
                raise ValueError('Raw response cache identity differs')
            result = transport['response']
        else:
            started = time.monotonic()
            payload = dict(model=MODEL_NAME, prompt=ids, n=1, return_token_ids=True,
                           skip_special_tokens=False, **params)
            with requests.Session() as session:
                session.trust_env = False
                response = session.post(self.endpoint + '/completions', json=payload, timeout=(30, 7200))
                if response.status_code != 200:
                    raise RuntimeError('Inference HTTP ' + str(response.status_code))
                result = response.json()
            transport = dict(request_hash=key, request_identity=identity, response=result,
                             elapsed_seconds=time.monotonic()-started)
            # Preserve evidence BEFORE validating token accounting or format.
            write_json(raw_path, transport)
        if result.get('model') != MODEL_NAME or len(result['choices']) != 1:
            raise ValueError('Unexpected inference model or choice count')
        choice = result['choices'][0]
        output_ids = choice.get('token_ids')
        if not isinstance(output_ids, list) or any(type(i) is not int for i in output_ids):
            raise ValueError('Raw output token IDs missing')
        if (result['usage']['prompt_tokens'] != len(ids)
                or result['usage']['completion_tokens'] != len(output_ids)
                or len(output_ids) > self.max_tokens
                or choice['finish_reason'] not in {'length', 'stop'}):
            raise ValueError('Token counts or termination reason differ')
        raw = self.tokenizer.decode(output_ids, skip_special_tokens=False)
        parsed, boundary = split_token_output(self.tokenizer, output_ids, choice['finish_reason'])
        record = dict(request_hash=key, request_identity=identity,
            raw_text=raw, output_token_ids=output_ids, server_text=choice['text'],
            prompt_tokens=len(ids), output_tokens=len(output_ids),
            reasoning_tokens=boundary if boundary is not None else len(output_ids),
            boundary_tokens=1 if boundary is not None else 0,
            after_reasoning_tokens=len(output_ids)-boundary-1 if boundary is not None else 0,
            finish_reason=choice['finish_reason'], elapsed_seconds=transport['elapsed_seconds'],
            stopped_without_final_boundary=boundary is None and choice['finish_reason']=='stop',
            **parsed)
        write_json(path, record)
        with self.lock:
            if recovered:
                self.recovered_raw += 1
            else:
                self.completed += 1
        return record

    def stats(self):
        with self.lock:
            return dict(completed_api_requests=self.completed, cached_requests=self.cached,
                        recovered_raw_requests=self.recovered_raw)
