"""Direct OpenAI Luna transport; retain the original weighted AlpacaEval parser."""
import json
from pathlib import Path
import shlex
import time

from .evaluation_common import digest, now, write_json
from .reference_eval import OpenRouterJudge, JUDGE_PARAMS as ORIGINAL_PARAMS, validate_judge_choice

MODEL = 'gpt-5.6-luna'
ENDPOINT = 'https://api.openai.com/v1'
PARAMS = dict(model_name=MODEL, max_completion_tokens=16, reasoning_effort='none',
              temperature=1, logprobs=True, top_logprobs=5)


def read_credentials(path):
    """Literal assignments only; never execute a user's credential file."""
    values = {}
    try:
        for line in Path(path).read_text().splitlines():
            parts = shlex.split(line, comments=True)
            if parts and parts[0] == 'export':
                parts = parts[1:]
            if not parts:
                continue
            if len(parts) != 1 or '=' not in parts[0]:
                raise ValueError()
            name, value = parts[0].split('=', 1)
            if name in values:
                raise ValueError()
            values[name] = value
    except (OSError, ValueError):
        raise ValueError('Invalid OpenAI credential file; values withheld') from None
    if values.get('OPENAI_BASE_URL', ENDPOINT).rstrip('/') != ENDPOINT:
        raise ValueError('OpenAI keys may only be sent to the official endpoint')
    key = values.get('OPENAI_API_KEY', '').strip()
    if not key or any(c.isspace() for c in key):
        raise ValueError('Missing OpenAI API key')
    return key


def validate_choice(choice):
    # A larger API budget must never turn the classifier into free-text judging.
    content = choice.get('message', {}).get('content')
    tokens = (choice.get('logprobs') or {}).get('content') or []
    if (content not in {'m', 'M'} or choice.get('finish_reason') != 'stop'
            or len(tokens) != 1 or tokens[0].get('token') != content):
        raise ValueError('Luna did not return exactly one completed verdict token')
    return validate_judge_choice(choice)


class LunaJudge(OpenRouterJudge):
    """Reuse tested deduplication/threading only; requests/cache identity are new."""
    def __init__(self, cache_dir, credentials, *, concurrency=4, progress=None, post=None):
        import threading
        self._key = read_credentials(credentials)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.concurrency, self.progress, self._post_override = concurrency, progress, post
        self.lock = threading.Lock()
        self.stats = dict(api_requests_completed=0, cached_requests=0, input_tokens=0,
                          output_tokens=0, reported_cost_usd=None)

    def _post(self, payload):
        if self._post_override is not None:
            return self._post_override(payload)
        import requests
        for attempt in range(5):
            try:
                with requests.Session() as session:
                    session.trust_env = False
                    response = session.post(ENDPOINT + '/chat/completions',
                        headers={'Authorization': 'Bearer ' + self._key}, json=payload,
                        timeout=(15, 90), allow_redirects=False)
                if response.status_code == 200:
                    return response.json()
                status = response.status_code
                if status != 429 and status < 500:
                    raise RuntimeError(f'OpenAI HTTP {status}; body withheld')
            except requests.RequestException:
                status = 'network_error'
            if attempt < 4:
                time.sleep(min(2 ** (attempt + 1), 30))
        raise RuntimeError(f'OpenAI failed after five attempts ({status}); body withheld')

    def _one(self, prompt):
        from alpaca_eval.utils import prompt_to_chatml
        payload = dict(model=MODEL, messages=prompt_to_chatml(prompt.strip()),
                       **{k: v for k, v in PARAMS.items() if k != 'model_name'})
        key = digest(dict(version='secpo-direct-openai-luna-weighted-v1', endpoint=ENDPOINT, request=payload))
        path = self.cache_dir / (key + '.json')
        cached = path.is_file()
        if cached:
            record = json.loads(path.read_text())
            if (record.get('request_hash') != key or record.get('requested_model') != MODEL
                    or record.get('returned_model') != MODEL):
                raise ValueError('Luna cache identity mismatch')
        else:
            start = time.monotonic()
            body = self._post(payload)
            if body.get('model') != MODEL or len(body.get('choices', [])) != 1:
                raise ValueError('Unexpected judge model/completion count')
            choice = body['choices'][0]
            validate_choice(choice)
            usage = body.get('usage') or {}
            record = dict(request_hash=key, created_at=now(), requested_model=MODEL,
                returned_model=body['model'], response_id=body.get('id'), choice=choice,
                usage={k: usage.get(k) for k in ('prompt_tokens', 'completion_tokens',
                    'prompt_tokens_details', 'completion_tokens_details')},
                elapsed_seconds=time.monotonic() - start)
            write_json(path, record)
        validate_choice(record['choice'])
        with self.lock:
            self.stats['cached_requests' if cached else 'api_requests_completed'] += 1
            if not cached:
                self.stats['input_tokens'] += record['usage'].get('prompt_tokens') or 0
                self.stats['output_tokens'] += record['usage'].get('completion_tokens') or 0
            if self.progress:
                self.progress(dict(self.stats))
        return record

    def __call__(self, prompts, **kwargs):
        if kwargs != PARAMS:
            raise ValueError('Unexpected Luna decoder settings')
        # The parent's concurrent dispatcher only checks this dictionary; actual
        # payloads use PARAMS above, and do not contact OpenRouter.
        return super().__call__(prompts, **ORIGINAL_PARAMS)
