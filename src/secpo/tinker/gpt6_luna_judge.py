"""Isolated GPT-6 Luna judge; leave the running GPT-5.6 experiment unchanged."""
import json
import time

from .evaluation_common import digest, now, write_json
from .luna_judge import ENDPOINT, LunaJudge, validate_choice
from .reference_eval import OpenRouterJudge, JUDGE_PARAMS as DISPATCH_PARAMS

MODEL = 'gpt-6-luna'
PARAMS = dict(model_name=MODEL, max_completion_tokens=16, reasoning_effort='none',
              temperature=1, logprobs=True, top_logprobs=5)


class GPT6LunaJudge(LunaJudge):
    """Reuse credential handling/official transport, not the older request identity."""

    def _one(self, prompt):
        from alpaca_eval.utils import prompt_to_chatml
        payload = dict(model=MODEL, messages=prompt_to_chatml(prompt.strip()),
                       **{k: v for k, v in PARAMS.items() if k != 'model_name'})
        key = digest(dict(version='secpo-direct-openai-gpt6-luna-weighted-v1',
                          endpoint=ENDPOINT, request=payload))
        path = self.cache_dir / (key + '.json')
        cached = path.is_file()
        if cached:
            record = json.loads(path.read_text())
            if (record.get('request_hash') != key or record.get('requested_model') != MODEL
                    or record.get('returned_model') != MODEL):
                raise ValueError('GPT-6 Luna cache identity mismatch')
        else:
            start = time.monotonic()
            body = self._post(payload)
            if body.get('model') != MODEL or len(body.get('choices', [])) != 1:
                raise ValueError('Unexpected judge model/completion count')
            choice = body['choices'][0]
            # Save failed verdicts separately for diagnosis, never as valid cache.
            try:
                validate_choice(choice)
            except ValueError:
                write_json(self.cache_dir / 'invalid_responses' / (key + '.json'),
                           dict(request_hash=key, requested_model=MODEL,
                                choice=choice, usage=body.get('usage'), at=now()))
                raise
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
            raise ValueError('Unexpected GPT-6 Luna decoder settings')
        # The shared dispatcher checks its original config; _one sends PARAMS.
        return OpenRouterJudge.__call__(self, prompts, **DISPATCH_PARAMS)
