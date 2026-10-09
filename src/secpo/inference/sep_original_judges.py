"""Provider transport only: SEP decisions remain in the original sep_judge module."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading
import time

from secpo.evaluation.sep_judge import JudgeConfig
from secpo.inference.sep_openrouter import credentials
from secpo.tinker.evaluation_common import ROOT, digest, now, write_json

CONFIG = ROOT/'configs/evaluate_sep_judges.json'
ENDPOINT = 'https://openrouter.ai/api/v1/chat/completions'
ROUTES = {
    'gemini-3.1-flash-lite-preview': dict(model='google/gemini-3.1-flash-lite-preview'),
    'gemini-3-flash-preview-thinking-4096': dict(model='google/gemini-3-flash-preview', reasoning={'max_tokens':4096}),
    'gpt-5-high-reasoning': dict(model='openai/gpt-5', reasoning={'effort':'high'}),
}
ORIGINALS = ('configs/evaluate_sep_judges.json', 'src/secpo/evaluation/sep_judge.py',
             'src/secpo/evaluation/sep_static.py', 'src/secpo/evaluation/witness.py')


def verify_originals():
    import subprocess
    hashes = {}
    for name in ORIGINALS:
        source = (ROOT/name).read_bytes()
        if source != subprocess.check_output(['git','-C',str(ROOT),'show','HEAD:'+name]):
            raise ValueError('Original SEP configuration or evaluator changed: '+name)
        hashes[name] = hashlib.sha256(source).hexdigest()
    return hashes


def protocol():
    config = JudgeConfig.from_path(CONFIG)
    if tuple(ROUTES) != config.judge_identifiers or config.trials_per_judge != 3:
        raise ValueError('Original judge identifiers/trials differ from provider mapping')
    return dict(config=config.to_dict(), endpoint=ENDPOINT, routes=deepcopy(ROUTES),
        provider={'require_parameters':True,'allow_fallbacks':False},
        temperature='omitted: not specified by SEP judge config',
        max_output_tokens='omitted: not specified by SEP judge config',
        aggregation='original evaluate_rows: witness gate then all nine YES',
        original_sources=verify_originals(),
        hosted_revision_note='Historical immutable provider revisions are not retained by the released repo.',
        gemini_budget_note='Send the original 4096 thinkingBudget request via reasoning.max_tokens. Google may internally map Gemini 3 budgets to thinking levels; this is not a guaranteed hard token cap.')


def payload(request):
    if request.judge_identifier not in ROUTES or request.trial_index not in (0,1,2):
        raise ValueError('Unapproved judge or trial')
    return dict(**deepcopy(ROUTES[request.judge_identifier]),
        messages=[dict(role='user', content=request.prompt)],
        provider={'require_parameters':True,'allow_fallbacks':False})


class OriginalJudges:
    def __init__(self, directory, key_file=None, post=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.key = credentials(key_file or ROOT/'secpo_openrouter.env') if post is None else ''
        self.post, self.identity = post, digest(protocol())
        self.lock = threading.Lock()
        self.stats = dict(api_requests_completed=0, cached_requests=0,
            prompt_tokens=0, completion_tokens=0, reported_cost_usd=0.0)

    def __call__(self, request):
        import requests
        body = payload(request)
        identity = digest(dict(protocol=self.identity, request=request.to_dict(), payload=body))
        path = self.directory/(identity+'.json')
        cached = path.exists()
        if cached:
            record = json.loads(path.read_text())
            if record['identity'] != identity or record['request'] != body:
                raise ValueError('Original judge transport cache identity mismatch')
        else:
            start = time.monotonic()
            if self.post is None:
                with requests.Session() as session:
                    session.trust_env = False
                    response = session.post(ENDPOINT, headers={'Authorization':'Bearer '+self.key},
                        json=body, timeout=(30,300))
                if response.status_code != 200:
                    raise RuntimeError(f'OpenRouter HTTP {response.status_code}; response body withheld')
                result = response.json()
            else:
                result = self.post(body)
            record = dict(identity=identity, request=body, judge_request=request.to_dict(),
                response=result, at=now(), elapsed_seconds=time.monotonic()-start)
            # Preserve malformed replies separately, but do not cache them as valid votes.
            try:
                self.validate(result, body['model'])
            except Exception:
                write_json(self.directory/'invalid_responses'/(identity+'.json'), record)
                raise
            write_json(path, record)
        text = self.validate(record['response'], body['model'])
        with self.lock:
            self.stats['cached_requests' if cached else 'api_requests_completed'] += 1
            if not cached:
                usage = record['response'].get('usage') or {}
                self.stats['prompt_tokens'] += usage.get('prompt_tokens') or 0
                self.stats['completion_tokens'] += usage.get('completion_tokens') or 0
                self.stats['reported_cost_usd'] += usage.get('cost') or 0.0
        return text

    @staticmethod
    def validate(result, model):
        if result.get('model') != model or len(result.get('choices',[])) != 1:
            raise ValueError('Unexpected judge model or completion count; no model substitution allowed')
        choice = result['choices'][0]
        text = (choice.get('message') or {}).get('content')
        if choice.get('finish_reason') != 'stop' or not isinstance(text,str) or text.strip().upper() not in ('YES','NO'):
            raise ValueError('Incomplete/non-binary judge response')
        return text
