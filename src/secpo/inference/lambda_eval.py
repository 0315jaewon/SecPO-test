"""Pinned Lambda/vLLM inference, retaining the original evaluation protocols."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlparse

from secpo.tinker.evaluation_common import (
    ROOT, MODEL, TOKENIZER_REVISION, RENDER_VERSION, CheckpointSampler, Generation,
    cut_at_stop, digest, evaluation_model_id, evaluation_output_root, write_json,
)
from secpo.tinker.evaluation_variants import sampler_model_kwargs

QWEN_SAMPLING = dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0,
                     presence_penalty=1.5, repetition_penalty=1.0, frequency_penalty=0.0)
AGENT_BENCHMARKS = {'agentdojo', 'agentdyn'}


def load_config(path, variant):
    config = json.loads(Path(path).read_text())
    url = urlparse(config['endpoint'])
    if (url.scheme != 'http' or url.hostname != '127.0.0.1' or url.username or url.password
            or url.path != '/v1' or url.query or url.fragment):
        raise ValueError('Lambda inference must use a loopback SSH tunnel')
    if (variant not in {'base', 'secpo'} or config['variant'] != variant
            or config['checkpoint'] != evaluation_model_id(variant)
            or config['base_revision'] != TOKENIZER_REVISION
            or config['served_model'] != 'secpo-suite-' + variant
            or config['sampling'] != QWEN_SAMPLING
            or config['backend'] != 'vllm-0.29.0'
            or config['max_model_len'] != 65536):
        raise ValueError('Unexpected Lambda model or sampling configuration')
    if variant == 'secpo' and not config.get('adapter_sha256'):
        raise ValueError('SecPO adapter identity is required')
    return config


def configure_protocol(manifest, path, variant):
    """Verify the old protocol BEFORE applying only the requested sampling change."""
    config = load_config(path, variant)
    benchmark = manifest['benchmark']
    agent = benchmark in AGENT_BENCHMARKS
    prior_path = evaluation_output_root(variant, None if agent else 16384) / benchmark / 'manifest.json'
    prior = json.loads(prior_path.read_text())
    stable = lambda m: {k: v for k, v in m.items() if k != 'implementation_sha256'}
    if stable(prior) != stable(manifest):
        changed = sorted(k for k in set(prior) | set(manifest)
                         if k != 'implementation_sha256' and prior.get(k) != manifest.get(k))
        raise ValueError(f'Unexpected protocol changes before Lambda sampling: {changed}')
    old_hashes = prior.get('implementation_sha256', {})
    for name in ('src/secpo/tinker/reference_eval.py', 'src/secpo/evaluation/reference_utility.py'):
        if name in old_hashes and old_hashes[name] != manifest['implementation_sha256'].get(name):
            raise ValueError('Judge or scoring implementation changed')
    manifest['inference'] = config
    manifest['comparison_manifest_sha256'] = digest(prior)
    if agent:
        manifest.update(QWEN_SAMPLING)
    elif 'generation' in manifest:
        manifest['generation'].update(QWEN_SAMPLING)
    else:
        for task in manifest['tasks'].values():
            task['generation_kwargs'] = {**task['generation_kwargs'], **QWEN_SAMPLING, 'do_sample': True}
    return config


def make_sampler(cache_dir, *, tokenizer, variant, config_path=None):
    if config_path is None:
        return CheckpointSampler(cache_dir, tokenizer=tokenizer, **sampler_model_kwargs(variant))
    return LambdaSampler(cache_dir, tokenizer=tokenizer, config=load_config(config_path, variant))


class LambdaSampler:
    """Send pinned prompt token IDs; store raw returned token IDs and exact settings."""
    def __init__(self, cache_dir, *, tokenizer, config):
        import requests
        self.config = deepcopy(config)
        self.tokenizer = tokenizer
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint = config['checkpoint']
        self.is_base_model = config['variant'] == 'base'
        self.completed = self.cached = self.input_tokens = self.output_tokens = 0
        self.lock = threading.Lock()
        self.key_locks = {}
        with requests.Session() as session:
            session.trust_env = False
            response = session.get(config['endpoint'] + '/models', timeout=30)
            response.raise_for_status()
            models = [row['id'] for row in response.json()['data']]
            if config['served_model'] not in models:
                raise ValueError('Lambda is serving the wrong model')

    def generate(self, prompt, *, max_tokens, stops=(), seed=42, **sampling):
        import requests
        if sampling != QWEN_SAMPLING:
            raise ValueError(f'Expected explicit Qwen sampling profile, got keys {sorted(sampling)}')
        if type(max_tokens) is not int or max_tokens not in {8192, 16384}:
            raise ValueError('Unapproved output cap')
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) + max_tokens > self.config['max_model_len']:
            raise ValueError('Prompt plus output exceeds context budget')
        stop_list = list(dict.fromkeys([*stops, '<|im_end|>', '<|endoftext|>']))
        params = dict(max_tokens=max_tokens, stop=stop_list, seed=seed, **sampling)
        identity = dict(inference=self.config, render_version=RENDER_VERSION,
                        tokenizer_revision=TOKENIZER_REVISION, prompt=prompt, params=params)
        key = digest(identity)
        path = self.cache_dir / (key + '.json')
        with self.lock:
            key_lock = self.key_locks.setdefault(key, threading.Lock())
        with key_lock:
            if path.exists():
                cached = json.loads(path.read_text())
                if cached['request_hash'] != key:
                    raise ValueError('Completion cache identity mismatch')
                with self.lock:
                    self.cached += 1
                return Generation(**cached['generation'])
            start = time.monotonic()
            payload = dict(model=self.config['served_model'], prompt=tokens, n=1,
                           return_token_ids=True, skip_special_tokens=False, **params)
            with requests.Session() as session:
                session.trust_env = False
                response = session.post(self.config['endpoint'] + '/completions',
                                        json=payload, timeout=(30, 3600))
                if response.status_code != 200:
                    raise RuntimeError(f'Lambda inference failed with HTTP {response.status_code}')
                result = response.json()
            if result.get('model') != self.config['served_model'] or len(result['choices']) != 1:
                raise ValueError('Unexpected inference response identity')
            choice = result['choices'][0]
            output_ids = choice.get('token_ids')
            if not isinstance(output_ids, list) or not all(type(t) is int for t in output_ids):
                raise ValueError('Missing raw generation token IDs')
            if (result['usage']['prompt_tokens'] != len(tokens)
                    or result['usage']['completion_tokens'] != len(output_ids)
                    or choice['finish_reason'] not in {'length', 'stop'}):
                raise ValueError('Unexpected inference token accounting or finish reason')
            text = cut_at_stop(self.tokenizer.decode(output_ids, skip_special_tokens=False), stop_list)
            generation = Generation(text, len(tokens), len(output_ids), choice['finish_reason'],
                                    time.monotonic() - start)
            write_json(path, dict(request_hash=key, generation=asdict(generation),
                       params=params, prompt_sha256=digest(prompt), prompt_token_ids=tokens,
                       output_token_ids=output_ids, server_text=choice['text']))
            with self.lock:
                self.completed += 1
                self.input_tokens += len(tokens)
                self.output_tokens += len(output_ids)
            return generation

    def stats(self):
        with self.lock:
            return dict(completed_api_requests=self.completed, cached_requests=self.cached,
                        input_tokens=self.input_tokens, output_tokens=self.output_tokens)
