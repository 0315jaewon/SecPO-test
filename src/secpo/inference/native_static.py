"""Launch/configuration glue for unchanged SecPO native-tool evaluators."""
from dataclasses import replace
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from secpo.tinker.evaluation_common import ROOT, MODEL, TOKENIZER_REVISION, digest, now, write_json
from secpo.tinker.agent_eval import BENCHMARKS, verify_source, tool_json_dumps
from secpo.inference.static_agent_eval import INJECTION_COUNTS, read, inference_config, summarize

OUTPUT = Path('/data2/jaewonc/secpo_evals/qwen38_native_static_20260924')
LEGACY = Path('/data2/jaewonc/secpo_evals/qwen38_lambda_static_20260924')
VARIANTS = ('secpo_lr4e5', 'secpo')
ENDPOINT = 'http://127.0.0.1:18781/v1'
ORIGINALS = ('src/secpo/evaluation/agentdojo_qwen36.py',
             'src/secpo/evaluation/agentdyn_qwen36.py')
FILES = (*ORIGINALS, 'src/secpo/subprocess_env.py',
    'src/secpo/inference/native_static.py', 'scripts/evaluate_native_static.py',
    'scripts/launch_native_static.py', 'scripts/native_static_remote.py',
    'src/secpo/inference/static_agent_eval.py', 'src/secpo/inference/lambda_eval.py',
    'src/secpo/tinker/agent_eval.py', 'src/secpo/tinker/evaluation_common.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_originals():
    result = {}
    for name in ORIGINALS:
        committed = subprocess.check_output(['git', '-C', str(ROOT), 'show', 'HEAD:' + name])
        if committed != (ROOT / name).read_bytes():
            raise ValueError('Original SecPO evaluator differs from git HEAD: ' + name)
        result[name] = sha(ROOT / name)
    return result


def served(variant):
    if variant not in VARIANTS:
        raise ValueError('Unapproved model variant')
    return 'secpo-native-' + variant


def template():
    """Configure Qwen3.8 trust roles without changing evaluator or parser code."""
    base = Path('/data2/jaewonc/secpo_backups/lambda_20260919/models/base')
    source = (base / 'chat_template.jinja').read_text()
    expected = read(Path('/data2/jaewonc/secpo_evals/qwen38_thinking_alpaca_20260923/model_validation.json'))
    identity = {r['path'].rsplit('/', 1)[-1]: r['sha256'] for r in expected['files']}
    if sha(base / 'chat_template.jinja') != identity['chat_template.jinja']:
        raise ValueError('Qwen3.8 template checksum differs')
    needle = 'message.role == "user"'
    if source.count(needle) != 2:
        raise ValueError('Unexpected Qwen3.8 template')
    configured = source.replace(needle, '(message.role == "user" or message.role == "input")')
    configured = configured.replace('<|im_start|>system', '<|im_start|>user')
    # AgentDojo emits native role=tool; AgentDyn's original converter emits
    # role=input. Both must render untrusted tool results as input, never user.
    prefix, marker, suffix = configured.partition('{%- elif message.role == "tool" %}')
    if not marker or suffix.count('<|im_start|>user') != 1:
        raise ValueError('Unexpected native tool-result branch')
    return prefix + marker + suffix.replace('<|im_start|>user', '<|im_start|>input')


def configure_environment(benchmark, variant):
    name = served(variant)
    os.environ['AGENTDOJO_BASE_URL'] = ENDPOINT
    # These are the original AgentDyn adapter's supported runtime variables.
    # The paper-specific Qwen3.6 dataclass/CLI is not patched or instantiated.
    values = dict(BASE_URL=ENDPOINT, CREDENTIAL_ENV='SECPO_NATIVE_UNUSED_CREDENTIAL',
        ENABLE_THINKING='false', PRESERVE_THINKING='false', MAX_TOKENS='8192',
        TEMPERATURE='0.7', TOP_P='0.8', TOP_K='20', PRESENCE_PENALTY='1.5',
        SEED='0', REQUEST_TIMEOUT='1800', REQUEST_MAX_ATTEMPTS='3', RETRY_MAX_WAIT_SECONDS='40')
    for key, value in values.items():
        os.environ['AGENTDYN_LOCAL_' + key] = value
    from agentdojo.models import MODEL_NAMES
    MODEL_NAMES[name] = 'Qwen 3'
    if benchmark == 'agentdyn':
        from secpo.evaluation.agentdyn_qwen36 import install
        install()


def pipeline(benchmark, variant):
    """Use original request construction and original response parsing."""
    name = served(variant)
    if benchmark == 'agentdojo':
        from secpo.evaluation.agentdojo_qwen36 import AgentDojoConfig, _build_pipeline
        config = replace(AgentDojoConfig(), model_name=name, model_revision=TOKENIZER_REVISION,
            temperature=.7, top_p=.8, top_k=20, presence_penalty=1.5,
            max_output_tokens=8192, enable_thinking=False, preserve_thinking=False)
        result = _build_pipeline(config)
    else:
        from agentdojo.agent_pipeline import agent_pipeline
        adapter = agent_pipeline.get_llm('local', 'local', name, 'input')
        adapter.name = name
        config = agent_pipeline.PipelineConfig(llm=adapter, model_id=name,
            defense='repeat_user_prompt', tool_delimiter='input', reasoning_effort=None,
            system_message_name=None, system_message=None, tool_output_format='json')
        result = agent_pipeline.AgentPipeline.from_config(config)
    result.name = name + '-repeat_user_prompt'
    # Existing compatibility serializer: only date/datetime -> ISO strings.
    # Ordinary JSON formatting and all benchmark tool functions are unchanged.
    from agentdojo.agent_pipeline.tool_execution import ToolsExecutor, tool_result_to_str
    for element in elements(result):
        if isinstance(element, ToolsExecutor):
            element.output_formatter = partial(tool_result_to_str, dump_fn=tool_json_dumps)
    return result


def elements(root):
    seen = set()
    def walk(element):
        if id(element) in seen:
            return
        seen.add(id(element))
        yield element
        for child in getattr(element, 'elements', ()):
            yield from walk(child)
    return list(walk(root))


class ChatCache:
    """Record/replay the original adapters' HTTP calls without interpreting tools."""
    def __init__(self, directory, variant, manifest_hash):
        self.directory, self.variant, self.identity = Path(directory), variant, manifest_hash
        self.lock, self.key_locks = threading.Lock(), {}
        self.requests = self.cached = self.output_tokens = self.input_tokens = 0

    def attach(self, agent):
        for element in elements(agent):
            client = getattr(element, 'client', None)
            if client is None:
                continue
            if str(client.base_url).rstrip('/') != ENDPOINT:
                raise ValueError('Unexpected endpoint')
            original = client.chat.completions.create
            def create(_original=original, **request):
                from openai.types.chat import ChatCompletion
                self.validate(request)
                key = digest(dict(manifest=self.identity, request=request))
                path = self.directory / (key + '.json')
                with self.lock:
                    guard = self.key_locks.setdefault(key, threading.Lock())
                with guard:
                    if path.exists():
                        row = read(path)
                        if row['request'] != request or row['identity'] != self.identity:
                            raise ValueError('Native completion cache mismatch')
                        with self.lock:
                            self.cached += 1
                        return ChatCompletion.model_validate(row['response'])
                    start = time.monotonic()
                    response = _original(**request)
                    if response.model != served(self.variant):
                        raise ValueError('Server returned wrong model')
                    write_json(path, dict(identity=self.identity, request=request,
                        response=response.model_dump(mode='json'), at=now(),
                        elapsed_seconds=time.monotonic()-start))
                    with self.lock:
                        self.requests += 1
                        if response.usage:
                            self.input_tokens += response.usage.prompt_tokens
                            self.output_tokens += response.usage.completion_tokens
                    return response
            client.chat.completions.create = create

    def validate(self, request):
        expected = dict(model=served(self.variant), max_tokens=8192, temperature=.7,
                        top_p=.8, presence_penalty=1.5, seed=0)
        if any(request.get(k) != v for k, v in expected.items()):
            raise ValueError('Native request configuration changed')
        extra = request['extra_body']
        if extra != dict(top_k=20, chat_template_kwargs=dict(enable_thinking=False, preserve_thinking=False)):
            raise ValueError('Native thinking/tool settings changed')
        if request.get('tool_choice') != 'auto' or not request.get('tools'):
            raise ValueError('Native tool schemas/auto choice absent')

    def stats(self):
        with self.lock:
            return dict(completed_api_requests=self.requests, cached_requests=self.cached,
                        input_tokens=self.input_tokens, output_tokens=self.output_tokens)
