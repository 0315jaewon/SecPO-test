"""Tinker inference for clean AgentDojo and AgentDyn utility episodes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from functools import partial
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

from secpo.evaluation.agentdyn_qwen36 import messages_to_qwen_input_roles
from .evaluation_common import MODEL, ROOT, CheckpointSampler, digest


BENCHMARKS = {
    "agentdojo": {"commit": "d3640b5b03a88eb44dad96852e1de1ef437d836e", "version": "v1.2.1",
                  "suites": {"banking": 16, "slack": 21, "travel": 20, "workspace": 40}},
    "agentdyn": {"commit": "5353cf7615b135cace8d07c8f12dac53a16b6db3", "version": "v1.2.2",
                 "suites": {"shopping": 20, "github": 20, "dailylife": 20}},
}


def verify_source(benchmark: str) -> dict:
    """Verify the pinned source tree, excluding upstream historical run archives.

    The sparse checkout contains source and root metadata. AgentDyn receives
    only the repository's reviewed patch. Its expected source is independently
    materialized in a temporary directory and compared to the installed code.
    """
    import agentdojo

    config = BENCHMARKS[benchmark]
    checkout = ROOT / "runtime/benchmarks" / benchmark
    def git(*args, text=True):
        return subprocess.check_output(["git", "-C", str(checkout), *args], text=text)
    if git("rev-parse", "HEAD").strip() != config["commit"]:
        raise ValueError("Benchmark checkout revision differs from the pinned source")
    entries = git("ls-tree", "-r", "HEAD", "--", "src/agentdojo", "LICENSE").splitlines()
    with tempfile.TemporaryDirectory(prefix="secpo-source-") as temp:
        expected_root = Path(temp)
        for entry in entries:
            metadata, path = entry.split("\t", 1)
            mode, kind, oid = metadata.split()
            if kind != "blob" or mode not in {"100644", "100755"}:
                raise ValueError("Unexpected benchmark source object")
            target = expected_root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(git("cat-file", "blob", oid, text=False))
        if benchmark == "agentdyn":
            patch = ROOT / "third_party/patches/agentdyn.patch"
            if hashlib.sha256(patch.read_bytes()).hexdigest() != "43a8e090daa08854d93cc8ea613d203354b004da645253b90380b3e59a7075f6":
                raise ValueError("AgentDyn patch checksum changed")
            subprocess.run(["git", "apply", str(patch)], cwd=expected_root, check=True)
        installed = Path(agentdojo.__file__).resolve().parent
        expected = expected_root / "src/agentdojo"
        hashes = {}
        for path in expected.rglob("*"):
            # macOS Finder metadata is omitted by the upstream wheel builder.
            if path.is_file() and path.name != ".DS_Store" and "__pycache__" not in path.parts:
                relative = path.relative_to(expected)
                observed = installed / relative
                if not observed.is_file() or path.read_bytes() != observed.read_bytes():
                    raise ValueError(f"Installed benchmark source mismatch: {relative}")
                hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for path in installed.rglob("*.py"):
            if str(path.relative_to(installed)) not in hashes:
                raise ValueError("Additional Python source in benchmark installation")
    return {"commit":config["commit"], "source_sha256":digest(hashes), "verified_files":len(hashes),
            "verification_scope":"src/agentdojo and reviewed patch; historical run archives excluded"}


def trust_role_template(tokenizer) -> str:
    """Extend the pinned native Qwen3.8 template for SecPO trust roles."""
    template = tokenizer.chat_template
    needle = 'message.role == "user"'
    if template.count(needle) != 2 or "<function=" not in template:
        raise ValueError("Unexpected native Qwen3.8 tool template")
    template = template.replace(needle, '(message.role == "user" or message.role == "input")')
    template = template.replace("<|im_start|>system", "<|im_start|>user")
    return template


def render_agent_prompt(tokenizer, messages, tools):
    converted = messages_to_qwen_input_roles(messages)
    # The older converter serializes arguments for OpenAI-compatible servers;
    # the native Qwen3.8 XML template expects mappings when rendering history.
    for message in converted:
        for call in message.get("tool_calls", []) or []:
            function = call["function"]
            if isinstance(function.get("arguments"), str):
                function["arguments"] = json.loads(function["arguments"])
    return tokenizer.apply_chat_template(
        converted, tools=tools, chat_template=trust_role_template(tokenizer), tokenize=False,
        add_generation_prompt=True, enable_thinking=False, preserve_thinking=False,
    )


@dataclass(frozen=True)
class ToolCall:
    function: str
    args: dict[str, Any]
    id: str


def parse_tool_response(text: str, tools: list[dict]) -> tuple[str, list[ToolCall]]:
    """Parse native Qwen XML calls using each function's parameter schema."""
    schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}
    calls = []
    for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
        body = match.group(1)
        function = re.fullmatch(r"<function\s*=\s*([^>]+)>\s*(.*?)\s*</function>", body.strip(), re.DOTALL)
        if function is None:
            raise ValueError("Malformed Qwen XML tool call")
        name = function.group(1).strip()
        properties = schemas.get(name, {}).get("properties", {})
        arguments = {}
        for parameter in re.finditer(r"<parameter\s*=\s*([^>]+)>\s*(.*?)\s*</parameter>", function.group(2), re.DOTALL):
            key, raw = parameter.group(1).strip(), parameter.group(2).strip()
            spec = properties.get(key, {})
            types = {spec.get("type"), *(item.get("type") for item in spec.get("anyOf", []))}
            if "string" in types:
                value = raw
            else:
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError:
                    value = raw
            arguments[key] = value
        calls.append(ToolCall(name, arguments, "call_" + digest([len(calls),body])[:24]))
    if "<tool_call>" in text and not calls:
        raise ValueError("Incomplete Qwen tool call")
    visible = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()
    return visible, calls


def make_pipeline(sampler: CheckpointSampler, *, sampling_params=None):
    from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
    from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
    from agentdojo.functions_runtime import FunctionCall
    from agentdojo.types import text_content_block_from_string

    class TinkerAgent(BasePipelineElement):
        name = ("Qwen3.8-27B-Base-Tinker-thinking-off" if sampler.is_base_model
                else "Qwen3.8-27B-SecPO-Tinker-thinking-off")

        def query(self, query, runtime, env, messages=(), extra_args=None):
            tools = [{"type":"function", "function":{
                "name":f.name, "description":f.description, "parameters":f.parameters.model_json_schema()
            }} for f in runtime.functions.values()]
            prompt = render_agent_prompt(sampler.tokenizer, messages, tools)
            params = sampling_params or dict(temperature=1.0, top_p=0.95, top_k=20)
            response = sampler.generate(prompt, max_tokens=8192, seed=0, **params)
            content, calls = parse_tool_response(response.text, tools)
            message = {"role":"assistant",
                       "content":[text_content_block_from_string(content)] if content else None,
                       "tool_calls":[FunctionCall(function=c.function, args=c.args, id=c.id) for c in calls]}
            return query, runtime, env, [*messages, message], dict(extra_args or {})

    config_values = dict(llm=TinkerAgent(), defense="repeat_user_prompt", system_message_name=None,
                         system_message=None, tool_output_format="json")
    # AgentDyn has additional config fields; the older AgentDojo version does not.
    for name, value in {"tool_delimiter":"input", "model_id":MODEL, "reasoning_effort":None}.items():
        if name in PipelineConfig.model_fields:
            config_values[name] = value
    pipeline = AgentPipeline.from_config(PipelineConfig(**config_values))
    from agentdojo.agent_pipeline.tool_execution import ToolsExecutor, tool_result_to_str

    def configure_formatters(element):
        if isinstance(element, ToolsExecutor):
            element.output_formatter = partial(tool_result_to_str, dump_fn=tool_json_dumps)
        for child in getattr(element, "elements", ()):
            configure_formatters(child)

    configure_formatters(pipeline)
    return pipeline


def tool_json_dumps(value):
    """Keep upstream JSON formatting, adding ISO serialization for date values.

    Both pinned versions dump Pydantic models in Python mode, which exposes
    datetime objects. Plain json.dumps crashes on those valid tool results.
    This does not change the simulated tool state or utility predicates.
    """
    def default(item):
        if isinstance(item, (date, datetime)):
            return item.isoformat()
        raise TypeError(f"Unsupported tool result value: {type(item).__name__}")
    return json.dumps(value, default=default)
