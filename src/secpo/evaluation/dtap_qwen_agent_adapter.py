"""DTap adapter that runs Qwen-Agent while preserving Qwen's ``input`` role.

Qwen-Agent's public message schema has no literal ``input`` role.  Trust is
therefore retained in ``Message.extra`` inside the framework.  Only the final
OpenAI-compatible transport conversion maps that provenance to private message
names.  The companion vLLM template renders those tagged messages as actual
``<|im_start|>input`` blocks.

The Qwen-Agent framework owns the function-calling loop, message schema, tool
registry, and raw OpenAI-compatible model adapter. DTap continues to own MCP
server lifecycle, hooks, injected-skill provenance, and judge-compatible
trajectories. The model service address is mandatory runtime configuration;
this module contains no deployment endpoint.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import json
import math
import os
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence, Union

from openai import BadRequestError, OpenAI
from qwen_agent.agents import FnCallAgent
from qwen_agent.llm.oai import TextChatAtOAI
from qwen_agent.llm.schema import (
    ASSISTANT,
    USER,
    ContentItem,
    Message,
)
from qwen_agent.tools.base import BaseTool

from agent.qwen_input.src.agent import (
    MIN_OUTPUT_TOKENS,
    TOOL_INJECTION_BOUNDARY,
    TRUSTED_TOOL_RESULT_NAME,
    UNTRUSTED_MESSAGE_NAME,
    UNTRUSTED_TOOL_RESULT_NAME,
    QwenInputRoleAgent,
    _env_bool,
    _fallback_tool_calls,
)
from dt_arena.src.types.agent import (
    Agent,
    AgentConfig,
    AgentResult,
    RuntimeConfig,
)
from dt_arena.src.types.trajectory import Trajectory


QWEN_AGENT_VERSION = "0.0.34"
FRAMEWORK_NAME = "qwen_agent_input"
TRANSPORT_TRUST_KEY = "dtap_trust"
MALFORMED_TOOL_NAME_MAX_CHARS = 128
MALFORMED_TOOL_ARGUMENT_PREVIEW_MAX_CHARS = 512
MALFORMED_TOOL_HISTORY_MAX_CHARS = 4096
MALFORMED_TOOL_ERROR_MAX_CHARS = 1024
CONTEXT_RECOVERY_MIN_OUTPUT_TOKENS = 1
CONTEXT_RECOVERY_POLICY = (
    "transport_only_recognized_vllm_400_max_tokens_halving_floor_1_v2"
)
DEFAULT_MCP_CLIENT_TIMEOUT_SECONDS = 30.0


def _positive_env_float(name: str, default: float) -> float:
    """Read finite positive runtime configuration without persisting it."""
    raw_value = os.getenv(name)
    value = default if raw_value is None else float(raw_value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


class Trust(str, Enum):
    """Trust classes retained inside Qwen-Agent messages."""

    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"
    TRUSTED_TOOL = "trusted_tool"
    UNTRUSTED_TOOL = "untrusted_tool"


def _to_qwen_content(
    content: Union[str, List[Dict[str, Any]], List[ContentItem]],
) -> Union[str, List[ContentItem]]:
    """Convert DTap/OpenAI content parts to Qwen-Agent content items."""
    if isinstance(content, str):
        return content
    items: List[ContentItem] = []
    for item in content:
        if isinstance(item, ContentItem):
            items.append(item)
            continue
        if not isinstance(item, dict):
            raise TypeError(f"Unsupported DTap content item: {type(item).__name__}")
        item_type = item.get("type")
        if item_type == "text" or "text" in item:
            items.append(ContentItem(text=str(item.get("text") or "")))
            continue
        if item_type in {"image", "image_url"} or "image" in item or "image_url" in item:
            image = item.get("image")
            if image is None:
                image = item.get("image_url")
            if isinstance(image, dict):
                image = image.get("url")
            if not isinstance(image, str) or not image:
                raise ValueError("Image content item has no URL")
            items.append(ContentItem(image=image))
            continue
        raise ValueError(f"Unsupported DTap content item type: {item_type!r}")
    return items


def qwen_message(
    content: Union[str, List[Dict[str, Any]], List[ContentItem]],
    trust: Trust,
) -> Message:
    """Create a Qwen-Agent user message with mandatory trust provenance."""
    return Message(
        role=USER,
        content=_to_qwen_content(content),
        extra={TRANSPORT_TRUST_KEY: trust.value},
    )


def _content_to_openai(content: Any) -> Union[str, List[Dict[str, Any]]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise TypeError(f"Unsupported Qwen-Agent content: {type(content).__name__}")
    parts: List[Dict[str, Any]] = []
    for item in content:
        payload = item.model_dump() if hasattr(item, "model_dump") else item
        if not isinstance(payload, dict):
            raise TypeError(
                f"Unsupported Qwen-Agent content item: {type(item).__name__}"
            )
        if payload.get("text") is not None:
            parts.append({"type": "text", "text": str(payload["text"])})
        elif payload.get("image"):
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": str(payload["image"])},
                }
            )
        else:
            raise ValueError(
                "Only text and image Qwen-Agent content is supported by DTap"
            )
    return parts


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    rendered = _content_to_openai(content)
    if isinstance(rendered, str):
        return rendered
    text: List[str] = []
    for item in rendered:
        if item["type"] != "text":
            raise ValueError("Assistant output cannot contain non-text content")
        text.append(str(item["text"]))
    return "".join(text)


def _parse_tool_arguments(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        parsed = raw
    else:
        try:
            parsed = json.loads(str(raw or "{}"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid tool arguments: {raw!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Tool arguments must decode to a JSON object")
    return parsed


def _bounded_text(value: Any, max_chars: int) -> str:
    """Render a model-provided value without allowing an unbounded echo."""
    if isinstance(value, str):
        rendered = value
    else:
        try:
            rendered = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            rendered = repr(value)
    if len(rendered) <= max_chars:
        return rendered
    suffix = "…[truncated]"
    if max_chars <= len(suffix):
        return suffix[:max_chars]
    return f"{rendered[: max_chars - len(suffix)]}{suffix}"


def _malformed_tool_error(name: str, raw_arguments: Any) -> str:
    """Build the trusted, bounded retry guidance for malformed tool JSON."""
    bounded_name = _bounded_text(name, MALFORMED_TOOL_NAME_MAX_CHARS)
    bounded_arguments = _bounded_text(
        raw_arguments,
        MALFORMED_TOOL_ARGUMENT_PREVIEW_MAX_CHARS,
    )
    message = (
        "Tool call "
        f"{json.dumps(bounded_name, ensure_ascii=False)} was not executed because "
        "its arguments were not one valid JSON object. Malformed arguments preview: "
        f"{json.dumps(bounded_arguments, ensure_ascii=False)}. Correct the JSON and "
        "try the tool call again."
    )
    return _bounded_text(message, MALFORMED_TOOL_ERROR_MAX_CHARS)


def _is_vllm_context_overflow(error: BadRequestError) -> bool:
    """Recognize only the vLLM 400 emitted for input+output overflow."""
    if getattr(error, "status_code", None) != 400:
        return False
    pieces = [str(error)]
    body = getattr(error, "body", None)
    if body is not None:
        try:
            pieces.append(json.dumps(body, ensure_ascii=False))
        except (TypeError, ValueError):
            pieces.append(repr(body))
    rendered = "\n".join(pieces).lower()
    return (
        "maximum context length" in rendered
        and "output tokens" in rendered
        and ("prompt contains" in rendered or "input tokens" in rendered)
    )


def _context_recovery_budgets(requested: Any) -> List[int]:
    """Return the bounded deterministic search for remaining context budget."""
    if isinstance(requested, bool) or not isinstance(requested, int):
        return []
    budgets: List[int] = []
    candidate = requested
    while candidate > CONTEXT_RECOVERY_MIN_OUTPUT_TOKENS:
        candidate = max(CONTEXT_RECOVERY_MIN_OUTPUT_TOKENS, candidate // 2)
        budgets.append(candidate)
    return budgets


def _render_tool_call_xml(name: str, raw_arguments: Any) -> str:
    try:
        arguments = _parse_tool_arguments(raw_arguments)
    except ValueError:
        # The malformed assistant turn must remain renderable on the retry turn.
        # JSON-string encoding keeps the raw preview inert inside the history while
        # retaining enough of it for the model to correct its own output.
        bounded_name = _bounded_text(name, MALFORMED_TOOL_NAME_MAX_CHARS)
        bounded_arguments = _bounded_text(
            raw_arguments,
            MALFORMED_TOOL_HISTORY_MAX_CHARS,
        )
        return "\n".join(
            [
                "<tool_call>",
                f"<function={bounded_name}>",
                "<malformed_arguments>",
                json.dumps(bounded_arguments, ensure_ascii=False),
                "</malformed_arguments>",
                "</function>",
                "</tool_call>",
            ]
        )
    lines = ["<tool_call>", f"<function={name}>"]
    for argument_name, argument_value in arguments.items():
        rendered = (
            argument_value
            if isinstance(argument_value, str)
            else json.dumps(argument_value, ensure_ascii=False)
        )
        lines.extend(
            [f"<parameter={argument_name}>", rendered, "</parameter>"]
        )
    lines.extend(["</function>", "</tool_call>"])
    return "\n".join(lines)


class DTapInputRoleOAI(TextChatAtOAI):
    """Qwen-Agent raw OAI transport with a fail-closed trust converter."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__(cfg)
        base_url = str(
            cfg.get("model_server")
            or cfg.get("base_url")
            or cfg.get("api_base")
            or ""
        ).strip()
        api_key = str(cfg.get("api_key") or "EMPTY").strip()
        request_timeout = float(cfg.get("request_timeout", 1800))
        self._dtap_client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=request_timeout,
            max_retries=0,
        )
        # TextChatAtOAI calls this hook for both raw streaming and regular chat.
        self._chat_complete_create = self._dtap_chat_complete_create

    @property
    def support_multimodal_input(self) -> bool:
        return True

    def _preprocess_messages(
        self,
        messages: List[Message],
        lang: Literal["en", "zh"],
        generate_cfg: dict,
        functions: Optional[List[Dict[str, Any]]] = None,
        use_raw_api: bool = False,
    ) -> List[Message]:
        del lang, generate_cfg, functions
        if not use_raw_api:
            raise RuntimeError("DTap Qwen-Agent transport requires use_raw_api=True")
        # Qwen-Agent's stock multimodal preprocessing may rewrite user messages
        # and discard their names.  DTap content is already normalized, so a
        # provenance-preserving copy is the only safe preprocessing here.
        return copy.deepcopy(messages)

    def _dtap_chat_complete_create(self, *args: Any, **kwargs: Any) -> Any:
        extra_params = ("top_k", "repetition_penalty")
        if any(key in kwargs for key in extra_params):
            kwargs["extra_body"] = copy.deepcopy(kwargs.get("extra_body", {}))
            for key in extra_params:
                if key in kwargs:
                    kwargs["extra_body"][key] = kwargs.pop(key)
        if "request_timeout" in kwargs:
            kwargs["timeout"] = kwargs.pop("request_timeout")
        try:
            return self._dtap_client.chat.completions.create(*args, **kwargs)
        except BadRequestError as error:
            if not _is_vllm_context_overflow(error):
                raise
            last_error = error

        # The failed validation generated no tokens.  Retry the exact same
        # messages, tools, seed, and sampling configuration while searching only
        # the output reservation.  This leaves every normal request unchanged and
        # never mutates or silently truncates trusted/untrusted history.
        for budget in _context_recovery_budgets(kwargs.get("max_tokens")):
            retry_kwargs = dict(kwargs)
            retry_kwargs["max_tokens"] = budget
            print(
                "[CONTEXT RECOVERY] recognized vLLM 400 context overflow; "
                f"retrying with max_tokens={budget}",
                file=sys.stderr,
                flush=True,
            )
            try:
                return self._dtap_client.chat.completions.create(
                    *args,
                    **retry_kwargs,
                )
            except BadRequestError as error:
                if not _is_vllm_context_overflow(error):
                    raise
                last_error = error
        raise RuntimeError(
            "Qwen-Agent bounded output-budget recovery exhausted"
        ) from last_error

    def convert_messages_to_dicts(
        self, messages: List[Message]
    ) -> List[Dict[str, Any]]:
        """Map explicit trust to transport tags and remove all private metadata."""
        converted: List[Dict[str, Any]] = []
        assistant_content: List[str] = []
        assistant_reasoning: List[str] = []
        assistant_calls: List[str] = []

        def flush_assistant() -> None:
            if not assistant_content and not assistant_reasoning and not assistant_calls:
                return
            content = "".join(assistant_content)
            if assistant_calls:
                rendered_calls = "\n\n".join(assistant_calls)
                content = f"{content}\n\n{rendered_calls}" if content else rendered_calls
            message: Dict[str, Any] = {"role": "assistant", "content": content}
            if assistant_reasoning:
                message["reasoning"] = "".join(assistant_reasoning)
            converted.append(message)
            assistant_content.clear()
            assistant_reasoning.clear()
            assistant_calls.clear()

        for source in messages:
            if source.role == ASSISTANT:
                if source.content:
                    assistant_content.append(_content_to_text(source.content))
                if source.reasoning_content:
                    assistant_reasoning.append(
                        _content_to_text(source.reasoning_content)
                    )
                if source.function_call:
                    assistant_calls.append(
                        _render_tool_call_xml(
                            source.function_call.name,
                            source.function_call.arguments,
                        )
                    )
                continue

            flush_assistant()
            if source.role != USER:
                raise ValueError(
                    "DTap Qwen-Agent history permits only explicit-trust user "
                    f"messages and assistant messages, got {source.role!r}"
                )
            extra = dict(source.extra or {})
            raw_trust = extra.pop(TRANSPORT_TRUST_KEY, None)
            if extra:
                raise ValueError(
                    f"Unexpected private Qwen-Agent message metadata: {sorted(extra)}"
                )
            try:
                trust = Trust(raw_trust)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Every DTap Qwen-Agent user message must have explicit trust"
                ) from exc
            message = {
                "role": "user",
                "content": _content_to_openai(source.content),
            }
            if trust is Trust.UNTRUSTED:
                message["name"] = UNTRUSTED_MESSAGE_NAME
            elif trust is Trust.TRUSTED_TOOL:
                message["name"] = TRUSTED_TOOL_RESULT_NAME
            elif trust is Trust.UNTRUSTED_TOOL:
                message["name"] = UNTRUSTED_TOOL_RESULT_NAME
            converted.append(message)

        flush_assistant()
        return converted

    def close(self) -> None:
        self._dtap_client.close()


@dataclass
class ToolOutcome:
    segments: List[
        tuple[bool, Union[str, List[Dict[str, Any]], List[ContentItem]]]
    ]
    server_name: Optional[str]


class DTapFunctionTool(BaseTool):
    """Dynamic Qwen-Agent BaseTool backed by DTap's FastMCP dispatcher."""

    name = "dtap_dynamic_tool"
    description = ""
    parameters: Union[List[dict], dict] = []

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        dispatcher: Any,
    ) -> None:
        self.name = name
        self.description = description
        # BaseTool's validator accepts only a narrow top-level schema.  Delay
        # assignment so valid MCP schemas retain constraints such as $defs and
        # additionalProperties exactly as the current experiment does.
        self.parameters = []
        super().__init__()
        self.parameters = copy.deepcopy(parameters)
        self._dispatcher = dispatcher

    def call(self, params: Union[str, dict], **kwargs: Any) -> ToolOutcome:
        return self._dispatcher(params, **kwargs)


class DTapQwenFnCallCore(FnCallAgent):
    """Qwen-Agent function loop with DTap's turn budget and provenance."""

    def __init__(
        self,
        *,
        owner: "QwenAgentInputRoleAgent",
        llm: DTapInputRoleOAI,
        tools: Sequence[DTapFunctionTool],
    ) -> None:
        self._owner = owner
        # Avoid Assistant/Memory/RAG behavior; DTap supplies all context itself.
        self.mem = SimpleNamespace(system_files=[])
        super().__init__(
            function_list=list(tools),
            llm=llm,
            system_message="",
            name=None,
            description=None,
        )

    @staticmethod
    def _native_calls(output: Sequence[Message]) -> List[tuple[str, Any, str]]:
        calls: List[tuple[str, Any, str]] = []
        for message in output:
            if message.function_call:
                extra = message.extra or {}
                calls.append(
                    (
                        message.function_call.name,
                        message.function_call.arguments,
                        str(extra.get("function_id") or uuid.uuid4().hex),
                    )
                )
        return calls

    @staticmethod
    def _fallback_calls(output: Sequence[Message]) -> List[tuple[str, Any, str]]:
        content = "".join(
            _content_to_text(message.content)
            for message in output
            if message.content
        )
        return [
            (call.function.name, call.function.arguments, str(call.id))
            for call in _fallback_tool_calls(content)
        ]

    def _run(
        self,
        messages: List[Message],
        lang: Literal["en", "zh"] = "en",
        **kwargs: Any,
    ) -> Iterator[List[Message]]:
        messages = copy.deepcopy(messages)
        response: List[Message] = []

        while self._owner._generation_count < self._owner.runtime_config.max_turns:
            extra_generate_cfg: Dict[str, Any] = {"lang": lang}
            if kwargs.get("seed") is not None:
                extra_generate_cfg["seed"] = kwargs["seed"]
            output_stream = self._call_llm(
                messages=messages,
                functions=[tool.function for tool in self.function_map.values()],
                extra_generate_cfg=extra_generate_cfg,
            )
            output: List[Message] = []
            for output in output_stream:
                if output:
                    yield response + output
            self._owner._generation_count += 1
            if not output:
                raise RuntimeError("Qwen-Agent model transport returned no messages")
            response.extend(output)
            messages.extend(output)

            calls = self._native_calls(output)
            if not calls:
                calls = self._fallback_calls(output)
            if not calls:
                break

            for tool_name, raw_arguments, tool_call_id in calls:
                try:
                    arguments = _parse_tool_arguments(raw_arguments)
                except ValueError:
                    # This is a model-output error, not a tool return.  Never guess
                    # at or dispatch malformed arguments.  A bounded framework-
                    # generated trusted result lets Qwen correct the JSON on the
                    # next model turn without converting the task to an execution
                    # failure.
                    result_message = qwen_message(
                        _malformed_tool_error(tool_name, raw_arguments),
                        Trust.TRUSTED_TOOL,
                    )
                    messages.append(result_message)
                    response.append(result_message)
                    yield response
                    continue
                tool = self.function_map.get(tool_name)
                if tool is None:
                    outcome = self._owner._dispatch_tool_from_qwen_thread(
                        tool_name, arguments, tool_call_id
                    )
                else:
                    outcome = tool.call(
                        arguments,
                        tool_call_id=tool_call_id,
                    )
                for trusted, content in outcome.segments:
                    trust = Trust.TRUSTED_TOOL if trusted else Trust.UNTRUSTED_TOOL
                    result_message = qwen_message(content, trust)
                    messages.append(result_message)
                    response.append(result_message)
                yield response
        else:
            exhausted = Message(
                role=ASSISTANT,
                content="Maximum agent turns reached before a final answer.",
            )
            messages.append(exhausted)
            response.append(exhausted)

        yield response


class QwenAgentInputRoleAgent(QwenInputRoleAgent):
    """DTap backend using Qwen-Agent with complete input-role separation."""

    FRAMEWORK_NAME = FRAMEWORK_NAME

    def __init__(
        self,
        agent_config: AgentConfig,
        runtime_config: Optional[RuntimeConfig] = None,
    ) -> None:
        # Deliberately bypass QwenInputRoleAgent.__init__: that constructor owns
        # a direct AsyncOpenAI client.  This adapter's model path is Qwen-Agent.
        Agent.__init__(self, agent_config, runtime_config)
        self.output_dir = Path(
            self.runtime_config.output_dir or Path.cwd() / "results"
        ).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.config and (self.config.sub_agents or self.config.a2a):
            raise ValueError(
                "qwen_agent_input fails closed for sub-agents/A2A; this DTap "
                "evaluation contains no such tasks"
            )
        if self.runtime_config.a2a_injection:
            raise ValueError("qwen_agent_input does not support A2A injections")

        self._messages: List[Message] = []
        self._tools: List[Dict[str, Any]] = []
        self._tool_to_server: Dict[str, str] = {}
        self._server_urls: Dict[str, str] = {}
        self._run_index = 0
        self._generation_count = 0
        self._mcp_tool_timeout = _positive_env_float(
            "DTAP_QWEN_TOOL_TIMEOUT",
            DEFAULT_MCP_CLIENT_TIMEOUT_SECONDS,
        )

        self._trajectory: Optional[Trajectory] = None
        self._trajectory_path: Optional[Path] = None
        self._started_at: Optional[float] = None

        self._skill_temp_dir: Optional[str] = None
        self._skill_paths: Dict[str, str] = {}
        self._skill_injection_keys: Dict[str, str] = {}
        self._skill_source_content: Dict[str, str] = {}
        self._skill_source_metadata: Dict[str, str] = {}
        self._trusted_skill_metadata: List[str] = []
        self._untrusted_skill_metadata: List[str] = []

        kwargs = self.runtime_config.agent_kwargs or {}
        raw_prompt_injections = kwargs.get("untrusted_prompt_injections", {})
        self._prompt_injections: Dict[int, List[str]] = {
            int(turn): [str(item) for item in items]
            for turn, items in raw_prompt_injections.items()
        }

        self._max_output_tokens = int(
            os.getenv("DTAP_QWEN_MAX_OUTPUT_TOKENS", str(MIN_OUTPUT_TOKENS))
        )
        if self._max_output_tokens < MIN_OUTPUT_TOKENS:
            raise ValueError(
                "DTAP_QWEN_MAX_OUTPUT_TOKENS must be at least "
                f"{MIN_OUTPUT_TOKENS}, got {self._max_output_tokens}"
            )
        if _env_bool("DTAP_QWEN_ENABLE_THINKING", False):
            raise ValueError("Qwen thinking must remain disabled for this backend")
        self._enable_thinking = False
        self._owner_loop: Optional[asyncio.AbstractEventLoop] = None
        self._llm: Optional[DTapInputRoleOAI] = None
        self._core: Optional[DTapQwenFnCallCore] = None

    async def initialize(self) -> None:
        if not self.config:
            raise ValueError("Agent config is required")
        await self._discover_mcp_tools()
        self._setup_skills()
        self._build_qwen_agent()

    def _build_qwen_agent(self) -> None:
        temperature = (
            self.runtime_config.temperature
            if self.runtime_config.temperature is not None
            else float(os.getenv("DTAP_QWEN_TEMPERATURE", "1.0"))
        )
        if not math.isfinite(float(temperature)):
            raise ValueError("Qwen-Agent temperature must be finite")
        generate_cfg: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": float(os.getenv("DTAP_QWEN_TOP_P", "0.95")),
            "top_k": int(os.getenv("DTAP_QWEN_TOP_K", "20")),
            "max_tokens": self._max_output_tokens,
            "seed": int(os.getenv("DTAP_QWEN_SEED", "0")),
            "max_input_tokens": 0,
            "max_retries": 0,
            "use_raw_api": True,
            "request_timeout": float(
                os.getenv("DTAP_QWEN_REQUEST_TIMEOUT", "1800")
            ),
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": self._enable_thinking,
                    "preserve_thinking": self._enable_thinking,
                }
            },
        }
        presence_penalty = float(
            os.getenv("DTAP_QWEN_PRESENCE_PENALTY", "0")
        )
        if presence_penalty:
            generate_cfg["presence_penalty"] = presence_penalty
        if self._tools:
            generate_cfg["tool_choice"] = "auto"
            generate_cfg["parallel_tool_calls"] = False

        llm_cfg = {
            "model": self.runtime_config.model,
            "model_server": os.environ["DTAP_QWEN_BASE_URL"],
            "api_key": os.getenv("DTAP_QWEN_API_KEY", "EMPTY"),
            "request_timeout": generate_cfg["request_timeout"],
            "generate_cfg": generate_cfg,
        }
        self._llm = DTapInputRoleOAI(llm_cfg)

        qwen_tools: List[DTapFunctionTool] = []
        for tool in self._tools:
            function = tool["function"]
            name = str(function["name"])

            def dispatch(
                params: Union[str, dict],
                *,
                _name: str = name,
                tool_call_id: Optional[str] = None,
                **_: Any,
            ) -> ToolOutcome:
                return self._dispatch_tool_from_qwen_thread(
                    _name,
                    _parse_tool_arguments(params),
                    str(tool_call_id or uuid.uuid4().hex),
                )

            qwen_tools.append(
                DTapFunctionTool(
                    name=name,
                    description=str(function.get("description") or ""),
                    parameters=copy.deepcopy(function.get("parameters") or {}),
                    dispatcher=dispatch,
                )
            )
        self._core = DTapQwenFnCallCore(
            owner=self,
            llm=self._llm,
            tools=qwen_tools,
        )

    def _start_trajectory(self, query: str, metadata: Dict[str, Any]) -> None:
        if self._trajectory is not None:
            return
        self._trajectory = Trajectory(
            task_id=str(metadata.get("task_id") or "unknown"),
            original_instruction=query,
            malicious_instruction=str(metadata.get("malicious_goal") or ""),
            domain=metadata.get("domain"),
            risk_category=metadata.get("risk_category"),
        )
        self._trajectory.start_timer()
        self._started_at = time.monotonic()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._trajectory_path = self.output_dir / f"qwen_agent_input_{stamp}.json"

    def _append_trusted_query(self, query: str) -> None:
        if not self._messages:
            trusted = self.config.system_prompt.strip() if self.config else ""
            if trusted:
                self._messages.append(qwen_message(trusted, Trust.TRUSTED))
            self._messages.extend(
                qwen_message(metadata, Trust.TRUSTED)
                for metadata in self._trusted_skill_metadata
            )
            self._messages.extend(
                qwen_message(metadata, Trust.UNTRUSTED)
                for metadata in self._untrusted_skill_metadata
            )
        self._messages.append(qwen_message(query, Trust.TRUSTED))
        for content in self._prompt_injections.get(self._run_index, []):
            self._messages.append(qwen_message(content, Trust.UNTRUSTED))

    async def _execute_tool_for_core(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        tool_call_id: str,
    ) -> ToolOutcome:
        segments, server_name = await self._call_tool(tool_name, arguments)
        assert self._trajectory is not None
        self._trajectory.append_agent_step(
            action=tool_name,
            tool_name=tool_name,
            tool_params=arguments,
            server=server_name,
            metadata={"tool_call_id": tool_call_id},
        )
        trajectory_content = "".join(
            content
            if isinstance(content, str)
            else json.dumps(
                [
                    item.model_dump() if hasattr(item, "model_dump") else item
                    for item in content
                ],
                ensure_ascii=False,
            )
            for _, content in segments
        )
        self._trajectory.append_tool_return(
            trajectory_content,
            tool_name=tool_name,
            server=server_name,
            metadata={"tool_call_id": tool_call_id},
        )
        self._save_trajectory()
        return ToolOutcome(segments=segments, server_name=server_name)

    def _dispatch_tool_from_qwen_thread(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        tool_call_id: str,
    ) -> ToolOutcome:
        if self._owner_loop is None:
            raise RuntimeError("Qwen-Agent tool dispatch has no owning event loop")
        future = asyncio.run_coroutine_threadsafe(
            self._execute_tool_for_core(tool_name, arguments, tool_call_id),
            self._owner_loop,
        )
        try:
            return future.result(timeout=self._mcp_tool_timeout + 10.0)
        except concurrent.futures.TimeoutError:
            future.cancel()
            message = (
                f"MCP tool {tool_name!r} timed out after "
                f"{self._mcp_tool_timeout:g} seconds. The tool did not "
                "complete; try a different action or continue without it."
            )
            return ToolOutcome(
                segments=[(False, message)],
                server_name=self._tool_to_server.get(tool_name),
            )

    async def run(
        self,
        user_input: Union[str, List[str]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        queries = [user_input] if isinstance(user_input, str) else list(user_input)
        result: Optional[AgentResult] = None
        for query in queries:
            result = await self._run_one(str(query), metadata or {})
        assert result is not None
        return result

    async def _run_one(self, query: str, metadata: Dict[str, Any]) -> AgentResult:
        if self._core is None:
            raise RuntimeError("Qwen-Agent backend was not initialized")
        self._run_index += 1
        self._start_trajectory(query, metadata)
        assert self._trajectory is not None
        self._trajectory.append_user_step(query)
        self._append_trusted_query(query)

        self._owner_loop = asyncio.get_running_loop()
        try:
            responses = await asyncio.to_thread(
                self._core.run_nonstream,
                self._messages,
                lang="en",
                seed=int(os.getenv("DTAP_QWEN_SEED", "0")),
            )
        finally:
            self._owner_loop = None
        normalized = [
            Message(**message) if isinstance(message, dict) else message
            for message in responses
        ]
        self._messages.extend(normalized)

        final_output = ""
        for message in reversed(normalized):
            if (
                message.role == ASSISTANT
                and message.content
                and not message.function_call
            ):
                final_output = _content_to_text(message.content)
                break
        if not final_output and self._generation_count >= self.runtime_config.max_turns:
            final_output = "Maximum agent turns reached before a final answer."

        self._trajectory.set_final_response(final_output)
        self._save_trajectory()
        return AgentResult(
            final_output=final_output,
            turn_count=self._generation_count,
            trajectory=self._trajectory,
            trace_id=uuid.uuid4().hex,
            duration=(
                time.monotonic() - self._started_at
                if self._started_at is not None
                else None
            ),
        )

    async def cleanup(self) -> None:
        self._save_trajectory()
        if self._llm is not None:
            await asyncio.to_thread(self._llm.close)
            self._llm = None
        self._core = None
        if self._skill_temp_dir:
            from utils.skill_helpers import cleanup_temp_directory

            cleanup_temp_directory(self._skill_temp_dir)
            self._skill_temp_dir = None
        self._server_urls.clear()
        self._tool_to_server.clear()


__all__ = [
    "DTapFunctionTool",
    "DTapInputRoleOAI",
    "DTapQwenFnCallCore",
    "QwenAgentInputRoleAgent",
    "QWEN_AGENT_VERSION",
    "TOOL_INJECTION_BOUNDARY",
    "Trust",
    "qwen_message",
]
