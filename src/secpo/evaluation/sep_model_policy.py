"""Model-family policy boundary for adaptive SEP victim calls.

The adaptive search algorithms operate only on visible assistant text.  This
module keeps tokenizer- and chat-template differences out of Genetic, PAIR,
and TAP, so all three attacks receive identical model-family handling.

Nothing is loaded at import time.  A concrete backend may use
``generation_options`` when constructing its request and may optionally
implement ``configure_model_policy(policy)``.  ``PolicyAppliedBackend`` also
normalizes every returned response before it reaches either the search or its
scorer; this is particularly important for Qwen reasoning blocks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Sequence


LLAMA31_EOT_TOKEN = "<|eot_id|>"
LLAMA31_EOT_TOKEN_ID = 128009
QWEN_IM_END_TOKEN = "<|im_end|>"
SUPPORTED_MODEL_FAMILIES = ("llama31", "qwen36")


def normalize_model_family(value: str) -> str:
    """Return the canonical released family name without guessing from paths."""

    normalized = re.sub(r"[^a-z0-9]+", "", value.casefold())
    aliases = {
        "llama31": "llama31",
        "llama318b": "llama31",
        "metallama31": "llama31",
        "qwen36": "qwen36",
        "qwen3627b": "qwen36",
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError(
            f"unsupported model family {value!r}; expected one of "
            f"{', '.join(SUPPORTED_MODEL_FAMILIES)}"
        ) from error


@dataclass(frozen=True)
class ModelSource:
    """Explicit base-model/adapter selection.

    The research Qwen launcher inferred LoRA use from underscores in a path.
    That corrupts ordinary checkpoint paths.  The public interface requires an
    adapter to be named explicitly and otherwise leaves the model path intact.
    """

    model_name_or_path: str
    adapter_name_or_path: str | None = None

    def __post_init__(self) -> None:
        if not self.model_name_or_path.strip():
            raise ValueError("model_name_or_path must not be empty")
        if self.adapter_name_or_path is not None and not self.adapter_name_or_path.strip():
            raise ValueError("adapter_name_or_path must be non-empty or null")

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


def resolve_model_source(
    model_name_or_path: str | Path,
    *,
    adapter_name_or_path: str | Path | None = None,
) -> ModelSource:
    """Resolve an explicit model source without inspecting underscore names."""

    model = str(model_name_or_path)
    adapter = None if adapter_name_or_path is None else str(adapter_name_or_path)
    return ModelSource(model_name_or_path=model, adapter_name_or_path=adapter)


def _valid_token_id(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _tokenizer_id(tokenizer: Any, token: str) -> int | None:
    converter = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(converter):
        return None
    value = _valid_token_id(converter(token))
    unknown = _valid_token_id(getattr(tokenizer, "unk_token_id", None))
    return None if value is None or value == unknown else value


def _tokenizer_eos_ids(tokenizer: Any) -> list[int]:
    raw = getattr(tokenizer, "eos_token_id", None)
    values = raw if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else (raw,)
    return [value for item in values if (value := _valid_token_id(item)) is not None]


def _merge_token_ids(*groups: Sequence[int]) -> tuple[int, ...]:
    result: list[int] = []
    for group in groups:
        for item in group:
            value = _valid_token_id(item)
            if value is not None and value not in result:
                result.append(value)
    return tuple(result)


@dataclass(frozen=True)
class ModelFamilyPolicy:
    """Immutable generation and visible-output policy for one model family."""

    family: str
    enable_thinking: bool = False
    max_output_tokens: int = 1536

    def __post_init__(self) -> None:
        canonical = normalize_model_family(self.family)
        object.__setattr__(self, "family", canonical)
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if canonical == "llama31" and self.enable_thinking:
            raise ValueError("enable_thinking is only supported for Qwen3.6")

    @property
    def untrusted_role(self) -> str:
        return "input" if self.family == "qwen36" else "user"

    def stop_token_ids(
        self,
        tokenizer: Any,
        requested: Sequence[int] = (),
    ) -> tuple[int, ...]:
        """Return model-correct termination IDs.

        Qwen explicitly removes Llama's ``128009`` even if a legacy caller
        supplied it.  End IDs are tokenizer-derived so the policy does not
        assume a particular Qwen vocabulary revision.
        """

        requested_ids = tuple(
            value for item in requested if (value := _valid_token_id(item)) is not None
        )
        eos = tuple(_tokenizer_eos_ids(tokenizer))
        if self.family == "qwen36":
            requested_ids = tuple(
                item for item in requested_ids if item != LLAMA31_EOT_TOKEN_ID
            )
            im_end = _tokenizer_id(tokenizer, QWEN_IM_END_TOKEN)
            extra = () if im_end is None else (im_end,)
            return tuple(
                item
                for item in _merge_token_ids(requested_ids, eos, extra)
                if item != LLAMA31_EOT_TOKEN_ID
            )

        eot = _tokenizer_id(tokenizer, LLAMA31_EOT_TOKEN)
        extra = (LLAMA31_EOT_TOKEN_ID,) if eot is None else (eot,)
        return _merge_token_ids(requested_ids, eos, extra)

    def generation_options(
        self,
        tokenizer: Any,
        *,
        requested_stop_token_ids: Sequence[int] = (),
    ) -> dict[str, Any]:
        """Return portable options a local or hosted victim backend can map."""

        result: dict[str, Any] = {
            "max_output_tokens": self.max_output_tokens,
            "stop_token_ids": list(
                self.stop_token_ids(tokenizer, requested_stop_token_ids)
            ),
        }
        if self.family == "qwen36":
            result["chat_template_kwargs"] = {
                "enable_thinking": self.enable_thinking,
            }
        return result

    def render_messages(
        self,
        *,
        instruction: str,
        input_text: str,
        injection: str,
    ) -> list[dict[str, str]]:
        """Render the released SEP user/input separation convention."""

        if not instruction:
            raise ValueError("instruction must not be empty")
        if not injection:
            raise ValueError("injection must not be empty")
        untrusted = input_text.replace("{injection}", injection)
        if "{injection}" not in input_text:
            untrusted = f"{input_text}\n\n{injection}" if input_text else injection
        if self.family == "qwen36":
            return [
                {"role": "user", "content": instruction},
                {"role": "input", "content": untrusted},
            ]
        return [
            {
                "role": "user",
                "content": (
                    f"{instruction}\n\n"
                    "<untrusted_input>\n"
                    f"{untrusted}\n"
                    "</untrusted_input>"
                ),
            }
        ]

    def normalize_response(self, value: str) -> str:
        """Return only text that is visible to the user and attack scorer."""

        if not isinstance(value, str):
            raise TypeError("victim response must be a string")
        if self.family != "qwen36":
            return value
        return _strip_qwen_reasoning(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "enable_thinking": self.enable_thinking,
            "max_output_tokens": self.max_output_tokens,
            "untrusted_role": self.untrusted_role,
        }


def _strip_qwen_reasoning(value: str) -> str:
    """Remove complete or truncated ``<think>`` regions conservatively."""

    text = value
    # Some OpenAI-compatible servers omit the opening tag while retaining the
    # closing delimiter.  When the prefix is only whitespace, keep only the
    # visible suffix.
    if "<think>" not in text and "</think>" in text:
        _, text = text.split("</think>", 1)

    visible: list[str] = []
    cursor = 0
    while True:
        opening = text.find("<think>", cursor)
        if opening < 0:
            visible.append(text[cursor:])
            break
        visible.append(text[cursor:opening])
        closing = text.find("</think>", opening + len("<think>"))
        if closing < 0:
            # Never expose or score an unfinished reasoning block.
            break
        cursor = closing + len("</think>")
    return "".join(visible).strip()


def model_family_policy(
    family: str,
    *,
    enable_thinking: bool | None = None,
    max_output_tokens: int | None = None,
) -> ModelFamilyPolicy:
    """Construct released Llama3.1/Qwen3.6 scientific generation defaults."""

    canonical = normalize_model_family(family)
    thinking = bool(enable_thinking) if enable_thinking is not None else False
    default_tokens = 32768 if canonical == "qwen36" and thinking else 1536
    return ModelFamilyPolicy(
        family=canonical,
        enable_thinking=thinking,
        max_output_tokens=(
            default_tokens if max_output_tokens is None else max_output_tokens
        ),
    )


class PolicyAppliedBackend:
    """Apply one family policy uniformly to an adaptive backend."""

    def __init__(self, backend: Any, policy: ModelFamilyPolicy) -> None:
        self.backend = backend
        self.policy = policy
        configure = getattr(backend, "configure_model_policy", None)
        if configure is not None:
            if not callable(configure):
                raise TypeError("backend configure_model_policy attribute is not callable")
            configure(policy)

    def propose(self, request: Any) -> Sequence[str]:
        return self.backend.propose(request)

    def victim(self, example: Any, injection: str) -> str:
        raw = self.backend.victim(example, injection)
        return self.policy.normalize_response(raw)

    def score(self, example: Any, injection: str, response: str) -> float:
        # ``response`` is guaranteed to have passed through normalize_response.
        return self.backend.score(example, injection, response)


__all__ = [
    "LLAMA31_EOT_TOKEN_ID",
    "ModelFamilyPolicy",
    "ModelSource",
    "PolicyAppliedBackend",
    "QWEN_IM_END_TOKEN",
    "SUPPORTED_MODEL_FAMILIES",
    "model_family_policy",
    "normalize_model_family",
    "resolve_model_source",
]
