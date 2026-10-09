"""Checkpoint sampling and durable, credential-free evaluation artifacts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from functools import partial
import json
import logging
import os
from pathlib import Path
import shlex
import tempfile
import threading
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
MODEL = "Qwen/Qwen3.8-27B"
TOKENIZER_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
RENDER_VERSION = "qwen38-secpo-eval-thinking-off-v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            json.dump(value, f, indent=2, ensure_ascii=False, default=json_default)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def json_default(value: Any) -> Any:
    if isinstance(value, partial):
        return {"callable":json_default(value.func), "args":value.args, "kwargs":value.keywords}
    if callable(value):
        return f"{value.__module__}:{value.__qualname__}"
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (Path, set, tuple)):
        return str(value) if isinstance(value, Path) else list(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def load_credentials(path: Path) -> None:
    """Read literal assignments; never execute the credential file as shell code."""
    for line in path.read_text().splitlines():
        parts = shlex.split(line, comments=True)
        if parts and parts[0] == "export":
            parts = parts[1:]
        if len(parts) == 1 and "=" in parts[0]:
            name, value = parts[0].split("=", 1)
            if name in {"TINKER_API_KEY", "HF_TOKEN"} and value:
                os.environ[name] = value
    for name in ("TINKER_API_KEY", "HF_TOKEN"):
        if not os.environ.get(name):
            raise ValueError(f"{name} is missing")
    os.environ["TINKER_TELEMETRY"] = "0"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HOME", str(ROOT / "runtime/hf_cache"))
    for name in ("tinker", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.ERROR)


def final_checkpoint() -> str:
    return json.loads((ROOT / "outputs/qwen38_27b_tinker/result.json").read_text())["final_sampler_path"]


def evaluation_model_id(variant: str) -> str:
    if variant == "base":
        return f"base:{MODEL}"
    if variant == "secpo":
        return final_checkpoint()
    raise ValueError(f"Unknown model variant: {variant}")


def validate_base_comparison(manifest: dict) -> None:
    """Require the baseline to differ from the SecPO protocol only in weights."""
    reference = ROOT / "outputs/qwen38_27b_tinker_utility" / manifest["benchmark"] / "manifest.json"
    prior = json.loads(reference.read_text())
    if manifest["checkpoint"] != evaluation_model_id("base"):
        raise ValueError("Baseline must use the unadapted base model")
    if prior["checkpoint"] != final_checkpoint():
        raise ValueError("Comparison manifest does not refer to the final SecPO sampler")
    settings = lambda value: {k:v for k,v in value.items() if k != "checkpoint"}
    if settings(manifest) != settings(prior):
        raise ValueError("Base-model protocol differs from the existing SecPO evaluation")


def evaluation_output_root(variant: str, max_output_tokens: int | None = None) -> Path:
    if variant not in {"secpo", "base"}:
        raise ValueError(f"Unknown model variant: {variant}")
    name = "qwen38_27b_base_tinker_utility" if variant == "base" else "qwen38_27b_tinker_utility"
    if max_output_tokens is not None:
        if type(max_output_tokens) is not int or not 0 < max_output_tokens < 65536:
            raise ValueError("Output cap must be between 1 and 65535 tokens")
        name += f"_max{max_output_tokens}"
    return ROOT / "outputs" / name


def validate_output_cap_comparison(manifest: dict, variant: str) -> None:
    """Only the output budget may change from the original matched-model run."""
    cap = manifest["default_max_gen_toks"]
    evaluation_output_root(variant, cap)  # Validate the budget before inference.
    reference = evaluation_output_root(variant) / manifest["benchmark"] / "manifest.json"
    expected = deepcopy(json.loads(reference.read_text()))
    if expected["checkpoint"] != evaluation_model_id(variant):
        raise ValueError("Comparison manifest refers to different model weights")
    expected["default_max_gen_toks"] = cap
    for task in expected["tasks"].values():
        task["generation_kwargs"] = {**(task["generation_kwargs"] or {}), "max_gen_toks": cap}
    if manifest != expected:
        raise ValueError("Output-cap experiment changes settings beyond the token budget")


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL, revision=TOKENIZER_REVISION, trust_remote_code=False)


def read_indices(path: Path, benchmark: str) -> dict[str, dict[str, Any]]:
    obj = json.loads(path.read_text())
    if obj.get("task") != benchmark:
        raise ValueError("Index file benchmark does not match")
    leaves = obj["leaf_tasks"]
    for task, record in leaves.items():
        indices = record["indices"]
        if len(indices) != record["num_sampled"] or len(set(indices)) != len(indices):
            raise ValueError(f"Duplicate indices or count mismatch for {task}")
        if not indices or any(type(i) is not int or not 0 <= i < record["num_docs"] for i in indices):
            raise ValueError(f"Invalid indices for {task}")
    return leaves


def cut_at_stop(text: str, stops: list[str]) -> str:
    positions = [text.find(stop) for stop in stops if stop and stop in text]
    return text[:min(positions)] if positions else text


@dataclass(frozen=True)
class Generation:
    text: str
    input_tokens: int
    output_tokens: int
    stop_reason: str
    elapsed_seconds: float


class CheckpointSampler:
    """A shared Tinker sampler with a content-addressed completion cache."""

    def __init__(self, cache_dir: Path, *, checkpoint: str | None = None, tokenizer=None,
                 base_model: str | None = None):
        import tinker

        if base_model is not None and (checkpoint is not None or base_model != MODEL):
            raise ValueError("Specify either the saved checkpoint or the matching base model")
        self.is_base_model = base_model is not None
        self.checkpoint = f"base:{base_model}" if self.is_base_model else checkpoint or final_checkpoint()
        self.tokenizer = tokenizer or load_tokenizer()
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.service = tinker.ServiceClient(timeout=120, max_retries=2)
        if self.is_base_model:
            # Direct base-model sampling: no training client, no new adapter,
            # and no saved SecPO weights loaded into this sampler.
            self.client = self.service.create_sampling_client(base_model=base_model)
        else:
            info = self.service.create_rest_client().get_weights_info_by_tinker_path(self.checkpoint).result(timeout=120)
            if info.base_model != MODEL:
                raise ValueError(f"Unexpected checkpoint base model: {info.base_model}")
            self.client = self.service.create_sampling_client(model_path=self.checkpoint)
        self.lock = threading.Lock()
        self.completed = 0
        self.cached = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def render(self, messages: list[dict], *, tools: list[dict] | None = None) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True,
            enable_thinking=False, preserve_thinking=False,
        )

    def generate(self, prompt: str, *, max_tokens: int, stops=(), temperature=0.0,
                 top_p=1.0, top_k=-1, seed=42) -> Generation:
        import tinker

        stop_list = list(dict.fromkeys([*stops, "<|im_end|>", "<|endoftext|>"]))
        params = dict(max_tokens=max_tokens, stop=stop_list, temperature=temperature,
                      top_p=top_p, top_k=top_k, seed=seed)
        key = digest({"checkpoint": self.checkpoint, "tokenizer_revision": TOKENIZER_REVISION,
                      "render_version": RENDER_VERSION, "prompt": prompt, "params": params})
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.is_file():
            stored = json.loads(cache_path.read_text())
            with self.lock:
                self.cached += 1
            return Generation(**stored["generation"])
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) + max_tokens > 65536:
            raise ValueError(f"Context budget exceeded: {len(tokens)} + {max_tokens} > 65536")
        start = time.monotonic()
        response = self.client.sample(
            prompt=tinker.types.ModelInput.from_ints(tokens), num_samples=1,
            sampling_params=tinker.types.SamplingParams(**params),
        ).result(timeout=1800)
        if len(response.sequences) != 1:
            raise RuntimeError("Expected exactly one sampled sequence")
        sequence = response.sequences[0]
        text = self.tokenizer.decode(sequence.tokens, skip_special_tokens=False)
        generation = Generation(cut_at_stop(text, stop_list), len(tokens), len(sequence.tokens),
                                str(sequence.stop_reason), time.monotonic() - start)
        from dataclasses import asdict
        write_json(cache_path, {"request_hash": key, "generation": asdict(generation)})
        with self.lock:
            self.completed += 1
            self.input_tokens += generation.input_tokens
            self.output_tokens += generation.output_tokens
        return generation

    def stats(self) -> dict:
        with self.lock:
            return dict(completed_api_requests=self.completed, cached_requests=self.cached,
                        input_tokens=self.input_tokens, output_tokens=self.output_tokens)
