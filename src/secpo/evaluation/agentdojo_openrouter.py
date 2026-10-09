"""Opt-in OpenRouter wiring; the released search, prompts and scoring stay intact.

This module is a small compatibility harness, not an alternative evaluator.
It records provider receipts without credentials and puts hard bounds on calls
and output tokens. Receipt totals are not a substitute for an account invoice.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import threading
import time
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .agentdojo_models import AgentDojoModelBackend, OpenAIChatConfig, OpenAIChatModel


BASE_URL = "https://openrouter.ai/api/v1"
ENDPOINT = BASE_URL + "/chat/completions"
MODELS = ("z-ai/glm-5.3-flash", "deepseek/deepseek-v4.1-flash")
KEY_ENV = "AGENTDOJO_OPENROUTER_API_KEY"
URL_ENV = "AGENTDOJO_OPENROUTER_BASE_URL"


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_key(path=None):
    """Parse dotenv data without sourcing it, evaluating Python, or logging it."""
    values = {}
    if path is not None:
        for line in Path(path).read_text().splitlines():
            parts = shlex.split(line, comments=True)
            if parts and parts[0] == "export":
                parts = parts[1:]
            if not parts:
                continue
            if len(parts) != 1 or "=" not in parts[0]:
                raise ValueError("Invalid credential file; contents withheld")
            name, value = parts[0].split("=", 1)
            if name in values:
                raise ValueError("Duplicate credential assignment")
            values[name] = value
    else:
        values = {name: os.environ.get(name, "") for name in
                  ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL")}
    if (values.get("OPENROUTER_BASE_URL") or BASE_URL).rstrip("/") != BASE_URL:
        raise ValueError("Credentials may only be sent to OpenRouter")
    key = values.get("OPENROUTER_API_KEY", "").strip()
    if not key or any(c.isspace() for c in key) or "$" in key or "`" in key:
        raise ValueError("Missing or invalid OpenRouter credential; contents withheld")
    return key


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Refusing to redirect an authenticated model request")


class RecordedTransport:
    """One receipt per attempted call; never records Authorization headers."""

    def __init__(self, directory, *, max_requests=4, max_tokens=4096, post=None):
        if max_requests < 1 or max_tokens < 1:
            raise ValueError("Request and token limits must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_requests, self.max_tokens = max_requests, max_tokens
        self.requests = 0
        self.lock = threading.Lock()
        self.receipts = []
        self.post = post

    def _post(self, url, payload, headers, timeout):
        request = Request(url, data=json.dumps(payload).encode(),
                          headers=dict(headers), method="POST")
        opener = build_opener(ProxyHandler({}), NoRedirect())
        with opener.open(request, timeout=timeout) as response:
            return json.load(response)

    def __call__(self, url, payload, headers, timeout):
        if url != ENDPOINT or payload.get("model") not in MODELS:
            raise ValueError("Unexpected endpoint or model; no substitutions allowed")
        if not 0 < payload.get("max_tokens", 0) <= self.max_tokens:
            raise ValueError("Output token limit exceeded")
        with self.lock:
            if self.requests >= self.max_requests:
                raise RuntimeError("Smoke-test request budget exhausted")
            self.requests += 1
            request_number = self.requests
        started = time.monotonic()
        record = {"request_number": request_number, "at": now(), "request": payload}
        try:
            result = (self.post or self._post)(url, payload, headers, timeout)
            record["response"] = result
            if result.get("model") != payload["model"]:
                raise ValueError("Unexpected response model")
            choices = result.get("choices", [])
            if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
                raise ValueError("Incomplete model response")
            record["status"] = "complete"
            return result
        except Exception as exc:
            # Do not retain server error bodies, request headers, or exception text.
            record["status"] = "failed"
            record["error_type"] = type(exc).__name__
            if isinstance(exc, HTTPError):
                record["http_status"] = exc.code
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            write_json(self.directory / f"{request_number:03d}.json", record)
            with self.lock:
                self.receipts.append(record)

    def summary(self):
        responses = [row["response"] for row in self.receipts if "response" in row]
        usages = [response.get("usage") or {} for response in responses]
        return {
            "attempted_requests": self.requests,
            "recorded_responses": len(responses),
            "prompt_tokens": sum(u.get("prompt_tokens", 0) or 0 for u in usages),
            "completion_tokens": sum(u.get("completion_tokens", 0) or 0 for u in usages),
            "reported_cost_usd": sum(u.get("cost", 0) or 0 for u in usages),
            "responses_missing_cost": sum(u.get("cost") is None for u in usages),
        }


def make_backend(key, *, attacker_model, judge_model, transport, max_tokens=4096):
    """Use the existing model classes, prompts, parser and critic score clamp.

    Preserve released decoding defaults: attacker T=.9/top_p=.9/high reasoning;
    critic T=.7/no explicit top_p or reasoning. Disable retries only for this
    bounded component probe. This does not configure a victim agent.
    """
    if attacker_model not in MODELS or judge_model not in MODELS:
        raise ValueError("Unsupported smoke-test model")
    os.environ[URL_ENV] = BASE_URL
    os.environ[KEY_ENV] = key
    common = dict(base_url_env=URL_ENV, api_key_env=KEY_ENV, max_tokens=max_tokens,
                  timeout=180, http_attempts=1)
    attacker = OpenAIChatModel(OpenAIChatConfig(
        model=attacker_model, temperature=.9, top_p=.9, reasoning_effort="high",
        **common), transport=transport)
    judge = OpenAIChatModel(OpenAIChatConfig(
        model=judge_model, temperature=.7, top_p=None, reasoning_effort=None,
        **common), transport=transport)
    return AgentDojoModelBackend(attacker, judge, workers=1, judge_format_retries=0)
