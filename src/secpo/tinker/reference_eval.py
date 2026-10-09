"""Tinker candidates and cached OpenRouter judgments for reference utility.

The existing AlpacaEval prompt, output-order randomization, log-probability
parser, and metrics are reused. Only the generation/judge transports change.
No key is included in configurations, cache keys, or saved response artifacts.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from importlib.metadata import version
import json
import math
from pathlib import Path
import re
import shlex
import threading
import time

from secpo.evaluation.reference_utility import (
    load_reference_utility_config, preflight_reference_utility, sha256_file,
)
from .agent_eval import trust_role_template
from .data import QWEN38_NON_THINKING_SUFFIX
from .evaluation_common import ROOT, digest, now, write_json


JUDGE_MODEL = "openai/gpt-4o-mini"
JUDGE_BASE_URL = "https://openrouter.ai/api/v1"
ALPACA_VERSION = "0.6.6"
ALPACA_METRIC_SHA256 = "97aeec3f1c7de6dee6fd31fe66f1702623f4614c3efe1c6fc5f4927cd5fd674d"
ANNOTATOR_NAME = "weighted_alpaca_eval_gpt-4o-mini-2024-07-18"
SEP_TEST_SHA256 = "e659683ae5b10905064d28bca5fd786d1a4ec96ce016a6b93a7282fd977a95a3"
JUDGE_PARAMS = {"model_name": JUDGE_MODEL, "max_tokens": 1, "temperature": 1,
                "logprobs": True, "top_logprobs": 5}


def read_openrouter_credentials(path: Path) -> str:
    """Return the key without executing assignments or changing process env."""
    values = {}
    allowed = {"OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "OPENROUTER_MODEL"}
    try:
        for line in path.read_text().splitlines():
            parts = shlex.split(line, comments=True)
            if parts and parts[0] == "export":
                parts = parts[1:]
            if not parts:
                continue
            if len(parts) != 1 or "=" not in parts[0]:
                raise ValueError()
            name, value = parts[0].split("=", 1)
            if name in allowed:
                if name in values:
                    raise ValueError()
                values[name] = value
    except (OSError, ValueError):
        raise ValueError("Cannot parse OpenRouter credential file; values withheld") from None
    if values.get("OPENROUTER_BASE_URL", JUDGE_BASE_URL).rstrip("/") != JUDGE_BASE_URL:
        raise ValueError("OpenRouter credentials may only be sent to the OpenRouter endpoint")
    if values.get("OPENROUTER_MODEL", "") not in {"", JUDGE_MODEL}:
        raise ValueError("Credential file model differs from the approved GPT-4o-mini judge")
    key = values.get("OPENROUTER_API_KEY", "").strip()
    if not key or any(character.isspace() for character in key):
        raise ValueError("A nonempty OpenRouter API key is required")
    return key


def reference_inputs(suite: str, path: Path, *, approved_sha256: str | None = None,
                     sep_test_path: Path | None = None):
    """Validate full references before any billable request.

    An explicitly approved alternate SEP reference retains all row, model,
    canonical-prompt, and order checks, plus alignment to the frozen test set.
    The original release configuration is never edited or silently relaxed.
    """
    if suite not in {"alpacaeval2", "sep"}:
        raise ValueError("Unsupported reference utility suite")
    config_name = "evaluate_alpacaeval2_utility.json" if suite == "alpacaeval2" else "evaluate_sep_utility.json"
    config = load_reference_utility_config(ROOT / "configs" / config_name)
    frozen_hash = config.reference.sha256
    if approved_sha256 is not None:
        if suite != "sep" or not re.fullmatch(r"[a-f0-9]{64}", approved_sha256):
            raise ValueError("Explicit reference overrides are limited to SHA-256-pinned SEP copies")
        if sep_test_path is None or sha256_file(sep_test_path) != SEP_TEST_SHA256:
            raise ValueError("An alternate SEP reference requires the exact frozen SEP test dataset")
        config = replace(config, reference=replace(config.reference, sha256=approved_sha256))
    # Checking the reference against itself exercises the original full-file,
    # identity, canonical-instruction, generator, and uniqueness validators.
    preflight_reference_utility(config, candidate_path=path, reference_path=path)
    rows = json.loads(path.read_text())
    if any(not row["output"].strip() for row in rows):
        raise ValueError("Reference answers must all be nonempty")
    if approved_sha256 is not None:
        tests = json.loads(sep_test_path.read_text())
        if len(tests) != len(rows) or not all(
            row["instruction_only"] == test["instruction"]
            and all(row[key] == test[key] for key in ("input", "injection", "witness"))
            for row, test in zip(rows, tests)
        ):
            raise ValueError("Alternate SEP reference does not align with the frozen test dataset")
    return config, rows, {"sha256": config.reference.sha256,
                         "release_sha256": frozen_hash,
                         "is_alternate_reference": config.reference.sha256 != frozen_hash,
                         "generator": config.reference.generator,
                         "rows": config.expected_rows}


def clean_messages(suite: str, row: dict) -> list[dict]:
    if suite == "alpacaeval2":
        return [{"role": "user", "content": row["instruction"]}]
    if suite != "sep":
        raise ValueError("Unsupported suite")
    messages = [{"role": "user", "content": row["instruction_only"]}]
    if row["input"]:
        messages.append({"role": "input", "content": row["input"]})
    return messages  # Deliberately never read or append row['injection'].


def render_clean_prompt(tokenizer, suite: str, row: dict) -> str:
    prompt = tokenizer.apply_chat_template(
        clean_messages(suite, row), chat_template=trust_role_template(tokenizer),
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
        preserve_thinking=False,
    )
    if not prompt.endswith(QWEN38_NON_THINKING_SUFFIX):
        raise ValueError("Evaluation thinking-off suffix differs from training")
    return prompt


def evaluator_sources() -> dict:
    import alpaca_eval
    if version("alpaca-eval") != ALPACA_VERSION:
        raise ValueError("Use the inspected alpaca-eval==0.6.6 environment")
    root = Path(alpaca_eval.__file__).resolve().parent
    paths = [f"evaluators_configs/{ANNOTATOR_NAME}/configs.yaml",
             "evaluators_configs/alpaca_eval_clf_gpt4_turbo/alpaca_eval_clf.txt",
             "annotators/base.py", "annotators/pairwise_evaluator.py",
             "processors.py", "completion_parsers.py", "utils.py",
             "metrics/winrate.py", "metrics/glm_winrate.py"]
    return {"version": ALPACA_VERSION,
            "sha256": {p: sha256_file(root / p) for p in paths}}


def validate_judge_choice(choice: dict) -> float:
    from alpaca_eval.completion_parsers import logprob_parser
    try:
        preferences = logprob_parser(choice, numerator_token="m", denominator_tokens=["m", "M"],
                                     is_binarize=False)
        if len(preferences) != 1:
            raise ValueError()
        value = float(preferences[0])
        if not math.isfinite(value) or not 1 <= value <= 2:
            raise ValueError()
    except (KeyError, TypeError, AssertionError, ValueError, IndexError):
        raise ValueError("Judge did not return one valid log-probability preference") from None
    return value


class OpenRouterJudge:
    """AlpacaEval-compatible decoder with bounded requests and durable caching."""

    def __init__(self, cache_dir: Path, credentials: Path, *, concurrency: int = 8,
                 progress=None, post=None):
        self._key = read_openrouter_credentials(credentials)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.concurrency = concurrency
        self.progress = progress
        self._post_override = post
        self.lock = threading.Lock()
        self.stats = {"api_requests_completed": 0, "cached_requests": 0,
                      "input_tokens": 0, "output_tokens": 0, "reported_cost_usd": 0.0}

    def _post(self, payload):
        if self._post_override is not None:
            return self._post_override(payload)
        import requests
        for attempt in range(3):
            try:
                with requests.Session() as session:
                    session.trust_env = False
                    response = session.post(JUDGE_BASE_URL + "/chat/completions",
                        headers={"Authorization": "Bearer " + self._key}, json=payload,
                        timeout=(15, 90), allow_redirects=False)
                if response.status_code == 200:
                    return response.json()
                status = response.status_code
                if status != 429 and status < 500:
                    raise RuntimeError(f"OpenRouter HTTP {status}; body withheld")
            except requests.RequestException:
                status = "network_error"
            if attempt < 2:
                time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenRouter failed after three attempts ({status}); body withheld")

    def _one(self, prompt: str) -> dict:
        from alpaca_eval.utils import prompt_to_chatml
        payload = {"model": JUDGE_MODEL, "messages": prompt_to_chatml(prompt.strip()),
                   **{k: v for k, v in JUDGE_PARAMS.items() if k != "model_name"},
                   "provider": {"require_parameters": True}}
        key = digest({"version": "secpo-openrouter-judge-v1", "endpoint": JUDGE_BASE_URL,
                      "request": payload})
        path = self.cache_dir / f"{key}.json"
        cached = path.is_file()
        if cached:
            record = json.loads(path.read_text())
            if (record.get("request_hash") != key
                    or record.get("requested_model") != JUDGE_MODEL
                    or "gpt-4o-mini" not in record.get("returned_model", "")):
                raise ValueError("Judge cache request identity mismatch")
        else:
            start = time.monotonic()
            body = self._post(payload)
            if len(body.get("choices", [])) != 1:
                raise ValueError("Expected one judge completion")
            choice = body["choices"][0]
            validate_judge_choice(choice)
            reported_model = body.get("model", "")
            if "gpt-4o-mini" not in reported_model:
                raise ValueError("OpenRouter returned an unexpected judge model")
            usage = body.get("usage") or {}
            record = {"request_hash": key, "created_at": now(), "requested_model": JUDGE_MODEL,
                      "returned_model": reported_model, "provider": body.get("provider"),
                      "response_id": body.get("id"), "choice": choice,
                      "usage": {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "cost")},
                      "elapsed_seconds": time.monotonic() - start}
            write_json(path, record)
        validate_judge_choice(record["choice"])
        with self.lock:
            self.stats["cached_requests" if cached else "api_requests_completed"] += 1
            if not cached:
                usage = record["usage"]
                self.stats["input_tokens"] += usage.get("prompt_tokens") or 0
                self.stats["output_tokens"] += usage.get("completion_tokens") or 0
                self.stats["reported_cost_usd"] += usage.get("cost") or 0.0
            if self.progress:
                self.progress(dict(self.stats))
        return record

    def __call__(self, prompts, **kwargs):
        if kwargs != JUDGE_PARAMS:
            raise ValueError("AlpacaEval decoder parameters differ from the approved judge configuration")
        # Identical comparisons need only one request, including within the
        # same concurrent batch. Preserve the caller's order and multiplicity.
        prompts = list(prompts)
        unique_prompts = list(dict.fromkeys(prompts))
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            try:
                by_prompt = dict(zip(unique_prompts, pool.map(self._one, unique_prompts)))
            except BaseException:
                pool.shutdown(wait=True, cancel_futures=True)
                raise
        records = [by_prompt[prompt] for prompt in prompts]
        return {"completions": [r["choice"]["message"]["content"] for r in records],
                "completions_all": [r["choice"] for r in records],
                "price_per_example": [r["usage"].get("cost") or 0.0 for r in records],
                "time_per_example": [r["elapsed_seconds"] for r in records]}


def make_annotator(judge, config):
    """Retain the installed mini judge's prompt, parser, and randomization."""
    from alpaca_eval.annotators import PairwiseAnnotator
    import pandas as pd
    annotator = PairwiseAnnotator(annotators_config=ANNOTATOR_NAME, caching_path=None,
                                 input_keys=list(config.identity_keys), seed=0,
                                 is_store_missing_annotations=False)
    # In 0.6.6 a fresh in-memory cache is None, but disabling missing-label
    # storage unconditionally queries it. Initialize the empty cache explicitly.
    if annotator.df_annotations is None:
        annotator.df_annotations = pd.DataFrame(columns=annotator.all_keys + [annotator.annotation_key])
        annotator.df_annotations[annotator.annotation_key] = pd.Series(dtype=float)
    if len(annotator.annotators) != 1:
        raise ValueError("Expected one utility judge model")
    single = next(iter(annotator.annotators.values()))
    # Provider-neutral configuration does not contain a client or credentials.
    single.fn_completions = judge
    single.completions_kwargs = dict(JUDGE_PARAMS)
    return annotator


def validate_annotations(annotations, candidates, references, identity_keys):
    if len(annotations) != len(candidates) or len(candidates) != len(references):
        raise ValueError("Missing utility judgments")
    for annotation, candidate, reference in zip(annotations, candidates, references):
        if (any(annotation[k] != candidate[k] or candidate[k] != reference[k] for k in identity_keys)
                or annotation["output_1"] != reference["output"]
                or annotation["output_2"] != candidate["output"]
                or annotation["generator_1"] != reference["generator"]
                or annotation["generator_2"] != candidate["generator"]):
            raise ValueError("Utility annotation rows or answer roles are misaligned")
        pref = annotation.get("preference")
        if not isinstance(pref, (float, int)) or not math.isfinite(pref) or not 1 <= pref <= 2:
            raise ValueError("Missing or invalid utility preference; refusing a partial score")


def score_annotations(suite, annotations, *, metric_asset: Path | None = None,
                      metric_asset_sha256: str | None = None):
    import pandas as pd
    from alpaca_eval.metrics import get_winrate
    if suite == "sep":
        return get_winrate(annotations)
    if suite != "alpacaeval2" or len(annotations) != 805:
        raise ValueError("AlpacaEval length control requires all 805 ordered examples")
    if metric_asset is None or sha256_file(metric_asset) != metric_asset_sha256:
        raise ValueError("AlpacaEval length-control asset is absent or changed")
    from alpaca_eval.metrics import glm_winrate
    # The package exposes no local-file argument. Bind only its auxiliary data
    # resolver to the verified local copy; the fit and scoring code are unchanged.
    original_download = glm_winrate.hf_hub_download
    def local_asset(*, repo_id, filename, **kwargs):
        if repo_id != "tatsu-lab/alpaca_eval" or filename != "df_gamed.csv":
            raise ValueError("Unexpected length-control data dependency")
        return str(metric_asset)
    try:
        glm_winrate.hf_hub_download = local_asset
        return glm_winrate.get_length_controlled_winrate(pd.DataFrame(annotations), save_weights_dir=None)
    finally:
        glm_winrate.hf_hub_download = original_download
