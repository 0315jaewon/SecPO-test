"""Fail-closed environment construction for external subprocesses.

Python's subprocess APIs inherit the complete parent environment by default.
Research shells commonly contain credentials for unrelated providers, so the
released launchers instead copy a small set of operational variables and only
the additional names explicitly approved by their caller.

This module is dependency-free and performs no process or filesystem work.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import os
import re
import threading
from typing import Any


_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SENSITIVE_COMPONENT = re.compile(
    r"(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|CREDENTIALS)(?:_|$)",
    re.IGNORECASE,
)

# Exact operational names that are useful to Python/model processes and do not
# carry generic provider credentials. HOME is retained for tools that require
# it; DTap replaces it with a task-private directory before launching.
OPERATIONAL_ENVIRONMENT_NAMES = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "TMPDIR",
        "TMP",
        "TEMP",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONNOUSERSITE",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONPYCACHEPREFIX",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "CPATH",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
        "XDG_RUNTIME_DIR",
        "UV_CACHE_DIR",
        "UV_PYTHON",
        "UV_PYTHON_DOWNLOADS",
        "UV_NO_SYNC",
        "PIP_CACHE_DIR",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "GIT_SSL_CAINFO",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "DOCKER_CERT_PATH",
        "DOCKER_TLS_VERIFY",
    }
)

# CUDA/NCCL tuning is operational, but a name containing a credential marker
# remains excluded even under one of these prefixes.
OPERATIONAL_ENVIRONMENT_PREFIXES = (
    "CUDA_",
    "NCCL_",
    "TORCH_",
    "PYTORCH_",
    "ROCR_",
    "HIP_",
)

_ENVIRONMENT_ACTIVATION_LOCK = threading.RLock()


def is_sensitive_environment_name(name: str) -> bool:
    """Return whether a variable name conventionally denotes a credential."""

    return bool(_SENSITIVE_COMPONENT.search(name))


def _validate_name(name: Any, *, label: str) -> str:
    if not isinstance(name, str) or not _ENVIRONMENT_NAME.fullmatch(name):
        raise ValueError(f"{label} contains an invalid environment name: {name!r}")
    return name


def _validate_value(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"environment value for {name} must be a string")
    if "\x00" in value:
        raise ValueError(f"environment value for {name} must not contain NUL")
    return value


def build_subprocess_environment(
    *,
    source: Mapping[str, str] | None = None,
    pass_env: Sequence[str] = (),
    require_pass_env: bool = False,
    overrides: Mapping[str, str] | None = None,
    approved_sensitive_names: Sequence[str] = (),
) -> dict[str, str]:
    """Build a child environment without ambient credential inheritance.

    ``pass_env`` names are explicit caller approvals and may therefore include
    credentials. Missing names are omitted unless ``require_pass_env`` is true.
    Sensitive names in ``overrides`` require an explicit entry in either
    ``pass_env`` or ``approved_sensitive_names``.
    """

    parent = os.environ if source is None else source
    passed = tuple(_validate_name(name, label="pass_env") for name in pass_env)
    if len(set(passed)) != len(passed):
        raise ValueError("pass_env must not contain duplicate names")
    approved = {
        *passed,
        *(
            _validate_name(name, label="approved_sensitive_names")
            for name in approved_sensitive_names
        ),
    }

    environment: dict[str, str] = {}
    for name, value in parent.items():
        if not isinstance(name, str) or not _ENVIRONMENT_NAME.fullmatch(name):
            continue
        operational = name in OPERATIONAL_ENVIRONMENT_NAMES or name.startswith(
            OPERATIONAL_ENVIRONMENT_PREFIXES
        )
        if operational and not is_sensitive_environment_name(name):
            environment[name] = _validate_value(value, name=name)

    for name in passed:
        if name not in parent:
            if require_pass_env:
                raise RuntimeError(
                    f"explicitly requested subprocess environment variable is unset: {name}"
                )
            continue
        environment[name] = _validate_value(parent[name], name=name)

    for raw_name, raw_value in (overrides or {}).items():
        name = _validate_name(raw_name, label="overrides")
        if is_sensitive_environment_name(name) and name not in approved:
            raise ValueError(
                f"sensitive subprocess override requires explicit approval: {name}"
            )
        environment[name] = _validate_value(raw_value, name=name)
    return environment


@contextmanager
def isolated_environment(environment: Mapping[str, str]):
    """Temporarily expose exactly ``environment`` to in-process integrations.

    Environment mutation is process-global, so activations are serialized and
    the complete parent mapping is restored even when external code raises.
    Callers should keep the boundary as narrow as practical.
    """

    selected = {
        _validate_name(name, label="environment"): _validate_value(value, name=name)
        for name, value in environment.items()
    }
    with _ENVIRONMENT_ACTIVATION_LOCK:
        previous = dict(os.environ)
        os.environ.clear()
        os.environ.update(selected)
        try:
            yield selected
        finally:
            os.environ.clear()
            os.environ.update(previous)


__all__ = [
    "OPERATIONAL_ENVIRONMENT_NAMES",
    "OPERATIONAL_ENVIRONMENT_PREFIXES",
    "build_subprocess_environment",
    "isolated_environment",
    "is_sensitive_environment_name",
]
