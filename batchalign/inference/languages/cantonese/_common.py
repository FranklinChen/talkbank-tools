"""Shared helpers for HK/Cantonese engines."""

from __future__ import annotations

import configparser
import logging
import os
from collections.abc import Mapping
from typing import Any

from batchalign.config import config_read
from batchalign.errors import ConfigError

L = logging.getLogger("batchalign.hk")

# Type alias for engine overrides (replaces the old batchalign.providers.EngineOverrides import)
EngineOverrides = dict[str, str]

_ASR_ENV_KEYS: dict[str, str] = {
    "engine.tencent.id": "BATCHALIGN_TENCENT_ID",
    "engine.tencent.key": "BATCHALIGN_TENCENT_KEY",
    "engine.tencent.region": "BATCHALIGN_TENCENT_REGION",
    "engine.tencent.bucket": "BATCHALIGN_TENCENT_BUCKET",
    "engine.aliyun.ak_id": "BATCHALIGN_ALIYUN_AK_ID",
    "engine.aliyun.ak_secret": "BATCHALIGN_ALIYUN_AK_SECRET",
    "engine.aliyun.ak_appkey": "BATCHALIGN_ALIYUN_AK_APPKEY",
}


# Cantonese normalization does not live here, or anywhere in Python. It has one
# owner, ``AlignedNormalization`` in ``batchalign-transform``, which the Rust
# server applies once per monologue. These helpers used to expose the Rust
# normalization to Python, where nothing in production called them.


def read_asr_config(
    keys: tuple[str, ...],
    *,
    engine: str,
    config: configparser.ConfigParser | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Read required ASR provider keys from injected env or configuration.

    Worker-launched HK providers should receive resolved credentials from the
    Rust control plane via environment variables. Direct Python callers may
    still fall back to an explicit or ambient `~/.batchalign.ini`.
    """
    env = environ if environ is not None else os.environ
    resolved_from_env: dict[str, str] = {}
    for key in keys:
        env_name = _ASR_ENV_KEYS.get(key)
        if env_name is None:
            break
        env_value = env.get(env_name, "").strip()
        if not env_value:
            resolved_from_env = {}
            break
        resolved_from_env[key] = env_value
    if len(resolved_from_env) == len(keys):
        return resolved_from_env

    resolved_config = config if config is not None else config_read()
    if not resolved_config.has_section("asr"):
        raise ConfigError(
            "No [asr] section in ~/.batchalign.ini. "
            f"{engine} requires provider credentials in that file."
        )

    missing = [k for k in keys if not resolved_config.has_option("asr", k)]
    if missing:
        raise ConfigError(
            f"Missing {engine} config keys in ~/.batchalign.ini: {', '.join(missing)}"
        )

    values: dict[str, str] = {}
    for k in keys:
        value = resolved_config.get("asr", k).strip()
        if not value:
            raise ConfigError(f"Empty {engine} config value in ~/.batchalign.ini: {k}")
        values[k] = value
    return values


def parse_timestamp_pair(value: Any) -> tuple[int | None, int | None]:
    """Parse a FunASR timestamp item into milliseconds."""
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None, None
    try:
        start = int(round(float(value[0])))
        end = int(round(float(value[1])))
    except Exception:
        return None, None
    return start, end
