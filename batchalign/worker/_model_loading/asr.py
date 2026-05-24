"""ASR-engine bootstrap helpers for worker startup."""

from __future__ import annotations

import logging
import os
import typing
from collections.abc import Mapping

from batchalign.inference._domain_types import RevAiApiKey
from batchalign.inference.asr import iso3_to_language_name
from batchalign.worker._types import AsrEngine, WorkerBootstrapRuntime, _state

L = logging.getLogger("batchalign.worker")


def load_asr_engine(bootstrap: WorkerBootstrapRuntime) -> None:
    """Load the ASR engine for this worker.

    The control plane may inject a resolved Rev.AI key directly into the worker
    bootstrap runtime. When it does, that injected value is authoritative and
    the worker does not rediscover credentials from ambient process state.

    Dispatches on the resolved ``AsrEngine`` so adding a new variant later
    forces a missing-arm error rather than silently loading Whisper.
    """
    lang = bootstrap.lang
    engine_overrides = bootstrap.engine_overrides or None
    rev_api_key = bootstrap.revai_api_key
    _state.rev_api_key = None

    backend = resolve_asr_engine(engine_overrides, rev_api_key)

    if backend is AsrEngine.REV:
        _state.rev_api_key = rev_api_key
        if rev_api_key is None:
            L.error("Rev.AI key not configured")
        _state.asr_engine = AsrEngine.REV
    elif backend is AsrEngine.TENCENT:
        from batchalign.inference.languages.cantonese._tencent_asr import load_tencent_asr

        load_tencent_asr(lang, engine_overrides)
        _state.asr_engine = AsrEngine.TENCENT
    elif backend is AsrEngine.ALIYUN:
        from batchalign.inference.languages.cantonese._aliyun_asr import load_aliyun_asr

        load_aliyun_asr(lang, engine_overrides)
        _state.asr_engine = AsrEngine.ALIYUN
    elif backend is AsrEngine.FUNAUDIO:
        from batchalign.inference.languages.cantonese._funaudio_asr import load_funaudio_asr

        load_funaudio_asr(lang, engine_overrides)
        _state.asr_engine = AsrEngine.FUNAUDIO
    elif backend is AsrEngine.WHISPER_HUB:
        # Community HF Whisper fine-tune loaded by model_id. Resolution
        # and the "unknown language" error path live in
        # ``batchalign.inference.whisper_hub``; the returned handle is
        # the same ``WhisperASRHandle`` stock Whisper uses, so downstream
        # V2 inference requires no branching on engine identity after
        # load time.
        from batchalign.inference.whisper_hub import load_whisper_hub_asr

        _state.whisper_asr_model = load_whisper_hub_asr(
            lang,
            engine_overrides,
            device_policy=bootstrap.device_policy,
        )
        _state.asr_engine = AsrEngine.WHISPER_HUB
    elif backend is AsrEngine.WHISPER:
        from batchalign.inference.asr import load_whisper_asr

        language = iso3_to_language_name(lang)
        # When auto-detecting, always use the multilingual model with no
        # language-specific overrides so Whisper detects per-segment.
        if language == "auto":
            _state.whisper_asr_model = load_whisper_asr(
                model="openai/whisper-large-v3",
                base="openai/whisper-large-v3",
                language="auto",
                device_policy=bootstrap.device_policy,
            )
        else:
            _state.whisper_asr_model = load_whisper_asr(
                language=language,
                device_policy=bootstrap.device_policy,
            )
        _state.asr_engine = AsrEngine.WHISPER
    else:
        # Exhaustive match. ``typing.assert_never`` makes the type
        # checker prove this branch is unreachable; if a new AsrEngine
        # variant is added without a load arm here, mypy / pyright
        # flags it at compile time. At runtime it raises ``AssertionError``
        # so a regression still fails loudly instead of silently
        # falling through.
        typing.assert_never(backend)


def resolve_asr_engine(
    engine_overrides: dict[str, str] | None,
    rev_api_key: RevAiApiKey | None,
) -> AsrEngine:
    """Resolve which ASR engine this worker should load.

    Precedence:

    1. Explicit engine override from the Rust control plane. Unknown
       wire strings raise ``ValueError`` rather than silently loading
       Whisper — a typo in a per-host override would otherwise produce
       wrong-model output.
    2. Rev.AI when a key is available.
    3. Local Whisper fallback.
    """
    if engine_overrides and "asr" in engine_overrides:
        choice = engine_overrides["asr"]
        try:
            return AsrEngine(choice)
        except ValueError as exc:
            supported = ", ".join(e.value for e in AsrEngine)
            raise ValueError(
                f"unknown asr engine {choice!r}; expected one of: {supported}"
            ) from exc
    return AsrEngine.REV if rev_api_key else AsrEngine.WHISPER


def resolve_injected_revai_api_key(
    environ: Mapping[str, str] | None = None,
) -> RevAiApiKey | None:
    """Resolve a pre-injected Rev.AI key from an explicit environment mapping."""
    env = environ if environ is not None else os.environ
    for key_name in ("BATCHALIGN_REV_API_KEY", "REVAI_API_KEY"):
        env_value = env.get(key_name)
        if env_value and env_value.strip():
            return env_value.strip()
    return None

__all__ = [
    "iso3_to_language_name",
    "load_asr_engine",
    "resolve_injected_revai_api_key",
    "resolve_asr_engine",
]
