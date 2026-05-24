"""Forced-alignment engine bootstrap helpers for worker startup."""

from __future__ import annotations

import typing

from batchalign.worker._types import FaEngine, WorkerBootstrapRuntime, _state


def load_fa_engine(bootstrap: WorkerBootstrapRuntime) -> None:
    """Load the forced-alignment engine for this worker.

    Dispatches on the resolved ``FaEngine`` so adding a new variant later
    forces a missing-arm error rather than silently loading Wave2Vec.
    """
    lang = bootstrap.lang
    engine_overrides = bootstrap.engine_overrides or None
    backend = resolve_fa_engine(engine_overrides)

    if backend is FaEngine.WAV2VEC_CANTO:
        from batchalign.inference.languages.cantonese._cantonese_fa import load_cantonese_fa

        load_cantonese_fa(
            lang,
            engine_overrides,
            device_policy=bootstrap.device_policy,
        )
        _state.fa_engine = FaEngine.WAV2VEC_CANTO
        _state.fa_model_name = "wav2vec-canto-v1"
    elif backend is FaEngine.WHISPER:
        from batchalign.inference.fa import load_whisper_fa

        _state.whisper_fa_model = load_whisper_fa(
            device_policy=bootstrap.device_policy
        )
        _state.fa_engine = FaEngine.WHISPER
        _state.fa_model_name = "whisper-fa-large-v2"
    elif backend is FaEngine.WAVE2VEC:
        import importlib.metadata as importlib_metadata

        from batchalign.inference.fa import load_wave2vec_fa

        _state.wave2vec_fa_model = load_wave2vec_fa(
            device_policy=bootstrap.device_policy
        )
        _state.fa_engine = FaEngine.WAVE2VEC
        try:
            torchaudio_version = importlib_metadata.version("torchaudio")
            _state.fa_model_name = f"wave2vec-fa-mms-{torchaudio_version}"
        except importlib_metadata.PackageNotFoundError:
            _state.fa_model_name = "wave2vec-fa-mms"
    else:
        # Exhaustive match — see the equivalent comment in
        # ``_model_loading.asr.load_asr_engine``.
        typing.assert_never(backend)


def resolve_fa_engine(engine_overrides: dict[str, str] | None) -> FaEngine:
    """Resolve which FA engine this worker should load.

    Precedence:

    1. Explicit engine override from the Rust control plane. Unknown
       wire strings raise ``ValueError`` rather than silently loading
       Wave2Vec — a typo in a per-host override would otherwise
       produce wrong-model output.
    2. Default to Whisper FA, preserving historical behavior.
    """
    if not engine_overrides or "fa" not in engine_overrides:
        return FaEngine.WHISPER
    choice = engine_overrides["fa"]
    try:
        return FaEngine(choice)
    except ValueError as exc:
        supported = ", ".join(e.value for e in FaEngine)
        raise ValueError(
            f"unknown fa engine {choice!r}; expected one of: {supported}"
        ) from exc


__all__ = [
    "load_fa_engine",
    "resolve_fa_engine",
]
