"""Forced-alignment engine bootstrap helpers for worker startup."""

from __future__ import annotations

import typing

from batchalign.inference.qwen_forced_alignment import QWEN_FORCED_ALIGNER_MODEL_ID
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
        from batchalign.inference.languages.cantonese._cantonese_fa import (
            load_cantonese_fa,
        )

        load_cantonese_fa(
            lang,
            engine_overrides,
            device_policy=bootstrap.device_policy,
        )
        _state.fa_engine = FaEngine.WAV2VEC_CANTO
        _state.fa_model_name = "wav2vec-canto-v1"
    elif backend is FaEngine.QWEN3:
        from batchalign.inference.qwen_forced_alignment import load_qwen_fa

        # The SAME aligner the Qwen3-ASR engine loads for its own word
        # timestamps, here on its own. `lang` fixes the model's language label
        # at bootstrap, so an unsupported language fails before the download
        # rather than after it.
        _state.qwen_fa_host = load_qwen_fa(lang, device_policy=bootstrap.device_policy)
        _state.fa_engine = FaEngine.QWEN3
        _state.fa_model_name = QWEN_FORCED_ALIGNER_MODEL_ID
    elif backend is FaEngine.WHISPER:
        from batchalign.inference.fa import load_whisper_fa

        _state.whisper_fa_model = load_whisper_fa(device_policy=bootstrap.device_policy)
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
        # Exhaustive match: see the equivalent comment in
        # ``_model_loading.asr.load_asr_engine``.
        typing.assert_never(backend)


# Legacy persistence wire names from the Rust control plane. Builds
# before 2026-06-12 serialized FA overrides with these names at the
# worker boundary (killing the worker at bootstrap; the four failed
# align jobs of 2026-06-11), and jobs persisted by those builds still
# carry the names in their stored options JSON. Accept them as aliases
# so a replayed or restarted job cannot kill a worker.
_LEGACY_FA_WIRE_NAMES: dict[str, FaEngine] = {
    "wav2vec_fa": FaEngine.WAVE2VEC,
    "whisper_fa": FaEngine.WHISPER,
    "cantonese_fa": FaEngine.WAV2VEC_CANTO,
}
# NOTE: `qwen3_fa` deliberately has no entry here. Its Rust persistence wire
# name and its dispatch name are the same string, so `FaEngine(choice)` already
# resolves it; an alias would be a second spelling of one name.


def resolve_fa_engine(engine_overrides: dict[str, str] | None) -> FaEngine:
    """Resolve which FA engine this worker should load.

    Precedence:

    1. Explicit engine override from the Rust control plane, accepting
       both the dispatch names (``FaEngine`` values) and the legacy
       persistence wire names (``_LEGACY_FA_WIRE_NAMES``). Unknown
       strings raise ``ValueError`` rather than silently loading
       Wave2Vec: a typo in a per-host override would otherwise
       produce wrong-model output.
    2. Otherwise a warm preload, because a missing key is not a choice.

    2. Nothing. There is no default, because there is no case left that needs
       one: the control plane names an engine for every task the worker's
       target preloads (`EngineSelection::for_target`), and a lazy-profile
       worker preloads nothing and names its engine per request. A missing
       ``fa`` key therefore means the boundary is broken, and guessing would
       load a model the request did not ask for.
    """
    if not engine_overrides or "fa" not in engine_overrides:
        raise ValueError(
            "no 'fa' engine in overrides; the control plane names one for every "
            "preloaded task, so its absence is a boundary bug rather than a "
            "request for a default"
        )
    choice = engine_overrides["fa"]
    legacy = _LEGACY_FA_WIRE_NAMES.get(choice)
    if legacy is not None:
        return legacy
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
