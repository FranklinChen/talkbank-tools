"""The forced-alignment load: builds the FA host one worker's requests use."""

from __future__ import annotations

import typing

from batchalign.inference._domain_types import parse_choice
from batchalign.worker._loads import FaLoad, LoadedWhisperFa, LoadedWordFa
from batchalign.worker._model_loading.pinned_hub import (
    resolve_transformers,
    unpinned,
)
from batchalign.worker._types import FaEngine, WorkerBootstrapRuntime


def load_fa_engine(bootstrap: WorkerBootstrapRuntime) -> FaLoad:
    """Load the forced-alignment engine the control plane named, and return
    its host: the runner over the loaded model and the FA identity it reports.

    Dispatches on the resolved ``FaEngine`` so adding a new variant later
    forces a missing-arm error rather than silently loading Wave2Vec. The
    identity (``engine``) is the FA cache namespace on the Rust side, named
    by the arm that loaded the model, never guessed.
    """

    lang = bootstrap.lang
    engine_overrides = bootstrap.engine_overrides or None
    backend = resolve_fa_engine(engine_overrides)

    if backend is FaEngine.WAV2VEC_CANTO:
        from batchalign.inference.languages.cantonese._cantonese_fa import (
            load_cantonese_fa,
        )

        host = load_cantonese_fa(
            lang, engine_overrides, device_policy=bootstrap.device_policy
        )
        return LoadedWordFa.from_cantonese(host)
    elif backend is FaEngine.QWEN3:
        from batchalign.inference.qwen_forced_alignment import (
            QWEN_FORCED_ALIGNER_MODEL_ID,
            load_qwen_fa,
            resolve_qwen_language,
        )

        # The SAME aligner the Qwen3-ASR engine loads for its own word
        # timestamps, here on its own. The language label is resolved first,
        # and `load_qwen_fa` takes the label rather than the code, so an
        # unsupported language fails before the download by construction.
        language = resolve_qwen_language(lang)
        snapshot = resolve_transformers(
            unpinned(QWEN_FORCED_ALIGNER_MODEL_ID), kind="forced alignment"
        )
        qwen_host = load_qwen_fa(
            language, snapshot, device_policy=bootstrap.device_policy
        )
        return LoadedWordFa.from_qwen(qwen_host)
    elif backend is FaEngine.WHISPER:
        from batchalign.inference.fa import (
            WHISPER_FA_MODEL_ID,
            load_whisper_fa,
        )

        whisper_handle = load_whisper_fa(
            resolve_transformers(
                unpinned(WHISPER_FA_MODEL_ID), kind="forced alignment"
            ),
            device_policy=bootstrap.device_policy,
        )
        return LoadedWhisperFa.from_model(whisper_handle)
    elif backend is FaEngine.WAVE2VEC:
        from batchalign.inference.fa import load_wave2vec_fa

        model = load_wave2vec_fa(device_policy=bootstrap.device_policy)
        return LoadedWordFa.from_wave2vec(model)
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
    2. Nothing. There is no default: the control plane names the FA engine
       in every FA load (``ensure_task``), so a missing ``fa`` key means the
       boundary is broken, and guessing would load a model the request did
       not ask for.
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
    return parse_choice(FaEngine, choice, "fa engine")


__all__ = [
    "load_fa_engine",
    "resolve_fa_engine",
]
