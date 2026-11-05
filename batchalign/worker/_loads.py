"""The hosts a worker's loads build, and the one key per load.

Every model a worker's loads build is an entry of ``_state.loads``
(:class:`~batchalign.worker._types.TaskLoadRegistry`), the Python mirror of
Rust's ``TaskLoads``. An entry is the host its load built: the model itself
(inside the runner that closes over it, never a copy) and the identity of what
loaded, read by whoever needs it from the host rather than from a second
field. The keys below are the only :class:`LoadKey` values; each is typed by
the host its load builds, so the registry pairs them by type::

    ensure_task(task) --task_load--> TaskLoad --load--> host
                                                  |
                                     registry.record(KEY, host)   (the only writer)

    request --its KEY--> registry.get(KEY) --absent--> model_unavailable
                                             --present--> handler(request, host)

A host holds no ``Optional`` slot: a load that built nothing is a host of its
own (:class:`ControlPlaneAsr`, the pyannoteAI :class:`LoadedDiarization`), and a
task that never loaded has no entry.

Two model stores sit outside the registry, because no load builds them: the
bounded per-language Stanza cache (``_state.stanza_pipelines``), filled on a
request's first use of a language, and the word segmenter's tokenizer, which
``batchalign.inference.word_segmentation`` builds on first use and caches.

Memory: an entry is the model its load built and nothing more; the registry
adds a reference, never a copy. The coreference entry holds a
:class:`~batchalign.inference.coref.PreparedCoref`, which owns no pipeline: each
batch builds the pipeline and releases it, so the registry holds no reference
to the coreference model between batches (what the process keeps after a
release is measured in ``worker/residency.rs`` on the Rust side).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, Protocol

from batchalign.inference._admission import _FactoryAdmission
from batchalign.worker._types import LoadKey, TaskLoad
from batchalign.worker._types_v2 import (
    AsrBackendV2,
    FaBackendV2,
    SpeakerBackendV2,
    SpeakerEmbeddingBackendV2,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from batchalign.inference._domain_types import LanguageCode
    from batchalign.inference.asr import AsrBatchItem, MonologueAsrResponse
    from batchalign.inference.avqi import AvqiResponse
    from batchalign.inference.languages.cantonese._cantonese_fa import CantoneseFaHost
    from batchalign.inference.opensmile import OpenSmileResponse
    from batchalign.inference.qwen_forced_alignment import (
        IndexedWordTimingRow,
        LoadedQwenFaHost,
    )
    from batchalign.inference.speaker import SpeakerResponse
    from batchalign.inference.speaker_embedding import SpeakerEmbeddingResponse
    from batchalign.inference.translate import TranslateInferenceRequest
    from batchalign.inference.types import (
        Wave2VecFAHandle,
        Wave2VecFAResult,
        WhisperFAHandle,
    )
    from batchalign.inference.utseg import UtsegInferenceRequest
    from batchalign.worker._types import BatchInferRequest, BatchInferResponse
    from batchalign.worker._types_v2 import (
        AsrModelIdentityV2,
        FaPreparationOutcomeV2,
        WhisperChunkResultPayloadV2,
        WordSegmentationResultPayloadV2,
    )

    # Runner shapes, named for the annotations below. Only under TYPE_CHECKING:
    # every annotation here is a string (`from __future__ import annotations`),
    # and the types they name import torch-heavy modules.
    WhisperFaRunner = Callable[[np.ndarray, str, str | None], list[tuple[str, float]]]
    """Whisper FA: audio, the shaped transcript, and the attention-capture
    context (``None`` when the request asked for no capture), to token
    onsets."""

    WordFaRunner = Callable[
        [np.ndarray, list[str]], Wave2VecFAResult | list[IndexedWordTimingRow]
    ]
    FaPreparer = Callable[[str, list[str], str], FaPreparationOutcomeV2]
    """Wave2Vec, Cantonese and Qwen3 FA: audio and the requested words, to one
    timing row per word."""

    DiarizationRunner = Callable[[np.ndarray, int, int | None], SpeakerResponse]
    """Prepared audio, its sample rate and the requested speaker count
    (``None`` asks the diarizer to estimate), to the diarization evidence."""

    ProgressCallback = Callable[[int, int], None]
    """Items completed and total, for a request's progress events."""


# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoadedLocalAsr:
    """A Whisper checkpoint (stock or a fine-tune) loaded in this process.

    ``runner`` transcribes prepared mono audio and stamps the handle's own
    identity on its payload. ``backend`` and ``model_identity`` are what the
    Rust executor reads from this host: the request's backend is checked
    against ``backend``.
    """

    backend: Literal[AsrBackendV2.LOCAL_WHISPER, AsrBackendV2.WHISPER_HUB]
    runner: Callable[[np.ndarray, LanguageCode], WhisperChunkResultPayloadV2]
    model_identity: AsrModelIdentityV2


@dataclass(frozen=True, slots=True)
class LoadedProviderAsr:
    """A Cantonese provider (Tencent, Aliyun, FunAudio, Qwen3-ASR) loaded here.

    ``runner`` transcribes one provider media item. ``model_identity`` is what
    the provider's load recorded, and the Rust executor admits it against the
    request's pin before spending a provider call.
    """

    backend: Literal[
        AsrBackendV2.HK_TENCENT,
        AsrBackendV2.HK_ALIYUN,
        AsrBackendV2.HK_FUNAUDIO,
        AsrBackendV2.HK_QWEN,
    ]
    runner: Callable[[AsrBatchItem], MonologueAsrResponse]
    model_identity: AsrModelIdentityV2


@dataclass(frozen=True, slots=True)
class ControlPlaneAsr:
    """The ASR load of a worker whose plan names no worker-hosted ASR engine.

    The control plane names the engine for every worker-hosted ASR load; it
    names none when the plan's ASR runs in Rust (Rev.AI, whisper.cpp), and it
    still prepares the command's task (the capability probe). That load builds
    nothing, and an ASR request reaching this worker is refused by name. It
    used to fall through to a guessed engine (Rev.AI when a key was in the
    environment, else FunAudio for Cantonese, else Whisper), loading a model
    the job never used.
    """


AsrLoad = LoadedLocalAsr | LoadedProviderAsr | ControlPlaneAsr
"""What an ASR load builds."""


# ---------------------------------------------------------------------------
# Forced alignment
# ---------------------------------------------------------------------------


class FaHostView(Protocol):
    """Read-only foreign-host attributes consumed by the native executor.

    Production loading still returns only the sealed ``FaLoad`` variants.
    This port lets boundary tests declare foreign hosts without forging one
    of those production model-loading proofs.
    """

    @property
    def backend(self) -> FaBackendV2: ...

    @property
    def engine(self) -> str: ...

    @property
    def runner(self) -> Callable[..., object]: ...

    @property
    def prepare(self) -> FaPreparer: ...


@dataclass(frozen=True, slots=True, init=False)
class LoadedWhisperFa(_FactoryAdmission):
    """The Whisper forced aligner loaded in this process.

    ``engine`` is the FA identity the capability report names, the namespace of
    every FA cache row (``FaCacheNamespace::from_loaded``, Rust side).
    """

    runner: WhisperFaRunner
    prepare: FaPreparer
    engine: str

    @classmethod
    def from_model(cls, handle: WhisperFAHandle) -> LoadedWhisperFa:
        from batchalign.inference.fa import WHISPER_FA_ENGINE_ID
        from batchalign.inference.fa_whisper_resources import whisper_request_preparer
        from batchalign.worker._fa_v2 import whisper_fa_runner

        value = object.__new__(cls)
        object.__setattr__(value, "runner", whisper_fa_runner(handle))
        object.__setattr__(value, "prepare", whisper_request_preparer(handle))
        object.__setattr__(value, "engine", WHISPER_FA_ENGINE_ID)
        return value

    @property
    def backend(self) -> FaBackendV2:
        """The backend this host serves: always Whisper."""
        return FaBackendV2.WHISPER


@dataclass(frozen=True, slots=True, init=False)
class LoadedWordFa(_FactoryAdmission):
    """A word-list forced aligner (Wave2Vec, Cantonese Wave2Vec, Qwen3).

    All three take the requested words and answer one row per word, so they
    share one call shape; ``backend`` says which loaded, and the Rust executor
    checks the request's against it. ``engine`` is the FA identity, as for
    :class:`LoadedWhisperFa`.
    """

    backend: Literal[FaBackendV2.WAVE2VEC, FaBackendV2.WAV2VEC_CANTO, FaBackendV2.QWEN3]
    runner: WordFaRunner
    prepare: FaPreparer
    engine: str

    @classmethod
    def from_wave2vec(cls, handle: Wave2VecFAHandle) -> LoadedWordFa:
        from importlib.metadata import version

        from batchalign.inference.fa import MMS_FA_ALGORITHM_ID
        from batchalign.inference.fa_resources import mms_request_preparer
        from batchalign.worker._fa_v2 import wave2vec_fa_runner

        value = object.__new__(cls)
        object.__setattr__(value, "backend", FaBackendV2.WAVE2VEC)
        object.__setattr__(value, "runner", wave2vec_fa_runner(handle))
        object.__setattr__(
            value, "prepare", mms_request_preparer(handle, lambda words: words)
        )
        object.__setattr__(
            value, "engine", f"{MMS_FA_ALGORITHM_ID}-{version('torchaudio')}"
        )
        return value

    @classmethod
    def from_cantonese(cls, host: CantoneseFaHost) -> LoadedWordFa:
        from batchalign.inference.fa_resources import mms_request_preparer
        from batchalign.inference.languages.cantonese._cantonese_fa import (
            CANTONESE_FA_ENGINE_ID,
        )
        from batchalign.worker._fa_v2 import cantonese_fa_runner

        value = object.__new__(cls)
        object.__setattr__(value, "backend", FaBackendV2.WAV2VEC_CANTO)
        object.__setattr__(value, "runner", cantonese_fa_runner(host))
        object.__setattr__(
            value, "prepare", mms_request_preparer(host.model, host.romanize_words)
        )
        object.__setattr__(value, "engine", CANTONESE_FA_ENGINE_ID)
        return value

    @classmethod
    def from_qwen(cls, host: LoadedQwenFaHost) -> LoadedWordFa:
        from batchalign.inference.fa_qwen_resources import qwen_request_preparer
        from batchalign.worker._fa_v2 import qwen_fa_runner

        value = object.__new__(cls)
        object.__setattr__(value, "backend", FaBackendV2.QWEN3)
        object.__setattr__(value, "runner", qwen_fa_runner(host))
        object.__setattr__(value, "prepare", qwen_request_preparer(host))
        object.__setattr__(value, "engine", host.cache_identity.wire_name)
        return value


FaLoad = LoadedWhisperFa | LoadedWordFa
"""What an FA load builds."""


# ---------------------------------------------------------------------------
# Speaker
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoadedDiarization:
    """A diarization engine: local pyannote (the pipeline its load built) or
    pyannoteAI (hosted, so its load built nothing and its runner calls the
    API). ``backend`` is checked against the request's by the Rust executor.
    """

    backend: SpeakerBackendV2
    runner: DiarizationRunner


@dataclass(frozen=True, slots=True)
class LoadedSpeakerEmbedding:
    """The speaker-embedding model its load built, inside ``runner``."""

    backend: SpeakerEmbeddingBackendV2
    runner: Callable[[np.ndarray, int, str], SpeakerEmbeddingResponse]


# ---------------------------------------------------------------------------
# Text tasks
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoadedMorphosyntax:
    """The morphosyntax runner over the worker's bounded Stanza cache.

    The progress callback is an argument of the call, not worker state: a V2
    request passes its own, the legacy batch route passes ``None`` (it has no
    progress channel).
    """

    runner: Callable[[BatchInferRequest, ProgressCallback | None], BatchInferResponse]


@dataclass(frozen=True, slots=True)
class LoadedUtseg:
    """The utseg runner over what its load built for the boundary model."""

    runner: Callable[[UtsegInferenceRequest], BatchInferResponse]


@dataclass(frozen=True, slots=True)
class LoadedTranslate:
    """The translation runner over the engine its load built."""

    runner: Callable[[TranslateInferenceRequest], BatchInferResponse]


@dataclass(frozen=True, slots=True)
class LoadedCoref:
    """The coreference runner over the prepared model.

    The runner closes over a ``PreparedCoref``, which holds the engine name
    and no pipeline: each batch builds the pipeline and releases it, so this
    entry keeps nothing resident between batches.
    """

    runner: Callable[[BatchInferRequest], BatchInferResponse]


# ---------------------------------------------------------------------------
# Model-free tasks
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoadedOpenSmile:
    """openSMILE has no model; its load records the feature extractor."""

    runner: Callable[[np.ndarray, int, str, str, str], OpenSmileResponse]


@dataclass(frozen=True, slots=True)
class LoadedAvqi:
    """AVQI has no model; its load records the voice-quality analysis."""

    runner: Callable[[np.ndarray, int, np.ndarray, int, str, str], AvqiResponse]


@dataclass(frozen=True, slots=True)
class LoadedWordSegmentation:
    """Word segmentation has no model; its load records the segmenter, which
    applies the one each request names."""

    runner: Callable[[str, str], WordSegmentationResultPayloadV2]


# ---------------------------------------------------------------------------
# The keys: one per load, typed by the host it builds
# ---------------------------------------------------------------------------

MORPHOSYNTAX_LOAD: Final = LoadKey[LoadedMorphosyntax](TaskLoad.MORPHOSYNTAX)
UTSEG_LOAD: Final = LoadKey[LoadedUtseg](TaskLoad.UTSEG)
TRANSLATE_LOAD: Final = LoadKey[LoadedTranslate](TaskLoad.TRANSLATE)
COREF_LOAD: Final = LoadKey[LoadedCoref](TaskLoad.COREF)
FA_LOAD: Final = LoadKey[FaLoad](TaskLoad.FA)
ASR_LOAD: Final = LoadKey[AsrLoad](TaskLoad.ASR)
OPENSMILE_LOAD: Final = LoadKey[LoadedOpenSmile](TaskLoad.OPENSMILE)
AVQI_LOAD: Final = LoadKey[LoadedAvqi](TaskLoad.AVQI)
WORD_SEGMENTATION_LOAD: Final = LoadKey[LoadedWordSegmentation](
    TaskLoad.WORD_SEGMENTATION
)
SPEAKER_HOSTED_LOAD: Final = LoadKey[LoadedDiarization](TaskLoad.SPEAKER_HOSTED)
SPEAKER_LOCAL_PYANNOTE_LOAD: Final = LoadKey[LoadedDiarization](
    TaskLoad.SPEAKER_LOCAL_PYANNOTE
)
SPEAKER_EMBEDDING_LOAD: Final = LoadKey[LoadedSpeakerEmbedding](
    TaskLoad.SPEAKER_EMBEDDING
)


def diarization_load(backend: SpeakerBackendV2) -> LoadKey[LoadedDiarization]:
    """The load a diarization request needs: its backend's."""
    match backend:
        case SpeakerBackendV2.PYANNOTE:
            return SPEAKER_LOCAL_PYANNOTE_LOAD
        case SpeakerBackendV2.PYANNOTE_AI:
            return SPEAKER_HOSTED_LOAD


__all__ = [
    "ASR_LOAD",
    "AVQI_LOAD",
    "COREF_LOAD",
    "FA_LOAD",
    "MORPHOSYNTAX_LOAD",
    "OPENSMILE_LOAD",
    "SPEAKER_EMBEDDING_LOAD",
    "SPEAKER_HOSTED_LOAD",
    "SPEAKER_LOCAL_PYANNOTE_LOAD",
    "TRANSLATE_LOAD",
    "UTSEG_LOAD",
    "WORD_SEGMENTATION_LOAD",
    "AsrLoad",
    "ControlPlaneAsr",
    "FaLoad",
    "LoadedAvqi",
    "LoadedCoref",
    "LoadedDiarization",
    "LoadedLocalAsr",
    "LoadedMorphosyntax",
    "LoadedOpenSmile",
    "LoadedProviderAsr",
    "LoadedSpeakerEmbedding",
    "LoadedTranslate",
    "LoadedUtseg",
    "LoadedWhisperFa",
    "LoadedWordFa",
    "LoadedWordSegmentation",
    "diarization_load",
]
