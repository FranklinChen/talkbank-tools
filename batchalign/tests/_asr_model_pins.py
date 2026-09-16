"""Pinned ASR model compositions for worker-protocol V2 tests.

Mirrors the resolved pins in ``crates/batchalign/src/model_manifest.rs``. Every V2
ASR test needs two agreeing halves: the composition a request pins, and the
loaded identity a worker reports for it. They must agree closely enough that
``AsrModelIdentityV2::admit`` accepts the pair, and hand-writing both at every
call site is exactly how two such copies drift apart.

Deliberately a plain module rather than a ``conftest.py`` fixture: the values are
wanted by ordinary helper functions as well as by tests, and a fixture can only
be reached from a test. The leading underscore keeps pytest from collecting it.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from batchalign.worker._types import _state
from batchalign.worker._types_v2 import (
    AliyunModelIdentityV2,
    AliyunModelsV2,
    AsrBackendV2,
    LoadedModelV2,
    ObservedCommitV2,
    ObservedNotExposedV2,
    QwenModelIdentityV2,
    QwenModelsV2,
    RequestedCommitV2,
    RequestedModelV2,
    RequestedProviderParameterV2,
    RequestedUnpinnedV2,
    RevModelIdentityV2,
    RevModelsV2,
    SenseVoiceModelIdentityV2,
    SenseVoiceModelsV2,
    TencentModelIdentityV2,
    TencentModelsV2,
    WhisperModelIdentityV2,
    WhisperModelsV2,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from batchalign.worker._types_v2 import AsrModelIdentityV2, AsrRequestedModelsV2

# The resolved revisions, copied from the Rust manifest. A copy, and knowingly
# so: these are test fixtures, and the Rust side pins the serialized bytes in
# `model_manifest::tests::the_pinned_composition_serializes_in_the_shape_python_parses`.
WHISPER_LARGE_V3 = "06f233fe06e710322aca913c1bc4249a0d71fce1"
SENSEVOICE = "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7"
FSMN_VAD = "df20e6b30c653645fa4ff125cacfcabd1020a669"
QWEN_ASR = "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"
QWEN_ALIGNER = "c07281df297b9905d24a508279258cccf987a064"

WHISPER_ID = "openai/whisper-large-v3"
SENSEVOICE_ID = "FunAudioLLM/SenseVoiceSmall"
FSMN_VAD_ID = "funasr/fsmn-vad"
QWEN_ASR_ID = "Qwen/Qwen3-ASR-1.7B-hf"
QWEN_ALIGNER_ID = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
TENCENT_ID = "tencent-asr"
TENCENT_PARAMETER = "16k_zh_large"
ALIYUN_ID = "aliyun-nls"
ALIYUN_PARAMETER = "speech-transcriber-v1"
REV_ID = "revai"
REV_PARAMETER = "asynchronous-transcript-v1"


def _pinned(model_id: str, commit: str) -> RequestedModelV2:
    """One model pinned to an exact hub commit."""

    return RequestedModelV2(id=model_id, revision=RequestedCommitV2(commit=commit))


def _selected(model_id: str, parameter: str) -> RequestedModelV2:
    """One cloud service selected by a provider parameter."""

    return RequestedModelV2(
        id=model_id, revision=RequestedProviderParameterV2(parameter=parameter)
    )


def _loaded_exactly(member: RequestedModelV2, commit: str) -> LoadedModelV2:
    """A member whose runtime reported the very commit that was asked for."""

    return LoadedModelV2(
        id=member.id,
        requested=member.revision,
        observed=ObservedCommitV2(commit=commit),
    )


def _loaded_unexposed(member: RequestedModelV2) -> LoadedModelV2:
    """A member whose runtime exposes no revision, as every cloud provider does."""

    return LoadedModelV2(
        id=member.id, requested=member.revision, observed=ObservedNotExposedV2()
    )


def request_models(backend: AsrBackendV2) -> AsrRequestedModelsV2:
    """The composition the control plane pins for one backend."""

    if backend is AsrBackendV2.LOCAL_WHISPER:
        return WhisperModelsV2(asr=_pinned(WHISPER_ID, WHISPER_LARGE_V3))
    if backend is AsrBackendV2.WHISPER_HUB:
        # whisper_hub seeds a fine-tune per language and seeds none for most, so
        # the realistic plan for an unseeded language is a floating identity.
        return WhisperModelsV2(
            asr=RequestedModelV2(id=WHISPER_ID, revision=RequestedUnpinnedV2())
        )
    if backend is AsrBackendV2.HK_TENCENT:
        return TencentModelsV2(
            engine_model_type=_selected(TENCENT_ID, TENCENT_PARAMETER)
        )
    if backend is AsrBackendV2.HK_ALIYUN:
        return AliyunModelsV2(service=_selected(ALIYUN_ID, ALIYUN_PARAMETER))
    if backend is AsrBackendV2.HK_FUNAUDIO:
        return SenseVoiceModelsV2(
            asr=_pinned(SENSEVOICE_ID, SENSEVOICE), vad=_pinned(FSMN_VAD_ID, FSMN_VAD)
        )
    if backend is AsrBackendV2.HK_QWEN:
        return QwenModelsV2(
            asr=_pinned(QWEN_ASR_ID, QWEN_ASR),
            aligner=_pinned(QWEN_ALIGNER_ID, QWEN_ALIGNER),
        )
    if backend is AsrBackendV2.REVAI:
        return RevModelsV2(provider=_selected(REV_ID, REV_PARAMETER))
    raise AssertionError(f"no pinned composition for ASR backend {backend!s}")


def loaded_identity(backend: AsrBackendV2) -> AsrModelIdentityV2:
    """What a worker reports after loading exactly what `request_models` pinned.

    Every pair this returns is one ``admit`` accepts: an exact commit comes back
    as that same commit, and a provider parameter comes back unexposed, which is
    the only observation a cloud API can honestly support.
    """

    models = request_models(backend)
    if isinstance(models, WhisperModelsV2):
        return WhisperModelIdentityV2(asr=_loaded_exactly(models.asr, WHISPER_LARGE_V3))
    if isinstance(models, TencentModelsV2):
        return TencentModelIdentityV2(
            engine_model_type=_loaded_unexposed(models.engine_model_type)
        )
    if isinstance(models, AliyunModelsV2):
        return AliyunModelIdentityV2(service=_loaded_unexposed(models.service))
    if isinstance(models, SenseVoiceModelsV2):
        return SenseVoiceModelIdentityV2(
            asr=_loaded_exactly(models.asr, SENSEVOICE),
            vad=_loaded_exactly(models.vad, FSMN_VAD),
        )
    if isinstance(models, QwenModelsV2):
        return QwenModelIdentityV2(
            asr=_loaded_exactly(models.asr, QWEN_ASR),
            aligner=_loaded_exactly(models.aligner, QWEN_ALIGNER),
        )
    if isinstance(models, RevModelsV2):
        return RevModelIdentityV2(provider=_loaded_unexposed(models.provider))
    raise AssertionError(f"no loaded identity for ASR backend {backend!s}")


@contextlib.contextmanager
def worker_loaded(backend: AsrBackendV2) -> Iterator[None]:
    """Record a loaded ASR identity in worker state for the duration of a test.

    Sets the same state the real engine loader sets, because the Rust bridge
    reads its load record from there before serving a provider request. Restores
    the previous value so one test cannot leak an identity into the next.
    """

    previous = _state.asr_model_identity
    _state.asr_model_identity = loaded_identity(backend)
    try:
        yield
    finally:
        _state.asr_model_identity = previous
