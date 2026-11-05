"""Live worker-protocol V2 execution boundary.

This module is the narrow typed entrypoint for live V2 execution over the
existing stdio worker process. Its job is intentionally small:

- resolve one request's load in the worker's load registry, ONCE, here
- refuse a request whose load is absent, by name (``model_unavailable``)
- hand the task executor the host that load built

No executor sees an absent model: each takes the loaded host itself, so a
``None`` test inside a handler has nothing to test.

**See also:** `../../INTERFACE_MAP.md` for the unified Python/Rust interface
reference, including Rust FFI function signatures, shared schema definitions,
and full Python caller locations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, assert_never

if TYPE_CHECKING:
    from collections.abc import Callable

    from batchalign.worker._types import LoadKey, TaskLoadRegistry
    from batchalign.worker._types_v2 import (
        AsrModelIdentityV2,
        AsrRequestV2,
        TaskResultV2,
    )

import time

from batchalign.worker._asr_v2 import execute_asr_request_v2
from batchalign.worker._avqi_v2 import execute_avqi_request_v2
from batchalign.worker._fa_v2 import execute_forced_alignment_request_v2
from batchalign.worker._loads import (
    ASR_LOAD,
    AVQI_LOAD,
    COREF_LOAD,
    FA_LOAD,
    MORPHOSYNTAX_LOAD,
    OPENSMILE_LOAD,
    SPEAKER_EMBEDDING_LOAD,
    TRANSLATE_LOAD,
    UTSEG_LOAD,
    WORD_SEGMENTATION_LOAD,
    ControlPlaneAsr,
    LoadedLocalAsr,
    LoadedProviderAsr,
    diarization_load,
)
from batchalign.worker._opensmile_v2 import execute_opensmile_request_v2
from batchalign.worker._speaker_embedding_v2 import (
    execute_speaker_embedding_request_v2,
)
from batchalign.worker._speaker_v2 import execute_speaker_request_v2
from batchalign.worker._text_v2 import (
    execute_coref_request_v2,
    execute_morphosyntax_request_v2,
    execute_translate_request_v2,
    execute_utseg_request_v2,
    execute_word_segmentation_request_v2,
)
from batchalign.worker._types import _state, not_loaded_message, sleep_test_delay
from batchalign.worker._types_v2 import (
    AsrRequestV2,
    AvqiRequestV2,
    CorefRequestV2,
    ExecuteErrorV2,
    ExecuteRequestV2,
    ExecuteResponseV2,
    ForcedAlignmentRequestV2,
    MorphosyntaxRequestV2,
    OpenSmileRequestV2,
    ProtocolErrorCodeV2,
    SpeakerEmbeddingRequestV2,
    SpeakerRequestV2,
    TranslateRequestV2,
    UtsegRequestV2,
    WordSegmentationRequestV2,
)

Host = TypeVar("Host")


def execute_request_v2(request: ExecuteRequestV2) -> ExecuteResponseV2:
    """Execute one typed V2 worker request against this worker's loads.

    The protocol loop's entry point: the one place the process-wide registry
    is read. Everything else takes the registry as an argument
    (:func:`execute_on_loads`), which is how tests hand it the hosts they
    built.
    """
    return execute_on_loads(request, _state.loads)


def execute_on_loads(
    request: ExecuteRequestV2, loads: TaskLoadRegistry
) -> ExecuteResponseV2:
    """Execute one typed V2 request on the hosts ``loads`` holds.

    Every response this function builds itself (a refused request, a test
    echo) reports the time from receipt to reply, measured here; none of them
    invents one.
    """
    started_at = time.monotonic()

    invalid_request_response = _validate_request_boundary(request, started_at)
    if invalid_request_response is not None:
        return invalid_request_response

    # Test-echo mode: return a successful echo response without model dispatch.
    # This enables integration tests for the concurrent dispatch path
    # (SharedGpuWorker) without loading real ML models.
    if _state.test_echo:
        from batchalign.worker._types_v2 import ExecuteSuccessV2

        sleep_test_delay()

        # The payload is an EMPTY placeholder of the requested task's own
        # result kind: transport tests read only the request_id, but the
        # protocol makes success-with-no-result unrepresentable (the Rust
        # reader refuses it at deserialization since 2026-08-21), and a
        # double that lies about the contract hangs every dispatch waiting
        # on its reply. Deriving the kind from the task keeps downstream
        # parsers honest; an echo answer must still never be READ as a
        # result, which is why echo-mode server jobs are short-circuited
        # before dispatch.
        return ExecuteResponseV2(
            request_id=request.request_id,
            outcome=ExecuteSuccessV2(),
            result=_echo_placeholder_result(request),
            elapsed_s=_seconds_since(started_at),
        )

    def on(
        key: LoadKey[Host],
        execute: Callable[[ExecuteRequestV2, Host], ExecuteResponseV2],
    ) -> ExecuteResponseV2:
        # The one resolution: absent is the named refusal, present is the
        # host the executor receives.
        host = loads.get(key)
        if host is None:
            return _refusal(
                request,
                ProtocolErrorCodeV2.MODEL_UNAVAILABLE,
                not_loaded_message(key.load),
                started_at,
            )
        return execute(request, host)

    payload = request.payload
    match payload:
        case MorphosyntaxRequestV2():
            return on(MORPHOSYNTAX_LOAD, execute_morphosyntax_request_v2)
        case UtsegRequestV2():
            return on(UTSEG_LOAD, execute_utseg_request_v2)
        case TranslateRequestV2():
            return on(TRANSLATE_LOAD, execute_translate_request_v2)
        case CorefRequestV2():
            return on(COREF_LOAD, execute_coref_request_v2)
        case WordSegmentationRequestV2():
            return on(WORD_SEGMENTATION_LOAD, execute_word_segmentation_request_v2)
        case AsrRequestV2(backend=asr_backend):

            def run_asr(
                request: ExecuteRequestV2,
                host: LoadedLocalAsr | LoadedProviderAsr | ControlPlaneAsr,
            ) -> ExecuteResponseV2:
                match host:
                    case LoadedLocalAsr() | LoadedProviderAsr():
                        return execute_asr_request_v2(request, host)
                    case ControlPlaneAsr():
                        return _refusal(
                            request,
                            ProtocolErrorCodeV2.MODEL_UNAVAILABLE,
                            "this worker hosts no ASR engine: its plan's ASR "
                            f"runs in the control plane, so a {asr_backend.value} "
                            "request cannot run here",
                            started_at,
                        )
                    case _:
                        assert_never(host)

            return on(ASR_LOAD, run_asr)
        case ForcedAlignmentRequestV2():
            return on(FA_LOAD, execute_forced_alignment_request_v2)
        case SpeakerRequestV2(backend=speaker_backend):
            return on(diarization_load(speaker_backend), execute_speaker_request_v2)
        case SpeakerEmbeddingRequestV2():
            return on(SPEAKER_EMBEDDING_LOAD, execute_speaker_embedding_request_v2)
        case OpenSmileRequestV2():
            return on(OPENSMILE_LOAD, execute_opensmile_request_v2)
        case AvqiRequestV2():
            return on(AVQI_LOAD, execute_avqi_request_v2)
        case _:
            assert_never(payload)


def _echo_model_identity(request: AsrRequestV2) -> AsrModelIdentityV2:
    """The identity an echo worker can honestly report: nothing observed.

    An echo worker loads no model at all, so every observation is an explicit
    absence and the requested half is the request's own pin, echoed unchanged.

    That makes the result deliberately INADMISSIBLE wherever the plan pinned an
    exact commit: `AsrModelIdentityV2::admit` refuses a pinned commit that comes
    back unexposed. This is the point, not a defect. A transport double must be
    able to answer (success with no result is unrepresentable, and a double that
    lies about the contract hangs every dispatch waiting on its reply), but its
    answer must never be readable as a real transcript. Shaping it so the
    admission check rejects it by name is a stronger guarantee than the comment
    that used to say echo answers are short-circuited before dispatch.
    """

    from batchalign.worker._types_v2 import (
        AliyunModelIdentityV2,
        AliyunModelsV2,
        LoadedModelV2,
        ObservedNotExposedV2,
        ParaformerModelIdentityV2,
        ParaformerModelsV2,
        QwenModelIdentityV2,
        QwenModelsV2,
        RequestedModelV2,
        RevModelIdentityV2,
        RevModelsV2,
        SenseVoiceModelIdentityV2,
        SenseVoiceModelsV2,
        TencentModelIdentityV2,
        TencentModelsV2,
        WhisperModelIdentityV2,
        WhisperModelsV2,
    )

    def unobserved(member: RequestedModelV2) -> LoadedModelV2:
        return LoadedModelV2(
            id=member.id,
            requested=member.revision,
            observed=ObservedNotExposedV2(),
        )

    models = request.models
    match models:
        case WhisperModelsV2():
            return WhisperModelIdentityV2(asr=unobserved(models.asr))
        case QwenModelsV2():
            return QwenModelIdentityV2(
                asr=unobserved(models.asr), aligner=unobserved(models.aligner)
            )
        case ParaformerModelsV2():
            return ParaformerModelIdentityV2(
                asr=unobserved(models.asr),
                vad=unobserved(models.vad),
                punc=unobserved(models.punc),
            )
        case SenseVoiceModelsV2():
            return SenseVoiceModelIdentityV2(
                asr=unobserved(models.asr), vad=unobserved(models.vad)
            )
        case TencentModelsV2():
            return TencentModelIdentityV2(
                engine_model_type=unobserved(models.engine_model_type)
            )
        case AliyunModelsV2():
            return AliyunModelIdentityV2(service=unobserved(models.service))
        case RevModelsV2():
            return RevModelIdentityV2(provider=unobserved(models.provider))
        case _:
            assert_never(models)


def _echo_placeholder_result(request: ExecuteRequestV2) -> TaskResultV2:
    """Derive an empty result from the typed request payload itself."""

    from batchalign.worker._types_v2 import (
        AsrRequestV2,
        AvqiRequestV2,
        AvqiResultPayloadV2,
        CorefRequestV2,
        CorefResultPayloadV2,
        FaCandidateV2,
        FaExecuteV2,
        FaNoModelMemoryV2,
        FaPreparationBindingV2,
        FaPreparationPlanV2,
        FaPreparationResultPayloadV2,
        FaPreparedV2,
        FaPrepareV2,
        FaProjectionUnavailableV2,
        FaRefusedV2,
        ForcedAlignmentRequestV2,
        IndexedWordTimingResultPayloadV2,
        LocalPyannoteSpeakerEvidenceV2,
        MonologueAsrResultPayloadV2,
        MorphosyntaxRequestV2,
        MorphosyntaxResultPayloadV2,
        OpenSmileRequestV2,
        OpenSmileResultPayloadV2,
        PreparedAudioRefV2,
        PyannoteAISpeakerEvidenceV2,
        SpanTooShortForEmbeddingV2,
        SpeakerBackendV2,
        SpeakerEmbeddingRequestV2,
        SpeakerEmbeddingResultPayloadV2,
        SpeakerEmbeddingSpanResultV2,
        SpeakerRequestV2,
        SpeakerResultPayloadV2,
        TranslateRequestV2,
        TranslationResultPayloadV2,
        UtsegRequestV2,
        UtsegResultPayloadV2,
        WordSegmentationRequestV2,
        WordSegmentationResultPayloadV2,
    )

    payload = request.payload
    match payload:
        case MorphosyntaxRequestV2():
            return MorphosyntaxResultPayloadV2(items=[])
        case UtsegRequestV2():
            return UtsegResultPayloadV2(items=[])
        case TranslateRequestV2():
            return TranslationResultPayloadV2(items=[])
        case CorefRequestV2():
            return CorefResultPayloadV2(items=[])
        case WordSegmentationRequestV2():
            return WordSegmentationResultPayloadV2(segmenter="test-echo", items=[])
        case AsrRequestV2():
            return MonologueAsrResultPayloadV2(
                lang="eng", monologues=[], model=_echo_model_identity(payload)
            )
        case ForcedAlignmentRequestV2():
            match payload.phase:
                case FaPrepareV2(execution_request_id=execution_id):
                    audio = next(
                        (
                            item
                            for item in request.attachments
                            if isinstance(item, PreparedAudioRefV2)
                            and item.id == payload.audio_ref_id
                        ),
                        None,
                    )
                    if audio is None:
                        return FaPreparationResultPayloadV2(
                            outcome=FaRefusedV2(
                                cause=FaProjectionUnavailableV2(
                                    message="test echo has no matching audio descriptor"
                                )
                            )
                        )
                    return FaPreparationResultPayloadV2(
                        outcome=FaPreparedV2(
                            plan=FaPreparationPlanV2(
                                binding=FaPreparationBindingV2(
                                    execution_request_id=execution_id,
                                    backend=payload.backend,
                                    engine="test-echo",
                                    payload_ref_id=payload.payload_ref_id,
                                    audio_ref_id=payload.audio_ref_id,
                                    samples=audio.frame_count,
                                    sample_rate_hz=audio.sample_rate_hz,
                                    text_mode=payload.text_mode,
                                    attention_capture=payload.attention_capture,
                                ),
                                memory=FaNoModelMemoryV2(),
                            )
                        )
                    )
                case FaExecuteV2():
                    return IndexedWordTimingResultPayloadV2(indexed_timings=[])
                case FaCandidateV2():
                    return FaPreparationResultPayloadV2(
                        outcome=FaRefusedV2(
                            cause=FaProjectionUnavailableV2(
                                message="an FA candidate cannot execute"
                            )
                        )
                    )
                case _:
                    assert_never(payload.phase)
        case SpeakerRequestV2(backend=backend):
            match backend:
                case SpeakerBackendV2.PYANNOTE_AI:
                    return SpeakerResultPayloadV2(
                        evidence=PyannoteAISpeakerEvidenceV2(
                            job_id="echo-job",
                            output={"exclusiveDiarization": []},
                        )
                    )
                case SpeakerBackendV2.PYANNOTE:
                    return SpeakerResultPayloadV2(
                        evidence=LocalPyannoteSpeakerEvidenceV2(segments=[])
                    )
                case _:
                    assert_never(backend)
        case SpeakerEmbeddingRequestV2(spans=spans):
            # Every requested span is echoed back by NAME, refused as
            # unmeasurable. That is the only honest placeholder available: an
            # echo worker holds no model, so it has measured nothing, and
            # returning a vector would put a fabricated acoustic identity into
            # a transport test's reach. Echoing the ids keeps the double
            # exercising the one property the real seam guarantees, which is
            # that the answer set equals the question set.
            return SpeakerEmbeddingResultPayloadV2(
                dimension=1,
                minimum_frames=1,
                spans=[
                    SpeakerEmbeddingSpanResultV2(
                        span_id=span.span_id,
                        outcome=SpanTooShortForEmbeddingV2(
                            frame_count=span.end_frame - span.start_frame
                        ),
                    )
                    for span in spans
                ],
            )
        case OpenSmileRequestV2():
            return OpenSmileResultPayloadV2(
                feature_set="echo",
                feature_level="echo",
                num_features=0,
                duration_segments=0,
                audio_file="echo",
                rows=[],
                success=True,
            )
        case AvqiRequestV2():
            return AvqiResultPayloadV2(
                avqi=0.0,
                cpps=0.0,
                hnr=0.0,
                shimmer_local=0.0,
                shimmer_local_db=0.0,
                slope=0.0,
                tilt=0.0,
                cs_file="echo",
                sv_file="echo",
                success=True,
            )
        case _:
            assert_never(payload)


def _seconds_since(started_at: float) -> float:
    """Seconds from a ``time.monotonic()`` reading to now."""
    return time.monotonic() - started_at


def _refusal(
    request: ExecuteRequestV2,
    code: ProtocolErrorCodeV2,
    message: str,
    started_at: float,
) -> ExecuteResponseV2:
    """One typed refusal this module builds itself, timed from receipt."""

    return ExecuteResponseV2(
        request_id=request.request_id,
        outcome=ExecuteErrorV2(code=code, message=message),
        result=None,
        elapsed_s=_seconds_since(started_at),
    )


def _validate_request_boundary(
    request: ExecuteRequestV2, started_at: float
) -> ExecuteResponseV2 | None:
    """Reject mismatched top-level task/payload combinations before dispatch."""

    payload_kind = getattr(request.payload, "kind", None)
    if payload_kind is None:
        return _refusal(
            request,
            ProtocolErrorCodeV2.INVALID_PAYLOAD,
            "execute payload did not include a task kind discriminator",
            started_at,
        )
    if request.task.value != payload_kind:
        return _refusal(
            request,
            ProtocolErrorCodeV2.INVALID_PAYLOAD,
            f"execute payload kind {payload_kind} does not match task {request.task.value}",
            started_at,
        )
    return None


__all__ = [
    "execute_on_loads",
    "execute_request_v2",
]
