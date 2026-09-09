"""Live worker-protocol V2 execution boundary.

This module is the narrow typed entrypoint for live V2 execution over the
existing stdio worker process. Its job is intentionally small:

- build model-host adapters from already loaded worker state
- route one typed V2 execute request to the correct task executor
- return one typed V2 execute response

The production worker loop should not assemble FA hosts or branch on loaded
model state inline. Keeping that wiring here makes the dispatch from
``BatchInferRequest`` payloads to typed V2 execute requests explicit and testable.

**See also:** `../../INTERFACE_MAP.md` for the unified Python/Rust interface
reference, including Rust FFI function signatures, shared schema definitions,
and full Python caller locations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

if TYPE_CHECKING:
    from batchalign.inference.languages.cantonese._cantonese_fa import CantoneseFaHost
    from batchalign.worker._types_v2 import TaskRequestV2, TaskResultV2

from dataclasses import dataclass, field

from batchalign.worker._asr_v2 import (
    AsrExecutionHostV2,
    build_default_asr_execution_host_v2,
    execute_asr_request_v2,
)
from batchalign.worker._avqi_v2 import (
    AvqiExecutionHostV2,
    build_default_avqi_execution_host_v2,
    execute_avqi_request_v2,
)
from batchalign.worker._fa_v2 import (
    ForcedAlignmentExecutionHostV2,
    build_default_fa_execution_host_v2,
    execute_forced_alignment_request_v2,
)
from batchalign.worker._opensmile_v2 import (
    OpenSmileExecutionHostV2,
    build_default_opensmile_execution_host_v2,
    execute_opensmile_request_v2,
)
from batchalign.worker._speaker_embedding_v2 import (
    SpeakerEmbeddingExecutionHostV2,
    build_default_speaker_embedding_execution_host_v2,
    execute_speaker_embedding_request_v2,
)
from batchalign.worker._speaker_v2 import (
    SpeakerExecutionHostV2,
    build_default_speaker_execution_host_v2,
    execute_speaker_request_v2,
)
from batchalign.worker._text_v2 import (
    TextExecutionHostV2,
    build_default_text_execution_host_v2,
    execute_coref_request_v2,
    execute_morphosyntax_request_v2,
    execute_translate_request_v2,
    execute_utseg_request_v2,
)
from batchalign.worker._types import _state
from batchalign.worker._types_v2 import (
    ExecuteErrorV2,
    ExecuteRequestV2,
    ExecuteResponseV2,
    InferenceTaskV2,
    ProtocolErrorCodeV2,
)


@dataclass(frozen=True, slots=True)
class WorkerExecutionHostV2:
    """Live V2 execution hosts built from already loaded worker state.

    The host groups the task-specific executors that the Python worker can run
    directly. Tests inject typed fake hosts here instead of replacing module
    globals.
    """

    asr: AsrExecutionHostV2 = field(default_factory=AsrExecutionHostV2)
    forced_alignment: ForcedAlignmentExecutionHostV2 = field(
        default_factory=ForcedAlignmentExecutionHostV2
    )
    speaker: SpeakerExecutionHostV2 = field(default_factory=SpeakerExecutionHostV2)
    speaker_embedding: SpeakerEmbeddingExecutionHostV2 = field(
        default_factory=SpeakerEmbeddingExecutionHostV2
    )
    opensmile: OpenSmileExecutionHostV2 = field(
        default_factory=OpenSmileExecutionHostV2
    )
    avqi: AvqiExecutionHostV2 = field(default_factory=AvqiExecutionHostV2)
    text: TextExecutionHostV2 = field(default_factory=TextExecutionHostV2)


def build_default_execution_host_v2() -> WorkerExecutionHostV2:
    """Build the live V2 execution hosts from loaded worker models."""

    return WorkerExecutionHostV2(
        asr=build_default_asr_execution_host_v2(
            asr_engine=_state.asr_engine,
            whisper_model=_state.whisper_asr_model,
        ),
        forced_alignment=build_default_fa_execution_host_v2(
            whisper_model=_state.whisper_fa_model,
            wave2vec_model=_state.wave2vec_fa_model,
            canto_host=_default_cantonese_fa_host(),
            qwen_host=_state.qwen_fa_host,
        ),
        speaker=build_default_speaker_execution_host_v2(
            _state.bootstrap.device_policy if _state.bootstrap is not None else None
        ),
        speaker_embedding=build_default_speaker_embedding_execution_host_v2(),
        opensmile=build_default_opensmile_execution_host_v2(),
        avqi=build_default_avqi_execution_host_v2(),
        text=build_default_text_execution_host_v2(),
    )


def execute_request_v2(
    request: ExecuteRequestV2,
    *,
    host: WorkerExecutionHostV2 | None = None,
) -> ExecuteResponseV2:
    """Execute one typed V2 worker request against the loaded runtime."""

    invalid_request_response = _validate_request_boundary(request)
    if invalid_request_response is not None:
        return invalid_request_response

    # Test-echo mode: return a successful echo response without model dispatch.
    # This enables integration tests for the concurrent dispatch path
    # (SharedGpuWorker) without loading real ML models.
    if _state.test_echo:
        import time

        from batchalign.worker._types_v2 import ExecuteSuccessV2

        if _state.test_delay_ms > 0:
            time.sleep(_state.test_delay_ms / 1000.0)

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
            result=_echo_placeholder_result(request.payload),
            elapsed_s=0.001,
        )

    execution_host = host or build_default_execution_host_v2()

    match request.task:
        case InferenceTaskV2.MORPHOSYNTAX:
            return execute_morphosyntax_request_v2(request, execution_host.text)
        case InferenceTaskV2.UTSEG:
            return execute_utseg_request_v2(request, execution_host.text)
        case InferenceTaskV2.TRANSLATE:
            return execute_translate_request_v2(request, execution_host.text)
        case InferenceTaskV2.COREF:
            return execute_coref_request_v2(request, execution_host.text)
        case InferenceTaskV2.ASR:
            return execute_asr_request_v2(request, execution_host.asr)
        case InferenceTaskV2.FORCED_ALIGNMENT:
            return execute_forced_alignment_request_v2(
                request,
                execution_host.forced_alignment,
            )
        case InferenceTaskV2.SPEAKER:
            return execute_speaker_request_v2(request, execution_host.speaker)
        case InferenceTaskV2.SPEAKER_EMBEDDING:
            return execute_speaker_embedding_request_v2(
                request, execution_host.speaker_embedding
            )
        case InferenceTaskV2.OPENSMILE:
            return execute_opensmile_request_v2(request, execution_host.opensmile)
        case InferenceTaskV2.AVQI:
            return execute_avqi_request_v2(request, execution_host.avqi)
        case _:
            return _unsupported_task_response(request)


def _echo_placeholder_result(payload: TaskRequestV2) -> TaskResultV2:
    """Derive an empty result from the typed request payload itself."""

    from batchalign.worker._types_v2 import (
        AsrRequestV2,
        AvqiRequestV2,
        AvqiResultPayloadV2,
        CorefRequestV2,
        CorefResultPayloadV2,
        ForcedAlignmentRequestV2,
        IndexedWordTimingResultPayloadV2,
        LocalPyannoteSpeakerEvidenceV2,
        MonologueAsrResultPayloadV2,
        MorphosyntaxRequestV2,
        MorphosyntaxResultPayloadV2,
        NemoSpeakerEvidenceV2,
        OpenSmileRequestV2,
        OpenSmileResultPayloadV2,
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
    )

    match payload:
        case MorphosyntaxRequestV2():
            return MorphosyntaxResultPayloadV2(items=[])
        case UtsegRequestV2():
            return UtsegResultPayloadV2(items=[])
        case TranslateRequestV2():
            return TranslationResultPayloadV2(items=[])
        case CorefRequestV2():
            return CorefResultPayloadV2(items=[])
        case AsrRequestV2():
            return MonologueAsrResultPayloadV2(lang="eng", monologues=[])
        case ForcedAlignmentRequestV2():
            return IndexedWordTimingResultPayloadV2(indexed_timings=[])
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
                case SpeakerBackendV2.NEMO:
                    return SpeakerResultPayloadV2(
                        evidence=NemoSpeakerEvidenceV2(segments=[])
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


def _unsupported_task_response(request: ExecuteRequestV2) -> ExecuteResponseV2:
    """Return a typed error for V2 tasks that are not live yet."""

    return ExecuteResponseV2(
        request_id=request.request_id,
        outcome=ExecuteErrorV2(
            code=ProtocolErrorCodeV2.MODEL_UNAVAILABLE,
            message=(
                f"worker protocol V2 task {request.task.value} is not wired into "
                "the live worker yet"
            ),
        ),
        result=None,
        elapsed_s=0.0,
    )


def _validate_request_boundary(request: ExecuteRequestV2) -> ExecuteResponseV2 | None:
    """Reject mismatched top-level task/payload combinations before dispatch."""

    payload_kind = getattr(request.payload, "kind", None)
    if payload_kind is None:
        return _invalid_payload_response(
            request,
            "execute payload did not include a task kind discriminator",
        )
    if request.task.value != payload_kind:
        return _invalid_payload_response(
            request,
            f"execute payload kind {payload_kind} does not match task {request.task.value}",
        )
    return None


def _invalid_payload_response(
    request: ExecuteRequestV2,
    message: str,
) -> ExecuteResponseV2:
    """Return one typed invalid-payload protocol response."""

    return ExecuteResponseV2(
        request_id=request.request_id,
        outcome=ExecuteErrorV2(
            code=ProtocolErrorCodeV2.INVALID_PAYLOAD,
            message=message,
        ),
        result=None,
        elapsed_s=0.0,
    )


__all__ = [
    "WorkerExecutionHostV2",
    "build_default_execution_host_v2",
    "execute_request_v2",
]


def _default_cantonese_fa_host() -> CantoneseFaHost | None:
    """The loaded Cantonese FA host, or None when that engine was not loaded.

    Imported lazily because the Cantonese module imports torch transitively via
    `batchalign.inference.fa`, and this module currently pulls no torch at all.
    (It does NOT pull pycantonese, which is imported inside `load_cantonese_fa`;
    an earlier version of this comment said otherwise.)
    """
    from batchalign.inference.languages.cantonese._cantonese_fa import (
        default_cantonese_fa_host,
    )

    return default_cantonese_fa_host()
