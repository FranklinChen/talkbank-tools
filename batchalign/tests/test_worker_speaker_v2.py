"""Focused tests for the live worker-protocol V2 speaker executor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from batchalign.inference.speaker import (
    LocalPyannoteSpeakerEvidence,
    PyannoteAISpeakerEvidence,
    SpeakerResponse,
    SpeakerSegment,
)
from batchalign.tests._loaded_hosts import diarization
from batchalign.worker._speaker_v2 import execute_speaker_request_v2
from batchalign.worker._types_v2 import (
    ExecuteErrorV2,
    ExecuteRequestV2,
    ExecuteSuccessV2,
    InferenceTaskV2,
    LocalPyannoteSpeakerEvidenceV2,
    PreparedAudioEncodingV2,
    PreparedAudioRefV2,
    ProtocolErrorCodeV2,
    SpeakerBackendV2,
    SpeakerPreparedAudioInputV2,
    SpeakerRequestV2,
    SpeakerResultV2,
)


def _write_pcm_f32le(path: Path, samples: np.ndarray) -> None:
    """Write little-endian float32 PCM speaker fixture data to disk."""

    path.write_bytes(samples.astype("<f4").tobytes())


def _pyannote_response(segments: list[SpeakerSegment] | None = None) -> SpeakerResponse:
    """Build evidence which can only satisfy a local pyannote request."""

    return SpeakerResponse(
        evidence=LocalPyannoteSpeakerEvidence(segments=segments or [])
    )


def _pyannote_ai_response() -> SpeakerResponse:
    """Build evidence which can only satisfy a pyannoteAI request."""

    return SpeakerResponse(
        evidence=PyannoteAISpeakerEvidence(
            job_id="job-paid-0",
            output={"exclusiveDiarization": []},
        )
    )


def _make_request(
    tmp_path: Path,
    *,
    backend: SpeakerBackendV2 = SpeakerBackendV2.PYANNOTE,
    expected_speakers: int | None = 2,
    channels: int = 1,
    byte_len: int | None = None,
) -> ExecuteRequestV2:
    """Create one live V2 speaker request with prepared audio."""

    frame_count = 4
    sample_count = frame_count * channels
    audio_path = tmp_path / "speaker-audio.pcm"
    _write_pcm_f32le(
        audio_path,
        np.asarray(
            [float(i) / 10.0 for i in range(1, sample_count + 1)], dtype=np.float32
        ),
    )
    if byte_len is None:
        byte_len = sample_count * 4

    return ExecuteRequestV2(
        request_id="req-speaker-v2-1",
        task=InferenceTaskV2.SPEAKER,
        payload=SpeakerRequestV2(
            backend=backend,
            input=SpeakerPreparedAudioInputV2(audio_ref_id="audio-ref-speaker-1"),
            expected_speakers=expected_speakers,
        ),
        attachments=[
            PreparedAudioRefV2(
                id="audio-ref-speaker-1",
                path=str(audio_path),
                encoding=PreparedAudioEncodingV2.PCM_F32LE,
                channels=channels,
                sample_rate_hz=16000,
                frame_count=frame_count,
                byte_offset=0,
                byte_len=byte_len,
            )
        ],
    )


def _assert_error_response(
    response,
    code: ProtocolErrorCodeV2,
    message_fragment: str | None = None,
) -> None:
    """Assert one typed speaker error response."""

    assert isinstance(response.outcome, ExecuteErrorV2)
    assert response.outcome.code is code
    if message_fragment is not None:
        assert message_fragment in response.outcome.message
    assert response.result is None


def test_execute_speaker_request_v2_returns_typed_segments(tmp_path: Path) -> None:
    """Speaker V2 execution should return the typed segment payload."""

    response = execute_speaker_request_v2(
        _make_request(tmp_path),
        diarization(
            lambda audio, sample_rate_hz, num_speakers: _pyannote_response(
                [
                    SpeakerSegment(
                        start_ms=10,
                        end_ms=25,
                        speaker=f"SPEAKER_{num_speakers}_{sample_rate_hz}_{audio.shape[0]}",
                    )
                ]
            )
        ),
    )

    assert isinstance(response.outcome, ExecuteSuccessV2)
    assert isinstance(response.result, SpeakerResultV2)
    assert isinstance(response.result.evidence, LocalPyannoteSpeakerEvidenceV2)
    assert response.result.evidence.segments[0].speaker == "SPEAKER_2_16000_4"
    assert response.result.evidence.segments[0].start_ms == 10


def test_execute_speaker_request_v2_preserves_an_unspecified_speaker_count(
    tmp_path: Path,
) -> None:
    """An omitted expected_speakers reaches the runner as None, not as 2.

    POLICY CHANGED 2026-08-21, and this test asserted the old behaviour, so it
    is rewritten rather than deleted. It used to be named
    ``..._defaults_expected_speakers_to_two`` and pinned ``num_speakers == 2``
    for a request that had specified nothing.

    That default was the defect. The CLI records an absent ``--num-speakers``
    as auto-detect (it says so in ``cli/args/mod.rs``), the protocol omits the
    field entirely, and pyannote estimates the count when given None. Rust
    collapsed all of that with ``.unwrap_or(2)`` at the FFI, so a request
    saying "I do not know" was indistinguishable from one asking for exactly
    two, and estimation was unreachable through the V2 path. Since ``diarize``
    hardcodes the pyannote backend, every run without an explicit count was
    quietly forced to two speakers.

    The value a caller SPECIFIED and a value we INVENTED must not arrive
    looking the same; keeping None intact is how that holds across the
    boundary.
    """

    captured: dict[str, object] = {}

    def runner(audio, sample_rate_hz, num_speakers):
        captured["shape"] = audio.shape
        captured["sample_rate_hz"] = sample_rate_hz
        captured["num_speakers"] = num_speakers
        return _pyannote_response()

    response = execute_speaker_request_v2(
        _make_request(tmp_path, expected_speakers=None),
        diarization(runner),
    )

    assert isinstance(response.outcome, ExecuteSuccessV2)
    assert captured == {"shape": (4,), "sample_rate_hz": 16000, "num_speakers": None}


def test_execute_speaker_request_v2_routes_pyannote_ai_backend(tmp_path: Path) -> None:
    """A pyannoteAI request runs on the hosted diarization host its load built."""

    called: list[str] = []

    def cloud_runner(audio, sample_rate_hz, num_speakers):
        called.append(f"cloud:{audio.shape[0]}:{sample_rate_hz}:{num_speakers}")
        return SpeakerResponse(
            evidence=PyannoteAISpeakerEvidence(
                job_id="job-paid-1",
                output={
                    "exclusiveDiarization": [
                        {"start": 0.0, "end": 0.01, "speaker": "SPEAKER_00"}
                    ]
                },
            )
        )

    response = execute_speaker_request_v2(
        _make_request(tmp_path, backend=SpeakerBackendV2.PYANNOTE_AI),
        diarization(cloud_runner, SpeakerBackendV2.PYANNOTE_AI),
    )

    assert isinstance(response.outcome, ExecuteSuccessV2)
    assert called == ["cloud:4:16000:2"]
    assert response.result is not None
    assert response.result.evidence.kind == "pyannote_ai"
    assert response.result.evidence.job_id == "job-paid-1"


def test_execute_speaker_request_v2_rejects_backend_evidence_mismatch(
    tmp_path: Path,
) -> None:
    """A local pyannote request cannot be satisfied by pyannoteAI provenance."""

    response = execute_speaker_request_v2(
        _make_request(tmp_path, backend=SpeakerBackendV2.PYANNOTE),
        diarization(lambda *_args: _pyannote_ai_response()),
    )

    _assert_error_response(
        response,
        ProtocolErrorCodeV2.RUNTIME_FAILURE,
        "evidence does not match requested backend",
    )


def test_execute_speaker_request_v2_rejects_wrong_task() -> None:
    """A task/payload mismatch fails with a typed protocol error.

    Policy change (2026-08-21): the agreement check moved from each executor
    into the shared Rust control plane, so the diagnostic is now the control
    plane's single wording ("declared task ... does not match payload task
    ...") rather than the speaker executor's old "expected speaker task"
    message. The wire code is unchanged (INVALID_PAYLOAD).
    """

    response = execute_speaker_request_v2(
        ExecuteRequestV2(
            request_id="req-speaker-v2-wrong-task",
            task=InferenceTaskV2.ASR,
            payload=SpeakerRequestV2(
                backend=SpeakerBackendV2.PYANNOTE,
                input=SpeakerPreparedAudioInputV2(audio_ref_id="audio-ref-speaker-2"),
            ),
            attachments=[],
        ),
        diarization(lambda *_args: _pyannote_response()),
    )

    _assert_error_response(
        response, ProtocolErrorCodeV2.INVALID_PAYLOAD, "does not match payload task"
    )


def test_speaker_provider_media_requests_fail_schema_validation() -> None:
    """Legacy speaker provider-media requests should be rejected at the schema boundary."""

    with pytest.raises(ValidationError):
        ExecuteRequestV2.model_validate(
            {
                "request_id": "req-speaker-v2-legacy",
                "task": "speaker",
                "payload": {
                    "kind": "speaker",
                    "backend": "pyannote",
                    "input": {
                        "kind": "provider_media",
                        "media_path": "/tmp/meeting.wav",
                    },
                },
                "attachments": [],
            }
        )


def test_the_hosted_speaker_load_builds_the_pyannote_ai_host_and_loads_nothing() -> (
    None
):
    """pyannoteAI is hosted: its load returns the host that calls the API,
    named by its backend, and builds no local model."""

    from batchalign.inference.speaker import infer_pyannote_ai_prepared_audio
    from batchalign.worker._model_loading.speaker import hosted_speaker_task

    host = hosted_speaker_task()
    assert host.backend is SpeakerBackendV2.PYANNOTE_AI
    assert host.runner is infer_pyannote_ai_prepared_audio


def test_missing_speaker_attachment_returns_typed_error(tmp_path: Path) -> None:
    """Missing prepared audio should become a typed protocol error."""

    request = _make_request(tmp_path)
    request.attachments = []

    response = execute_speaker_request_v2(
        request,
        diarization(lambda *_args: _pyannote_response()),
    )

    _assert_error_response(
        response,
        ProtocolErrorCodeV2.MISSING_ATTACHMENT,
        "missing worker protocol V2 attachment",
    )


def test_multichannel_speaker_audio_is_rejected(tmp_path: Path) -> None:
    """The live speaker path should reject multi-channel prepared audio explicitly."""

    response = execute_speaker_request_v2(
        _make_request(tmp_path, channels=2, byte_len=32),
        diarization(lambda *_args: _pyannote_response()),
    )

    _assert_error_response(
        response, ProtocolErrorCodeV2.INVALID_PAYLOAD, "mono prepared audio"
    )


def test_malformed_speaker_audio_returns_attachment_unreadable(tmp_path: Path) -> None:
    """Truncated prepared audio should map to ATTACHMENT_UNREADABLE."""

    response = execute_speaker_request_v2(
        _make_request(tmp_path, byte_len=15),
        diarization(lambda *_args: _pyannote_response()),
    )

    _assert_error_response(
        response,
        ProtocolErrorCodeV2.ATTACHMENT_UNREADABLE,
        "has 15 bytes, expected 16",
    )


def test_a_request_for_another_backend_than_the_loaded_one_is_refused(
    tmp_path: Path,
) -> None:
    """The executor reads the backend from the host it is handed: a pyannoteAI
    request handed the local pyannote host is refused by name, and the local
    runner never runs (a paid call is not silently replaced by a local one)."""

    def must_not_run(*_args):
        raise AssertionError("the loaded runner serves a different backend")

    response = execute_speaker_request_v2(
        _make_request(tmp_path, backend=SpeakerBackendV2.PYANNOTE_AI),
        diarization(must_not_run, SpeakerBackendV2.PYANNOTE),
    )

    _assert_error_response(
        response,
        ProtocolErrorCodeV2.MODEL_UNAVAILABLE,
        "this worker loaded the pyannote speaker backend, and the request asks for "
        "pyannote_ai",
    )


def test_speaker_runtime_failures_are_typed(tmp_path: Path) -> None:
    """Unexpected speaker host failures should surface as runtime failures."""

    def boom(*_args):
        raise RuntimeError("speaker host exploded")

    response = execute_speaker_request_v2(
        _make_request(tmp_path),
        diarization(boom),
    )

    _assert_error_response(
        response, ProtocolErrorCodeV2.RUNTIME_FAILURE, "speaker host exploded"
    )


def test_invalid_numeric_attachment_becomes_invalid_payload_even_if_validation_is_bypassed(
    tmp_path: Path,
) -> None:
    """Rust should still reject bad prepared-audio numerics if a caller bypasses Pydantic."""

    request = _make_request(tmp_path)
    raw_attachment = request.attachments[0].model_dump()
    raw_attachment["sample_rate_hz"] = 0
    bad_attachment = PreparedAudioRefV2.model_construct(**raw_attachment)
    bad_request = ExecuteRequestV2.model_construct(
        request_id=request.request_id,
        task=request.task,
        payload=request.payload,
        attachments=[bad_attachment],
    )

    response = execute_speaker_request_v2(
        bad_request,
        diarization(lambda *_args: _pyannote_response()),
    )

    _assert_error_response(
        response,
        ProtocolErrorCodeV2.INVALID_PAYLOAD,
        "positive sample_rate_hz",
    )


def test_invalid_speaker_host_output_becomes_runtime_failure(tmp_path: Path) -> None:
    """Malformed speaker host output should be classified as runtime failure."""

    response = execute_speaker_request_v2(
        _make_request(tmp_path),
        diarization(
            lambda *_args: _pyannote_response(
                [
                    SpeakerSegment(
                        start_ms=1200,
                        end_ms=200,
                        speaker="SPEAKER_BAD",
                    )
                ]
            )
        ),
    )

    _assert_error_response(
        response,
        ProtocolErrorCodeV2.RUNTIME_FAILURE,
        "invalid speaker host output",
    )
