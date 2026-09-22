"""Tests for the live worker-protocol V2 ASR executor."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from batchalign.inference.asr import (
    AsrElement,
    AsrMonologue,
    AttributedSpeaker,
    MonologueAsrResponse,
)
from batchalign.tests._asr_model_pins import (
    loaded_identity,
    request_models,
    worker_loaded,
)
from batchalign.worker._asr_v2 import AsrExecutionHostV2, execute_asr_request_v2
from batchalign.worker._types_v2 import (
    AsrBackendV2,
    AsrElementKindV2,
    AsrRequestV2,
    ExecuteErrorV2,
    ExecuteRequestV2,
    ExecuteSuccessV2,
    InferenceTaskV2,
    IntegratedDiarizationV2,
    MonologueAsrResultV2,
    PreparedAudioEncodingV2,
    PreparedAudioInputV2,
    PreparedAudioRefV2,
    ProtocolErrorCodeV2,
    ProviderMediaInputV2,
    WhisperChunkResultPayloadV2,
    WhisperChunkResultV2,
    WhisperChunkSpanV2,
)


def _write_pcm_f32le(path: Path, samples: np.ndarray) -> None:
    """Write little-endian float32 PCM test data to disk."""

    path.write_bytes(samples.astype("<f4").tobytes())


def _make_request(
    tmp_path: Path,
    *,
    backend: AsrBackendV2 = AsrBackendV2.LOCAL_WHISPER,
) -> ExecuteRequestV2:
    """Create one live V2 ASR execute request with prepared audio."""

    audio_path = tmp_path / "audio.pcm"
    _write_pcm_f32le(audio_path, np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32))

    return ExecuteRequestV2(
        request_id="req-asr-v2-1",
        task=InferenceTaskV2.ASR,
        payload=AsrRequestV2(
            lang="eng",
            backend=backend,
            input=PreparedAudioInputV2(audio_ref_id="audio-ref-1"),
            models=request_models(backend),
        ),
        attachments=[
            PreparedAudioRefV2(
                id="audio-ref-1",
                path=str(audio_path),
                encoding=PreparedAudioEncodingV2.PCM_F32LE,
                channels=1,
                sample_rate_hz=16000,
                frame_count=4,
                byte_offset=0,
                byte_len=16,
            )
        ],
    )


def _make_provider_request(
    *,
    backend: AsrBackendV2 = AsrBackendV2.HK_TENCENT,
) -> ExecuteRequestV2:
    """Create one live V2 ASR request that uses provider-media input."""

    return ExecuteRequestV2(
        request_id="req-asr-v2-provider-1",
        task=InferenceTaskV2.ASR,
        payload=AsrRequestV2(
            lang="yue",
            backend=backend,
            input=ProviderMediaInputV2(
                media_path="/tmp/provider.wav",
                diarization=IntegratedDiarizationV2(speakers=2),
            ),
            models=request_models(backend),
        ),
        attachments=[],
    )


def test_executes_local_whisper_asr_v2_request(tmp_path: Path) -> None:
    """The live V2 executor should return typed Whisper chunk output."""

    captured: dict[str, object] = {}

    def runner(audio: np.ndarray, lang: str) -> WhisperChunkResultPayloadV2:
        captured["shape"] = audio.shape
        captured["lang"] = lang
        return WhisperChunkResultPayloadV2(
            lang=lang,
            text="hello world",
            chunks=[
                WhisperChunkSpanV2(text="hello", start_s=0.0, end_s=0.5),
                WhisperChunkSpanV2(text="world", start_s=0.5, end_s=1.0),
            ],
            model=loaded_identity(AsrBackendV2.LOCAL_WHISPER),
        )

    response = execute_asr_request_v2(
        _make_request(tmp_path),
        AsrExecutionHostV2(local_whisper_runner=runner),
    )

    assert isinstance(response.outcome, ExecuteSuccessV2)
    assert isinstance(response.result, WhisperChunkResultV2)
    assert response.result.text == "hello world"
    assert response.result.chunks[1].end_s == 1.0
    assert captured == {"shape": (4,), "lang": "eng"}


def test_returns_missing_attachment_for_invalid_asr_request(tmp_path: Path) -> None:
    """Missing prepared audio should become a typed protocol error."""

    request = _make_request(tmp_path)
    request.attachments = []

    response = execute_asr_request_v2(
        request,
        AsrExecutionHostV2(local_whisper_runner=lambda *_args: None),  # type: ignore[arg-type]
    )

    assert isinstance(response.outcome, ExecuteErrorV2)
    assert response.outcome.code is ProtocolErrorCodeV2.MISSING_ATTACHMENT
    assert response.result is None


def test_invalid_numeric_attachment_becomes_invalid_payload_even_if_validation_is_bypassed(
    tmp_path: Path,
) -> None:
    """Rust should reject bad prepared-audio numerics even when Python is bypassed."""

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

    response = execute_asr_request_v2(
        bad_request,
        AsrExecutionHostV2(
            local_whisper_runner=lambda *_args: WhisperChunkResultPayloadV2(
                lang="eng",
                text="unused",
                chunks=[],
                model=loaded_identity(AsrBackendV2.LOCAL_WHISPER),
            )
        ),
    )

    assert isinstance(response.outcome, ExecuteErrorV2)
    assert response.outcome.code is ProtocolErrorCodeV2.INVALID_PAYLOAD
    assert "positive sample_rate_hz" in response.outcome.message
    assert response.result is None


def test_returns_model_unavailable_for_unwired_asr_backend(tmp_path: Path) -> None:
    """Cloud/provider ASR backends should fail explicitly on the live V2 path."""

    response = execute_asr_request_v2(
        _make_provider_request(backend=AsrBackendV2.REVAI),
        AsrExecutionHostV2(),
    )

    assert isinstance(response.outcome, ExecuteErrorV2)
    assert response.outcome.code is ProtocolErrorCodeV2.MODEL_UNAVAILABLE
    assert "Rust control plane" in response.outcome.message


@worker_loaded(AsrBackendV2.HK_TENCENT)
def test_executes_provider_media_asr_v2_request() -> None:
    """Provider-media ASR requests should return typed monologue output.

    Records a loaded identity first, as the real engine loader does: the bridge
    reads this worker's load record before calling a provider, because the
    provider's own reply describes speech and never the model behind it.
    """

    captured: dict[str, object] = {}

    def runner(item) -> MonologueAsrResponse:
        captured["audio_path"] = item.audio_path
        captured["lang"] = item.lang
        captured["diarization"] = item.diarization
        return MonologueAsrResponse(
            lang=item.lang,
            monologues=[
                AsrMonologue(
                    speaker=AttributedSpeaker(label="1"),
                    elements=[
                        AsrElement(
                            value="nei5",
                            ts=0.1,
                            end_ts=0.4,
                            type="text",
                            confidence=0.9,
                        ),
                        AsrElement(value="。", type="punctuation"),
                    ],
                )
            ],
        )

    response = execute_asr_request_v2(
        _make_provider_request(),
        AsrExecutionHostV2(hk_tencent_runner=runner),
    )

    assert isinstance(response.outcome, ExecuteSuccessV2)
    assert isinstance(response.result, MonologueAsrResultV2)
    assert response.result.monologues[0].speaker.kind == "attributed"
    assert response.result.monologues[0].speaker.label == "1"
    assert response.result.monologues[0].elements[0].kind is AsrElementKindV2.TEXT
    assert (
        response.result.monologues[0].elements[1].kind is AsrElementKindV2.PUNCTUATION
    )
    # What reaches the provider adapter is the request's own typed answer, not a
    # bare count whose zero would have to mean "do not separate".
    assert captured == {
        "audio_path": "/tmp/provider.wav",
        "lang": "yue",
        "diarization": IntegratedDiarizationV2(speakers=2),
    }


def test_invalid_local_whisper_host_output_becomes_runtime_failure(
    tmp_path: Path,
) -> None:
    """Malformed local Whisper host output should be classified as runtime failure."""

    def runner(_audio: np.ndarray, lang: str) -> dict[str, object]:
        return {
            "lang": lang,
            "text": "hello world",
            "chunks": [
                {"text": "hello", "start_s": 0.6, "end_s": 0.2},
            ],
        }

    response = execute_asr_request_v2(
        _make_request(tmp_path),
        AsrExecutionHostV2(local_whisper_runner=runner),  # type: ignore[arg-type]
    )

    assert isinstance(response.outcome, ExecuteErrorV2)
    assert response.outcome.code is ProtocolErrorCodeV2.RUNTIME_FAILURE
    assert "invalid ASR host output" in response.outcome.message
    assert response.result is None


@worker_loaded(AsrBackendV2.HK_TENCENT)
def test_invalid_provider_asr_host_output_becomes_runtime_failure() -> None:
    """Malformed provider ASR host output should be classified as runtime failure."""

    def runner(_item) -> MonologueAsrResponse:
        return MonologueAsrResponse(
            lang="yue",
            monologues=[
                AsrMonologue(
                    speaker=AttributedSpeaker(label="1"),
                    elements=[
                        AsrElement(
                            value="nei5",
                            ts=0.8,
                            end_ts=0.4,
                            type="text",
                        )
                    ],
                )
            ],
        )

    response = execute_asr_request_v2(
        _make_provider_request(),
        AsrExecutionHostV2(hk_tencent_runner=runner),
    )

    assert isinstance(response.outcome, ExecuteErrorV2)
    assert response.outcome.code is ProtocolErrorCodeV2.RUNTIME_FAILURE
    assert "invalid ASR host output" in response.outcome.message
    assert response.result is None


def test_whisper_chunk_inverted_timestamps_travel_raw(tmp_path: Path) -> None:
    """Whisper occasionally returns chunks with end_s < start_s on long audio
    (job 696870c7-02b, maria16.wav). The executor passes them through as
    emitted: the Rust consumer settles every producer's spans in one place, so
    a per-chunk refusal or swap here would be a second, guessing copy.
    """

    def runner(audio: np.ndarray, lang: str) -> WhisperChunkResultPayloadV2:
        return WhisperChunkResultPayloadV2(
            lang=lang,
            text=" Thank you.",
            chunks=[
                WhisperChunkSpanV2(text=" Thank you.", start_s=2020.0, end_s=2017.0),
            ],
            model=loaded_identity(AsrBackendV2.LOCAL_WHISPER),
        )

    response = execute_asr_request_v2(
        _make_request(tmp_path),
        AsrExecutionHostV2(local_whisper_runner=runner),
    )

    assert isinstance(response.outcome, ExecuteSuccessV2)
    assert isinstance(response.result, WhisperChunkResultV2)
    assert response.result.chunks[0].start_s == 2020.0
    assert response.result.chunks[0].end_s == 2017.0
