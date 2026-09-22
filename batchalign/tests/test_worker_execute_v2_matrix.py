"""Matrix tests for live worker-protocol V2 backend routing and shape guards.

These tests complement the task-by-task router checks by covering the backend
matrix and a few intentionally invalid boundary combinations that should fail
before any real model host is required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from batchalign.inference._domain_types import TranslationBackend
from batchalign.inference.asr import (
    AsrElement,
    AsrMonologue,
    AttributedSpeaker,
    MonologueAsrResponse,
    UndiarizedSpeaker,
)
from batchalign.inference.speaker import (
    LocalPyannoteSpeakerEvidence,
    NemoSpeakerEvidence,
    PyannoteAISpeakerEvidence,
    SpeakerResponse,
    SpeakerSegment,
)
from batchalign.inference.types import Wave2VecWordAlignment
from batchalign.tests._asr_model_pins import request_models, worker_loaded
from batchalign.worker._asr_v2 import AsrExecutionHostV2
from batchalign.worker._execute_v2 import WorkerExecutionHostV2, execute_request_v2
from batchalign.worker._fa_v2 import ForcedAlignmentExecutionHostV2
from batchalign.worker._speaker_v2 import SpeakerExecutionHostV2
from batchalign.worker._text_v2 import TextExecutionHostV2
from batchalign.worker._types import BatchInferResponse, InferResponse, WorkerJSONValue
from batchalign.worker._types_v2 import (
    ArtifactRefV2,
    AsrBackendV2,
    AsrInputV2,
    AsrRequestV2,
    CorefRequestV2,
    CorefResultV2,
    ExecuteErrorV2,
    ExecuteRequestV2,
    ExecuteSuccessV2,
    FaBackendV2,
    FaTextModeV2,
    ForcedAlignmentRequestV2,
    IndexedWordTimingResultV2,
    InferenceTaskV2,
    IntegratedDiarizationV2,
    MonologueAsrResultV2,
    MorphosyntaxRequestV2,
    MorphosyntaxResultV2,
    NotRequestedDiarizationV2,
    PreparedAudioEncodingV2,
    PreparedAudioInputV2,
    PreparedAudioRefV2,
    PreparedTextEncodingV2,
    PreparedTextRefV2,
    ProtocolErrorCodeV2,
    ProviderMediaInputV2,
    SpeakerBackendV2,
    SpeakerPreparedAudioInputV2,
    SpeakerRequestV2,
    SpeakerResultV2,
    SubmittedJobInputV2,
    TranslateRequestV2,
    TranslationResultV2,
    UtsegRequestV2,
    UtsegResultV2,
    WhisperTokenTimingResultV2,
)

TextTaskName = Literal["morphosyntax", "utseg", "translate", "coref"]
AsrInputKind = Literal["prepared_audio", "provider_media", "submitted_job"]


def _write_pcm_f32le(path: Path, samples: np.ndarray) -> None:
    """Write little-endian float32 PCM test data to disk."""

    path.write_bytes(samples.astype("<f4").tobytes())


def _write_json_payload(path: Path, value: WorkerJSONValue) -> None:
    """Write one prepared JSON payload for V2 matrix tests."""

    path.write_text(json.dumps(value), encoding="utf-8")


def _make_prepared_audio_attachment(tmp_path: Path, stem: str) -> PreparedAudioRefV2:
    """Create one small prepared-audio attachment for V2 matrix tests."""

    audio_path = tmp_path / f"{stem}.pcm"
    _write_pcm_f32le(
        audio_path,
        np.asarray([0.2, 0.1, -0.1, 0.0], dtype=np.float32),
    )
    return PreparedAudioRefV2(
        id=f"{stem}-audio-ref",
        path=str(audio_path),
        encoding=PreparedAudioEncodingV2.PCM_F32LE,
        channels=1,
        sample_rate_hz=16000,
        frame_count=4,
        byte_offset=0,
        byte_len=16,
    )


def _make_asr_request(
    tmp_path: Path,
    backend: AsrBackendV2,
    input_kind: AsrInputKind,
) -> ExecuteRequestV2:
    """Build one ASR execute request for the requested backend/input pair."""

    attachments: list[ArtifactRefV2] = []
    input_payload: AsrInputV2
    if input_kind == "prepared_audio":
        audio_attachment = _make_prepared_audio_attachment(
            tmp_path, f"asr-{backend.value}"
        )
        attachments = [audio_attachment]
        input_payload = PreparedAudioInputV2(audio_ref_id=audio_attachment.id)
    elif input_kind == "provider_media":
        # Ask each provider for what it can actually do. Tencent separates
        # speakers; Aliyun, FunASR and Qwen separate none, and asking them to
        # while their own output attributes nobody is the contradiction the
        # bridge refuses. One request shape for all four hid that.
        input_payload = ProviderMediaInputV2(
            media_path=f"/tmp/{backend.value}.wav",
            diarization=(
                IntegratedDiarizationV2(speakers=2)
                if _separates_speakers(backend)
                else NotRequestedDiarizationV2()
            ),
        )
    else:
        input_payload = SubmittedJobInputV2(
            provider_job_id=f"submitted-{backend.value}-1"
        )

    return ExecuteRequestV2(
        request_id=f"req-asr-{backend.value}-{input_kind}",
        task=InferenceTaskV2.ASR,
        payload=AsrRequestV2(
            lang="yue",
            backend=backend,
            input=input_payload,
            models=request_models(backend),
        ),
        attachments=attachments,
    )


def _separates_speakers(backend: AsrBackendV2) -> bool:
    """Whether this provider attributes speakers of its own.

    Tencent is the only HK provider that does. Aliyun streams one track,
    FunASR's speaker model is never built (its `spk_model` argument reaches
    `generate`, which does not read it), and Qwen3-ASR emits no diarization;
    for those three, BA3's dedicated diarization stage attributes speakers
    afterwards.
    """

    return backend is AsrBackendV2.HK_TENCENT


def _provider_asr_host(backend: AsrBackendV2) -> AsrExecutionHostV2:
    """Build one provider-ASR host that marks which backend was selected."""

    def _response(label: str, media_path: str) -> MonologueAsrResponse:
        return MonologueAsrResponse(
            lang="yue",
            monologues=[
                AsrMonologue(
                    # A provider that separates speakers names one; a provider
                    # that does not says so. Reporting "undiarized" from an
                    # engine the request asked to separate is refused at the
                    # bridge, which is what this double used to do for all four.
                    speaker=(
                        AttributedSpeaker(label="1")
                        if _separates_speakers(backend)
                        else UndiarizedSpeaker()
                    ),
                    elements=[AsrElement(value=f"{label}:{media_path}", type="text")],
                )
            ],
        )

    if backend is AsrBackendV2.HK_TENCENT:
        return AsrExecutionHostV2(
            hk_tencent_runner=lambda item: _response("tencent", item.audio_path)
        )
    if backend is AsrBackendV2.HK_ALIYUN:
        return AsrExecutionHostV2(
            hk_aliyun_runner=lambda item: _response("aliyun", item.audio_path)
        )
    if backend is AsrBackendV2.HK_FUNAUDIO:
        return AsrExecutionHostV2(
            hk_funaudio_runner=lambda item: _response("funaudio", item.audio_path)
        )
    if backend is AsrBackendV2.HK_QWEN:
        return AsrExecutionHostV2(
            hk_qwen_runner=lambda item: _response("qwen", item.audio_path)
        )
    raise AssertionError(f"unexpected provider ASR backend {backend!s}")


def _make_fa_request(
    tmp_path: Path,
    backend: FaBackendV2,
    text_mode: FaTextModeV2,
) -> ExecuteRequestV2:
    """Build one FA execute request for the requested backend/text mode."""

    words = ["天", "氣"] if backend is FaBackendV2.WAV2VEC_CANTO else ["hello", "world"]
    payload_path = tmp_path / f"fa-{backend.value}.json"
    _write_json_payload(
        payload_path,
        {
            "words": words,
            "word_ids": ["u0:w0", "u0:w1"],
            "word_utterance_indices": [0, 0],
            "word_utterance_word_indices": [0, 1],
        },
    )
    payload_attachment = PreparedTextRefV2(
        id=f"fa-{backend.value}-payload-ref",
        path=str(payload_path),
        encoding=PreparedTextEncodingV2.UTF8_JSON,
        byte_offset=0,
        byte_len=payload_path.stat().st_size,
    )
    audio_attachment = _make_prepared_audio_attachment(tmp_path, f"fa-{backend.value}")
    return ExecuteRequestV2(
        request_id=f"req-fa-{backend.value}",
        task=InferenceTaskV2.FORCED_ALIGNMENT,
        payload=ForcedAlignmentRequestV2(
            backend=backend,
            payload_ref_id=payload_attachment.id,
            audio_ref_id=audio_attachment.id,
            text_mode=text_mode,
        ),
        attachments=[payload_attachment, audio_attachment],
    )


def _fa_host(backend: FaBackendV2) -> ForcedAlignmentExecutionHostV2:
    """Build one FA host that exposes which backend/result shape was selected."""

    if backend is FaBackendV2.WHISPER:
        return ForcedAlignmentExecutionHostV2(
            whisper_runner=lambda audio, text: [
                (text.split()[0], 0.1 if audio.shape == (4,) else 0.0),
                (text.split()[-1], 0.25),
            ]
        )
    if backend is FaBackendV2.WAVE2VEC:
        return ForcedAlignmentExecutionHostV2(
            wave2vec_runner=lambda audio, words: [
                Wave2VecWordAlignment(words[0], (100, 180), 0.8),
                Wave2VecWordAlignment(
                    words[1], (240, 320 if audio.shape == (4,) else 0), 0.9
                ),
            ]
        )
    if backend is FaBackendV2.WAV2VEC_CANTO:
        return ForcedAlignmentExecutionHostV2(
            canto_runner=lambda audio, payload, request: [
                Wave2VecWordAlignment(
                    payload.words[0],
                    (50, 120 if request.text_mode is FaTextModeV2.CHAR_JOINED else 0),
                    None,
                ),
                Wave2VecWordAlignment(
                    payload.words[1], (130, 220 if audio.shape == (4,) else 0), None
                ),
            ]
        )
    raise AssertionError(f"unexpected FA backend {backend!s}")


def _make_speaker_request(
    tmp_path: Path, backend: SpeakerBackendV2
) -> ExecuteRequestV2:
    """Build one speaker execute request for the requested backend."""

    audio_attachment = _make_prepared_audio_attachment(
        tmp_path, f"speaker-{backend.value}"
    )
    return ExecuteRequestV2(
        request_id=f"req-speaker-{backend.value}",
        task=InferenceTaskV2.SPEAKER,
        payload=SpeakerRequestV2(
            backend=backend,
            input=SpeakerPreparedAudioInputV2(audio_ref_id=audio_attachment.id),
            expected_speakers=3,
        ),
        attachments=[audio_attachment],
    )


def _speaker_host(backend: SpeakerBackendV2) -> SpeakerExecutionHostV2:
    """Build one speaker host that marks which backend was selected."""

    segment = SpeakerSegment(start_ms=0, end_ms=1000, speaker=backend.value)

    if backend is SpeakerBackendV2.PYANNOTE_AI:
        return SpeakerExecutionHostV2(
            pyannote_ai_prepared_audio_runner=lambda audio, sample_rate_hz, num_speakers: (
                SpeakerResponse(
                    evidence=PyannoteAISpeakerEvidence(
                        job_id="job-matrix",
                        output={
                            "exclusiveDiarization": [
                                {
                                    "start": 0.0,
                                    "end": 1.0,
                                    "speaker": f"pyannote_ai-{sample_rate_hz}-{num_speakers}-{audio.shape[0]}",
                                }
                            ]
                        },
                    )
                )
            )
        )
    if backend is SpeakerBackendV2.PYANNOTE:
        return SpeakerExecutionHostV2(
            pyannote_prepared_audio_runner=lambda audio, sample_rate_hz, num_speakers: (
                SpeakerResponse(
                    evidence=LocalPyannoteSpeakerEvidence(
                        segments=[
                            segment.model_copy(
                                update={
                                    "speaker": f"pyannote-{sample_rate_hz}-{num_speakers}-{audio.shape[0]}"
                                }
                            )
                        ]
                    )
                )
            )
        )
    if backend is SpeakerBackendV2.NEMO:
        return SpeakerExecutionHostV2(
            nemo_prepared_audio_runner=lambda audio, sample_rate_hz, num_speakers: (
                SpeakerResponse(
                    evidence=NemoSpeakerEvidence(
                        segments=[
                            segment.model_copy(
                                update={
                                    "speaker": f"nemo-{sample_rate_hz}-{num_speakers}-{audio.shape[0]}"
                                }
                            )
                        ]
                    )
                )
            )
        )
    raise AssertionError(f"unexpected speaker backend {backend!s}")


def _make_text_request(tmp_path: Path, task_name: TextTaskName) -> ExecuteRequestV2:
    """Build one text-task execute request for result-shape guard tests."""

    payload_path = tmp_path / f"{task_name}.json"
    if task_name == "morphosyntax":
        attachment_id = "text-ref-morphosyntax-matrix"
        _write_json_payload(
            payload_path,
            {
                "items": [
                    {
                        "words": ["I", "saw", "it"],
                        "terminator": ".",
                        "special_forms": [[None, None], [None, None], [None, None]],
                        "lang": "eng",
                    }
                ],
                "mwt": {},
            },
        )
        payload = MorphosyntaxRequestV2(
            lang="eng",
            payload_ref_id=attachment_id,
            item_count=1,
        )
        task = InferenceTaskV2.MORPHOSYNTAX
    elif task_name == "utseg":
        attachment_id = "text-ref-utseg-matrix"
        _write_json_payload(
            payload_path,
            {"items": [{"words": ["hello", "world"], "text": "hello world"}]},
        )
        payload = UtsegRequestV2(
            lang="eng",
            payload_ref_id=attachment_id,
            item_count=1,
        )
        task = InferenceTaskV2.UTSEG
    elif task_name == "translate":
        attachment_id = "text-ref-translate-matrix"
        _write_json_payload(payload_path, {"items": [{"text": "hello there"}]})
        payload = TranslateRequestV2(
            source_lang="eng",
            target_lang="spa",
            engine=TranslationBackend.GOOGLE,
            payload_ref_id=attachment_id,
            item_count=1,
        )
        task = InferenceTaskV2.TRANSLATE
    else:
        attachment_id = "text-ref-coref-matrix"
        _write_json_payload(payload_path, {"items": [{"sentences": [["she"]]}]})
        payload = CorefRequestV2(
            lang="eng",
            payload_ref_id=attachment_id,
            item_count=1,
        )
        task = InferenceTaskV2.COREF

    attachment = PreparedTextRefV2(
        id=attachment_id,
        path=str(payload_path),
        encoding=PreparedTextEncodingV2.UTF8_JSON,
        byte_offset=0,
        byte_len=payload_path.stat().st_size,
    )
    return ExecuteRequestV2(
        request_id=f"req-{task_name}-matrix",
        task=task,
        payload=payload,
        attachments=[attachment],
    )


def _text_host_with_results(
    task_name: TextTaskName, results: list[InferResponse]
) -> TextExecutionHostV2:
    """Build one text host that returns the supplied batch-infer result list."""

    response = BatchInferResponse(results=results)
    if task_name == "morphosyntax":
        return TextExecutionHostV2(morphosyntax_runner=lambda request: response)
    if task_name == "utseg":
        return TextExecutionHostV2(utseg_runner=lambda request: response)
    if task_name == "translate":
        return TextExecutionHostV2(translate_runner=lambda request: response)
    return TextExecutionHostV2(coref_runner=lambda request: response)


def _assert_error_response(
    response,
    code: ProtocolErrorCodeV2,
    message_fragment: str,
) -> None:
    """Assert that one execute response is a typed protocol/runtime error."""

    assert isinstance(response.outcome, ExecuteErrorV2)
    assert response.outcome.code is code
    assert response.result is None
    assert message_fragment in response.outcome.message


@pytest.mark.parametrize(
    "backend",
    [
        AsrBackendV2.HK_TENCENT,
        AsrBackendV2.HK_ALIYUN,
        AsrBackendV2.HK_FUNAUDIO,
        AsrBackendV2.HK_QWEN,
    ],
    ids=lambda backend: backend.value,
)
def test_routes_provider_asr_backend_matrix(
    backend: AsrBackendV2, tmp_path: Path
) -> None:
    """Provider ASR backends should route to the matching host runner."""

    request = _make_asr_request(tmp_path, backend, "provider_media")
    # The backend varies per parameter case, so this one records its load
    # identity inline rather than through the decorator the fixed-backend
    # tests use.
    with worker_loaded(backend):
        response = execute_request_v2(
            request=request,
            host=WorkerExecutionHostV2(asr=_provider_asr_host(backend)),
        )

    assert isinstance(response.outcome, ExecuteSuccessV2)
    assert isinstance(response.result, MonologueAsrResultV2)
    assert (
        response.result.monologues[0]
        .elements[0]
        .value.startswith(backend.value.removeprefix("hk_"))
    )


@pytest.mark.parametrize(
    ("backend", "text_mode"),
    [
        pytest.param(
            FaBackendV2.WHISPER,
            FaTextModeV2.SPACE_JOINED,
            id="whisper-space_joined",
        ),
        pytest.param(
            FaBackendV2.WAVE2VEC,
            FaTextModeV2.SPACE_JOINED,
            id="wave2vec-space_joined",
        ),
        pytest.param(
            FaBackendV2.WAV2VEC_CANTO,
            FaTextModeV2.CHAR_JOINED,
            id="wav2vec_canto-char_joined",
        ),
        # What `--pauses` selects: the request carries the shaping, and the
        # host receives text already shaped.
        pytest.param(
            FaBackendV2.WHISPER,
            FaTextModeV2.CHAR_SPACED,
            id="whisper-char_spaced",
        ),
    ],
)
def test_routes_forced_alignment_backend_matrix(
    tmp_path: Path,
    backend: FaBackendV2,
    text_mode: FaTextModeV2,
) -> None:
    """FA backends should route to the matching host and result shape."""

    request = _make_fa_request(tmp_path, backend, text_mode)
    response = execute_request_v2(
        request=request,
        host=WorkerExecutionHostV2(forced_alignment=_fa_host(backend)),
    )

    assert isinstance(response.outcome, ExecuteSuccessV2)
    if backend is FaBackendV2.WHISPER:
        assert isinstance(response.result, WhisperTokenTimingResultV2)
        # The host echoes the first whitespace-separated token of the text it
        # was given, so this asserts which shaping actually reached it.
        expected_first = "h" if text_mode is FaTextModeV2.CHAR_SPACED else "hello"
        assert response.result.tokens[0].text == expected_first
        assert response.result.tokens[1].time_s == 0.25
    else:
        assert isinstance(response.result, IndexedWordTimingResultV2)
        assert response.result.indexed_timings[0] is not None
        assert response.result.indexed_timings[0].start_ms >= 50
        assert response.result.indexed_timings[1] is not None


@pytest.mark.parametrize(
    "backend",
    [
        SpeakerBackendV2.PYANNOTE_AI,
        SpeakerBackendV2.PYANNOTE,
        SpeakerBackendV2.NEMO,
    ],
    ids=lambda backend: backend.value,
)
def test_routes_speaker_backend_matrix(
    backend: SpeakerBackendV2, tmp_path: Path
) -> None:
    """Speaker backends should route to the matching host runner."""

    request = _make_speaker_request(tmp_path, backend)
    response = execute_request_v2(
        request=request,
        host=WorkerExecutionHostV2(speaker=_speaker_host(backend)),
    )

    assert isinstance(response.outcome, ExecuteSuccessV2)
    assert isinstance(response.result, SpeakerResultV2)
    if response.result.evidence.kind == "pyannote_ai":
        first = response.result.evidence.output["exclusiveDiarization"][0]
        assert isinstance(first, dict)
        assert str(first["speaker"]).startswith(backend.value)
    else:
        assert response.result.evidence.segments[0].speaker.startswith(backend.value)


@pytest.mark.parametrize(
    ("backend", "input_kind", "expected_code", "message_fragment"),
    [
        pytest.param(
            AsrBackendV2.LOCAL_WHISPER,
            "provider_media",
            ProtocolErrorCodeV2.INVALID_PAYLOAD,
            "prepared_audio",
            id="local_whisper-provider_media",
        ),
        pytest.param(
            AsrBackendV2.HK_TENCENT,
            "prepared_audio",
            ProtocolErrorCodeV2.INVALID_PAYLOAD,
            "provider_media",
            id="hk_tencent-prepared_audio",
        ),
        pytest.param(
            AsrBackendV2.HK_ALIYUN,
            "prepared_audio",
            ProtocolErrorCodeV2.INVALID_PAYLOAD,
            "provider_media",
            id="hk_aliyun-prepared_audio",
        ),
        pytest.param(
            AsrBackendV2.HK_FUNAUDIO,
            "prepared_audio",
            ProtocolErrorCodeV2.INVALID_PAYLOAD,
            "provider_media",
            id="hk_funaudio-prepared_audio",
        ),
        pytest.param(
            AsrBackendV2.HK_TENCENT,
            "submitted_job",
            ProtocolErrorCodeV2.INVALID_PAYLOAD,
            "provider_media",
            id="hk_tencent-submitted_job",
        ),
        pytest.param(
            AsrBackendV2.REVAI,
            "provider_media",
            ProtocolErrorCodeV2.MODEL_UNAVAILABLE,
            "Rust control plane",
            id="revai-provider_media",
        ),
    ],
)
def test_rejects_invalid_asr_backend_input_pairs(
    tmp_path: Path,
    backend: AsrBackendV2,
    input_kind: AsrInputKind,
    expected_code: ProtocolErrorCodeV2,
    message_fragment: str,
) -> None:
    """Invalid ASR backend/input pairs should fail with typed protocol errors."""

    request = _make_asr_request(tmp_path, backend, input_kind)
    response = execute_request_v2(
        request=request,
        host=WorkerExecutionHostV2(asr=AsrExecutionHostV2()),
    )
    _assert_error_response(response, expected_code, message_fragment)


@pytest.mark.parametrize(
    "task_name",
    ["morphosyntax", "utseg", "translate", "coref"],
)
def test_text_execute_v2_rejects_result_count_mismatch(
    tmp_path: Path,
    task_name: TextTaskName,
) -> None:
    """Text-task V2 execution should reject host result-count drift."""

    request = _make_text_request(tmp_path, task_name)
    response = execute_request_v2(
        request=request,
        host=WorkerExecutionHostV2(
            text=_text_host_with_results(task_name, []),
        ),
    )

    _assert_error_response(response, ProtocolErrorCodeV2.RUNTIME_FAILURE, "expected 1")


@pytest.mark.parametrize(
    ("task_name", "good_result", "expected_kind"),
    [
        pytest.param(
            "morphosyntax",
            {
                "kind": "analyzed",
                "raw_sentences": [],
                "model": {
                    "stanza_version": "1.99.0",
                    "lang": "eng",
                    "pipeline": "standard",
                },
                "repairs": [],
            },
            MorphosyntaxResultV2,
            id="morphosyntax",
        ),
        pytest.param(
            "morphosyntax",
            {"kind": "no_words"},
            MorphosyntaxResultV2,
            id="morphosyntax-no-words",
        ),
        pytest.param(
            "utseg",
            {"trees": ["(ROOT hello)"]},
            UtsegResultV2,
            id="utseg",
        ),
        pytest.param(
            "utseg",
            {"assignments": [0, 1]},
            UtsegResultV2,
            id="utseg-assignments",
        ),
        pytest.param(
            "translate",
            {
                "kind": "translated",
                "raw_translation": "hola",
                "engine": "googletrans-v1",
            },
            TranslationResultV2,
            id="translate",
        ),
        pytest.param(
            "translate",
            {"kind": "blank_input"},
            TranslationResultV2,
            id="translate-blank-input",
        ),
        pytest.param(
            "coref",
            {"kind": "resolved", "annotations": [], "engine": "stanza-1.99.0"},
            CorefResultV2,
            id="coref",
        ),
        pytest.param(
            "coref",
            {"kind": "no_sentences"},
            CorefResultV2,
            id="coref-no-sentences",
        ),
    ],
)
def test_text_execute_v2_accepts_valid_result_shapes(
    tmp_path: Path,
    task_name: TextTaskName,
    good_result: WorkerJSONValue,
    expected_kind: type,
) -> None:
    """Text-task V2 execution should accept valid provider-shaped host output."""

    request = _make_text_request(tmp_path, task_name)
    response = execute_request_v2(
        request=request,
        host=WorkerExecutionHostV2(
            text=_text_host_with_results(
                task_name,
                [InferResponse(result=good_result, elapsed_s=0.0)],
            ),
        ),
    )

    assert isinstance(response.outcome, ExecuteSuccessV2)
    assert isinstance(response.result, expected_kind)


@pytest.mark.parametrize(
    ("task_name", "bad_result", "message_fragment"),
    [
        pytest.param(
            "utseg",
            {"trees": [1]},
            "list[str]",
            id="utseg-trees",
        ),
        pytest.param(
            "utseg",
            {"assignments": ["bad"]},
            "list[usize]",
            id="utseg-assignments",
        ),
    ],
)
def test_text_execute_v2_rejects_invalid_result_shapes(
    tmp_path: Path,
    task_name: TextTaskName,
    bad_result: WorkerJSONValue,
    message_fragment: str,
) -> None:
    """Text-task V2 execution should reject malformed per-item result shapes."""

    request = _make_text_request(tmp_path, task_name)
    response = execute_request_v2(
        request=request,
        host=WorkerExecutionHostV2(
            text=_text_host_with_results(
                task_name,
                [InferResponse(result=bad_result, elapsed_s=0.0)],
            ),
        ),
    )

    _assert_error_response(
        response, ProtocolErrorCodeV2.RUNTIME_FAILURE, message_fragment
    )


_MODEL = {"stanza_version": "1.99.0", "lang": "eng", "pipeline": "standard"}


@pytest.mark.parametrize(
    ("task_name", "bad_item"),
    [
        pytest.param("morphosyntax", {"raw_sentences": []}, id="morphosyntax-untagged"),
        pytest.param(
            "morphosyntax",
            {"kind": "analyzed", "raw_sentences": {}, "model": _MODEL, "repairs": []},
            id="morphosyntax-raw_sentences",
        ),
        pytest.param(
            "morphosyntax",
            {"kind": "analyzed", "raw_sentences": [], "repairs": []},
            id="morphosyntax-missing-model",
        ),
        # A host that reports no repairs must say so. Defaulting the field
        # would make "repaired nothing" and "does not report repairs" the same
        # item, which is the shape that kept relation rewrites in the log only.
        pytest.param(
            "morphosyntax",
            {"kind": "analyzed", "raw_sentences": [], "model": _MODEL},
            id="morphosyntax-missing-repairs",
        ),
        pytest.param(
            "morphosyntax",
            {
                "kind": "analyzed",
                "raw_sentences": [],
                "model": _MODEL,
                "repairs": [{"kind": "transposed", "word": "ne"}],
            },
            id="morphosyntax-unknown-repair-kind",
        ),
        pytest.param(
            "morphosyntax",
            {
                "kind": "analyzed",
                "raw_sentences": [],
                "model": {**_MODEL, "stanza_version": ""},
                "repairs": [],
            },
            id="morphosyntax-blank-model-version",
        ),
        pytest.param(
            "morphosyntax",
            {
                "kind": "analyzed",
                "raw_sentences": [],
                "model": {**_MODEL, "pipeline": "retokenize"},
                "repairs": [],
            },
            id="morphosyntax-unknown-pipeline",
        ),
        pytest.param(
            "translate",
            {"kind": "translated", "raw_translation": ["hola"], "engine": "x"},
            id="translate-raw_translation",
        ),
        pytest.param(
            "translate",
            {"kind": "translated", "raw_translation": "hola"},
            id="translate-missing-engine",
        ),
        pytest.param(
            "translate",
            {"kind": "translated", "raw_translation": "hola", "engine": "google|v1"},
            id="translate-separator-in-engine",
        ),
        pytest.param(
            "coref",
            {"kind": "resolved", "annotations": {}, "engine": "stanza-1.99.0"},
            id="coref-annotations",
        ),
        pytest.param(
            "coref",
            {"kind": "resolved", "annotations": [], "engine": " "},
            id="coref-blank-engine",
        ),
    ],
)
def test_text_execute_v2_fails_only_the_item_that_does_not_parse(
    tmp_path: Path,
    task_name: TextTaskName,
    bad_item: WorkerJSONValue,
) -> None:
    """A host item outside its tagged V2 shape becomes that item's failure.

    The bridge parses each item through the canonical tagged type, so an
    analysis without its model or an engine name a stamp cannot hold is not a
    value that crosses; the batch still succeeds and only this item fails.
    """

    request = _make_text_request(tmp_path, task_name)
    response = execute_request_v2(
        request=request,
        host=WorkerExecutionHostV2(
            text=_text_host_with_results(
                task_name,
                [InferResponse(result=bad_item, elapsed_s=0.0)],
            ),
        ),
    )

    assert isinstance(response.outcome, ExecuteSuccessV2)
    assert response.result is not None
    item = response.result.items[0]
    assert item.kind == "failed"
    assert f"invalid {task_name} host item" in item.error
