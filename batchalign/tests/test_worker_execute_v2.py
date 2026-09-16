# affects: batchalign/worker/_execute_v2.py
# affects: batchalign/worker/_types_v2.py
"""Tests for the live worker-protocol V2 execution router."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from batchalign.inference.asr import (
    AsrElement,
    AsrMonologue,
    AttributedSpeaker,
    MonologueAsrResponse,
)
from batchalign.inference.avqi import AvqiResponse
from batchalign.inference.opensmile import OpenSmileResponse
from batchalign.inference.speaker import (
    LocalPyannoteSpeakerEvidence,
    SpeakerResponse,
    SpeakerSegment,
)
from batchalign.tests._asr_model_pins import (
    loaded_identity,
    request_models,
    worker_loaded,
)
from batchalign.worker._asr_v2 import AsrExecutionHostV2
from batchalign.worker._avqi_v2 import AvqiExecutionHostV2
from batchalign.worker._execute_v2 import WorkerExecutionHostV2, execute_request_v2
from batchalign.worker._fa_v2 import ForcedAlignmentExecutionHostV2
from batchalign.worker._opensmile_v2 import OpenSmileExecutionHostV2
from batchalign.worker._speaker_v2 import SpeakerExecutionHostV2
from batchalign.worker._text_v2 import TextExecutionHostV2
from batchalign.worker._types import (
    BatchInferRequest,
    BatchInferResponse,
    InferResponse,
    _state,
)
from batchalign.worker._types_v2 import (
    AsrBackendV2,
    AsrRequestV2,
    AttributedSpeakerV2,
    AvqiRequestV2,
    AvqiResultV2,
    CorefRequestV2,
    CorefResolvedItemV2,
    CorefResultV2,
    ExecuteErrorV2,
    ExecuteRequestV2,
    ExecuteResponseV2,
    ExecuteSuccessV2,
    FaBackendV2,
    FaTextModeV2,
    ForcedAlignmentRequestV2,
    InferenceTaskV2,
    IntegratedDiarizationV2,
    MonologueAsrResultV2,
    MorphosyntaxAnalyzedItemV2,
    MorphosyntaxRequestV2,
    MorphosyntaxResultV2,
    OpenSmileRequestV2,
    OpenSmileResultV2,
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
    TranslateRequestV2,
    TranslationResultV2,
    UtsegRequestV2,
    UtsegResultV2,
    WhisperChunkResultPayloadV2,
    WhisperChunkResultV2,
    WhisperChunkSpanV2,
    WhisperTokenTimingResultV2,
)


def _write_pcm_f32le(path: Path, samples: np.ndarray) -> None:
    """Write little-endian float32 PCM test data to disk."""

    path.write_bytes(samples.astype("<f4").tobytes())


def _write_fa_payload(path: Path) -> None:
    """Write a minimal prepared FA payload for live V2 router tests."""

    path.write_text(
        json.dumps(
            {
                "words": ["hello", "world"],
                "word_ids": ["u0:w0", "u0:w1"],
                "word_utterance_indices": [0, 0],
                "word_utterance_word_indices": [0, 1],
            }
        ),
        encoding="utf-8",
    )


def _write_json_payload(path: Path, value: object) -> None:
    """Write one prepared JSON payload for live V2 router tests."""

    path.write_text(json.dumps(value), encoding="utf-8")


def _assert_runtime_failure_response(response, message_fragment: str) -> None:
    """Assert one typed runtime-failure response from the V2 router."""

    assert isinstance(response.outcome, ExecuteErrorV2)
    assert response.outcome.code is ProtocolErrorCodeV2.RUNTIME_FAILURE
    assert message_fragment in response.outcome.message
    assert response.result is None


def test_test_echo_rejects_mismatched_task_payload_boundary() -> None:
    """Test-echo mode must still reject malformed top-level V2 task/payload pairs."""

    previous_test_echo = _state.test_echo
    previous_test_delay_ms = _state.test_delay_ms
    _state.test_echo = True
    _state.test_delay_ms = 0
    try:
        response = execute_request_v2(
            request=ExecuteRequestV2(
                request_id="req-execute-v2-mismatch-1",
                task=InferenceTaskV2.MORPHOSYNTAX,
                payload=AsrRequestV2(
                    lang="eng",
                    backend=AsrBackendV2.HK_TENCENT,
                    input=ProviderMediaInputV2(
                        media_path="/tmp/provider.wav",
                        diarization=IntegratedDiarizationV2(speakers=2),
                    ),
                    models=request_models(AsrBackendV2.HK_TENCENT),
                ),
                attachments=[],
            )
        )
    finally:
        _state.test_echo = previous_test_echo
        _state.test_delay_ms = previous_test_delay_ms

    assert isinstance(response.outcome, ExecuteErrorV2)
    assert response.outcome.code is ProtocolErrorCodeV2.INVALID_PAYLOAD
    assert "does not match task morphosyntax" in response.outcome.message
    assert response.result is None


def test_test_echo_derives_speaker_evidence_from_backend_payload() -> None:
    """Echo placeholders preserve speaker backend provenance structurally."""

    previous_test_echo = _state.test_echo
    previous_test_delay_ms = _state.test_delay_ms
    _state.test_echo = True
    _state.test_delay_ms = 0
    try:
        for backend in SpeakerBackendV2:
            response = execute_request_v2(
                request=ExecuteRequestV2(
                    request_id=f"req-execute-v2-echo-{backend.value}",
                    task=InferenceTaskV2.SPEAKER,
                    payload=SpeakerRequestV2(
                        backend=backend,
                        input=SpeakerPreparedAudioInputV2(
                            audio_ref_id="unused-in-echo"
                        ),
                    ),
                    attachments=[],
                )
            )
            assert isinstance(response.outcome, ExecuteSuccessV2)
            assert isinstance(response.result, SpeakerResultV2)
            assert response.result.evidence.kind == backend.value
    finally:
        _state.test_echo = previous_test_echo
        _state.test_delay_ms = previous_test_delay_ms


def _make_fa_request(
    tmp_path: Path,
) -> tuple[
    PreparedTextRefV2,
    PreparedAudioRefV2,
    ForcedAlignmentRequestV2,
]:
    """Build the attachments and payload wrapper for one FA V2 request."""

    payload_path = tmp_path / "payload.json"
    audio_path = tmp_path / "audio.pcm"
    _write_fa_payload(payload_path)
    _write_pcm_f32le(audio_path, np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32))

    return (
        PreparedTextRefV2(
            id="payload-ref-1",
            path=str(payload_path),
            encoding=PreparedTextEncodingV2.UTF8_JSON,
            byte_offset=0,
            byte_len=payload_path.stat().st_size,
        ),
        PreparedAudioRefV2(
            id="audio-ref-1",
            path=str(audio_path),
            encoding=PreparedAudioEncodingV2.PCM_F32LE,
            channels=1,
            sample_rate_hz=16000,
            frame_count=4,
            byte_offset=0,
            byte_len=16,
        ),
        ForcedAlignmentRequestV2(
            backend=FaBackendV2.WHISPER,
            payload_ref_id="payload-ref-1",
            audio_ref_id="audio-ref-1",
            # `--pauses` is expressed as the text mode now, not as a second
            # boolean the host applies after this one.
            text_mode=FaTextModeV2.CHAR_SPACED,
        ),
    )


def test_routes_forced_alignment_execute_v2_request(tmp_path: Path) -> None:
    """The live V2 router should hand FA requests to the staged FA executor."""

    payload_attachment, audio_attachment, payload = _make_fa_request(tmp_path)

    response = execute_request_v2(
        request=ExecuteRequestV2(
            request_id="req-execute-v2-fa-1",
            task=InferenceTaskV2.FORCED_ALIGNMENT,
            payload=payload,
            attachments=[payload_attachment, audio_attachment],
        ),
        host=WorkerExecutionHostV2(
            asr=AsrExecutionHostV2(),
            forced_alignment=ForcedAlignmentExecutionHostV2(
                # The shaping the request asked for must have been applied
                # before the host is called: with CHAR_SPACED, "hello world"
                # arrives one character per token. Asserting on the text is
                # what asserting on the old `pauses` flag was standing in for.
                whisper_runner=lambda audio, text: [
                    ("hello", 0.1 if audio.shape == (4,) else 0.0),
                    ("world", 0.3 if text == "h e l l o   w o r l d" else 0.0),
                ]
            ),
            speaker=SpeakerExecutionHostV2(),
        ),
    )

    assert isinstance(response.outcome, ExecuteSuccessV2)
    assert isinstance(response.result, WhisperTokenTimingResultV2)
    assert response.result.tokens[0].text == "hello"
    assert response.result.tokens[1].time_s == 0.3


def test_routes_asr_execute_v2_request(tmp_path: Path) -> None:
    """The live V2 router should hand ASR requests to the ASR executor."""

    _payload_attachment, audio_attachment, _payload = _make_fa_request(tmp_path)

    response = execute_request_v2(
        request=ExecuteRequestV2(
            request_id="req-execute-v2-asr-1",
            task=InferenceTaskV2.ASR,
            payload=AsrRequestV2(
                lang="eng",
                backend=AsrBackendV2.LOCAL_WHISPER,
                input=PreparedAudioInputV2(audio_ref_id="audio-ref-1"),
                models=request_models(AsrBackendV2.LOCAL_WHISPER),
            ),
            attachments=[audio_attachment],
        ),
        host=WorkerExecutionHostV2(
            asr=AsrExecutionHostV2(
                local_whisper_runner=lambda audio, lang: WhisperChunkResultPayloadV2(
                    lang=lang,
                    text="hello",
                    chunks=[
                        WhisperChunkSpanV2(
                            text="hello",
                            start_s=0.0 if audio.shape == (4,) else 1.0,
                            end_s=0.5,
                        )
                    ],
                    model=loaded_identity(AsrBackendV2.LOCAL_WHISPER),
                )
            ),
            forced_alignment=ForcedAlignmentExecutionHostV2(),
            speaker=SpeakerExecutionHostV2(),
        ),
    )

    assert isinstance(response.outcome, ExecuteSuccessV2)
    assert isinstance(response.result, WhisperChunkResultV2)
    assert response.result.text == "hello"


@worker_loaded(AsrBackendV2.HK_TENCENT)
def test_routes_provider_media_asr_execute_v2_request() -> None:
    """The live V2 router should hand provider-media ASR requests to ASR hosts."""

    response = execute_request_v2(
        request=ExecuteRequestV2(
            request_id="req-execute-v2-asr-provider-1",
            task=InferenceTaskV2.ASR,
            payload=AsrRequestV2(
                lang="yue",
                backend=AsrBackendV2.HK_TENCENT,
                input=ProviderMediaInputV2(
                    media_path="/tmp/provider.wav",
                    diarization=IntegratedDiarizationV2(speakers=2),
                ),
                models=request_models(AsrBackendV2.HK_TENCENT),
            ),
            attachments=[],
        ),
        host=WorkerExecutionHostV2(
            asr=AsrExecutionHostV2(
                hk_tencent_runner=lambda item: MonologueAsrResponse(
                    lang=item.lang,
                    monologues=[
                        AsrMonologue(
                            speaker=AttributedSpeaker(label="1"),
                            elements=[AsrElement(value=item.audio_path, type="text")],
                        )
                    ],
                )
            ),
            forced_alignment=ForcedAlignmentExecutionHostV2(),
            speaker=SpeakerExecutionHostV2(),
        ),
    )

    assert isinstance(response.outcome, ExecuteSuccessV2)
    assert isinstance(response.result, MonologueAsrResultV2)
    assert response.result.monologues[0].speaker == AttributedSpeakerV2(label="1")
    assert response.result.monologues[0].elements[0].value == "/tmp/provider.wav"


def test_routes_speaker_execute_v2_request(tmp_path: Path) -> None:
    """The live V2 router should hand speaker requests to the speaker host."""

    audio_path = tmp_path / "speaker-audio.pcm"
    _write_pcm_f32le(audio_path, np.asarray([0.2, 0.1, 0.0, -0.1], dtype=np.float32))
    response = execute_request_v2(
        request=ExecuteRequestV2(
            request_id="req-execute-v2-speaker-1",
            task=InferenceTaskV2.SPEAKER,
            payload=SpeakerRequestV2(
                backend=SpeakerBackendV2.PYANNOTE,
                input=SpeakerPreparedAudioInputV2(audio_ref_id="audio-ref-speaker-1"),
                expected_speakers=3,
            ),
            attachments=[
                PreparedAudioRefV2(
                    id="audio-ref-speaker-1",
                    path=str(audio_path),
                    encoding=PreparedAudioEncodingV2.PCM_F32LE,
                    channels=1,
                    sample_rate_hz=16000,
                    frame_count=4,
                    byte_offset=0,
                    byte_len=16,
                )
            ],
        ),
        host=WorkerExecutionHostV2(
            asr=AsrExecutionHostV2(),
            forced_alignment=ForcedAlignmentExecutionHostV2(),
            speaker=SpeakerExecutionHostV2(
                pyannote_prepared_audio_runner=lambda audio, sample_rate_hz, num_speakers: (
                    SpeakerResponse(
                        evidence=LocalPyannoteSpeakerEvidence(
                            segments=[
                                SpeakerSegment(
                                    start_ms=0,
                                    end_ms=1200,
                                    speaker=f"SPEAKER_{num_speakers}_{sample_rate_hz}_{audio.shape[0]}",
                                )
                            ]
                        )
                    )
                )
            ),
        ),
    )

    assert isinstance(response.outcome, ExecuteSuccessV2)
    assert isinstance(response.result, SpeakerResultV2)
    assert response.result.evidence.kind == "pyannote"
    assert response.result.evidence.segments[0].speaker == "SPEAKER_3_16000_4"
    assert response.result.evidence.segments[0].end_ms == 1200


def test_routes_opensmile_execute_v2_request(tmp_path: Path) -> None:
    """The live V2 router should hand openSMILE requests to the staged executor."""

    audio_path = tmp_path / "opensmile-audio.pcm"
    _write_pcm_f32le(audio_path, np.asarray([0.0, 0.5, -0.5, 0.25], dtype=np.float32))

    response = execute_request_v2(
        request=ExecuteRequestV2(
            request_id="req-execute-v2-opensmile-1",
            task=InferenceTaskV2.OPENSMILE,
            payload=OpenSmileRequestV2(
                audio_ref_id="audio-ref-opensmile-1",
                feature_set="eGeMAPSv02",
                feature_level="functionals",
            ),
            attachments=[
                PreparedAudioRefV2(
                    id="audio-ref-opensmile-1",
                    path=str(audio_path),
                    encoding=PreparedAudioEncodingV2.PCM_F32LE,
                    channels=1,
                    sample_rate_hz=16000,
                    frame_count=4,
                    byte_offset=0,
                    byte_len=16,
                )
            ],
        ),
        host=WorkerExecutionHostV2(
            opensmile=OpenSmileExecutionHostV2(
                prepared_audio_runner=lambda audio, sample_rate_hz, feature_set, feature_level, audio_label: (
                    OpenSmileResponse(
                        feature_set=feature_set,
                        feature_level=feature_level,
                        num_features=2 if sample_rate_hz == 16000 else 0,
                        duration_segments=1 if audio.shape == (4,) else 0,
                        audio_file=audio_label,
                        rows=[{"f0_mean": 100.0, "jitter_local": 0.1}],
                        success=True,
                    )
                )
            ),
        ),
    )

    assert isinstance(response.outcome, ExecuteSuccessV2)
    assert isinstance(response.result, OpenSmileResultV2)
    assert response.result.rows[0]["f0_mean"] == 100.0
    assert response.result.feature_set == "eGeMAPSv02"


def test_invalid_opensmile_host_output_becomes_runtime_failure(tmp_path: Path) -> None:
    """Malformed openSMILE host output should be classified as runtime failure."""

    audio_path = tmp_path / "opensmile-invalid-audio.pcm"
    _write_pcm_f32le(audio_path, np.asarray([0.0, 0.5, -0.5, 0.25], dtype=np.float32))

    response = execute_request_v2(
        request=ExecuteRequestV2(
            request_id="req-execute-v2-opensmile-invalid-1",
            task=InferenceTaskV2.OPENSMILE,
            payload=OpenSmileRequestV2(
                audio_ref_id="audio-ref-opensmile-invalid-1",
                feature_set="eGeMAPSv02",
                feature_level="functionals",
            ),
            attachments=[
                PreparedAudioRefV2(
                    id="audio-ref-opensmile-invalid-1",
                    path=str(audio_path),
                    encoding=PreparedAudioEncodingV2.PCM_F32LE,
                    channels=1,
                    sample_rate_hz=16000,
                    frame_count=4,
                    byte_offset=0,
                    byte_len=16,
                )
            ],
        ),
        host=WorkerExecutionHostV2(
            opensmile=OpenSmileExecutionHostV2(
                prepared_audio_runner=lambda *_args: OpenSmileResponse(
                    feature_set="eGeMAPSv02",
                    feature_level="functionals",
                    num_features=1,
                    duration_segments=1,
                    audio_file="invalid.wav",
                    rows=[{"f0_mean": float("nan")}],
                    success=True,
                )
            ),
        ),
    )

    _assert_runtime_failure_response(response, "invalid openSMILE host output")


def test_routes_avqi_execute_v2_request(tmp_path: Path) -> None:
    """The live V2 router should hand AVQI requests to the staged executor."""

    cs_path = tmp_path / "sample.cs.pcm"
    sv_path = tmp_path / "sample.sv.pcm"
    _write_pcm_f32le(cs_path, np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
    _write_pcm_f32le(sv_path, np.asarray([0.5, 0.6, 0.7, 0.8], dtype=np.float32))

    response = execute_request_v2(
        request=ExecuteRequestV2(
            request_id="req-execute-v2-avqi-1",
            task=InferenceTaskV2.AVQI,
            payload=AvqiRequestV2(
                cs_audio_ref_id="audio-ref-avqi-cs-1",
                sv_audio_ref_id="audio-ref-avqi-sv-1",
            ),
            attachments=[
                PreparedAudioRefV2(
                    id="audio-ref-avqi-cs-1",
                    path=str(cs_path),
                    encoding=PreparedAudioEncodingV2.PCM_F32LE,
                    channels=1,
                    sample_rate_hz=16000,
                    frame_count=4,
                    byte_offset=0,
                    byte_len=16,
                ),
                PreparedAudioRefV2(
                    id="audio-ref-avqi-sv-1",
                    path=str(sv_path),
                    encoding=PreparedAudioEncodingV2.PCM_F32LE,
                    channels=1,
                    sample_rate_hz=16000,
                    frame_count=4,
                    byte_offset=0,
                    byte_len=16,
                ),
            ],
        ),
        host=WorkerExecutionHostV2(
            avqi=AvqiExecutionHostV2(
                prepared_audio_runner=lambda cs_audio, cs_rate, sv_audio, sv_rate, cs_label, sv_label: (
                    AvqiResponse(
                        avqi=3.14
                        if cs_audio.shape == (4,) and sv_audio.shape == (4,)
                        else 0.0,
                        cpps=5.0,
                        hnr=10.0,
                        shimmer_local=0.2,
                        shimmer_local_db=0.3,
                        slope=0.4,
                        tilt=0.5,
                        cs_file=cs_label if cs_rate == 16000 else "bad",
                        sv_file=sv_label if sv_rate == 16000 else "bad",
                        success=True,
                    )
                )
            ),
        ),
    )

    assert isinstance(response.outcome, ExecuteSuccessV2)
    assert isinstance(response.result, AvqiResultV2)
    assert response.result.avqi == 3.14
    assert response.result.cs_file.endswith("sample.cs.pcm")


def test_invalid_avqi_host_output_becomes_runtime_failure(tmp_path: Path) -> None:
    """Malformed AVQI host output should be classified as runtime failure."""

    cs_path = tmp_path / "invalid.cs.pcm"
    sv_path = tmp_path / "invalid.sv.pcm"
    _write_pcm_f32le(cs_path, np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
    _write_pcm_f32le(sv_path, np.asarray([0.5, 0.6, 0.7, 0.8], dtype=np.float32))

    response = execute_request_v2(
        request=ExecuteRequestV2(
            request_id="req-execute-v2-avqi-invalid-1",
            task=InferenceTaskV2.AVQI,
            payload=AvqiRequestV2(
                cs_audio_ref_id="audio-ref-avqi-invalid-cs-1",
                sv_audio_ref_id="audio-ref-avqi-invalid-sv-1",
            ),
            attachments=[
                PreparedAudioRefV2(
                    id="audio-ref-avqi-invalid-cs-1",
                    path=str(cs_path),
                    encoding=PreparedAudioEncodingV2.PCM_F32LE,
                    channels=1,
                    sample_rate_hz=16000,
                    frame_count=4,
                    byte_offset=0,
                    byte_len=16,
                ),
                PreparedAudioRefV2(
                    id="audio-ref-avqi-invalid-sv-1",
                    path=str(sv_path),
                    encoding=PreparedAudioEncodingV2.PCM_F32LE,
                    channels=1,
                    sample_rate_hz=16000,
                    frame_count=4,
                    byte_offset=0,
                    byte_len=16,
                ),
            ],
        ),
        host=WorkerExecutionHostV2(
            avqi=AvqiExecutionHostV2(
                prepared_audio_runner=lambda *_args: AvqiResponse(
                    avqi=float("inf"),
                    cpps=5.0,
                    hnr=10.0,
                    shimmer_local=0.2,
                    shimmer_local_db=0.3,
                    slope=0.4,
                    tilt=0.5,
                    cs_file="invalid.cs.pcm",
                    sv_file="invalid.sv.pcm",
                    success=True,
                )
            ),
        ),
    )

    _assert_runtime_failure_response(response, "invalid AVQI host output")


def test_routes_morphosyntax_execute_v2_request(tmp_path: Path) -> None:
    """The live V2 router should hand morphosyntax requests to the text host."""

    payload_path = tmp_path / "morphosyntax-batch.json"
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

    response = execute_request_v2(
        request=ExecuteRequestV2(
            request_id="req-execute-v2-morphosyntax-1",
            task=InferenceTaskV2.MORPHOSYNTAX,
            payload=MorphosyntaxRequestV2(
                lang="eng",
                payload_ref_id="text-ref-morphosyntax-1",
                item_count=1,
            ),
            attachments=[
                PreparedTextRefV2(
                    id="text-ref-morphosyntax-1",
                    path=str(payload_path),
                    encoding=PreparedTextEncodingV2.UTF8_JSON,
                    byte_offset=0,
                    byte_len=payload_path.stat().st_size,
                )
            ],
        ),
        host=WorkerExecutionHostV2(
            asr=AsrExecutionHostV2(),
            forced_alignment=ForcedAlignmentExecutionHostV2(),
            speaker=SpeakerExecutionHostV2(),
            text=TextExecutionHostV2(
                morphosyntax_runner=lambda req: BatchInferResponse(
                    results=[
                        InferResponse(
                            result={
                                "kind": "analyzed",
                                "raw_sentences": [
                                    [
                                        {
                                            "id": 1,
                                            "text": "I",
                                            "lemma": "I",
                                            "upos": "PRON",
                                            "head": 2,
                                            "deprel": "nsubj",
                                        },
                                        {
                                            "id": 2,
                                            "text": "saw",
                                            "lemma": "see",
                                            "upos": "VERB",
                                            "head": 0,
                                            "deprel": "root",
                                        },
                                        {
                                            "id": 3,
                                            "text": "it",
                                            "lemma": "it",
                                            "upos": "PRON",
                                            "head": 2,
                                            "deprel": "obj",
                                        },
                                    ]
                                ],
                                "model": {
                                    "stanza_version": "1.99.0",
                                    "lang": "eng",
                                    "pipeline": "standard",
                                },
                                "repairs": [],
                            },
                            elapsed_s=0.0,
                        )
                    ]
                )
            ),
        ),
    )

    assert isinstance(response.outcome, ExecuteSuccessV2)
    assert isinstance(response.result, MorphosyntaxResultV2)
    assert isinstance(response.result.items[0], MorphosyntaxAnalyzedItemV2)
    assert response.result.items[0].raw_sentences[0][1]["lemma"] == "see"


def test_routes_morphosyntax_unicode_special_forms_request(tmp_path: Path) -> None:
    """Unicode morphosyntax payloads should preserve special forms through the V2 router."""

    payload_path = tmp_path / "morphosyntax-unicode-batch.json"
    _write_json_payload(
        payload_path,
        {
            "items": [
                {
                    "words": ["我", "食", "緊", "飯"],
                    # CHAT terminators are the ASCII set `talkbank_model::Terminator`
                    # defines; a CJK full stop is not one, and this
                    # fixture asserted a shape Rust cannot send.
                    "terminator": ".",
                    "special_forms": [
                        [None, None],
                        ["食緊", "eat.PROG"],
                        [None, None],
                        [None, None],
                    ],
                    "lang": "yue",
                }
            ],
            "mwt": {"食緊": ["食", "緊"]},
        },
    )

    captured: dict[str, object] = {}

    def runner(req: BatchInferRequest) -> BatchInferResponse:
        captured["items"] = req.items
        captured["mwt"] = req.mwt
        return BatchInferResponse(
            results=[
                InferResponse(
                    result={
                        "kind": "analyzed",
                        "raw_sentences": [
                            [
                                {
                                    "id": 1,
                                    "text": "我",
                                    "lemma": "我",
                                    "upos": "PRON",
                                    "head": 2,
                                    "deprel": "nsubj",
                                },
                                {
                                    "id": 2,
                                    "text": "食緊",
                                    "lemma": "食緊",
                                    "upos": "VERB",
                                    "head": 0,
                                    "deprel": "root",
                                },
                                {
                                    "id": 3,
                                    "text": "飯",
                                    "lemma": "飯",
                                    "upos": "NOUN",
                                    "head": 2,
                                    "deprel": "obj",
                                },
                            ]
                        ],
                        "model": {
                            "stanza_version": "1.99.0",
                            "lang": "yue",
                            "pipeline": "cantonese_pycantonese_pos",
                        },
                        "repairs": [],
                    },
                    elapsed_s=0.0,
                )
            ]
        )

    response = execute_request_v2(
        request=ExecuteRequestV2(
            request_id="req-execute-v2-morphosyntax-unicode-1",
            task=InferenceTaskV2.MORPHOSYNTAX,
            payload=MorphosyntaxRequestV2(
                lang="yue",
                payload_ref_id="text-ref-morphosyntax-unicode-1",
                item_count=1,
            ),
            attachments=[
                PreparedTextRefV2(
                    id="text-ref-morphosyntax-unicode-1",
                    path=str(payload_path),
                    encoding=PreparedTextEncodingV2.UTF8_JSON,
                    byte_offset=0,
                    byte_len=payload_path.stat().st_size,
                )
            ],
        ),
        host=WorkerExecutionHostV2(
            text=TextExecutionHostV2(morphosyntax_runner=runner),
        ),
    )

    assert isinstance(response.outcome, ExecuteSuccessV2)
    assert isinstance(response.result, MorphosyntaxResultV2)
    assert response.result.items[0].raw_sentences[0][1]["text"] == "食緊"
    assert isinstance(captured["items"], list)
    assert captured["items"][0]["special_forms"][1] == ["食緊", "eat.PROG"]
    assert captured["mwt"] == {"食緊": ["食", "緊"]}


def test_invalid_morphosyntax_host_item_fails_only_that_item(
    tmp_path: Path,
) -> None:
    """A morphosyntax host item that does not parse fails that item alone."""

    payload_path = tmp_path / "morphosyntax-invalid-batch.json"
    _write_json_payload(
        payload_path,
        {
            "items": [
                {
                    "words": ["hello"],
                    "terminator": ".",
                    "special_forms": [[None, None]],
                    "lang": "eng",
                }
            ],
            "mwt": {},
        },
    )

    response = execute_request_v2(
        request=ExecuteRequestV2(
            request_id="req-execute-v2-morphosyntax-invalid-1",
            task=InferenceTaskV2.MORPHOSYNTAX,
            payload=MorphosyntaxRequestV2(
                lang="eng",
                payload_ref_id="text-ref-morphosyntax-invalid-1",
                item_count=1,
            ),
            attachments=[
                PreparedTextRefV2(
                    id="text-ref-morphosyntax-invalid-1",
                    path=str(payload_path),
                    encoding=PreparedTextEncodingV2.UTF8_JSON,
                    byte_offset=0,
                    byte_len=payload_path.stat().st_size,
                )
            ],
        ),
        host=WorkerExecutionHostV2(
            text=TextExecutionHostV2(
                morphosyntax_runner=lambda _req: BatchInferResponse(
                    results=[
                        InferResponse(result={"raw_sentences": "bad"}, elapsed_s=0.0)
                    ]
                )
            ),
        ),
    )

    _assert_item_failure(response, "invalid morphosyntax host item")


def test_morphosyntax_item_count_mismatch_is_invalid_payload(tmp_path: Path) -> None:
    """A batch whose length disagrees with the request metadata is refused
    before any model runs.

    Pins the Rust control plane's count check (`CountedBatch::load`), which
    replaced `_text_v2.py`'s `_validate_item_count` in the 2026-08-21 port:
    the runner must never be reached with a miscounted batch.
    """

    payload_path = tmp_path / "morphosyntax-batch-miscounted.json"
    item = {
        "words": ["hi"],
        "terminator": ".",
        "special_forms": [[None, None]],
        "lang": "eng",
    }
    _write_json_payload(payload_path, {"items": [item, item], "mwt": {}})

    def _must_not_run(_req: object) -> BatchInferResponse:
        raise AssertionError("the runner must not be called for a miscounted batch")

    response = execute_request_v2(
        request=ExecuteRequestV2(
            request_id="req-execute-v2-morphosyntax-miscount",
            task=InferenceTaskV2.MORPHOSYNTAX,
            payload=MorphosyntaxRequestV2(
                lang="eng",
                payload_ref_id="text-ref-morphosyntax-miscount",
                item_count=1,
            ),
            attachments=[
                PreparedTextRefV2(
                    id="text-ref-morphosyntax-miscount",
                    path=str(payload_path),
                    encoding=PreparedTextEncodingV2.UTF8_JSON,
                    byte_offset=0,
                    byte_len=payload_path.stat().st_size,
                )
            ],
        ),
        host=WorkerExecutionHostV2(
            text=TextExecutionHostV2(morphosyntax_runner=_must_not_run)
        ),
    )

    assert isinstance(response.outcome, ExecuteErrorV2)
    assert response.outcome.code is ProtocolErrorCodeV2.INVALID_PAYLOAD
    assert "payload had 2 items, expected 1" in response.outcome.message
    assert response.result is None


def test_malformed_batch_with_no_runner_reports_model_unavailable(
    tmp_path: Path,
) -> None:
    """PRECEDENCE, stated as intended (2026-08-21 port): runner availability
    is checked before the batch's per-item shape.

    The Rust control plane checks only the item COUNT before requiring a
    runner; per-item validation lives in the runner adapter, which cannot run
    without a runner. So a shape-malformed batch on a host with no runner
    reports model_unavailable, where the old Python order (pydantic first)
    reported invalid_payload. Both facts are true of such a request; the
    ruling here is that a host that cannot run the task says so first.
    """

    payload_path = tmp_path / "morphosyntax-batch-malformed.json"
    _write_json_payload(
        payload_path,
        {"items": [{"words": 7}], "mwt": {}},
    )

    response = execute_request_v2(
        request=ExecuteRequestV2(
            request_id="req-execute-v2-morphosyntax-malformed-no-runner",
            task=InferenceTaskV2.MORPHOSYNTAX,
            payload=MorphosyntaxRequestV2(
                lang="eng",
                payload_ref_id="text-ref-morphosyntax-malformed",
                item_count=1,
            ),
            attachments=[
                PreparedTextRefV2(
                    id="text-ref-morphosyntax-malformed",
                    path=str(payload_path),
                    encoding=PreparedTextEncodingV2.UTF8_JSON,
                    byte_offset=0,
                    byte_len=payload_path.stat().st_size,
                )
            ],
        ),
        host=WorkerExecutionHostV2(text=TextExecutionHostV2()),
    )

    assert isinstance(response.outcome, ExecuteErrorV2)
    assert response.outcome.code is ProtocolErrorCodeV2.MODEL_UNAVAILABLE
    assert "no morphosyntax host loaded" in response.outcome.message


def test_corrupt_prepared_batch_artifact_is_attachment_unreadable(
    tmp_path: Path,
) -> None:
    """A frozen batch artifact that is not valid JSON is corrupt STORAGE.

    Adjudicated 2026-08-21: the category is attachment_unreadable, because
    the artifact is a file Rust froze, not part of the request envelope; the
    old Python path's invalid_payload here was an accident of its exception
    ladder, not a decision.
    """

    payload_path = tmp_path / "morphosyntax-batch-corrupt.json"
    payload_path.write_text("{not json", encoding="utf-8")

    response = execute_request_v2(
        request=ExecuteRequestV2(
            request_id="req-execute-v2-morphosyntax-corrupt",
            task=InferenceTaskV2.MORPHOSYNTAX,
            payload=MorphosyntaxRequestV2(
                lang="eng",
                payload_ref_id="text-ref-morphosyntax-corrupt",
                item_count=1,
            ),
            attachments=[
                PreparedTextRefV2(
                    id="text-ref-morphosyntax-corrupt",
                    path=str(payload_path),
                    encoding=PreparedTextEncodingV2.UTF8_JSON,
                    byte_offset=0,
                    byte_len=payload_path.stat().st_size,
                )
            ],
        ),
        host=WorkerExecutionHostV2(text=TextExecutionHostV2()),
    )

    assert isinstance(response.outcome, ExecuteErrorV2)
    assert response.outcome.code is ProtocolErrorCodeV2.ATTACHMENT_UNREADABLE
    assert "held invalid JSON" in response.outcome.message


def test_batch_without_items_list_is_invalid_payload(tmp_path: Path) -> None:
    """Valid JSON with no items list is a malformed PAYLOAD, not bad storage."""

    payload_path = tmp_path / "morphosyntax-batch-itemless.json"
    _write_json_payload(payload_path, {"mwt": {}})

    response = execute_request_v2(
        request=ExecuteRequestV2(
            request_id="req-execute-v2-morphosyntax-itemless",
            task=InferenceTaskV2.MORPHOSYNTAX,
            payload=MorphosyntaxRequestV2(
                lang="eng",
                payload_ref_id="text-ref-morphosyntax-itemless",
                item_count=1,
            ),
            attachments=[
                PreparedTextRefV2(
                    id="text-ref-morphosyntax-itemless",
                    path=str(payload_path),
                    encoding=PreparedTextEncodingV2.UTF8_JSON,
                    byte_offset=0,
                    byte_len=payload_path.stat().st_size,
                )
            ],
        ),
        host=WorkerExecutionHostV2(text=TextExecutionHostV2()),
    )

    assert isinstance(response.outcome, ExecuteErrorV2)
    assert response.outcome.code is ProtocolErrorCodeV2.INVALID_PAYLOAD
    assert "must carry an items list" in response.outcome.message


def test_routes_utseg_execute_v2_request(tmp_path: Path) -> None:
    """The live V2 router should hand utseg requests to the text host."""

    payload_path = tmp_path / "utseg-batch.json"
    _write_json_payload(
        payload_path,
        {"items": [{"words": ["hello", "world"], "text": "hello world"}]},
    )

    response = execute_request_v2(
        request=ExecuteRequestV2(
            request_id="req-execute-v2-utseg-1",
            task=InferenceTaskV2.UTSEG,
            payload=UtsegRequestV2(
                lang="eng",
                payload_ref_id="text-ref-utseg-1",
                item_count=1,
            ),
            attachments=[
                PreparedTextRefV2(
                    id="text-ref-utseg-1",
                    path=str(payload_path),
                    encoding=PreparedTextEncodingV2.UTF8_JSON,
                    byte_offset=0,
                    byte_len=payload_path.stat().st_size,
                )
            ],
        ),
        host=WorkerExecutionHostV2(
            asr=AsrExecutionHostV2(),
            forced_alignment=ForcedAlignmentExecutionHostV2(),
            speaker=SpeakerExecutionHostV2(),
            text=TextExecutionHostV2(
                utseg_runner=lambda req: BatchInferResponse(
                    results=[
                        InferResponse(
                            result={"trees": ["(ROOT (S hello world))"]}, elapsed_s=0.0
                        )
                    ]
                )
            ),
        ),
    )

    assert isinstance(response.outcome, ExecuteSuccessV2)
    assert isinstance(response.result, UtsegResultV2)
    assert response.result.items[0].trees == ["(ROOT (S hello world))"]


def test_routes_utseg_execute_v2_request_with_assignments(tmp_path: Path) -> None:
    """The live V2 router should preserve direct utseg assignments."""

    payload_path = tmp_path / "utseg-batch-assignments.json"
    _write_json_payload(
        payload_path,
        {"items": [{"words": ["hello", "world"], "text": "hello world"}]},
    )

    response = execute_request_v2(
        request=ExecuteRequestV2(
            request_id="req-execute-v2-utseg-assignments-1",
            task=InferenceTaskV2.UTSEG,
            payload=UtsegRequestV2(
                lang="eng",
                payload_ref_id="text-ref-utseg-assignments-1",
                item_count=1,
            ),
            attachments=[
                PreparedTextRefV2(
                    id="text-ref-utseg-assignments-1",
                    path=str(payload_path),
                    encoding=PreparedTextEncodingV2.UTF8_JSON,
                    byte_offset=0,
                    byte_len=payload_path.stat().st_size,
                )
            ],
        ),
        host=WorkerExecutionHostV2(
            text=TextExecutionHostV2(
                utseg_runner=lambda req: BatchInferResponse(
                    results=[
                        InferResponse(result={"assignments": [0, 1]}, elapsed_s=0.0)
                    ]
                )
            ),
        ),
    )

    assert isinstance(response.outcome, ExecuteSuccessV2)
    assert isinstance(response.result, UtsegResultV2)
    assert response.result.items[0].assignments == [0, 1]


def test_routes_utseg_execute_v2_request_with_boundary_evidence(tmp_path: Path) -> None:
    """The real PyO3 boundary must preserve typed per-word model evidence."""

    payload_path = tmp_path / "utseg-batch-boundary-evidence.json"
    _write_json_payload(
        payload_path,
        {
            "items": [
                {
                    "words": ["hello", "!!!", "world"],
                    "text": "hello !!! world",
                }
            ]
        },
    )
    evidence = {
        "model_id": "talkbank/CHATUtterance-en",
        "model_revision": "0123456789abcdef0123456789abcdef01234567",
        "normalization_revision": "lower-strip-ascii-punctuation-v1",
        "adjacency_policy_revision": "suppress-earlier-adjacent-nonordinary-v1",
        "word_evidence": [
            {
                "kind": "classified",
                "raw_action": "period_boundary",
                "applied_action": "period_boundary",
                "boundary_probability_micros": 812_345,
            },
            {"kind": "normalization_omission"},
            {
                "kind": "classified",
                "raw_action": "ordinary",
                "applied_action": "ordinary",
                "boundary_probability_micros": 123_456,
            },
        ],
    }

    response = execute_request_v2(
        request=ExecuteRequestV2(
            request_id="req-execute-v2-utseg-boundary-evidence-1",
            task=InferenceTaskV2.UTSEG,
            payload=UtsegRequestV2(
                lang="eng",
                payload_ref_id="text-ref-utseg-boundary-evidence-1",
                item_count=1,
            ),
            attachments=[
                PreparedTextRefV2(
                    id="text-ref-utseg-boundary-evidence-1",
                    path=str(payload_path),
                    encoding=PreparedTextEncodingV2.UTF8_JSON,
                    byte_offset=0,
                    byte_len=payload_path.stat().st_size,
                )
            ],
        ),
        host=WorkerExecutionHostV2(
            text=TextExecutionHostV2(
                utseg_runner=lambda _req: BatchInferResponse(
                    results=[
                        InferResponse(
                            result={
                                "assignments": [0, 1, 1],
                                "boundary_model_evidence": evidence,
                            },
                            elapsed_s=0.0,
                        )
                    ]
                )
            ),
        ),
    )

    assert isinstance(response.outcome, ExecuteSuccessV2)
    assert isinstance(response.result, UtsegResultV2)
    boundary_model_evidence = response.result.items[0].boundary_model_evidence
    assert boundary_model_evidence is not None
    assert boundary_model_evidence.model_dump(mode="json") == evidence


def test_invalid_utseg_host_output_becomes_runtime_failure(tmp_path: Path) -> None:
    """Malformed utseg host output should be classified as runtime failure."""

    payload_path = tmp_path / "utseg-invalid-batch.json"
    _write_json_payload(
        payload_path, {"items": [{"words": ["hello"], "text": "hello"}]}
    )

    response = execute_request_v2(
        request=ExecuteRequestV2(
            request_id="req-execute-v2-utseg-invalid-1",
            task=InferenceTaskV2.UTSEG,
            payload=UtsegRequestV2(
                lang="eng",
                payload_ref_id="text-ref-utseg-invalid-1",
                item_count=1,
            ),
            attachments=[
                PreparedTextRefV2(
                    id="text-ref-utseg-invalid-1",
                    path=str(payload_path),
                    encoding=PreparedTextEncodingV2.UTF8_JSON,
                    byte_offset=0,
                    byte_len=payload_path.stat().st_size,
                )
            ],
        ),
        host=WorkerExecutionHostV2(
            text=TextExecutionHostV2(
                utseg_runner=lambda _req: BatchInferResponse(
                    results=[InferResponse(result={"trees": [1, 2]}, elapsed_s=0.0)]
                )
            ),
        ),
    )

    _assert_runtime_failure_response(response, "invalid utseg host output")


def test_invalid_utseg_assignment_host_output_becomes_runtime_failure(
    tmp_path: Path,
) -> None:
    """Malformed utseg assignments should be classified as runtime failure."""

    payload_path = tmp_path / "utseg-invalid-assignments-batch.json"
    _write_json_payload(
        payload_path, {"items": [{"words": ["hello"], "text": "hello"}]}
    )

    response = execute_request_v2(
        request=ExecuteRequestV2(
            request_id="req-execute-v2-utseg-invalid-assignments-1",
            task=InferenceTaskV2.UTSEG,
            payload=UtsegRequestV2(
                lang="eng",
                payload_ref_id="text-ref-utseg-invalid-assignments-1",
                item_count=1,
            ),
            attachments=[
                PreparedTextRefV2(
                    id="text-ref-utseg-invalid-assignments-1",
                    path=str(payload_path),
                    encoding=PreparedTextEncodingV2.UTF8_JSON,
                    byte_offset=0,
                    byte_len=payload_path.stat().st_size,
                )
            ],
        ),
        host=WorkerExecutionHostV2(
            text=TextExecutionHostV2(
                utseg_runner=lambda _req: BatchInferResponse(
                    results=[
                        InferResponse(result={"assignments": ["bad"]}, elapsed_s=0.0)
                    ]
                )
            ),
        ),
    )

    _assert_runtime_failure_response(response, "invalid utseg host output")


def test_routes_translate_execute_v2_request(tmp_path: Path) -> None:
    """The live V2 router should hand translate requests to the text host."""

    payload_path = tmp_path / "translate-batch.json"
    _write_json_payload(payload_path, {"items": [{"text": "hello there"}]})

    response = execute_request_v2(
        request=ExecuteRequestV2(
            request_id="req-execute-v2-translate-1",
            task=InferenceTaskV2.TRANSLATE,
            payload=TranslateRequestV2(
                source_lang="eng",
                target_lang="spa",
                payload_ref_id="text-ref-translate-1",
                item_count=1,
            ),
            attachments=[
                PreparedTextRefV2(
                    id="text-ref-translate-1",
                    path=str(payload_path),
                    encoding=PreparedTextEncodingV2.UTF8_JSON,
                    byte_offset=0,
                    byte_len=payload_path.stat().st_size,
                )
            ],
        ),
        host=WorkerExecutionHostV2(
            asr=AsrExecutionHostV2(),
            forced_alignment=ForcedAlignmentExecutionHostV2(),
            speaker=SpeakerExecutionHostV2(),
            text=TextExecutionHostV2(
                translate_runner=lambda req: BatchInferResponse(
                    results=[
                        InferResponse(
                            result={
                                "kind": "translated",
                                "raw_translation": "hola",
                                "engine": "googletrans-v1",
                            },
                            elapsed_s=0.0,
                        )
                    ]
                )
            ),
        ),
    )

    assert isinstance(response.outcome, ExecuteSuccessV2)
    assert isinstance(response.result, TranslationResultV2)
    assert response.result.items[0].raw_translation == "hola"


def test_invalid_translate_host_item_fails_only_that_item(tmp_path: Path) -> None:
    """A translate host item that does not parse fails that item alone."""

    payload_path = tmp_path / "translate-invalid-batch.json"
    _write_json_payload(payload_path, {"items": [{"text": "hello there"}]})

    response = execute_request_v2(
        request=ExecuteRequestV2(
            request_id="req-execute-v2-translate-invalid-1",
            task=InferenceTaskV2.TRANSLATE,
            payload=TranslateRequestV2(
                source_lang="eng",
                target_lang="spa",
                payload_ref_id="text-ref-translate-invalid-1",
                item_count=1,
            ),
            attachments=[
                PreparedTextRefV2(
                    id="text-ref-translate-invalid-1",
                    path=str(payload_path),
                    encoding=PreparedTextEncodingV2.UTF8_JSON,
                    byte_offset=0,
                    byte_len=payload_path.stat().st_size,
                )
            ],
        ),
        host=WorkerExecutionHostV2(
            text=TextExecutionHostV2(
                translate_runner=lambda _req: BatchInferResponse(
                    results=[
                        InferResponse(
                            result={"raw_translation": ["hola"]}, elapsed_s=0.0
                        )
                    ]
                )
            ),
        ),
    )

    _assert_item_failure(response, "invalid translate host item")


def test_routes_coref_execute_v2_request(tmp_path: Path) -> None:
    """The live V2 router should hand coref requests to the text host."""

    payload_path = tmp_path / "coref-batch.json"
    _write_json_payload(payload_path, {"items": [{"sentences": [["she"]]}]})

    response = execute_request_v2(
        request=ExecuteRequestV2(
            request_id="req-execute-v2-coref-1",
            task=InferenceTaskV2.COREF,
            payload=CorefRequestV2(
                lang="eng",
                payload_ref_id="text-ref-coref-1",
                item_count=1,
            ),
            attachments=[
                PreparedTextRefV2(
                    id="text-ref-coref-1",
                    path=str(payload_path),
                    encoding=PreparedTextEncodingV2.UTF8_JSON,
                    byte_offset=0,
                    byte_len=payload_path.stat().st_size,
                )
            ],
        ),
        host=WorkerExecutionHostV2(
            asr=AsrExecutionHostV2(),
            forced_alignment=ForcedAlignmentExecutionHostV2(),
            speaker=SpeakerExecutionHostV2(),
            text=TextExecutionHostV2(
                coref_runner=lambda req: BatchInferResponse(
                    results=[
                        InferResponse(
                            result={
                                "kind": "resolved",
                                "engine": "stanza-1.99.0",
                                "annotations": [
                                    {
                                        "sentence_idx": 0,
                                        "words": [
                                            [
                                                {
                                                    "chain_id": 0,
                                                    "is_start": True,
                                                    "is_end": True,
                                                }
                                            ]
                                        ],
                                    }
                                ],
                            },
                            elapsed_s=0.0,
                        )
                    ]
                )
            ),
        ),
    )

    assert isinstance(response.outcome, ExecuteSuccessV2)
    assert isinstance(response.result, CorefResultV2)
    assert isinstance(response.result.items[0], CorefResolvedItemV2)
    assert response.result.items[0].annotations[0].words[0][0].chain_id == 0


def test_invalid_coref_host_item_fails_only_that_item(tmp_path: Path) -> None:
    """A coref host item that does not parse fails that item alone."""

    payload_path = tmp_path / "coref-invalid-batch.json"
    _write_json_payload(payload_path, {"items": [{"sentences": [["she"]]}]})

    response = execute_request_v2(
        request=ExecuteRequestV2(
            request_id="req-execute-v2-coref-invalid-1",
            task=InferenceTaskV2.COREF,
            payload=CorefRequestV2(
                lang="eng",
                payload_ref_id="text-ref-coref-invalid-1",
                item_count=1,
            ),
            attachments=[
                PreparedTextRefV2(
                    id="text-ref-coref-invalid-1",
                    path=str(payload_path),
                    encoding=PreparedTextEncodingV2.UTF8_JSON,
                    byte_offset=0,
                    byte_len=payload_path.stat().st_size,
                )
            ],
        ),
        host=WorkerExecutionHostV2(
            text=TextExecutionHostV2(
                coref_runner=lambda _req: BatchInferResponse(
                    results=[
                        InferResponse(result={"annotations": "bad"}, elapsed_s=0.0)
                    ]
                )
            ),
        ),
    )

    _assert_item_failure(response, "invalid coref host item")


def test_returns_typed_error_for_missing_text_v2_host(tmp_path: Path) -> None:
    """Text tasks without a live V2 host should fail explicitly."""

    payload_path = tmp_path / "unsupported-text-batch.json"
    _write_json_payload(payload_path, {"items": [{"text": "hello"}]})

    response = execute_request_v2(
        request=ExecuteRequestV2(
            request_id="req-execute-v2-unsupported-1",
            task=InferenceTaskV2.TRANSLATE,
            payload=TranslateRequestV2(
                source_lang="eng",
                target_lang="spa",
                payload_ref_id="text-ref-unsupported-1",
                item_count=1,
            ),
            attachments=[
                PreparedTextRefV2(
                    id="text-ref-unsupported-1",
                    path=str(payload_path),
                    encoding=PreparedTextEncodingV2.UTF8_JSON,
                    byte_offset=0,
                    byte_len=payload_path.stat().st_size,
                )
            ],
        ),
        host=WorkerExecutionHostV2(
            asr=AsrExecutionHostV2(),
            forced_alignment=ForcedAlignmentExecutionHostV2(),
            speaker=SpeakerExecutionHostV2(),
            text=TextExecutionHostV2(),
        ),
    )

    assert isinstance(response.outcome, ExecuteErrorV2)
    assert response.outcome.code is ProtocolErrorCodeV2.MODEL_UNAVAILABLE
    assert response.result is None
    assert "no translate host loaded" in response.outcome.message


def _assert_item_failure(response: ExecuteResponseV2, message_fragment: str) -> None:
    """Assert the batch succeeded and its one item carries a parse failure.

    The bridge parses each host item into its tagged V2 type and turns one
    that does not parse into THAT item's ``failed`` outcome, the same
    per-item attribution the server applies, instead of failing the batch.
    """
    assert isinstance(response.outcome, ExecuteSuccessV2)
    assert response.result is not None
    item = response.result.items[0]
    assert item.kind == "failed"
    assert message_fragment in item.error
