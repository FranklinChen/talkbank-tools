"""Window, absence, placement and publication boundaries use real consumers."""

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from batchalign.inference.languages.cantonese._qwen_chunking import (
    SAMPLE_RATE,
    AudioChunk,
)
from batchalign.inference.languages.cantonese._qwen_common import (
    QwenTranscription,
    ReadyQwenRecognizer,
)
from batchalign.inference.qwen_forced_alignment import (
    IndexedWindowUnit,
    PairedWindowAlignment,
    QwenFaHost,
    QwenTimingInvalid,
    TimedChatWord,
    UntimedChatWord,
    UntimedReason,
    WindowAlignedWord,
    resolve_qwen_language,
)
from batchalign.inference.qwen_timing import (
    AudioWindow,
    CompletedRecording,
    ObservedRecordingSpan,
    ObservedWindowSpan,
    QwenRecordingOrderInvalid,
    RecordingChunkSpeech,
    RecordingStart,
    SpokenWindowWords,
    UnavailableWindowTiming,
    WindowAlignment,
)
from batchalign.tests.pipelines.fa.test_qwen_forced_alignment import (
    _QwenLikeAligner,
    _ready_qwen,
    _RecordingAligner,
)


def _speech(words, duration=1.0):
    paired = PairedWindowAlignment.of([word.text for word in words], words)
    return SpokenWindowWords(WindowAlignment(paired, AudioWindow(duration)))


@pytest.mark.parametrize("start,end", [(1.1, 1.2), (0.8, 1.01), (2.0, 3.0)])
def test_the_asr_boundary_names_out_of_window_times_and_recording_location(start, end):
    recognizer = _ready_qwen(
        _RecordingAligner([WindowAlignedWord("hello", start, end)])
    )
    with pytest.raises(QwenTimingInvalid) as error:
        recognizer._align_chunk(AudioChunk(np.zeros(SAMPLE_RATE), 180.0), "hello")
    detail = str(error.value)
    assert "recording window 180..181 s" in detail
    assert "unit 1 'hello'" in detail
    assert "outside its 0..1 s audio window" in detail


def test_an_exact_window_end_is_admitted_and_positionless_time_is_explicit():
    speech = _speech(
        [WindowAlignedWord("hello", 0.1, 1.0), WindowAlignedWord("unknown", 1.0, 1.0)]
    )
    assert isinstance(speech.words[0].timing, ObservedWindowSpan)
    assert isinstance(speech.words[1].timing, UnavailableWindowTiming)
    with pytest.raises(FrozenInstanceError):
        speech.window.duration_s = 2.0


@pytest.mark.parametrize(
    "spans",
    [
        [(0.0, 0.0), (0.2, 0.4)],
        [(0.1, 0.4), (0.0, 0.0)],
        [(0.0, 0.0), (0.0, 0.0)],
        [(0.1, 0.2), (0.0, 0.0), (0.3, 0.4)],
    ],
)
def test_an_unavailable_constituent_cannot_become_a_timed_word(spans):
    word = "你" * len(spans)
    host = QwenFaHost(_QwenLikeAligner(spans), resolve_qwen_language("yue"))
    fold = host.align_words(np.zeros(SAMPLE_RATE), [word])
    assert isinstance(fold.words[0], UntimedChatWord)
    assert fold.words[0].reason is UntimedReason.ALIGNMENT_UNIT_UNAVAILABLE
    assert fold.to_indexed_timings()[0].interval_ms is None


def test_a_positive_span_that_rounds_away_is_explicitly_unavailable():
    host = QwenFaHost(
        _QwenLikeAligner([(0.10001, 0.10002)]), resolve_qwen_language("yue")
    )
    fold = host.align_words(np.zeros(SAMPLE_RATE), ["你"])
    assert isinstance(fold.words[0], UntimedChatWord)
    assert fold.words[0].reason is UntimedReason.BELOW_MILLISECOND_RESOLUTION


def test_direct_constructors_cannot_forge_observed_or_unavailable_states():
    zero = IndexedWindowUnit(0, WindowAlignedWord("hello", 0.0, 0.0))
    positive = IndexedWindowUnit(0, WindowAlignedWord("hello", 0.1, 0.4))
    with pytest.raises(ValueError):
        ObservedWindowSpan(zero, AudioWindow(1.0))
    with pytest.raises(ValueError):
        UnavailableWindowTiming(positive, AudioWindow(1.0))
    with pytest.raises(ValueError):
        TimedChatWord("hello", 100, 0)


@pytest.mark.parametrize("offset", [179.0, 180.0, 180.5])
def test_recording_publication_refuses_rewinding_or_overlapping_chunks(offset):
    speech = _speech([WindowAlignedWord("hello", 0.1, 0.4)])
    first = RecordingChunkSpeech("hello", 180.0, speech)
    prefix = RecordingStart().append(first)
    with pytest.raises(QwenRecordingOrderInvalid) as error:
        prefix.append(RecordingChunkSpeech("hello", offset, speech))
    assert "180..181 s" in str(error.value)
    assert CompletedRecording(prefix).chunks == (first,)


def test_valid_gaps_and_contiguous_windows_publish_an_immutable_sequence():
    speech = _speech([WindowAlignedWord("hello", 0.1, 0.4)])
    chunks = tuple(
        RecordingChunkSpeech("hello", offset, speech)
        for offset in [180.0, 181.0, 184.0]
    )
    prefix = RecordingStart()
    for chunk in chunks:
        prefix = prefix.append(chunk)
    recording = CompletedRecording(prefix)
    assert recording.chunks == chunks
    spans = [chunk.words[0].timing for chunk in recording.chunks]
    assert all(isinstance(span, ObservedRecordingSpan) for span in spans)
    assert [span.start_s for span in spans] == [180.1, 181.1, 184.1]


def test_asr_wire_keeps_unavailable_word_text_without_fabricating_a_timestamp(
    monkeypatch,
):
    recognizer = _ready_qwen(
        _RecordingAligner([WindowAlignedWord("unknown", 0.0, 0.0)])
    )
    chunk = recognizer._align_chunk(AudioChunk(np.zeros(SAMPLE_RATE), 180.0), "unknown")
    result = QwenTranscription(CompletedRecording(RecordingStart().append(chunk)), ())
    monkeypatch.setattr(
        ReadyQwenRecognizer, "_run_model", lambda *_args, **_kwargs: result
    )
    payload, timed = recognizer.transcribe("unused.wav")
    element = payload["monologues"][0]["elements"][0]
    assert element["value"] == "unknown"
    assert element["ts"] is None and element["end_ts"] is None
    assert timed == []


def test_the_loaded_host_names_the_shared_admission_revision_for_cache_replay(
    monkeypatch,
):
    from batchalign.inference import qwen_forced_alignment
    from batchalign.worker._handlers import _reported_engine
    from batchalign.worker._model_loading.forced_alignment import load_fa_engine
    from batchalign.worker._types import (
        InferTask,
        NumSpeakers,
        WorkerBootstrapRuntime,
        _state,
    )

    fixture = json.loads(
        (
            Path(__file__).resolve().parents[4]
            / "test-fixtures/qwen_fa_admission_identity.json"
        ).read_text()
    )
    host = QwenFaHost(_QwenLikeAligner(), resolve_qwen_language("yue"))
    assert host.cache_identity.model_id == fixture["legacy"]
    assert host.cache_identity.wire_name == fixture["current"]
    assert fixture["legacy"] != fixture["current"]
    monkeypatch.setattr(_state, "fa_model_name", None)
    monkeypatch.setattr(_state, "qwen_fa_host", None)
    monkeypatch.setattr(
        qwen_forced_alignment, "load_qwen_fa", lambda *_args, **_kwargs: host
    )
    load_fa_engine(
        WorkerBootstrapRuntime(
            task=InferTask.FA,
            lang="yue",
            num_speakers=NumSpeakers(2),
            engine_overrides={"fa": "qwen3_fa"},
        )
    )
    assert _reported_engine(InferTask.FA) == fixture["current"]


def _recording_run(monkeypatch, second_offset):
    import librosa

    from batchalign.inference.languages.cantonese import _qwen_common
    from batchalign.inference.languages.cantonese._qwen_chunking import Completed

    recognizer = _ready_qwen(_RecordingAligner([WindowAlignedWord("hello", 0.1, 0.4)]))
    chunks = [
        AudioChunk(np.zeros(SAMPLE_RATE), offset) for offset in [0.0, second_offset]
    ]
    monkeypatch.setattr(
        librosa,
        "load",
        lambda *_args, **_kwargs: (np.zeros(2 * SAMPLE_RATE), SAMPLE_RATE),
    )
    monkeypatch.setattr(
        _qwen_common, "split_audio_into_chunks", lambda *_args, **_kwargs: chunks
    )
    monkeypatch.setattr(
        ReadyQwenRecognizer, "_transcribe_chunk", lambda *_args: Completed("hello")
    )
    return recognizer._run_model("unused.wav")


def test_the_recording_consumer_cannot_publish_overlapping_chunks(monkeypatch):
    with pytest.raises(QwenRecordingOrderInvalid):
        _recording_run(monkeypatch, 0.5)


def test_the_recording_consumer_can_publish_contiguous_chunks(monkeypatch):
    result = _recording_run(monkeypatch, 1.0)
    assert len(result.segments) == 2
    assert [chunk.window.start_s for chunk in result.segments] == [0.0, 1.0]
