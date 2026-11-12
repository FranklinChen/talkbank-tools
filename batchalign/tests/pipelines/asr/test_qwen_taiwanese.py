"""Taiwanese recognition preserves language, lexical text and word timing."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from batchalign.inference.languages.cantonese._qwen_chunking import (
    MAX_FORCE_ALIGN_INPUT_SECONDS,
    AudioChunk,
)
from batchalign.inference.languages.cantonese._qwen_common import (
    LoadedQwen,
    QwenRecognizerConfig,
    QwenTranscription,
    ReadyQwenRecognizer,
)
from batchalign.inference.qwen_asr_profile import QwenAsrProfile
from batchalign.inference.qwen_forced_alignment import (
    PairedWindowAlignment,
    WindowAlignedWord,
    resolve_qwen_language,
)
from batchalign.inference.qwen_punctuation import (
    LexicalDrift,
    NonlexicalUnit,
    PunctuatedAlignment,
    restore_punctuation,
)
from batchalign.inference.qwen_timing import (
    AudioWindow,
    CompletedRecording,
    ObservedRecordingSpan,
    RecordingStart,
    SpokenWindowWords,
    WindowAlignment,
)
from batchalign.tests._qwen_recognition import (
    FAKE_QWEN_SNAPSHOTS,
    SEGMENTATION_MODEL_POLICY,
    TAIWANESE_POLICY,
)
from batchalign.worker._model_loading.asr import QWEN_RECOGNITION_KEY, qwen_recognition
from batchalign.worker._types_v2 import RecognitionPunctuationV2


def _spoken(words):
    reported = [
        WindowAlignedWord(word, float(i), float(i + 1)) for i, word in enumerate(words)
    ]
    return SpokenWindowWords(
        WindowAlignment(
            PairedWindowAlignment.of(words, reported), AudioWindow(float(len(words)))
        )
    )


def test_taiwanese_recognition_does_not_widen_standalone_alignment_support():
    profile = QwenAsrProfile.for_language("nan", TAIWANESE_POLICY)
    assert profile.recognition_language == "Chinese"
    assert profile.alignment_language == "Chinese"
    assert profile.prompt is None
    assert profile.chunk_seconds == 30
    assert profile.punctuation is RecognitionPunctuationV2.RESTORE
    with pytest.raises(ValueError):
        resolve_qwen_language("nan")
    for code, label in [("eng", "English"), ("yue", "Cantonese"), ("zho", "Chinese")]:
        other = QwenAsrProfile.for_language(code, SEGMENTATION_MODEL_POLICY)
        assert other.recognition_language == label
        assert other.prompt is None
        # The window and punctuation are the policy's, not the language's.
        assert other.chunk_seconds == MAX_FORCE_ALIGN_INPUT_SECONDS
        assert other.punctuation is RecognitionPunctuationV2.DISCARD


@pytest.mark.parametrize(
    "text, words, expected",
    [
        ("你好！我好。", ["你", "好", "我", "好"], ("你", "好！", "我", "好。")),
        ("你好，hello?", ["你", "好", "hello"], ("你", "好，", "hello?")),
        ("『好。』", ["好"], ("好。",)),
    ],
)
def test_punctuation_joins_only_lexically_identical_units(text, words, expected):
    restored = restore_punctuation(text, _spoken(words))
    assert isinstance(restored, PunctuatedAlignment)
    assert tuple(word.text for word in restored.words) == expected


@pytest.mark.parametrize(
    "words, refusal",
    [
        (["你", "壞"], LexicalDrift),
        (["你"], LexicalDrift),
        (["你", "好", "啊"], LexicalDrift),
        (["你", "!", "好"], NonlexicalUnit),
    ],
)
def test_punctuation_refuses_changed_or_missing_words_without_raising(words, refusal):
    """The production edge never raises on model output: it says why."""
    assert isinstance(restore_punctuation("你好。", _spoken(words)), refusal)


def _drifting_recognizer(units):
    class Aligner:
        def align(self, audio, text, language):
            return PairedWindowAlignment.of(
                [unit for unit, _start, _end in units],
                [WindowAlignedWord(unit, start, end) for unit, start, end in units],
            )

    return ReadyQwenRecognizer(
        QwenRecognizerConfig(lang="nan", device="cpu", recognition=TAIWANESE_POLICY),
        LoadedQwen(processor=object(), model=object(), aligner=Aligner()),
    )


def test_a_respelled_chunk_keeps_the_recognized_text_untimed(caplog):
    """The aligner spelled different words than were recognized: its spelling
    never replaces the recognized text, and no time is attributed to it."""
    recognizer = _drifting_recognizer([("你", 0.1, 0.4), ("壞", 0.5, 0.9)])
    chunk = AudioChunk(samples=np.zeros(16000, dtype=np.float32), offset=30.0)
    with caplog.at_level("WARNING"):
        speech = recognizer._align_chunk(chunk, "你好。")
    assert speech.text == "你好。"
    assert speech.words == ()
    assert "does not spell the recognized text" in caplog.text
    assert "recording window 30..31 s" in caplog.text


def test_a_respelled_chunk_reaches_the_transcript_as_recognized_untimed_text(
    monkeypatch,
):
    recognizer = _drifting_recognizer([("你", 0.1, 0.4), ("壞", 0.5, 0.9)])
    chunk = recognizer._align_chunk(
        AudioChunk(samples=np.zeros(16000, dtype=np.float32), offset=30.0), "你好。"
    )
    result = QwenTranscription(CompletedRecording(RecordingStart().append(chunk)), ())
    monkeypatch.setattr(
        ReadyQwenRecognizer, "_run_model", lambda *_args, **_kwargs: result
    )
    (element,) = recognizer.transcribe("unused.wav")["monologues"][0]["elements"]
    assert (element["value"], element["ts"], element["end_ts"]) == (
        "你好。",
        None,
        None,
    )


def test_a_nonlexical_unit_keeps_timed_words_without_punctuation(caplog):
    """The units still spell the recognized text, so their timing stands; only
    punctuation is lost, and the log says why."""
    recognizer = _drifting_recognizer(
        [("你", 0.1, 0.4), ("!", 0.45, 0.5), ("好", 0.5, 0.9)]
    )
    chunk = AudioChunk(samples=np.zeros(16000, dtype=np.float32), offset=30.0)
    with caplog.at_level("WARNING"):
        speech = recognizer._align_chunk(chunk, "你好。")
    assert [
        (word.timing.start_s, word.timing.end_s, word.text)
        for word in speech.words
        if isinstance(word.timing, ObservedRecordingSpan)
    ] == [(30.1, 30.4, "你"), (30.45, 30.5, "!"), (30.5, 30.9, "好")]
    assert "punctuation not restored in recording window 30..31 s" in caplog.text


def test_recognizer_retains_punctuation_and_exact_file_relative_times():
    class Aligner:
        def align(self, audio, text, language):
            assert language == "Chinese"
            return PairedWindowAlignment.of(
                ["你", "好"],
                [WindowAlignedWord("你", 0.1, 0.4), WindowAlignedWord("好", 0.5, 0.9)],
            )

    recognizer = ReadyQwenRecognizer(
        QwenRecognizerConfig(
            lang="nan",
            device="cpu",
            recognition=TAIWANESE_POLICY,
        ),
        LoadedQwen(processor=object(), model=object(), aligner=Aligner()),
    )
    chunk = AudioChunk(samples=np.zeros(16000, dtype=np.float32), offset=30.0)
    words = recognizer._align_chunk(chunk, "你好。")
    assert [
        (word.timing.start_s, word.timing.end_s, word.text)
        for word in words.words
        if isinstance(word.timing, ObservedRecordingSpan)
    ] == [(30.1, 30.4, "你"), (30.5, 30.9, "好。")]


def test_a_discarding_profile_keeps_only_the_aligned_units():
    """Cantonese leaves boundaries to segmentation: the same text and alignment
    yield the bare units, with the recognizer's punctuation dropped."""

    class Aligner:
        def align(self, audio, text, language):
            assert language == "Cantonese"
            return PairedWindowAlignment.of(
                ["你", "好"],
                [WindowAlignedWord("你", 0.1, 0.4), WindowAlignedWord("好", 0.5, 0.9)],
            )

    recognizer = ReadyQwenRecognizer(
        QwenRecognizerConfig(
            lang="yue",
            device="cpu",
            recognition=SEGMENTATION_MODEL_POLICY,
        ),
        LoadedQwen(processor=object(), model=object(), aligner=Aligner()),
    )
    chunk = AudioChunk(samples=np.zeros(16000, dtype=np.float32), offset=30.0)
    words = recognizer._align_chunk(chunk, "你好。")
    assert [
        (word.timing.start_s, word.timing.end_s, word.text)
        for word in words.words
        if isinstance(word.timing, ObservedRecordingSpan)
    ] == [(30.1, 30.4, "你"), (30.5, 30.9, "好")]


def test_config_load_transitions_to_a_ready_recognizer(monkeypatch):
    from batchalign.inference.languages.cantonese import _qwen_asr

    loaded = LoadedQwen(processor=object(), model=object(), aligner=object())
    monkeypatch.setattr(LoadedQwen, "load", classmethod(lambda cls, **kwargs: loaded))
    ready = _qwen_asr.load_qwen_asr(
        "nan",
        None,
        recognition=TAIWANESE_POLICY,
        snapshots=FAKE_QWEN_SNAPSHOTS,
    ).recognizer
    assert isinstance(ready, ReadyQwenRecognizer)
    assert ready.models is loaded
    assert ready.config.profile.recognition_language == "Chinese"
    assert ready.config.profile.punctuation is RecognitionPunctuationV2.RESTORE
    assert not hasattr(ready.config, "transcribe")
    with pytest.raises(FrozenInstanceError):
        ready.config.lang = "eng"


def test_a_failed_load_raises_and_publishes_no_recognizer(monkeypatch):
    """A failed load raises, so there is no returned recognizer to record, and
    the module holds no recognizer that could be left half published."""
    from batchalign.inference.languages.cantonese import _qwen_asr

    def fail(cls, **kwargs):
        raise RuntimeError("model load failed")

    monkeypatch.setattr(LoadedQwen, "load", classmethod(fail))
    with pytest.raises(RuntimeError, match="model load failed"):
        _qwen_asr.load_qwen_asr(
            "nan",
            None,
            recognition=TAIWANESE_POLICY,
            snapshots=FAKE_QWEN_SNAPSHOTS,
        )
    assert not hasattr(_qwen_asr, "_recognizer")


def test_the_worker_boundary_requires_the_control_plane_policy():
    """The worker never decides the policy from the language code itself: a
    spawn without it, or with a field only the other side knows, is refused."""
    spawn = {QWEN_RECOGNITION_KEY: TAIWANESE_POLICY.model_dump_json()}
    assert qwen_recognition(spawn) == TAIWANESE_POLICY
    with pytest.raises(ValueError, match=QWEN_RECOGNITION_KEY):
        qwen_recognition({"qwen_model": "Qwen/Qwen3-ASR-1.7B-hf"})
    with pytest.raises(ValueError):
        qwen_recognition(
            {
                QWEN_RECOGNITION_KEY: (
                    '{"window":{"kind":"chunker_default"},'
                    '"punctuation":"discard","novel":1}'
                )
            }
        )


def test_punctuation_does_not_erase_combining_tone_marks():
    assert isinstance(restore_punctuation("a\u0301.", _spoken(["a"])), LexicalDrift)
    aligned = _spoken(["a\u0301"])
    result = restore_punctuation("a\u0301.", aligned)
    assert isinstance(result, PunctuatedAlignment)
    assert result.words[0].text == "a\u0301."
    assert result.words[0].timing is aligned.words[0].timing


@pytest.mark.parametrize("context", ["", "臺灣台語與國語混合對話"])
def test_worker_context_override_is_part_of_the_ready_profile(monkeypatch, context):
    from batchalign.inference.languages.cantonese import _qwen_asr

    loaded = LoadedQwen(processor=object(), model=object(), aligner=object())
    monkeypatch.setattr(LoadedQwen, "load", classmethod(lambda cls, **kwargs: loaded))
    ready = _qwen_asr.load_qwen_asr(
        "nan",
        {"qwen_context": context},
        recognition=TAIWANESE_POLICY,
        snapshots=FAKE_QWEN_SNAPSHOTS,
    ).recognizer
    assert ready.config.profile.prompt == (context or None)
    assert ready.config.lang == "nan"


def test_invalid_context_cannot_be_admitted_to_the_ready_configuration():
    with pytest.raises(TypeError, match="qwen_context must be text"):
        QwenRecognizerConfig(
            lang="nan", context_override=42, recognition=TAIWANESE_POLICY
        )
