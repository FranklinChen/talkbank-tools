"""Taiwanese recognition preserves language, lexical text and word timing."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from batchalign.inference.languages.cantonese._qwen_chunking import AudioChunk
from batchalign.inference.languages.cantonese._qwen_common import (
    LoadedQwen,
    QwenRecognizerConfig,
    ReadyQwenRecognizer,
)
from batchalign.inference.qwen_asr_profile import QwenAsrProfile
from batchalign.inference.qwen_forced_alignment import (
    SpokenWindowWords,
    WindowAlignedWord,
    WindowAlignment,
    resolve_qwen_language,
)
from batchalign.inference.qwen_punctuation import PunctuatedAlignment


def _spoken(words):
    return SpokenWindowWords.of_units(
        WindowAlignedWord(word, float(i), float(i + 1)) for i, word in enumerate(words)
    )


def test_taiwanese_recognition_does_not_widen_standalone_alignment_support():
    profile = QwenAsrProfile.for_language("nan")
    assert profile.recognition_language == "Chinese"
    assert profile.alignment_language == "Chinese"
    assert profile.prompt is None
    assert profile.chunk_seconds == 30
    with pytest.raises(ValueError):
        resolve_qwen_language("nan")
    for code, label in [("eng", "English"), ("yue", "Cantonese"), ("zho", "Chinese")]:
        other = QwenAsrProfile.for_language(code)
        assert other.recognition_language == label
        assert other.prompt is None
        assert other.chunk_seconds == 180


@pytest.mark.parametrize(
    "text, words, expected",
    [
        ("你好！我好。", ["你", "好", "我", "好"], ("你", "好！", "我", "好。")),
        ("你好，hello?", ["你", "好", "hello"], ("你", "好，", "hello?")),
        ("『好。』", ["好"], ("好。",)),
    ],
)
def test_punctuation_joins_only_lexically_identical_units(text, words, expected):
    assert (
        tuple(word.text for word in PunctuatedAlignment(text, _spoken(words)).words)
        == expected
    )


@pytest.mark.parametrize(
    "words", [["你", "壞"], ["你"], ["你", "好", "啊"], ["你", "!", "好"]]
)
def test_punctuation_refuses_changed_or_missing_words(words):
    with pytest.raises(ValueError):
        PunctuatedAlignment("你好。", _spoken(words))


def test_recognizer_retains_punctuation_and_exact_file_relative_times():
    class Aligner:
        def align(self, audio, text, language):
            assert language == "Chinese"
            return WindowAlignment.of(
                ["你", "好"],
                [WindowAlignedWord("你", 0.1, 0.4), WindowAlignedWord("好", 0.5, 0.9)],
            )

    recognizer = ReadyQwenRecognizer(
        QwenRecognizerConfig(
            lang="nan", model_id="Qwen/Qwen3-ASR-1.7B-hf", device="cpu"
        ),
        LoadedQwen(processor=object(), model=object(), aligner=Aligner()),
    )
    chunk = AudioChunk(samples=np.zeros(16000, dtype=np.float32), offset=30.0)
    words = recognizer._align_chunk(chunk, "你好。")
    assert words == [(30.1, 30.4, "你"), (30.5, 30.9, "好。")]


def test_config_load_transitions_to_a_ready_recognizer(monkeypatch):
    from batchalign.inference.languages.cantonese import _qwen_asr

    loaded = LoadedQwen(processor=object(), model=object(), aligner=object())
    monkeypatch.setattr(LoadedQwen, "load", classmethod(lambda cls, **kwargs: loaded))
    monkeypatch.setattr(_qwen_asr, "_recognizer", None)
    _qwen_asr.load_qwen_asr("nan", {"qwen_model": "Qwen/Qwen3-ASR-1.7B-hf"})
    ready = _qwen_asr._recognizer
    assert isinstance(ready, ReadyQwenRecognizer)
    assert ready.models is loaded
    assert ready.config.profile.recognition_language == "Chinese"
    assert not hasattr(ready.config, "transcribe")
    with pytest.raises(FrozenInstanceError):
        ready.config.lang = "eng"


def test_a_failed_load_publishes_no_ready_recognizer(monkeypatch):
    from batchalign.inference.languages.cantonese import _qwen_asr

    def fail(cls, **kwargs):
        raise RuntimeError("model load failed")

    monkeypatch.setattr(LoadedQwen, "load", classmethod(fail))
    monkeypatch.setattr(_qwen_asr, "_recognizer", None)
    with pytest.raises(RuntimeError, match="model load failed"):
        _qwen_asr.load_qwen_asr("nan", None)
    assert _qwen_asr._recognizer is None


def test_punctuation_does_not_erase_combining_tone_marks():
    with pytest.raises(ValueError):
        PunctuatedAlignment("a\u0301.", _spoken(["a"]))
    aligned = _spoken(["a\u0301"])
    result = PunctuatedAlignment("a\u0301.", aligned)
    assert result.words == (WindowAlignedWord("a\u0301.", 0.0, 1.0),)


@pytest.mark.parametrize("context", ["", "臺灣台語與國語混合對話"])
def test_worker_context_override_is_part_of_the_ready_profile(monkeypatch, context):
    from batchalign.inference.languages.cantonese import _qwen_asr

    loaded = LoadedQwen(processor=object(), model=object(), aligner=object())
    monkeypatch.setattr(LoadedQwen, "load", classmethod(lambda cls, **kwargs: loaded))
    monkeypatch.setattr(_qwen_asr, "_recognizer", None)
    _qwen_asr.load_qwen_asr("nan", {"qwen_context": context})
    ready = _qwen_asr._recognizer
    assert ready is not None
    assert ready.config.profile.prompt == (context or None)
    assert ready.config.lang == "nan"


def test_invalid_context_cannot_be_admitted_to_the_ready_configuration():
    with pytest.raises(TypeError, match="qwen_context must be text"):
        QwenRecognizerConfig(lang="nan", context_override=42)
