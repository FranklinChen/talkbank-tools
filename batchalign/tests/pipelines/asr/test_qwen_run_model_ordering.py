"""`ReadyQwenRecognizer._run_model` must classify a chunk's decode outcome BEFORE
deciding whether its text counts as "no speech".

Origin: `_run_model` used to check `if not text.strip(): continue` before
ever calling `detect_runaway`. A `Bounded(DEADLINE, "")` chunk -- the
deadline fired before the model emitted anything -- has empty text exactly
like an ordinary silent `Completed` chunk, so the blank-text check silently
swallowed it: the fact that a chunk's decode was cut off by the deadline
never reached `refused`, `QwenTranscription`, or the log line downstream of
it. `detect_runaway` must run first; only a `Completed` outcome with empty
text is legitimate silence.

No model: The loaded models, `_transcribe_chunk`, and the audio load are stubbed,
so this exercises the loop's control flow alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from batchalign.inference.languages.cantonese import _qwen_common as qwen_common
from batchalign.inference.languages.cantonese._qwen_chunking import (
    SAMPLE_RATE,
    AudioChunk,
    Bounded,
    BoundReason,
    Completed,
    RunawayReason,
)


def _stub_recognizer(monkeypatch: pytest.MonkeyPatch, chunk_seconds: float = 1.0):
    recognizer = qwen_common.ReadyQwenRecognizer(
        qwen_common.QwenRecognizerConfig(
            lang="yue", model_id="Qwen/Qwen3-ASR-0.6B-hf", device="cpu"
        ),
        qwen_common.LoadedQwen(processor=object(), model=object(), aligner=object()),
    )
    fake_chunk = AudioChunk(
        samples=np.zeros(int(chunk_seconds * SAMPLE_RATE), dtype=np.float32), offset=0.0
    )
    monkeypatch.setattr(
        qwen_common, "split_audio_into_chunks", lambda wav: [fake_chunk]
    )

    import librosa

    monkeypatch.setattr(
        librosa,
        "load",
        lambda *args, **kwargs: (
            np.zeros(int(chunk_seconds * SAMPLE_RATE), dtype=np.float32),
            SAMPLE_RATE,
        ),
    )
    return recognizer


def test_a_bounded_deadline_chunk_with_no_text_still_lands_in_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recognizer = _stub_recognizer(monkeypatch)
    monkeypatch.setattr(
        qwen_common.ReadyQwenRecognizer,
        "_transcribe_chunk",
        lambda self, chunk, budget: Bounded(reason=BoundReason.DEADLINE, text=""),
    )

    result = recognizer._run_model("fake.wav")

    assert result.segments == []
    assert len(result.refused) == 1
    assert result.refused[0].reason is RunawayReason.DEADLINE


def test_a_completed_outcome_with_no_text_is_ordinary_silence_not_a_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one case the blank-text `continue` still exists for: a chunk that
    stopped on its own and simply had nothing to say."""
    recognizer = _stub_recognizer(monkeypatch)
    monkeypatch.setattr(
        qwen_common.ReadyQwenRecognizer,
        "_transcribe_chunk",
        lambda self, chunk, budget: Completed(text="   "),
    )

    result = recognizer._run_model("fake.wav")

    assert result.segments == []
    assert result.refused == []
