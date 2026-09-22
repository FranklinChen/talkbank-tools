"""A Whisper chunk without a full timestamp span keeps its words.

The HuggingFace pipeline can return a chunk whose ``timestamp`` is absent or
half-populated: ``(start, None)`` for a chunk cut off mid-word, and
``(None, None)`` for a whole transcript when the model predicted no
timestamps at all. A missing bound is an absence of TIMING, never of words.

Two earlier rules were both wrong. ``or 0.0`` invented a time (a chunk
missing only its start claimed to run from the beginning of the audio, and
reached the Rust boundary as ``Some(0.0)`` rather than ``None``). Dropping
the chunk, the rule between 536f29c8 and 2026-09-22, discarded the words with
the timing: every Cantonese fixture of the benchmark set came back as "the
ASR engine recognized no words", because the model emitted one untimed
chunk for the whole file. The bound now travels as ``None`` and the words go
downstream untimed, as a Tencent or Aliyun word with a missing offset does.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from batchalign.inference.asr import infer_whisper_prepared_audio
from batchalign.tests._asr_model_pins import loaded_identity
from batchalign.worker._types_v2 import AsrBackendV2


class _FakeWhisper:
    """Minimal stand-in for ``WhisperASRHandle``.

    Returns whatever chunk list it was constructed with, so a test can present
    the timestamp shapes the real pipeline emits without loading a model.
    """

    sample_rate = 16_000

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = chunks
        # An ASR result now names the checkpoint behind it, so even a stand-in
        # has to say what it loaded. These tests are about timestamp handling
        # and nothing here reads the identity; it is a stock pinned Whisper so
        # that the double stays a double rather than becoming a second fixture.
        self.model_identity = loaded_identity(AsrBackendV2.LOCAL_WHISPER)

    def gen_kwargs(self, _language_name: str) -> dict[str, Any]:
        return {}

    def __call__(self, _inputs: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"text": "whatever", "chunks": self._chunks}


def _run(chunks: list[dict[str, Any]]):
    return infer_whisper_prepared_audio(
        _FakeWhisper(chunks),  # type: ignore[arg-type]
        np.zeros(16_000, dtype=np.float32),
        "eng",
    )


def _spans(result) -> list[tuple[str, float | None, float | None]]:
    return [(c.text, c.start_s, c.end_s) for c in result.chunks]


def test_a_chunk_missing_its_start_keeps_its_words_and_its_end() -> None:
    """Half a span: the known bound travels, the missing one is ``None``, and
    nothing is invented in either direction."""
    assert _spans(_run([{"text": "hello", "timestamp": (None, 5.2)}])) == [
        ("hello", None, 5.2)
    ]


def test_a_whole_transcript_without_timestamps_is_kept_untimed() -> None:
    """The shape that emptied every Cantonese transcript: one chunk, no
    timestamps at all."""
    assert _spans(_run([{"text": "有個小朋友在戶外踢球", "timestamp": None}])) == [
        ("有個小朋友在戶外踢球", None, None)
    ]


def test_chunks_with_real_spans_are_kept_untouched() -> None:
    assert _spans(
        _run(
            [
                {"text": "one", "timestamp": (0.0, 0.5)},
                {"text": "two", "timestamp": (0.5, 1.25)},
            ]
        )
    ) == [("one", 0.0, 0.5), ("two", 0.5, 1.25)]


def test_a_genuine_zero_start_survives() -> None:
    """0.0 is a legal start, and must not be confused with absence."""
    assert _spans(_run([{"text": "first", "timestamp": (0.0, 0.4)}])) == [
        ("first", 0.0, 0.4)
    ]


def test_untimed_chunks_are_reported_not_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        _run(
            [
                {"text": "kept", "timestamp": (1.0, 2.0)},
                {"text": "untimed", "timestamp": (None, 5.2)},
            ]
        )
    assert any("untimed" in record.message for record in caplog.records)


def test_an_inverted_chunk_travels_raw_for_rust_to_settle() -> None:
    """Order is not this producer's to guess: the Rust consumer projects every
    producer's spans in one place, so an inverted chunk is passed as emitted."""
    assert _spans(_run([{"text": "x", "timestamp": (2.0, 1.0)}])) == [("x", 2.0, 1.0)]
