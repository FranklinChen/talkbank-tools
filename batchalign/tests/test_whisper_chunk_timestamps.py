"""A Whisper chunk without a usable span must not be given one.

The HuggingFace pipeline can return a chunk whose ``timestamp`` is absent or
half-populated. The wire type it feeds, ``WhisperChunkSpanV2``, requires two
non-negative floats and validates ``end_s >= start_s``, so a producer that
hands it a missing value has to do something. It used to substitute ``0.0``,
which is not an absence: it is a real time at the very start of the audio, and
a chunk missing only its start then claimed to span from zero to its real end.

That also defeated the ``Option`` on the Rust side, which receives ``Some(0.0)``
rather than ``None`` and so never filters the token out.
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


def test_a_chunk_missing_its_start_is_not_stretched_back_to_zero() -> None:
    """The case that produced the worst value: half a span.

    With ``or 0.0`` this became a chunk spanning 0 to 5.2 seconds, which passes
    every downstream check because both numbers are real and ordered.
    """
    result = _run([{"text": "hello", "timestamp": (None, 5.2)}])
    assert [c.text for c in result.chunks] == []


def test_a_chunk_missing_both_ends_is_dropped_rather_than_placed_at_zero() -> None:
    result = _run([{"text": "hello", "timestamp": None}])
    assert result.chunks == []


def test_chunks_with_real_spans_are_kept_untouched() -> None:
    result = _run(
        [
            {"text": "one", "timestamp": (0.0, 0.5)},
            {"text": "two", "timestamp": (0.5, 1.25)},
        ]
    )
    assert [(c.text, c.start_s, c.end_s) for c in result.chunks] == [
        ("one", 0.0, 0.5),
        ("two", 0.5, 1.25),
    ]


def test_a_genuine_zero_start_survives() -> None:
    """0.0 is a legal start, and must not be confused with absence.

    This is the case that makes ``or 0.0`` wrong in both directions: it cannot
    tell a missing value from a real first word.
    """
    result = _run([{"text": "first", "timestamp": (0.0, 0.4)}])
    assert [(c.start_s, c.end_s) for c in result.chunks] == [(0.0, 0.4)]


def test_dropped_chunks_are_reported_not_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Dropping without saying so would trade one silent defect for another."""
    with caplog.at_level("WARNING"):
        _run(
            [
                {"text": "kept", "timestamp": (1.0, 2.0)},
                {"text": "dropped", "timestamp": (None, 5.2)},
            ]
        )
    assert any("timestamp" in record.message.lower() for record in caplog.records)


def test_an_inverted_chunk_travels_raw_for_rust_to_settle() -> None:
    """Order is not this producer's to guess: the Rust consumer projects every
    producer's spans in one place, so an inverted chunk is passed as emitted."""
    result = _run([{"text": "x", "timestamp": (2.0, 1.0)}])
    assert [(c.start_s, c.end_s) for c in result.chunks] == [(2.0, 1.0)]
