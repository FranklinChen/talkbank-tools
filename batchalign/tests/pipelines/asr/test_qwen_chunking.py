"""Chunking for the native Qwen3-ASR engine.

This is one of two jobs BA3 took over from the `qwen-asr` package when the
engine moved onto `transformers`' native `qwen3_asr` module. Needs no model,
so these run in milliseconds. Runaway detection, the other job, and the
per-chunk decode budget it now reads first, are covered in
`test_qwen_decode_budget.py`.

What each property here is protecting:

* **Exact reconstruction.** A splitter that dropped or duplicated samples
  would still produce a fluent transcript, so nothing downstream would notice
  audio going missing.
* **Offsets.** The forced aligner reports times against the window it was
  handed. The offset is what makes them recording-relative, and omitting it
  produces timings that look reasonable and are silently wrong. FA emitted
  word timings 28.2 seconds past the end of their own audio on 6 of 226
  sessions once, undetected for two months, for exactly this reason.
"""

from __future__ import annotations

import numpy as np
import pytest

from batchalign.inference.languages.cantonese._qwen_chunking import (
    MIN_ASR_INPUT_SECONDS,
    SAMPLE_RATE,
    split_audio_into_chunks,
)

SR = SAMPLE_RATE


def _noise(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(seconds * SR)).astype(np.float32)


def test_short_audio_is_one_chunk_at_zero_offset() -> None:
    chunks = split_audio_into_chunks(_noise(10.0))
    assert len(chunks) == 1
    assert chunks[0].offset == 0.0


def test_chunks_reconstruct_the_original_exactly() -> None:
    wav = _noise(430.0)
    chunks = split_audio_into_chunks(wav)
    assert len(chunks) > 1
    np.testing.assert_array_equal(np.concatenate([c.samples for c in chunks]), wav)


def test_offsets_are_the_true_start_time_of_each_chunk() -> None:
    """Each offset must be exactly where its chunk begins in the recording.

    This is the number added to every word timing, so an offset that merely
    looks plausible is not good enough.
    """
    wav = _noise(500.0)
    chunks = split_audio_into_chunks(wav)

    expected = 0.0
    for chunk in chunks:
        assert chunk.offset == pytest.approx(expected, abs=1e-9)
        expected += chunk.seconds
    assert expected == pytest.approx(len(wav) / SR, abs=1e-9)


def test_the_effective_cap_exceeds_the_named_one_by_the_search_window() -> None:
    """A chunk can run past `max_chunk_sec`, by up to `search_expand_sec`.

    Worth pinning because it is easy to assume otherwise: the boundary search
    looks on BOTH sides of the target cut, so the real ceiling is 185 s rather
    than the 180 s the constant names. Anything downstream keyed to 180 has
    five seconds of slack it did not ask for.
    """
    chunks = split_audio_into_chunks(_noise(700.0))
    longest = max(c.seconds for c in chunks)
    assert longest <= 185.0 + 1e-6
    assert longest > 180.0


def test_a_short_tail_is_padded_up_to_the_model_minimum() -> None:
    """The one place chunking does not preserve the input, pinned deliberately.

    The search window is switched off so the cut lands on the target and the
    tail length is a fact about the input rather than about where the audio
    happened to go quiet.
    """
    chunks = split_audio_into_chunks(_noise(180.1), search_expand_sec=0.0)
    tail = chunks[-1]
    assert len(tail.samples) == int(MIN_ASR_INPUT_SECONDS * SR)
    assert tail.samples[-1] == 0.0
