"""Per-chunk and per-request decode budgeting, and the typed decode outcome,
for the native Qwen3-ASR engine.

Origin: a 411 s fleet job ran past the 1800 s job ceiling twice. Every ~180 s
chunk was decoded against one flat `max_new_tokens` cap with no wall-clock
stop, so a repetition-loop chunk paid the full cost of 4096 tokens of
greedy decoding before the runaway detector ever looked at the result, and a
10 s chunk was budgeted identically to a 180 s one.

`DecodeBudget` is the file-level wall-clock decode budget for one request,
built exactly once. `ChunkDecodeBudget.for_chunk` draws a chunk's own limits
down from what remains of it, so the SUM of every chunk's deadline can never
exceed the budget the request started with -- a per-chunk deadline computed
from the chunk alone, with no file-level ceiling, is exactly the shape that
let the 1800 s job ceiling blow past twice in one run.

`ChunkDecodeOutcome` (`Bounded` / `Completed`) records which bound fired, if
either, so `detect_runaway` no longer has to infer it from character density
alone.

None of this needs a model: everything here is a pure function over
`AudioChunk`, `DecodeBudget`, and the typed outcome.
"""

from __future__ import annotations

import numpy as np
import pytest

from batchalign.inference.languages.cantonese._qwen_chunking import (
    DEADLINE_REALTIME_FACTOR,
    MAX_NEW_TOKENS_CEILING,
    SAMPLE_RATE,
    AudioChunk,
    Bounded,
    BoundReason,
    ChunkDecodeBudget,
    Completed,
    DecodeBudget,
    RunawayReason,
    detect_runaway,
)

SR = SAMPLE_RATE


def _chunk(seconds: float, offset: float = 0.0) -> AudioChunk:
    return AudioChunk(
        samples=np.zeros(int(seconds * SR), dtype=np.float32), offset=offset
    )


def _ample_budget() -> DecodeBudget:
    """A remaining file budget large enough that it never constrains a
    single chunk under test, so tests about per-chunk PROPORTIONAL scaling
    are not incidentally also tests about the file-level cap."""
    return DecodeBudget(seconds=1_000_000.0)


# --- DecodeBudget ------------------------------------------------------------


def test_for_total_seconds_derives_from_the_named_realtime_factor() -> None:
    budget = DecodeBudget.for_total_seconds(100.0)
    assert budget.seconds == pytest.approx(100.0 * DEADLINE_REALTIME_FACTOR)


def test_remaining_after_subtracts_what_was_spent() -> None:
    budget = DecodeBudget(seconds=100.0)
    assert budget.remaining_after(40.0).seconds == pytest.approx(60.0)


def test_remaining_after_floors_at_zero_rather_than_going_negative() -> None:
    budget = DecodeBudget(seconds=10.0)
    assert budget.remaining_after(1_000.0).seconds == 0.0


# --- ChunkDecodeBudget.for_chunk ---------------------------------------------


def test_a_short_chunk_gets_fewer_tokens_than_a_long_one() -> None:
    """The defect this replaces: every chunk got the same flat 4096-token
    cap regardless of how little audio it held."""
    short_budget = ChunkDecodeBudget.for_chunk(_chunk(10.0), _ample_budget())
    long_budget = ChunkDecodeBudget.for_chunk(_chunk(180.0), _ample_budget())
    assert short_budget.max_new_tokens < long_budget.max_new_tokens


def test_max_new_tokens_never_exceeds_the_ceiling() -> None:
    """The 185 s chunk is the reference length the ceiling was calibrated
    for (the effective max after the chunker's own boundary search), so the
    scaled budget must land exactly on it, never above."""
    budget = ChunkDecodeBudget.for_chunk(_chunk(185.0), _ample_budget())
    assert budget.max_new_tokens == MAX_NEW_TOKENS_CEILING

    over_length_budget = ChunkDecodeBudget.for_chunk(_chunk(600.0), _ample_budget())
    assert over_length_budget.max_new_tokens == MAX_NEW_TOKENS_CEILING


def test_max_new_tokens_is_never_zero_for_a_real_chunk() -> None:
    """A chunk this short only ever occurs padded to the model minimum by
    `split_audio_into_chunks`, but the budget must not divide-by-zero or
    hand `generate` a cap of 0 regardless of who constructs the chunk."""
    budget = ChunkDecodeBudget.for_chunk(_chunk(0.5), _ample_budget())
    assert budget.max_new_tokens >= 1


def test_deadline_scales_with_chunk_length_when_the_file_budget_is_ample() -> None:
    budget = ChunkDecodeBudget.for_chunk(_chunk(10.0), _ample_budget())
    assert budget.deadline_seconds == pytest.approx(10.0 * DEADLINE_REALTIME_FACTOR)


def test_a_chunks_deadline_is_capped_by_what_remains_of_the_file_budget() -> None:
    """A chunk's proportional share (`chunk.seconds * DEADLINE_REALTIME_FACTOR`)
    is 128 s for a 10 s chunk; a request with only 5 s left must cap it there,
    not hand out more than the file has."""
    remaining = DecodeBudget(seconds=5.0)
    budget = ChunkDecodeBudget.for_chunk(_chunk(10.0), remaining)
    assert budget.deadline_seconds == pytest.approx(5.0)


def test_the_sum_of_chunk_deadlines_never_exceeds_the_file_budget() -> None:
    """Threads a `DecodeBudget` across many chunks exactly the way
    `ReadyQwenRecognizer._run_model` does, and checks the invariant `for_chunk`
    exists to guarantee: no matter how the audio is sliced, the running
    total of every chunk's own deadline never exceeds what the request
    started with."""
    total_seconds = 700.0
    file_budget = DecodeBudget.for_total_seconds(total_seconds)

    chunk_lengths = [180.0, 180.0, 180.0, 160.0]
    assert sum(chunk_lengths) == pytest.approx(total_seconds)

    remaining = file_budget
    spent = 0.0
    for length in chunk_lengths:
        chunk_budget = ChunkDecodeBudget.for_chunk(_chunk(length), remaining)
        spent += chunk_budget.deadline_seconds
        remaining = remaining.remaining_after(chunk_budget.deadline_seconds)
        assert spent <= file_budget.seconds + 1e-9

    assert spent == pytest.approx(file_budget.seconds)


# --- detect_runaway over the typed outcome ----------------------------------


def test_a_token_cap_bound_is_always_a_runaway_with_that_reason() -> None:
    chunk = _chunk(180.0)
    outcome = Bounded(reason=BoundReason.TOKEN_CAP, text="x" * 10)
    runaway = detect_runaway(outcome, chunk)
    assert runaway is not None
    assert runaway.reason is RunawayReason.TOKEN_CAP
    # The partial text is never dropped, even when refused.
    assert runaway.text == "x" * 10


def test_a_deadline_bound_is_always_a_runaway_with_that_reason() -> None:
    chunk = _chunk(180.0)
    outcome = Bounded(reason=BoundReason.DEADLINE, text="partial transcript")
    runaway = detect_runaway(outcome, chunk)
    assert runaway is not None
    assert runaway.reason is RunawayReason.DEADLINE
    assert runaway.text == "partial transcript"


def test_a_deadline_bound_with_no_text_at_all_is_still_a_runaway() -> None:
    """The chunk this pins: a deadline that fires before the model has
    emitted anything. Its outcome's text is empty, exactly like an
    ordinary `Completed` silent chunk, so a caller that checked
    blank-text before consulting `detect_runaway` would silently treat a
    real deadline overrun as unremarkable silence."""
    chunk = _chunk(180.0)
    outcome = Bounded(reason=BoundReason.DEADLINE, text="")
    runaway = detect_runaway(outcome, chunk)
    assert runaway is not None
    assert runaway.reason is RunawayReason.DEADLINE


def test_ordinary_speech_density_on_a_completed_outcome_is_not_a_runaway() -> None:
    """The densest legitimate chunk measured across the whole Cantonese
    benchmark was 4.3 characters per second; the median was 2.9."""
    chunk = _chunk(180.0)
    assert detect_runaway(Completed(text="x" * 780), chunk) is None


def test_the_measured_runaway_is_caught_on_a_completed_outcome() -> None:
    """The real failure: 4,584 characters for 179.6 s of audio, 25.5 per
    second, on fixture A020 of the Cantonese benchmark. It completed on its
    own (never hit the token cap or the deadline), so density is the only
    signal that catches it."""
    chunk = _chunk(179.6, offset=360.0)
    runaway = detect_runaway(Completed(text="x" * 4584), chunk)

    assert runaway is not None
    assert runaway.reason is RunawayReason.DENSITY
    assert runaway.chars == 4584
    assert runaway.chars_per_second == pytest.approx(25.5, abs=0.2)
    assert runaway.offset == 360.0
    assert "4584 chars" in runaway.describe()


def test_an_empty_chunk_is_not_reported_as_a_runaway() -> None:
    """Zero-length audio would divide by zero; silence is not a runaway."""
    empty = AudioChunk(samples=np.array([], dtype=np.float32), offset=0.0)
    assert detect_runaway(Completed(text="anything"), empty) is None
