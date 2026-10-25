"""Long-audio chunking and runaway detection for the native Qwen3-ASR engine.

Two jobs that the ``qwen-asr`` sidecar used to do for us, and that we own now
that BA3 drives ``transformers``' ``qwen3_asr`` module directly.

**Chunking.** The native path has no long-audio handling of its own: a whole
session fed to it does not merely score worse, it does not complete. This is a
port of the sidecar's ``split_audio_into_chunks``, which cuts at the quietest
window near each target boundary so a chunk edge lands in silence rather than
mid-word. Keeping the same algorithm is deliberate: the migration was decided
on a CER differential measured under this exact chunking, so changing it here
would invalidate the evidence that justified the change.

**Coordinate spaces.** The forced aligner returns times relative to THE AUDIO
IT WAS GIVEN, which under chunking is a 180-second window, not the recording.
Turning one into the other is a single addition that is invisible when omitted
and produces timings that look entirely reasonable until they land past the end
of the file. That has happened here before: FA emitted word timings 28.2
seconds past the end of their own audio on 6 of 226 sessions and went unnoticed
for two months. So the two spaces are named apart, and ``AudioChunk.to_file``
is the only conversion between them.

**Runaway detection.** Greedy decoding on this model can fall into a repetition
loop and emit until it hits the token cap. Measured on the 18-fixture Cantonese
benchmark: one chunk produced 4,584 characters for 179.6 seconds of audio while
every other chunk in the corpus stayed under 4.4 characters per second. A
silent hallucination of that size is worse than a crash, because nothing
downstream can tell it from speech.

**Decode budgeting.** A flat ``max_new_tokens`` cap sized for the longest
possible chunk, with no wall-clock stop, means a short chunk is decoded
against a budget it never needed and a repetition loop pays the FULL cost of
that cap before anything downstream looks at what it produced. Measured on a
fleet host (CPU, float32): a 411 s file ran past its 1800 s job ceiling
twice, driven by chunks decoded exactly this way. ``ChunkDecodeBudget`` is
built only from the ``AudioChunk`` it bounds, so a 10 s chunk and a 180 s
chunk are never decoded against the same numbers. Its outcome is typed too:
``ChunkDecodeOutcome`` is ``Bounded`` (the token cap or the deadline cut the
decode off before it stopped on its own) or ``Completed`` (it stopped on its
own, within budget), and ``detect_runaway`` reads that distinction first,
falling back to character density only for a ``Completed`` outcome.

**One owner of the file's time budget.** A per-chunk deadline computed
independently from ``chunk.seconds`` alone has no relationship to any ceiling
on the FILE as a whole; a file split into many chunks could accumulate an
unbounded total wall-clock spend even though each individual chunk looked
disciplined. ``DecodeBudget`` is built exactly ONCE per request, from the
file's total audio seconds and the same realtime factor
``ChunkDecodeBudget`` uses, and every chunk's deadline is drawn down from it
via ``ChunkDecodeBudget.for_chunk(chunk, remaining)``: a chunk's deadline is
the smaller of its own proportional share and whatever is left after the
chunks before it, so the SUM of every chunk's deadline can never exceed the
budget it was drawn from.

**Transport ceiling.** ``DecodeBudget`` bounds decode time only. The outer
transport/job-level timeout that carries a request end-to-end (audio load,
every chunk's decode and alignment, IPC) must be configured to at least this
budget PLUS a margin for that surrounding work, or the two ceilings can
disagree exactly the way the flat 1800 s job ceiling and the flat per-chunk
token cap disagreed in the reported defect. The Rust side of the job
pipeline is being changed to derive its own timeout from the SAME audio
duration this module uses, via a ``decode_budget_seconds`` field the request
will carry, so the two ceilings are computed from one duration instead of
drifting apart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import assert_never

import numpy as np

# Qwen3-ASR's audio front end expects 16 kHz.
SAMPLE_RATE = 16000

# The model has a minimum input length; a short tail is zero-padded up to it.
MIN_ASR_INPUT_SECONDS = 0.5

# The sidecar picks between a 1200 s cap for plain ASR and a 180 s cap when
# timestamps are requested. We always want timestamps, so 180 is the one that
# governs. NOTE: the effective ceiling is this plus SEARCH_EXPAND_SECONDS,
# because the boundary search may move a cut later as well as earlier.
MAX_FORCE_ALIGN_INPUT_SECONDS = 180.0
SEARCH_EXPAND_SECONDS = 5.0
MIN_ENERGY_WINDOW_MS = 100.0

# Characters of transcript per second of audio, above which output is treated
# as a decoding runaway rather than speech. Measured, not guessed, over 119
# chunks of the Cantonese benchmark: the median is 2.9, the densest legitimate
# chunk is 4.3, and the one degenerate chunk is 25.5. Eight sits nearly twice
# above anything real we have seen and three times below the failure, so there
# is no plausible transcript in the gap.
MAX_CHARS_PER_SECOND = 8.0

# Caps generation for ONE chunk at the chunker's own effective ceiling (a
# chunk can run up to `MAX_FORCE_ALIGN_INPUT_SECONDS + SEARCH_EXPAND_SECONDS`,
# see the boundary-search test below). Generous for that much speech: the
# densest legitimate chunk measured across the Cantonese benchmark was 4.3
# characters per second. Reaching this cap is the signature of a decoding
# runaway rather than of dense speech, which is why `ChunkDecodeOutcome`
# exists rather than this bound being trusted to contain the damage on its
# own: hitting it still yields thousands of characters of repetition, and
# nothing before this change stopped the decode any earlier.
MAX_NEW_TOKENS_CEILING = 4096

# The chunk length `MAX_NEW_TOKENS_CEILING` was calibrated for: the chunker's
# own effective maximum, not the named `MAX_FORCE_ALIGN_INPUT_SECONDS` alone,
# because the boundary search can move a cut later as well as earlier.
_REFERENCE_CHUNK_SECONDS = MAX_FORCE_ALIGN_INPUT_SECONDS + SEARCH_EXPAND_SECONDS

# Tokens per second of audio a chunk may be budgeted, derived from the two
# constants above rather than a fresh characters-per-token estimate: it
# reuses the same generosity `MAX_NEW_TOKENS_CEILING` already claims for the
# reference-length chunk, decomposed into a per-second rate so a shorter
# chunk gets a proportionally shorter budget instead of the same flat cap
# regardless of how little audio it holds.
_TOKENS_PER_SECOND_CEILING = MAX_NEW_TOKENS_CEILING / _REFERENCE_CHUNK_SECONDS

# Measured wall-clock seconds of decode per second of audio, for a normal
# (non-runaway) chunk, on the fleet host where the 1800 s job-ceiling defect
# was measured (CPU, float32).
_MEASURED_REALTIME_FACTOR = 6.4

# Chosen, not measured: headroom over the measured baseline so a chunk that
# happens to decode somewhat slower than that one host (a different CPU,
# contention from another job) is not itself misclassified as a runaway.
# Doubling the observed factor is generous slack while still bounding any
# chunk that is many multiples slower than a legitimate decode has ever been
# measured to be; `detect_runaway` still applies the character-density check
# to whatever text a `Completed` decode returns; a `Bounded` one is reported,
# with its partial text intact, never silently discarded.
_REALTIME_FACTOR_SAFETY_MARGIN = 2.0

# Wall-clock seconds of decode budget per second of audio.
DEADLINE_REALTIME_FACTOR = _MEASURED_REALTIME_FACTOR * _REALTIME_FACTOR_SAFETY_MARGIN


@dataclass(frozen=True)
class AudioChunk:
    """One slice of audio, and where it starts in the recording.

    The offset is carried WITH the samples rather than alongside them, because
    the pairing of a chunk with its own offset is the thing that must never be
    got wrong, and a pair that travels together cannot be mismatched.
    """

    samples: np.ndarray
    offset: float

    @property
    def seconds(self) -> float:
        return len(self.samples) / float(SAMPLE_RATE)


@dataclass(frozen=True)
class DecodeBudget:
    """The wall-clock decode budget for an entire request, in seconds.

    ONE owner. Every chunk of a request draws its own deadline down from a
    single ``DecodeBudget`` (see ``ChunkDecodeBudget.for_chunk``) rather than
    each chunk computing an independent deadline from its own length alone,
    which is how the sum of per-chunk deadlines used to have no relationship
    to any ceiling on the file as a whole.

    Construct via ``for_total_seconds`` until the request itself carries a
    ``decode_budget_seconds`` field (a value the Rust side will derive from
    the same audio duration and hand down); once that lands, callers build a
    ``DecodeBudget`` directly from that field instead of re-deriving one from
    ``for_total_seconds``. Both are legitimate construction paths for the
    same fact, computed from the same duration; neither is a bypass of the
    other.
    """

    seconds: float

    @classmethod
    def for_total_seconds(cls, total_seconds: float) -> DecodeBudget:
        """Derive a request's budget from its total audio, absent a
        request-supplied one. Uses the one named realtime factor every
        chunk's own share is also measured against."""
        return cls(seconds=total_seconds * DEADLINE_REALTIME_FACTOR)

    def remaining_after(self, spent_seconds: float) -> DecodeBudget:
        """The budget left after ``spent_seconds`` has been drawn down.

        Never negative: a chunk that was allocated the last of the budget
        still leaves a (zero, not unrepresentable-negative) budget for
        whatever chunk follows it.
        """
        return DecodeBudget(seconds=max(0.0, self.seconds - spent_seconds))


@dataclass(frozen=True)
class ChunkDecodeBudget:
    """The decode limits for one chunk, built only from the chunk and what
    remains of the request's file-level ``DecodeBudget``.

    Construct only via ``for_chunk``. A budget assembled elsewhere from raw
    numbers could disagree with the chunk it claims to bound, which is
    exactly the pairing-by-convention shape this type exists to close off:
    the flat cap it replaces was one number reused for every chunk
    regardless of length, with no file-level ceiling on the sum at all.
    """

    max_new_tokens: int
    deadline_seconds: float

    @classmethod
    def for_chunk(cls, chunk: AudioChunk, remaining: DecodeBudget) -> ChunkDecodeBudget:
        """Scale the token cap from the chunk's own length; cap the deadline
        at whatever the file-level budget has left.

        ``max_new_tokens`` never exceeds ``MAX_NEW_TOKENS_CEILING`` (the
        value the flat cap used unconditionally) and shrinks proportionally
        for a chunk shorter than the reference length that ceiling was
        calibrated for.

        ``deadline_seconds`` is the smaller of the chunk's own proportional
        share (``chunk.seconds * DEADLINE_REALTIME_FACTOR``, what it would
        get if it were the only chunk left) and ``remaining.seconds`` (what
        every chunk before it has left of the file's own budget). Callers
        thread the SAME ``DecodeBudget`` through the whole request, drawing
        it down by each returned ``deadline_seconds`` after every chunk, so
        the running sum can never exceed what the request started with.
        """
        tokens = math.ceil(chunk.seconds * _TOKENS_PER_SECOND_CEILING)
        max_new_tokens = min(MAX_NEW_TOKENS_CEILING, max(tokens, 1))
        proportional_share = chunk.seconds * DEADLINE_REALTIME_FACTOR
        deadline_seconds = min(proportional_share, remaining.seconds)
        return cls(max_new_tokens=max_new_tokens, deadline_seconds=deadline_seconds)


class BoundReason(Enum):
    """Which limit cut a decode off before it stopped on its own."""

    TOKEN_CAP = "token_cap"
    DEADLINE = "deadline"


@dataclass(frozen=True)
class Bounded:
    """A chunk decode that was cut off by the token cap or the deadline.

    Carries the text produced up to that point: a cut-off decode is refused
    downstream (see ``detect_runaway``), but the partial text is never
    dropped, only classified.
    """

    reason: BoundReason
    text: str


@dataclass(frozen=True)
class Completed:
    """A chunk decode that stopped on its own, within budget."""

    text: str


# The typed result of decoding one chunk. A caller that only reads `.text`
# gets identical behaviour for both members; a caller that needs to know
# WHY (`detect_runaway`, logging) matches on the type.
type ChunkDecodeOutcome = Bounded | Completed


class RunawayReason(Enum):
    """Why a chunk's transcript was refused as unusable."""

    TOKEN_CAP = "token cap"
    DEADLINE = "decode deadline"
    DENSITY = "character density"


@dataclass(frozen=True)
class RunawayOutput:
    """A chunk whose transcript was refused as unlikely to be real speech.

    Returned rather than merely logged. The caller has to decide what reaches
    the transcript, and a refusal that exists only in a log line is invisible
    to the artifact, unqueryable afterwards, and tested by nothing.
    """

    offset: float
    audio_seconds: float
    chars: int
    limit: float
    reason: RunawayReason
    text: str

    @property
    def chars_per_second(self) -> float:
        """Derived, not stored, so it cannot disagree with its own inputs."""
        return self.chars / self.audio_seconds

    def describe(self) -> str:
        return (
            f"{self.chars} chars for {self.audio_seconds:.1f}s of audio at "
            f"{self.offset:.1f}s ({self.chars_per_second:.1f} chars/s, "
            f"limit {self.limit:.1f}, reason: {self.reason.value})"
        )


def _runaway_reason_for(bound: BoundReason) -> RunawayReason:
    """The one place a ``BoundReason`` becomes a ``RunawayReason``.

    An exhaustive ``match`` rather than a lookup table: a future
    ``BoundReason`` member with no arm here fails mypy at ``assert_never``
    instead of a silent ``KeyError`` (a dict) or a silently-wrong default
    (an ``if``/``else`` chain).
    """
    match bound:
        case BoundReason.TOKEN_CAP:
            return RunawayReason.TOKEN_CAP
        case BoundReason.DEADLINE:
            return RunawayReason.DEADLINE
        case _:
            assert_never(bound)


def detect_runaway(
    outcome: ChunkDecodeOutcome,
    chunk: AudioChunk,
    limit: float = MAX_CHARS_PER_SECOND,
) -> RunawayOutput | None:
    """Classify one chunk's decode outcome as speech or runaway.

    A ``Bounded`` outcome is a runaway by construction: the model was cut off
    by the token cap or the wall-clock deadline before it could stop on its
    own, and legitimate speech at the calibrated density limit never reaches
    either bound (see ``ChunkDecodeBudget`` and the module docstring for the
    measurements). A ``Completed`` outcome still gets the character-density
    check that used to be the only signal, because a model that stopped on
    its own can still have emitted more text than the audio could plausibly
    contain.

    Returns ``None`` for ordinary output, so the caller branches on a value
    that names the problem rather than on a bare boolean.
    """
    seconds = chunk.seconds
    if seconds <= 0:
        return None

    if isinstance(outcome, Bounded):
        return RunawayOutput(
            offset=chunk.offset,
            audio_seconds=seconds,
            chars=len(outcome.text),
            limit=limit,
            reason=_runaway_reason_for(outcome.reason),
            text=outcome.text,
        )

    density = len(outcome.text) / seconds
    if density <= limit:
        return None
    return RunawayOutput(
        offset=chunk.offset,
        audio_seconds=seconds,
        chars=len(outcome.text),
        limit=limit,
        reason=RunawayReason.DENSITY,
        text=outcome.text,
    )


def split_audio_into_chunks(
    wav: np.ndarray,
    sr: int = SAMPLE_RATE,
    max_chunk_sec: float = MAX_FORCE_ALIGN_INPUT_SECONDS,
    search_expand_sec: float = SEARCH_EXPAND_SECONDS,
    min_window_ms: float = MIN_ENERGY_WINDOW_MS,
) -> list[AudioChunk]:
    """Split audio at low-energy boundaries, as the sidecar did.

    Concatenating the chunks reproduces the input exactly: no overlaps, no
    gaps, so no audio is transcribed twice or dropped. The only exception is a
    final chunk shorter than the model's minimum input, which is zero-padded.
    """
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1).astype(np.float32)

    total_len = int(wav.shape[0])
    if total_len / float(sr) <= max_chunk_sec:
        return [AudioChunk(samples=wav, offset=0.0)]

    max_len = int(max_chunk_sec * sr)
    expand = int(search_expand_sec * sr)
    win = max(4, int((min_window_ms / 1000.0) * sr))

    chunks: list[AudioChunk] = []
    start = 0
    offset_sec: float = 0.0

    while (total_len - start) > max_len:
        cut = start + max_len
        left = max(start, cut - expand)
        right = min(total_len, cut + expand)

        if right - left <= win:
            boundary = cut
        else:
            # Quietest window in the search band, then the quietest single
            # sample inside it: cut where the speech is not.
            seg_abs = np.abs(wav[left:right])
            window_sums = np.convolve(
                seg_abs, np.ones(win, dtype=np.float32), mode="valid"
            )
            min_pos = int(np.argmin(window_sums))
            inner = int(np.argmin(seg_abs[min_pos : min_pos + win]))
            boundary = left + min_pos + inner

        boundary = int(min(max(boundary, start + 1), total_len))
        chunks.append(AudioChunk(samples=wav[start:boundary], offset=offset_sec))
        offset_sec += (boundary - start) / float(sr)
        start = boundary

    chunks.append(AudioChunk(samples=wav[start:total_len], offset=offset_sec))

    min_len = int(MIN_ASR_INPUT_SECONDS * sr)
    return [
        AudioChunk(
            samples=(
                np.pad(
                    c.samples, (0, min_len - c.samples.shape[0]), mode="constant"
                ).astype(np.float32)
                if c.samples.shape[0] < min_len
                else c.samples
            ),
            offset=c.offset,
        )
        for c in chunks
    ]
