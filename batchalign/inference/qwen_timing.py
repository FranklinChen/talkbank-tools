"""Admit Qwen timing against audio, then assemble an ordered recording.

Paired model reports are not timing evidence. These constructors are the only
edges from reports to bounded spans, explicit absence, and recording placement.
No ML library is imported here.

A unit whose own span is unusable (non-finite, negative, reversed, or outside
its audio window) is admitted as ``RejectedWindowTiming``: explicit absence
that keeps the defect, so that unit is untimed and every other unit keeps its
timing. Units that run backwards against each other make the window's order
untrustworthy, a defect of neither unit alone, so every observed span in that
window is admitted as ``DisorderedWindowTiming``: untimed, keeping the pair.
Other windows keep their timing. No timing defect fails the file.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import TYPE_CHECKING, assert_never

from batchalign.inference.qwen_forced_alignment import (
    BackwardsUnitSpan,
    IndexedWindowUnit,
    InvalidUnitSpan,
    MissingUnitTime,
    OutsideAudioWindow,
    PairedWindowAlignment,
    RejectedAlignmentDiagnostic,
    ReportedUnit,
    SpanDefect,
    TimingBoundary,
    UntimedReportedUnit,
    WindowAlignedWord,
)
from batchalign.worker._model_errors import ModelOutputInvalidError

if TYPE_CHECKING:
    from collections.abc import Sized

    from batchalign.inference.qwen_punctuation import PunctuatedAlignment


_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RelativeAudioWindow:
    """Times relative to the aligned window, with no recording placement:
    the standalone aligner's coordinates."""


RELATIVE_AUDIO_WINDOW = RelativeAudioWindow()


@dataclass(frozen=True, slots=True)
class AudioWindow:
    """The actual duration of the samples handed to the aligner, in seconds."""

    duration_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.duration_s) or self.duration_s < 0:
            raise ValueError("an audio window needs a finite nonnegative duration")

    @classmethod
    def from_samples(cls, samples: Sized, sample_rate: int) -> AudioWindow:
        if sample_rate <= 0:
            raise ValueError("sample rate must be positive")
        return cls(len(samples) / sample_rate)


type SpanDefectVerdict = InvalidUnitSpan | OutsideAudioWindow
"""What `_judge_span` can find wrong with a reported span."""

type UnitSpanDefect = SpanDefectVerdict | MissingUnitTime
"""What is wrong with one unit's own report. That unit alone is untimed."""


@dataclass(frozen=True, slots=True)
class CollapsedSpan:
    """A positive window span that is not positive once placed at its
    recording offset in floating point (a long offset and a tiny span)."""

    unit: IndexedWindowUnit
    offset_s: float


type TimingDefect = UnitSpanDefect | BackwardsUnitSpan | CollapsedSpan
"""Every defect admission records. A backwards pair is a defect of the
window's order, not of either unit, so it untimes the window's spans."""


def describe_timing_defect(defect: TimingDefect) -> str:
    """The defect with the original reported values, for logs and errors."""
    match defect:
        case InvalidUnitSpan(unit=unit, defect=span_defect):
            return (
                f"unit {unit.index + 1} {unit.word.text!r} has {span_defect.value} "
                f"timing {unit.word.start_s!r}..{unit.word.end_s!r} s"
            )
        case BackwardsUnitSpan(previous=previous, current=current, boundary=boundary):
            return (
                f"unit {current.index + 1} {current.word.text!r} has a backwards "
                f"{boundary.value}: {current.word.start_s!r}..{current.word.end_s!r} s "
                f"after unit {previous.index + 1} {previous.word.text!r} "
                f"{previous.word.start_s!r}..{previous.word.end_s!r} s"
            )
        case OutsideAudioWindow(unit=unit, duration_s=duration):
            return (
                f"unit {unit.index + 1} {unit.word.text!r} reports "
                f"{unit.word.start_s!r}..{unit.word.end_s!r} s outside its "
                f"0..{duration:g} s audio window"
            )
        case MissingUnitTime(index=index, reported=reported):
            return (
                f"unit {index + 1} {reported.text!r} has no usable "
                f"{reported.field.value}: decoded {reported.decoded}"
            )
        case CollapsedSpan(unit=unit, offset_s=offset):
            return (
                f"unit {unit.index + 1} {unit.word.text!r} relative "
                f"{unit.word.start_s!r}..{unit.word.end_s!r} s is not positive at "
                f"offset {offset!r} s"
            )
        case unreachable:
            assert_never(unreachable)


class QwenTimingInvalid(ModelOutputInvalidError):
    """A span constructor was handed a span admission did not approve.

    Admission (`_admit_timing`) never raises this: it turns every defect into
    explicit untimed evidence. This guards the constructors that claim a usable
    span, so a direct caller cannot build one from a bad span.
    """

    def __init__(self, problem: SpanDefectVerdict) -> None:
        self.problem = problem
        super().__init__(
            "Qwen timing constructor given a span admission rejects "
            f"(window-relative): {describe_timing_defect(problem)}"
        )


@dataclass(frozen=True, slots=True)
class SpanInWindow:
    """The unit's own span is finite, ordered and inside its window."""


type SpanVerdict = SpanInWindow | SpanDefectVerdict


def _judge_span(unit: IndexedWindowUnit, window: AudioWindow) -> SpanVerdict:
    """The one rule for a unit's own span: in its window, or which defect."""
    word = unit.word
    if not math.isfinite(word.start_s) or not math.isfinite(word.end_s):
        return InvalidUnitSpan(unit, SpanDefect.NON_FINITE)
    if word.start_s < 0 or word.end_s < 0:
        return InvalidUnitSpan(unit, SpanDefect.NEGATIVE)
    if word.end_s < word.start_s:
        return InvalidUnitSpan(unit, SpanDefect.REVERSED)
    if word.end_s > window.duration_s:
        return OutsideAudioWindow(unit, window.duration_s)
    return SpanInWindow()


def _admit_span(unit: IndexedWindowUnit, window: AudioWindow) -> None:
    """Guard for the timing constructors that claim a usable span."""
    match _judge_span(unit, window):
        case SpanInWindow():
            pass
        case InvalidUnitSpan() | OutsideAudioWindow() as defect:
            raise QwenTimingInvalid(defect)
        case unreachable:
            assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class ObservedWindowSpan:
    """A positive observed interval contained in its own audio window."""

    unit: IndexedWindowUnit
    window: AudioWindow

    def __post_init__(self) -> None:
        _admit_span(self.unit, self.window)
        if self.start_s == self.end_s:
            raise ValueError("zero-width model timing must become unavailable evidence")

    @property
    def start_s(self) -> float:
        return self.unit.word.start_s

    @property
    def end_s(self) -> float:
        return self.unit.word.end_s


class UntimedWindowTiming:
    """Explicit absence: the unit keeps its position with no usable span.

    Consumers match `ObservedWindowSpan` and this base; only the log reads
    which subclass (and so which reason) a unit has.
    """

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class UnavailableWindowTiming(UntimedWindowTiming):
    """A zero-width report retains its position without claiming an interval."""

    unit: IndexedWindowUnit
    window: AudioWindow

    def __post_init__(self) -> None:
        _admit_span(self.unit, self.window)
        if self.unit.word.start_s != self.unit.word.end_s:
            raise ValueError("unavailable timing requires a zero-width report")


@dataclass(frozen=True, slots=True)
class RejectedWindowTiming(UntimedWindowTiming):
    """A unit whose reported span is unusable: untimed, keeping the defect."""

    defect: UnitSpanDefect


@dataclass(frozen=True, slots=True)
class DisorderedWindowTiming(UntimedWindowTiming):
    """A unit whose own span was usable, untimed because its window's reported
    order runs backwards. Every such unit of a window keeps the same pair."""

    unit: IndexedWindowUnit
    defect: BackwardsUnitSpan


type WindowTiming = ObservedWindowSpan | UntimedWindowTiming


def _admit_timing(unit: IndexedWindowUnit, window: AudioWindow) -> WindowTiming:
    """The one edge from a reported unit to its window timing."""
    match _judge_span(unit, window):
        case SpanInWindow():
            if unit.word.start_s == unit.word.end_s:
                return UnavailableWindowTiming(unit, window)
            return ObservedWindowSpan(unit, window)
        case InvalidUnitSpan() | OutsideAudioWindow() as defect:
            return RejectedWindowTiming(defect)
        case unreachable:
            assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class AlignedWindowWord:
    text: str
    timing: WindowTiming


@dataclass(frozen=True, slots=True, init=False)
class WindowAlignment:
    """Immutable bounded evidence, admitted from one paired model response."""

    units: tuple[str, ...]
    words: tuple[AlignedWindowWord, ...]
    window: AudioWindow

    def __init__(
        self,
        reported: PairedWindowAlignment,
        window: AudioWindow,
        location: TimingLocation = RELATIVE_AUDIO_WINDOW,
    ) -> None:
        """Admit every unit, then the window's order. Never raises on model
        output: each defect becomes explicit untimed evidence. ``location``
        only says which coordinates the log names; the times stay relative."""
        admitted = [
            _admit_reported(index, word, window)
            for index, word in enumerate(reported.words)
        ]
        disorder = _first_disorder(admitted)
        if disorder is not None:
            admitted = [
                AlignedWindowWord(
                    word.text, DisorderedWindowTiming(word.timing.unit, disorder)
                )
                if isinstance(word.timing, ObservedWindowSpan)
                else word
                for word in admitted
            ]
        _log_untimed_defects(admitted, disorder, reported, window, location)
        object.__setattr__(self, "units", reported.units)
        object.__setattr__(self, "words", tuple(admitted))
        object.__setattr__(self, "window", window)


def _admit_reported(
    index: int, reported: ReportedUnit, window: AudioWindow
) -> AlignedWindowWord:
    """One decoded unit, at its position: its span admitted, or untimed with
    the defect when the decoder gave it no usable time."""
    match reported:
        case WindowAlignedWord():
            timing = _admit_timing(IndexedWindowUnit(index, reported), window)
            return AlignedWindowWord(reported.text, timing)
        case UntimedReportedUnit():
            missing = RejectedWindowTiming(MissingUnitTime(index, reported))
            return AlignedWindowWord(reported.text, missing)
        case unreachable:
            assert_never(unreachable)


def _first_disorder(words: list[AlignedWindowWord]) -> BackwardsUnitSpan | None:
    """The first pair of spoken, observed units whose start or end runs
    backwards. Untimed and blank units carry no order to check."""
    spoken = [
        word.timing
        for word in words
        if isinstance(word.timing, ObservedWindowSpan) and word.text.strip()
    ]
    for previous, current in itertools.pairwise(spoken):
        if current.start_s < previous.start_s:
            return BackwardsUnitSpan(previous.unit, current.unit, TimingBoundary.START)
        if current.end_s < previous.end_s:
            return BackwardsUnitSpan(previous.unit, current.unit, TimingBoundary.END)
    return None


def _log_untimed_defects(
    words: list[AlignedWindowWord],
    disorder: BackwardsUnitSpan | None,
    reported: PairedWindowAlignment,
    window: AudioWindow,
    location: TimingLocation,
) -> None:
    """One warning per window that untimed units for a defect, with the
    unchanged report: evidence about the model, whatever the log level."""
    defects: list[TimingDefect] = [
        word.timing.defect
        for word in words
        if isinstance(word.timing, RejectedWindowTiming)
    ]
    if disorder is not None:
        defects.append(disorder)
    if not defects:
        return
    untimed = sum(
        isinstance(word.timing, RejectedWindowTiming | DisorderedWindowTiming)
        for word in words
    )
    _LOG.warning(
        "Qwen aligner timing in %s is unusable for %d of %d units; they are "
        "untimed and the rest of the file keeps its timing. Defects: %s. "
        "Qwen rejected alignment diagnostic: %s",
        describe_timing_location(location),
        untimed,
        len(words),
        "; ".join(describe_timing_defect(defect) for defect in defects),
        json.dumps(
            asdict(RejectedAlignmentDiagnostic(reported, window.duration_s)),
            ensure_ascii=False,
        ),
    )


@dataclass(frozen=True, slots=True, init=False)
class SpokenWindowWords:
    """ASR's nonblank view; unavailable units remain explicitly unavailable."""

    words: tuple[AlignedWindowWord, ...]
    window: AudioWindow

    def __init__(self, alignment: WindowAlignment) -> None:
        object.__setattr__(
            self, "words", tuple(word for word in alignment.words if word.text.strip())
        )
        object.__setattr__(self, "window", alignment.window)


@dataclass(frozen=True, slots=True)
class UntimedRecognition:
    """A window whose recognized text stands without word timing: the
    aligner's units do not spell that text, so none of their times can be
    attributed to its words. Nothing is invented and the text is not changed."""

    window: AudioWindow

    @property
    def words(self) -> tuple[AlignedWindowWord, ...]:
        return ()


@dataclass(frozen=True, slots=True)
class RecordingWindow:
    """Where one audio window sits in the recording."""

    start_s: float
    audio: AudioWindow

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.start_s)
            or self.start_s < 0
            or not math.isfinite(self.end_s)
        ):
            raise ValueError("a recording window needs finite nonnegative coordinates")

    @property
    def end_s(self) -> float:
        return self.start_s + self.audio.duration_s

    def describe(self) -> str:
        return f"recording window {self.start_s:g}..{self.end_s:g} s"


type TimingLocation = RelativeAudioWindow | RecordingWindow
"""Which coordinates a logged defect's reported times are in."""


def describe_timing_location(location: TimingLocation) -> str:
    match location:
        case RelativeAudioWindow():
            return "window-relative coordinates"
        case RecordingWindow():
            return f"{location.describe()} (reported times are relative to that window)"
        case unreachable:
            assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class ObservedRecordingSpan:
    source: ObservedWindowSpan
    placement: RecordingWindow

    def __post_init__(self) -> None:
        if self.source.window != self.placement.audio:
            raise ValueError("recording placement belongs to a different audio window")
        if not _stays_positive(self.source, self.placement):
            raise ValueError(
                "a recording span must stay positive; `_place` turns a collapse "
                "into CollapsedPlacement"
            )

    @property
    def start_s(self) -> float:
        return self.placement.start_s + self.source.start_s

    @property
    def end_s(self) -> float:
        return self.placement.start_s + self.source.end_s


@dataclass(frozen=True, slots=True)
class CollapsedPlacement(UntimedWindowTiming):
    """A positive window span that placement cannot keep positive: untimed,
    keeping the defect, never invented."""

    defect: CollapsedSpan


def _stays_positive(span: ObservedWindowSpan, window: RecordingWindow) -> bool:
    """The one placement rule, shared by `_place` and the span's guard."""
    return window.start_s + span.end_s > window.start_s + span.start_s


def _place(
    span: ObservedWindowSpan, window: RecordingWindow
) -> ObservedRecordingSpan | CollapsedPlacement:
    """The one edge from window time to recording time."""
    if _stays_positive(span, window):
        return ObservedRecordingSpan(span, window)
    return CollapsedPlacement(CollapsedSpan(span.unit, window.start_s))


@dataclass(frozen=True, slots=True)
class RecordingWord:
    text: str
    timing: ObservedRecordingSpan | UntimedWindowTiming


@dataclass(frozen=True, slots=True, init=False)
class RecordingChunkSpeech:
    """A bounded speech view placed once in recording coordinates."""

    text: str
    window: RecordingWindow
    words: tuple[RecordingWord, ...]

    def __init__(
        self,
        text: str,
        offset_s: float,
        speech: SpokenWindowWords | PunctuatedAlignment | UntimedRecognition,
    ) -> None:
        window = RecordingWindow(offset_s, speech.window)
        words: list[RecordingWord] = []
        for word in speech.words:
            match word.timing:
                case ObservedWindowSpan() as span:
                    words.append(RecordingWord(word.text, _place(span, window)))
                case UntimedWindowTiming() as absent:
                    words.append(RecordingWord(word.text, absent))
                case unreachable:
                    assert_never(unreachable)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "window", window)
        object.__setattr__(self, "words", tuple(words))


def log_placement_defects(speech: RecordingChunkSpeech) -> None:
    """One warning per chunk whose placement untimed spans, naming each."""
    collapsed = [
        word.timing.defect
        for word in speech.words
        if isinstance(word.timing, CollapsedPlacement)
    ]
    if collapsed:
        _LOG.warning(
            "Qwen timing placed in %s lost %d positive span(s) to floating-point "
            "resolution; they are untimed: %s",
            speech.window.describe(),
            len(collapsed),
            "; ".join(describe_timing_defect(defect) for defect in collapsed),
        )


class QwenRecordingOrderInvalid(ModelOutputInvalidError):
    """Chunk placement attempted to rewind audio already assembled."""

    def __init__(self, previous: RecordingWindow, current: RecordingWindow) -> None:
        self.previous = previous
        self.current = current
        super().__init__(
            f"Qwen recording assembly rejected window {current.start_s:g}..{current.end_s:g} s "
            f"after {previous.start_s:g}..{previous.end_s:g} s: the audio windows overlap or go backwards. "
            "Keep the chunk/window diagnostic and audio; this is generated alignment/placement, not a CHAT input error."
        )


@dataclass(frozen=True, slots=True)
class RecordingStart:
    """No audio has been published to the recording sequence yet."""

    def append(self, chunk: RecordingChunkSpeech) -> RecordingPrefix:
        return RecordingPrefix(self, chunk)


@dataclass(frozen=True, slots=True)
class RecordingPrefix:
    """An immutable ordered prefix; extending consumes its placement bound."""

    previous: RecordingStart | RecordingPrefix
    chunk: RecordingChunkSpeech

    def __post_init__(self) -> None:
        match self.previous:
            case RecordingStart():
                pass
            case RecordingPrefix(chunk=previous):
                if self.chunk.window.start_s < previous.window.end_s:
                    raise QwenRecordingOrderInvalid(previous.window, self.chunk.window)
            case unreachable:
                assert_never(unreachable)

    def append(self, chunk: RecordingChunkSpeech) -> RecordingPrefix:
        return RecordingPrefix(self, chunk)


@dataclass(frozen=True, slots=True, init=False)
class CompletedRecording:
    """Only an ordered prefix can publish immutable recording chunks."""

    chunks: tuple[RecordingChunkSpeech, ...]

    def __init__(self, prefix: RecordingStart | RecordingPrefix) -> None:
        chunks: list[RecordingChunkSpeech] = []
        while True:
            match prefix:
                case RecordingStart():
                    break
                case RecordingPrefix():
                    chunks.append(prefix.chunk)
                    prefix = prefix.previous
                case unreachable:
                    assert_never(unreachable)
        object.__setattr__(self, "chunks", tuple(reversed(chunks)))


class QwenFaAdmissionRevision(Enum):
    BOUNDED_UNIT_EVIDENCE_V1 = "bounded-unit-evidence-v1"
    UNIT_DEFECTS_UNTIMED_V2 = "unit-defects-untimed-v2"
    """A unit with an unusable span is untimed instead of failing the file."""
    TIMING_DEFECTS_UNTIMED_V3 = "timing-defects-untimed-v3"
    """No timing defect fails the file: a window whose units run backwards
    against each other is untimed, and so is a unit the decoder gave no usable
    time."""


@dataclass(frozen=True, slots=True)
class QwenFaCacheIdentity:
    """Model selection and admission semantics jointly authorize replay."""

    model_id: str
    admission: QwenFaAdmissionRevision

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("a Qwen FA cache identity needs the loaded model ID")

    @property
    def wire_name(self) -> str:
        return f"{self.model_id}@{self.admission.value}"
