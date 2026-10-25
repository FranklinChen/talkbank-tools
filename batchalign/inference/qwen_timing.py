"""Admit Qwen timing against audio, then assemble an ordered recording.

Paired model reports are not timing evidence. These constructors are the only
edges from reports to bounded spans, explicit absence, and recording placement.
No ML library is imported here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, assert_never

from batchalign.inference.qwen_forced_alignment import (
    BackwardsUnitSpan,
    IndexedWindowUnit,
    InvalidUnitSpan,
    OutsideAudioWindow,
    PairedWindowAlignment,
    QwenTimingInvalid,
    SpanDefect,
    TimingBoundary,
)
from batchalign.worker._model_errors import ModelOutputInvalidError

if TYPE_CHECKING:
    from collections.abc import Sized

    from batchalign.inference.qwen_punctuation import PunctuatedAlignment


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


def _admit_span(unit: IndexedWindowUnit, window: AudioWindow) -> None:
    word = unit.word
    if not math.isfinite(word.start_s) or not math.isfinite(word.end_s):
        raise QwenTimingInvalid(InvalidUnitSpan(unit, SpanDefect.NON_FINITE))
    if word.start_s < 0 or word.end_s < 0:
        raise QwenTimingInvalid(InvalidUnitSpan(unit, SpanDefect.NEGATIVE))
    if word.end_s < word.start_s:
        raise QwenTimingInvalid(InvalidUnitSpan(unit, SpanDefect.REVERSED))
    if word.end_s > window.duration_s:
        raise QwenTimingInvalid(OutsideAudioWindow(unit, window.duration_s))


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


@dataclass(frozen=True, slots=True)
class UnavailableWindowTiming:
    """A zero-width report retains its position without claiming an interval."""

    unit: IndexedWindowUnit
    window: AudioWindow

    def __post_init__(self) -> None:
        _admit_span(self.unit, self.window)
        if self.unit.word.start_s != self.unit.word.end_s:
            raise ValueError("unavailable timing requires a zero-width report")


type WindowTiming = ObservedWindowSpan | UnavailableWindowTiming


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

    def __init__(self, reported: PairedWindowAlignment, window: AudioWindow) -> None:
        admitted: list[AlignedWindowWord] = []
        previous: ObservedWindowSpan | None = None
        for index, word in enumerate(reported.words):
            unit = IndexedWindowUnit(index, word)
            timing: WindowTiming
            if word.start_s == word.end_s:
                timing = UnavailableWindowTiming(unit, window)
            else:
                timing = ObservedWindowSpan(unit, window)
                if word.text.strip():
                    if previous is not None:
                        if timing.start_s < previous.start_s:
                            raise QwenTimingInvalid(
                                BackwardsUnitSpan(
                                    previous.unit, unit, TimingBoundary.START
                                )
                            )
                        if timing.end_s < previous.end_s:
                            raise QwenTimingInvalid(
                                BackwardsUnitSpan(
                                    previous.unit, unit, TimingBoundary.END
                                )
                            )
                    previous = timing
            admitted.append(AlignedWindowWord(word.text, timing))
        object.__setattr__(self, "units", reported.units)
        object.__setattr__(self, "words", tuple(admitted))
        object.__setattr__(self, "window", window)


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
class RecordingWindow:
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


@dataclass(frozen=True, slots=True)
class ObservedRecordingSpan:
    source: ObservedWindowSpan
    placement: RecordingWindow

    def __post_init__(self) -> None:
        if self.source.window != self.placement.audio:
            raise ValueError("recording placement belongs to a different audio window")
        if self.end_s <= self.start_s:
            raise ModelOutputInvalidError(
                f"Qwen recording placement lost the positive span of unit "
                f"{self.source.unit.index + 1} {self.source.unit.word.text!r}: "
                f"relative {self.source.start_s!r}..{self.source.end_s!r} s, "
                f"offset {self.placement.start_s!r} s"
            )

    @property
    def start_s(self) -> float:
        return self.placement.start_s + self.source.start_s

    @property
    def end_s(self) -> float:
        return self.placement.start_s + self.source.end_s


@dataclass(frozen=True, slots=True)
class RecordingWord:
    text: str
    timing: ObservedRecordingSpan | UnavailableWindowTiming


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
        speech: SpokenWindowWords | PunctuatedAlignment,
    ) -> None:
        window = RecordingWindow(offset_s, speech.window)
        words: list[RecordingWord] = []
        for word in speech.words:
            match word.timing:
                case ObservedWindowSpan() as span:
                    words.append(
                        RecordingWord(word.text, ObservedRecordingSpan(span, window))
                    )
                case UnavailableWindowTiming() as unavailable:
                    words.append(RecordingWord(word.text, unavailable))
                case unreachable:
                    assert_never(unreachable)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "window", window)
        object.__setattr__(self, "words", tuple(words))


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
