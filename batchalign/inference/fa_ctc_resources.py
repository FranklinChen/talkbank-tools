"""MMS labels and native CPU CTC allocation shapes, without inference.

The allocation layout is read from TorchAudio's CPU implementation at
455331b1b3bb4aabfe3228f6f911fc3f11ae93b1:
https://github.com/pytorch/audio/blob/455331b1b3bb4aabfe3228f6f911fc3f11ae93b1/src/libtorchaudio/forced_align/cpu/compute.cpp
It is not an audio-duration cutoff. Runtime qualification must bind this
layout to the installed native backend before it can authorize a lease.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType

import torch

from batchalign.inference._admission import _FactoryAdmission
from batchalign.worker._model_errors import ModelOutputInvalidError


class CtcFrameCapacityError(ValueError):
    """The actual CTC label path cannot fit the encoder's frame population."""

    def __init__(self, available_frames: int, required_frames: int) -> None:
        self.available_frames = available_frames
        self.required_frames = required_frames
        super().__init__(
            f"CTC needs {required_frames} frames for its labels and repeats; "
            f"the encoder supplies {available_frames}"
        )


@dataclass(frozen=True, slots=True, init=False)
class MmsAlphabet(_FactoryAdmission):
    """One admitted immutable model alphabet, shared by planning and execution."""

    labels: Mapping[str, int]
    wildcard: int

    @classmethod
    def admit(cls, dictionary: Mapping[str, int]) -> MmsAlphabet:
        labels = dict(dictionary)
        if any(type(label) is not int or label < 0 for label in labels.values()):
            raise ValueError("MMS vocabulary has an invalid label id")
        if "*" not in labels or labels["*"] <= 0:
            raise ValueError("MMS vocabulary must name a nonblank wildcard label")
        value = object.__new__(cls)
        object.__setattr__(value, "labels", MappingProxyType(labels))
        object.__setattr__(value, "wildcard", labels["*"])
        return value


@dataclass(frozen=True, slots=True)
class WildcardSymbol:
    """An input symbol represented by the model's explicit wildcard label."""

    word: int
    symbol: str


@dataclass(frozen=True, slots=True, init=False)
class _MmsTargetWord(_FactoryAdmission):
    """A source word owns the exact nonempty native labels assigned to it."""

    source: str
    labels: tuple[int, ...]


@dataclass(frozen=True, slots=True, init=False)
class MmsTargetSequence(_FactoryAdmission):
    """One ordered label sequence owning its partition into source words."""

    words: tuple[_MmsTargetWord, ...]
    wildcard_symbols: tuple[WildcardSymbol, ...]
    blank_only_words: tuple[int, ...]

    @classmethod
    def from_words(
        cls, words: Sequence[str], alphabet: MmsAlphabet
    ) -> MmsTargetSequence:
        if not words or any(not word for word in words):
            raise ValueError("MMS targets require nonempty requested words")
        dictionary = alphabet.labels
        wildcard = alphabet.wildcard
        target_words: list[_MmsTargetWord] = []
        substitutions: list[WildcardSymbol] = []
        blank_only: list[int] = []
        for ordinal, word in enumerate(words):
            labels: list[int] = []
            for symbol in word.lower():
                if symbol in dictionary:
                    label = dictionary[symbol]
                else:
                    label = wildcard
                    substitutions.append(WildcardSymbol(word=ordinal, symbol=symbol))
                # CTC's blank is not a target label. A source word remains
                # present even when all its symbols map to blank; record that
                # shape and use the model's wildcard, as the runner does.
                if label != 0:
                    labels.append(label)
            if not labels:
                labels.append(wildcard)
                blank_only.append(ordinal)
            target = object.__new__(_MmsTargetWord)
            object.__setattr__(target, "source", word)
            object.__setattr__(target, "labels", tuple(labels))
            target_words.append(target)
        value = object.__new__(cls)
        object.__setattr__(value, "words", tuple(target_words))
        object.__setattr__(value, "wildcard_symbols", tuple(substitutions))
        object.__setattr__(value, "blank_only_words", tuple(blank_only))
        return value

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(label for word in self.words for label in word.labels)

    @property
    def word_lengths(self) -> tuple[int, ...]:
        return tuple(len(word.labels) for word in self.words)

    def bind_spans[Span](self, spans: Sequence[Span]) -> MmsAlignedWords[Span]:
        """Admit the native result once, before constructing word intervals."""
        return MmsAlignedWords._admit(self, spans)

    @property
    def minimum_frames(self) -> int:
        ids = self.ids
        repeats = sum(left == right for left, right in pairwise(ids))
        return len(ids) + repeats


@dataclass(frozen=True, slots=True, init=False)
class _MmsAlignedWord[Span](_FactoryAdmission):
    """A source word paired with all and only its native label spans."""

    source: str
    spans: tuple[Span, ...]


@dataclass(frozen=True, slots=True, init=False)
class MmsAlignedWords[Span](_FactoryAdmission):
    """The complete native population, partitioned by the prepared targets."""

    words: tuple[_MmsAlignedWord[Span], ...]

    @classmethod
    def _admit(
        cls, targets: MmsTargetSequence, spans: Sequence[Span]
    ) -> MmsAlignedWords[Span]:
        if len(spans) != len(targets.ids):
            raise ModelOutputInvalidError(
                "CTC label population mismatch: "
                f"requested {len(targets.ids)}, received {len(spans)}"
            )
        words: list[_MmsAlignedWord[Span]] = []
        start = 0
        for word in targets.words:
            end = start + len(word.labels)
            aligned = object.__new__(_MmsAlignedWord)
            object.__setattr__(aligned, "source", word.source)
            object.__setattr__(aligned, "spans", tuple(spans[start:end]))
            words.append(aligned)
            start = end
        value = object.__new__(cls)
        object.__setattr__(value, "words", tuple(words))
        return value


@dataclass(frozen=True, slots=True, init=False)
class AdmittedCtcShape(_FactoryAdmission):
    """Labels paired with an encoder population that can contain their path.

    This establishes CTC shape feasibility, not acoustic accuracy or a host
    memory grant. The native-workspace quantity excludes already-produced
    emissions, input/target tensors and Python output shaping.
    """

    targets: MmsTargetSequence
    frames: int
    scalar_bytes: int

    @classmethod
    def admit(
        cls, targets: MmsTargetSequence, *, frames: int, classes: int, scalar_bytes: int
    ) -> AdmittedCtcShape:
        if frames <= 0 or classes <= 0:
            raise ValueError("CTC encoder dimensions must be positive")
        if scalar_bytes not in {
            torch.float16.itemsize,
            torch.float32.itemsize,
            torch.float64.itemsize,
        }:
            raise ValueError("CPU CTC requires a supported floating-point scalar width")
        if any(label >= classes for label in targets.ids):
            raise ValueError("target label lies outside the encoder vocabulary")
        if frames < targets.minimum_frames:
            raise CtcFrameCapacityError(frames, targets.minimum_frames)
        value = object.__new__(cls)
        object.__setattr__(value, "targets", targets)
        object.__setattr__(value, "frames", frames)
        object.__setattr__(value, "scalar_bytes", scalar_bytes)
        return value

    @property
    def native_workspace_bytes(self) -> int:
        states = 2 * len(self.targets.ids) + 1
        alpha_rows = 2 * states * self.scalar_bytes
        backpointers = self.frames * states * torch.int8.itemsize
        paths = self.frames * torch.int64.itemsize
        return alpha_rows + backpointers + paths
