"""Retain ASR punctuation without inventing or shifting word timestamps.

Punctuation moves from the recognized text onto the aligner's units only
under an exact lexical correspondence. When there is none the refusal says
why, and the caller keeps what remains honest: the spoken units without
punctuation when only a nonlexical unit is in the way, or the recognized
text untimed when the aligner spelled different words. Neither fails the file.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, replace
from typing import assert_never

from batchalign.inference.qwen_timing import (
    AlignedWindowWord,
    AudioWindow,
    SpokenWindowWords,
)


def _lexical(character: str) -> bool:
    return character == "'" or unicodedata.category(character)[0] in "LNM"


@dataclass(frozen=True, slots=True)
class LexicalDrift:
    """The aligner's units do not spell the recognized text's lexical
    characters (letters, numbers, combining marks, the ASCII apostrophe)."""

    recognized: str
    aligned: str


@dataclass(frozen=True, slots=True)
class NonlexicalUnit:
    """A spoken unit with no lexical character, so no text position to anchor
    punctuation to."""

    index: int
    text: str


type PunctuationRefusal = LexicalDrift | NonlexicalUnit


def describe_punctuation_refusal(refusal: PunctuationRefusal) -> str:
    match refusal:
        case LexicalDrift(recognized=recognized, aligned=aligned):
            return (
                f"aligned units spell {aligned!r} where the recognized text has "
                f"{recognized!r}"
            )
        case NonlexicalUnit(index=index, text=text):
            return f"spoken unit {index + 1} {text!r} has no lexical character"
        case unreachable:
            assert_never(unreachable)


_KEPT = "。！？.!?，,；;：:"
"""Sentence and clause punctuation carried onto the preceding unit. Quotes and
arbitrary symbols are not restored."""


@dataclass(frozen=True, slots=True)
class _Correspondence:
    """Proof that every unit maps onto the text's lexical characters in order,
    carrying each unit's punctuation: the kept marks in the text gap from just
    past its last lexical character to the next unit's first one."""

    punctuation: tuple[str, ...]


def _judge_correspondence(
    text: str, aligned: SpokenWindowWords
) -> _Correspondence | PunctuationRefusal:
    """The one rule: an exact lexical correspondence, or why there is none."""
    source = [i for i, c in enumerate(text) if _lexical(c)]
    lexical_units = [
        "".join(c for c in word.text if _lexical(c)) for word in aligned.words
    ]
    recognized = "".join(text[i] for i in source)
    spelled = "".join(lexical_units)
    if spelled != recognized:
        return LexicalDrift(recognized, spelled)
    punctuation: list[str] = []
    offset = 0
    for index, (word, lexical) in enumerate(
        zip(aligned.words, lexical_units, strict=True)
    ):
        if not lexical:
            return NonlexicalUnit(index, word.text)
        offset += len(lexical)
        following = source[offset] if offset < len(source) else len(text)
        gap = text[source[offset - 1] + 1 : following]
        punctuation.append("".join(c for c in gap if c in _KEPT))
    return _Correspondence(tuple(punctuation))


@dataclass(frozen=True, slots=True, init=False)
class PunctuatedAlignment:
    """An exact lexical correspondence, with punctuation attached to its word.

    Built only from the `_Correspondence` proof that `restore_punctuation`
    obtains, so there is no unproven way to make one and no second judgment.
    Each unit keeps its immutable timing evidence. Text is provider output,
    never CHAT.
    """

    words: tuple[AlignedWindowWord, ...]
    window: AudioWindow

    def __init__(self, aligned: SpokenWindowWords, proof: _Correspondence) -> None:
        object.__setattr__(
            self,
            "words",
            tuple(
                replace(word, text=word.text + marks)
                for word, marks in zip(aligned.words, proof.punctuation, strict=True)
            ),
        )
        object.__setattr__(self, "window", aligned.window)


def restore_punctuation(
    text: str, spoken: SpokenWindowWords
) -> PunctuatedAlignment | PunctuationRefusal:
    """Punctuation restored onto the spoken units, or why it cannot be. Never
    raises on model output; the caller decides what each refusal keeps."""
    match _judge_correspondence(text, spoken):
        case _Correspondence() as proof:
            return PunctuatedAlignment(spoken, proof)
        case LexicalDrift() | NonlexicalUnit() as refusal:
            return refusal
        case unreachable:
            assert_never(unreachable)
