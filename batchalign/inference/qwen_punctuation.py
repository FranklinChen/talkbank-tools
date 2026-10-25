"""Retain ASR punctuation without inventing or shifting word timestamps."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, replace

from batchalign.inference.qwen_timing import (
    AlignedWindowWord,
    AudioWindow,
    SpokenWindowWords,
)


def _lexical(character: str) -> bool:
    return character == "'" or unicodedata.category(character)[0] in "LNM"


@dataclass(frozen=True, slots=True, init=False)
class PunctuatedAlignment:
    """An exact lexical correspondence, with punctuation attached to its word.

    Construction refuses text drift. Only punctuation and whitespace may differ
    between the ASR text and the aligner's units; no fuzzy matching transfers
    timestamps to changed words. Text is provider output, never CHAT.
    """

    words: tuple[AlignedWindowWord, ...]
    window: AudioWindow

    def __init__(self, text: str, aligned: SpokenWindowWords) -> None:
        source = [(i, c) for i, c in enumerate(text) if _lexical(c)]
        expected = "".join(c for _, c in source)
        actual = "".join(c for word in aligned.words for c in word.text if _lexical(c))
        if actual != expected:
            raise ValueError(
                "Qwen alignment changed recognized words; cannot transfer punctuation"
            )
        result: list[AlignedWindowWord] = []
        offset = 0
        for word in aligned.words:
            count = sum(_lexical(c) for c in word.text)
            if count == 0:
                raise ValueError("Qwen alignment returned a nonlexical word")
            offset += count
            end = source[offset - 1][0] + 1
            next_start = source[offset][0] if offset < len(source) else len(text)
            # Preserve sentence/clause evidence, not quotes or arbitrary symbols.
            punctuation = "".join(
                c for c in text[end:next_start] if c in "。！？.!?，,；;：:"
            )
            result.append(replace(word, text=word.text + punctuation))
        object.__setattr__(self, "words", tuple(result))
        object.__setattr__(self, "window", aligned.window)
