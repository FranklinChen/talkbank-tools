"""Build morphosyntax batch items the way the Rust side sends them.

Why this exists: the worker's ``MorphosyntaxBatchItem`` keeps the CHAT
main-tier words and the utterance terminator apart, and rejects an item whose
``words`` end with the terminator (the terminator is evidence for Stanza, never
a ``%mor`` word). Golden tests that wrote their items by hand kept the old
shape, a trailing ``"."`` in ``words``, for two months after that rule landed:
the worker rejected every item, the result was ``None``, and the tests failed
with ``'NoneType' object has no attribute 'get'``, which said nothing about the
cause. One test also left out the required ``special_forms`` field.

``morphosyntax_item`` takes the words and the terminator separately and sizes
``special_forms`` from the words, so neither mistake can be written again.
``first_raw_sentence`` reads a response and fails with the worker's own error
when an item was rejected, so the next contract change explains itself.
"""

from __future__ import annotations

from typing import Any


def morphosyntax_item(
    chat_words: list[str], *, terminator: str = ".", lang: str = "eng"
) -> dict[str, Any]:
    """One batch item: CHAT words, its terminator, and no special forms."""
    return {
        "words": list(chat_words),
        "terminator": terminator,
        "special_forms": [[None, None]] * len(chat_words),
        "lang": lang,
    }


def chat_words_of(stanza_tokens: list[str], terminator: str = ".") -> list[str]:
    """The CHAT words of a token list that ends with its terminator.

    For fixtures that serve two consumers: Stanza, which is given the
    terminator as text, and the batch item, which must not carry it.
    """
    assert stanza_tokens and stanza_tokens[-1] == terminator, (
        f"expected tokens ending with {terminator!r}, got {stanza_tokens}"
    )
    return stanza_tokens[:-1]


def first_raw_sentence(response: Any) -> list[dict[str, Any]]:
    """``raw_sentences[0]`` of the first result, or the worker's own error."""
    first = response.results[0]
    assert first.error is None, f"worker rejected the item: {first.error}"
    raw = first.result.get("raw_sentences", [[]])
    return raw[0] if raw else []
