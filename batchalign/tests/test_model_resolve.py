"""Tests for ``batchalign.models.resolve``: per-language model_id resolver.

The resolver is a nested dict keyed on model family then ISO-639-3 code.
These tests pin the entries we rely on in production, anyone removing or
renaming an entry must update both the resolver and any callers.
"""

from __future__ import annotations

from batchalign.models.resolve import resolve


class TestResolveUtteranceIsGone:
    """The utterance family no longer lives here.

    The boundary model a language loads is named by the Rust manifest and sent
    with the worker spawn, because an id must be known BEFORE a load in order
    to pin its revision. This asserts the copy is actually gone rather than
    merely unused: an entry that reappeared here would be a second answer to
    the question the manifest already answers, and the worker would load from
    it again.
    """

    def test_utterance_family_is_not_resolved_here(self) -> None:
        assert resolve("utterance", "eng") is None
        assert resolve("utterance", "yue") is None


class TestResolveWhisperHub:
    """WhisperHub ASR fine-tune model_id resolution.

    The table seeds `mal` because an empirical evaluation showed that
    `thennal/whisper-medium-ml` is the only path that produces coherent
    Malayalam: stock Whisper and Rev.AI both fail for that language.
    See `book/src/reference/whisper-hub-asr.md` for the comparison.
    """

    def test_whisper_hub_malayalam_returns_thennal_medium_ml(self) -> None:
        # RED: "whisper_hub" family is not yet seeded in _RESOLVER.
        assert resolve("whisper_hub", "mal") == "thennal/whisper-medium-ml"

    def test_whisper_hub_unseeded_language_returns_none(self) -> None:
        # The resolver must return None for languages we haven't
        # characterized. Callers are responsible for surfacing a typed
        # error to the user with a clear "no default for this language,
        # pass a model_id" message.
        assert resolve("whisper_hub", "eng") is None

    def test_unknown_family_returns_none(self) -> None:
        assert resolve("nonexistent_family", "mal") is None
