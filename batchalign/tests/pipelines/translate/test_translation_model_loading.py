"""Tests for translation worker bootstrap engine selection.

Mirrors the ASR engine-resolution tests in
``batchalign/tests/pipelines/asr/test_asr_model_loading.py``. The translation
loader previously discarded its ``engine_overrides`` parameter outright,
hard-coding Google as the backend even when the Rust control plane passed
``{"translate": "seamless"}``. These tests pin the resolver behavior so the
discard cannot recur silently.
"""

from __future__ import annotations

import pytest

from batchalign.inference._domain_types import TranslationBackend
from batchalign.worker._model_loading.translation import resolve_translate_engine


class TestResolveTranslateEngine:
    """Engine selection must stay deterministic, typed, and loud on bad input."""

    def test_seamless_override_wins(self) -> None:
        assert (
            resolve_translate_engine({"translate": "seamless"})
            is TranslationBackend.SEAMLESS
        )

    def test_google_override_wins(self) -> None:
        assert (
            resolve_translate_engine({"translate": "google"})
            is TranslationBackend.GOOGLE
        )

    def test_default_without_overrides_is_google(self) -> None:
        assert resolve_translate_engine(None) is TranslationBackend.GOOGLE

    def test_empty_dict_falls_through_to_default(self) -> None:
        assert resolve_translate_engine({}) is TranslationBackend.GOOGLE

    def test_unrelated_override_keys_are_ignored(self) -> None:
        # Only the ``translate`` key matters here; other engine keys
        # belong to other resolvers.
        assert (
            resolve_translate_engine({"asr": "seamless"})
            is TranslationBackend.GOOGLE
        )

    def test_unknown_engine_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unknown translate engine 'gogle'"):
            resolve_translate_engine({"translate": "gogle"})

    def test_unknown_engine_error_mentions_supported_options(self) -> None:
        with pytest.raises(ValueError, match="google, seamless"):
            resolve_translate_engine({"translate": "whisper"})
