"""Tests for FA worker bootstrap engine selection.

Mirrors the ASR engine-resolution tests in
``batchalign/tests/pipelines/asr/test_asr_model_loading.py`` and the
translate ones in
``batchalign/tests/pipelines/translate/test_translation_model_loading.py``.

The FA resolver previously lived inline in ``load_fa_engine`` and
silently fell through to Wave2Vec on any unknown wire string. These
tests pin the extracted typed resolver behavior so the silent
fallthrough cannot recur.
"""

from __future__ import annotations

import pytest

from batchalign.worker._model_loading.forced_alignment import resolve_fa_engine
from batchalign.worker._types import FaEngine


class TestResolveFaEngine:
    """Engine selection must stay deterministic, typed, and loud on bad input."""

    def test_whisper_override_wins(self) -> None:
        assert resolve_fa_engine({"fa": "whisper"}) is FaEngine.WHISPER

    def test_wave2vec_override_wins(self) -> None:
        assert resolve_fa_engine({"fa": "wave2vec"}) is FaEngine.WAVE2VEC

    def test_wav2vec_canto_override_wins(self) -> None:
        assert (
            resolve_fa_engine({"fa": "wav2vec_canto"})
            is FaEngine.WAV2VEC_CANTO
        )

    def test_default_without_overrides_is_whisper(self) -> None:
        assert resolve_fa_engine(None) is FaEngine.WHISPER

    def test_empty_dict_falls_through_to_default(self) -> None:
        assert resolve_fa_engine({}) is FaEngine.WHISPER

    def test_unrelated_override_keys_are_ignored(self) -> None:
        # Only the ``fa`` key matters here; other engine keys belong
        # to other resolvers.
        assert resolve_fa_engine({"asr": "whisper"}) is FaEngine.WHISPER

    def test_unknown_engine_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unknown fa engine 'wisper'"):
            resolve_fa_engine({"fa": "wisper"})

    def test_unknown_engine_error_mentions_supported_options(self) -> None:
        # The supported-engines list is derived from the FaEngine
        # enum, so adding a 4th variant requires zero changes here.
        with pytest.raises(ValueError) as exc_info:
            resolve_fa_engine({"fa": "x"})
        msg = str(exc_info.value)
        for variant in FaEngine:
            assert variant.value in msg, (
                f"error message {msg!r} missing variant {variant.value!r}"
            )
