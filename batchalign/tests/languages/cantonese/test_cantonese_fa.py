# affects: batchalign/inference/languages/cantonese/_cantonese_fa.py
"""Unit tests for _cantonese_fa.py: jyutping conversion and FA provider."""

from __future__ import annotations

import builtins
import sys

import pytest

from batchalign.inference.languages.cantonese._cantonese_fa import (
    CantoneseFaHost,
    _hanzi_to_jyutping,
    _maybe_romanize,
    load_cantonese_fa,
)

from .conftest import PyCantoneseFake

# ---------------------------------------------------------------------------
# _hanzi_to_jyutping
# ---------------------------------------------------------------------------


class TestHanziToJyutping:
    """Tests use real PyCantonese, no faked jyutping dictionary."""

    def test_single_char(self, pc_real) -> None:  # type: ignore[no-untyped-def]
        assert _hanzi_to_jyutping(pc_real, "好") == "hou"

    def test_multi_char(self, pc_real) -> None:  # type: ignore[no-untyped-def]
        assert _hanzi_to_jyutping(pc_real, "你好") == "nei'hou"

    def test_multisyllable_single_word_separates_every_syllable(self, pc_real) -> None:  # type: ignore[no-untyped-def]
        """The case that had no test, and the one that was silently wrong.

        `你好` above passes under BOTH pycantonese 4.x and 5.x, because its
        segmenter splits it into two words and the apostrophe then falls out of
        the word join either way. So the existing coverage could not see the
        difference between separating syllables and separating whatever the
        segmenter happened to split.

        These words are kept as ONE word by the segmenter, so under 4.x their
        syllables ran together (`gwongdungwaa`) despite the function promising
        syllable separation. Pinned here because the guarantee is what the
        aligner is handed, and it must not depend on segmenter behaviour.
        """
        assert _hanzi_to_jyutping(pc_real, "廣東話") == "gwong'dung'waa"
        assert _hanzi_to_jyutping(pc_real, "香港") == "hoeng'gong"
        assert _hanzi_to_jyutping(pc_real, "唔該") == "m'goi"

    def test_separation_does_not_depend_on_segmentation(self, pc_real) -> None:  # type: ignore[no-untyped-def]
        """Two-syllable input separates the same whether or not it is one word.

        `我哋` is one segmented word and `一個` is two, and the aligner should
        not be able to tell. This is the invariant the 4.x behaviour broke.
        """
        assert _hanzi_to_jyutping(pc_real, "我哋") == "ngo'dei"
        assert _hanzi_to_jyutping(pc_real, "一個") == "jat'go"

    def test_unknown_char_passthrough(self, pc_real) -> None:  # type: ignore[no-untyped-def]
        assert _hanzi_to_jyutping(pc_real, "xyz") == "xyz"

    def test_mixed_known_unknown(self, pc_real) -> None:  # type: ignore[no-untyped-def]
        # The old expectation was "hou", pinning the loss of the unknown
        # source span. A known pronunciation must not erase its neighbour.
        assert _hanzi_to_jyutping(pc_real, "好x") == "hou'x"

    def test_missing_romanizer_source_span_is_a_named_failure(self) -> None:
        from batchalign.inference.languages.cantonese._cantonese_fa import (
            RomanizationCoverageError,
        )

        class IncompleteRomanizer:
            @staticmethod
            def characters_to_jyutping(_text: str):
                return [("好", "hou2")]

        with pytest.raises(RomanizationCoverageError, match="source"):
            _hanzi_to_jyutping(IncompleteRomanizer(), "好x")

    def test_empty_string(self, pc_real) -> None:  # type: ignore[no-untyped-def]
        assert _hanzi_to_jyutping(pc_real, "") == ""

    def test_corpus_word_gam(self, pc_real) -> None:  # type: ignore[no-untyped-def]
        assert _hanzi_to_jyutping(pc_real, "咁") == "gam"

    def test_corpus_word_gaau(self, pc_real) -> None:  # type: ignore[no-untyped-def]
        assert _hanzi_to_jyutping(pc_real, "搞") == "gaau"

    def test_tone_stripping(self, pc_real) -> None:  # type: ignore[no-untyped-def]
        assert _hanzi_to_jyutping(pc_real, "我") == "ngo"
        assert _hanzi_to_jyutping(pc_real, "係") == "hai"

    def test_malformed_romanizer_result_is_a_named_failure(self) -> None:
        from batchalign.inference.languages.cantonese._cantonese_fa import (
            RomanizationCoverageError,
        )

        class _BrokenPc:
            @staticmethod
            def characters_to_jyutping(_text: str):
                return None

        with pytest.raises(RomanizationCoverageError):
            _hanzi_to_jyutping(_BrokenPc(), "你好")


# ---------------------------------------------------------------------------
# _maybe_romanize
# ---------------------------------------------------------------------------


class TestMaybeRomanize:
    """Tests use real PyCantonese, no faked jyutping dictionary."""

    def test_yue_romanizes(self, pc_real) -> None:  # type: ignore[no-untyped-def]
        assert _maybe_romanize(pc_real, "好", "yue") == "hou"

    def test_non_yue_passthrough(self, pc_real) -> None:  # type: ignore[no-untyped-def]
        assert _maybe_romanize(pc_real, "hello", "eng") == "hello"

    def test_corpus_phrase_words(self, pc_real) -> None:  # type: ignore[no-untyped-def]
        words = ["咁", "搞", "笑"]
        romanized = [_maybe_romanize(pc_real, w, "yue") for w in words]
        assert romanized == ["gam", "gaau", "siu"]


# ---------------------------------------------------------------------------
# CantoneseFaHost.align_words (with faked model environment)
# ---------------------------------------------------------------------------


class TestAlignWords:
    def test_mixed_source_span_reaches_the_aligner(self, pc_real) -> None:  # type: ignore[no-untyped-def]
        class ReachedAlignment(Exception):
            def __init__(self, words: tuple[str, ...]) -> None:
                self.words = words

        def capture(_model: object, _audio: object, words: list[str]):
            # Stop at the model boundary instead of inventing model output.
            raise ReachedAlignment(tuple(words))

        host = CantoneseFaHost(
            model=object(), romanizer=pc_real, infer_wave2vec_fa=capture, lang="yue"
        )
        with pytest.raises(ReachedAlignment) as reached:
            host.align_words(object(), ["好x", "你好"])
        assert reached.value.words == ("hou'x", "nei'hou")

    def test_words_are_romanized_before_alignment(
        self, cantonese_fa_host: CantoneseFaHost
    ) -> None:
        rows = cantonese_fa_host.align_words(object(), ["你", "好"])

        assert [row[0] for row in rows] == ["nei", "hou"]
        assert [row[1] for row in rows] == [(0, 100), (100, 200)]

    def test_no_words_aligns_nothing(self, cantonese_fa_host: CantoneseFaHost) -> None:
        assert cantonese_fa_host.align_words(object(), []) == []

    def test_a_non_cantonese_host_passes_words_through(self) -> None:
        seen: list[list[str]] = []

        def record(_model, _audio, words):
            seen.append(words)
            return []

        host = CantoneseFaHost(
            model=object(),
            romanizer=PyCantoneseFake(),
            infer_wave2vec_fa=record,
            lang="eng",
        )

        host.align_words(object(), ["你", "hello"])

        assert seen == [["你", "hello"]]

    def test_aligner_failures_propagate(self) -> None:
        def boom(*_args, **_kwargs):
            raise RuntimeError("alignment failed")

        host = CantoneseFaHost(
            model=object(),
            romanizer=PyCantoneseFake(),
            infer_wave2vec_fa=boom,
            lang="yue",
        )

        with pytest.raises(RuntimeError, match="alignment failed"):
            host.align_words(object(), ["你"])


def test_load_cantonese_fa_reports_missing_dependency(monkeypatch) -> None:
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pycantonese":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="pycantonese"):
        load_cantonese_fa("yue", None)


def test_load_cantonese_fa_returns_the_host_it_built(monkeypatch) -> None:
    import batchalign.inference.languages.cantonese._cantonese_fa as module

    fake_pc = object()

    def fake_infer(model, audio, words):
        return [("word", (0, 100))]

    monkeypatch.setitem(sys.modules, "pycantonese", fake_pc)
    monkeypatch.setattr(
        "batchalign.inference.fa.load_wave2vec_fa",
        lambda device_policy=None: {"device_policy": device_policy},
    )
    monkeypatch.setattr("batchalign.inference.fa.infer_wave2vec_fa", fake_infer)

    host = module.load_cantonese_fa("yue", None, device_policy="policy")

    assert host.model == {"device_policy": "policy"}
    assert host.romanizer is fake_pc
    assert host.infer_wave2vec_fa is fake_infer
    # The language the load was asked for is the host's, not a module default.
    assert host.lang == "yue"
    assert module.load_cantonese_fa("eng", None).lang == "eng"
    # Nothing is stored at module level.
    assert not hasattr(module, "_model")
