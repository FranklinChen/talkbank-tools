"""Tests for _tokenizer_realign.py: MWT hint annotation and merge logic.

Verifies that merged tokens are returned as (text, bool) MWT hint tuples
matching Python master's tokenizer_processor rules exactly:
  - English tokens containing ' (except o' forms) → (text, True)
  - All other merges → (text, False)

The contraction rule itself is Rust's (``RealignLanguage`` in
``batchalign-transform``) and is tested there; these tests exercise it
through the boundary Python calls.
"""

from __future__ import annotations

import pytest

from batchalign.inference._domain_types import StanzaCode
from batchalign.inference._tokenizer_realign import (
    TokenizerContext,
    _conform,
    _realign_sentence,
    make_tokenizer_postprocessor,
)
from batchalign_core import RealignLanguage

EN = RealignLanguage.English
FR = RealignLanguage.French

# ---------------------------------------------------------------------------
# TokenizerContext: the language is read from a Stanza code once
# ---------------------------------------------------------------------------


class TestTokenizerContextLanguage:
    """A context cannot exist without the language its rules depend on."""

    def test_stanza_code_names_the_realignment_language(self) -> None:
        assert TokenizerContext(StanzaCode("en")).language == EN
        assert TokenizerContext(StanzaCode("fr")).language == FR
        assert TokenizerContext(StanzaCode("it")).language == RealignLanguage.Italian
        assert TokenizerContext(StanzaCode("de")).language == RealignLanguage.Other

    def test_empty_stanza_code_is_refused(self) -> None:
        """The empty code used to be the postprocessor's default, and an
        English pipeline built without its code silently lost contraction
        expansion."""
        with pytest.raises(ValueError, match="Stanza language code"):
            TokenizerContext(StanzaCode(""))


# ---------------------------------------------------------------------------
# _realign_sentence: merged token tuples
# ---------------------------------------------------------------------------


class TestRealignSentenceMwtTuples:
    """Verify merged tokens become (text, bool) tuples with correct MWT hint."""

    def test_spurious_split_becomes_false_tuple(self) -> None:
        """Non-English/non-apostrophe merge → (text, False)."""
        # Stanza split "ice-cream" into ["ice", "-", "cream"]
        tokens = ["ice", "-", "cream"]
        words = ["ice-cream"]
        result = _realign_sentence(tokens, words, EN)
        assert result == [("ice-cream", False)]

    def test_english_contraction_split_becomes_true_tuple(self) -> None:
        """English apostrophe merge → (text, True)."""
        # Stanza split "don't" into ["don", "'t"], maps to one word "don't"
        tokens = ["don", "'t"]
        words = ["don't"]
        result = _realign_sentence(tokens, words, EN)
        assert result == [("don't", True)]

    def test_possessive_becomes_true_tuple(self) -> None:
        """English possessive merge → (text, True)."""
        tokens = ["Claus", "'"]
        words = ["Claus'"]
        result = _realign_sentence(tokens, words, EN)
        assert result == [("Claus'", True)]

    def test_french_apostrophe_becomes_false_tuple(self) -> None:
        """Non-English apostrophe merge → (text, False)."""
        tokens = ["l", "'"]
        words = ["l'"]
        result = _realign_sentence(tokens, words, FR)
        assert result == [("l'", False)]

    def test_french_native_components_remain_one_authoritative_word(self) -> None:
        result = _realign_sentence(
            ["l'", "escargot", "dort", "."],
            ["l'escargot", "dort", "."],
            FR,
        )
        assert result == [("l'escargot", ["l'", "escargot"]), "dort", "."]

    def test_stanza_accepts_checked_expansion_without_loading_models(self) -> None:
        from stanza.models.common.doc import Document
        from stanza.models.tokenization.utils import postprocess_doc

        surface = "l'escargot dort ."
        native = [
            [
                {"id": (1,), "text": "l'"},
                {"id": (2,), "text": "escargot"},
                {"id": (3,), "text": "dort"},
                {"id": (4,), "text": "."},
            ]
        ]
        ctx = TokenizerContext(StanzaCode("fr"))
        ctx.original_words = [["l'escargot", "dort", "."]]
        rows = postprocess_doc(native, make_tokenizer_postprocessor(ctx), surface)
        document = Document(rows, text=surface)
        assert [token.text for token in document.sentences[0].tokens] == [
            "l'escargot",
            "dort",
            ".",
        ]
        assert [word.text for word in document.sentences[0].words] == [
            "l'",
            "escargot",
            "dort",
            ".",
        ]

    def test_single_token_unchanged(self) -> None:
        """Single-token words pass through unchanged (no merge, no tuple)."""
        tokens = ["hello", "world"]
        words = ["hello", "world"]
        result = _realign_sentence(tokens, words, EN)
        # 1:1 mapping → fast path, tokens unchanged
        assert result == ["hello", "world"]

    def test_mixed_merge_and_single(self) -> None:
        """Merged tokens get tuple; single tokens stay as-is."""
        # "ice-cream" split, "good" and "bye" each single
        tokens = ["ice", "-", "cream", "good", "bye"]
        words = ["ice-cream", "good", "bye"]
        result = _realign_sentence(tokens, words, EN)
        assert result[0] == ("ice-cream", False)
        assert _conform(result[1]) == "good"
        assert _conform(result[2]) == "bye"

    def test_merge_elsewhere_does_not_drop_later_mwt_hint(self) -> None:
        """A preceding merge must not strip a later unaffected MWT tuple.

        Regression for the English preserve-mode defect on:
        ``mm-hmm that's right .``

        Stanza tokenizes this as:
        - ``mm``, ``-``, ``hmm``  (spurious split that must be merged)
        - ``(\"that's\", True)``   (native MWT hint that must survive)
        """
        tokens = ["mm", "-", "hmm", ("that's", True), "right", "."]
        words = ["mm-hmm", "that's", "right", "."]
        result = _realign_sentence(tokens, words, EN)
        assert result[0] == ("mm-hmm", False)
        assert result[1] == ("that's", True)
        assert _conform(result[2]) == "right"
        assert _conform(result[3]) == "."

    def test_language_without_a_rule_suppresses_apostrophe_merges(self) -> None:
        """A language with no contraction rule → False for apostrophe merges."""
        tokens = ["don", "'t"]
        words = ["don't"]
        result = _realign_sentence(tokens, words, RealignLanguage.Other)
        assert result == [("don't", False)]

    def test_character_mismatch_refuses_unbound_native_tokens(self) -> None:
        tokens = ["totally", "different"]
        words = ["something", "else"]
        with pytest.raises(ValueError, match="does not fit"):
            _realign_sentence(tokens, words, EN)

    def test_native_token_cannot_cross_word_boundary(self) -> None:
        with pytest.raises(ValueError, match="does not fit"):
            _realign_sentence(["foobar"], ["foo", "bar"], EN)

    def test_embedded_space_fragments_preserve_later_native_hint(self) -> None:
        result = _realign_sentence(
            ["ice cream", ("that's", True)], ["ice", "cream", "that's"], EN
        )
        assert result == ["ice", "cream", ("that's", True)]


# ---------------------------------------------------------------------------
# make_tokenizer_postprocessor: the context's language
# ---------------------------------------------------------------------------


class TestMakeTokenizerPostprocessor:
    """Verify the context's language is applied correctly."""

    def test_english_context_contraction_tuple(self) -> None:
        ctx = TokenizerContext(StanzaCode("en"))
        ctx.original_words = [["don't"]]
        pp = make_tokenizer_postprocessor(ctx)
        batch = [["don", "'t"]]
        result = pp(batch)  # type: ignore[operator]
        # Should be (don't, True), English contraction
        assert result[0][0] == ("don't", True)

    def test_non_english_context_no_contraction(self) -> None:
        ctx = TokenizerContext(StanzaCode("fr"))
        ctx.original_words = [["l'"]]
        pp = make_tokenizer_postprocessor(ctx)
        batch = [["l", "'"]]
        result = pp(batch)  # type: ignore[operator]
        assert result[0][0] == ("l'", False)

    def test_empty_context_returns_batch_unchanged(self) -> None:
        ctx = TokenizerContext(StanzaCode("en"))
        # No original_words set
        pp = make_tokenizer_postprocessor(ctx)
        batch = [["hello", "world"]]
        result = pp(batch)  # type: ignore[operator]
        assert result is batch

    def test_extra_native_sentence_cannot_escape_bound_request(self) -> None:
        ctx = TokenizerContext(StanzaCode("en"))
        ctx.original_words = [["hello"]]
        pp = make_tokenizer_postprocessor(ctx)
        with pytest.raises(ValueError, match="sentence count"):
            pp([["hello"], ["unrequested"]])
