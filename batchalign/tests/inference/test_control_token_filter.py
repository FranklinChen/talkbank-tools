"""Unit tests for the Stanza control-token ingress workaround.

Pure-function tests — no Stanza, no Pipeline, no model downloads. Runs
on every `make test` pass. The matching integration test lives at
``batchalign/tests/pipelines/morphosyntax/test_control_token_leak_propagation.py``
and exercises the full ``batch_infer_morphosyntax`` path on real
Stanza output.
"""

from __future__ import annotations

import pytest

from batchalign.inference._control_token_filter import (
    CONTROL_TOKEN_RE,
    ControlTokenLeak,
    LeakField,
    strip_control_tokens,
    strip_control_tokens_in_sentence,
)


# ── strip_control_tokens ─────────────────────────────────────────────


class TestStripControlTokens:
    """The 2026-04-14 Finnish MWT leak was ``<SOS>tos`` → should become
    ``tos``. Broaden coverage to all the canonical neural-LM control
    tokens so future leaks in other languages are caught the same way.
    """

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("<SOS>tos", "tos"),  # the Finnish MWT leak
            ("<sos>tos", "tos"),  # lemmatizer lowercases the same leak
            ("tos<EOS>", "tos"),
            ("<SOS>tos<EOS>", "tos"),
            ("<UNK>", ""),
            ("<s>hello</s>", "hello"),
            ("<PAD>", ""),
            ("<MASK>word", "word"),
        ],
    )
    def test_strips_known_control_tokens(self, raw: str, expected: str) -> None:
        assert strip_control_tokens(raw) == expected

    def test_preserves_text_without_control_tokens(self) -> None:
        assert strip_control_tokens("kato") == "kato"
        assert strip_control_tokens("tollei") == "tollei"

    def test_preserves_legitimate_angle_bracket_content(self) -> None:
        # Not a known control token — must be preserved. Guards against
        # overaggressive stripping that would mangle CHAT annotations
        # like <foo> if they ever appeared in Stanza output.
        assert strip_control_tokens("<foo>bar") == "<foo>bar"
        assert strip_control_tokens("<some-word>") == "<some-word>"

    def test_empty_string_returns_empty(self) -> None:
        assert strip_control_tokens("") == ""

    def test_idempotent(self) -> None:
        once = strip_control_tokens("<SOS>tos<EOS>")
        twice = strip_control_tokens(once)
        assert once == twice == "tos"


# ── strip_control_tokens_in_sentence ──────────────────────────────────


class TestStripInSentence:
    """Integration with Stanza's to_dict() shape. The fixture data
    mirrors the exact Document.to_dict() output captured from Stanza
    1.11.1 on ``"a tollei b"`` — see
    ``test_stanza_fi_mwt_sos_leak.py`` for the reproducer.
    """

    def test_empty_sentence_yields_no_leaks_and_no_mutation(self) -> None:
        sent: list[dict] = []
        leaks = strip_control_tokens_in_sentence(sent)
        assert leaks == []
        assert sent == []

    def test_clean_sentence_yields_no_leaks_and_no_mutation(self) -> None:
        sent = [
            {"id": 1, "text": "the", "lemma": "the", "upos": "DET"},
            {"id": 2, "text": "cat", "lemma": "cat", "upos": "NOUN"},
        ]
        snapshot = [dict(tok) for tok in sent]
        leaks = strip_control_tokens_in_sentence(sent)
        assert leaks == []
        assert sent == snapshot

    def test_strips_sos_from_mwt_expansion_word_reporting_both_fields(
        self,
    ) -> None:
        # The exact shape Stanza 1.11.1 emits on `"a tollei b"`:
        # token id=[2,3] is the MWT parent, id=2 is the leaked first
        # expansion word.
        sent = [
            {"id": 1, "text": "a", "lemma": "a", "upos": "NOUN"},
            {"id": [2, 3], "text": "tollei"},
            {"id": 2, "text": "<SOS>tos", "lemma": "<SOS>tos", "upos": "SYM"},
            {"id": 3, "text": "ei", "lemma": "ei", "upos": "VERB"},
            {"id": 4, "text": "b", "lemma": "b", "upos": "NOUN"},
        ]

        leaks = strip_control_tokens_in_sentence(sent)

        # Two leaks reported: text + lemma, both on token id 2.
        assert leaks == [
            ControlTokenLeak(
                token_id=2,
                field=LeakField.TEXT,
                value="<SOS>tos",
                stripped="tos",
            ),
            ControlTokenLeak(
                token_id=2,
                field=LeakField.LEMMA,
                value="<SOS>tos",
                stripped="tos",
            ),
        ]

        # The affected token is rewritten in place.
        affected = sent[2]
        assert affected["text"] == "tos"
        assert affected["lemma"] == "tos"

        # Every other token is untouched.
        assert sent[0] == {"id": 1, "text": "a", "lemma": "a", "upos": "NOUN"}
        assert sent[1] == {"id": [2, 3], "text": "tollei"}
        assert sent[3] == {"id": 3, "text": "ei", "lemma": "ei", "upos": "VERB"}
        assert sent[4] == {"id": 4, "text": "b", "lemma": "b", "upos": "NOUN"}

    def test_mwt_parent_token_without_lemma_is_safe(self) -> None:
        # MWT parent tokens have id=list and no lemma/upos. The filter
        # must not choke on the missing lemma key and must rewrite
        # only the text field.
        sent = [
            {"id": [1, 2], "text": "<SOS>gonna"},
            {"id": 1, "text": "gon", "lemma": "go", "upos": "VERB"},
            {"id": 2, "text": "na", "lemma": "to", "upos": "PART"},
        ]
        leaks = strip_control_tokens_in_sentence(sent)
        assert len(leaks) == 1
        assert leaks[0].field == LeakField.TEXT
        assert sent[0]["text"] == "gonna"
        assert "lemma" not in sent[0]

    def test_ignores_angle_bracket_content_that_is_not_a_control_token(
        self,
    ) -> None:
        sent = [{"id": 1, "text": "<foo>bar", "lemma": "<foo>bar", "upos": "X"}]
        leaks = strip_control_tokens_in_sentence(sent)
        assert leaks == []
        assert sent[0]["text"] == "<foo>bar"
        assert sent[0]["lemma"] == "<foo>bar"


# ── regex guards ──────────────────────────────────────────────────────


class TestControlTokenRegex:
    """Pin the exact vocabulary the filter matches so a future change
    to the regex is visible in a test diff."""

    @pytest.mark.parametrize(
        "token",
        ["<SOS>", "<EOS>", "<UNK>", "<PAD>", "<BOS>", "<CLS>", "<SEP>", "<MASK>", "<s>", "</s>"],
    )
    def test_matches_canonical_control_tokens(self, token: str) -> None:
        assert CONTROL_TOKEN_RE.fullmatch(token), (
            f"{token!r} should be recognized as a control token"
        )

    @pytest.mark.parametrize(
        "token",
        ["<foo>", "<some-word>", "<123>", "<SOS", "SOS>", "<>", ""],
    )
    def test_does_not_match_non_control_angle_bracket_content(
        self, token: str
    ) -> None:
        assert CONTROL_TOKEN_RE.fullmatch(token) is None, (
            f"{token!r} must not be treated as a control token"
        )
