"""Tests for UdWord Pydantic validation of Stanza output.

Stanza's doc.to_dict() can omit required fields in two cases:
1. MWT Range tokens (id=[start, end]), only have id and text
2. Regular tokens where a processor (e.g. lemma) fails silently

The Rust UdWord struct requires: id, text, lemma, upos, head, deprel.
Missing any of these causes serde deserialization failure.  The Pydantic
``UdWord`` model mirrors the Rust struct and fills safe defaults.
"""

from __future__ import annotations

import pytest

from batchalign.inference.morphosyntax import (
    RelationRepair,
    RelationRepairKind,
    RepairedSentence,
    UdWord,
    _repaired_relation,
)

# --- Direct model tests ---


def test_udword_complete_token() -> None:
    """Complete token should pass validation unchanged."""
    w = UdWord.model_validate(
        {
            "id": 1,
            "text": "hello",
            "lemma": "hello",
            "upos": "INTJ",
            "head": 0,
            "deprel": "root",
            "xpos": "UH",
            "feats": None,
        }
    )
    assert w.lemma == "hello"
    assert w.upos == "INTJ"
    assert w.head == 0
    assert w.deprel == "root"
    assert w.xpos == "UH"


def test_udword_missing_lemma_defaults_to_text() -> None:
    """Regular token missing lemma should default to surface text."""
    w = UdWord.model_validate(
        {
            "id": 14,
            "text": "assistante",
            "upos": "NOUN",
            "head": 6,
            "deprel": "conj",
        }
    )
    assert w.lemma == "assistante"


def test_udword_range_token_gets_empty_lemma() -> None:
    """MWT Range token should get empty string lemma (not surface text)."""
    w = UdWord.model_validate({"id": [2, 3], "text": "au"})
    assert w.lemma == ""
    assert w.upos == "X"
    assert w.head == 0
    assert w.deprel == "dep"


def test_udword_missing_upos() -> None:
    """Missing upos should default to 'X'."""
    w = UdWord.model_validate(
        {
            "id": 1,
            "text": "foo",
            "lemma": "foo",
            "head": 0,
            "deprel": "root",
        }
    )
    assert w.upos == "X"


def test_udword_missing_head() -> None:
    """Missing head should default to 0."""
    w = UdWord.model_validate(
        {
            "id": 1,
            "text": "foo",
            "lemma": "foo",
            "upos": "NOUN",
            "deprel": "root",
        }
    )
    assert w.head == 0


def test_udword_missing_deprel() -> None:
    """Missing deprel should default to 'dep'."""
    w = UdWord.model_validate(
        {
            "id": 1,
            "text": "foo",
            "lemma": "foo",
            "upos": "NOUN",
            "head": 0,
        }
    )
    assert w.deprel == "dep"


def test_pad_relation_is_repaired_to_dep_and_reported() -> None:
    """A Stanza <PAD> label is no relation; it becomes 'dep', and is reported."""
    relation, repair = _repaired_relation("<PAD>", "etxean")
    assert relation == "dep"
    assert repair == RelationRepair(
        kind=RelationRepairKind.PAD_RELATION,
        word="etxean",
        from_relation="<PAD>",
        to_relation="dep",
    )


def test_any_angle_bracketed_relation_is_repaired() -> None:
    """Any angle-bracketed label (e.g. <UNK>) is padding, not a relation."""
    relation, repair = _repaired_relation("<UNK>", "foo")
    assert relation == "dep"
    assert repair is not None
    assert repair.kind is RelationRepairKind.PAD_RELATION


def test_relation_case_is_repaired_and_reported() -> None:
    """An upper-case UD relation is lowercased, and the subtype is kept.

    This was the one rewrite with no log line at all: it changed the value
    silently, so a corpus could not be asked how often it happened.
    """
    relation, repair = _repaired_relation("NMOD:poss", "his")
    assert relation == "nmod:poss"
    assert repair == RelationRepair(
        kind=RelationRepairKind.RELATION_CASE,
        word="his",
        from_relation="NMOD:poss",
        to_relation="nmod:poss",
    )


def test_a_repair_cannot_record_a_rewrite_that_changed_nothing() -> None:
    """The constructor refuses it, so no count can be inflated by a no-op."""
    with pytest.raises(ValueError, match="must change the relation"):
        RelationRepair(
            kind=RelationRepairKind.RELATION_CASE,
            word="foo",
            from_relation="nsubj",
            to_relation="nsubj",
        )


def test_a_repair_cannot_record_a_rewrite_to_a_non_ud_relation() -> None:
    """A repair that leaves the label invalid is not a repair."""
    with pytest.raises(ValueError, match="must produce a UD relation"):
        RelationRepair(
            kind=RelationRepairKind.UNKNOWN_RELATION,
            word="foo",
            from_relation="iob",
            to_relation="notarelation",
        )


def test_udword_leaves_a_relation_alone() -> None:
    """The model fills defaults only: repairing relations is not its job."""
    w = UdWord.model_validate(
        {
            "id": 1,
            "text": "hello",
            "lemma": "hello",
            "upos": "INTJ",
            "head": 0,
            "deprel": "root",
        }
    )
    assert w.deprel == "root"


def test_udword_extra_fields_preserved() -> None:
    """Extra fields (ner, start_char, etc.) should be kept."""
    w = UdWord.model_validate(
        {
            "id": 1,
            "text": "Paris",
            "lemma": "Paris",
            "upos": "PROPN",
            "head": 0,
            "deprel": "root",
            "ner": "B-LOC",
            "start_char": 0,
            "end_char": 5,
        }
    )
    d = w.model_dump()
    assert d["ner"] == "B-LOC"
    assert d["start_char"] == 0


def test_udword_tuple_id_coerced_to_list() -> None:
    """Stanza uses tuples for Range IDs; Pydantic should accept them."""
    w = UdWord.model_validate({"id": (2, 3), "text": "au"})
    assert w.id == [2, 3]
    assert w.lemma == ""


# --- Integration: RepairedSentence ---


def test_repaired_sentence_fills_every_word() -> None:
    """Every token of a sentence is validated, Range parents included."""
    first = RepairedSentence(
        [
            {"id": 1, "text": "je", "upos": "PRON", "head": 2, "deprel": "nsubj"},
            {"id": [2, 3], "text": "au"},
            {"id": 2, "text": "à", "upos": "ADP", "head": 4, "deprel": "case"},
            {"id": 3, "text": "le", "upos": "DET", "head": 4, "deprel": "det"},
        ]
    )
    second = RepairedSentence([{"id": 1, "text": "oui", "head": 0, "deprel": "root"}])

    # Sentence 1: regular token missing lemma
    assert first.words[0]["lemma"] == "je"
    # Sentence 1: Range token
    assert first.words[1]["lemma"] == ""
    assert first.words[1]["upos"] == "X"
    # Sentence 1: component tokens missing lemma
    assert first.words[2]["lemma"] == "à"
    assert first.words[3]["lemma"] == "le"
    # Sentence 2: missing lemma and upos
    assert second.words[0]["lemma"] == "oui"
    assert second.words[0]["upos"] == "X"
    # Nothing needed repairing, and the sentences say so rather than leaving
    # the question to a log.
    assert first.repairs == ()
    assert second.repairs == ()


def test_iob_relation_is_repaired_to_iobj_and_reported() -> None:
    """Stanza's Italian model emits `iob`, which is not a UD relation.

    Reproduced live on stanza 1.13.0, 2026-07-28, running the Italian
    pipeline directly on "attenzione ."::

        id=1 text='attenzi' upos=VERB head=0 deprel='root'
        id=2 text='ne'      upos=PRON head=1 deprel='iob'
        id=3 text='.'       upos=PUNCT head=1 deprel='punct'

    Universal Dependencies defines `iobj`, never `iob`. Passing it through
    put `2|1|IOB` into `%gra` across the published corpora, where it sat
    undetected until chatter's E761 relation-vocabulary rule shipped in
    v0.4.0. CLAN CHECK never flagged it.
    """
    relation, repair = _repaired_relation("iob", "ne")
    assert relation == "iobj"
    assert repair == RelationRepair(
        kind=RelationRepairKind.RELATION_ALIAS,
        word="ne",
        from_relation="iob",
        to_relation="iobj",
    )


def test_unknown_relation_falls_back_to_dep_and_is_reported() -> None:
    """A relation outside the UD closed set must not reach `%gra` verbatim.

    The failure mode this prevents is silent pass-through: `iob` reached the
    corpora precisely because nothing validated the label against UD. An
    unrecognised relation degrades to `dep`, which is a real UD relation, and
    the degradation is reported rather than only logged.
    """
    relation, repair = _repaired_relation("notarelation", "foo")
    assert relation == "dep"
    assert repair == RelationRepair(
        kind=RelationRepairKind.UNKNOWN_RELATION,
        word="foo",
        from_relation="notarelation",
        to_relation="dep",
    )


def test_valid_ud_relations_pass_through_untouched() -> None:
    """Legitimate relations, including subtypes, must be preserved exactly.

    The corpora use many language-specific subtypes (`nmod:poss`,
    `acl:relcl`, ...). UD defines subtypes as open and language-specific, so
    only the HEAD is a closed set; over-eager normalisation here would
    corrupt far more data than the bug it fixes.
    """
    for deprel in (
        "iobj",
        "nsubj",
        "root",
        "punct",
        "expl",
        "discourse",
        "nmod:poss",
        "acl:relcl",
        "flat:foreign",
    ):
        assert _repaired_relation(deprel, "foo") == (deprel, None), (
            f"{deprel!r} must survive untouched, and report no repair"
        )
