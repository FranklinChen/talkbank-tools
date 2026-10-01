"""The worker's verb-reading evidence for the Rust finite-verb rescue.

A fake lemma processor stands in for Stanza's, so these run without models;
the golden ``test_preserve_mwt_end_to_end.py`` covers the real pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from batchalign.inference._english_verb_reading import (
    VERB_READING_LEMMA_MISC_KEY,
    annotate_verb_reading_lemmas,
)


def _word(text: str, upos: str, misc: str | None = None) -> dict[str, Any]:
    """One word as ``doc.to_dict()`` serializes it."""
    return {"id": 1, "text": text, "upos": upos, "misc": misc}


@dataclass
class _FakeLemma:
    """Answers from a table and records which (text, upos) it was asked about."""

    answers: dict[str, str]
    asked: list[tuple[str, str]] = field(default_factory=list)

    def process(self, probe: Any) -> Any:
        for sentence in probe.sentences:
            for word in sentence.words:
                self.asked.append((word.text, word.upos))
                word.lemma = self.answers.get(word.text, "_")
        return probe


@dataclass
class _FakeNlp:
    lemma: _FakeLemma

    @property
    def processors(self) -> dict[str, _FakeLemma]:
        return {"lemma": self.lemma}


def test_only_ing_nouns_are_asked_about_and_asked_as_verbs() -> None:
    lemma = _FakeLemma({"barking": "bark"})
    sentences = [
        [_word("dog", "NOUN"), _word("barking", "NOUN"), _word("going", "VERB")]
    ]
    annotate_verb_reading_lemmas(sentences, _FakeNlp(lemma))
    assert lemma.asked == [("barking", "VERB")]
    assert sentences[0][1]["misc"] == f"{VERB_READING_LEMMA_MISC_KEY}=bark"
    assert sentences[0][0]["misc"] is None


def test_an_existing_misc_entry_is_kept() -> None:
    sentences = [[_word("washing", "NOUN", misc="SpaceAfter=No")]]
    annotate_verb_reading_lemmas(sentences, _FakeNlp(_FakeLemma({"washing": "wash"})))
    assert (
        sentences[0][0]["misc"] == f"SpaceAfter=No|{VERB_READING_LEMMA_MISC_KEY}=wash"
    )


def test_no_lemma_means_no_annotation() -> None:
    sentences = [[_word("glorping", "NOUN")]]
    annotate_verb_reading_lemmas(sentences, _FakeNlp(_FakeLemma({})))
    assert sentences[0][0]["misc"] is None


def test_a_document_without_candidates_never_reaches_the_lemmatizer() -> None:
    lemma = _FakeLemma({})
    annotate_verb_reading_lemmas([[_word("dog", "NOUN")]], _FakeNlp(lemma))
    assert lemma.asked == []


def test_each_spelling_is_asked_once() -> None:
    lemma = _FakeLemma({"barking": "bark"})
    sentences = [[_word("barking", "NOUN")], [_word("barking", "NOUN")]]
    annotate_verb_reading_lemmas(sentences, _FakeNlp(lemma))
    assert lemma.asked == [("barking", "VERB")]
    assert all(s[0]["misc"] == f"{VERB_READING_LEMMA_MISC_KEY}=bark" for s in sentences)
