"""The verb lemma of an English ``-ing`` word Stanza analysed as a noun.

Why this exists
---------------
The Rust finite-verb rescue (``crates/batchalign-transform/src/morphosyntax/
invariants/finite_verb_main_clause.rs``) repairs Stanza's possessive misreading
of ``the lady's washing dishes .``: ``'s`` becomes the copula and ``washing``
the present participle heading the clause. Stanza's analysis gave ``washing``
the NOUN lemma ``washing``, and the rescue used to keep it, emitting
``verb|washing``, a verb whose lemma is a noun's (and ``verb|barking`` for
``the dog's barking .``). The rescue runs in Rust, where no lemmatizer exists,
so the verb lemma has to come from here.

What it does
------------
For every English NOUN ending in ``-ing``, ask Stanza's own lemma processor
for the word's lemma under a VERB/VBG reading, on a one-word pretagged
document. That is the dictionary and, for unseen words, the neural model the
pipeline itself uses (``barking`` gives ``bark``), so no rule of ours guesses
a stem. The answer travels in the UD MISC field as
``VerbReadingLemma=<lemma>``, which Rust reads only through the rescue's typed
accessor and nowhere else; a word without it is not rescued.

The condition is a deliberate superset of the rescue's candidates: every
NOUN ending in ``-ing``, with nothing about length or the possessive context.
This is evidence about the word, computed where the model is; which words the
rescue may promote is decided in Rust alone, so the two sides share only the
MISC key, not a rule that could drift.
"""

from __future__ import annotations

from typing import Any

# Must match `VERB_READING_LEMMA_MISC_KEY` in finite_verb_main_clause.rs.
VERB_READING_LEMMA_MISC_KEY = "VerbReadingLemma"

# The tags the pretagged probe gives each word: the reading whose lemma we want.
_VERB_UPOS = "VERB"
_PRESENT_PARTICIPLE_XPOS = "VBG"
# Stanza writes `_` for a lemma it could not produce.
_NO_LEMMA = "_"


def _is_ing_noun(word: dict[str, Any]) -> bool:
    text = word.get("text")
    return (
        word.get("upos") == "NOUN"
        and isinstance(text, str)
        and text.lower().endswith("ing")
    )


def annotate_verb_reading_lemmas(
    sentences: list[list[dict[str, Any]]], nlp: Any
) -> None:
    """Attach ``VerbReadingLemma`` to every English ``-ing`` NOUN.

    ``sentences`` is ``doc.to_dict()`` of the document the English pipeline
    ``nlp`` produced; its ``lemma`` processor answers, once per distinct
    spelling. Adds a ``misc`` entry to the matching word dicts in place.
    """
    targets = [
        word for sentence in sentences for word in sentence if _is_ing_noun(word)
    ]
    if not targets:
        return

    from stanza.models.common.doc import Document

    spellings = sorted({word["text"] for word in targets})
    probe = Document(
        [
            [
                {
                    "id": 1,
                    "text": text,
                    "upos": _VERB_UPOS,
                    "xpos": _PRESENT_PARTICIPLE_XPOS,
                }
            ]
            for text in spellings
        ]
    )
    lemmatized = nlp.processors["lemma"].process(probe)
    verb_lemma = {
        text: sentence.words[0].lemma
        for text, sentence in zip(spellings, lemmatized.sentences, strict=True)
    }
    for word in targets:
        lemma = verb_lemma[word["text"]]
        if lemma and lemma != _NO_LEMMA:
            entry = f"{VERB_READING_LEMMA_MISC_KEY}={lemma}"
            misc = word.get("misc")
            word["misc"] = entry if not misc else f"{misc}|{entry}"
