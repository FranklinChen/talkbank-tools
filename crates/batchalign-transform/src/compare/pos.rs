//! Whether a compared document carries part-of-speech tags, decided once for
//! the whole file.
//!
//! `%mor` is a per-utterance tier, so "this word has no tag" and "this document
//! tags nothing at all" reach a consumer as the same `None`. They are not the
//! same fact, and the difference decides two things no per-word absence can:
//! which side's tag a matched pair reports, and whether the punctuation filter
//! is able to consult a tag at all.
//!
//! # The defect this replaces
//!
//! compare morphotags the MAIN transcript itself and reads the gold companion
//! off disk as it is, so the ordinary case is a tagged main side and an
//! untagged gold one. Every matched word took its part of speech from the gold
//! form, found none there, and reported the literal `?`: the `%xsmor` tier came
//! out as a row of question marks and every per-POS metric landed in one `?`
//! bucket, which is the whole of the per-POS breakdown for such a run.
//!
//! batchalign2 has the same defect, and is deliberately not the standard
//! matched here. Measured 2026-09-16 against `TalkBank/batchalign2` `master`
//! commit `d8bb0cd0`, file blob `37270401`,
//! `batchalign/pipelines/analysis/compare.py` lines 355-359 and 544-588: its
//! `_get_pos` returns `form.morphology[0].pos.upper()`, or the literal `"?"`
//! whenever the form is absent or carries no morphology, and it has no
//! file-level notion of a tagged side anywhere. Matching that would reproduce
//! the defect, so BA3 differs on purpose and the difference is recorded on the
//! compare page.
//!
//! # Two parity details that are unchanged
//!
//! BA3 reads ONE entry per `%mor` item, `item.main.pos`, and uppercases it,
//! which is what `morphology[0].pos.upper()` amounts to; and a form whose
//! morphology list is empty is untagged rather than tagged with nothing, so a
//! document in which every list is empty is [`GoldPos::Untagged`].

use talkbank_model::model::{ChatFile, Line};

use super::engine::is_punct_or_filler;

/// Per-utterance, per-word `%mor` labels for one document.
///
/// The field is private and [`GoldPos::of`] is the only thing that builds one,
/// so [`GoldPos::Tagged`] cannot be spelled for a document that tags nothing:
/// such a document is [`GoldPos::Untagged`], and the rules below are total over
/// the two.
pub(in crate::compare) struct MorPosLabels(Vec<Vec<Option<String>>>);

/// The part-of-speech evidence one compared document carries.
pub(in crate::compare) enum GoldPos {
    /// At least one `%mor` item exists somewhere in the document, so a word
    /// without a tag genuinely lacks one and a `PUNCT` tag is a fact the
    /// document states.
    Tagged(MorPosLabels),
    /// No `%mor` item exists anywhere in the document. No word in it has a part
    /// of speech, and no word in it can be recognized as punctuation by tag.
    Untagged,
}

/// The gold document's tag for a matched word.
pub(in crate::compare) struct GoldTag<'a>(pub(in crate::compare) Option<&'a str>);

/// The main document's tag for the same matched word.
///
/// A separate type from [`GoldTag`] for one reason: the two are the same
/// primitive carrying opposite meanings into [`GoldPos::pos_for_match`], and
/// passing them the wrong way round is the one mistake that call site can make.
/// Now the compiler refuses it instead of a reviewer catching it.
pub(in crate::compare) struct MainTag<'a>(pub(in crate::compare) Option<&'a str>);

impl GoldPos {
    /// Decide, once, what part-of-speech evidence this document carries.
    pub(in crate::compare) fn of(chat_file: &ChatFile) -> Self {
        let mut labels: Vec<Vec<Option<String>>> = Vec::new();
        for line in &chat_file.lines {
            if let Line::Utterance(utt) = line {
                labels.push(
                    utt.mor_tier()
                        .map(|mor| {
                            mor.items()
                                .iter()
                                .map(|item| Some(item.main.pos.to_string().to_uppercase()))
                                .collect()
                        })
                        .unwrap_or_default(),
                );
            }
        }

        // An empty `%mor` tier tags nothing, so a document holding only empty
        // ones is Untagged rather than Tagged-with-no-items: the two behave
        // identically per word, and collapsing them here keeps the file-level
        // question answerable by the variant alone.
        if labels.iter().any(|utterance| !utterance.is_empty()) {
            Self::Tagged(MorPosLabels(labels))
        } else {
            Self::Untagged
        }
    }

    /// The label this document records for one word, if it records one.
    fn label(&self, utterance: usize, word: usize) -> Option<&str> {
        match self {
            Self::Tagged(MorPosLabels(labels)) => labels
                .get(utterance)
                .and_then(|utterance| utterance.get(word))
                .and_then(Option::as_deref),
            Self::Untagged => None,
        }
    }

    /// The part of speech to record for one word of this document.
    pub(in crate::compare) fn tag(&self, utterance: usize, word: usize) -> Option<String> {
        self.label(utterance, word).map(str::to_owned)
    }

    /// Whether the word at this position is left out of the comparison.
    ///
    /// batchalign2 asks this as a disjunction: the surface is in the
    /// punctuation sets, OR the form's part of speech is `PUNCT`. The second
    /// half can only ever fire on a document that HAS tags, which is why the
    /// choice is made here, per document, rather than per form.
    ///
    /// The consequence is worth stating because it reaches further than the
    /// reported column. This filter decides what enters the word ALIGNMENT, so
    /// on an untagged side a token that is punctuation only by its tag stays in
    /// and is aligned as an ordinary word. Nothing can recover the tag for a
    /// document that has none; what the file-level decision buys is that the
    /// weaker rule is now a stated property of an untagged document instead of
    /// an accident of a per-form `None`.
    pub(in crate::compare) fn excludes_from_comparison(
        &self,
        utterance: usize,
        word: usize,
        text: &str,
    ) -> bool {
        match self {
            Self::Tagged(_) => {
                is_punct_or_filler(text)
                    || self
                        .label(utterance, word)
                        .is_some_and(|label| label.eq_ignore_ascii_case("PUNCT"))
            }
            Self::Untagged => is_punct_or_filler(text),
        }
    }

    /// The part of speech a MATCHED pair reports, read off the gold side's
    /// own evidence.
    ///
    /// Tagged: the gold tag, which is the point of running compare at all. When
    /// the two transcripts disagree about a word they both contain, the
    /// gold-standard tag is what the reviewer needs to see, and a gold word
    /// with no item of its own still reports nothing rather than borrowing.
    ///
    /// Untagged: the main tag. A match means the two sides hold the same word,
    /// the main side was morphotagged by this very run, and that tag therefore
    /// describes the matched word rather than a different one. It is the only
    /// tag in existence for that word, and reporting it is what turns a per-POS
    /// breakdown of nothing but `?` into the breakdown the CSV promises. A
    /// deletion has no main word at all, so it keeps reporting nothing, which
    /// is the honest answer: no side tagged it.
    pub(in crate::compare) fn pos_for_match(
        &self,
        gold: GoldTag<'_>,
        main: MainTag<'_>,
    ) -> Option<String> {
        match self {
            Self::Tagged(_) => gold.0.map(str::to_owned),
            Self::Untagged => main.0.map(str::to_owned),
        }
    }
}
