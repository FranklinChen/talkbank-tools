//! Where a tokenizer may merge words: the runs of one utterance's `%mor` words.
//!
//! A segmenter that sees an utterance as one string merges across whatever
//! the string hides. Cantonese child speech is full of retraces
//! (`a b [/] b c`) and untranscribed material (`a xxx b`); PyCantonese given
//! the joined words merged characters on either side of them, and the
//! rebuild, which cannot write one word over two places, had to refuse the
//! whole utterance. The cure is at the producer: the request says where the
//! runs are, and the segmenter segments each run on its own, so a merged
//! token can only join words with nothing between them.
//!
//! Chatter decides the runs (`talkbank_transform::extract::collect_mor_runs`)
//! in the same walk that decides which items are `%mor` words, and returns
//! the words already partitioned. BA3 used to walk the main tier a second
//! time with its own copy of the word rules, and refused an utterance when
//! the two counts disagreed: BA3 0.8.8 refused whole files that way, over
//! annotated fragments and `xxx [*]` that the second walk counted and the
//! extractor rightly skipped. A [`WordRuns`] is now
//! taken from Chatter's runs by a total conversion, so the runs partition
//! the words by construction and no count can disagree.
//!
//! A run is never coarser than the boundaries the rebuild checks
//! (`RetokenizeContext::at_boundary`), so tokens segmented within runs never
//! cross one. The rebuild's check stays as the admission check on model
//! output, which this module does not replace.

use std::num::NonZeroUsize;

use talkbank_transform::extract::MorRuns;

/// One utterance's `%mor` words partitioned into runs, in order. Made only
/// from Chatter's runs ([`WordRuns::from`]) or as [`WordRuns::unmergeable`],
/// so the lengths always partition exactly the words they describe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordRuns(Vec<NonZeroUsize>);

impl From<&MorRuns> for WordRuns {
    /// The run lengths of Chatter's partition. Total: every run Chatter
    /// makes holds at least one word.
    fn from(runs: &MorRuns) -> Self {
        Self(runs.run_lengths().collect())
    }
}

impl WordRuns {
    /// `words` words of which none may merge with another: the honest runs
    /// of an item that has no authority to retokenize (a secondary-language
    /// span keeps its CHAT words as written).
    pub fn unmergeable(words: usize) -> Self {
        Self(std::iter::repeat_n(NonZeroUsize::MIN, words).collect())
    }

    /// The run lengths, in order; they sum to the item's word count.
    pub fn lengths(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().map(|run| run.get())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    /// The runs Chatter gives `body`'s one utterance: BA3's boundary contract,
    /// checked where the runs enter BA3.
    fn runs_of(body: &str) -> Vec<usize> {
        let source = format!(
            "@UTF8\n@Begin\n@Languages:\tyue\n@Participants:\tCHI Target_Child\n\
             @ID:\tyue|test|CHI|||||Target_Child|||\n*CHI:\t{body}\n@End\n"
        );
        let parser = talkbank_parser::TreeSitterParser::new().expect("parser");
        let chat = crate::parse_and_validate_with_parser(
            &parser,
            &source,
            talkbank_model::ParseValidateOptions::default(),
        )
        .expect("fixture parses");
        let utterance = chat.utterances().next().expect("one utterance");
        let runs = crate::extract::collect_mor_runs(&utterance.main.content.content);
        WordRuns::from(&runs).lengths().collect()
    }

    /// The shapes from real Cantonese child speech that the joined-string
    /// segmenter merged across (each written here with synthetic words).
    #[test]
    fn retraces_untranscribed_material_and_brackets_end_runs() {
        // A retraced word is not in the %mor list, and ends the run before it.
        assert_eq!(runs_of("a b [/] b c d ."), vec![1, 3]);
        // A retraced group likewise; the correction starts a new run.
        assert_eq!(runs_of("a <b c> [/] b c d ."), vec![1, 3]);
        // Untranscribed material is not a %mor word, and separates its neighbours.
        assert_eq!(runs_of("a b xxx c ."), vec![2, 1]);
        // Pauses and events separate too.
        assert_eq!(runs_of("a (.) b &=laughs c ."), vec![1, 1, 1]);
        // A group's words form their own run inside its brackets.
        assert_eq!(runs_of("a <b c> [!] d ."), vec![1, 2, 1]);
        // Uninterrupted speech is one run.
        assert_eq!(runs_of("a b c d ."), vec![4]);
    }

    /// A replacement's words and an error-marked word stand alone: the
    /// rebuild expands each scoped word on its own.
    #[test]
    fn scoped_words_are_runs_of_their_own() {
        assert_eq!(runs_of("a b [: c d] e f ."), vec![1, 1, 1, 2]);
        assert_eq!(runs_of("a b [*] c d ."), vec![1, 1, 2]);
    }

    /// A scoped word that is not a %mor word (a fragment, a filler,
    /// untranscribed material) counts none and ends the run, as the
    /// extractor and the rebuild treat it: annotating it does not make it a
    /// word. Counting it once stopped every file holding such a word (an
    /// annotated fragment or `xxx`) as an internal fault.
    #[test]
    fn a_scoped_word_outside_the_mor_list_counts_none() {
        assert_eq!(runs_of("a &+w [=! laughs] ."), vec![1]);
        assert_eq!(runs_of("a b xxx [*] ."), vec![2]);
        assert_eq!(runs_of("a &-um [=! shrugs] b ."), vec![1, 1]);
        assert_eq!(runs_of("a <b &+c [=! laughs]> [!] d ."), vec![1, 1, 1]);
    }

    #[test]
    fn unmergeable_words_are_each_their_own_run() {
        let runs = WordRuns::unmergeable(3);
        assert_eq!(runs.lengths().collect::<Vec<_>>(), vec![1, 1, 1]);
    }
}
