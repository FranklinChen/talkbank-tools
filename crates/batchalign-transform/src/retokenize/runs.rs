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
//! A run is a maximal sequence of plain, retokenizable words in the same
//! container with no other main-tier content between them. Every other word
//! the `%mor` list counts (an annotated word, each word of a replacement, a
//! tag separator) is a run of its own, and every container edge ends a run.
//! That is never coarser than the boundaries the rebuild checks
//! (`RetokenizeContext::at_boundary`), so tokens segmented within runs never
//! cross one. The rebuild's check stays as the admission check on model
//! output, which this module does not replace.
//!
//! The walk counts words by the rebuild walk's own rules (`should_retokenize`,
//! `excluded_from_mor`, `is_tag_marker_separator`). Its total must equal the
//! extractor's word count; [`WordRuns::of`] refuses otherwise, so a counting
//! disagreement fails the utterance loudly instead of misplacing a boundary.

use std::num::NonZeroUsize;

use talkbank_model::alignment::helpers::is_tag_marker_separator;
use talkbank_model::model::{BracketedItem, UtteranceContent};

use super::rebuild::{excluded_from_mor, should_retokenize};

/// One utterance's `%mor` words partitioned into runs, in order. Only
/// [`WordRuns::of`] (from the main tier) and [`WordRuns::unmergeable`] make
/// one, so the lengths always partition exactly the words they describe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordRuns(Vec<NonZeroUsize>);

/// The run walk counted a different number of words than extraction did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("the run walk counted {walked} words where extraction found {extracted}")]
pub struct RunsDisagree {
    /// Words the run walk counted.
    pub walked: usize,
    /// Words chatter's extractor produced for the `%mor` list.
    pub extracted: usize,
}

impl WordRuns {
    /// The runs of a main tier whose extracted `%mor` word list has
    /// `extracted` words.
    pub fn of(content: &[UtteranceContent], extracted: usize) -> Result<Self, RunsDisagree> {
        let mut walk = RunWalk::default();
        walk.utterance(content);
        walk.boundary();
        let walked = walk.runs.iter().map(|run| run.get()).sum();
        if walked == extracted {
            Ok(Self(walk.runs))
        } else {
            Err(RunsDisagree { walked, extracted })
        }
    }

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

    /// How many words the runs cover.
    pub fn word_count(&self) -> usize {
        self.lengths().sum()
    }
}

/// The walk's state: the closed runs and the length of the one still open.
#[derive(Default)]
struct RunWalk {
    runs: Vec<NonZeroUsize>,
    open: usize,
}

impl RunWalk {
    /// A plain retokenizable word extends the open run.
    fn joinable_word(&mut self) {
        self.open += 1;
    }

    /// Anything else ends the open run.
    fn boundary(&mut self) {
        if let Some(run) = NonZeroUsize::new(self.open) {
            self.runs.push(run);
        }
        self.open = 0;
    }

    /// `count` counted words of which each stands alone.
    fn lone_words(&mut self, count: usize) {
        self.boundary();
        self.runs
            .extend(std::iter::repeat_n(NonZeroUsize::MIN, count));
    }

    /// A container's items, with a boundary at both edges.
    fn container(&mut self, items: &[BracketedItem]) {
        self.boundary();
        self.bracketed(items);
        self.boundary();
    }

    fn utterance(&mut self, content: &[UtteranceContent]) {
        for item in content {
            match item {
                UtteranceContent::Word(word) if should_retokenize(word) => self.joinable_word(),
                UtteranceContent::AnnotatedWord(annotated) => {
                    if excluded_from_mor(&annotated.scoped_annotations) {
                        self.boundary();
                    } else {
                        self.lone_words(1);
                    }
                }
                UtteranceContent::ReplacedWord(replaced) => {
                    if excluded_from_mor(replaced.scoped_annotations.as_slice()) {
                        self.boundary();
                    } else {
                        self.lone_words(replaced.replacement.words.len());
                    }
                }
                UtteranceContent::Group(group) => self.container(&group.content.content),
                UtteranceContent::PhoGroup(pho) => self.container(&pho.content.content),
                UtteranceContent::SinGroup(sin) => self.container(&sin.content.content),
                UtteranceContent::Quotation(quot) => self.container(&quot.content.content),
                UtteranceContent::AnnotatedGroup(annotated) => {
                    if excluded_from_mor(&annotated.scoped_annotations) {
                        self.boundary();
                    } else {
                        self.container(&annotated.inner.content.content);
                    }
                }
                UtteranceContent::AnnotatedQuotation(annotated) => {
                    if excluded_from_mor(&annotated.scoped_annotations) {
                        self.boundary();
                    } else {
                        self.container(&annotated.inner.content.content);
                    }
                }
                UtteranceContent::Separator(sep) if is_tag_marker_separator(sep) => {
                    self.lone_words(1);
                }
                // Enumerated, not `_`, like the rebuild walk's passthrough
                // arm: each is content between words and counts none.
                UtteranceContent::Word(_)
                | UtteranceContent::OrphanAnnotation(_)
                | UtteranceContent::Event(_)
                | UtteranceContent::AnnotatedEvent(_)
                | UtteranceContent::Pause(_)
                | UtteranceContent::Retrace(_)
                | UtteranceContent::AnnotatedRetrace(_)
                | UtteranceContent::Action(_)
                | UtteranceContent::AnnotatedAction(_)
                | UtteranceContent::Freecode(_)
                | UtteranceContent::Separator(_)
                | UtteranceContent::OverlapPoint(_)
                | UtteranceContent::InternalBullet(_)
                | UtteranceContent::LongFeatureBegin(_)
                | UtteranceContent::LongFeatureEnd(_)
                | UtteranceContent::UnderlineBegin(_)
                | UtteranceContent::UnderlineEnd(_)
                | UtteranceContent::NonvocalBegin(_)
                | UtteranceContent::NonvocalEnd(_)
                | UtteranceContent::NonvocalSimple(_)
                | UtteranceContent::OtherSpokenEvent(_) => self.boundary(),
            }
        }
    }

    fn bracketed(&mut self, items: &[BracketedItem]) {
        for item in items {
            match item {
                BracketedItem::Word(word) if should_retokenize(word) => self.joinable_word(),
                BracketedItem::AnnotatedWord(annotated) => {
                    if excluded_from_mor(&annotated.scoped_annotations) {
                        self.boundary();
                    } else {
                        self.lone_words(1);
                    }
                }
                BracketedItem::ReplacedWord(replaced) => {
                    if excluded_from_mor(replaced.scoped_annotations.as_slice()) {
                        self.boundary();
                    } else {
                        self.lone_words(replaced.replacement.words.len());
                    }
                }
                BracketedItem::Group(group) => self.container(&group.content.content),
                BracketedItem::PhoGroup(pho) => self.container(&pho.content.content),
                BracketedItem::SinGroup(sin) => self.container(&sin.content.content),
                BracketedItem::Quotation(quot) => self.container(&quot.content.content),
                BracketedItem::AnnotatedGroup(annotated) => {
                    if excluded_from_mor(&annotated.scoped_annotations) {
                        self.boundary();
                    } else {
                        self.container(&annotated.inner.content.content);
                    }
                }
                BracketedItem::AnnotatedQuotation(annotated) => {
                    if excluded_from_mor(&annotated.scoped_annotations) {
                        self.boundary();
                    } else {
                        self.container(&annotated.inner.content.content);
                    }
                }
                BracketedItem::Separator(sep) if is_tag_marker_separator(sep) => {
                    self.lone_words(1);
                }
                BracketedItem::Word(_)
                | BracketedItem::OrphanAnnotation(_)
                | BracketedItem::Event(_)
                | BracketedItem::AnnotatedEvent(_)
                | BracketedItem::Pause(_)
                | BracketedItem::Action(_)
                | BracketedItem::AnnotatedAction(_)
                | BracketedItem::Retrace(_)
                | BracketedItem::AnnotatedRetrace(_)
                | BracketedItem::OverlapPoint(_)
                | BracketedItem::Separator(_)
                | BracketedItem::InternalBullet(_)
                | BracketedItem::Freecode(_)
                | BracketedItem::LongFeatureBegin(_)
                | BracketedItem::LongFeatureEnd(_)
                | BracketedItem::UnderlineBegin(_)
                | BracketedItem::UnderlineEnd(_)
                | BracketedItem::NonvocalBegin(_)
                | BracketedItem::NonvocalEnd(_)
                | BracketedItem::NonvocalSimple(_)
                | BracketedItem::OtherSpokenEvent(_) => self.boundary(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The runs of `body`, checked against the extractor's word count.
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
        let mut words = Vec::new();
        crate::extract::collect_utterance_content(
            &utterance.main.content.content,
            talkbank_model::alignment::helpers::PositionalDomain::Mor,
            &mut words,
        );
        WordRuns::of(&utterance.main.content.content, words.len())
            .unwrap_or_else(|error| panic!("{body:?}: {error}"))
            .lengths()
            .collect()
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

    #[test]
    fn a_walk_that_disagrees_with_extraction_is_refused() {
        let parser = talkbank_parser::TreeSitterParser::new().expect("parser");
        let chat = crate::parse_and_validate_with_parser(
            &parser,
            "@UTF8\n@Begin\n@Languages:\tyue\n@Participants:\tCHI Target_Child\n\
             @ID:\tyue|test|CHI|||||Target_Child|||\n*CHI:\ta b c .\n@End\n",
            talkbank_model::ParseValidateOptions::default(),
        )
        .expect("fixture parses");
        let content = &chat
            .utterances()
            .next()
            .expect("utterance")
            .main
            .content
            .content;
        assert_eq!(
            WordRuns::of(content, 2),
            Err(RunsDisagree {
                walked: 3,
                extracted: 2
            })
        );
    }

    #[test]
    fn unmergeable_words_are_each_their_own_run() {
        let runs = WordRuns::unmergeable(3);
        assert_eq!(runs.lengths().collect::<Vec<_>>(), vec![1, 1, 1]);
        assert_eq!(runs.word_count(), 3);
    }
}
