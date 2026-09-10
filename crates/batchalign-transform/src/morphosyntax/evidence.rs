//! Transcriber evidence about an utterance that Stanza never sees.
//!
//! Stanza receives the cleaned words and the terminator. The transcriber
//! wrote more than that: CHAT pauses (`(.)`, `(..)`, timed) mark prosodic
//! breaks, and a word set off by a break on both sides (or by a break and
//! the edge of the utterance) is a discourse element, not a constituent of
//! the clause next to it. `put the lady on the chair (.) okay ?` is a request
//! plus a tag; `you okay ?` is a predicate. Only the pause tells them apart,
//! and only the CHAT AST has the pause.
//!
//! This module reads that evidence off the utterance, aligned to the words
//! the morphosyntax payload was built from, so a rewrite at the post-depparse
//! stage can use it without re-walking the tree.

use talkbank_model::WriteChat;
use talkbank_model::alignment::helpers::{
    ContentItem, TierDomain, counts_for_tier, is_tag_marker_separator, walk_content,
};
use talkbank_model::model::UtteranceContent;
use talkbank_transform::extract::ExtractedWord;

/// Whether a transcriber-marked break (a pause or the utterance edge) sits on
/// each side of one payload word.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WordIsolation {
    /// A pause precedes the word, or it is the first word.
    pub break_before: bool,
    /// A pause follows the word, or it is the last word.
    pub break_after: bool,
}

impl WordIsolation {
    /// Set off on both sides: the shape of a tag or a stand-alone response.
    pub fn is_isolated(self) -> bool {
        self.break_before && self.break_after
    }
}

/// Per-word transcriber evidence for one utterance, in payload word order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UtteranceEvidence {
    isolation: Vec<WordIsolation>,
}

impl UtteranceEvidence {
    /// No evidence at all: every word reads as unbroken on both sides, so no
    /// evidence-driven rewrite can fire. For callers that have no utterance.
    pub fn none(word_count: usize) -> Self {
        Self {
            isolation: vec![WordIsolation::default(); word_count],
        }
    }

    /// Read the breaks off `content`, aligned to `words`, the payload's
    /// extracted words for the same utterance.
    ///
    /// The walk is the same Mor-domain descent the extractor used, so its
    /// word-like items come in payload order; each is matched to the next
    /// payload word by cleaned text. A word-like item the payload did not
    /// keep (`xxx`, an omitted `0word`, a word under an alignment-ignored
    /// annotation) still occupies a slot in the speech: it is not a break,
    /// and a word next to it is not at the utterance edge. A separator the
    /// payload did keep (a comma or a CHAT tag marker) is a break by
    /// definition. A genuine mismatch leaves every later word unbroken,
    /// which is the safe direction: no rewrite fires on words whose evidence
    /// could not be placed.
    pub fn from_utterance(content: &[UtteranceContent], words: &[ExtractedWord]) -> Self {
        let mut placer = Placer {
            isolation: vec![WordIsolation::default(); words.len()],
            words,
            next: 0,
            pending_break: true, // the utterance edge precedes word 0
            slot_after_last: false,
        };
        walk_content(content, Some(TierDomain::Mor), &mut |item| match item {
            ContentItem::Word(word) => placer.place(Kept::from_word(word)),
            ContentItem::ReplacedWord(replaced) => {
                if replaced.replacement.words.is_empty() {
                    placer.place(Kept::from_word(&replaced.word));
                } else {
                    for word in &replaced.replacement.words {
                        placer.place(Kept::from_word(word));
                    }
                }
            }
            ContentItem::Separator(separator) => {
                // A separator is a break. The extractor keeps the tag-marker
                // kinds as `%mor` items, so those also occupy a payload slot.
                if is_tag_marker_separator(separator) {
                    placer.place(Kept::Separator(separator.to_chat_string()));
                } else {
                    placer.pending_break = true;
                }
            }
            ContentItem::Pause(_) => placer.pending_break = true,
            ContentItem::Event(_)
            | ContentItem::Action(_)
            | ContentItem::OverlapPoint(_)
            | ContentItem::OtherSpokenEvent(_)
            | ContentItem::Freecode(_)
            | ContentItem::InternalBullet(_)
            | ContentItem::LongFeatureBegin(_)
            | ContentItem::LongFeatureEnd(_)
            | ContentItem::UnderlineBegin(_)
            | ContentItem::UnderlineEnd(_)
            | ContentItem::NonvocalBegin(_)
            | ContentItem::NonvocalEnd(_)
            | ContentItem::NonvocalSimple(_) => {}
        });
        // The utterance edge follows the last payload word only if nothing
        // word-like came after it.
        if placer.next == words.len()
            && !placer.slot_after_last
            && let Some(last) = placer.isolation.last_mut()
        {
            last.break_after = true;
        }
        Self {
            isolation: placer.isolation,
        }
    }

    /// Evidence assembled by a caller that already knows the breaks (tests,
    /// or a producer that read them another way).
    pub fn from_isolation(isolation: Vec<WordIsolation>) -> Self {
        Self { isolation }
    }

    /// The evidence for payload word `index`, if the utterance has that many.
    pub fn isolation(&self, index: usize) -> Option<WordIsolation> {
        self.isolation.get(index).copied()
    }

    /// Number of words the evidence covers.
    pub fn len(&self) -> usize {
        self.isolation.len()
    }

    /// Whether the evidence covers no words.
    pub fn is_empty(&self) -> bool {
        self.isolation.is_empty()
    }
}

/// One word-like item of the walk, classified the way the extractor
/// classifies it: does it occupy a payload slot, and is it a break.
enum Kept<'a> {
    /// A word the extractor keeps as a `%mor` item.
    Word(&'a str),
    /// A word the extractor drops (`xxx`, an omitted `0word`): speech that
    /// occupies no slot and is not a break.
    Dropped,
    /// A tag-marker separator: kept as a `%mor` item, and a break.
    Separator(String),
}

impl<'a> Kept<'a> {
    fn from_word(word: &'a talkbank_model::model::Word) -> Self {
        if counts_for_tier(word, TierDomain::Mor) {
            Self::Word(word.cleaned_text())
        } else {
            Self::Dropped
        }
    }
}

/// The alignment state of one walk over an utterance's content.
struct Placer<'w> {
    isolation: Vec<WordIsolation>,
    words: &'w [ExtractedWord],
    /// Index of the next payload word to place.
    next: usize,
    /// A break (pause, separator or the utterance edge) has been seen since
    /// the last placed word.
    pending_break: bool,
    /// A word-like item the payload did not keep came after the last placed
    /// word, so that word is not at the utterance edge.
    slot_after_last: bool,
}

impl Placer<'_> {
    /// Consume one word-like item of the walk.
    fn place(&mut self, item: Kept<'_>) {
        let is_break = matches!(item, Kept::Separator(_));
        // A break seen since the last placed word (or this item being one)
        // closes that word off, whatever this item turns out to be.
        if (self.pending_break || is_break)
            && let Some(previous) = self.next.checked_sub(1)
        {
            self.isolation[previous].break_after = true;
        }
        let text = match &item {
            Kept::Word(text) => *text,
            Kept::Separator(text) => text.as_str(),
            Kept::Dropped => {
                // Speech, not silence: nothing before it is at the edge.
                self.pending_break = false;
                self.slot_after_last = true;
                return;
            }
        };
        let expected = self.words.get(self.next);
        if !expected.is_some_and(|word| word.text == text) {
            // The extractor and this walk disagree about what the payload
            // holds. Fail closed (no later word gets evidence) and say so:
            // this is a defect, not a data condition.
            tracing::warn!(
                item = text,
                expected = expected.map(|w| w.text.as_str()),
                index = self.next,
                "utterance evidence: walk does not align with the payload"
            );
            self.pending_break = false;
            self.slot_after_last = true;
            return;
        }
        if self.pending_break || is_break {
            self.isolation[self.next].break_before = true;
        }
        self.pending_break = is_break;
        self.slot_after_last = false;
        self.next += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use talkbank_model::alignment::helpers::PositionalDomain;
    use talkbank_model::model::Line;
    use talkbank_parser::TreeSitterParser;

    fn evidence_for(main_tier: &str) -> (Vec<String>, UtteranceEvidence) {
        let chat = format!(
            "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tMOT Mother, CHI Target_Child\n\
             @ID:\teng|test|MOT|||||Mother|||\n@ID:\teng|test|CHI|||||Target_Child|||\n{main_tier}\n@End\n"
        );
        let parser = TreeSitterParser::new().expect("parser builds");
        let file = parser.parse_chat_file(&chat).expect_built();
        let utt = file
            .lines
            .as_slice()
            .iter()
            .find_map(|l| match l {
                Line::Utterance(u) => Some(u.as_ref()),
                _ => None,
            })
            .expect("one utterance");
        let mut words = Vec::new();
        talkbank_transform::extract::collect_utterance_content(
            &utt.main.content.content,
            PositionalDomain::Mor,
            &mut words,
        );
        let evidence = UtteranceEvidence::from_utterance(&utt.main.content.content, &words);
        (
            words.iter().map(|w| w.text.as_str().to_string()).collect(),
            evidence,
        )
    }

    fn isolated(evidence: &UtteranceEvidence) -> Vec<bool> {
        (0..evidence.len())
            .map(|i| {
                evidence
                    .isolation(i)
                    .map(WordIsolation::is_isolated)
                    .unwrap_or(false)
            })
            .collect()
    }

    #[test]
    fn a_pause_before_the_final_word_isolates_it() {
        let (words, evidence) = evidence_for("*MOT:\tput the lady on the chair (.) okay ?");
        assert_eq!(words, ["put", "the", "lady", "on", "the", "chair", "okay"]);
        assert_eq!(
            isolated(&evidence),
            [false, false, false, false, false, false, true]
        );
        assert_eq!(
            evidence.isolation(5),
            Some(WordIsolation {
                break_before: false,
                break_after: true
            })
        );
    }

    #[test]
    fn a_predicate_okay_is_not_isolated() {
        let (words, evidence) = evidence_for("*MOT:\tyou okay ?");
        assert_eq!(words, ["you", "okay"]);
        assert_eq!(isolated(&evidence), [false, false]);
        assert_eq!(
            evidence.isolation(0),
            Some(WordIsolation {
                break_before: true,
                break_after: false
            })
        );
    }

    #[test]
    fn an_initial_word_before_a_pause_is_isolated() {
        let (words, evidence) = evidence_for("*MOT:\tokay (.) I'm gonna get some medicine .");
        assert_eq!(words[0], "okay");
        assert!(
            evidence
                .isolation(0)
                .is_some_and(WordIsolation::is_isolated)
        );
        assert!(
            !evidence
                .isolation(1)
                .is_some_and(WordIsolation::is_isolated)
        );
    }

    #[test]
    fn a_one_word_utterance_is_isolated() {
        let (_, evidence) = evidence_for("*MOT:\tokay .");
        assert_eq!(isolated(&evidence), [true]);
    }

    #[test]
    fn fillers_the_payload_drops_do_not_break_the_alignment() {
        let (words, evidence) = evidence_for("*MOT:\t&-um (.) okay .");
        assert_eq!(words, ["okay"]);
        assert_eq!(isolated(&evidence), [true]);
    }

    #[test]
    fn a_word_the_payload_drops_is_speech_not_a_break() {
        // `xxx` has no `%mor` item, but it was said: `my` is not alone.
        let (words, evidence) = evidence_for("*CHI:\tmy xxx xxx .");
        assert_eq!(words, ["my"]);
        assert_eq!(isolated(&evidence), [false]);
        let (words, evidence) = evidence_for("*CHI:\toh (.) yyy my xxx .");
        assert_eq!(words, ["oh", "my"]);
        assert_eq!(isolated(&evidence), [true, false]);
    }

    #[test]
    fn a_comma_is_a_break_on_both_sides() {
        let (words, evidence) = evidence_for("*MOT:\tthat's the cat , right ?");
        assert_eq!(words, ["that's", "the", "cat", ",", "right"]);
        assert_eq!(isolated(&evidence), [false, false, false, true, true]);
    }

    #[test]
    fn none_has_no_breaks() {
        let evidence = UtteranceEvidence::none(3);
        assert_eq!(evidence.len(), 3);
        assert!(!isolated(&evidence).iter().any(|b| *b));
    }
}
