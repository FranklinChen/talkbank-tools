//! Word segmentation of recognized Han text, for languages that need it.
//!
//! A recognizer for Taiwanese returns one word per Han character, each with
//! its own timing. TalkBank transcripts write words, so the runs of
//! consecutive one-character words are segmented by a lexicon-based segmenter
//! (which one, and for which languages, is `WordSegmenterV2::for_language`;
//! the Python worker runs it) and each segmented word takes the span its
//! timed characters cover.
//!
//! ```text
//! monologue words --(PlannedSegmentation::new)--> NothingToSegment(words)
//!                                            \-> Runs(plan) --(texts)--> worker segmenter
//! plan + token lists --(SegmentationPlan::apply)--> monologue words, segmented
//! ```
//!
//! It runs on whole monologues ([`MonologueWords`]), before they are chunked,
//! so a long-turn split never cuts a run in two and turn length is counted in
//! words. The plan owns the monologues between the two steps, so tokens can
//! only be applied to the words they were computed from, and a plan always
//! has at least one run, so an empty request cannot reach the worker.
//! [`SegmentationPlan::apply`] refuses any token list that does not re-spell
//! its run exactly: a segmenter that dropped, added or changed a character
//! would otherwise move timings onto characters they do not belong to.

use super::prepare::is_cjk_ideograph;
use super::{AdmittedInterval, AsrWord, MonologueWords, WordKind, WordTiming};

/// One run of consecutive one-character Han words inside one monologue.
#[derive(Debug)]
struct Run {
    /// Index of the monologue the run is in.
    monologue: usize,
    /// Index of the run's first word in that monologue.
    start: usize,
    /// The run's characters, one per word, so its word count is its
    /// character count.
    text: String,
}

impl Run {
    /// How many words (one character each) the run spans.
    fn len(&self) -> usize {
        self.text.chars().count()
    }
}

/// What a search of the monologues for runs found.
#[derive(Debug)]
pub enum PlannedSegmentation {
    /// No run of one-character Han words: the monologues, unchanged.
    NothingToSegment(Vec<MonologueWords>),
    /// At least one run, awaiting the segmenter's tokens.
    Runs(SegmentationPlan),
}

/// Monologues awaiting segmentation, with the runs (at least one) to send to
/// the segmenter.
#[derive(Debug)]
pub struct SegmentationPlan {
    monologues: Vec<MonologueWords>,
    runs: Vec<Run>,
}

/// Why a segmenter's answer was refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SegmentationRefusal {
    /// The segmenter answered for a different number of runs.
    #[error("the segmenter returned {returned} token lists for {sent} runs")]
    RunCount {
        /// Runs sent.
        sent: usize,
        /// Token lists returned.
        returned: usize,
    },
    /// A token list does not re-spell its run.
    #[error("the segmenter re-spelled run {run:?} as {tokens:?}")]
    Respelled {
        /// The run's characters.
        run: String,
        /// The tokens returned for it.
        tokens: Vec<String>,
    },
}

/// A regular word that is exactly one Han character.
fn single_han(word: &AsrWord) -> bool {
    if word.kind != WordKind::Regular {
        return false;
    }
    let mut chars = word.text.as_str().chars();
    matches!((chars.next(), chars.next()), (Some(c), None) if is_cjk_ideograph(c))
}

/// A character's admitted, non-empty span, if it has one. Its bounds were
/// admitted when the words were prepared, so a refusal here cannot arise; a
/// character without a whole span contributes no time to its word.
fn timed_span(word: &AsrWord) -> Option<AdmittedInterval> {
    WordTiming::from_millis(word.start_ms, word.end_ms)
        .ok()?
        .interval()
}

impl PlannedSegmentation {
    /// Find the runs of one-character Han words in `monologues`. The only
    /// constructor of a [`SegmentationPlan`].
    pub fn new(monologues: Vec<MonologueWords>) -> Self {
        let mut runs = Vec::new();
        for (index, monologue) in monologues.iter().enumerate() {
            let words = &monologue.words;
            let mut position = 0;
            while position < words.len() {
                let start = position;
                while position < words.len() && single_han(&words[position]) {
                    position += 1;
                }
                if position == start {
                    position += 1;
                    continue;
                }
                runs.push(Run {
                    monologue: index,
                    start,
                    text: words[start..position]
                        .iter()
                        .map(|w| w.text.as_str())
                        .collect(),
                });
            }
        }
        if runs.is_empty() {
            Self::NothingToSegment(monologues)
        } else {
            Self::Runs(SegmentationPlan { monologues, runs })
        }
    }
}

impl SegmentationPlan {
    /// The runs' texts, in the order their token lists must come back.
    pub fn texts(&self) -> impl Iterator<Item = &str> {
        self.runs.iter().map(|run| run.text.as_str())
    }

    /// Replace each run by the words the segmenter found in it.
    ///
    /// `tokens` holds one list per run, in [`Self::texts`] order. Each list
    /// must re-spell its run exactly. A word's timing is the span covering
    /// its characters that have one; a word with none stays untimed.
    pub fn apply(
        self,
        tokens: Vec<Vec<String>>,
    ) -> Result<Vec<MonologueWords>, SegmentationRefusal> {
        let Self {
            mut monologues,
            runs,
        } = self;
        if tokens.len() != runs.len() {
            return Err(SegmentationRefusal::RunCount {
                sent: runs.len(),
                returned: tokens.len(),
            });
        }
        // Check every run before changing any monologue.
        for (run, list) in runs.iter().zip(&tokens) {
            let respelled = list.iter().flat_map(|token| token.chars());
            if list.iter().any(String::is_empty) || !respelled.eq(run.text.chars()) {
                return Err(SegmentationRefusal::Respelled {
                    run: run.text.clone(),
                    tokens: list.clone(),
                });
            }
        }
        // One forward pass per monologue: copy the words between runs and
        // merge each run's characters into its tokens.
        let mut pending = runs.into_iter().zip(tokens).peekable();
        for (index, monologue) in monologues.iter_mut().enumerate() {
            let mut old = std::mem::take(&mut monologue.words).into_iter();
            let mut words = Vec::with_capacity(old.len());
            let mut position = 0;
            while let Some((run, list)) = pending.next_if(|(run, _)| run.monologue == index) {
                words.extend(old.by_ref().take(run.start - position));
                position = run.start + run.len();
                for token in list {
                    let span = old
                        .by_ref()
                        .take(token.chars().count())
                        .filter_map(|character| timed_span(&character))
                        .reduce(AdmittedInterval::covering);
                    let (start_ms, end_ms) = span.map_or((None, None), |span| {
                        WordTiming::from_admitted(span).into_optional_millis()
                    });
                    words.push(AsrWord::new(token, start_ms, end_ms));
                }
            }
            words.extend(old);
            monologue.words = words;
        }
        Ok(monologues)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asr_postprocess::SpeakerIndex;

    type Timed<'a> = (&'a str, Option<(i64, i64)>);

    fn monologue(words: &[Timed<'_>]) -> MonologueWords {
        MonologueWords {
            speaker: SpeakerIndex(0),
            words: words
                .iter()
                .map(|(text, ms)| AsrWord::new(*text, ms.map(|m| m.0), ms.map(|m| m.1)))
                .collect(),
        }
    }

    fn texts(monologue: &MonologueWords) -> Vec<&str> {
        monologue.words.iter().map(|w| w.text.as_str()).collect()
    }

    fn plan(monologues: Vec<MonologueWords>) -> SegmentationPlan {
        match PlannedSegmentation::new(monologues) {
            PlannedSegmentation::Runs(plan) => plan,
            PlannedSegmentation::NothingToSegment(_) => panic!("test: expected runs"),
        }
    }

    /// Runs stop at anything that is not a one-character Han word.
    #[test]
    fn runs_are_consecutive_single_han_words() {
        let plan = plan(vec![monologue(&[
            ("臺", None),
            ("灣", None),
            (",", None),
            ("甜", None),
            ("ABC", None),
            ("今仔日", None),
            ("米", None),
        ])]);
        assert_eq!(plan.texts().collect::<Vec<_>>(), ["臺灣", "甜", "米"]);
    }

    /// With no run there is no plan, so nothing is sent to a segmenter.
    #[test]
    fn no_run_means_nothing_to_segment() {
        let words = vec![monologue(&[("今仔日", None), (".", None)])];
        match PlannedSegmentation::new(words.clone()) {
            PlannedSegmentation::NothingToSegment(unchanged) => assert_eq!(unchanged, words),
            PlannedSegmentation::Runs(_) => panic!("no one-character Han word"),
        }
    }

    /// Words take the span of their timed characters.
    #[test]
    fn segmented_words_take_their_characters_span() {
        let plan = plan(vec![monologue(&[
            ("你", Some((0, 80))),
            ("今", Some((100, 160))),
            ("仔", None),
            ("日", Some((240, 320))),
            (".", None),
        ])]);
        let monologues = plan
            .apply(vec![vec!["你".into(), "今仔日".into()]])
            .expect("tokens re-spell the run");
        let words = &monologues[0].words;
        assert_eq!(texts(&monologues[0]), ["你", "今仔日", "."]);
        assert_eq!((words[0].start_ms, words[0].end_ms), (Some(0), Some(80)));
        assert_eq!((words[1].start_ms, words[1].end_ms), (Some(100), Some(320)));
    }

    /// A character with only one bound contributes nothing, so a word's start
    /// and end are never taken from two different half-timed characters.
    #[test]
    fn half_timed_characters_contribute_no_time() {
        let plan = plan(vec![MonologueWords {
            speaker: SpeakerIndex(0),
            words: vec![
                AsrWord::new("出", Some(40), None),
                AsrWord::new("門", Some(100), Some(160)),
                AsrWord::new("去", None, Some(400)),
            ],
        }]);
        let monologues = plan
            .apply(vec![vec!["出門".into(), "去".into()]])
            .expect("re-spelled");
        let words = &monologues[0].words;
        assert_eq!((words[0].start_ms, words[0].end_ms), (Some(100), Some(160)));
        assert_eq!((words[1].start_ms, words[1].end_ms), (None, None));
    }

    /// Several runs in several monologues, applied in place.
    #[test]
    fn every_run_is_replaced_where_it_was() {
        let plan = plan(vec![
            monologue(&[
                ("馬", None),
                ("上", None),
                (",", None),
                ("馬", None),
                ("上", None),
            ]),
            monologue(&[("ABC", None)]),
            monologue(&[("出", None), ("門", None), ("去", None)]),
        ]);
        let monologues = plan
            .apply(vec![
                vec!["馬上".into()],
                vec!["馬上".into()],
                vec!["出門".into(), "去".into()],
            ])
            .expect("re-spelled");
        assert_eq!(texts(&monologues[0]), ["馬上", ",", "馬上"]);
        assert_eq!(texts(&monologues[1]), ["ABC"]);
        assert_eq!(texts(&monologues[2]), ["出門", "去"]);
    }

    /// A segmenter that changes, drops or adds a character is refused, and so
    /// is one that answers for the wrong number of runs.
    #[test]
    fn a_segmenter_that_respells_is_refused() {
        let taiwan = || plan(vec![monologue(&[("臺", None), ("灣", None)])]);
        for bad in [
            vec![vec!["臺湾".to_owned()]],
            vec![vec!["臺".to_owned()]],
            vec![vec!["臺灣灣".to_owned()]],
            vec![vec!["臺灣".to_owned(), String::new()]],
        ] {
            assert!(matches!(
                taiwan().apply(bad),
                Err(SegmentationRefusal::Respelled { .. })
            ));
        }
        assert_eq!(
            taiwan().apply(vec![]).map(|_| ()),
            Err(SegmentationRefusal::RunCount {
                sent: 1,
                returned: 0
            })
        );
    }
}
