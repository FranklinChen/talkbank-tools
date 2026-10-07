//! Source-bound lexical timing obligations, discharged only by final output.
//!
//! Final output admission measures every required word of the source against
//! the aligned document and returns one of two states. Complete: every word
//! has a positive interval. Partial: some do not, and the result carries a
//! typed account of each utterance with untimed words, which words, and why
//! (its audio window was refused, so no request was made; or a request was
//! made and produced no usable interval; or the transcript marks it as not in
//! the recording, as `[+ diary]` does). A partial result is still written:
//! the timing that was measured is kept, the untimed words stay in the
//! transcript without bullets, and the file is reported diagnosed with the
//! account as a shortfall. Nothing here can certify a partial result as
//! complete, and a changed lexical structure is still an internal failure,
//! as is any timing on an utterance not in the recording that the input did
//! not give it under a policy keeping given bullets.

use crate::api::OffRecordPostcode;
use crate::chat_ops::fa::RecordingPresence;
use crate::chat_ops::fa::{collect_existing_fa_word_timings, split_compound_filler};
use crate::error::{AlignmentCompletionFailure, ServerError};
use crate::types::results::FaResult;
use crate::types::traces::RefusedWindowTrace;
use std::collections::BTreeMap;
use std::sync::Arc;
use talkbank_model::alignment::helpers::{TierDomain, WordItem, counts_for_tier, walk_words};
use talkbank_model::model::{Line, Utterance};
use talkbank_model::{UtteranceIdx, WordIdx};

/// Captured by input admission, shared across retries without copying labels.
/// One entry per source utterance, in order.
#[derive(Clone)]
pub(super) struct RequiredFaTiming(Arc<[UtteranceObligation]>);

/// What one source utterance obliges the output to carry.
struct UtteranceObligation {
    /// Its lexical words. One entry is one typed CHAT word, even when FA
    /// splits a compound filler.
    words: Vec<Vec<String>>,
    presence: ObligedPresence,
}

/// Whether the output may time the utterance at all.
enum ObligedPresence {
    /// Speech in the recording: every word is owed a positive interval.
    InRecording,
    /// Not in the recording: no word may be timed, and the only bullet it
    /// may carry is the one the input gave it, under a policy that keeps
    /// given bullets. Anything else is a producer fault, never output.
    NotInRecording {
        postcode: OffRecordPostcode,
        permitted_bullet: Option<BulletExtent>,
    },
}

/// A bullet's two endpoints, compared exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BulletExtent {
    start_ms: u64,
    end_ms: u64,
}

impl BulletExtent {
    fn of(bullet: &talkbank_model::model::Bullet) -> Self {
        Self {
            start_ms: bullet.timing.start_ms,
            end_ms: bullet.timing.end_ms,
        }
    }
}

/// Owns the exact result whose source correspondence and timing are complete.
/// No mutable result or independent completion flag can leave this owner.
pub(super) struct CompleteFaResult(FaResult);

/// The result whose source correspondence holds but whose timing does not
/// cover every required word, with the account of what is missing. Built
/// only by [`RequiredFaTiming::admit`], so the account always describes
/// exactly this result.
pub(super) struct PartialFaResult {
    result: FaResult,
    account: UntimedAccount,
}

/// What final output admission established about the source's timing
/// obligations.
pub(super) enum FaCompletion {
    /// Every required word has a positive interval.
    Complete(CompleteFaResult),
    /// Some required words do not; the account says which and why.
    Partial(PartialFaResult),
}

/// Every utterance with untimed required words, in transcript order, never
/// empty: the first is a field of its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UntimedAccount {
    /// Required lexical words in the whole file.
    required_words: usize,
    first: UntimedUtterance,
    rest: Vec<UntimedUtterance>,
}

/// One utterance with untimed required words. `untimed` is never empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UntimedUtterance {
    utterance: UtteranceIdx,
    /// Required lexical words in this utterance.
    words: usize,
    /// The untimed ones, in word order.
    untimed: Vec<WordIdx>,
    cause: UntimedCause,
}

/// Why an utterance's words have no timing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum UntimedCause {
    /// Grouping refused the utterance's audio window, so no alignment
    /// request was made for it.
    WindowRefused(RefusedWindowTrace),
    /// Grouping placed it in no request: the audio left for a run of untimed
    /// utterances could not physically contain their words.
    NotPlaced,
    /// No refusal names it, and the final document has no positive interval
    /// for these words: the aligner returned none, or the words could not be
    /// sent to it.
    NoUsableTiming,
    /// The transcript marks it as not speech in the recording; alignment
    /// never looked for it.
    NotInRecording(OffRecordPostcode),
}

impl UntimedAccount {
    /// Every utterance with untimed words, in transcript order.
    pub(crate) fn utterances(&self) -> impl Iterator<Item = &UntimedUtterance> {
        std::iter::once(&self.first).chain(&self.rest)
    }

    /// Untimed required words in the whole file, at least one.
    pub(crate) fn untimed_words(&self) -> usize {
        self.utterances()
            .map(|utterance| utterance.untimed.len())
            .sum()
    }

    /// The bounded record of this account: the totals and the first
    /// utterances. The complete account is logged once by the caller.
    pub(crate) fn record(&self) -> crate::api::OutputShortfallRecord {
        crate::api::OutputShortfallRecord::TimingIncomplete {
            required_words: self.required_words as u64,
            untimed_words: self.untimed_words() as u64,
            untimed_utterances: (1 + self.rest.len()) as u64,
            first_untimed: self
                .utterances()
                .take(crate::api::FileOutputDiagnostics::FIRST_FINDINGS)
                .map(UntimedUtterance::record)
                .collect(),
        }
    }
}

impl UntimedUtterance {
    fn record(&self) -> crate::api::UntimedUtteranceRecord {
        crate::api::UntimedUtteranceRecord {
            utterance: self.utterance.raw() as u64 + 1,
            words: self.words as u64,
            untimed_words: self.untimed.len() as u64,
            cause: match &self.cause {
                UntimedCause::WindowRefused(window) => {
                    crate::api::UntimedCauseRecord::WindowRefused {
                        window: window.clone(),
                    }
                }
                UntimedCause::NotPlaced => crate::api::UntimedCauseRecord::NotPlaced,
                UntimedCause::NoUsableTiming => crate::api::UntimedCauseRecord::NoUsableTiming,
                UntimedCause::NotInRecording(postcode) => {
                    crate::api::UntimedCauseRecord::NotInRecording {
                        postcode: *postcode,
                    }
                }
            },
        }
    }
}

impl std::fmt::Display for UntimedAccount {
    /// Every utterance and word, for the one full log line.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} of {} required words untimed",
            self.untimed_words(),
            self.required_words
        )?;
        for utterance in self.utterances() {
            write!(f, "; utterance {} words", utterance.utterance.raw() + 1)?;
            for word in &utterance.untimed {
                write!(f, " {}", word.raw() + 1)?;
            }
            match &utterance.cause {
                UntimedCause::WindowRefused(_) => f.write_str(" (window refused)")?,
                UntimedCause::NotPlaced => f.write_str(" (not placed)")?,
                UntimedCause::NoUsableTiming => f.write_str(" (no usable timing)")?,
                UntimedCause::NotInRecording(postcode) => {
                    write!(f, " (not in the recording: {postcode})")?;
                }
            }
        }
        Ok(())
    }
}

impl PartialFaResult {
    /// What is untimed, and why.
    pub(super) fn account(&self) -> &UntimedAccount {
        &self.account
    }

    pub(super) fn into_timeline_trace(self) -> crate::types::traces::FaTimelineTrace {
        self.result.into_timeline_trace()
    }
}

enum WordTimingCoverage {
    Positive,
    Missing,
}

/// Written `%wor` is the output timing surface when present. A no-`%wor`
/// projection retains timings on the typed main words instead. Reuse does not
/// require a consumer to refresh a second representation to prove completion.
fn word_coverage(utterance: &Utterance) -> Vec<WordTimingCoverage> {
    match utterance.wor_tier() {
        Some(tier) => tier
            .words()
            .filter(|word| counts_for_tier(word, TierDomain::Wor))
            .map(|word| match &word.inline_bullet {
                Some(bullet) if bullet.timing.start_ms < bullet.timing.end_ms => {
                    WordTimingCoverage::Positive
                }
                _ => WordTimingCoverage::Missing,
            })
            .collect(),
        None => collect_existing_fa_word_timings(utterance)
            .into_iter()
            .map(|timing| match timing {
                Some(_) => WordTimingCoverage::Positive,
                None => WordTimingCoverage::Missing,
            })
            .collect(),
    }
}

impl CompleteFaResult {
    pub(super) fn into_timeline_trace(self) -> crate::types::traces::FaTimelineTrace {
        self.0.into_timeline_trace()
    }
}

fn lexical_shape(utterance: &Utterance) -> Vec<Vec<String>> {
    let mut words = Vec::new();
    walk_words(&utterance.main.content.content, None, &mut |leaf| {
        let word = match leaf {
            WordItem::Word(word) => word,
            WordItem::ReplacedWord(replaced) => &replaced.word,
            WordItem::Separator(_) => return,
        };
        if counts_for_tier(word, TierDomain::Wor) {
            words.push(split_compound_filler(word));
        }
    });
    words
}

impl RequiredFaTiming {
    /// Bind the obligations to the input as parsed, before any stage edits
    /// the working model. The main-bullet binding decides which bullet an
    /// utterance not in the recording may keep: its given one under `keep`
    /// and `exact`, none under `derive`. One owner, so the bullet completion
    /// permits is the bullet the projection restores.
    ///
    /// # Errors
    /// The binding was made from a different document (an utterance it does
    /// not cover): an internal fault.
    pub(super) fn bind(
        source: &talkbank_model::ChatFile,
        main_bullets: &crate::chat_ops::fa::MainBulletAuthority,
    ) -> Result<Self, crate::chat_ops::fa::KeptBulletError> {
        Ok(Self(
            source
                .lines
                .iter()
                .filter_map(|line| match line {
                    Line::Utterance(utterance) => Some(utterance),
                    _ => None,
                })
                .enumerate()
                .map(|(ordinal, utterance)| {
                    Ok(UtteranceObligation {
                        words: lexical_shape(utterance),
                        presence: match RecordingPresence::of(utterance) {
                            RecordingPresence::InRecording => ObligedPresence::InRecording,
                            RecordingPresence::NotInRecording(postcode) => {
                                ObligedPresence::NotInRecording {
                                    postcode,
                                    permitted_bullet: main_bullets
                                        .off_record_bullet(UtteranceIdx::new(ordinal))?
                                        .as_ref()
                                        .map(BulletExtent::of),
                                }
                            }
                        },
                    })
                })
                .collect::<Result<_, crate::chat_ops::fa::KeptBulletError>>()?,
        ))
    }

    /// Measure `result` against the source's obligations.
    ///
    /// A changed lexical structure is an internal failure (the aligner may
    /// change timing, never words). Otherwise the result is complete, or
    /// partial with the account of every untimed word; both are written.
    pub(super) fn admit(&self, result: FaResult) -> Result<FaCompletion, ServerError> {
        // Why an utterance is untimed is decided by grouping, which records a
        // refused window, or a run it could not place, as a decision on the
        // utterance's line (grouping's line indices are the output's: no
        // later step adds or removes a line). Indexed once: decisions and
        // untimed utterances can each number in the thousands.
        let unplaceable = batchalign_transform::decisions::FaStrategy::UnplaceableRun.as_str();
        let causes: BTreeMap<usize, UntimedCause> = result
            .decisions
            .iter()
            .filter_map(|decision| match &decision.refused_window {
                Some(window) => Some((
                    decision.line_idx,
                    UntimedCause::WindowRefused(window.clone()),
                )),
                None => (decision.strategy == unplaceable)
                    .then_some((decision.line_idx, UntimedCause::NotPlaced)),
            })
            .collect();
        let mut utterances = result
            .output
            .as_chat_file()
            .lines
            .iter()
            .enumerate()
            .filter_map(|(line, entry)| match entry {
                Line::Utterance(utterance) => Some((line, utterance.as_ref())),
                _ => None,
            });
        let mut required_words = 0;
        let mut untimed = Vec::new();
        for (ordinal, expected) in self.0.iter().enumerate() {
            let index = UtteranceIdx::new(ordinal);
            let Some((line, utterance)) = utterances.next() else {
                return Err(AlignmentCompletionFailure::SourceChanged { utterance: index }.into());
            };
            if lexical_shape(utterance) != expected.words {
                return Err(AlignmentCompletionFailure::SourceChanged { utterance: index }.into());
            }
            let timings = word_coverage(utterance);
            if timings.len() != expected.words.len() {
                return Err(AlignmentCompletionFailure::SourceChanged { utterance: index }.into());
            }
            required_words += expected.words.len();
            let missing: Vec<WordIdx> = timings
                .iter()
                .enumerate()
                .filter(|(_, timing)| matches!(timing, WordTimingCoverage::Missing))
                .map(|(word, _)| WordIdx::new(word))
                .collect();
            let cause = match &expected.presence {
                ObligedPresence::InRecording => causes
                    .get(&line)
                    .cloned()
                    .unwrap_or(UntimedCause::NoUsableTiming),
                ObligedPresence::NotInRecording {
                    postcode,
                    permitted_bullet,
                } => {
                    // Nothing in the recording can time it: a timed word, or
                    // a bullet other than the one the input gave under a
                    // keeping policy, is a placement the pipeline invented.
                    let bullet = utterance.main.content.bullet.as_ref().map(BulletExtent::of);
                    if missing.len() != timings.len() || bullet != *permitted_bullet {
                        return Err(AlignmentCompletionFailure::OffRecordUtteranceTimed {
                            utterance: index,
                        }
                        .into());
                    }
                    UntimedCause::NotInRecording(*postcode)
                }
            };
            if !missing.is_empty() {
                untimed.push(UntimedUtterance {
                    utterance: index,
                    words: expected.words.len(),
                    untimed: missing,
                    cause,
                });
            }
        }
        if utterances.next().is_some() {
            return Err(AlignmentCompletionFailure::SourceChanged {
                utterance: UtteranceIdx::new(self.0.len()),
            }
            .into());
        }
        let mut untimed = untimed.into_iter();
        Ok(match untimed.next() {
            None => FaCompletion::Complete(CompleteFaResult(result)),
            Some(first) => FaCompletion::Partial(PartialFaResult {
                result,
                account: UntimedAccount {
                    required_words,
                    first,
                    rest: untimed.collect(),
                },
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_ops::fa::WordGapHealing;
    use crate::engine_reports::FaCacheNamespace;
    use crate::fa::{FaInputDocument, input::FaAdmission};
    use talkbank_model::alignment::helpers::{WordItemMut, walk_words_mut};
    use talkbank_model::model::{Bullet, FileStem, TranscriptName};

    fn attempt(words: &str) -> (talkbank_model::ChatFile, FaAdmission) {
        let source = format!(
            "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n@Media:\tinput, audio, unlinked\n*PAR:\t{words} .\n@End\n"
        );
        let read = crate::fa::read_fa_source_named(
            &source,
            TranscriptName::Named(
                FileStem::from_path(std::path::Path::new("input.cha")).expect("name"),
            ),
            crate::chat_ops::fa::DEFAULT_MAIN_BULLET_POLICY,
        )
        .expect("complete named source admission");
        match read.attempt() {
            FaInputDocument::Active(active) => (active.chat_file, active.admission),
            FaInputDocument::Preserved(_) => panic!("ordinary source must align"),
        }
    }

    fn time_words(file: &mut talkbank_model::ChatFile, only_last: bool) -> usize {
        for line in &mut file.lines {
            let Line::Utterance(utterance) = line else {
                continue;
            };
            let words = lexical_shape(utterance).len();
            let mut ordinal = 0;
            walk_words_mut(
                utterance.main.content.content.as_mut_slice(),
                None,
                &mut |leaf| {
                    let word = match leaf {
                        WordItemMut::Word(word) => word,
                        WordItemMut::ReplacedWord(replaced) => &mut replaced.word,
                        WordItemMut::Separator(_) => return,
                    };
                    if counts_for_tier(word, TierDomain::Wor) {
                        if !only_last || ordinal + 1 == words {
                            word.inline_bullet = Some(if only_last {
                                Bullet::new(19_980, 20_000)
                            } else {
                                Bullet::new(100 + ordinal as u64 * 100, 180 + ordinal as u64 * 100)
                            });
                        }
                        ordinal += 1;
                    }
                },
            );
            utterance.main.content.bullet = Some(Bullet::new(0, 20_000));
            return words;
        }
        panic!("fixture must have an utterance");
    }

    fn result(file: talkbank_model::ChatFile) -> FaResult<talkbank_model::ChatFile> {
        FaResult::without_groups(
            file,
            WordGapHealing::PreserveMeasured,
            "test_engine",
            &FaCacheNamespace::for_test("test-build"),
        )
    }

    /// One timed word out of nine is a partial result: written, with the
    /// account of the eight untimed words as the file's shortfall, never a
    /// refusal and never a clean success.
    #[test]
    fn eight_missing_timings_are_written_with_their_account() {
        let (mut file, admission) = attempt("one two three four five six seven eight nine");
        assert_eq!(time_words(&mut file, true), 9);
        let admitted = admission
            .finish(result(file))
            .expect("partial timing is written, not refused");
        let (_document, shortfalls) = admitted.into_document();
        insta::assert_json_snapshot!(shortfalls, @r#"
        [
          {
            "kind": "timing_incomplete",
            "required_words": 9,
            "untimed_words": 8,
            "untimed_utterances": 1,
            "first_untimed": [
              {
                "utterance": 1,
                "words": 9,
                "untimed_words": 8,
                "cause": {
                  "kind": "no_usable_timing"
                }
              }
            ]
          }
        ]
        "#);
        insta::assert_snapshot!(shortfalls[0].to_string(), @"timing incomplete: 8 of 9 words in 1 utterance(s) have no timing and were written without it (first: utterance 1, 8 of 9 words, no usable timing)");
    }

    /// No timed word at all is still a partial result, written with its
    /// account: the words and the source structure are the output's value.
    #[test]
    fn a_document_with_no_word_timing_is_written_with_its_account() {
        let (file, admission) = attempt("hello world");
        let admitted = admission
            .finish(result(file))
            .expect("untimed output is written, not refused");
        let (_document, shortfalls) = admitted.into_document();
        let [
            crate::api::OutputShortfallRecord::TimingIncomplete {
                required_words: 2,
                untimed_words: 2,
                untimed_utterances: 1,
                first_untimed,
            },
        ] = shortfalls.as_slice()
        else {
            panic!("expected one timing shortfall, got {shortfalls:?}");
        };
        assert_eq!(first_untimed.len(), 1);
    }

    #[test]
    fn completed_timing_retains_its_payload_through_output_admission() {
        let (mut file, admission) = attempt("hello world");
        assert_eq!(time_words(&mut file, false), 2);
        let admitted = admission
            .finish(result(file))
            .expect("complete timing admission");
        let (_document, shortfalls, _timeline) = admitted.into_document_and_timeline();
        assert!(shortfalls.is_empty(), "complete timing is clean");
    }

    #[test]
    fn a_different_valid_lexical_output_cannot_discharge_source_obligations() {
        let (_, admission) = attempt("hello world");
        for words in ["hello", "hello there", "hello world now"] {
            let (mut file, _) = attempt(words);
            time_words(&mut file, false);
            let Err(error) = admission.clone().finish(result(file)) else {
                panic!("source drift: {words}");
            };
            assert!(matches!(
                error,
                ServerError::AlignmentCompletion(AlignmentCompletionFailure::SourceChanged { .. })
            ));
            assert_eq!(
                crate::runner::util::classify_server_error(&error),
                crate::scheduling::FailureCategory::System
            );
        }
    }

    #[test]
    fn typed_compound_fillers_and_replacements_use_chat_word_obligations() {
        let (mut file, admission) = attempt("&-uh_huh hello [: hi] &-uh");
        assert_eq!(
            time_words(&mut file, false),
            3,
            "compound filler parts are model labels, not separate source words"
        );
        assert!(
            admission
                .finish(result(file))
                .expect("complete")
                .into_document()
                .1
                .is_empty()
        );
    }

    /// Admit a two-utterance source: spoken words with a bullet, then a
    /// `[+ diary]` note, with the bullet `diary_bullet` if given.
    fn attempt_with_diary(
        diary_bullet: &str,
        policy: crate::chat_ops::fa::MainBulletPolicy,
    ) -> (talkbank_model::ChatFile, FaAdmission) {
        let source = format!(
            "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tMOT Mother\n@ID:\teng|test|MOT|||||Mother|||\n@Media:\tinput, audio\n*MOT:\tgo there now . \u{15}0_20000\u{15}\n*MOT:\twent to the park . [+ diary]{diary_bullet}\n@End\n"
        );
        let read = crate::fa::read_fa_source_named(
            &source,
            TranscriptName::Named(
                FileStem::from_path(std::path::Path::new("input.cha")).expect("name"),
            ),
            policy,
        )
        .expect("a diary note is admitted");
        match read.attempt() {
            FaInputDocument::Active(active) => (active.chat_file, active.admission),
            FaInputDocument::Preserved(_) => panic!("ordinary source must align"),
        }
    }

    fn diary_utterance(file: &mut talkbank_model::ChatFile) -> &mut Utterance {
        (&mut file.lines)
            .into_iter()
            .filter_map(|line| match line {
                Line::Utterance(utterance) => Some(utterance.as_mut()),
                _ => None,
            })
            .nth(1)
            .expect("the diary note is the second utterance")
    }

    /// A `[+ diary]` note is written untimed and reported with its cause,
    /// never dropped from the account. Its words count among the file's, and
    /// the spoken utterance's timing is unaffected.
    #[test]
    fn a_diary_note_is_reported_untimed_because_it_is_not_in_the_recording() {
        let (mut file, admission) =
            attempt_with_diary("", crate::chat_ops::fa::DEFAULT_MAIN_BULLET_POLICY);
        assert_eq!(time_words(&mut file, false), 3);
        let (_document, shortfalls) = admission
            .finish(result(file))
            .expect("the note is written untimed")
            .into_document();
        insta::assert_json_snapshot!(shortfalls, @r#"
        [
          {
            "kind": "timing_incomplete",
            "required_words": 7,
            "untimed_words": 4,
            "untimed_utterances": 1,
            "first_untimed": [
              {
                "utterance": 2,
                "words": 4,
                "untimed_words": 4,
                "cause": {
                  "kind": "not_in_recording",
                  "postcode": "diary"
                }
              }
            ]
          }
        ]
        "#);
        insta::assert_snapshot!(shortfalls[0].to_string(), @"timing incomplete: 4 of 7 words in 1 utterance(s) have no timing and were written without it (first: utterance 2, 4 of 4 words, not in the recording: [+ diary])");
    }

    /// The input's bullet on a diary note: admission removes it from the
    /// working model under every policy, so no stage reads it as evidence.
    /// Under `derive` the output may not carry one; under `keep` it may carry
    /// exactly the given one (which imposition restores) and nothing else.
    #[test]
    fn a_diary_notes_given_bullet_is_kept_only_where_the_policy_keeps_given_bullets() {
        use crate::chat_ops::fa::MainBulletPolicy;
        let given = " \u{15}20000_21000\u{15}";
        for policy in [
            MainBulletPolicy::DeriveFromWords,
            MainBulletPolicy::KeepGiven,
        ] {
            let (mut file, _) = attempt_with_diary(given, policy);
            assert!(
                diary_utterance(&mut file).main.content.bullet.is_none(),
                "{policy:?}: the working model never holds it"
            );
        }

        let (mut file, admission) = attempt_with_diary(given, MainBulletPolicy::KeepGiven);
        time_words(&mut file, false);
        diary_utterance(&mut file).main.content.bullet = Some(Bullet::new(20_000, 21_000));
        assert!(
            admission.finish(result(file)).is_ok(),
            "keep: the given bullet, restored, is the input's own"
        );

        for (policy, bullet) in [
            (
                MainBulletPolicy::DeriveFromWords,
                Bullet::new(20_000, 21_000),
            ),
            (MainBulletPolicy::KeepGiven, Bullet::new(20_000, 22_000)),
        ] {
            let (mut file, admission) = attempt_with_diary(given, policy);
            time_words(&mut file, false);
            diary_utterance(&mut file).main.content.bullet = Some(bullet);
            let Err(error) = admission.finish(result(file)) else {
                panic!("{policy:?}: a bullet the policy does not keep is invented timing");
            };
            assert!(matches!(
                error,
                ServerError::AlignmentCompletion(
                    AlignmentCompletionFailure::OffRecordUtteranceTimed { .. }
                )
            ));
        }
    }

    /// Under `keep`, a diary note's given bullet is restored exactly but
    /// orders nothing, so timing aligned around it can conflict with it (here
    /// the same speaker's aligned utterance overlaps it). The output cannot
    /// be valid; the refusal names the note and the remedy, as the input and
    /// policy conflict it is, not an internal fault.
    #[test]
    fn a_kept_diary_bullet_that_conflicts_with_aligned_timing_is_named() {
        let source = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tMOT Mother\n@ID:\teng|test|MOT|||||Mother|||\n@Media:\tinput, audio\n*MOT:\tgo there now .\n*MOT:\twent to the park . [+ diary] \u{15}1000_2000\u{15}\n@End\n";
        let read = crate::fa::read_fa_source_named(
            source,
            TranscriptName::Named(
                FileStem::from_path(std::path::Path::new("input.cha")).expect("name"),
            ),
            crate::chat_ops::fa::MainBulletPolicy::KeepGiven,
        )
        .expect("the note's bullet is the input's only timing, and admitted");
        let FaInputDocument::Active(active) = read.attempt() else {
            panic!("ordinary source must align");
        };
        let (mut file, admission) = (active.chat_file, active.admission);
        // The spoken utterance aligned across the note's given extent.
        time_words(&mut file, false);
        diary_utterance(&mut file).main.content.bullet = Some(Bullet::new(1_000, 2_000));
        let Err(error) = admission.finish(result(file)) else {
            panic!("an overlapping same-speaker bullet cannot be written");
        };
        assert!(
            matches!(
                &error,
                ServerError::KeptOffRecordBulletConflict { utterances } if utterances == &[2]
            ),
            "{error}"
        );
        assert_eq!(
            crate::runner::util::classify_server_error(&error),
            crate::scheduling::FailureCategory::Validation
        );
    }

    /// A timed word on a diary note is a placement nothing in the recording
    /// supports: an internal fault, never output.
    #[test]
    fn a_timed_word_on_a_diary_note_is_a_producer_fault() {
        let (mut file, admission) =
            attempt_with_diary("", crate::chat_ops::fa::DEFAULT_MAIN_BULLET_POLICY);
        time_words(&mut file, false);
        let mut first = true;
        walk_words_mut(
            diary_utterance(&mut file)
                .main
                .content
                .content
                .as_mut_slice(),
            None,
            &mut |leaf| {
                if let WordItemMut::Word(word) = leaf
                    && std::mem::take(&mut first)
                {
                    word.inline_bullet = Some(Bullet::new(20_100, 20_400));
                }
            },
        );
        let Err(error) = admission.finish(result(file)) else {
            panic!("a timed diary word must not be written");
        };
        assert!(matches!(
            error,
            ServerError::AlignmentCompletion(
                AlignmentCompletionFailure::OffRecordUtteranceTimed { .. }
            )
        ));
        assert_eq!(
            crate::runner::util::classify_server_error(&error),
            crate::scheduling::FailureCategory::System
        );
    }

    #[test]
    fn an_untranscribed_turn_has_no_lexical_timing_obligation() {
        let (file, admission) = attempt("xxx");
        assert!(
            admission
                .finish(result(file))
                .expect("nothing to time")
                .into_document()
                .1
                .is_empty()
        );
    }
}
