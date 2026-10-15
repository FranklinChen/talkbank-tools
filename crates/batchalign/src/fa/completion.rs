//! Source-bound lexical timing obligations, discharged only by final output.
//!
//! Final output admission measures every required word of the source against
//! the aligned document and returns one of two states. Complete: every word
//! has a positive interval. Partial: some do not, and the result carries a
//! typed account of each utterance with untimed words, which words, and why
//! (its audio window was refused, so no request was made; or a request was
//! made and produced no usable interval). A partial result is still written:
//! the timing that was measured is kept, the untimed words stay in the
//! transcript without bullets, and the file is reported diagnosed with the
//! account as a shortfall. Nothing here can certify a partial result as
//! complete, and a changed lexical structure is still an internal failure.
//!
//! An utterance the transcript marks as not in the recording (`[+ diary]`)
//! owes no timing at all, so its words are not required words and it is
//! never in the untimed account. Whatever the result's state, it carries the
//! separate [`ExclusionAccount`] of every such utterance, which
//! is reported as information and does not diagnose the file (ruling of
//! 2026-10-07: an intentional exclusion must not count against a file). Any
//! timing on one that the input did not give it, under a policy keeping
//! given bullets, is an internal failure.

use crate::api::OffRecordPostcode;
use crate::chat_ops::fa::RecordingPresence;
use crate::chat_ops::fa::{collect_existing_fa_word_timings, split_compound_filler};
use crate::error::{AlignmentCompletionFailure, ServerError};
use crate::pipeline::post_validate::PostValidated;
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
pub(super) struct CompleteFaResult {
    result: WritableFaResult,
    /// The utterances left out on purpose; `None` when there is none.
    exclusions: Option<ExclusionAccount>,
}

/// The result whose source correspondence holds but whose timing does not
/// cover every required word, with the account of what is missing. Built
/// only by [`RequiredFaTiming::admit`], so the account always describes
/// exactly this result.
pub(super) struct PartialFaResult {
    result: WritableFaResult,
    account: UntimedAccount,
    /// The utterances left out on purpose; `None` when there is none.
    exclusions: Option<ExclusionAccount>,
}

/// What final output admission established about the source's timing
/// obligations.
pub(super) enum FaCompletion {
    /// Every required word has a positive interval.
    Complete(CompleteFaResult),
    /// Some required words do not; the account says which and why.
    Partial(PartialFaResult),
}

/// The construction-admitted output and the run that produced it, inseparable
/// until completion has been measured against its writable timing surface.
/// Only `gate` can construct this phase; a raw in-memory result is not enough.
pub(super) struct WritableFaResult {
    result: FaResult,
    document: PostValidated,
}

pub(super) struct FinalFaOutput {
    pub(super) output: super::AlignedOutput,
    pub(super) evidence: FaResult,
}

impl WritableFaResult {
    pub(super) fn gate(result: FaResult) -> Result<Self, ServerError> {
        let document = PostValidated::gate(
            result.output.as_chat_file(),
            crate::api::ReleasedCommand::Align,
        )
        .map_err(|failure| super::explain_gate_refusal(result.output.as_chat_file(), failure))?;
        Ok(Self { result, document })
    }

    pub(super) fn into_output(
        self,
        shortfalls: Vec<crate::pipeline::post_validate::Shortfall>,
        exclusions: Vec<crate::pipeline::post_validate::Exclusion>,
    ) -> FinalFaOutput {
        FinalFaOutput {
            output: super::AlignedOutput {
                document: self.document,
                shortfalls,
                exclusions,
            },
            evidence: self.result,
        }
    }
}

/// Every utterance with untimed required words, in transcript order, never
/// empty: the first is a field of its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UntimedAccount {
    /// Required lexical words in the whole file: every word of an utterance
    /// in the recording. A word of an excluded utterance is not required.
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
    /// No supplied timing or source-bound UTR search evidence; no request.
    LocalizationUnavailable,
    /// No refusal names it, and the final document has no positive interval
    /// for these words: the aligner returned none, or the words could not be
    /// sent to it.
    NoUsableTiming,
}

/// Every utterance alignment left out because the transcript marks it as not
/// in the recording, in transcript order, never empty: the first is a field
/// of its own. Built only by [`RequiredFaTiming::admit`], after it checked
/// that the output gave none of them timing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExclusionAccount {
    first: ExcludedUtterance,
    rest: Vec<ExcludedUtterance>,
}

/// One utterance left out of alignment.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ExcludedUtterance {
    utterance: UtteranceIdx,
    /// Its lexical words, none owed timing.
    words: usize,
    postcode: OffRecordPostcode,
}

impl ExclusionAccount {
    fn utterances(&self) -> impl Iterator<Item = &ExcludedUtterance> {
        std::iter::once(&self.first).chain(&self.rest)
    }

    /// The bounded record of this account: the totals and the first
    /// utterances. The complete account is logged once by the caller.
    pub(crate) fn record(&self) -> crate::api::OutputExclusionRecord {
        crate::api::OutputExclusionRecord::NotInRecording {
            excluded_utterances: (1 + self.rest.len()) as u64,
            excluded_words: self
                .utterances()
                .map(|utterance| utterance.words as u64)
                .sum(),
            first_excluded: self
                .utterances()
                .take(crate::api::FileOutputDiagnostics::FIRST_FINDINGS)
                .map(|excluded| crate::api::ExcludedUtteranceRecord {
                    utterance: excluded.utterance.raw() as u64 + 1,
                    words: excluded.words as u64,
                    postcode: excluded.postcode,
                })
                .collect(),
        }
    }
}

impl std::fmt::Display for ExclusionAccount {
    /// Every utterance, for the one full log line.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("not in the recording:")?;
        for excluded in self.utterances() {
            write!(
                f,
                " utterance {} {}",
                excluded.utterance.raw() + 1,
                excluded.postcode
            )?;
        }
        Ok(())
    }
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
                UntimedCause::LocalizationUnavailable => {
                    crate::api::UntimedCauseRecord::LocalizationUnavailable
                }
                UntimedCause::NoUsableTiming => crate::api::UntimedCauseRecord::NoUsableTiming,
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
                UntimedCause::LocalizationUnavailable => {
                    f.write_str(" (localization unavailable)")?
                }
                UntimedCause::NoUsableTiming => f.write_str(" (no usable timing)")?,
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

    /// The utterances left out on purpose, checked untimed by the same
    /// admission that measured the result.
    pub(super) fn exclusions(&self) -> Option<&ExclusionAccount> {
        self.exclusions.as_ref()
    }

    pub(super) fn into_result(self) -> WritableFaResult {
        self.result
    }
}

enum WordTimingCoverage {
    Positive,
    Missing,
}

/// `%wor` explicitly writes each word's bullet. Main-tier `Word.inline_bullet`
/// is working state, not emitted by the Word writer. Do not certify that state
/// as durable word timing when the chosen projection has no `%wor` tier.
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
        None => lexical_shape(utterance)
            .into_iter()
            .map(|_| WordTimingCoverage::Missing)
            .collect(),
    }
}

impl CompleteFaResult {
    /// The utterances left out on purpose, checked untimed by the same
    /// admission that measured the result.
    pub(super) fn exclusions(&self) -> Option<&ExclusionAccount> {
        self.exclusions.as_ref()
    }

    pub(super) fn into_result(self) -> WritableFaResult {
        self.result
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

/// The lexical/timing projection a completion account actually described.
/// Retained in output admission so later cosmetic transforms cannot change
/// that projection while carrying the old complete/partial account to disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DurableFaTimingCensus(Arc<[UtteranceProjection]>);

#[derive(Debug, PartialEq, Eq)]
struct UtteranceProjection {
    speaker: talkbank_model::model::SpeakerCode,
    words: Vec<Vec<String>>,
    bullet: Option<BulletExtent>,
    wor: Option<Vec<WordProjection>>,
    off_record: Option<OffRecordPostcode>,
}

#[derive(Debug, PartialEq, Eq)]
struct WordProjection {
    text: String,
    bullet: Option<BulletExtent>,
}

impl DurableFaTimingCensus {
    pub(crate) fn of(file: &talkbank_model::ChatFile) -> Self {
        Self(
            file.utterances()
                .map(|utterance| UtteranceProjection {
                    speaker: utterance.main.speaker.clone(),
                    words: lexical_shape(utterance),
                    bullet: utterance.main.content.bullet.as_ref().map(BulletExtent::of),
                    wor: utterance.wor_tier().map(|tier| {
                        tier.words()
                            .map(|word| WordProjection {
                                text: word.cleaned_text().to_owned(),
                                bullet: word.inline_bullet.as_ref().map(BulletExtent::of),
                            })
                            .collect()
                    }),
                    off_record: match RecordingPresence::of(utterance) {
                        RecordingPresence::InRecording => None,
                        RecordingPresence::NotInRecording(postcode) => Some(postcode),
                    },
                })
                .collect(),
        )
    }

    pub(crate) fn is_preserved_in(&self, file: &talkbank_model::ChatFile) -> bool {
        self == &Self::of(file)
    }
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
    /// change timing, never words), and so is timing on an utterance not in
    /// the recording. Otherwise the result is complete, or partial with the
    /// account of every untimed word owed timing; both are written, with
    /// the account of the utterances left out on purpose beside them.
    pub(super) fn admit(&self, result: WritableFaResult) -> Result<FaCompletion, ServerError> {
        // Why an utterance is untimed is decided by grouping, which records a
        // refused window, or a run it could not place, as a decision on the
        // utterance's line (grouping's line indices are the output's: no
        // later step adds or removes a line). Indexed once: decisions and
        // untimed utterances can each number in the thousands.
        let unplaceable = batchalign_transform::decisions::FaStrategy::UnplaceableRun.as_str();
        let unlocalized =
            batchalign_transform::decisions::FaStrategy::LocalizationUnavailable.as_str();
        let causes: BTreeMap<usize, UntimedCause> = result
            .result
            .decisions
            .iter()
            .filter_map(|decision| match &decision.refused_window {
                Some(window) => Some((
                    decision.line_idx,
                    UntimedCause::WindowRefused(window.clone()),
                )),
                None if decision.strategy == unplaceable => {
                    Some((decision.line_idx, UntimedCause::NotPlaced))
                }
                None if decision.strategy == unlocalized => {
                    Some((decision.line_idx, UntimedCause::LocalizationUnavailable))
                }
                None => None,
            })
            .collect();
        let mut utterances = result
            .document
            .document()
            .lines
            .iter()
            .enumerate()
            .filter_map(|(line, entry)| match entry {
                Line::Utterance(utterance) => Some((line, utterance.as_ref())),
                _ => None,
            });
        let mut required_words = 0;
        let mut untimed = Vec::new();
        let mut excluded = Vec::new();
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
            let missing: Vec<WordIdx> = timings
                .iter()
                .enumerate()
                .filter(|(_, timing)| matches!(timing, WordTimingCoverage::Missing))
                .map(|(word, _)| WordIdx::new(word))
                .collect();
            match &expected.presence {
                ObligedPresence::InRecording => {
                    required_words += expected.words.len();
                    if !missing.is_empty() {
                        untimed.push(UntimedUtterance {
                            utterance: index,
                            words: expected.words.len(),
                            untimed: missing,
                            cause: causes
                                .get(&line)
                                .cloned()
                                .unwrap_or(UntimedCause::NoUsableTiming),
                        });
                    }
                }
                ObligedPresence::NotInRecording {
                    postcode,
                    permitted_bullet,
                } => {
                    // Nothing in the recording can time it: a timed word, or
                    // a bullet other than the one the input gave under a
                    // keeping policy, is a placement the pipeline invented.
                    let bullet = utterance.main.content.bullet.as_ref().map(BulletExtent::of);
                    if missing.len() != timings.len()
                        || collect_existing_fa_word_timings(utterance)
                            .iter()
                            .any(Option::is_some)
                        || bullet != *permitted_bullet
                    {
                        return Err(AlignmentCompletionFailure::OffRecordUtteranceTimed {
                            utterance: index,
                        }
                        .into());
                    }
                    excluded.push(ExcludedUtterance {
                        utterance: index,
                        words: expected.words.len(),
                        postcode: *postcode,
                    });
                }
            }
        }
        if utterances.next().is_some() {
            return Err(AlignmentCompletionFailure::SourceChanged {
                utterance: UtteranceIdx::new(self.0.len()),
            }
            .into());
        }
        let mut excluded = excluded.into_iter();
        let exclusions = excluded.next().map(|first| ExclusionAccount {
            first,
            rest: excluded.collect(),
        });
        let mut untimed = untimed.into_iter();
        Ok(match untimed.next() {
            None => FaCompletion::Complete(CompleteFaResult { result, exclusions }),
            Some(first) => FaCompletion::Partial(PartialFaResult {
                result,
                account: UntimedAccount {
                    required_words,
                    first,
                    rest: untimed.collect(),
                },
                exclusions,
            }),
        })
    }
}

#[cfg(test)]
mod tests;
