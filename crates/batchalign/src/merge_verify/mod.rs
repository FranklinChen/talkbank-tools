//! The verify-flag tier pass over merged drafts (`merge-verify`).
//!
//! Consumes a merged draft directory plus an engine-verdicts JSON
//! (forced-alignment score, pitch band, machine-ear answer per flagged
//! utterance, produced upstream by the verify engines or replayed from
//! a cache) and emits a rewritten draft plus a review queue.
//!
//! Tier semantics (human-calibrated against blind listening verdicts;
//! measured 97.5% precision on the auto-trust tier):
//!
//! - AUTO_TRUST: trusted category AND ear YES AND pitch CHILD. The
//!   verify `%com` flag is REWRITTEN to a machine-verified provenance
//!   note (maintainer ruling: provenance survives in the transcript,
//!   flags are never silently deleted).
//! - REVIEW: trusted category failing a gate, or an uncalibrated
//!   category. Flag untouched; the line exports to the review queue.
//! - HOLD: clock and region categories; untouched entirely (clock
//!   placements measured 0/7 correct; interpolation drift is a
//!   session-level problem, not a per-line one).
//! - Demotion: a `confident` line with adverse verdicts (pitch ADULT
//!   or ear NO) GAINS a review flag and joins the queue; text and
//!   timing are never moved by this pass.
//!
//! Corpus-specific flag vocabularies stay OUTSIDE this module: the
//! verdicts JSON carries each line's category (mapped at the corpus
//! seam), and verify flags are identified by a caller-supplied prefix.
//!
//! Complete named source admission precedes editing. A source-bound plan owns
//! both the admitted draft and its unique verdicts. Only checked typed output
//! can be written; main-tier structure is preserved, not original formatting.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use talkbank_model::model::{
    BulletContentSegment, ComTier, DependentTier, DependentTierEntry, Line, TranscriptName,
};
use talkbank_model::validation::{
    AlignmentValidation, ValidChatFile, ValidationFailure, ValidationPolicy,
};
use talkbank_model::{NullErrorSink, RuleSelection, WriteChat};
use unicode_normalization::UnicodeNormalization;

/// Ordinal of a main-tier utterance within one file (0-based, in
/// document order). The verdicts JSON keys lines by this ordinal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Hash)]
#[serde(transparent)]
pub struct UtteranceOrdinal(pub usize);

/// Calibrated flag category, supplied per line by the corpus seam.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerifyCategory {
    /// Diarization-mislabel rescues (the primary promotion target).
    Other,
    /// Medium-confidence anchors.
    Medium,
    /// Medium-confidence anchors under their measurement-era name.
    UnknownFlag,
    /// Approximate/pitch-ambiguous placements.
    Approx,
    /// Interpolated (clock) placements; never auto-trusted.
    Clock,
    /// Child-voice-region placements; never auto-trusted.
    Region,
    /// Weak matches (uncalibrated; always reviewed).
    Weak,
    /// Boundary-suspect glued placements (uncalibrated; always reviewed).
    Glued,
    /// Child-orphan placements (uncalibrated; always reviewed).
    ChildOrphan,
    /// Not flagged; eligible only for demotion re-flagging.
    Confident,
}

impl VerifyCategory {
    /// Whether the ear sample calibrated this category as safe for
    /// silent trust when the remaining gates pass.
    fn is_trusted(self) -> bool {
        matches!(
            self,
            VerifyCategory::Other
                | VerifyCategory::Medium
                | VerifyCategory::UnknownFlag
                | VerifyCategory::Approx
        )
    }
}

/// Pitch band verdict for the placed span (from the pitch engine).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PitchBand {
    /// Voiced f0 sits in the child band for the placed span.
    Child,
    /// Voiced f0 sits in the adult band.
    Adult,
    /// Too few voiced frames, or mixed banding (whisper, murmur, overlap).
    Ambiguous,
}

/// Machine-ear verdict (does the child say the claimed text here?).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EarVerdict {
    /// The machine ear heard the child say the claimed text.
    Yes,
    /// It did not (routes to review, never auto-demotes on its own).
    No,
}

/// One line's engine verdicts, keyed by utterance ordinal.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LineVerdict {
    /// Which main-tier utterance this verdict describes.
    pub utterance_index: UtteranceOrdinal,
    /// Calibrated flag category (mapped at the corpus seam).
    pub category: VerifyCategory,
    /// Mean per-word forced-alignment confidence for the placed text.
    /// None = UNSCORABLE: the token sequence exceeded what the audio
    /// window could hold, so CTC alignment was impossible. FA is
    /// ordering-only (never a promote/demote gate), so an unscorable
    /// line is tiered by its category, pitch, and ear alone.
    pub fa_mean_score: Option<f64>,
    /// Pitch band of the placed span.
    pub pitch: PitchBand,
    /// Machine-ear answer for the placed text.
    pub ear: EarVerdict,
}

/// All verdicts for one session file (`<session>.cha`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionVerdicts {
    /// Session id; the draft file is `<session>.cha`.
    pub session: String,
    /// Per-line verdicts, keyed by utterance ordinal.
    pub lines: Vec<LineVerdict>,
}

/// The verdicts document consumed by the pass.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerdictsDoc {
    /// Every session the pass should process.
    pub sessions: Vec<SessionVerdicts>,
}

/// What the calibrated rule decides for one line.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TierOutcome {
    /// Rewrite the verify flag to a machine-verified provenance note.
    AutoTrust,
    /// Leave the flag; export the line to the review queue.
    Review,
    /// Untouched (clock/region).
    Hold,
    /// Confident line with adverse verdicts: gains a review flag.
    Demote,
    /// Confident line with benign verdicts: nothing to do.
    Untouched,
}

/// Assign the calibrated tier for one verdict line.
pub fn tier_outcome(verdict: &LineVerdict) -> TierOutcome {
    match verdict.category {
        VerifyCategory::Clock | VerifyCategory::Region => TierOutcome::Hold,
        VerifyCategory::Confident => {
            if verdict.pitch == PitchBand::Adult || verdict.ear == EarVerdict::No {
                TierOutcome::Demote
            } else {
                TierOutcome::Untouched
            }
        }
        VerifyCategory::Weak | VerifyCategory::Glued | VerifyCategory::ChildOrphan => {
            TierOutcome::Review
        }
        category if category.is_trusted() => {
            if verdict.ear == EarVerdict::Yes && verdict.pitch == PitchBand::Child {
                TierOutcome::AutoTrust
            } else {
                TierOutcome::Review
            }
        }
        // is_trusted covers every remaining variant; keep the compiler
        // honest if a category is ever added without a tier decision.
        VerifyCategory::Other
        | VerifyCategory::Medium
        | VerifyCategory::UnknownFlag
        | VerifyCategory::Approx => unreachable_category(),
    }
}

/// The guard arm above is structurally unreachable (`is_trusted`
/// matches exactly those four variants); modeled as a typed dead end
/// rather than a panic per the no-panic policy.
fn unreachable_category() -> TierOutcome {
    TierOutcome::Review
}

/// One review-queue entry (REVIEW tier or demotion).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueueEntry {
    /// Session id of the queued line.
    pub session: String,
    /// Which main-tier utterance to review.
    pub utterance_index: UtteranceOrdinal,
    /// The line's calibrated flag category.
    pub category: VerifyCategory,
    /// Why the line queued (REVIEW gate failure or demotion).
    pub tier: TierOutcome,
    /// Mean per-word forced-alignment confidence (None = unscorable).
    pub fa_mean_score: Option<f64>,
    /// Pitch band of the placed span.
    pub pitch: PitchBand,
    /// Machine-ear answer.
    pub ear: EarVerdict,
}

/// The exported review queue.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReviewQueue {
    /// Queued lines across all sessions, in document order per session.
    pub entries: Vec<QueueEntry>,
}

/// Per-run counts, printed by the CLI.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct VerifySummary {
    /// Sessions processed.
    pub sessions: usize,
    /// Lines promoted to machine-verified provenance notes.
    pub auto_trusted: usize,
    /// Lines exported to the review queue (gate failures).
    pub reviewed: usize,
    /// Clock/region lines left untouched.
    pub held: usize,
    /// Previously-confident lines re-flagged for review.
    pub demoted: usize,
}

/// Typed failures of the pass.
#[derive(Debug, thiserror::Error)]
pub enum MergeVerifyError {
    /// Reading or writing a filesystem path failed.
    #[error("i/o failure on {path}: {source}")]
    Io {
        /// Path the i/o operation targeted.
        path: PathBuf,
        /// Underlying i/o error.
        #[source]
        source: std::io::Error,
    },
    /// The verdicts document is not valid JSON for the expected shape.
    #[error("verdicts JSON at {path} did not parse: {source}")]
    VerdictsParse {
        /// Path of the verdicts document.
        path: PathBuf,
        /// Underlying JSON error.
        #[source]
        source: serde_json::Error,
    },
    /// A session named in the verdicts has no draft file.
    #[error("draft file for session '{session}' not found at {path}")]
    MissingSession {
        /// Session named by the verdicts document.
        session: String,
        /// Draft path that was expected to exist.
        path: PathBuf,
    },
    /// Complete named source admission refused the draft.
    #[error("draft {path} failed CHAT admission: {source}")]
    SourceAdmission {
        /// Draft that was judged.
        path: PathBuf,
        /// Typed refusal, retaining internal failure versus invalidity.
        #[source]
        source: batchalign_transform::ValidatedParseError,
    },
    /// The transform could not establish checked output admission.
    #[error("output for session '{session}' failed CHAT admission: {source}")]
    OutputAdmission {
        /// Session whose typed output was judged.
        session: String,
        /// Producer failure, never a claim of invalid source CHAT.
        #[source]
        source: ValidationFailure,
    },
    /// A session name is not a single safe filename stem.
    #[error("unsafe session filename stem: {session:?}")]
    UnsafeSession {
        /// Refused wire value.
        session: String,
    },
    /// A document names the same session more than once.
    #[error("duplicate session verdicts for '{session}'")]
    DuplicateSession {
        /// Conflicting session identity.
        session: String,
    },
    /// An ordinal must have exactly one verdict, even if duplicates agree.
    #[error("duplicate verdict in session '{session}' for utterance {ordinal:?}")]
    DuplicateVerdict {
        /// Session containing the duplicate.
        session: String,
        /// Duplicate ordinal.
        ordinal: UtteranceOrdinal,
    },
    /// A verdict names an utterance ordinal past the draft's end.
    #[error(
        "verdict for session '{session}' names utterance ordinal {ordinal:?} \
         but the draft has only {utterance_count} main-tier utterances"
    )]
    OrdinalOutOfRange {
        /// Session whose verdicts overran the draft.
        session: String,
        /// The out-of-range ordinal.
        ordinal: UtteranceOrdinal,
        /// How many main-tier utterances the draft actually has.
        utterance_count: usize,
    },
    /// The preservation invariant broke: a main tier changed.
    #[error(
        "preservation invariant violated in session '{session}': main tier \
         {ordinal:?} changed through the pass\n  before: {before}\n  after:  {after}"
    )]
    MainTierChanged {
        /// Session where the invariant broke.
        session: String,
        /// Ordinal of the changed main tier, without sentinel values.
        ordinal: UtteranceOrdinal,
        /// Debug representation of the original typed main tier.
        before: String,
        /// Debug representation of the transformed typed main tier.
        after: String,
    },
    /// The review queue could not be serialized.
    #[error("review queue at {path} failed to serialize: {source}")]
    QueueSerialize {
        /// Queue output path.
        path: PathBuf,
        /// Underlying JSON error.
        #[source]
        source: serde_json::Error,
    },
}

/// Render a pitch band for provenance notes.
fn pitch_label(pitch: PitchBand) -> &'static str {
    match pitch {
        PitchBand::Child => "child",
        PitchBand::Adult => "adult",
        PitchBand::Ambiguous => "ambiguous",
    }
}

/// Render an ear verdict for provenance notes.
fn ear_label(ear: EarVerdict) -> &'static str {
    match ear {
        EarVerdict::Yes => "yes",
        EarVerdict::No => "no",
    }
}

/// The machine-verified provenance note replacing a promoted flag.
/// Carries the three signals and the known residual failure mode
/// (whispered adult speech defeats the pitch leg; ~2.5% measured).
/// Render the FA leg for a note: a number, or an honest n/a for an
/// unscorable line (never a fabricated value).
fn fa_label(score: Option<f64>) -> String {
    match score {
        Some(value) => format!("{value:.2}"),
        None => "n/a".to_owned(),
    }
}

fn provenance_note(verdict: &LineVerdict) -> String {
    format!(
        "machine-verified placement (fa={}, pitch={}, ear={}); \
         residual risk: whispered adult speech (~2.5% measured)",
        fa_label(verdict.fa_mean_score),
        pitch_label(verdict.pitch),
        ear_label(verdict.ear),
    )
}

/// The review flag added to a demoted (previously confident) line.
fn demotion_note(verdict: &LineVerdict) -> String {
    format!(
        "review: machine-demoted placement (fa={}, pitch={}, ear={})",
        fa_label(verdict.fa_mean_score),
        pitch_label(verdict.pitch),
        ear_label(verdict.ear),
    )
}

/// Rewrite the flag segment of a %com tier's text into the provenance
/// note, preserving any human transcriber note ahead of it verbatim.
///
/// The merge writer appends its flag to a pre-existing human comment as
/// `<human note> ; <prefix>: <flag>`, so the flag is not always at the
/// start of the tier; it always runs from the `<prefix>:` marker to the
/// end. Returns None when the tier carries no flag (a plain human
/// comment is never touched). A bare `starts_with` predicate here
/// silently skipped four human-prefixed bundle flags (2026-07-17).
fn rewrite_flag_text(text: &str, flag_prefix: &str, verdict: &LineVerdict) -> Option<String> {
    let marker = format!("{flag_prefix}:");
    let flag_start = text.find(&marker)?;
    Some(format!(
        "{}{}",
        &text[..flag_start],
        provenance_note(verdict)
    ))
}

/// Safe wire identity: this value can only name one file in each directory.
struct SessionStem(String);

impl SessionStem {
    fn admit(session: String) -> Result<Self, MergeVerifyError> {
        if session.is_empty()
            || matches!(session.as_str(), "." | "..")
            || session
                .chars()
                .any(|c| c.is_control() || matches!(c, '/' | '\\' | ':'))
        {
            return Err(MergeVerifyError::UnsafeSession { session });
        }
        Ok(Self(session))
    }

    fn path(&self, directory: &Path) -> PathBuf {
        directory.join(format!("{}.cha", self.0))
    }
}

/// Source-bound plan: neither a raw document nor independently swappable edits
/// can enter the transform. Verdict indices are unique and admitted in range.
struct PreparedSession {
    session: SessionStem,
    source: ValidChatFile,
    verdicts: BTreeMap<UtteranceOrdinal, LineVerdict>,
}

impl PreparedSession {
    fn admit(
        session: SessionVerdicts,
        draft_dir: &Path,
        parser: &batchalign_transform::parse::TreeSitterParser,
    ) -> Result<Self, MergeVerifyError> {
        let stem = SessionStem::admit(session.session)?;
        let mut verdicts = BTreeMap::new();
        for verdict in session.lines {
            let ordinal = verdict.utterance_index;
            if verdicts.insert(ordinal, verdict).is_some() {
                return Err(MergeVerifyError::DuplicateVerdict {
                    session: stem.0,
                    ordinal,
                });
            }
        }
        let path = stem.path(draft_dir);
        let text = std::fs::read_to_string(&path).map_err(|source| {
            if source.kind() == std::io::ErrorKind::NotFound {
                MergeVerifyError::MissingSession {
                    session: stem.0.clone(),
                    path: path.clone(),
                }
            } else {
                MergeVerifyError::Io {
                    path: path.clone(),
                    source,
                }
            }
        })?;
        let source = batchalign_transform::parse_source_with_parser(parser, &text)
            .admit(TranscriptName::for_path(&path), &NullErrorSink)
            .map_err(|source| MergeVerifyError::SourceAdmission { path, source })?
            .into_valid_file();
        let utterance_count = source
            .document()
            .lines
            .iter()
            .filter(|line| matches!(line, Line::Utterance(_)))
            .count();
        if let Some(&ordinal) = verdicts.keys().find(|index| index.0 >= utterance_count) {
            return Err(MergeVerifyError::OrdinalOutOfRange {
                session: stem.0,
                ordinal,
                utterance_count,
            });
        }
        Ok(Self {
            session: stem,
            source,
            verdicts,
        })
    }

    fn apply(
        self,
        flag_prefix: &str,
        summary: &mut VerifySummary,
    ) -> Result<CheckedSession, MergeVerifyError> {
        let Self {
            session,
            source,
            verdicts,
        } = self;
        let policy = ValidationPolicy::new(
            RuleSelection::new(),
            AlignmentValidation::IncludeTierAlignment,
        );
        let mut file = source.into_unchecked();
        let mut queue = Vec::new();
        // This traversal exposes only dependent tiers to the edit operation.
        // Main tiers, headers, and other dependent tiers retain their structure.
        for (ordinal, utterance) in (&mut file.lines)
            .into_iter()
            .filter_map(|line| match line {
                Line::Utterance(u) => Some(u),
                _ => None,
            })
            .enumerate()
        {
            let Some(verdict) = verdicts.get(&UtteranceOrdinal(ordinal)) else {
                continue;
            };
            let main_before = utterance.main.clone();
            let outcome = tier_outcome(verdict);
            match outcome {
                TierOutcome::AutoTrust => {
                    summary.auto_trusted += 1;
                    for entry in &mut utterance.dependent_tiers {
                        if let DependentTier::Com(com) = &mut entry.tier {
                            for segment in &mut com.content.segments {
                                if let BulletContentSegment::Text(text) = segment
                                    && let Some(replacement) =
                                        rewrite_flag_text(&text.text, flag_prefix, verdict)
                                {
                                    text.text = replacement.into();
                                }
                            }
                        }
                    }
                }
                TierOutcome::Review => {
                    summary.reviewed += 1;
                    queue.push(queue_entry(&session.0, verdict, outcome));
                }
                TierOutcome::Hold => summary.held += 1,
                TierOutcome::Demote => {
                    summary.demoted += 1;
                    utterance
                        .dependent_tiers
                        .push(DependentTierEntry::new(DependentTier::Com(
                            ComTier::from_text(demotion_note(verdict)),
                        )));
                    queue.push(queue_entry(&session.0, verdict, outcome));
                }
                TierOutcome::Untouched => {}
            }
            if utterance.main != main_before {
                return Err(MergeVerifyError::MainTierChanged {
                    session: session.0.clone(),
                    ordinal: UtteranceOrdinal(ordinal),
                    before: format!("{main_before:?}"),
                    after: format!("{:?}", utterance.main),
                });
            }
        }
        let path = session.path(Path::new(""));
        let output = file
            .validate_construction_with_policy(
                policy,
                &NullErrorSink,
                TranscriptName::for_path(&path),
            )
            .map_err(|source| MergeVerifyError::OutputAdmission {
                session: session.0.clone(),
                source,
            })?;
        summary.sessions += 1;
        Ok(CheckedSession {
            session,
            output,
            queue,
        })
    }
}

/// Only this producer-admitted state has write permission.
struct CheckedSession {
    session: SessionStem,
    output: ValidChatFile,
    queue: Vec<QueueEntry>,
}

impl CheckedSession {
    fn write(&self, directory: &Path) -> Result<(), MergeVerifyError> {
        let path = self.session.path(directory);
        std::fs::write(&path, self.output.to_chat_string())
            .map_err(|source| MergeVerifyError::Io { path, source })
    }
}

fn queue_entry(session: &str, verdict: &LineVerdict, tier: TierOutcome) -> QueueEntry {
    QueueEntry {
        session: session.to_owned(),
        utterance_index: verdict.utterance_index,
        category: verdict.category,
        tier,
        fa_mean_score: verdict.fa_mean_score,
        pitch: verdict.pitch,
        ear: verdict.ear,
    }
}

/// Run the pass: read every session named in the verdicts document from
/// `draft_dir`, admit and transform the complete set, then write checked
/// drafts and the review queue into `out_dir`. Semantic refusal writes nothing;
/// filesystem failure during persistence can still leave partial output.
pub fn run(
    draft_dir: &Path,
    verdicts_path: &Path,
    out_dir: &Path,
    flag_prefix: &str,
) -> Result<VerifySummary, MergeVerifyError> {
    let verdicts_text =
        std::fs::read_to_string(verdicts_path).map_err(|source| MergeVerifyError::Io {
            path: verdicts_path.to_path_buf(),
            source,
        })?;
    let doc: VerdictsDoc =
        serde_json::from_str(&verdicts_text).map_err(|source| MergeVerifyError::VerdictsParse {
            path: verdicts_path.to_path_buf(),
            source,
        })?;

    let parser = crate::chat_parser();
    let mut summary = VerifySummary::default();
    let mut sessions = BTreeSet::new();
    let mut prepared = Vec::new();
    for session in doc.sessions {
        // Canonically equivalent names must not overwrite each other on
        // Unicode-normalizing or case-insensitive filesystems.
        let collision_key: String = session.session.to_lowercase().nfc().collect();
        if !sessions.insert(collision_key) {
            return Err(MergeVerifyError::DuplicateSession {
                session: session.session,
            });
        }
        prepared.push(PreparedSession::admit(session, draft_dir, &parser)?);
    }
    let checked = prepared
        .into_iter()
        .map(|session| session.apply(flag_prefix, &mut summary))
        .collect::<Result<Vec<_>, _>>()?;
    let queue = ReviewQueue {
        entries: checked
            .iter()
            .flat_map(|session| session.queue.iter().cloned())
            .collect(),
    };

    let queue_path = out_dir.join("review-queue.json");
    let queue_json = serde_json::to_string_pretty(&queue).map_err(|source| {
        MergeVerifyError::QueueSerialize {
            path: queue_path.clone(),
            source,
        }
    })?;
    std::fs::create_dir_all(out_dir).map_err(|source| MergeVerifyError::Io {
        path: out_dir.to_path_buf(),
        source,
    })?;
    for session in &checked {
        session.write(out_dir)?;
    }
    std::fs::write(&queue_path, queue_json).map_err(|source| MergeVerifyError::Io {
        path: queue_path,
        source,
    })?;

    Ok(summary)
}

impl MergeVerifyError {
    /// Preserve producer failure taxonomy at the CLI/server presentation seam.
    pub(crate) fn is_internal(&self) -> bool {
        match self {
            Self::SourceAdmission { source, .. } => {
                crate::error::chat_admission_is_internal(source)
            }
            Self::Io { .. }
            | Self::OutputAdmission { .. }
            | Self::MainTierChanged { .. }
            | Self::QueueSerialize { .. } => true,
            Self::VerdictsParse { .. }
            | Self::MissingSession { .. }
            | Self::UnsafeSession { .. }
            | Self::DuplicateSession { .. }
            | Self::DuplicateVerdict { .. }
            | Self::OrdinalOutOfRange { .. } => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_verify_failure_taxonomy_is_not_diagnostic_prose() {
        use axum::response::IntoResponse;
        let io = crate::error::ServerError::MergeVerification(Box::new(MergeVerifyError::Io {
            path: PathBuf::from("output.cha"),
            source: std::io::Error::other("validation error in an infrastructure message"),
        }));
        assert_eq!(
            io.into_response().status(),
            axum::http::StatusCode::INTERNAL_SERVER_ERROR
        );
        let verdict = crate::error::ServerError::MergeVerification(Box::new(
            MergeVerifyError::DuplicateVerdict {
                session: "S1".into(),
                ordinal: UtteranceOrdinal(0),
            },
        ));
        assert_eq!(
            verdict.into_response().status(),
            axum::http::StatusCode::BAD_REQUEST
        );
    }
}
