//! Server error types: maps to HTTP status codes.
//!
//! Error responses use `{"detail": "..."}` to match FastAPI's `HTTPException`.

#[cfg(feature = "server")]
use axum::http::StatusCode;
#[cfg(feature = "server")]
use axum::response::{IntoResponse, Response};

use crate::api::JobId;
use crate::chat_ops::CacheKey;
use crate::chat_ops::fa::coordinates::WindowFault;
use crate::media::probe::ProbeError;
use crate::media::window::EmptySegment;
pub use crate::revai::RetainedRevLanguageRejection;

/// Why a recording's duration could not be established.
///
/// Typed rather than a rendered string because the variants ask for different
/// people. A file ffprobe refuses, or one that probes as zero length, is the
/// submitter's to fix and says so; a host without ffprobe, or a window over
/// the whole file that does not fit it, is the operator's. While this was a
/// `String` every case classified as an internal error, so a folder of empty
/// audio files told its owner to restart the job, and each restart failed the
/// same way.
#[derive(Debug, thiserror::Error)]
pub enum RecordingDurationError {
    /// ffprobe could not answer for this file. Whose problem that is belongs
    /// to the probe error's own variant; its message already names the file.
    #[error(transparent)]
    Probe(ProbeError),
    /// The whole recording is not a valid window over itself, which a
    /// recording of positive length always is: an internal invariant.
    #[error("the whole recording is not a valid window over itself: {0}")]
    WholeFileWindow(WindowFault),
}

/// Whose problem a [`RecordingDurationError`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecordingDurationFault {
    /// The submitted media: the person who sent it can fix it; a retry cannot.
    Media,
    /// The host or this build: the operator's.
    Host,
}

impl RecordingDurationError {
    /// Whose problem this is.
    ///
    /// The one question classification asks, answered by an exhaustive match
    /// so a new probe failure has to take a side.
    pub(crate) fn fault(&self) -> RecordingDurationFault {
        match self {
            Self::Probe(
                ProbeError::Refused { .. }
                | ProbeError::Unreadable { .. }
                | ProbeError::NoAudioStream { .. }
                | ProbeError::UnmeasuredContainer { .. }
                | ProbeError::LengthNotStated { .. }
                | ProbeError::PacketsWithoutDuration { .. }
                | ProbeError::EmptyAudio { .. },
            ) => RecordingDurationFault::Media,
            Self::Probe(ProbeError::FfprobeMissing { .. } | ProbeError::Spawn { .. })
            | Self::WholeFileWindow(_) => RecordingDurationFault::Host,
        }
    }
}

/// Non-empty forced-alignment cache misses behind `--require-media-cache`.
///
/// Construction requires a head index, so the error cannot represent the
/// contradictory state "required evidence is unavailable, but nothing is
/// missing." The remaining indices retain the pipeline's request order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingForcedAlignmentEvidence {
    request_indices: Vec<usize>,
}

impl MissingForcedAlignmentEvidence {
    /// The missed FA requests (dispatch units), first one separate so the
    /// list cannot be empty. Requests, not groups: the cache is keyed per
    /// request, and an anchored group is several requests.
    pub(crate) fn new(first_request: usize, remaining: impl IntoIterator<Item = usize>) -> Self {
        let request_indices = std::iter::once(first_request).chain(remaining).collect();
        Self { request_indices }
    }

    /// FA request ordinals whose evidence was absent from the reusable cache.
    pub fn request_indices(&self) -> &[usize] {
        &self.request_indices
    }
}

impl std::fmt::Display for MissingForcedAlignmentEvidence {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "required forced-alignment evidence is unavailable for requests {:?}; \
             --require-media-cache prevented new inference",
            self.request_indices
        )
    }
}

/// One absent speaker-evidence entry behind `--require-media-cache`.
///
/// The content-derived key remains a [`CacheKey`] rather than an arbitrary
/// string, so an actionable replay refusal cannot accidentally cite a path,
/// job id, or unrelated cache namespace as the missing evidence identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingSpeakerEvidence {
    cache_key: CacheKey,
}

/// One absent Rev.AI transcript-evidence entry behind cache-only mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingRevAsrEvidence {
    cache_key: CacheKey,
}

impl MissingRevAsrEvidence {
    pub(crate) fn new(cache_key: CacheKey) -> Self {
        Self { cache_key }
    }

    /// Content identity whose reusable Rev.AI response was absent.
    pub fn cache_key(&self) -> &CacheKey {
        &self.cache_key
    }
}

impl std::fmt::Display for MissingRevAsrEvidence {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "required Rev.AI evidence is unavailable for cache key {}; \
             --require-media-cache prevented a new provider call",
            self.cache_key
        )
    }
}

impl MissingSpeakerEvidence {
    pub(crate) fn new(cache_key: CacheKey) -> Self {
        Self { cache_key }
    }

    /// Content identity whose reusable speaker evidence was absent.
    pub fn cache_key(&self) -> &CacheKey {
        &self.cache_key
    }
}

impl std::fmt::Display for MissingSpeakerEvidence {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "required speaker evidence is unavailable for cache key {}; \
             --require-media-cache prevented new inference",
            self.cache_key
        )
    }
}

/// An outstanding source-bound timing obligation. This is missing evidence,
/// not invalid submitted CHAT or permission to write an untimed result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingTimingRegenerationEvidence {
    header_span: talkbank_model::Span,
    origin: TimingObligationOrigin,
    reason: TimingRegenerationFailure,
}

/// Why an admitted source owes timing before its `@Media` linkage can be
/// written. Both leave the same obligation (the output must carry timing,
/// because a linked declaration without timing is E544), and the message
/// differs: one source lost timing it had, the other never had any.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimingObligationOrigin {
    /// Recorded word timing was unusable and Chatter's admission removed it,
    /// so alignment must regenerate it.
    DiscardedWordTiming,
    /// The source declares linked media (`@Media: name, audio`, no status)
    /// and its retained model carries no timing: the state before a first
    /// alignment, which alignment exists to leave. (A `%wor` tier Chatter
    /// could not read is removed before that check, so timing it may have
    /// carried is not seen; the obligation is the same either way.)
    NeverTimed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TimingRegenerationFailure {
    NoRequest {
        line_idx: batchalign_transform::decisions::LineIdx,
        window: batchalign_transform::decisions::RefusedWindow,
    },
    NoRestoredTiming,
}

impl MissingTimingRegenerationEvidence {
    /// Location of the source declaration whose obligation remains outstanding.
    pub fn header_span(&self) -> talkbank_model::Span {
        self.header_span
    }

    /// Only a checked pending source and an actually refused empty grouping
    /// can establish this disposition. An unexplained empty plan stays internal.
    pub(crate) fn from_refused_grouping(
        obligation: &talkbank_model::validation::MediaTimingObligation,
        origin: TimingObligationOrigin,
        grouping: &crate::chat_ops::fa::Grouping,
        timing: talkbank_model::model::TranscriptTimingEvidence<'_>,
    ) -> Option<Self> {
        use batchalign_transform::decisions::{DecisionStrategy, FaStrategy};
        if !grouping.groups.is_empty()
            || matches!(
                timing,
                talkbank_model::model::TranscriptTimingEvidence::Recorded(_)
            )
        {
            return None;
        }
        grouping
            .decisions
            .iter()
            .find_map(|decision| match &decision.strategy {
                DecisionStrategy::Fa(FaStrategy::WindowRefused(window)) => Some(Self {
                    header_span: obligation.header_span(),
                    origin,
                    reason: TimingRegenerationFailure::NoRequest {
                        line_idx: decision.line_idx,
                        window: *window,
                    },
                }),
                _ => None,
            })
    }

    /// The source-admission owner checks the actual attempted output, not a
    /// diagnostic string or a claimed inference success.
    pub(crate) fn from_untimed_output(
        obligation: &talkbank_model::validation::MediaTimingObligation,
        origin: TimingObligationOrigin,
        timing: talkbank_model::model::TranscriptTimingEvidence<'_>,
    ) -> Option<Self> {
        matches!(
            timing,
            talkbank_model::model::TranscriptTimingEvidence::Absent
        )
        .then(|| Self {
            header_span: obligation.header_span(),
            origin,
            reason: TimingRegenerationFailure::NoRestoredTiming,
        })
    }
}

impl std::fmt::Display for MissingTimingRegenerationEvidence {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Exhaustive over both axes: a new origin or failure must say what
        // the user can change, not inherit another case's sentence.
        match (&self.reason, self.origin) {
            (
                TimingRegenerationFailure::NoRequest { line_idx, window },
                TimingObligationOrigin::DiscardedWordTiming,
            ) => write!(
                formatter,
                "timing regeneration has no admissible alignment request at transcript entry {}: {}; \
                 recover narrower acoustic timing with a compatible UTR backend or supply corrected timing; \
                 no output was written",
                line_idx.raw() + 1,
                window
            ),
            (
                TimingRegenerationFailure::NoRequest { line_idx, window },
                TimingObligationOrigin::NeverTimed,
            ) => write!(
                formatter,
                "alignment has no admissible request at transcript entry {}: {}; the transcript's \
                 @Media header declares it linked, and a linked transcript must carry timing (E544); \
                 recover narrower acoustic timing with a compatible UTR backend, or add `, unlinked` \
                 to the @Media header to have the transcript written without timing; no output was written",
                line_idx.raw() + 1,
                window
            ),
            (
                TimingRegenerationFailure::NoRestoredTiming,
                TimingObligationOrigin::DiscardedWordTiming,
            ) => formatter.write_str(
                "timing regeneration produced no retained timing evidence; the source media linkage \
                 obligation remains outstanding; use a compatible timing backend or supply corrected \
                 timing; no output was written",
            ),
            (TimingRegenerationFailure::NoRestoredTiming, TimingObligationOrigin::NeverTimed) => {
                formatter.write_str(
                    "alignment produced no timing, and the transcript's @Media header declares it \
                     linked, which requires timing (E544); use a compatible timing backend, or add \
                     `, unlinked` to the @Media header to have the transcript written without timing; \
                     no output was written",
                )
            }
        }
    }
}

/// The source's `@Media` declaration cannot be the subject of alignment.
///
/// Decided at input admission, before media is resolved or any inference
/// runs, from the declaration alone: every case is something the user changes
/// in the transcript's header, and resubmitting unchanged can never help.
/// Each message names the change.
///
/// The cases mirror the preconditions of Chatter's
/// `batchalign_transform::media_timing::reconcile_media_timing`, the
/// transition that turns aligned output into linked media. Deciding them here
/// is what lets that transition run only on a declaration known to admit it.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AlignmentMediaRefusal {
    /// No `@Media` header. CHAT requires one on any timed transcript (E752),
    /// and align does not author it: the media type is a declaration about
    /// the recording that the media search does not establish.
    #[error(
        "the transcript has no @Media header, so the timing align writes would name no \
         recording; add the header `@Media:\t{suggested}, audio, unlinked` (or `video` for a \
         video recording) after the @ID headers and resubmit; `unlinked` is CHAT's mark for a \
         transcript not yet linked to its media, and align removes it once it writes timing"
    )]
    Undeclared {
        /// The name the header must carry: the transcript's own file name,
        /// as CHAT requires (E531).
        suggested: SuggestedMediaName,
    },
    /// Linked `@Media`, no timing anywhere, and `--main-bullets exact`:
    /// `exact` keeps every utterance without a bullet untimed and admits no
    /// UTR, so the timing a linked declaration requires (E544) can never be
    /// written.
    #[error(
        "the @Media header declares the transcript linked but it has no timing, and \
         `--main-bullets exact` keeps every utterance without a bullet untimed, so no timing \
         could be written; run with `--main-bullets derive` or `keep`, or add `, unlinked` to \
         the @Media header"
    )]
    LinkedUntimedUnderExactBullets,
    /// More than one `@Media` header leaves the timeline ambiguous.
    ///
    /// Not reached through align's admission today: Chatter's validation
    /// refuses a second `@Media` (E501) first, as a validation failure with
    /// its own message. Kept so the decision stays total over what the
    /// transition refuses.
    #[error(
        "the transcript has {count} @Media headers; a transcript links to one recording, so \
         keep only the header naming it and resubmit"
    )]
    Multiple {
        /// Number of `@Media` headers found.
        count: usize,
    },
    /// The declaration says the recording is missing, as a media type
    /// (`@Media: name, missing`) or a status (`, missing`).
    #[error(
        "the @Media header declares the recording `{media}` missing, so there is nothing to \
         align against; if the recording exists, change the header to \
         `@Media:\t{media}, audio, unlinked` (or `video`) and resubmit"
    )]
    DeclaredMissing {
        /// The declared media name, as written.
        media: String,
    },
    /// `notrans` declares the recording untranscribed, which contradicts
    /// aligning a transcript to it.
    #[error(
        "the @Media header marks the recording `{media}` as not transcribed (`notrans`), which \
         contradicts aligning this transcript to it; replace `notrans` with `unlinked` and resubmit"
    )]
    NotTranscribed {
        /// The declared media name, as written.
        media: String,
    },
    /// A media type or status CHAT does not define. Not reached through
    /// align's admission today: Chatter's validation refuses these first
    /// (E535, E536). Kept so the match over Chatter's enums stays exhaustive.
    #[error(
        "the @Media header's {field} `{as_written}` is not one CHAT defines; use `audio` or \
         `video`, optionally followed by `unlinked`, and resubmit"
    )]
    Unsupported {
        /// Which field: `media type` or `status`.
        field: &'static str,
        /// The token as written.
        as_written: String,
    },
}

/// The name a missing `@Media` header must carry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SuggestedMediaName {
    /// The transcript's file name without its extension.
    Transcript(String),
    /// The transcript was submitted without a file name.
    Unnamed,
}

impl std::fmt::Display for SuggestedMediaName {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transcript(stem) => formatter.write_str(stem),
            Self::Unnamed => formatter.write_str("<transcript file name without extension>"),
        }
    }
}

/// Alignment completion is distinct from CHAT validity and provider success.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlignmentCompletionFailure {
    /// Final lexical structure no longer corresponds to its admitted source.
    SourceChanged {
        /// First utterance whose required word structure differs.
        utterance: talkbank_model::UtteranceIdx,
    },
    /// An utterance the transcript marks as not in the recording came out
    /// with timing it could only have been given by the pipeline: a timed
    /// word, or a bullet other than the one the input gave it under a policy
    /// that keeps given bullets.
    OffRecordUtteranceTimed {
        /// The utterance.
        utterance: talkbank_model::UtteranceIdx,
    },
}

impl std::fmt::Display for AlignmentCompletionFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SourceChanged { utterance } => write!(
                formatter,
                "alignment output no longer corresponds to admitted lexical structure at utterance {}; no output was written",
                utterance.raw() + 1
            ),
            Self::OffRecordUtteranceTimed { utterance } => write!(
                formatter,
                "alignment output timed utterance {}, which the transcript marks as not in the recording; no output was written",
                utterance.raw() + 1
            ),
        }
    }
}

impl std::error::Error for AlignmentCompletionFailure {}

#[cfg(all(test, feature = "server"))]
mod alignment_completion_tests {
    use super::*;

    #[test]
    fn alignment_completion_has_typed_http_and_retry_dispositions() {
        use crate::runner::util::{classify_server_error, is_retryable_worker_failure};
        use crate::scheduling::FailureCategory;
        // Untimed words are no longer a refusal (they are a partial result,
        // written and diagnosed); a changed lexical structure still is, and
        // it is our own fault, never retried.
        // A timed utterance the transcript marks as not in the recording is
        // the same kind of fault: the pipeline invented a placement.
        for (failure, category, status) in [
            (
                AlignmentCompletionFailure::SourceChanged {
                    utterance: talkbank_model::UtteranceIdx::new(0),
                },
                FailureCategory::System,
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
            (
                AlignmentCompletionFailure::OffRecordUtteranceTimed {
                    utterance: talkbank_model::UtteranceIdx::new(0),
                },
                FailureCategory::System,
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
        ] {
            let error = ServerError::from(failure);
            assert_eq!(classify_server_error(&error), category);
            assert_eq!(error.status_code(), status);
            assert!(!is_retryable_worker_failure(category));
        }
    }
}

/// Typed family of intentional required-evidence precondition refusals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MissingRequiredEvidence {
    /// One or more forced-alignment groups were absent.
    ForcedAlignment(MissingForcedAlignmentEvidence),
    /// Speaker evidence for one semantic media request was absent.
    Speaker(MissingSpeakerEvidence),
    /// Raw Rev.AI transcript evidence for one provider request was absent.
    RevAsr(MissingRevAsrEvidence),
    /// Source timing was discarded and its regeneration remains unfulfilled.
    TimingRegeneration(MissingTimingRegenerationEvidence),
}

impl std::fmt::Display for MissingRequiredEvidence {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ForcedAlignment(missing) => missing.fmt(formatter),
            Self::Speaker(missing) => missing.fmt(formatter),
            Self::RevAsr(missing) => missing.fmt(formatter),
            Self::TimingRegeneration(missing) => missing.fmt(formatter),
        }
    }
}

/// Detail of a single file that conflicts with an already-active job.
///
/// Returned inside the `conflicts` array of a [`ServerError::JobConflict`]
/// 409 response body. Each entry identifies exactly which file, in which
/// existing job, caused the conflict. Callers can use this to show the user
/// which files need to finish (or be cancelled) before resubmission.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConflictDetail {
    /// The filename that overlaps between the new submission and an active job.
    pub filename: crate::api::DisplayPath,
    /// The `job_id` of the existing active job that owns this file.
    pub job_id: JobId,
    /// The command the conflicting job is running.
    pub command: crate::api::ReleasedCommand,
    /// The current status of the conflicting job.
    pub status: crate::api::JobStatus,
}

/// Whether an upstream ASR provider failure is worth retrying.
///
/// A public mirror of the provider client's own verdict. The client's error
/// type is crate-private, so the typed fact travels rather than the type; what
/// must not travel is a string the control plane then has to re-interpret.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsrProviderDisposition {
    /// The same request may succeed later (transport fault, 5xx, exhausted
    /// upload retries).
    Transient,
    /// Repeating the request cannot help (4xx, a job the provider failed, an
    /// undecodable response).
    Terminal,
}

impl std::fmt::Display for AsrProviderDisposition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transient => write!(f, "transient"),
            Self::Terminal => write!(f, "terminal"),
        }
    }
}

/// All errors that can occur in the server.
///
/// Each variant maps to an HTTP status code via [`IntoResponse`]. The response
/// body is always `{"detail": "..."}` (matching FastAPI's `HTTPException`
/// convention), except for [`JobConflict`](Self::JobConflict) which includes
/// a structured `conflicts` array.
#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    /// The provider response was retained but its language was not admitted.
    /// The payload can only be issued after successful evidence persistence.
    /// HTTP 502: provider evidence was unusable, not malformed client input.
    #[error(transparent)]
    UnresolvedAsrLanguage(RetainedRevLanguageRejection),
    /// A database operation failed (schema migration, insert, query, etc.).
    ///
    /// **HTTP 500.** Callers should retry or report the error. Typically
    /// indicates a corrupt DB, disk-full condition, or schema mismatch.
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),

    /// A database migration failed.
    ///
    /// **HTTP 500.** Indicates a schema version mismatch or corrupt migration state.
    #[error("migration error: {0}")]
    Migration(#[from] sqlx::migrate::MigrateError),

    /// Persisted structured data could not be serialized or deserialized.
    ///
    /// **HTTP 500.** Indicates an internal schema/shape mismatch or corrupt
    /// stored JSON payload in SQLite.
    #[error("persistence error: {0}")]
    Persistence(String),

    /// A stored job lease names no lease that could have been held: part of
    /// one, or one that does not expire after its heartbeat.
    ///
    /// **HTTP 500.** Corrupt stored state, refused where it is read.
    #[error("persistence error: {0}")]
    StoredLease(#[from] crate::db::StoredLeaseError),

    /// Typed ASR-to-CHAT assembly failure, including requested diagnostics.
    #[error(transparent)]
    TranscriptBuild(#[from] batchalign_transform::build_chat::TranscriptBuildError),

    /// Input admission refused the source's `@Media` declaration for
    /// alignment, before any media was resolved or inference ran.
    ///
    /// **HTTP 400.** The user changes the header; the message says how.
    #[error(transparent)]
    AlignmentMedia(#[from] AlignmentMediaRefusal),

    /// Chatter's complete admission refused a never-timed source that its
    /// timing-regeneration admission, running the same rules with only E544
    /// deferred, accepts with nothing deferred.
    ///
    /// **HTTP 500.** The two admissions disagree: a tool fault, not the input's.
    #[error(
        "internal fault: Chatter's complete and timing-regeneration admissions disagree about \
         this transcript; no output was written"
    )]
    AdmissionDisagreement,

    /// Under `--main-bullets keep` or `exact`, the bullets the input gave
    /// these utterances (not in the recording, e.g. `[+ diary]`) conflict
    /// with the timing aligned around them, so the output cannot be valid.
    ///
    /// **HTTP 400.** The input and the requested policy conflict: such a
    /// bullet is kept exactly but is never an anchor for its neighbours.
    #[error(
        "--main-bullets keep: the bullets the input gave utterance(s) {utterances:?}, which \
         are not in the recording, conflict with the timing aligned around them; nothing \
         was written. Run with --main-bullets derive, or correct or remove those bullets"
    )]
    KeptOffRecordBulletConflict {
        /// The utterances, counting from 1.
        utterances: Vec<usize>,
    },

    /// Chatter's tier-replacement admission selected a replacement other
    /// than the one the planner chose from the headers.
    ///
    /// **HTTP 500.** The planner's choice is the admission's input, so a
    /// different selection is a tool fault, never the transcript's.
    #[error(
        "internal fault: Chatter admitted tier replacement {admitted:?} where {planned:?} was \
         planned; no output was written"
    )]
    ReplacementPlanContradicted {
        /// What the planner chose (`None`: retain every tier).
        planned: Option<talkbank_parser::ReplacementTiers>,
        /// What the admission reports it selected.
        admitted: Option<talkbank_parser::ReplacementTiers>,
    },

    /// Aligned output could not take the media/timing transition its source
    /// declaration was admitted for.
    ///
    /// **HTTP 500.** Every condition this transition refuses is decided at
    /// input admission ([`Self::AlignmentMedia`]), so reaching it means the
    /// alignment itself changed the declaration: an internal fault, never
    /// the user's input. Deliberately not `#[from]`: its one producer is the
    /// admitted transition in `fa::input`, named there.
    #[error("internal fault: aligned output contradicts its admitted @Media declaration: {0}")]
    MediaTiming(batchalign_transform::media_timing::MediaTimingError),

    /// Required morphology could not be completely applied. This is a failed
    /// analysis/transform, not a declaration that the submitted CHAT is invalid.
    /// **HTTP 500.** Preserve the typed refusal and do not automatically retry.
    #[error("Result injection failed: {0}")]
    MorphosyntaxInjection(#[from] batchalign_transform::morphosyntax::InjectionError),

    /// The proposed segmentation cannot preserve the admitted CHAT structure.
    /// This is a failed transform, not a declaration that its input is invalid.
    /// HTTP 500; retain the typed refusal and do not automatically retry.
    #[error(transparent)]
    UtterancePartition(#[from] batchalign_transform::utseg::UtsegApplyRefusal),

    /// Valid CHAT requests analysis unavailable from the configured backend.
    /// HTTP 412, non-retryable; never advise changing truthful source language.
    #[error(transparent)]
    AnalysisUnavailable(#[from] crate::morphosyntax::AnalysisUnavailable),

    /// Serialized pipeline output failed parsing before provenance publication.
    ///
    /// **HTTP 500.** Preserve diagnostics from the generated output rather than
    /// blaming the submitted request or retrying a worker that already finished.
    #[error("pipeline output could not receive provenance: {0}")]
    OutputParse(#[source] talkbank_model::ParseErrors),

    /// Required evidence is unavailable, either in cache-only mode or because
    /// source-bound timing regeneration could not admit an acoustic request.
    ///
    /// This is an intentional, actionable precondition refusal, not corrupt
    /// persistence and not an internal system failure.
    #[error("{0}")]
    RequiredEvidenceUnavailable(MissingRequiredEvidence),

    /// CHAT validity is not complete alignment. Missing timing is unavailable
    /// evidence; loss of source correspondence is an internal producer fault.
    #[error(transparent)]
    AlignmentCompletion(#[from] AlignmentCompletionFailure),

    /// Transcription produced no words, so there is no transcript to write.
    ///
    /// Neither a malformed request nor an internal fault: the run reached the
    /// end and had nothing in it. Reported rather than written, because the
    /// headers-only CHAT file this used to produce is indistinguishable from a
    /// transcript of a silent recording and was delivered as a success.
    #[error(transparent)]
    EmptyTranscription(#[from] EmptyTranscription),

    /// The requested `job_id` does not exist in the [`JobStore`](crate::store::JobStore).
    ///
    /// **HTTP 404.** Callers should verify the job ID. The job may have been
    /// pruned after expiry (`job_ttl_days`) or explicitly deleted.
    #[error("job {0} not found")]
    JobNotFound(JobId),

    /// A new job submission overlaps with files already being processed by
    /// an active job from the same submitter.
    ///
    /// **HTTP 409.** The `conflicts` field lists each overlapping file and
    /// the active job that owns it. Callers should wait for the conflicting
    /// job to finish, cancel it, or remove the overlapping files from the
    /// new submission.
    #[error("{message}")]
    JobConflict {
        /// Human-readable description of the conflict.
        message: String,
        /// Per-file details showing which active jobs overlap.
        conflicts: Vec<ConflictDetail>,
    },

    /// An operation (e.g. restart, delete) was attempted on a job that is
    /// still queued or running and has not yet reached a terminal state.
    ///
    /// **HTTP 409.** Callers should cancel the job first, or wait for it
    /// to complete before retrying the operation.
    #[error("job {0} is not in a terminal state")]
    JobNotTerminal(JobId),

    /// A result file was requested (e.g. `GET /jobs/{id}/results/{filename}`)
    /// but the file does not exist on disk.
    ///
    /// **HTTP 404.** The job may not have produced output for this file,
    /// or the staging directory may have been cleaned up.
    #[error("file not found: {0}")]
    FileNotFound(String),

    /// A result file was requested but the file has not finished processing
    /// yet (still queued or in progress).
    ///
    /// **HTTP 409.** Callers should poll the job status and retry once the
    /// file reaches `"done"` status.
    #[error("file not ready: {0}")]
    FileNotReady(String),

    /// The submitted command name is not recognized by the server (not in
    /// the set of worker-advertised capabilities).
    ///
    /// **HTTP 400.** Callers should check `GET /health` for the list of
    /// supported `capabilities` and resubmit with a valid command.
    #[error("unknown command: {0}")]
    UnknownCommand(String),

    /// A request failed input validation (e.g. empty filename list, missing
    /// required fields, invalid language code).
    ///
    /// **HTTP 400.** Callers should fix the request payload and resubmit.
    #[error("validation error: {0}")]
    Validation(String),

    /// Offline verification retains typed input refusal versus producer failure.
    #[error(transparent)]
    MergeVerification(Box<crate::merge_verify::MergeVerifyError>),

    /// Speaker evidence could not be produced. Original worker failures retain
    /// their retry/memory classification; only enrollment refusals are input errors.
    #[error(transparent)]
    SpeakerIdentity(crate::chat_ops::speaker_identity::SpeakerIdentityFailure),

    /// The packaged embedding model identity cannot be established.
    #[error(transparent)]
    SpeakerModelManifest(#[from] crate::chat_ops::speaker_identity::InvalidModelManifest),

    /// Preparing the shared recording failed before any embedding request.
    #[error(transparent)]
    SpeakerAudioPreparation(
        #[from] crate::worker::speaker_embedding_request_v2::SpeakerEmbeddingRequestBuildErrorV2,
    ),

    /// Complete CHAT admission refused the input. The producing failure keeps
    /// internal tool faults distinct from CHAT invalidity.
    #[error("CHAT pre-validation failed: {}", chat_admission_details(.0))]
    ChatAdmission(batchalign_transform::ValidatedParseError),

    /// Source-bound tier replacement refused retained input, without treating
    /// an internal producer failure as invalid CHAT.
    #[error("CHAT replacement admission failed: {}", chat_replacement_details(.0))]
    ChatReplacementAdmission(talkbank_parser::ReplacementFailure),

    /// A producer could not establish admission of its output. This is a tool
    /// failure, not a verdict that the caller submitted invalid CHAT.
    #[error("{command} output admission failed: {details}")]
    OutputAdmission {
        /// The command whose produced output could not be admitted.
        command: crate::api::ReleasedCommand,
        /// Why, typed: the judgement's findings, or the producer's statement.
        details: OutputAdmissionRefusal,
    },

    /// A Python worker process failed (crashed, timed out, or returned an
    /// error response over the stdio IPC protocol).
    ///
    /// **HTTP 500.** The worker pool will automatically restart crashed
    /// workers. Callers can retry the job.
    #[error("worker error: {0}")]
    Worker(#[from] crate::worker::error::WorkerError),

    /// An in-process native inference engine (whisper.cpp) failed for a
    /// non-input reason: model resolution/download, build configuration,
    /// or an internal invariant.
    ///
    /// **HTTP 500.** Distinct from `Validation` so infrastructure
    /// failures are never reported to clients as bad input.
    #[error("whisper engine error: {0}")]
    WhisperEngine(String),

    /// A filesystem I/O operation failed (reading/writing staging files,
    /// creating directories, etc.).
    ///
    /// **HTTP 500.** Typically indicates a permissions problem or full disk.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The system's available memory is below the critical threshold
    /// (configurable via `ServerConfig.memory_gate_mb`, default 2048 MB)
    /// and no idle workers can be reused for the requested command.
    ///
    /// **HTTP 500.** The memory gate polls every 5 seconds and waits up to
    /// 120 seconds for memory to free up before triggering this error.
    /// Callers should wait and retry, or reduce the number of concurrent
    /// jobs.
    #[error("memory pressure: {0}")]
    MemoryPressure(String),

    /// A per-file text-workflow failure the control plane has ALREADY
    /// classified, carried with its verdict.
    ///
    /// `run_text_pipeline` used to render its typed `TextWorkflowFileError`
    /// into `Validation(String)`, which retyped every PER-ITEM PROVIDER
    /// failure as bad input: `TextWorkflowFileError::ItemErrors` classifies as
    /// `ProviderTerminal`, and `Validation` does not, so the runner's retry
    /// and reporting policy could not see the case it exists for. The verdict
    /// travels with the message now instead of being re-derived from it by
    /// `classify_server_error`.
    ///
    /// Build one only with [`ServerError::from_classified_failure`], so the
    /// category is always the producing error's own answer and never a
    /// caller's guess.
    #[error("{message}")]
    ClassifiedFailure {
        /// The control plane's verdict for this failure.
        category: crate::scheduling::FailureCategory,
        /// The failure, rendered by the workflow that produced it.
        message: String,
    },

    /// An FA audio segment request produced no whole sample frames.
    ///
    /// The segment says WHY, in `EmptyReason`, issued by the code that
    /// measured the decode's byte length. This doc claimed until 2026-09-07
    /// that the cause was "the requested time window falls past the end of the
    /// source audio file", which nothing had established: the measuring party
    /// holds a byte length and no source duration, and `fa::transport`'s own
    /// handler says the opposite in a comment ("that does not prove the window
    /// is past EOF; very short in-range windows can do this"). The reason is
    /// evidence now, so no reader has to guess it off a log line.
    ///
    /// This is not a fatal error at the file level: the FA pipeline handles it
    /// by leaving the affected group's words unaligned rather than aborting.
    /// It is surfaced as a `ServerError` variant so the transport layer can
    /// match on it without inspecting error message strings.
    #[error("empty FA audio segment {0}")]
    EmptyFaAudioSegment(EmptySegment),

    /// The runner was asked to execute a job that is not present in this
    /// server's local `JobStore`.
    ///
    /// **HTTP 500: internal consistency error, not a 404.** The local
    /// server runner was asked to execute a job that its own `JobStore` does
    /// not contain. Surfacing this error means local state drifted or was
    /// concurrently truncated while execution was being scheduled. Either case
    /// must fail loudly; silently reporting success would mask a real
    /// correctness bug.
    #[error(
        "job {0} is not in this server's local JobStore; local execution state is inconsistent"
    )]
    JobNotInLocalStore(JobId),

    /// The recording's duration could not be established, so no timing this
    /// job produces could be checked against it.
    ///
    /// **HTTP 500.** Deliberately fatal rather than degrading to an unbounded
    /// run. Timings are only meaningful relative to the audio they describe,
    /// and a pass that cannot state the audio's length cannot tell a
    /// measurement from a moment that does not exist: that is precisely how
    /// alignment output came to carry timings 28.2 seconds past the end of
    /// their own media. Forced alignment always has an audio file and the
    /// engine must read the same bytes, so failing here means something is
    /// wrong with the media, not with the request.
    #[error("cannot establish the recording's duration: {0}")]
    RecordingDuration(RecordingDurationError),

    /// A pinned Hugging Face Hub artifact refused this machine's request: a
    /// gated repository requiring accepted terms, a missing/invalid token,
    /// or no cached copy while offline. The message is the worker's own
    /// typed diagnostic, already actionable (which repository, and the two
    /// remedies), so it is surfaced verbatim.
    ///
    /// **HTTP 500.** Distinct from [`Validation`](Self::Validation): this is
    /// a configuration/credential condition on the SERVER's machine, not a
    /// malformed request, and must never be reported to the caller as bad
    /// input. Before 2026-09-02 every worker-protocol V2 speaker parse
    /// failure, this one included, collapsed into `Validation`, which the
    /// dashboard renders as "pipeline bug, filed automatically" even though
    /// nothing about batchalign was broken.
    #[error("model access required: {0}")]
    ModelAccessDenied(String),

    /// An upstream ASR provider (Rev.AI) failed.
    ///
    /// **HTTP 502.** Distinct from [`Validation`](Self::Validation), and for
    /// the same reason [`ModelAccessDenied`](Self::ModelAccessDenied) is:
    /// nothing about the request was wrong. Until 2026-09-03 every Rev.AI
    /// failure was flattened with `to_string()` into `Validation`, so 94
    /// uploads killed by a dropped connection reached the dashboard as
    /// `error_category: validation`, which reads as "you sent bad input" and
    /// sent a whole shift looking for a broken daemon.
    ///
    /// `disposition` is the typed half and is what the control plane acts on;
    /// `message` is the operator-facing rendering of the provider's own error,
    /// cause chain included.
    #[error("ASR provider failure ({disposition}): {message}")]
    AsrProvider {
        /// Whether repeating the request could plausibly succeed.
        disposition: AsrProviderDisposition,
        /// The provider error rendered with its full cause chain.
        message: String,
    },

    /// A worker-protocol V2 retry loop
    /// ([`crate::infer_retry::dispatch_execute_v2_with_retry_and_progress`])
    /// observed job cancellation before returning a definite outcome.
    ///
    /// **Internal signal, not expected to reach HTTP**: this dispatch path
    /// runs from background job execution, not a request handler. Distinct
    /// from [`Worker`](Self::Worker): a cancellation is a deliberate stop,
    /// never a transient engine failure, and must never be classified by
    /// [`crate::runner::util::classify_worker_error`]'s retry logic, the
    /// same logic that a stop needs to interrupt.
    #[error("execute_v2 retry cancelled by job cancellation")]
    Cancelled,
}

/// Why a producer's output could not be admitted.
///
/// Typed so a caller that reports it can bound it: a judgement can make
/// thousands of findings, and its rendered list must not travel where a
/// bounded record belongs (see `api::StageRefusalRecord`).
#[derive(Debug, Clone)]
pub enum OutputAdmissionRefusal {
    /// The output was judged and failed: the bar and every finding, the
    /// first one its own field so the list is never empty.
    Judged {
        /// The bar it was judged against.
        bar: crate::api::JudgementBar,
        /// The first finding.
        first: crate::api::OutputFindingRecord,
        /// The others, in the order the judgement found them.
        rest: Vec<crate::api::OutputFindingRecord>,
    },
    /// The producer could not establish an output to judge (a malformed or
    /// incomplete worker result, a stage run out of order). One statement.
    Unestablished(String),
}

impl OutputAdmissionRefusal {
    /// The producer's statement of why it established no output.
    pub fn unestablished(statement: impl Into<String>) -> Self {
        Self::Unestablished(statement.into())
    }
}

impl std::fmt::Display for OutputAdmissionRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Judged { bar, first, rest } => {
                write!(f, "post-validation failed ({bar}): {first}")?;
                for finding in rest {
                    write!(f, "; {finding}")?;
                }
                Ok(())
            }
            Self::Unestablished(statement) => f.write_str(statement),
        }
    }
}

/// Which transcription stage produced nothing.
///
/// Three stages can each end with no words, and they mean different things to
/// whoever reads the failure, so the variant names the one that came up empty
/// rather than leaving a reader to guess from a message. A silence reported as
/// a completed job is the defect this type exists to prevent: before
/// 2026-09-16 the first of these wrote a headers-only transcript and the job
/// finished successfully.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum EmptyTranscription {
    /// The ASR engine returned no words at all.
    #[error(
        "the ASR engine recognized no words in this recording, so there is no transcript to \
         write. If the recording contains speech, the engine or the language selected for it \
         is at fault; if it does not, there is nothing here to transcribe."
    )]
    Asr,

    /// ASR post-processing kept no utterance from the engine's words.
    #[error(
        "ASR post-processing kept no utterance from the words the engine returned, so there \
         is no transcript to write."
    )]
    Postprocess,

    /// CHAT assembly produced no utterance line.
    #[error(
        "none of the {described} utterance(s) that reached CHAT assembly held any content, \
         so there is no transcript to write. Every token was a terminator, a separator or \
         empty text."
    )]
    ChatBuild {
        /// How many utterances reached CHAT assembly.
        described: usize,
    },
}

impl From<crate::chat_ops::speaker_identity::SpeakerIdentityFailure> for ServerError {
    fn from(error: crate::chat_ops::speaker_identity::SpeakerIdentityFailure) -> Self {
        use crate::chat_ops::speaker_identity::{
            EmbeddingInferenceFailure, SpeakerIdentityFailure,
        };
        match error {
            // The audio shell's worker-retry capability requires the original
            // worker variant, not merely an equal rendered message or category.
            SpeakerIdentityFailure::Inference(EmbeddingInferenceFailure::Dispatch(worker)) => {
                Self::Worker(worker)
            }
            other => Self::SpeakerIdentity(other),
        }
    }
}

pub(crate) fn chat_admission_is_internal(
    error: &batchalign_transform::ValidatedParseError,
) -> bool {
    use batchalign_transform::ValidatedParseError;
    match error {
        ValidatedParseError::InternalFailure { .. } => true,
        ValidatedParseError::Validation(failure) => failure.has_internal_failure(),
        ValidatedParseError::Parse(_) => false,
    }
}

fn chat_admission_details(error: &batchalign_transform::ValidatedParseError) -> String {
    match error {
        batchalign_transform::ValidatedParseError::Parse(product) => product
            .diagnostics()
            .iter()
            .map(|diagnostic| format!("{} {}", diagnostic.code.as_str(), diagnostic.message))
            .collect::<Vec<_>>()
            .join("; "),
        _ => error.to_string(),
    }
}

fn chat_replacement_details(error: &talkbank_parser::ReplacementFailure) -> String {
    if error.diagnostics().is_empty() {
        error.to_string()
    } else {
        error
            .diagnostics()
            .iter()
            .map(|diagnostic| format!("{} {}", diagnostic.code.as_str(), diagnostic.message))
            .collect::<Vec<_>>()
            .join("; ")
    }
}

#[cfg(feature = "server")]
impl ServerError {
    fn status_code(&self) -> StatusCode {
        match self {
            Self::TranscriptBuild(error) => {
                use batchalign_transform::build_chat::TranscriptBuildError;
                match error {
                    TranscriptBuildError::Diagnostic(_) => StatusCode::INTERNAL_SERVER_ERROR,
                    TranscriptBuildError::MissingParticipantCode(_)
                    | TranscriptBuildError::MissingPrimaryLanguage
                    | TranscriptBuildError::InvalidLanguageCode { .. }
                    | TranscriptBuildError::WordFailedValidation { .. } => StatusCode::BAD_REQUEST,
                }
            }
            Self::Database(_) | Self::Migration(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::Persistence(_) | Self::StoredLease(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::AlignmentMedia(_) | Self::KeptOffRecordBulletConflict { .. } => {
                StatusCode::BAD_REQUEST
            }
            Self::AdmissionDisagreement | Self::ReplacementPlanContradicted { .. } => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
            Self::MediaTiming(_)
            | Self::OutputParse(_)
            | Self::MorphosyntaxInjection(_)
            | Self::UtterancePartition(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::RequiredEvidenceUnavailable(_) | Self::AnalysisUnavailable(_) => {
                StatusCode::PRECONDITION_FAILED
            }
            Self::AlignmentCompletion(failure) => match failure {
                AlignmentCompletionFailure::SourceChanged { .. }
                | AlignmentCompletionFailure::OffRecordUtteranceTimed { .. } => {
                    StatusCode::INTERNAL_SERVER_ERROR
                }
            },
            // The request was fine and the server worked; the submitted media
            // yielded no words. Neither a 400 (nothing wrong with the payload)
            // nor a 500 (nothing broke).
            Self::EmptyTranscription(_) => StatusCode::UNPROCESSABLE_ENTITY,
            Self::JobNotFound(_) => StatusCode::NOT_FOUND,
            Self::JobConflict { .. } => StatusCode::CONFLICT,
            Self::JobNotTerminal(_) => StatusCode::CONFLICT,
            Self::FileNotFound(_) => StatusCode::NOT_FOUND,
            Self::FileNotReady(_) => StatusCode::CONFLICT,
            Self::UnknownCommand(_) => StatusCode::BAD_REQUEST,
            Self::Validation(_) => StatusCode::BAD_REQUEST,
            Self::MergeVerification(error) => {
                if error.is_internal() {
                    StatusCode::INTERNAL_SERVER_ERROR
                } else {
                    StatusCode::BAD_REQUEST
                }
            }
            Self::SpeakerIdentity(_) => {
                // One policy owner also drives the task's retry decision.
                if crate::runner::util::classify_server_error(self)
                    == crate::scheduling::FailureCategory::Validation
                {
                    StatusCode::BAD_REQUEST
                } else {
                    StatusCode::INTERNAL_SERVER_ERROR
                }
            }
            Self::SpeakerModelManifest(_) | Self::SpeakerAudioPreparation(_) => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
            Self::ChatAdmission(error) => {
                if chat_admission_is_internal(error) {
                    StatusCode::INTERNAL_SERVER_ERROR
                } else {
                    StatusCode::BAD_REQUEST
                }
            }
            Self::ChatReplacementAdmission(error) => {
                if error.has_internal_failure() {
                    StatusCode::INTERNAL_SERVER_ERROR
                } else {
                    StatusCode::BAD_REQUEST
                }
            }
            Self::UnresolvedAsrLanguage(_) => StatusCode::BAD_GATEWAY,
            Self::OutputAdmission { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Self::Worker(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::WhisperEngine(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::Io(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::MemoryPressure(_) => StatusCode::INTERNAL_SERVER_ERROR,
            // A per-file workflow verdict, recorded against the job's file
            // rows rather than answered over HTTP. `BAD_GATEWAY` matches the
            // `AsrProvider` arm, since the commonest carried verdict is a
            // provider failure and the caller's request was fine.
            Self::ClassifiedFailure { .. } => StatusCode::BAD_GATEWAY,
            // EmptyFaAudioSegment is an internal skip signal, never returned as HTTP.
            Self::EmptyFaAudioSegment(..) => StatusCode::INTERNAL_SERVER_ERROR,
            // JobNotInLocalStore is an internal consistency error, not a
            // user-facing 404. It only surfaces during local runner dispatch,
            // so HTTP mapping is defensive but rare.
            Self::JobNotInLocalStore(_) => StatusCode::INTERNAL_SERVER_ERROR,
            // The request was fine; the media could not be measured. A 400
            // would tell the caller to fix a payload that has nothing wrong
            // with it.
            Self::RecordingDuration(_) => StatusCode::INTERNAL_SERVER_ERROR,
            // The server's own machine lacks Hub access/credentials; the
            // caller's request was fine.
            Self::ModelAccessDenied(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::AsrProvider { .. } => StatusCode::BAD_GATEWAY,
            // Internal cancellation signal from a background retry loop;
            // never returned as an HTTP response (see the variant's doc).
            Self::Cancelled => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

#[cfg(feature = "server")]
impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let body = match &self {
            Self::UnresolvedAsrLanguage(retained) => serde_json::json!({
                "detail": self.to_string(),
                "diagnostic_key": retained.diagnostic_key(),
                "admission": "rejected_unresolved_language",
            }),
            Self::JobConflict { message, conflicts } => {
                serde_json::json!({
                    "detail": {
                        "message": message,
                        "conflicts": conflicts,
                    }
                })
            }
            _ => serde_json::json!({ "detail": self.to_string() }),
        };
        (status, axum::Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn speaker_protocol_setup_and_enrollment_failures_keep_their_typed_dispositions() {
        use super::ServerError;
        use crate::chat_ops::speaker_identity::{
            EmbeddingInferenceFailure, EnrolledLabel, InvalidModelManifest, OutsidePreparedAudio,
            SpeakerIdentityFailure, TrackAnalysisFailure,
        };
        use crate::runner::util::{classify_server_error, is_retryable_worker_failure};
        use crate::scheduling::FailureCategory;
        use crate::worker::speaker_embedding_request_v2::{
            SpeakerEmbeddingRequestBuildErrorV2, SpeakerEmbeddingResultParseError,
        };
        use axum::http::StatusCode;

        let label = || EnrolledLabel::parse("VOICE").unwrap();
        for (error, category, status) in [
            (
                ServerError::from(SpeakerIdentityFailure::EnrollmentOutsideRecording {
                    label: label(),
                    source: OutsidePreparedAudio {
                        start_ms: 20,
                        end_ms: 30,
                        recording_ms: 10,
                    },
                }),
                FailureCategory::Validation,
                StatusCode::BAD_REQUEST,
            ),
            (
                ServerError::from(SpeakerIdentityFailure::EnrollmentTooShort {
                    label: label(),
                    frames: 1,
                    minimum_frames: 2,
                }),
                FailureCategory::Validation,
                StatusCode::BAD_REQUEST,
            ),
            (
                ServerError::from(SpeakerIdentityFailure::MissingOutcome {
                    span_id: "utt:0".into(),
                }),
                FailureCategory::WorkerProtocol,
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
            (
                ServerError::from(SpeakerIdentityFailure::Inference(
                    EmbeddingInferenceFailure::from(
                        SpeakerEmbeddingResultParseError::SpanSetMismatch {
                            missing: vec!["utt:0".into()],
                            unexpected: vec![],
                        },
                    ),
                )),
                FailureCategory::WorkerProtocol,
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
            (
                ServerError::from(SpeakerIdentityFailure::Tracks(
                    TrackAnalysisFailure::EnrolledVoiceWithoutDirection { label: label() },
                )),
                FailureCategory::System,
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
            (
                ServerError::from(InvalidModelManifest {
                    detail: "same detail".into(),
                }),
                FailureCategory::System,
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
            (
                ServerError::from(SpeakerEmbeddingRequestBuildErrorV2::MissingAudioPath),
                FailureCategory::System,
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
            (
                ServerError::from(std::io::Error::from(std::io::ErrorKind::PermissionDenied)),
                FailureCategory::System,
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
        ] {
            assert_eq!(classify_server_error(&error), category);
            assert!(!is_retryable_worker_failure(category));
            let detail = error.to_string();
            let response = axum::response::IntoResponse::into_response(error);
            assert_eq!(response.status(), status);
            let bytes = axum::body::to_bytes(response.into_body(), 8192)
                .await
                .unwrap();
            let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(body["detail"].as_str(), Some(detail.as_str()));
        }
    }

    #[tokio::test]
    async fn unavailable_morphology_is_a_nonretryable_http_precondition_not_bad_chat() {
        use crate::morphosyntax::AnalysisUnavailable;
        let language = crate::api::LanguageCode3::try_new("que").unwrap();
        let error =
            super::ServerError::from(AnalysisUnavailable::admit_primary(&language).unwrap_err());
        let category = crate::runner::util::classify_server_error(&error);
        assert_eq!(
            category,
            crate::scheduling::FailureCategory::AnalysisUnavailable
        );
        assert!(!crate::runner::util::is_retryable_worker_failure(category));
        let message = error.to_string();
        assert!(message.contains("que") && message.contains("Keep truthful language declarations"));
        assert!(!message.contains("internal error") && !message.contains("Fix the @Languages"));
        let response = axum::response::IntoResponse::into_response(error);
        assert_eq!(
            response.status(),
            axum::http::StatusCode::PRECONDITION_FAILED
        );
        let bytes = axum::body::to_bytes(response.into_body(), 8192)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(
            body["detail"]
                .as_str()
                .unwrap()
                .contains("analysis unavailable")
        );
    }

    #[test]
    fn morphosyntax_completion_failure_is_system_not_bad_chat() {
        use batchalign_transform::morphosyntax::{
            InjectionError, ResponseCountMismatch, UnexpectedSentenceCount,
        };
        for injection in [
            InjectionError::ResponseCount(ResponseCountMismatch {
                expected: 1,
                actual: 0,
            }),
            InjectionError::SentenceCount {
                index: 1,
                error: UnexpectedSentenceCount { actual: 2 },
            },
        ] {
            let error = super::ServerError::from(injection);
            assert_eq!(
                error.status_code(),
                axum::http::StatusCode::INTERNAL_SERVER_ERROR
            );
            assert_eq!(
                crate::runner::util::classify_server_error(&error),
                crate::scheduling::FailureCategory::System,
            );
            assert!(matches!(
                error,
                super::ServerError::MorphosyntaxInjection(_)
            ));
        }
    }

    #[test]
    fn asr_diagnostic_failure_is_system_not_bad_transcript() {
        use batchalign_transform::build_chat::{AsrDiagnosticError, TranscriptBuildError};
        let error = super::ServerError::from(TranscriptBuildError::Diagnostic(
            AsrDiagnosticError::Write {
                path: std::path::PathBuf::from("diagnostic.json"),
                source: std::io::Error::from(std::io::ErrorKind::PermissionDenied),
            },
        ));
        assert_eq!(
            error.status_code(),
            axum::http::StatusCode::INTERNAL_SERVER_ERROR
        );
        assert_eq!(
            crate::runner::util::classify_server_error(&error),
            crate::scheduling::FailureCategory::System
        );
        assert!(
            matches!(error, super::ServerError::TranscriptBuild(TranscriptBuildError::Diagnostic(AsrDiagnosticError::Write { source, .. })) if source.kind() == std::io::ErrorKind::PermissionDenied)
        );
        let invalid = super::ServerError::from(TranscriptBuildError::MissingPrimaryLanguage);
        assert_eq!(invalid.status_code(), axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(
            crate::runner::util::classify_server_error(&invalid),
            crate::scheduling::FailureCategory::Validation
        );
    }

    use super::*;

    #[test]
    fn job_not_found_is_404() {
        let err = ServerError::JobNotFound(JobId::from("abc123"));
        assert_eq!(err.status_code(), StatusCode::NOT_FOUND);
        assert_eq!(err.to_string(), "job abc123 not found");
    }

    #[test]
    fn validation_is_400() {
        let err = ServerError::Validation("bad input".into());
        assert_eq!(err.status_code(), StatusCode::BAD_REQUEST);
    }

    /// A Hugging Face Hub access failure is a server-side configuration
    /// condition, never bad input, so it must map to 500 and never to the
    /// 400 `Validation` gets.
    #[test]
    fn model_access_denied_is_500_not_400_and_carries_the_worker_message() {
        let err = ServerError::ModelAccessDenied(
            "could not download the Hugging Face model at \
             pyannote/speaker-diarization-community-1: its repository is gated"
                .into(),
        );
        assert_eq!(err.status_code(), StatusCode::INTERNAL_SERVER_ERROR);
        assert!(err.to_string().contains("speaker-diarization-community-1"));
    }

    #[test]
    fn conflict_is_409() {
        let err = ServerError::JobConflict {
            message: "files overlap".into(),
            conflicts: vec![ConflictDetail {
                filename: crate::api::DisplayPath::from("a.cha"),
                job_id: JobId::from("j1"),
                command: crate::api::ReleasedCommand::Morphotag,
                status: crate::api::JobStatus::Running,
            }],
        };
        assert_eq!(err.status_code(), StatusCode::CONFLICT);
    }

    #[test]
    fn error_response_has_detail_field() {
        let err = ServerError::JobNotFound(JobId::from("abc"));
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    /// `JobNotInLocalStore` is an internal consistency error, NOT a
    /// user-facing missing resource. It must map to 500, not 404; a 404
    /// would suggest the job was pruned or never existed, when in fact the
    /// local runner lost sync with its own store.
    #[test]
    fn job_not_in_local_store_is_500_and_mentions_local_state() {
        let err = ServerError::JobNotInLocalStore(JobId::from("abc123"));
        assert_eq!(err.status_code(), StatusCode::INTERNAL_SERVER_ERROR);
        let rendered = err.to_string();
        assert!(
            rendered.contains("abc123"),
            "error message should include the job id: {rendered}"
        );
        assert!(
            rendered.contains("local execution state"),
            "error message should describe local execution-state drift: {rendered}"
        );
    }
}
