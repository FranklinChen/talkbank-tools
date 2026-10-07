//! Error classification and user-facing message translation.
//!
//! Raw system errors (e.g. "Broken pipe (os error 32)") must never surface
//! to end users. This module translates classified failures into messages
//! that help users understand what happened and what to do about it.

use crate::error::{AsrProviderDisposition, RecordingDurationFault, ServerError};
use crate::scheduling::FailureCategory;
use crate::worker::error::WorkerError;

/// Truncate a string to keep only the last `max_chars` characters.
///
/// Python tracebacks have the actual error at the END, so keeping the tail
/// is more useful than keeping the head.
fn truncate_tail(s: &str, max_chars: usize) -> &str {
    if s.len() <= max_chars {
        return s;
    }
    // Find a char boundary near the truncation point.
    let mut start = s.len() - max_chars;
    while !s.is_char_boundary(start) {
        start += 1;
    }
    match s[start..].find('\n') {
        Some(offset) => &s[start + offset + 1..],
        None => &s[start..],
    }
}

/// Classify worker errors into control-plane failure categories.
pub(crate) fn classify_worker_error(error: &WorkerError) -> FailureCategory {
    match error {
        WorkerError::ReadyTimeout { .. } => FailureCategory::WorkerTimeout,
        WorkerError::HealthCheckFailed(_) => FailureCategory::WorkerTimeout,
        WorkerError::ProcessExited { .. } => FailureCategory::WorkerCrash,
        // The worker went away under the request, as a crash takes it away;
        // the remedy is the same, a retry on another worker.
        WorkerError::WorkerRetired => FailureCategory::WorkerCrash,
        // The worker's stream is no longer trusted, so it is retired; a
        // retry runs on another worker.
        WorkerError::OutputNoise { .. } => FailureCategory::WorkerCrash,
        WorkerError::Timeout { .. } => FailureCategory::WorkerTimeout,
        WorkerError::Protocol(_) => FailureCategory::WorkerProtocol,
        // A report that breaks the capability contract is a protocol
        // disagreement between the server and the worker build.
        WorkerError::CapabilitiesRefused(_) => FailureCategory::WorkerProtocol,
        WorkerError::Io(_) => FailureCategory::WorkerCrash,
        WorkerError::WorkerResponse(_) => FailureCategory::ProviderTransient,
        // The worker refused the request the server built: the two disagree
        // about the protocol, and the same request is refused again.
        WorkerError::RequestRefused(_) => FailureCategory::WorkerProtocol,
        // Bootstrap-class worker errors are deterministic across retries:
        // a missing model file, a failed catalog download, or an
        // unsupported language will produce the same failure on every
        // attempt. The orchestrator must NOT retry these, historically,
        // 3-attempt retries of a deterministic Stanza catalog miss
        // produced multi-GB log explosions because each attempt dumped a
        // full Python traceback before the worker exited.
        WorkerError::Bootstrap(_) | WorkerError::RuntimeIdentityMismatch => {
            FailureCategory::WorkerBootstrap
        }
        WorkerError::MemoryGuard(_) => FailureCategory::MemoryPressure,
        // Deliberately NOT `Cancelled`, which means "cancelled intentionally"
        // by an operator and is a TERMINAL user-intent state. The server
        // shutting down is infrastructure: the work was not repudiated, and a
        // job that dies this way should be recoverable when the server comes
        // back. Marking it as a user cancellation is how a job becomes
        // permanently dead, the same shape as the 2026-07-28 `finalize_job`
        // defect where a status the caller ASKED for outranked the job's real
        // one and left a row recovery never revisits.
        WorkerError::PoolShuttingDown => FailureCategory::System,
        // A count the pool cannot reconcile is our defect; a retry would
        // wait on the same count.
        WorkerError::PoolAccountingBroken { .. } => FailureCategory::System,
        WorkerError::SpawnFailed(_)
        | WorkerError::ReadyParseFailed(_)
        | WorkerError::NativeCommand { .. }
        | WorkerError::NoWorker { .. } => FailureCategory::System,
    }
}

/// Classify server-side orchestration errors into control-plane failure categories.
pub(crate) fn classify_server_error(error: &ServerError) -> FailureCategory {
    match error {
        ServerError::TranscriptBuild(error) => {
            use batchalign_transform::build_chat::TranscriptBuildError;
            match error {
                TranscriptBuildError::Diagnostic(_) => FailureCategory::System,
                TranscriptBuildError::MissingParticipantCode(_)
                | TranscriptBuildError::MissingPrimaryLanguage
                | TranscriptBuildError::InvalidLanguageCode { .. }
                | TranscriptBuildError::WordFailedValidation { .. } => FailureCategory::Validation,
            }
        }
        ServerError::Worker(worker_error) => classify_worker_error(worker_error),
        ServerError::SpeakerIdentity(error) => {
            use crate::chat_ops::speaker_identity::{
                EmbeddingInferenceFailure, SpeakerIdentityFailure,
            };
            match error {
                SpeakerIdentityFailure::Inference(EmbeddingInferenceFailure::Dispatch(worker)) => {
                    classify_worker_error(worker)
                }
                SpeakerIdentityFailure::Inference(EmbeddingInferenceFailure::InvalidResponse(
                    _,
                ))
                | SpeakerIdentityFailure::MissingOutcome { .. } => FailureCategory::WorkerProtocol,
                SpeakerIdentityFailure::EnrollmentOutsideRecording { .. }
                | SpeakerIdentityFailure::EnrollmentTooShort { .. } => FailureCategory::Validation,
                SpeakerIdentityFailure::EmptySpeakerCode { .. }
                | SpeakerIdentityFailure::Tracks(_) => FailureCategory::System,
            }
        }
        ServerError::SpeakerModelManifest(_) | ServerError::SpeakerAudioPreparation(_) => {
            FailureCategory::System
        }
        // Native-engine failures (model resolution, build config, internal
        // invariants) are infrastructure: deterministic per attempt, so the
        // orchestrator must not retry them as if transient.
        ServerError::WhisperEngine(_) => FailureCategory::System,
        ServerError::Validation(_) => FailureCategory::Validation,
        // The transcript's own `@Media` header, decided before any work: the
        // message names the header change, and a restart cannot help.
        ServerError::AlignmentMedia(_) | ServerError::KeptOffRecordBulletConflict { .. } => {
            FailureCategory::Validation
        }
        ServerError::MergeVerification(error) => {
            if error.is_internal() {
                FailureCategory::System
            } else {
                FailureCategory::Validation
            }
        }
        ServerError::ChatAdmission(error) => {
            if crate::error::chat_admission_is_internal(error) {
                FailureCategory::System
            } else {
                FailureCategory::Validation
            }
        }
        ServerError::ChatReplacementAdmission(error) => {
            if error.has_internal_failure() {
                FailureCategory::System
            } else {
                FailureCategory::Validation
            }
        }
        // A retained provider response needs an explicit language decision,
        // not another automatic attempt or a claim of malformed client input.
        ServerError::UnresolvedAsrLanguage(_) => FailureCategory::ProviderTerminal,
        ServerError::OutputAdmission { .. } => FailureCategory::System,
        // The producing workflow already classified this one; its verdict is
        // carried, never re-derived. Re-deriving is what turned a per-item
        // PROVIDER failure into `Validation` (and so killed its retry) when
        // `run_text_pipeline` rendered its typed error into a string.
        ServerError::ClassifiedFailure { category, .. } => *category,
        ServerError::MemoryPressure(_) => FailureCategory::MemoryPressure,
        ServerError::RequiredEvidenceUnavailable(_) => FailureCategory::EvidenceUnavailable,
        ServerError::AlignmentCompletion(failure) => match failure {
            crate::error::AlignmentCompletionFailure::SourceChanged { .. }
            | crate::error::AlignmentCompletionFailure::OffRecordUtteranceTimed { .. } => {
                FailureCategory::System
            }
        },
        ServerError::AnalysisUnavailable(_) => FailureCategory::AnalysisUnavailable,
        // Deterministic across retries: the same recording yields the same
        // silence, so this must never be retried. `Validation` is chosen for
        // what it RENDERS as rather than as a claim that the request was
        // malformed. `user_facing_error` passes a validation message through
        // with light framing, so the typed text reaches the operator verbatim,
        // naming which stage came up empty and what to check.
        // `System` would replace that with "contact your administrator".
        ServerError::EmptyTranscription(_) => FailureCategory::Validation,
        ServerError::Io(_) => FailureCategory::System,
        // `MediaTiming` is internal by construction: input admission decides
        // every declaration its transition refuses (`AlignmentMedia`, above),
        // so only an alignment that changed the header can reach it.
        ServerError::AdmissionDisagreement
        | ServerError::ReplacementPlanContradicted { .. }
        | ServerError::Database(_)
        | ServerError::Migration(_)
        | ServerError::Persistence(_)
        | ServerError::StoredLease(_)
        | ServerError::MediaTiming(_)
        | ServerError::MorphosyntaxInjection(_)
        | ServerError::UtterancePartition(_)
        | ServerError::OutputParse(_) => FailureCategory::System,
        ServerError::JobNotFound(_)
        | ServerError::JobConflict { .. }
        | ServerError::JobNotTerminal(_)
        | ServerError::FileNotFound(_)
        | ServerError::FileNotReady(_)
        | ServerError::UnknownCommand(_) => FailureCategory::System,
        // EmptyFaAudioSegment is an internal skip signal consumed before reaching
        // file-level error classification.  Treat as validation if it ever leaks.
        ServerError::EmptyFaAudioSegment(..) => FailureCategory::Validation,
        // JobNotInLocalStore signals that an activity was dispatched to the
        // wrong server (shared-queue misconfiguration). It's a control-plane
        // topology bug, not a per-file validation or worker failure, the
        // closest fit is `System`, and the error message itself directs the
        // operator to check task-queue configuration.
        ServerError::JobNotInLocalStore(_) => FailureCategory::System,
        // A file ffprobe refuses, or one that probes as zero length, is the
        // submitter's to fix, and retrying cannot change it. `Validation` is
        // chosen for what it RENDERS as, the same reasoning as
        // `EmptyTranscription` above: the typed message names the file and
        // what was wrong with it, where `System` replaced it with "internal
        // error, try restarting the job". A host without ffprobe, or a broken
        // invariant, stays `System`, which is the operator's.
        ServerError::RecordingDuration(error) => match error.fault() {
            RecordingDurationFault::Media => FailureCategory::Validation,
            RecordingDurationFault::Host => FailureCategory::System,
        },
        // A gated/credential-denied Hugging Face Hub artifact is deterministic
        // across retries (the operator's credentials do not change mid-job)
        // and actionable, so it gets its own category rather than `System`
        // (which reads as "contact your administrator") or `Validation`
        // (which reads as "you sent bad input").
        ServerError::ModelAccessDenied(_) => FailureCategory::ModelAccessDenied,
        // The provider's own typed verdict decides this; nothing here re-reads
        // a message to guess. `ProviderTransient` is retryable
        // (`is_retryable_worker_failure`), which is the whole point: a dropped
        // upload should be attempted again rather than reported as bad input.
        ServerError::AsrProvider { disposition, .. } => match disposition {
            AsrProviderDisposition::Transient => FailureCategory::ProviderTransient,
            AsrProviderDisposition::Terminal => FailureCategory::ProviderTerminal,
        },
        // A deliberate stop, never a transient or infrastructure failure:
        // see `ServerError::Cancelled`'s own doc for why this must stay
        // distinct from `System`.
        ServerError::Cancelled => FailureCategory::Cancelled,
    }
}

/// Whether a classified worker failure should be retried automatically.
pub(crate) fn is_retryable_worker_failure(category: FailureCategory) -> bool {
    matches!(
        category,
        FailureCategory::WorkerCrash
            | FailureCategory::WorkerTimeout
            | FailureCategory::ProviderTransient
    )
}

/// Translate a classified failure into a user-facing error message.
///
/// The returned message is what end users see in the dashboard. It must be
/// actionable and free of system internals (no "Broken pipe", no "os error
/// 32", no stack traces). The raw technical error is preserved in server
/// logs via `tracing` for developer debugging.
///
/// `command_label` is the human-readable command name (e.g. "Alignment",
/// "Morphosyntax"). `filename` is the file that failed.
pub(crate) fn user_facing_error(
    category: FailureCategory,
    command_label: &str,
    filename: &str,
    raw_error: &str,
) -> String {
    match category {
        FailureCategory::WorkerCrash => {
            // Include the raw error (which now contains worker stderr via
            // ProcessExited's Display impl) so users see the actual Python
            // traceback or OOM message, not a generic "contact administrator."
            let detail = truncate_tail(raw_error, 500);
            format!(
                "{command_label} failed for {filename}: the processing engine crashed.\n{detail}"
            )
        }
        FailureCategory::WorkerTimeout => format!(
            "{command_label} timed out for {filename}: the processing engine did not \
             respond in time. The file may be too large or the server may be overloaded. \
             Try restarting the job or processing fewer files at once."
        ),
        FailureCategory::WorkerProtocol => format!(
            "{command_label} failed for {filename}: communication error with the \
             processing engine. Try restarting the job."
        ),
        FailureCategory::WorkerBootstrap => {
            // Bootstrap-class errors are user-actionable: network failure,
            // disk full, missing auth token, etc. Surface the worker's typed
            // message verbatim: that's exactly the actionable hint the user
            // needs. Do NOT prepend "an internal error" framing; the worker
            // already produced the user-facing wording.
            let detail = truncate_tail(raw_error, 1000);
            format!("{command_label} failed for {filename}: {detail}")
        }
        FailureCategory::ProviderTransient => format!(
            "{command_label} failed for {filename}: the external service returned a \
             temporary error. Try restarting the job."
        ),
        FailureCategory::ProviderTerminal => {
            // A provider may be local (Praat, for example). The typed category
            // establishes terminality, not a network or credential diagnosis.
            let detail = truncate_tail(raw_error, 1000);
            format!("{command_label} failed for {filename}: {detail}")
        }
        FailureCategory::MemoryPressure => format!(
            "{command_label} was deferred for {filename}: the server does not have \
             enough free memory. Try again later or process fewer files at once."
        ),
        FailureCategory::InputMissing => format!(
            "{command_label} failed for {filename}: a required input file could not be \
             found. Check that all referenced media files exist."
        ),
        FailureCategory::EvidenceUnavailable => {
            format!("{command_label} could not complete for {filename}: {raw_error}")
        }
        FailureCategory::AnalysisUnavailable => {
            format!("{command_label} could not complete for {filename}: {raw_error}")
        }
        // Validation, ParseError, and System categories typically already have
        // well-formed messages from the validation/parse layer, so we pass
        // them through with light framing.
        FailureCategory::Validation | FailureCategory::ParseError => {
            // These messages are already user-facing (validation error codes, etc.)
            format!("{command_label} failed for {filename}: {raw_error}")
        }
        FailureCategory::Cancelled => format!("{command_label} was cancelled for {filename}."),
        FailureCategory::System => format!(
            "{command_label} failed for {filename} due to an internal error. \
             Try restarting the job. If the problem persists, please contact \
             your administrator."
        ),
        FailureCategory::ModelAccessDenied => {
            // The worker's typed message already names the repository and
            // both remedies (accept its terms, authenticate with `hf auth
            // login`, or choose a different engine); surface it verbatim,
            // the same treatment `WorkerBootstrap` gets above.
            let detail = truncate_tail(raw_error, 1000);
            format!("{command_label} failed for {filename}: {detail}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_provider_message_keeps_local_failure_without_authentication_guess() {
        let message = user_facing_error(
            FailureCategory::ProviderTerminal,
            "Analysis",
            "silent.cs.wav",
            "Praat could not extract voiced intervals",
        );
        assert_eq!(
            message,
            "Analysis failed for silent.cs.wav: Praat could not extract voiced intervals"
        );
        assert!(!message.contains("API keys") && !message.contains("external service"));
        assert!(!is_retryable_worker_failure(
            FailureCategory::ProviderTerminal
        ));
    }

    #[test]
    fn long_unicode_provider_detail_truncates_at_a_character_boundary() {
        let raw = "é".repeat(600);
        let detail = truncate_tail(&raw, 999);
        assert_eq!(detail.len(), 998);
        assert_eq!(detail, "é".repeat(499));
    }

    #[test]
    fn morphosyntax_capability_refusal_preserves_actionable_source_language() {
        let language = crate::api::LanguageCode3::try_new("que").unwrap();
        let error = ServerError::from(
            crate::morphosyntax::AnalysisUnavailable::admit_primary(&language).unwrap_err(),
        );
        let category = classify_server_error(&error);
        assert_eq!(category, FailureCategory::AnalysisUnavailable);
        assert!(!is_retryable_worker_failure(category));
        let message = user_facing_error(category, "Morphology", "sample.cha", &error.to_string());
        assert!(message.contains("que") && message.contains("Keep truthful language declarations"));
        assert!(!message.contains("internal error") && !message.contains("Fix the @Languages"));
    }

    /// The regression this module exists for (2026-09-02): a Hugging Face
    /// Hub access failure (gated repo, missing token) must classify as its
    /// own category, never `Validation`. `Validation` tells the caller their
    /// REQUEST was wrong; the request was fine, the server's own machine
    /// lacks Hub access.
    #[test]
    fn model_access_denied_classifies_to_its_own_category_never_validation() {
        let error = ServerError::ModelAccessDenied(
            "could not download the Hugging Face model at \
             pyannote/speaker-diarization-community-1: its repository is gated"
                .to_owned(),
        );

        let category = classify_server_error(&error);

        assert_eq!(category, FailureCategory::ModelAccessDenied);
        assert_ne!(category, FailureCategory::Validation);
    }

    /// Audio ffprobe cannot read is the submitter's problem and says which
    /// file; a host without ffprobe is the operator's.
    ///
    /// The regression (2026-09-16): a folder of trimmed audio files that held
    /// no audio failed every file with "internal error, try restarting the
    /// job", because the probe's typed verdict had been rendered to a string
    /// and could only classify as `System`.
    #[test]
    fn unreadable_audio_names_the_file_and_a_missing_ffprobe_stays_internal() {
        use crate::error::RecordingDurationError;
        use crate::media::probe::ProbeError;

        let refused =
            ServerError::RecordingDuration(RecordingDurationError::Probe(ProbeError::Refused {
                input: "/audio/empty.mp3".to_owned(),
                diagnostics: "Invalid data found when processing input".to_owned(),
            }));
        let category = classify_server_error(&refused);
        assert_eq!(category, FailureCategory::Validation);
        assert!(!is_retryable_worker_failure(category));
        let message = user_facing_error(category, "Alignment", "empty.cha", &refused.to_string());
        assert!(
            message.contains("/audio/empty.mp3"),
            "the user must be told which file could not be read: {message}"
        );

        let zero_length =
            ServerError::RecordingDuration(RecordingDurationError::Probe(ProbeError::EmptyAudio {
                input: "/audio/silent.mp3".to_owned(),
            }));
        assert_eq!(
            classify_server_error(&zero_length),
            FailureCategory::Validation
        );

        let missing = ServerError::RecordingDuration(RecordingDurationError::Probe(
            ProbeError::FfprobeMissing {
                input: "/audio/fine.mp3".to_owned(),
            },
        ));
        assert_eq!(classify_server_error(&missing), FailureCategory::System);
    }

    /// A transcript's own `@Media` header is the user's to change: input,
    /// never "internal error, try restarting the job", and not retried. The
    /// media/timing transition after alignment is internal by construction,
    /// because admission decides every declaration it would refuse.
    ///
    /// The regression (2026-10-07): a transcript with no `@Media` ran all of
    /// UTR and FA, then failed with `MediaTiming(MissingMedia)`, classified
    /// `System`, and the user was told to restart the job.
    #[test]
    fn a_media_declaration_refusal_is_input_and_the_transition_is_internal() {
        use crate::error::{AlignmentMediaRefusal, SuggestedMediaName};
        let refused = ServerError::from(AlignmentMediaRefusal::Undeclared {
            suggested: SuggestedMediaName::Transcript("sample".to_owned()),
        });
        let category = classify_server_error(&refused);
        assert_eq!(category, FailureCategory::Validation);
        assert!(!is_retryable_worker_failure(category));
        let message = user_facing_error(category, "Alignment", "sample.cha", &refused.to_string());
        assert!(
            message.contains("@Media:\tsample, audio, unlinked"),
            "{message}"
        );
        assert!(
            !message.contains("internal error") && !message.contains("restarting"),
            "{message}"
        );

        let transition = ServerError::MediaTiming(
            batchalign_transform::media_timing::MediaTimingError::MissingMedia,
        );
        assert_eq!(classify_server_error(&transition), FailureCategory::System);
        assert!(
            transition.to_string().starts_with("internal fault"),
            "{transition}"
        );
    }

    /// A deterministic, credential-class failure must not be auto-retried:
    /// retrying does not change whether the operator has accepted a model's
    /// terms or holds a valid token.
    #[test]
    fn model_access_denied_is_not_retryable() {
        assert!(!is_retryable_worker_failure(
            FailureCategory::ModelAccessDenied
        ));
    }

    /// A recording that produced no words is never retried, and the operator
    /// sees WHICH stage came up empty.
    ///
    /// Retrying cannot change what a recording contains, and the message is
    /// the whole value of the failure: it distinguishes "the engine returned
    /// nothing" from "post-processing kept nothing" from "every token was
    /// punctuation". `ProviderTerminal` would have replaced all of it with
    /// boilerplate about API keys, which is why the category is chosen for its
    /// rendering and asserted here rather than left to a reader.
    #[test]
    fn an_empty_transcription_is_not_retryable_and_keeps_its_own_message() {
        let error = ServerError::EmptyTranscription(crate::error::EmptyTranscription::Asr);
        let category = classify_server_error(&error);

        assert!(
            !is_retryable_worker_failure(category),
            "a recording with no speech in it says the same thing on every attempt"
        );

        let message = user_facing_error(category, "Transcription", "clip.wav", &error.to_string());
        assert!(
            message.contains("recognized no words"),
            "the typed message must reach the operator, got: {message}"
        );
    }

    /// The user-facing message must surface the worker's own actionable text
    /// (which repository, and the two remedies) rather than a generic
    /// "contact your administrator" framing.
    #[test]
    fn model_access_denied_user_message_surfaces_the_worker_detail() {
        let raw = "could not download the Hugging Face model at \
                    pyannote/speaker-diarization-community-1: its repository is gated. \
                    Visit https://huggingface.co/pyannote/speaker-diarization-community-1 \
                    to accept its terms and authenticate with `hf auth login`, or choose a \
                    different --speaker-engine.";

        let message = user_facing_error(
            FailureCategory::ModelAccessDenied,
            "Diarize",
            "session.cha",
            raw,
        );

        assert!(message.contains("speaker-diarization-community-1"));
        assert!(message.contains("hf auth login"));
    }
}
