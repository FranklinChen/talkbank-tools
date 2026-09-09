//! Server error types: maps to HTTP status codes.
//!
//! Error responses use `{"detail": "..."}` to match FastAPI's `HTTPException`.

#[cfg(feature = "server")]
use axum::http::StatusCode;
#[cfg(feature = "server")]
use axum::response::{IntoResponse, Response};

use crate::api::JobId;
use crate::chat_ops::CacheKey;
use crate::media::window::EmptySegment;

/// Non-empty forced-alignment cache misses behind `--require-media-cache`.
///
/// Construction requires a head index, so the error cannot represent the
/// contradictory state "required evidence is unavailable, but nothing is
/// missing." The remaining indices retain the pipeline's group order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingForcedAlignmentEvidence {
    group_indices: Vec<usize>,
}

impl MissingForcedAlignmentEvidence {
    pub(crate) fn new(first_group: usize, remaining_groups: &[usize]) -> Self {
        let mut group_indices = Vec::with_capacity(remaining_groups.len() + 1);
        group_indices.push(first_group);
        group_indices.extend_from_slice(remaining_groups);
        Self { group_indices }
    }

    /// FA group ordinals whose evidence was absent from the reusable cache.
    pub fn group_indices(&self) -> &[usize] {
        &self.group_indices
    }
}

impl std::fmt::Display for MissingForcedAlignmentEvidence {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "required forced-alignment evidence is unavailable for groups {:?}; \
             --require-media-cache prevented new inference",
            self.group_indices
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

/// Typed family of intentional cache-only precondition refusals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MissingRequiredEvidence {
    /// One or more forced-alignment groups were absent.
    ForcedAlignment(MissingForcedAlignmentEvidence),
    /// Speaker evidence for one semantic media request was absent.
    Speaker(MissingSpeakerEvidence),
    /// Raw Rev.AI transcript evidence for one provider request was absent.
    RevAsr(MissingRevAsrEvidence),
}

impl std::fmt::Display for MissingRequiredEvidence {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ForcedAlignment(missing) => missing.fmt(formatter),
            Self::Speaker(missing) => missing.fmt(formatter),
            Self::RevAsr(missing) => missing.fmt(formatter),
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

    /// A timing-producing CHAT transform could not reconcile its output with
    /// the document's typed media declaration.
    ///
    /// **HTTP 500.** Alignment reached an internally contradictory output
    /// state and must not serialize it as a successful result.
    #[error("media/timing transition failed: {0}")]
    MediaTiming(#[from] batchalign_transform::media_timing::MediaTimingError),

    /// Serialized pipeline output failed parsing before provenance publication.
    ///
    /// **HTTP 500.** Preserve diagnostics from the generated output rather than
    /// blaming the submitted request or retrying a worker that already finished.
    #[error("pipeline output could not receive provenance: {0}")]
    OutputParse(#[source] talkbank_model::ParseErrors),

    /// A cache-only request reached a non-empty set of FA cache misses.
    ///
    /// This is an intentional, actionable precondition refusal, not corrupt
    /// persistence and not an internal system failure.
    #[error("{0}")]
    RequiredEvidenceUnavailable(MissingRequiredEvidence),

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
    RecordingDuration(String),

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

#[cfg(feature = "server")]
impl ServerError {
    fn status_code(&self) -> StatusCode {
        match self {
            Self::Database(_) | Self::Migration(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::Persistence(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::MediaTiming(_) | Self::OutputParse(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::RequiredEvidenceUnavailable(_) => StatusCode::PRECONDITION_FAILED,
            Self::JobNotFound(_) => StatusCode::NOT_FOUND,
            Self::JobConflict { .. } => StatusCode::CONFLICT,
            Self::JobNotTerminal(_) => StatusCode::CONFLICT,
            Self::FileNotFound(_) => StatusCode::NOT_FOUND,
            Self::FileNotReady(_) => StatusCode::CONFLICT,
            Self::UnknownCommand(_) => StatusCode::BAD_REQUEST,
            Self::Validation(_) => StatusCode::BAD_REQUEST,
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
