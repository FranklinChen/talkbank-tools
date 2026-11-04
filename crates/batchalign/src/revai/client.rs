//! Blocking Rev.AI HTTP client.
//!
//! The shared client stays blocking on purpose. The PyO3 binding can release
//! the Python GIL around an entire request, and the Rust server can move upload
//! work onto `spawn_blocking` threads. That keeps the client simple while still
//! fitting both host runtimes cleanly.

use batchalign_types::interval::AdmittedInterval;
use std::thread;
use std::time::Duration;

use reqwest::blocking::multipart::{Form, Part};
use reqwest::blocking::{Client, RequestBuilder};
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE};

use crate::api::{AudioPositionSeconds, RevAiJobId};

use super::poll::{Awaited, PollEnd, PollPolicy, PollTiming, PolledJob, Progress, SubmittedJob};
use super::types::{
    Job, JobStatus, LangIdJob, LangIdJobStatus, LangIdResult, RevTranscriptEvidence, SubmitOptions,
    TimedWord, Transcript,
};

/// Transcript plus optional detected language from Rev.AI auto-detection.
///
/// When `language: "auto"` is used in `SubmitOptions`, Rev.AI returns the
/// detected language as an ISO 639-1 code on the completed `Job`. This struct
/// bundles the transcript with that detection result so callers can propagate
/// the real language to downstream pipeline stages.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TranscriptResult {
    /// The full transcript payload.
    transcript: RevTranscriptEvidence,
    /// ISO 639-1 language code detected by Rev.AI (e.g. `"es"`, `"en"`).
    /// `None` when a concrete language was specified (not auto-detected).
    detected_language: Option<String>,
}

impl TranscriptResult {
    pub(super) fn into_parts(self) -> (RevTranscriptEvidence, Option<String>) {
        (self.transcript, self.detected_language)
    }
}

const PRODUCTION_BASE_URL: &str = "https://api.rev.ai/speechtotext/v1";
const PRODUCTION_LANGID_BASE_URL: &str = "https://api.rev.ai/languageid/v1";
const TRANSCRIPT_ACCEPT: &str = "application/vnd.rev.transcript.v1.0+json";
const LANGID_ACCEPT: &str = "application/vnd.rev.languageid.v1.0+json";

/// The two Rev.AI service roots a client talks to.
///
/// These were bare `const`s until 2026-09-03. A hardcoded host is not a
/// configuration inconvenience, it is the reason a transport fault against
/// this provider could not be reproduced anywhere but against the paid
/// endpoint: the fleet met `request or response body error` on 94 uploads and
/// there was no seam at which to ask what our own client does when a
/// connection goes away. Carrying the roots as a value gives the test suite a
/// real socket to fail against and costs production nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RevAiEndpoints {
    speech_to_text: String,
    language_id: String,
}

impl RevAiEndpoints {
    /// The live Rev.AI service. The only value production ever constructs.
    fn production() -> Self {
        Self {
            speech_to_text: PRODUCTION_BASE_URL.to_owned(),
            language_id: PRODUCTION_LANGID_BASE_URL.to_owned(),
        }
    }
}

/// How long one Rev.AI request may take, decided by what it carries.
///
/// reqwest's blocking `Client::new()` gives EVERY request a 30 s total
/// timeout. A media upload that shares the uplink with others takes longer
/// than that, and the timeout then drops the request while its body is still
/// being written, which reqwest reports as "request or response body error
/// ... send failed because receiver is gone". Every upload of a 48-recording
/// fold-in failed that way at exactly 30 s. No single total fits both a
/// status poll and an 80 MB upload, so the deadline is a property of the
/// request kind, and an upload's grows with its body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestDeadline {
    /// A small exchange: a status poll, a transcript or result download.
    Exchange,
    /// A media upload: a fixed allowance plus the body at a floor rate.
    Upload { body_bytes: usize },
}

impl RequestDeadline {
    /// Bound on a small exchange; a long recording's transcript JSON is a
    /// few megabytes at most.
    const EXCHANGE: Duration = Duration::from_secs(120);
    /// Fixed part of an upload's allowance (connection, TLS, Rev's reply).
    const UPLOAD_BASE: Duration = Duration::from_secs(120);
    /// The slowest sustained upload rate still treated as progress. Uploads
    /// run concurrently (one per busy worker, 48 in a six-job fold-in) and
    /// share one uplink, so the floor is set well below a per-stream share:
    /// at 16 KiB/s an 80 MB recording is allowed about 83 minutes. The bound
    /// exists so a stalled connection cannot hang a worker forever, not to
    /// race a healthy one.
    const UPLOAD_FLOOR_BYTES_PER_SECOND: u64 = 16 * 1024;

    fn duration(self) -> Duration {
        match self {
            Self::Exchange => Self::EXCHANGE,
            Self::Upload { body_bytes } => {
                let bytes = u64::try_from(body_bytes).unwrap_or(u64::MAX);
                Self::UPLOAD_BASE
                    + Duration::from_secs(bytes.div_ceil(Self::UPLOAD_FLOOR_BYTES_PER_SECOND))
            }
        }
    }
}

/// The HTTP client, reachable only through requests that carry a deadline.
///
/// The inner client has a connect timeout and NO total timeout; each request
/// gets its own from [`RequestDeadline`], and these two methods are the only
/// way to build one, so a request without a deadline has no route. An upload
/// builds its own form from the media bytes, so the deadline is computed from
/// the body actually sent.
struct DeadlineClient(Client);

impl DeadlineClient {
    const CONNECT: Duration = Duration::from_secs(30);

    fn new() -> Result<Self> {
        Client::builder()
            .connect_timeout(Self::CONNECT)
            .timeout(None)
            .build()
            .map(Self)
            .map_err(RevAiError::Http)
    }

    fn get(&self, url: String) -> RequestBuilder {
        self.0
            .get(url)
            .timeout(RequestDeadline::Exchange.duration())
    }

    /// A multipart POST whose `media` part is `media`, plus an optional
    /// second named part (the transcription options).
    fn upload(
        &self,
        url: String,
        media: &[u8],
        file_name: &str,
        mime: &str,
        extra: Option<(&'static str, Part)>,
    ) -> Result<RequestBuilder> {
        let mut form = Form::new().part(
            "media",
            Part::bytes(media.to_vec())
                .file_name(file_name.to_owned())
                .mime_str(mime)?,
        );
        if let Some((name, part)) = extra {
            form = form.part(name, part);
        }
        let deadline = RequestDeadline::Upload {
            body_bytes: media.len(),
        };
        Ok(self
            .0
            .post(url)
            .timeout(deadline.duration())
            .multipart(form))
    }
}

/// How long to wait before each upload retry.
///
/// A schedule is data, not a `sleep` buried in a loop: production backs off
/// exponentially, and a test that wants to observe the exhausted-retry
/// outcome should not have to spend six seconds of wall clock doing it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UploadRetryBackoff {
    /// `2^attempt` seconds before attempts 2 and 3.
    Exponential,
    /// No wait at all. Test-only: see [`RevAiClient::for_test`].
    #[cfg(test)]
    None,
}

impl UploadRetryBackoff {
    fn delay_before(self, attempt: u32) -> Option<Duration> {
        if attempt == 0 {
            return None;
        }
        match self {
            Self::Exponential => Some(Duration::from_secs(2u64.pow(attempt))),
            #[cfg(test)]
            Self::None => None,
        }
    }
}

/// Whether a Rev.AI failure is worth another attempt.
///
/// The verdict is derived from the typed error, never from its message, and
/// it is what the control plane needs in order to classify the failure.
/// Before 2026-09-03 every Rev.AI failure was stringified into
/// `ServerError::Validation`, so a dropped TCP connection reached the operator
/// as `error_category: validation`, which reads as "your input was bad".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RevAiDisposition {
    /// The same request may succeed later: transport faults, 5xx, exhausted
    /// upload retries.
    Transient,
    /// Repeating the request cannot help: a 4xx, a job Rev.AI itself failed,
    /// or a response we cannot decode.
    Terminal,
    /// Rev.AI refuses this account (HTTP 401, 402 or 403: the key is not
    /// accepted, the account is not permitted, or it has no credit left). No
    /// request from it succeeds until an account holder acts, so neither a
    /// retry nor a narrower UTR fallback can help.
    AccountRefused,
}

impl std::fmt::Display for RevAiDisposition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transient => write!(f, "transient"),
            Self::Terminal => write!(f, "terminal"),
            Self::AccountRefused => write!(f, "account_refused"),
        }
    }
}

/// Errors produced by Rev.AI client operations.
#[derive(Debug, thiserror::Error)]
pub enum RevAiError {
    /// The HTTP client failed before receiving a usable response.
    ///
    /// The Display of a `reqwest::Error` names only the STAGE that failed
    /// ("request or response body error for url (...)"); the cause that says
    /// WHAT happened to the socket lives in its source chain. Rendering only
    /// the top line is how 94 upload failures on 2026-09-03 were read as a
    /// credential problem: nothing printed said "connection reset". The chain
    /// is walked here so the operator sees the actual fault.
    #[error("HTTP error: {}", render_source_chain(.0))]
    Http(#[from] reqwest::Error),

    /// Rev.AI returned a non-success HTTP status and response body.
    #[error("Rev.AI {endpoint} returned HTTP {status}: {body}")]
    ApiError {
        /// The request the status answers, which decides what it means.
        endpoint: RevAiEndpoint,
        /// Numeric HTTP status code from Rev.AI.
        status: u16,
        /// Response body returned by Rev.AI.
        body: String,
    },

    /// A submitted job reached the failed terminal state.
    #[error("Rev.AI job failed: {0}")]
    JobFailed(String),

    /// Every upload attempt failed.
    ///
    /// Carries what happened on EACH attempt. Keeping only the last one hides
    /// the shape of the failure: three identical transport faults and a 502
    /// followed by two resets are different diagnoses, and the operator can
    /// only tell them apart if all of them survive.
    #[error(
        "Rev.AI upload failed on all {} attempt(s): {}",
        attempts.len(),
        attempts.iter().enumerate()
            .map(|(index, error)| format!("attempt {}: {error}", index + 1))
            .collect::<Vec<_>>()
            .join("; ")
    )]
    RetriesExhausted {
        /// One entry per attempt, in the order they were made.
        attempts: Vec<RevAiError>,
    },

    /// JSON decoding failed for a Rev.AI response body.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// The transcript endpoint returned bytes that cannot be retained as
    /// lossless UTF-8 JSON text.
    #[error("Rev.AI transcript body is not UTF-8: {0}")]
    TranscriptEncoding(#[from] std::string::FromUtf8Error),
}

/// Which Rev.AI request a failed status answers.
///
/// A status means different things on different requests: a 401 or 403 on a
/// submission refuses the account (its key, its permission, its credit), but
/// on a job's status or transcript a 403 may refuse only that resource. So
/// [`RevAiError::disposition`] reads the account refusal only from a
/// submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RevAiEndpoint {
    /// `POST /jobs`: submitting media for transcription.
    SubmitJob,
    /// `GET /jobs/{id}`: a transcription job's status.
    JobDetails,
    /// `GET /jobs/{id}/transcript`.
    Transcript,
    /// `POST /languageid/.../jobs`: submitting media for language id.
    SubmitLanguageId,
    /// `GET .../jobs/{id}` of a language-id job.
    LanguageIdJob,
    /// `GET .../jobs/{id}/result` of a language-id job.
    LanguageIdResult,
}

impl RevAiEndpoint {
    /// Whether the request submits work on the account's behalf, so a
    /// refusal of it is a refusal of the account.
    const fn submits(self) -> bool {
        match self {
            Self::SubmitJob | Self::SubmitLanguageId => true,
            Self::JobDetails | Self::Transcript | Self::LanguageIdJob | Self::LanguageIdResult => {
                false
            }
        }
    }
}

impl std::fmt::Display for RevAiEndpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::SubmitJob => "job submission",
            Self::JobDetails => "job status",
            Self::Transcript => "transcript",
            Self::SubmitLanguageId => "language-id submission",
            Self::LanguageIdJob => "language-id job status",
            Self::LanguageIdResult => "language-id result",
        })
    }
}

/// HTTP 429: Rev.AI is rate limiting this account. Transient on every
/// request: the limit lifts, and the request is not at fault.
const TOO_MANY_REQUESTS: u16 = 429;

/// Render an error together with every cause beneath it.
///
/// `reqwest::Error`'s own Display stops at the stage that failed, so the
/// interesting half (`connection reset by peer`, `broken pipe`) is only
/// reachable through `source()`. Used by [`RevAiError::Http`]'s message.
fn render_source_chain(error: &(dyn std::error::Error + 'static)) -> String {
    let mut rendered = error.to_string();
    let mut cause = error.source();
    while let Some(next) = cause {
        rendered.push_str(": ");
        rendered.push_str(&next.to_string());
        cause = next.source();
    }
    rendered
}

impl RevAiError {
    /// Whether repeating the request could plausibly succeed.
    ///
    /// Exhaustive on purpose: a new variant must state its own verdict rather
    /// than inherit a catch-all, because the verdict decides both whether the
    /// control plane retries and what the operator is told went wrong.
    pub fn disposition(&self) -> RevAiDisposition {
        match self {
            // The socket, not the request: a reset, a hang-up mid-body, a DNS
            // blip. Nothing about the submission changes by trying again.
            Self::Http(_) => RevAiDisposition::Transient,
            // The service, or its rate limit, on any request: both lift.
            Self::ApiError { status, .. } if *status >= 500 || *status == TOO_MANY_REQUESTS => {
                RevAiDisposition::Transient
            }
            // The account, not the request: an unaccepted key, a forbidden
            // account, no credit. Read only from a submission; on a job's
            // status or transcript a 403 may refuse that resource alone.
            Self::ApiError {
                endpoint,
                status: 401..=403,
                ..
            } if endpoint.submits() => RevAiDisposition::AccountRefused,
            Self::ApiError { .. } => RevAiDisposition::Terminal,
            // Every attempt already failed; the loop only keeps retrying what
            // it judged retryable, so the exhaustion itself is transient.
            Self::RetriesExhausted { .. } => RevAiDisposition::Transient,
            // Rev.AI reached a terminal verdict on the job, or handed back
            // something we cannot decode. Neither improves on a second run.
            Self::JobFailed(_) | Self::Json(_) | Self::TranscriptEncoding(_) => {
                RevAiDisposition::Terminal
            }
        }
    }
}

/// A failure sorted by its verdict, so a match on it is a match on what to do.
///
/// The one route from a [`RevAiError`] to a decision: the upload loop and the
/// poll both consume this rather than re-reading [`RevAiError::disposition`]
/// and a status code of their own. Each arm's payload can only be built here,
/// so a [`TerminalFailure`] in hand is proof the table called it terminal.
#[derive(Debug)]
pub(crate) enum ClassifiedFailure {
    Transient(TransientFailure),
    Terminal(TerminalFailure),
    AccountRefused(AccountRefusal),
}

/// A failure the table judged transient: the same request may succeed later.
#[derive(Debug)]
pub(crate) struct TransientFailure(RevAiError);

/// A failure the table judged terminal: repeating the request cannot help.
#[derive(Debug)]
pub(crate) struct TerminalFailure(RevAiError);

/// Rev.AI refusing the account itself.
#[derive(Debug)]
pub(crate) struct AccountRefusal(RevAiError);

impl TransientFailure {
    /// The failure itself, once the verdict has been acted on.
    pub(crate) fn into_error(self) -> RevAiError {
        self.0
    }
}

impl TerminalFailure {
    /// The failure itself, once the verdict has been acted on.
    pub(crate) fn into_error(self) -> RevAiError {
        self.0
    }
}

impl AccountRefusal {
    /// The failure itself, once the verdict has been acted on.
    pub(crate) fn into_error(self) -> RevAiError {
        self.0
    }
}

impl std::fmt::Display for TransientFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl RevAiError {
    /// Sort this failure by its verdict, consuming it.
    pub(crate) fn classify(self) -> ClassifiedFailure {
        match self.disposition() {
            RevAiDisposition::Transient => ClassifiedFailure::Transient(TransientFailure(self)),
            RevAiDisposition::Terminal => ClassifiedFailure::Terminal(TerminalFailure(self)),
            RevAiDisposition::AccountRefused => {
                ClassifiedFailure::AccountRefused(AccountRefusal(self))
            }
        }
    }
}

impl From<RevAiError> for crate::error::ServerError {
    /// The ONLY route from a Rev.AI failure into the control plane.
    ///
    /// It exists so no call site can reach for `Validation(error.to_string())`
    /// again: that spelling is what turned a dropped connection into "bad
    /// input" for 94 files. The typed disposition survives; the message keeps
    /// the provider's full cause chain.
    fn from(error: RevAiError) -> Self {
        let disposition = match error.disposition() {
            RevAiDisposition::Transient => crate::error::AsrProviderDisposition::Transient,
            RevAiDisposition::Terminal => crate::error::AsrProviderDisposition::Terminal,
            RevAiDisposition::AccountRefused => {
                crate::error::AsrProviderDisposition::AccountRefused
            }
        };
        Self::AsrProvider {
            disposition,
            message: render_source_chain(&error),
        }
    }
}

/// Standard result type for Rev.AI client operations.
pub type Result<T> = std::result::Result<T, RevAiError>;

/// Blocking Rev.AI HTTP client.
pub struct RevAiClient {
    api_key: String,
    client: DeadlineClient,
    endpoints: RevAiEndpoints,
    upload_retry_backoff: UploadRetryBackoff,
    /// Where a job wait's time comes from: real for production, virtual for a
    /// test that needs a deadline to pass without the wall clock doing so.
    poll_timing: PollTiming,
}

impl RevAiClient {
    /// Create a new client bound to one API key, talking to the live service.
    ///
    /// Fails only if the HTTP client cannot be built (its TLS backend cannot
    /// initialize), which `Client::new()` used to turn into a panic.
    pub fn new(api_key: &str) -> Result<Self> {
        Ok(Self {
            api_key: api_key.to_owned(),
            client: DeadlineClient::new()?,
            endpoints: RevAiEndpoints::production(),
            upload_retry_backoff: UploadRetryBackoff::Exponential,
            poll_timing: PollTiming::Real,
        })
    }

    /// A client pointed at a local test server, with the backoff removed and
    /// the poll clock virtual.
    ///
    /// The only other constructor is [`new`][Self::new], which is always the
    /// production endpoint, so no production path can reach a different host:
    /// this one is compiled out of the shipped binary entirely.
    #[cfg(test)]
    pub(super) fn for_test(speech_to_text_base: &str) -> Self {
        Self {
            api_key: "test-key".to_owned(),
            client: DeadlineClient::new().expect("test: HTTP client builds"),
            endpoints: RevAiEndpoints {
                speech_to_text: speech_to_text_base.to_owned(),
                language_id: format!("{speech_to_text_base}/languageid"),
            },
            upload_retry_backoff: UploadRetryBackoff::None,
            poll_timing: PollTiming::Virtual,
        }
    }

    fn submit_media_bytes(
        &self,
        file_bytes: &[u8],
        file_name: &str,
        mime: &str,
        opts: &SubmitOptions,
    ) -> Result<Job> {
        let options_json = serde_json::to_string(opts)?;

        // Every attempt's failure is kept. The loop used to overwrite one
        // `last_err`, so an operator reading a failed upload saw the third
        // fault and had no way to tell it apart from the first two.
        let mut attempts_failed: Vec<RevAiError> = Vec::new();

        for attempt in 0..3u32 {
            if let Some(delay) = self.upload_retry_backoff.delay_before(attempt) {
                eprintln!(
                    "talkbank-revai: retry {}/3 for upload of {} (waiting {}s)",
                    attempt + 1,
                    file_name,
                    delay.as_secs(),
                );
                thread::sleep(delay);
            }

            let options_part = Part::text(options_json.clone()).mime_str("application/json")?;
            match self
                .client
                .upload(
                    format!("{}/jobs", self.endpoints.speech_to_text),
                    file_bytes,
                    file_name,
                    mime,
                    Some(("options", options_part)),
                )?
                .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
                .send()
            {
                Ok(resp) => {
                    if !resp.status().is_success() {
                        let status = resp.status().as_u16();
                        let body = read_error_body(resp);
                        let error = RevAiError::ApiError {
                            endpoint: RevAiEndpoint::SubmitJob,
                            status,
                            body,
                        };
                        // Retry only what the error itself says is worth
                        // retrying, rather than re-deciding the 5xx boundary
                        // here in a second place.
                        match error.classify() {
                            ClassifiedFailure::Transient(failure) => {
                                attempts_failed.push(failure.into_error());
                            }
                            ClassifiedFailure::Terminal(failure) => {
                                return Err(failure.into_error());
                            }
                            ClassifiedFailure::AccountRefused(refusal) => {
                                return Err(refusal.into_error());
                            }
                        }
                        continue;
                    }
                    return Ok(resp.json()?);
                }
                Err(err) => {
                    attempts_failed.push(RevAiError::Http(err));
                }
            }
        }

        Err(RevAiError::RetriesExhausted {
            attempts: attempts_failed,
        })
    }

    /// Fetch the current status for one previously submitted job.
    fn get_job_details(&self, job_id: &RevAiJobId) -> Result<Job> {
        let resp = self
            .client
            .get(format!("{}/jobs/{job_id}", self.endpoints.speech_to_text))
            .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
            .send()?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = read_error_body(resp);
            return Err(RevAiError::ApiError {
                endpoint: RevAiEndpoint::JobDetails,
                status,
                body,
            });
        }

        Ok(resp.json()?)
    }

    /// Download the transcript for a completed job.
    fn get_transcript(&self, job_id: &RevAiJobId) -> Result<RevTranscriptEvidence> {
        let resp = self
            .client
            .get(format!(
                "{}/jobs/{job_id}/transcript",
                self.endpoints.speech_to_text
            ))
            .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
            .header(ACCEPT, TRANSCRIPT_ACCEPT)
            .header(CONTENT_TYPE, "application/json")
            .send()?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = read_error_body(resp);
            return Err(RevAiError::ApiError {
                endpoint: RevAiEndpoint::Transcript,
                status,
                body,
            });
        }

        retain_transcript_body(resp.bytes()?.to_vec())
    }

    /// Upload media, wait for its job under `policy`, and download the
    /// transcript.
    ///
    /// `Ok(Awaited::Unfinished)` is a paid job the wait did not see finish; it
    /// names the job so the caller can record it and a later run can resume
    /// it with [`Self::resume_transcription_blocking`] instead of paying again.
    pub(super) fn transcribe_bytes_blocking(
        &self,
        bytes: &[u8],
        file_name: &str,
        mime: &str,
        opts: &SubmitOptions,
        policy: PollPolicy,
    ) -> Result<Awaited<TranscriptResult>> {
        let job = self.submit_media_bytes(bytes, file_name, mime, opts)?;
        self.await_transcription(SubmittedJob::named(job.id), policy)
    }

    /// Wait for a job an earlier run submitted and left unfinished, then
    /// download its transcript. Uploads nothing.
    pub(super) fn resume_transcription_blocking(
        &self,
        job: SubmittedJob,
        policy: PollPolicy,
    ) -> Result<Awaited<TranscriptResult>> {
        self.await_transcription(job, policy)
    }

    /// Poll a submitted transcription job until the wait ends, then download
    /// a finished job's transcript.
    ///
    /// The wait is the [`super::poll`] typestate: it cannot run without a
    /// deadline, and a transient status failure cannot end it. Rev.AI's own
    /// verdict on the job (failed, or a terminal refusal of the poll) is the
    /// error it was; an unfinished job is returned by name, not as an error.
    fn await_transcription(
        &self,
        job: SubmittedJob,
        policy: PollPolicy,
    ) -> Result<Awaited<TranscriptResult>> {
        match job
            .begin::<TranscriptionJob>(policy, self.poll_timing)
            .run(self)
        {
            PollEnd::Ready(job_id, detected_language) => {
                let transcript = self.get_transcript(&job_id)?;
                Ok(Awaited::Ready(TranscriptResult {
                    transcript,
                    detected_language,
                }))
            }
            PollEnd::Unfinished(unfinished) => Ok(Awaited::Unfinished(unfinished)),
            PollEnd::Rejected(failure) => Err(failure.into_error()),
            PollEnd::Refused(refusal) => Err(refusal.into_error()),
        }
    }

    // -------------------------------------------------------------------
    // Language Identification API
    // -------------------------------------------------------------------

    fn submit_langid_bytes(
        &self,
        file_bytes: &[u8],
        file_name: &str,
        mime: &str,
    ) -> Result<LangIdJob> {
        let resp = self
            .client
            .upload(
                format!("{}/jobs", self.endpoints.language_id),
                file_bytes,
                file_name,
                mime,
                None,
            )?
            .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
            .send()?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = read_error_body(resp);
            return Err(RevAiError::ApiError {
                endpoint: RevAiEndpoint::SubmitLanguageId,
                status,
                body,
            });
        }

        Ok(resp.json()?)
    }

    /// Poll a language identification job status.
    fn get_langid_job(&self, job_id: &RevAiJobId) -> Result<LangIdJob> {
        let resp = self
            .client
            .get(format!("{}/jobs/{job_id}", self.endpoints.language_id))
            .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
            .send()?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = read_error_body(resp);
            return Err(RevAiError::ApiError {
                endpoint: RevAiEndpoint::LanguageIdJob,
                status,
                body,
            });
        }

        Ok(resp.json()?)
    }

    /// Download the language identification result for a completed job.
    fn get_langid_result(&self, job_id: &RevAiJobId) -> Result<LangIdResult> {
        let resp = self
            .client
            .get(format!(
                "{}/jobs/{job_id}/result",
                self.endpoints.language_id
            ))
            .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
            .header(ACCEPT, LANGID_ACCEPT)
            .send()?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = read_error_body(resp);
            return Err(RevAiError::ApiError {
                endpoint: RevAiEndpoint::LanguageIdResult,
                status,
                body,
            });
        }

        Ok(resp.json()?)
    }

    /// Upload media for language identification, wait for its job under
    /// `policy`, and download the result. The same bounded wait as a
    /// transcription; an unfinished identification is returned by name.
    pub(super) fn identify_language_bytes_blocking(
        &self,
        bytes: &[u8],
        file_name: &str,
        mime: &str,
        policy: PollPolicy,
    ) -> Result<Awaited<LangIdResult>> {
        let job = self.submit_langid_bytes(bytes, file_name, mime)?;
        match SubmittedJob::named(job.id)
            .begin::<LanguageIdJob>(policy, self.poll_timing)
            .run(self)
        {
            PollEnd::Ready(job_id, ()) => Ok(Awaited::Ready(self.get_langid_result(&job_id)?)),
            PollEnd::Unfinished(unfinished) => Ok(Awaited::Unfinished(unfinished)),
            PollEnd::Rejected(failure) => Err(failure.into_error()),
            PollEnd::Refused(refusal) => Err(refusal.into_error()),
        }
    }
}

/// A transcription job, as the poll typestate reads it. Ready carries the
/// language Rev.AI detected (`None` when a concrete language was submitted).
struct TranscriptionJob;

impl PolledJob for TranscriptionJob {
    type Ready = Option<String>;

    fn progress(client: &RevAiClient, job: &RevAiJobId) -> Result<Progress<Self::Ready>> {
        let details = client.get_job_details(job)?;
        match details.status {
            JobStatus::InProgress => Ok(Progress::InProgress),
            JobStatus::Transcribed => Ok(Progress::Ready(details.language)),
            // Rev.AI's own verdict on the job, handed to the one failure table
            // (which calls it terminal) rather than ended here.
            JobStatus::Failed => Err(RevAiError::JobFailed(
                details
                    .failure_detail
                    .unwrap_or_else(|| "unknown error".into()),
            )),
        }
    }
}

/// A language-identification job, as the poll typestate reads it. Its result
/// is a separate download, so Ready carries nothing.
struct LanguageIdJob;

impl PolledJob for LanguageIdJob {
    type Ready = ();

    fn progress(client: &RevAiClient, job: &RevAiJobId) -> Result<Progress<Self::Ready>> {
        let status = client.get_langid_job(job)?;
        match status.status {
            LangIdJobStatus::InProgress => Ok(Progress::InProgress),
            LangIdJobStatus::Completed => Ok(Progress::Ready(())),
            LangIdJobStatus::Failed => Err(RevAiError::JobFailed(
                status
                    .failure_detail
                    .unwrap_or_else(|| "unknown error".into()),
            )),
        }
    }
}

/// Admit one transcript response body without text-decoder normalization.
///
/// JSON is required to be UTF-8. A strict conversion makes the retained
/// `String` a lossless representation of the exact HTTP body bytes; malformed
/// encodings fail before they can acquire exact-provider fidelity.
fn retain_transcript_body(body: Vec<u8>) -> Result<RevTranscriptEvidence> {
    let raw_json = String::from_utf8(body)?;
    RevTranscriptEvidence::from_provider_json(raw_json).map_err(Into::into)
}

fn read_error_body(resp: reqwest::blocking::Response) -> String {
    match resp.text() {
        Ok(body) => body,
        Err(error) => format!("<failed to read response body: {error}>"),
    }
}

/// Project a full Rev.AI transcript into the simplified timed-word shape used
/// by the UTR path.
pub fn extract_timed_words(transcript: &Transcript) -> Vec<TimedWord> {
    struct TimedElement<'a> {
        value: &'a str,
        start_s: AudioPositionSeconds,
        end_s: AudioPositionSeconds,
    }

    let mut raw: Vec<TimedElement<'_>> = Vec::new();
    for monologue in &transcript.monologues {
        for elem in &monologue.elements {
            if let (Some(ts), Some(end_ts)) = (elem.ts, elem.end_ts) {
                raw.push(TimedElement {
                    value: &elem.value,
                    start_s: ts,
                    end_s: end_ts,
                });
            }
        }
    }

    // Positions are totally ordered, so no incomparable pair needs a fallback.
    raw.sort_by(|a, b| a.start_s.order(b.start_s));

    let mut result = Vec::with_capacity(raw.len());
    let mut prev_end_ms: f64 = 0.0;

    for elem in &raw {
        let cleaned = elem.value.trim();
        if cleaned.is_empty() {
            continue;
        }
        // The pair is admitted once as an interval (rounded to the nearest
        // millisecond, ordered, in range); a pair that is not one is no
        // timing, and the word is left out of the timed words like a word
        // the transcript gave no times.
        let interval = match AdmittedInterval::admit_positions(elem.start_s, elem.end_s) {
            Ok(interval) => interval,
            Err(refusal) => {
                tracing::warn!(word = cleaned, %refusal, "Rev.ai word's times are not an interval; left untimed");
                continue;
            }
        };
        let start_ms = interval.start_millis();
        let end_ms = interval.end_millis();

        if (start_ms as f64) < prev_end_ms * 0.5 && prev_end_ms > 2000.0 {
            tracing::warn!(
                word = cleaned,
                start_ms,
                prev_end_ms = prev_end_ms as u64,
                "Rev.ai timestamp regression"
            );
        }

        result.push(TimedWord {
            word: cleaned.to_owned(),
            start_ms,
            end_ms,
        });
        prev_end_ms = end_ms as f64;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcript_body_retention_preserves_exact_utf8_bytes() {
        let body = "{\n  \"monologues\": [], \"provider_extension\": \"café\"\n}\n"
            .as_bytes()
            .to_vec();

        let evidence = retain_transcript_body(body.clone()).expect("valid provider JSON");

        assert_eq!(
            evidence
                .exact_provider_json()
                .expect("live response retains exact JSON")
                .as_bytes(),
            body
        );
    }

    #[test]
    fn transcript_body_retention_rejects_non_utf8_without_lossy_decoding() {
        let error = retain_transcript_body(vec![b'{', 0xff, b'}'])
            .expect_err("non-UTF-8 provider bytes must not acquire exact fidelity");

        assert!(matches!(error, RevAiError::TranscriptEncoding(_)));
    }

    #[test]
    fn parse_job_in_progress() {
        let json = r#"{"id":"abc123","status":"in_progress"}"#;
        let job: Job = serde_json::from_str(json).unwrap();
        assert_eq!(job.id, "abc123");
        assert_eq!(job.status, JobStatus::InProgress);
        assert!(job.failure_detail.is_none());
    }

    #[test]
    fn parse_job_failed() {
        let json = r#"{"id":"ghi789","status":"failed","failure_detail":"Audio too short"}"#;
        let job: Job = serde_json::from_str(json).unwrap();
        assert_eq!(job.status, JobStatus::Failed);
        assert_eq!(job.failure_detail.as_deref(), Some("Audio too short"));
    }

    #[test]
    fn serialize_submit_options_full() {
        let opts = SubmitOptions {
            language: "en".into(),
            speakers_count: Some(2),
            skip_postprocessing: Some(true),
            metadata: Some("test_job".into()),
        };
        let json = serde_json::to_string(&opts).unwrap();
        assert!(json.contains(r#""speakers_count":2"#));
        assert!(json.contains(r#""skip_postprocessing":true"#));
        assert!(json.contains(r#""metadata":"test_job""#));
    }

    #[test]
    fn extract_timed_words_basic() {
        let transcript: Transcript = serde_json::from_str(
            r#"{
            "monologues": [{
                "speaker": 0,
                "elements": [
                    {"type": "text", "value": "hello", "ts": 0.5, "end_ts": 0.9},
                    {"type": "text", "value": "world", "ts": 1.0, "end_ts": 1.5},
                    {"type": "punct", "value": "."}
                ]
            }]
        }"#,
        )
        .unwrap();

        let words = extract_timed_words(&transcript);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "hello");
        assert_eq!(words[0].start_ms, 500);
        assert_eq!(words[1].word, "world");
        assert_eq!(words[1].end_ms, 1500);
    }

    // -------------------------------------------------------------------
    // Language Identification type tests
    // -------------------------------------------------------------------

    #[test]
    fn parse_langid_job_in_progress() {
        let json = r#"{"id":"Umx5c6F7pH7r","status":"in_progress","type":"language_id","created_on":"2021-09-15T05:14:38.13"}"#;
        let job: crate::revai::types::LangIdJob = serde_json::from_str(json).unwrap();
        assert_eq!(job.id, "Umx5c6F7pH7r");
        assert_eq!(job.status, crate::revai::types::LangIdJobStatus::InProgress);
    }

    #[test]
    fn parse_langid_job_completed() {
        let json = r#"{"id":"abc","status":"completed"}"#;
        let job: crate::revai::types::LangIdJob = serde_json::from_str(json).unwrap();
        assert_eq!(job.status, crate::revai::types::LangIdJobStatus::Completed);
    }

    #[test]
    fn parse_langid_result() {
        let json = r#"{
            "top_language": "es",
            "language_confidences": [
                {"language": "es", "confidence": 0.907},
                {"language": "en", "confidence": 0.08},
                {"language": "nl", "confidence": 0.023}
            ]
        }"#;
        let result: crate::revai::types::LangIdResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.top_language, "es");
        assert_eq!(result.language_confidences.len(), 3);
        assert_eq!(result.language_confidences[0].language, "es");
        assert!(result.language_confidences[0].confidence > 0.9);
        assert_eq!(result.language_confidences[1].language, "en");
    }

    #[test]
    fn parse_langid_result_english_dominant() {
        let json = r#"{
            "top_language": "en",
            "language_confidences": [
                {"language": "en", "confidence": 0.95}
            ]
        }"#;
        let result: crate::revai::types::LangIdResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.top_language, "en");
        assert_eq!(result.language_confidences.len(), 1);
    }
}

#[cfg(test)]
mod transport_failure_tests {
    use super::*;
    use crate::error::ServerError;
    use crate::runner::util::classify_server_error;
    use crate::scheduling::FailureCategory;
    use std::io::Read;
    use std::net::TcpListener;

    /// The deadline is a property of the request: an upload's grows with its
    /// body, so a large recording is never held to an exchange's bound (the
    /// old fixed 30 s total killed every fold-in upload), and the floor rate
    /// is what the allowance is computed from.
    #[test]
    fn upload_deadlines_grow_with_the_body_and_exchanges_stay_fixed() {
        let small = RequestDeadline::Upload { body_bytes: 0 }.duration();
        let large = RequestDeadline::Upload {
            body_bytes: 80_000_000,
        }
        .duration();
        assert_eq!(small, RequestDeadline::UPLOAD_BASE);
        assert_eq!(
            large,
            RequestDeadline::UPLOAD_BASE + Duration::from_secs(80_000_000u64.div_ceil(16 * 1024))
        );
        assert!(
            large > Duration::from_secs(60 * 60),
            "an 80 MB upload gets an hour or more, not seconds"
        );
        assert_eq!(
            RequestDeadline::Exchange.duration(),
            RequestDeadline::EXCHANGE
        );
    }

    /// RED FIRST: an account-level refusal (401, 402, 403) was a terminal
    /// failure of the request, classified with a failed job or an
    /// undecodable response, and kept the same-engine UTR fallback that the
    /// same account would be refused. It is its own category now, not
    /// retried and withholding that fallback; other 4xx stay terminal.
    #[test]
    fn an_account_refusal_is_its_own_category() {
        for status in [401_u16, 402, 403] {
            let error = RevAiError::ApiError {
                endpoint: RevAiEndpoint::SubmitJob,
                status,
                body: "refused".into(),
            };
            assert_eq!(
                error.disposition(),
                RevAiDisposition::AccountRefused,
                "{status}"
            );
            assert_eq!(
                classify_server_error(&ServerError::from(error)),
                FailureCategory::ProviderAccountRefused,
                "{status}"
            );
        }
        let bad_request = RevAiError::ApiError {
            endpoint: RevAiEndpoint::SubmitJob,
            status: 400,
            body: "bad media".into(),
        };
        assert_eq!(bad_request.disposition(), RevAiDisposition::Terminal);
    }

    /// RED FIRST: the account mapping applied to every request, so a 403 on
    /// one job's status or transcript (which may refuse only that resource)
    /// was read as the whole account refused; and a 429 was terminal, so a
    /// rate-limited request was not retried and its UTR fallback was sent
    /// into the same limit. The account refusal is read only from a
    /// submission, and a 429 is transient on every request.
    ///
    /// The bodies here are placeholders: the mapping reads the status and the
    /// request, never the body. No recorded Rev.AI response body backs these
    /// statuses yet; a fixture is owed from a real response.
    #[test]
    fn a_status_means_what_it_means_on_the_request_it_answers() {
        let status = |endpoint, status| RevAiError::ApiError {
            endpoint,
            status,
            body: String::new(),
        };
        for (endpoint, forbidden) in [
            (RevAiEndpoint::SubmitJob, RevAiDisposition::AccountRefused),
            (
                RevAiEndpoint::SubmitLanguageId,
                RevAiDisposition::AccountRefused,
            ),
            (RevAiEndpoint::JobDetails, RevAiDisposition::Terminal),
            (RevAiEndpoint::Transcript, RevAiDisposition::Terminal),
            (RevAiEndpoint::LanguageIdJob, RevAiDisposition::Terminal),
            (RevAiEndpoint::LanguageIdResult, RevAiDisposition::Terminal),
        ] {
            assert_eq!(
                status(endpoint, 429).disposition(),
                RevAiDisposition::Transient,
                "{endpoint}"
            );
            assert_eq!(status(endpoint, 403).disposition(), forbidden, "{endpoint}");
        }
    }

    /// A Rev.AI endpoint that accepts the connection and drops it without
    /// answering, reproducing the transport fault the fleet met on
    /// 2026-09-03: `request or response body error for url (.../jobs)`.
    ///
    /// Reaching the real boundary is the point. The failure being reproduced
    /// is a property of the socket, not of any code of ours, so no double
    /// can stand in for it: only a real client writing a real request body
    /// into a real connection that goes away can produce this error.
    fn endpoint_that_hangs_up_mid_request() -> (String, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("test: bind loopback");
        let addr = listener.local_addr().expect("test: local addr");
        let handle = std::thread::spawn(move || {
            // Accept every connection the retry loop opens, read a little of
            // the request so the client has started writing, then drop the
            // socket. Ends when the listener is dropped with the thread.
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { return };
                let mut scratch = [0u8; 64];
                let _ = stream.read(&mut scratch);
            }
        });
        (format!("http://{addr}/speechtotext/v1"), handle)
    }

    /// An upstream provider that drops the connection is NOT bad input, and
    /// must never be reported to an operator as a validation failure: that is
    /// what sent the 2026-09-03 fresh-ASR run looking for a broken daemon.
    #[test]
    fn rev_transport_failure_classifies_as_a_transient_provider_failure() {
        let (base, _server) = endpoint_that_hangs_up_mid_request();
        let client = RevAiClient::for_test(&base);

        let error = client
            .transcribe_bytes_blocking(
                &vec![0u8; 512 * 1024],
                "provider-media.mp3",
                "audio/mpeg",
                &SubmitOptions {
                    language: "en".to_string(),
                    speakers_count: Some(2),
                    skip_postprocessing: Some(true),
                    metadata: None,
                },
                PollPolicy::TRANSCRIPTION,
            )
            .map(|_| ())
            .expect_err("test: a hung-up endpoint cannot produce a transcript");

        let server_error = ServerError::from(error);
        assert_eq!(
            classify_server_error(&server_error),
            FailureCategory::ProviderTransient,
            "a dropped connection to Rev.AI is a transient provider failure, got: {server_error}"
        );
    }
}

/// The bounded wait, driven through the production path (`submit`, then the
/// poll typestate, then the transcript download) against a scripted loopback
/// Rev.AI. The client's clock is virtual, so a deadline passes in no wall time.
#[cfg(test)]
mod wait_tests {
    use super::super::poll::{PollDeadline, PollInterval, TransientBudget, UnfinishedCause};
    use super::super::scripted_server::{ScriptedReply, ScriptedRevServer, SeenRequest};
    use super::*;

    const SUBMITTED: ScriptedReply =
        ScriptedReply::Json(r#"{"id":"job-1","status":"in_progress"}"#);
    const IN_PROGRESS: ScriptedReply =
        ScriptedReply::Json(r#"{"id":"job-1","status":"in_progress"}"#);
    const TRANSCRIBED: ScriptedReply =
        ScriptedReply::Json(r#"{"id":"job-1","status":"transcribed","language":"en"}"#);
    const FAILED: ScriptedReply = ScriptedReply::Json(
        r#"{"id":"job-1","status":"failed","failure_detail":"media holds no speech"}"#,
    );
    const TRANSCRIPT: ScriptedReply = ScriptedReply::Json(r#"{"monologues":[]}"#);
    /// Rev.AI's service failing on a status poll: transient by the table.
    const UNAVAILABLE: ScriptedReply = ScriptedReply::Status(503);

    fn options() -> SubmitOptions {
        SubmitOptions {
            language: "en".to_string(),
            speakers_count: Some(2),
            skip_postprocessing: Some(true),
            metadata: None,
        }
    }

    /// A policy whose numbers a test can reason about: a budget of two
    /// transient failures, a ten-second deadline, five-second polls.
    fn policy() -> PollPolicy {
        PollPolicy {
            deadline: PollDeadline::of(Duration::from_secs(10)),
            transient_budget: TransientBudget::of(2),
            interval: PollInterval {
                initial: Duration::from_secs(5),
                max: Duration::from_secs(5),
            },
        }
    }

    /// A policy under which the deadline never comes first.
    fn without_deadline_pressure() -> PollPolicy {
        PollPolicy {
            deadline: PollDeadline::of(Duration::from_secs(3600)),
            ..policy()
        }
    }

    fn transcribe(
        server: &ScriptedRevServer,
        policy: PollPolicy,
    ) -> Result<Awaited<TranscriptResult>> {
        RevAiClient::for_test(&server.speech_to_text_base).transcribe_bytes_blocking(
            b"media bytes",
            "provider-media.mp3",
            "audio/mpeg",
            &options(),
            policy,
        )
    }

    fn seen(method: &str, path: &str) -> SeenRequest {
        SeenRequest {
            method: method.to_owned(),
            path: path.to_owned(),
        }
    }

    fn unfinished(
        awaited: Result<Awaited<TranscriptResult>>,
    ) -> super::super::poll::UnfinishedRevJob {
        match awaited {
            Ok(Awaited::Unfinished(unfinished)) => unfinished,
            Ok(Awaited::Ready(_)) => panic!("the wait was expected to end unfinished"),
            Err(error) => panic!("the wait was expected to end unfinished, not fail: {error}"),
        }
    }

    /// The ordinary wait: in progress twice, then transcribed, then the
    /// transcript is downloaded once.
    #[test]
    fn a_job_that_finishes_is_polled_until_transcribed_then_downloaded() {
        let server = ScriptedRevServer::start(vec![
            SUBMITTED,
            IN_PROGRESS,
            IN_PROGRESS,
            TRANSCRIBED,
            TRANSCRIPT,
        ]);
        match transcribe(&server, without_deadline_pressure()) {
            Ok(Awaited::Ready(result)) => {
                let (_, detected) = result.into_parts();
                assert_eq!(detected.as_deref(), Some("en"));
            }
            Ok(Awaited::Unfinished(unfinished)) => {
                panic!("finished job left unfinished: {unfinished}")
            }
            Err(error) => panic!("finished job failed: {error}"),
        }
        assert_eq!(
            server.seen(),
            vec![
                seen("POST", "/speechtotext/v1/jobs"),
                seen("GET", "/speechtotext/v1/jobs/job-1"),
                seen("GET", "/speechtotext/v1/jobs/job-1"),
                seen("GET", "/speechtotext/v1/jobs/job-1"),
                seen("GET", "/speechtotext/v1/jobs/job-1/transcript"),
            ]
        );
    }

    /// RED FIRST: one transient failure of a status poll used to end the job
    /// (the poll's `?`), and the file's retry then re-submitted and re-paid.
    /// The budget absorbs it and the wait goes on to the transcript.
    #[test]
    fn one_transient_poll_failure_is_absorbed_and_the_job_still_completes() {
        let server = ScriptedRevServer::start(vec![
            SUBMITTED,
            IN_PROGRESS,
            UNAVAILABLE,
            TRANSCRIBED,
            TRANSCRIPT,
        ]);
        assert!(
            matches!(
                transcribe(&server, without_deadline_pressure()),
                Ok(Awaited::Ready(_))
            ),
            "a 503 on one status poll must not end the job"
        );
        let polls = server
            .seen()
            .iter()
            .filter(|request| request.path == "/speechtotext/v1/jobs/job-1")
            .count();
        assert_eq!(polls, 3, "in progress, the failed poll, transcribed");
        assert_eq!(
            server.seen().last(),
            Some(&seen("GET", "/speechtotext/v1/jobs/job-1/transcript"))
        );
    }

    /// The transport fault the fleet met on 2026-09-03, on a status poll: the
    /// connection goes away with no answer. Transient by the table, absorbed
    /// like a 503, and the job still completes.
    #[test]
    fn a_hung_up_status_poll_is_absorbed_and_the_job_still_completes() {
        let server = ScriptedRevServer::start(vec![
            SUBMITTED,
            ScriptedReply::HangUp,
            TRANSCRIBED,
            TRANSCRIPT,
        ]);
        assert!(
            matches!(
                transcribe(&server, without_deadline_pressure()),
                Ok(Awaited::Ready(_))
            ),
            "a dropped connection on one status poll must not end the job"
        );
        assert_eq!(
            server.seen().last(),
            Some(&seen("GET", "/speechtotext/v1/jobs/job-1/transcript"))
        );
    }

    /// Budget plus one transient failures end the wait with the job named and
    /// every failure recorded, in order; the job is not abandoned.
    #[test]
    fn exceeding_the_transient_budget_hands_the_job_back_by_name() {
        let server = ScriptedRevServer::start(vec![SUBMITTED, UNAVAILABLE]);
        let unfinished = unfinished(transcribe(&server, without_deadline_pressure()));
        assert_eq!(unfinished.job, "job-1");
        match &unfinished.cause {
            UnfinishedCause::TransientBudgetExhausted { budget, failures } => {
                assert_eq!(*budget, TransientBudget::of(2));
                assert_eq!(
                    failures.len(),
                    3,
                    "budget of two, plus the one that exhausted it"
                );
                assert!(
                    failures.iter().all(|failure| failure.contains("503")),
                    "{failures:?}"
                );
            }
            UnfinishedCause::DeadlineExceeded { .. } => panic!("the budget ran out first"),
        }
        assert!(unfinished.to_string().contains("job-1"), "{unfinished}");
        let polls = server
            .seen()
            .iter()
            .filter(|request| request.path == "/speechtotext/v1/jobs/job-1")
            .count();
        assert_eq!(polls, 3, "no poll after the budget is exhausted");
    }

    /// RED FIRST: a job Rev.AI reported in progress forever held the file
    /// forever. The deadline ends the wait with the job named; the virtual
    /// clock advances by the interval per wait, so ten seconds pass in two
    /// polls and no wall time.
    #[test]
    fn a_job_in_progress_past_the_deadline_is_handed_back_by_name() {
        let server = ScriptedRevServer::start(vec![SUBMITTED, IN_PROGRESS]);
        let unfinished = unfinished(transcribe(&server, policy()));
        assert_eq!(unfinished.job, "job-1");
        match &unfinished.cause {
            UnfinishedCause::DeadlineExceeded { deadline, elapsed } => {
                assert_eq!(*deadline, PollDeadline::of(Duration::from_secs(10)));
                assert!(*elapsed >= Duration::from_secs(10), "{elapsed:?}");
            }
            UnfinishedCause::TransientBudgetExhausted { .. } => {
                panic!("every poll answered; nothing was transient")
            }
        }
        let polls = server
            .seen()
            .iter()
            .filter(|request| request.path == "/speechtotext/v1/jobs/job-1")
            .count();
        assert_eq!(
            polls, 2,
            "two five-second waits reach the ten-second deadline"
        );
    }

    /// Rev.AI's own verdict on the job ends the wait at once, as the terminal
    /// failure it is: no further poll, no transcript request.
    #[test]
    fn a_failed_job_ends_the_wait_immediately_as_a_terminal_failure() {
        let server = ScriptedRevServer::start(vec![SUBMITTED, FAILED]);
        let error = transcribe(&server, without_deadline_pressure())
            .map(|_| ())
            .expect_err("Rev.AI failed the job");
        assert!(
            matches!(error, RevAiError::JobFailed(ref detail) if detail == "media holds no speech"),
            "{error}"
        );
        assert_eq!(error.disposition(), RevAiDisposition::Terminal);
        assert_eq!(
            server.seen(),
            vec![
                seen("POST", "/speechtotext/v1/jobs"),
                seen("GET", "/speechtotext/v1/jobs/job-1"),
            ]
        );
    }

    /// A terminal refusal of the poll itself (the job is gone) ends the wait
    /// at once too; only transient failures consume budget.
    #[test]
    fn a_terminal_poll_refusal_ends_the_wait_immediately() {
        let server = ScriptedRevServer::start(vec![SUBMITTED, ScriptedReply::Status(404)]);
        let error = transcribe(&server, without_deadline_pressure())
            .map(|_| ())
            .expect_err("the job is gone");
        assert!(
            matches!(
                error,
                RevAiError::ApiError {
                    endpoint: RevAiEndpoint::JobDetails,
                    status: 404,
                    ..
                }
            ),
            "{error}"
        );
        assert_eq!(server.seen().len(), 2, "no poll after a terminal refusal");
    }

    /// Resuming an earlier run's job uploads nothing: the first request is the
    /// status poll of the named job.
    #[test]
    fn resuming_a_named_job_polls_it_without_submitting() {
        let server = ScriptedRevServer::start(vec![IN_PROGRESS, TRANSCRIBED, TRANSCRIPT]);
        let client = RevAiClient::for_test(&server.speech_to_text_base);
        let awaited = client
            .resume_transcription_blocking(
                SubmittedJob::named(RevAiJobId::try_new("job-1").expect("a job id")),
                without_deadline_pressure(),
            )
            .expect("the resumed job finishes");
        assert!(matches!(awaited, Awaited::Ready(_)));
        let requests = server.seen();
        assert!(
            requests.iter().all(|request| request.method == "GET"),
            "a resume must not upload: {requests:?}"
        );
        assert_eq!(requests[0], seen("GET", "/speechtotext/v1/jobs/job-1"));
    }
}
