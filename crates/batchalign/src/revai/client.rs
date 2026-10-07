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

use crate::api::AudioPositionSeconds;

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
}

impl std::fmt::Display for RevAiDisposition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transient => write!(f, "transient"),
            Self::Terminal => write!(f, "terminal"),
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
    #[error("Rev.AI returned HTTP {status}: {body}")]
    ApiError {
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
            Self::ApiError { status, .. } if *status >= 500 => RevAiDisposition::Transient,
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
        })
    }

    /// A client pointed at a local test server, with the backoff removed.
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
                        let error = RevAiError::ApiError { status, body };
                        // Retry only what the error itself says is worth
                        // retrying, rather than re-deciding the 5xx boundary
                        // here in a second place.
                        match error.disposition() {
                            RevAiDisposition::Transient => attempts_failed.push(error),
                            RevAiDisposition::Terminal => return Err(error),
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
    fn get_job_details(&self, job_id: &str) -> Result<Job> {
        let resp = self
            .client
            .get(format!("{}/jobs/{job_id}", self.endpoints.speech_to_text))
            .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
            .send()?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = read_error_body(resp);
            return Err(RevAiError::ApiError { status, body });
        }

        Ok(resp.json()?)
    }

    /// Download the transcript for a completed job.
    fn get_transcript(&self, job_id: &str) -> Result<RevTranscriptEvidence> {
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
            return Err(RevAiError::ApiError { status, body });
        }

        retain_transcript_body(resp.bytes()?.to_vec())
    }

    pub(super) fn transcribe_bytes_blocking(
        &self,
        bytes: &[u8],
        file_name: &str,
        mime: &str,
        opts: &SubmitOptions,
        max_poll_secs: u64,
    ) -> Result<TranscriptResult> {
        let job = self.submit_media_bytes(bytes, file_name, mime, opts)?;
        self.poll_and_download(&job.id, 5, max_poll_secs)
    }

    /// Poll a previously submitted job until it completes, then download it.
    ///
    /// Returns a [`TranscriptResult`] that includes the detected language
    /// (when `language: "auto"` was used at submission time).
    fn poll_and_download(
        &self,
        job_id: &str,
        initial_interval_secs: u64,
        max_interval_secs: u64,
    ) -> Result<TranscriptResult> {
        let mut interval = initial_interval_secs;
        let mut attempts: u32 = 0;

        loop {
            let job = self.get_job_details(job_id)?;
            match job.status {
                JobStatus::InProgress => {
                    thread::sleep(Duration::from_secs(interval));
                    attempts += 1;
                    if attempts.is_multiple_of(3) {
                        interval = (interval * 2).min(max_interval_secs);
                    }
                }
                JobStatus::Transcribed => {
                    let transcript = self.get_transcript(job_id)?;
                    return Ok(TranscriptResult {
                        transcript,
                        detected_language: job.language,
                    });
                }
                JobStatus::Failed => {
                    let detail = job.failure_detail.unwrap_or_else(|| "unknown error".into());
                    return Err(RevAiError::JobFailed(detail));
                }
            }
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
            return Err(RevAiError::ApiError { status, body });
        }

        Ok(resp.json()?)
    }

    /// Poll a language identification job status.
    fn get_langid_job(&self, job_id: &str) -> Result<LangIdJob> {
        let resp = self
            .client
            .get(format!("{}/jobs/{job_id}", self.endpoints.language_id))
            .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
            .send()?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = read_error_body(resp);
            return Err(RevAiError::ApiError { status, body });
        }

        Ok(resp.json()?)
    }

    /// Download the language identification result for a completed job.
    fn get_langid_result(&self, job_id: &str) -> Result<LangIdResult> {
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
            return Err(RevAiError::ApiError { status, body });
        }

        Ok(resp.json()?)
    }

    pub(super) fn identify_language_bytes_blocking(
        &self,
        bytes: &[u8],
        file_name: &str,
        mime: &str,
        max_poll_secs: u64,
    ) -> Result<LangIdResult> {
        let job = self.submit_langid_bytes(bytes, file_name, mime)?;
        self.poll_langid_blocking(&job.id, max_poll_secs)
    }

    fn poll_langid_blocking(&self, job_id: &str, max_poll_secs: u64) -> Result<LangIdResult> {
        let mut interval: u64 = 3;
        let mut attempts: u32 = 0;

        loop {
            let status = self.get_langid_job(job_id)?;
            match status.status {
                LangIdJobStatus::InProgress => {
                    thread::sleep(Duration::from_secs(interval));
                    attempts += 1;
                    if attempts.is_multiple_of(3) {
                        interval = (interval * 2).min(max_poll_secs);
                    }
                }
                LangIdJobStatus::Completed => {
                    return self.get_langid_result(job_id);
                }
                LangIdJobStatus::Failed => {
                    let detail = status
                        .failure_detail
                        .unwrap_or_else(|| "unknown error".into());
                    return Err(RevAiError::JobFailed(detail));
                }
            }
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
                1,
            )
            .expect_err("test: a hung-up endpoint cannot produce a transcript");

        let server_error = ServerError::from(error);
        assert_eq!(
            classify_server_error(&server_error),
            FailureCategory::ProviderTransient,
            "a dropped connection to Rev.AI is a transient provider failure, got: {server_error}"
        );
    }
}
