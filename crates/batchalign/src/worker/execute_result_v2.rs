//! One reader for worker-protocol V2 execute responses.
//!
//! [`ExecuteResponseV2`] used to split a single fact across two public fields
//! (an outcome beside an `Option<TaskResultV2>`), so every result adapter
//! checked both by hand, in an order nobody wrote down. Four adapters checked
//! the outcome first and two checked the payload first; since a failed request
//! carries no payload, the two that led with the payload reported every typed
//! error response as "missing a result payload" and threw away the protocol
//! code and message the worker had actually sent.
//!
//! The pairing is structural since 2026-08-21 (the type's interior is a
//! validated enum, and deserialization refuses a disagreeing wire value), so
//! `ExecuteResponseV2::read` is total over the two legal shapes. This module
//! keeps only the shared message shape every adapter uses.

use crate::types::worker_v2::{
    ExecuteOutcomeRef, ExecuteResponseV2, ProtocolErrorCodeV2, TaskResultV2,
};

/// A worker-protocol V2 execute response's failure, as read from the wire.
///
/// Carries the failed [`ProtocolErrorCodeV2`] alongside the fully formatted
/// diagnostic, rather than collapsing them into a bare `String` (the shape
/// every caller used before 2026-09-02). A caller that only needs text for
/// display keeps writing `?` into a `String`-returning function via the
/// [`From`] impl below; the one caller that needs to route on the failure's
/// CATEGORY (`parse_speaker_result_v2`, for a Hugging Face Hub access
/// failure) can read `code` instead of re-deriving it from message text,
/// which the house rules ban.
#[derive(Debug, Clone)]
pub struct ExecuteFailureRead {
    /// The protocol code the worker's response carried.
    pub code: ProtocolErrorCodeV2,
    /// The fully formatted diagnostic: `"worker protocol V2 {task} request
    /// failed with {code:?}: {message}"`.
    pub message: String,
}

impl std::fmt::Display for ExecuteFailureRead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl From<ExecuteFailureRead> for String {
    fn from(value: ExecuteFailureRead) -> Self {
        value.message
    }
}

/// Require that one execute response succeeded and produced a result payload.
///
/// `task` names the task in the diagnostics, so the four adapters share these
/// two message shapes rather than each spelling them out.
pub fn require_success_result<'a>(
    response: &'a ExecuteResponseV2,
    task: &str,
) -> Result<&'a TaskResultV2, ExecuteFailureRead> {
    match response.read() {
        ExecuteOutcomeRef::Success(result) => Ok(result),
        ExecuteOutcomeRef::Failed { code, message } => {
            Err(ExecuteFailureRead::new(task, code, message))
        }
    }
}

/// The owned counterpart of [`require_success_result`]: move the payload out
/// of a successful response, so a consumer can move its parts into its own
/// types instead of cloning them.
pub fn take_success_result(
    response: ExecuteResponseV2,
    task: &str,
) -> Result<TaskResultV2, ExecuteFailureRead> {
    response
        .into_outcome()
        .map_err(|(code, message)| ExecuteFailureRead::new(task, code, &message))
}

impl ExecuteFailureRead {
    /// The one message shape both readers share.
    fn new(task: &str, code: ProtocolErrorCodeV2, message: &str) -> Self {
        Self {
            code,
            message: format!("worker protocol V2 {task} request failed with {code:?}: {message}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::api::DurationSeconds;
    use crate::types::worker_v2::{ProtocolErrorCodeV2, SpeakerResultV2, WorkerRequestIdV2};

    // NOTE (2026-08-21): this module used to also test "success with no
    // payload yields the missing-a-result-payload message". That scenario is
    // no longer WRITABLE: `ExecuteResponseV2` became structural, its interior
    // is a validated enum, and a success without a payload has no constructor
    // and is refused at deserialization. The test went with the state it
    // guarded, per types-over-tests.

    /// The regression this module exists for. A failed response carries no
    /// payload, and the openSMILE and AVQI dispatch paths used to read the
    /// payload first and so answer "missing a result payload". The code and
    /// message must survive instead.
    #[test]
    fn a_failed_response_reports_its_code_and_message_not_a_missing_payload() {
        let failed = ExecuteResponseV2::failure(
            WorkerRequestIdV2::from("req-1".to_owned()),
            ProtocolErrorCodeV2::ModelUnavailable,
            "no speaker host loaded".to_owned(),
            DurationSeconds(0.0),
        );

        let error = require_success_result(&failed, "speaker")
            .expect_err("a failed response must not read as success");

        assert_eq!(error.code, ProtocolErrorCodeV2::ModelUnavailable);
        let rendered = error.to_string();
        assert!(rendered.contains("ModelUnavailable"), "{rendered}");
        assert!(rendered.contains("no speaker host loaded"), "{rendered}");
        assert!(
            !rendered.contains("missing a result payload"),
            "the reported failure was replaced by the missing-payload message: {rendered}"
        );
    }

    /// The code survives so a caller can route on it (this is the type this
    /// module exists for, since 2026-09-02): a Hugging Face Hub access
    /// failure must be distinguishable from a generic runtime crash without
    /// re-parsing the message text.
    #[test]
    fn a_model_access_denied_failure_preserves_its_code() {
        let failed = ExecuteResponseV2::failure(
            WorkerRequestIdV2::from("req-1".to_owned()),
            ProtocolErrorCodeV2::ModelAccessDenied,
            "could not download the Hugging Face model at pyannote/speaker-diarization-community-1"
                .to_owned(),
            DurationSeconds(0.0),
        );

        let error = require_success_result(&failed, "speaker")
            .expect_err("a failed response must not read as success");

        assert_eq!(error.code, ProtocolErrorCodeV2::ModelAccessDenied);
        assert!(
            error
                .to_string()
                .contains("speaker-diarization-community-1")
        );
    }

    /// Existing callers that only need text (ASR, FA, text, opensmile/AVQI)
    /// keep writing `require_success_result(...)?` into a `String`-returning
    /// function: the `?` operator's implicit `From::from` must still work.
    #[test]
    fn converts_into_a_plain_string_via_from() {
        let failed = ExecuteResponseV2::failure(
            WorkerRequestIdV2::from("req-1".to_owned()),
            ProtocolErrorCodeV2::RuntimeFailure,
            "boom".to_owned(),
            DurationSeconds(0.0),
        );

        let error = require_success_result(&failed, "ASR")
            .expect_err("a failed response must not read as success");
        let as_string: String = error.into();

        assert!(as_string.contains("boom"));
    }

    #[test]
    fn a_successful_response_yields_its_payload() {
        let ok = ExecuteResponseV2::success(
            WorkerRequestIdV2::from("req-1".to_owned()),
            TaskResultV2::SpeakerResult(SpeakerResultV2 {
                evidence: crate::types::worker_v2::SpeakerInferenceEvidenceV2::Pyannote {
                    segments: Vec::new(),
                },
            }),
            DurationSeconds(0.0),
        );

        assert!(matches!(
            require_success_result(&ok, "speaker"),
            Ok(TaskResultV2::SpeakerResult(_))
        ));
    }
}
