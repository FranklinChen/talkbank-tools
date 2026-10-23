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
/// failure) can read [`Self::code`] instead of re-deriving it from message
/// text, which the house rules ban.
///
/// Built only by the readers below, from a response the worker sent: the
/// fields are private, so no caller pairs a code with a message of its own.
#[derive(Debug, Clone)]
pub struct ExecuteFailureRead {
    /// The protocol code the worker's response carried.
    code: ProtocolErrorCodeV2,
    /// The fully formatted diagnostic: `"worker protocol V2 {task} request
    /// failed with {code:?}: {message}"`.
    message: String,
}

impl std::fmt::Display for ExecuteFailureRead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ExecuteFailureRead {}

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

/// Where the model behind a failed request runs, which decides what its
/// `RuntimeFailure` means.
///
/// Locality selects the diagnostic category, not a claim about determinism.
/// Both local and remote generic failures can retry within the bounded policy.
/// Decided from the engine the request named, by an exhaustive match, never
/// from the message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelLocality {
    /// The model runs in the worker on this machine.
    Local,
    /// The worker calls a remote provider.
    Remote,
}

impl ModelLocality {
    /// Where an ASR backend's model runs.
    pub(crate) fn of_asr(backend: crate::types::worker_v2::AsrBackendV2) -> Self {
        use crate::types::worker_v2::AsrBackendV2;
        match backend {
            AsrBackendV2::LocalWhisper
            | AsrBackendV2::WhisperHub
            | AsrBackendV2::HkFunaudio
            | AsrBackendV2::HkQwen => Self::Local,
            AsrBackendV2::HkTencent | AsrBackendV2::HkAliyun | AsrBackendV2::Revai => Self::Remote,
        }
    }

    /// Where a speaker backend's model runs.
    pub(crate) fn of_speaker(backend: crate::types::worker_v2::SpeakerBackendV2) -> Self {
        use crate::types::worker_v2::SpeakerBackendV2;
        match backend {
            SpeakerBackendV2::Pyannote | SpeakerBackendV2::Nemo => Self::Local,
            SpeakerBackendV2::PyannoteAi => Self::Remote,
        }
    }

    /// Where a translation backend's model runs.
    pub(crate) fn of_translate(backend: crate::types::worker_v2::TranslateBackendV2) -> Self {
        use crate::types::worker_v2::TranslateBackendV2;
        match backend {
            TranslateBackendV2::Seamless | TranslateBackendV2::Nllb => Self::Local,
            TranslateBackendV2::Google
            | TranslateBackendV2::Tencent
            | TranslateBackendV2::Aliyun => Self::Remote,
        }
    }
}

impl ExecuteFailureRead {
    /// The protocol code the worker's response carried.
    pub fn code(&self) -> ProtocolErrorCodeV2 {
        self.code
    }

    /// The fully formatted diagnostic.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The same failure, its message prefixed with where it happened (a
    /// request, a group). The code is unchanged.
    pub(crate) fn in_context(self, context: impl std::fmt::Display) -> Self {
        Self {
            code: self.code,
            message: format!("{context}: {}", self.message),
        }
    }

    /// A failure as a reader would build it, for tests of what consumes one.
    #[cfg(test)]
    pub(crate) fn for_test(code: ProtocolErrorCodeV2, message: &str) -> Self {
        Self {
            code,
            message: message.to_owned(),
        }
    }

    /// The server error this worker-reported failure is, decided by its
    /// protocol code alone (never its message), for every code.
    ///
    /// Exhaustive with no catch-all, so a new protocol code must choose its
    /// category here rather than falling into "invalid input". None of these
    /// is a refusal of the client's input: the request the server built was
    /// refused (`WorkerProtocol`), a model could not load (`WorkerBootstrap`),
    /// the task failed while running (bounded retries with the local or remote
    /// category selected by `locality`), the provider's quota is exhausted, or
    /// Hugging Face refused this machine.
    pub(crate) fn into_server_error(self, locality: ModelLocality) -> crate::error::ServerError {
        use crate::error::ServerError;
        use crate::worker::error::{WorkerAfterError, WorkerError};
        match self.code {
            ProtocolErrorCodeV2::UnsupportedProtocol
            | ProtocolErrorCodeV2::InvalidPayload
            | ProtocolErrorCodeV2::MissingAttachment
            | ProtocolErrorCodeV2::AttachmentUnreadable => {
                ServerError::Worker(WorkerError::RequestRefused(self.message))
            }
            // A worker with no pipeline for the language fails the same way
            // on every retry, as when the language is refused while the
            // worker starts: deterministic, and its message is the operator's
            // answer.
            ProtocolErrorCodeV2::ModelUnavailable | ProtocolErrorCodeV2::UnsupportedLanguage => {
                ServerError::Worker(WorkerError::Bootstrap {
                    message: self.message,
                    worker: WorkerAfterError::Serves,
                })
            }
            // A download that failed in transit, or a device that ran out of
            // memory, may succeed later, wherever the model runs. The worker
            // answered with a response, so it goes on serving.
            ProtocolErrorCodeV2::ModelDownloadFailed => {
                ServerError::Worker(WorkerError::ModelDownload {
                    message: self.message,
                    worker: WorkerAfterError::Serves,
                })
            }
            ProtocolErrorCodeV2::DeviceOutOfMemory => {
                ServerError::Worker(WorkerError::DeviceOutOfMemory {
                    message: self.message,
                    worker: WorkerAfterError::Serves,
                })
            }
            ProtocolErrorCodeV2::RuntimeFailure => match locality {
                ModelLocality::Local => {
                    ServerError::Worker(WorkerError::ModelRuntime(self.message))
                }
                ModelLocality::Remote => {
                    ServerError::Worker(WorkerError::WorkerResponse(self.message))
                }
            },
            ProtocolErrorCodeV2::ProviderQuotaExhausted => {
                ServerError::ProviderQuotaExhausted(self.message)
            }
            ProtocolErrorCodeV2::ModelAccessDenied => ServerError::ModelAccessDenied(self.message),
        }
    }

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
    use crate::api::NonNegativeSeconds;
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
            NonNegativeSeconds::ZERO,
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
            NonNegativeSeconds::ZERO,
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

    /// RED FIRST: a download that failed in transit and a device that ran
    /// out of memory had no protocol code of their own; the worker reported
    /// both as `runtime_failure`, which for a local model is not retried.
    /// Each is now its own code and is retried, wherever the model runs, and
    /// the worker that answered goes on serving.
    #[test]
    fn a_download_and_a_device_shortfall_are_retried_on_a_serving_worker() {
        use crate::error::ServerError;
        use crate::runner::util::classify_worker_error;
        use crate::worker::error::{WorkerAfterError, WorkerError};

        for code in [
            ProtocolErrorCodeV2::ModelDownloadFailed,
            ProtocolErrorCodeV2::DeviceOutOfMemory,
        ] {
            let failed = ExecuteResponseV2::failure(
                WorkerRequestIdV2::from("req-1".to_owned()),
                code,
                "failed".to_owned(),
                NonNegativeSeconds::ZERO,
            );
            let read = require_success_result(&failed, "utseg").expect_err("a failure");
            let ServerError::Worker(error) = read.into_server_error(ModelLocality::Local) else {
                panic!("a worker failure for {code:?}");
            };
            match &error {
                WorkerError::ModelDownload { worker, .. }
                | WorkerError::DeviceOutOfMemory { worker, .. } => {
                    assert_eq!(*worker, WorkerAfterError::Serves);
                }
                other => panic!("{code:?} became {other:?}"),
            }
            assert!(
                ((classify_worker_error(&error)).retryability()
                    == crate::types::scheduling::Retryability::Retryable),
                "{code:?} must be retried"
            );
        }
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
            NonNegativeSeconds::ZERO,
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
            NonNegativeSeconds::ZERO,
        );

        assert!(matches!(
            require_success_result(&ok, "speaker"),
            Ok(TaskResultV2::SpeakerResult(_))
        ));
    }
}
