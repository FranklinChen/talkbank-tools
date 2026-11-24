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
    pub(crate) const fn of_asr(backend: crate::types::worker_v2::AsrBackendV2) -> Self {
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
    pub(crate) const fn of_speaker(backend: crate::types::worker_v2::SpeakerBackendV2) -> Self {
        use crate::types::worker_v2::SpeakerBackendV2;
        match backend {
            SpeakerBackendV2::Pyannote => Self::Local,
            SpeakerBackendV2::PyannoteAi => Self::Remote,
        }
    }

    /// Where a translation backend's model runs.
    pub(crate) const fn of_translate(backend: crate::types::worker_v2::TranslateBackendV2) -> Self {
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
    #[must_use]
    pub const fn code(&self) -> ProtocolErrorCodeV2 {
        self.code
    }

    /// The fully formatted diagnostic.
    #[must_use]
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

    /// The server error this worker-reported failure is, decided by its
    /// protocol code alone (never its message), for every code.
    ///
    /// Exhaustive with no catch-all, so a new protocol code must choose its
    /// category here rather than falling into "invalid input". None of these
    /// is a refusal of the client's input: the request the server built was
    /// refused (`WorkerProtocol`), a model could not load (`WorkerBootstrap`),
    /// the task failed while running (bounded retries with the local or remote
    /// category selected by `locality`), the provider refused to bill the
    /// account, or Hugging Face refused this machine.
    ///
    /// `billed` is the product the request was sent as, which an account
    /// refusal names. Every caller states it, `NotNamed` included, so a
    /// request billed per product cannot lose that name to a default.
    pub(crate) fn into_server_error(
        self,
        locality: ModelLocality,
        billed: crate::error::BilledProduct,
    ) -> crate::error::ServerError {
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
            ProtocolErrorCodeV2::ProviderQuotaExhausted => ServerError::ProviderBillingRefused(
                crate::error::ProviderBillingRefusal::new(billed, self.message),
            ),
            ProtocolErrorCodeV2::ModelAccessDenied => ServerError::ModelAccessDenied(self.message),
            ProtocolErrorCodeV2::ModelOutputInvalid => {
                ServerError::ModelOutputInvalid(self.message)
            }
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
mod tests;
