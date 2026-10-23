//! The shared control plane for every worker-protocol V2 executor: one failure
//! taxonomy, and one operation that turns a Python request into a serialized
//! [`ExecuteResponseV2`].
//!
//! **See also:** [INTERFACE_MAP.md](../../../INTERFACE_MAP.md) for the per-task
//! Python callers; this module owns only what all of them share.
//!
//! The ASR, FA and media executors each used to carry a private copy of this
//! control plane. The duplicated primitives moved here, and they are PRIVATE:
//! a public primitive is an invitation to re-implement the verb. Callers reach
//! the whole operation ([`execute_request_v2`]) or nothing.

use std::time::Instant;

use batchalign_types::api::NonNegativeSeconds;
use batchalign_types::worker_v2::{
    ExecuteRequestV2, ExecuteResponseV2, ProtocolErrorCodeV2, TaskRequestV2, TaskResultV2,
};
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;

use crate::error::BatchalignBoundaryError;
use crate::py_json_bridge::py_to_json_value;
use crate::worker_artifacts::{ArtifactFailure, validate_attachment_descriptors};

/// Every way executing a V2 request can fail, and the protocol code each maps
/// to. One owner for a mapping that executors previously wrote out by hand at
/// each of their dispatch sites.
pub(crate) enum ExecuteFailure {
    /// A prepared artifact was missing or unreadable.
    Artifact(ArtifactFailure),
    /// The request shape was wrong for the task.
    InvalidPayload(String),
    /// The model or SDK for the task is not loaded.
    ModelUnavailable(String),
    /// A pinned Hugging Face Hub artifact refused this machine's request: a
    /// gated repository requiring accepted terms, a missing/invalid token,
    /// or no cached copy while offline. Distinct from `Runtime` so the
    /// server can categorize it as a configuration/credential condition on
    /// the OPERATOR's machine rather than a batchalign defect. Produced only
    /// by [`classify_runner_error`], which recognizes the raised Python
    /// exception by CLASS, never by parsing its message text.
    ModelAccessDenied(String),
    /// A provider's structured quota refusal; no identical-request retry can
    /// succeed until the account's allocation changes.
    ProviderQuotaExhausted(String),
    /// The worker has no pipeline for the requested language. Produced only
    /// by [`classify_runner_error`], from the Python class
    /// `UnsupportedLanguageError`, which subclasses `ValueError` and so used
    /// to be read as a malformed payload.
    UnsupportedLanguage(String),
    /// A model or catalog download the task needed failed in transit.
    /// Produced only by [`classify_runner_error`], from the Python owner of
    /// that question (`batchalign.inference._transient_failures`).
    ModelDownloadFailed(String),
    /// The accelerator ran out of memory (`torch.OutOfMemoryError`).
    /// Produced only by [`classify_runner_error`], from the same owner.
    DeviceOutOfMemory(String),
    /// The task ran and failed.
    Runtime(String),
}

impl ExecuteFailure {
    /// Convert into the wire code and message together.
    ///
    /// Deliberately one method rather than a `code()` beside a
    /// `into_message()`: separate accessors are only ever correct when used as
    /// a pair, and nothing would stop one failure's code being joined to
    /// another's message. Returning the pair makes the pairing the only thing
    /// you can express, and it is exactly what
    /// [`ExecuteResponseV2::failure`] consumes.
    fn into_code_and_message(self) -> (ProtocolErrorCodeV2, String) {
        match self {
            Self::Artifact(failure) => failure.into_code_and_message(),
            Self::InvalidPayload(message) => (ProtocolErrorCodeV2::InvalidPayload, message),
            Self::ModelUnavailable(message) => (ProtocolErrorCodeV2::ModelUnavailable, message),
            Self::ModelAccessDenied(message) => (ProtocolErrorCodeV2::ModelAccessDenied, message),
            Self::ProviderQuotaExhausted(message) => {
                (ProtocolErrorCodeV2::ProviderQuotaExhausted, message)
            }
            Self::UnsupportedLanguage(message) => {
                (ProtocolErrorCodeV2::UnsupportedLanguage, message)
            }
            Self::ModelDownloadFailed(message) => {
                (ProtocolErrorCodeV2::ModelDownloadFailed, message)
            }
            Self::DeviceOutOfMemory(message) => (ProtocolErrorCodeV2::DeviceOutOfMemory, message),
            Self::Runtime(message) => (ProtocolErrorCodeV2::RuntimeFailure, message),
        }
    }
}

// The stable identities of the Python exceptions the classifier below
// recognises. Each maps a Python class to a Rust marker type at compile time,
// so `classify_runner_error` checks the ACTUAL exception class raised, never
// its message text. Each class lives in a module that holds nothing heavier,
// so the first failure a worker classifies does not import a model library.
pyo3::import_exception!(
    batchalign.inference._model_access_errors,
    ModelAccessDeniedError
);

pyo3::import_exception!(
    batchalign.inference._provider_errors,
    ProviderQuotaExhaustedError
);

pyo3::import_exception!(batchalign.worker._language_errors, UnsupportedLanguageError);

pyo3::import_exception!(batchalign.worker._model_errors, ModelNotLoadedError);

// pydantic's validation failure: what a runner adapter raises when the batch
// or request it parses has the wrong shape.
pyo3::import_exception!(pydantic, ValidationError);

/// The Python module that owns the two transient-failure questions (is this
/// a download that failed in transit; is this an accelerator out of memory),
/// which the Python request loops read as well, so a failure is classified
/// alike whether it escapes an op or a request.
const TRANSIENT_FAILURES_MODULE: &str = "batchalign.inference._transient_failures";

/// Which of the owner's questions to ask.
#[derive(Clone, Copy)]
enum TransientQuestion {
    ModelDownload,
    DeviceOutOfMemory,
}

impl TransientQuestion {
    const fn function(self) -> &'static str {
        match self {
            Self::ModelDownload => "is_model_download_failure",
            Self::DeviceOutOfMemory => "is_device_out_of_memory",
        }
    }

    /// The failure a yes answer makes of the runner's error.
    fn failure(self, message: String) -> ExecuteFailure {
        match self {
            Self::ModelDownload => ExecuteFailure::ModelDownloadFailed(message),
            Self::DeviceOutOfMemory => ExecuteFailure::DeviceOutOfMemory(message),
        }
    }

    /// Ask the owner about `error`. The owner reads exception classes, and
    /// the cause of a Hub "no cached copy" error, which no single
    /// `import_exception!` class can express.
    fn ask(self, py: Python<'_>, error: &PyErr) -> PyResult<bool> {
        py.import(TRANSIENT_FAILURES_MODULE)?
            .getattr(self.function())?
            .call1((error.value(py),))?
            .extract()
    }
}

/// Classify a `PyErr` raised by an inference-runner callback into the shared
/// V2 failure taxonomy: the one classifier every executor uses.
///
/// By exception class, never message text, most specific first:
///
/// | Python class | Failure |
/// | --- | --- |
/// | `ModelAccessDeniedError` | `ModelAccessDenied` |
/// | `ProviderQuotaExhaustedError` | `ProviderQuotaExhausted` |
/// | a download that failed in transit (see `_transient_failures`) | `ModelDownloadFailed` |
/// | `torch.OutOfMemoryError` (see `_transient_failures`) | `DeviceOutOfMemory` |
/// | `UnsupportedLanguageError` | `UnsupportedLanguage` |
/// | `ModelNotLoadedError` (e.g. `UtsegModelNotFoundError`) | `ModelUnavailable` |
/// | `NotImplementedError` | `ModelUnavailable` |
/// | pydantic `ValidationError` | `InvalidPayload` |
/// | anything else | `Runtime` |
///
/// A plain `ValueError` is a runtime failure, not a malformed payload: models
/// raise it for input they cannot process (a forced-alignment window too
/// short for its transcript, say), and the server recognises those
/// constraints only among runtime failures. Only pydantic's validation
/// failure says the payload's shape was wrong. `UnsupportedLanguageError`
/// subclasses `ValueError`, and `ModelAccessDeniedError` and
/// `ModelNotLoadedError` `RuntimeError`, so the specific classes are checked
/// first.
///
/// If the transient-failure owner cannot be asked (it failed to import, a
/// packaging defect), the failure is a runtime failure whose message says
/// so beside the original, never a silent guess.
fn classify_runner_error(py: Python<'_>, error: PyErr) -> ExecuteFailure {
    let message = error.to_string();
    if error.is_instance_of::<ModelAccessDeniedError>(py) {
        return ExecuteFailure::ModelAccessDenied(message);
    }
    if error.is_instance_of::<ProviderQuotaExhaustedError>(py) {
        return ExecuteFailure::ProviderQuotaExhausted(message);
    }
    for question in [
        TransientQuestion::ModelDownload,
        TransientQuestion::DeviceOutOfMemory,
    ] {
        match question.ask(py, &error) {
            Ok(true) => return question.failure(message),
            Ok(false) => {}
            Err(unclassifiable) => {
                return ExecuteFailure::Runtime(format!(
                    "{message} (the failure could not be classified: \
                     {TRANSIENT_FAILURES_MODULE}.{}: {unclassifiable})",
                    question.function()
                ));
            }
        }
    }
    if error.is_instance_of::<UnsupportedLanguageError>(py) {
        ExecuteFailure::UnsupportedLanguage(message)
    } else if error.is_instance_of::<ModelNotLoadedError>(py)
        || error.is_instance_of::<PyNotImplementedError>(py)
    {
        ExecuteFailure::ModelUnavailable(message)
    } else if error.is_instance_of::<ValidationError>(py) {
        ExecuteFailure::InvalidPayload(message)
    } else {
        ExecuteFailure::Runtime(message)
    }
}

impl From<ArtifactFailure> for ExecuteFailure {
    /// Lets an artifact helper propagate with `?` instead of a `map_err`
    /// closure that stringifies its category away.
    fn from(failure: ArtifactFailure) -> Self {
        Self::Artifact(failure)
    }
}

/// One owner of "is this payload the one I run".
///
/// Task/payload agreement is already checked once by [`validate_request`];
/// this refuses a payload handed to the WRONG executor. It replaced nine
/// hand-written two-arm matches across the executor files, each with its own
/// copy of the message.
pub(crate) fn extract_task_payload<'a, T>(
    request: &'a ExecuteRequestV2,
    project: impl for<'r> Fn(&'r TaskRequestV2) -> Option<&'r T>,
    label: &str,
) -> Result<&'a T, ExecuteFailure> {
    project(&request.payload).ok_or_else(|| {
        ExecuteFailure::InvalidPayload(format!(
            "execute payload did not contain {label} request data"
        ))
    })
}

/// A loaded inference runner: the Python callable an executor hands its
/// prepared inputs to.
///
/// The field is private and the only way to call it is [`Self::call1`], which
/// classifies whatever the runner raises through [`classify_runner_error`].
/// Executors used to call the raw `Py<PyAny>` and map its error themselves,
/// and half of them stringified every exception into a runtime failure, so a
/// gated model or an unsupported language was reported as a crash on those
/// paths and correctly on the others.
pub(crate) struct LoadedRunner(Py<PyAny>);

impl LoadedRunner {
    /// Call the runner with positional `args`, classifying a raised exception
    /// into the shared failure taxonomy.
    pub(crate) fn call1<'py, A>(
        &self,
        py: Python<'py>,
        args: A,
    ) -> Result<Bound<'py, PyAny>, ExecuteFailure>
    where
        A: pyo3::call::PyCallArgs<'py>,
    {
        self.0
            .bind(py)
            .call1(args)
            .map_err(|error| classify_runner_error(py, error))
    }
}

/// One owner of the "this runner is not loaded" refusal every executor makes,
/// and the only constructor of a [`LoadedRunner`].
///
/// The message stays the caller's, because it names the specific runner
/// ("no wave2vec FA host loaded ...", "no pyannote speaker host loaded ...");
/// what this owns is the refusal's CATEGORY, so no site can drift to a code
/// other than `model_unavailable`.
pub(crate) fn require_runner(
    runner: Option<Py<PyAny>>,
    unavailable_message: &str,
) -> Result<LoadedRunner, ExecuteFailure> {
    runner
        .map(LoadedRunner)
        .ok_or_else(|| ExecuteFailure::ModelUnavailable(unavailable_message.to_owned()))
}

fn parse_execute_request(request: &Bound<'_, PyAny>) -> PyResult<ExecuteRequestV2> {
    serde_json::from_value(py_to_json_value(request)?)
        .map_err(|error| BatchalignBoundaryError::internal(error).into_py_err())
}

/// Deserialize whatever a Python inference host returned into `T`.
///
/// The `"invalid <label> host output"` wording is a contract the Python test
/// matrix asserts on, so it has one owner here rather than a copy per executor
/// per failure mode.
pub(crate) fn parse_host_output<T: serde::de::DeserializeOwned>(
    response: &Bound<'_, PyAny>,
    label: &str,
) -> Result<T, ExecuteFailure> {
    let value = py_to_json_value(response).map_err(|error| {
        ExecuteFailure::Runtime(format!("invalid {label} host output: {error}"))
    })?;
    serde_json::from_value(value)
        .map_err(|error| ExecuteFailure::Runtime(format!("invalid {label} host output: {error}")))
}

/// Proof that a parsed request passed the control plane's shared validation:
/// its declared `task` agrees with its payload variant, and its attachment
/// descriptors are self-consistent.
///
/// This is a phase type, not a convenience wrapper. The only constructor is
/// [`validate_request`], private to this module, so an executor's `run`
/// closure CANNOT receive a request that skipped validation: the ordering
/// "validate, then run" has no other signature to travel through. Derefs to
/// the request so executors read fields as before.
#[derive(Clone, Copy)]
pub(crate) struct ValidatedRequestV2<'a>(&'a ExecuteRequestV2);

impl std::ops::Deref for ValidatedRequestV2<'_> {
    type Target = ExecuteRequestV2;

    fn deref(&self) -> &ExecuteRequestV2 {
        self.0
    }
}

/// The single transition from a parsed request to a validated one.
///
/// The task/payload agreement is checked HERE, once. `ExecuteRequestV2`
/// stores the discriminant twice (`task`, and again as the payload variant);
/// five executors used to re-check the pairing with bespoke messages. What
/// remains per executor is only "is this payload the one I run", in its
/// `extract_*` match.
fn validate_request(request: &ExecuteRequestV2) -> Result<ValidatedRequestV2<'_>, ExecuteFailure> {
    let payload_task = request.payload.task();
    if request.task != payload_task {
        return Err(ExecuteFailure::InvalidPayload(format!(
            "declared task {:?} does not match payload task {:?}",
            request.task, payload_task
        )));
    }
    validate_attachment_descriptors(&request.attachments).map_err(ArtifactFailure::from)?;
    if let TaskRequestV2::Asr(asr) = &request.payload
        && let batchalign_types::worker_v2::AsrInputV2::ProviderMedia(media) = &asr.input
        && let Some(speakers) = media.diarization.requested_speakers()
        && !batchalign_types::worker_v2::ProviderDiarizationV2::for_backend(
            asr.backend,
            batchalign_types::api::NumSpeakers(speakers.get()),
        )
        .is_requested()
    {
        return Err(ExecuteFailure::InvalidPayload(format!(
            "ASR backend {:?} does not offer integrated speaker separation",
            asr.backend,
        )));
    }
    Ok(ValidatedRequestV2(request))
}

/// Run one worker-protocol V2 request and serialize its response.
///
/// This owns the whole verb from the parsed request onward: start the clock,
/// validate (task/payload agreement plus attachment descriptors, producing the
/// [`ValidatedRequestV2`] proof `run` requires), invoke `run`, and fold
/// whichever outcome came back into an [`ExecuteResponseV2`]. (Parsing itself
/// still raises across FFI rather than answering, because a request too
/// malformed to parse carries no `request_id` to correlate a response to.)
///
/// Two invariants stop being the caller's problem, which is the point of typing
/// the operation rather than merely sharing its parts:
///
/// - **`outcome` and `result` cannot disagree.** Since 2026-08-21 that is a
///   property of [`ExecuteResponseV2`] itself: its interior is a validated
///   enum and its only constructors are the two legal shapes, so no caller
///   here or anywhere else can pair them wrongly.
/// - **The elapsed time is measured from the right instant.** `started_at` was
///   previously threaded through every response call by hand; a caller passing
///   a fresh `Instant` would have reported a plausible near-zero duration that
///   nothing could distinguish from a fast task.
pub(crate) fn execute_request_v2<F>(request: &Bound<'_, PyAny>, run: F) -> PyResult<String>
where
    F: FnOnce(ValidatedRequestV2<'_>) -> Result<TaskResultV2, ExecuteFailure>,
{
    let request = parse_execute_request(request)?;
    let started_at = Instant::now();

    let outcome = validate_request(&request).and_then(run);

    let elapsed_s = NonNegativeSeconds::from(started_at.elapsed());
    let response = match outcome {
        Ok(result) => ExecuteResponseV2::success(request.request_id.clone(), result, elapsed_s),
        Err(failure) => {
            let (code, message) = failure.into_code_and_message();
            ExecuteResponseV2::failure(request.request_id.clone(), code, message, elapsed_s)
        }
    };

    serde_json::to_string(&response)
        .map_err(|error| BatchalignBoundaryError::internal(error).into_py_err())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The mapping is a POLICY: each variant's protocol code is a contract with
    /// the Python side and with anything that buckets these codes, so it has
    /// real alternatives and a type cannot pin it. Executors used to write this
    /// mapping out at each dispatch site, where a copy could drift silently.
    #[test]
    fn failure_variants_map_to_their_protocol_codes() {
        let cases = [
            (
                ExecuteFailure::Artifact(ArtifactFailure::Missing("m".to_owned())),
                ProtocolErrorCodeV2::MissingAttachment,
            ),
            (
                ExecuteFailure::Artifact(ArtifactFailure::Unreadable("m".to_owned())),
                ProtocolErrorCodeV2::AttachmentUnreadable,
            ),
            (
                ExecuteFailure::InvalidPayload("m".to_owned()),
                ProtocolErrorCodeV2::InvalidPayload,
            ),
            (
                ExecuteFailure::ModelUnavailable("m".to_owned()),
                ProtocolErrorCodeV2::ModelUnavailable,
            ),
            (
                ExecuteFailure::ModelAccessDenied("m".to_owned()),
                ProtocolErrorCodeV2::ModelAccessDenied,
            ),
            (
                ExecuteFailure::Runtime("m".to_owned()),
                ProtocolErrorCodeV2::RuntimeFailure,
            ),
            (
                ExecuteFailure::ProviderQuotaExhausted("m".to_owned()),
                ProtocolErrorCodeV2::ProviderQuotaExhausted,
            ),
            (
                ExecuteFailure::UnsupportedLanguage("m".to_owned()),
                ProtocolErrorCodeV2::UnsupportedLanguage,
            ),
        ];
        for (failure, expected) in cases {
            // Asserting the PAIR: the message travels with its own code, never
            // another's.
            assert_eq!(failure.into_code_and_message(), (expected, "m".to_owned()));
        }
    }

    #[test]
    fn provider_capability_is_checked_before_admitting_a_request() {
        use batchalign_types::worker_v2::{AsrBackendV2, AsrInputV2, ProviderDiarizationV2};
        let mut request: ExecuteRequestV2 = serde_json::from_str(include_str!(
            "../../../tests/fixtures/worker_protocol_v2/execute_request_asr_provider_media.json"
        ))
        .unwrap();
        for backend in [
            AsrBackendV2::HkTencent,
            AsrBackendV2::HkAliyun,
            AsrBackendV2::HkFunaudio,
            AsrBackendV2::HkQwen,
        ] {
            let TaskRequestV2::Asr(asr) = &mut request.payload else {
                panic!("ASR fixture")
            };
            asr.backend = backend;
            let AsrInputV2::ProviderMedia(media) = &mut asr.input else {
                panic!("media fixture")
            };
            media.diarization = ProviderDiarizationV2::for_backend(
                AsrBackendV2::HkTencent,
                batchalign_types::api::NumSpeakers(2),
            );
            assert_eq!(
                validate_request(&request).is_ok(),
                backend == AsrBackendV2::HkTencent
            );
            let TaskRequestV2::Asr(asr) = &mut request.payload else {
                unreachable!()
            };
            let AsrInputV2::ProviderMedia(media) = &mut asr.input else {
                unreachable!()
            };
            media.diarization = ProviderDiarizationV2::NotRequested;
            assert!(validate_request(&request).is_ok());
        }
    }

    /// A missing attachment and an unreadable one are different operator
    /// actions (supply the file, versus investigate the file), and the code
    /// used to be recovered by substring match on the message. This pins that
    /// the category now survives a reworded message, which is exactly what the
    /// old arrangement could not do.
    #[test]
    fn artifact_category_survives_a_reworded_message() {
        // One identical message, two categories: the codes must still differ.
        let text = "no such thing anywhere".to_owned();

        assert_eq!(
            ExecuteFailure::from(ArtifactFailure::Missing(text.clone())).into_code_and_message(),
            (ProtocolErrorCodeV2::MissingAttachment, text.clone())
        );
        assert_eq!(
            ExecuteFailure::from(ArtifactFailure::Unreadable(text.clone())).into_code_and_message(),
            (ProtocolErrorCodeV2::AttachmentUnreadable, text)
        );
    }
}
