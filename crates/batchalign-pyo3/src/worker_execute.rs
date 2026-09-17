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

use batchalign_types::api::DurationSeconds;
use batchalign_types::worker_v2::{
    ExecuteRequestV2, ExecuteResponseV2, ProtocolErrorCodeV2, TaskRequestV2, TaskResultV2,
};
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
            Self::Runtime(message) => (ProtocolErrorCodeV2::RuntimeFailure, message),
        }
    }
}

// The stable identity of the Python exception raised for a Hugging Face Hub
// access/credential failure. Maps a Python class to a Rust marker type at
// compile time, so `classify_runner_error` below can check the ACTUAL
// exception class raised, never its message text (see
// `batchalign/inference/_model_access_errors.py`, the module this exception
// is defined in).
pyo3::import_exception!(
    batchalign.inference._model_access_errors,
    ModelAccessDeniedError
);

/// Classify a `PyErr` raised by an inference-runner callback into the shared
/// V2 failure taxonomy.
///
/// Distinguishes [`ModelAccessDeniedError`] (matched by EXCEPTION CLASS
/// identity, never by parsing the error's message text) from every other
/// exception, which stays a generic [`ExecuteFailure::Runtime`]. Python is
/// the only side that inspects the underlying `huggingface_hub` exception
/// classes (see `classify_huggingface_access_error`); this function exists
/// so that reclassification does not have to be repeated at every Rust call
/// site that invokes a Python runner.
pub(crate) fn classify_runner_error(py: Python<'_>, error: PyErr) -> ExecuteFailure {
    if error.is_instance_of::<ModelAccessDeniedError>(py) {
        ExecuteFailure::ModelAccessDenied(error.to_string())
    } else {
        ExecuteFailure::Runtime(error.to_string())
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

/// One owner of the "this runner is not loaded" refusal every executor makes.
///
/// The message stays the caller's, because it names the specific runner
/// ("no wave2vec FA host loaded ...", "no pyannote speaker host loaded ...");
/// what this owns is the refusal's CATEGORY, so no site can drift to a code
/// other than `model_unavailable`.
pub(crate) fn require_runner(
    runner: Option<Py<PyAny>>,
    unavailable_message: &str,
) -> Result<Py<PyAny>, ExecuteFailure> {
    runner.ok_or_else(|| ExecuteFailure::ModelUnavailable(unavailable_message.to_owned()))
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

    let elapsed_s = DurationSeconds(started_at.elapsed().as_secs_f64());
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
