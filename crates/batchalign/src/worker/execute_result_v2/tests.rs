//! Wire-level failure facts and exhaustive classification policy.
use super::*;

/// Guards: actual failure envelopes retain their code/message and select bootstrap,
/// access, download and memory policies without turning into missing-payload errors.
/// Kind: wire
/// Not a type: external error-code bytes and the chosen retry/locality policy remain
/// observable decisions after structural outcome/payload and failure fields are private.
/// State: local JSON decoding and result readers require no model, pool or filesystem.
/// Concurrency: none
#[test]
fn failure_wire_preserves_diagnostics_and_retry_policy() {
    use crate::error::{BilledProduct, ServerError};
    use crate::runner::util::classify_worker_error;
    use crate::types::scheduling::Retryability;
    use crate::worker::error::{WorkerAfterError, WorkerError};
    enum FailureCase {
        Unavailable,
        Access,
        Download,
        Memory,
    }
    for case in [
        FailureCase::Unavailable,
        FailureCase::Access,
        FailureCase::Download,
        FailureCase::Memory,
    ] {
        let (wire_code, code, message) = match case {
            FailureCase::Unavailable => (
                "model_unavailable",
                ProtocolErrorCodeV2::ModelUnavailable,
                "no speaker host loaded",
            ),
            FailureCase::Access => (
                "model_access_denied",
                ProtocolErrorCodeV2::ModelAccessDenied,
                "gated speaker-diarization-community-1",
            ),
            FailureCase::Download => (
                "model_download_failed",
                ProtocolErrorCodeV2::ModelDownloadFailed,
                "download failed",
            ),
            FailureCase::Memory => (
                "device_out_of_memory",
                ProtocolErrorCodeV2::DeviceOutOfMemory,
                "device shortfall",
            ),
        };
        let bytes = serde_json::json!({"request_id":"failure-contract","outcome":{"kind":"error","code":wire_code,"message":message},"elapsed_s":0.0}).to_string();
        let response: ExecuteResponseV2 = serde_json::from_str(&bytes).unwrap();
        let failure = require_success_result(&response, "speaker").unwrap_err();
        assert_eq!(failure.code(), code);
        assert!(failure.to_string().contains(message));
        assert!(failure.to_string().contains(&format!("{code:?}")));
        let owned = take_success_result(response, "speaker").unwrap_err();
        assert_eq!(failure.message(), owned.message());
        let error = failure.into_server_error(ModelLocality::Local, BilledProduct::NotNamed);
        match case {
            FailureCase::Unavailable => assert!(matches!(
                error,
                ServerError::Worker(WorkerError::Bootstrap { .. })
            )),
            FailureCase::Access => assert!(matches!(error, ServerError::ModelAccessDenied(_))),
            FailureCase::Download | FailureCase::Memory => {
                let ServerError::Worker(worker) = error else {
                    panic!("expected worker failure");
                };
                match &worker {
                    WorkerError::ModelDownload { worker, .. }
                    | WorkerError::DeviceOutOfMemory { worker, .. } => {
                        assert_eq!(*worker, WorkerAfterError::Serves)
                    }
                    other => panic!("unexpected failure {other:?}"),
                }
                assert_eq!(
                    classify_worker_error(&worker).retryability(),
                    Retryability::Retryable
                );
            }
        }
    }
}
