use super::*;
use crate::types::worker_v2::{ProtocolErrorCodeV2, TranslateBackendV2, WorkerRequestIdV2};

fn failed(code: ProtocolErrorCodeV2, message: &str) -> ExecuteResponseV2 {
    ExecuteResponseV2::failure(
        WorkerRequestIdV2::from("req-text"),
        code,
        message.to_owned(),
        crate::api::NonNegativeSeconds::ZERO,
    )
}

/// Guards: text failure codes retain unsupported-language and access diagnoses,
/// and runtime failures use the selected model's actual local/provider category.
/// Kind: policy
/// Not a type: category choices for a decoded protocol code and backend locality
/// are semantic mappings even though invalid outcome/payload combinations are excluded.
/// State: typed response fixtures enter through the normal failure readers.
/// Concurrency: none
#[test]
fn text_failures_retain_protocol_and_backend_locality() {
    let refusal = parse_morphosyntax_result_v2(failed(
        ProtocolErrorCodeV2::UnsupportedLanguage,
        "no Stanza pipeline for 'que'",
    ))
    .unwrap_err();
    match refusal.into_server_error(ModelLocality::Local) {
        ServerError::Worker(WorkerError::Bootstrap { message, .. }) => {
            assert!(message.contains("no Stanza pipeline for 'que'"))
        }
        other => panic!("expected bootstrap, got {other:?}"),
    }
    let response = failed(ProtocolErrorCodeV2::ModelAccessDenied, "gated repository");
    assert!(matches!(
        parse_utseg_result_v2(&response)
            .unwrap_err()
            .into_server_error(ModelLocality::Local),
        ServerError::ModelAccessDenied(_)
    ));
    enum Backend {
        Local,
        Provider,
    }
    for case in [Backend::Local, Backend::Provider] {
        let backend = match case {
            Backend::Local => TranslateBackendV2::Nllb,
            Backend::Provider => TranslateBackendV2::Google,
        };
        let error =
            parse_translate_result_v2(failed(ProtocolErrorCodeV2::RuntimeFailure, "model failure"))
                .unwrap_err()
                .into_server_error(ModelLocality::of_translate(backend));
        match case {
            Backend::Local => assert!(matches!(
                error,
                ServerError::Worker(WorkerError::ModelRuntime(_))
            )),
            Backend::Provider => assert!(matches!(
                error,
                ServerError::Worker(WorkerError::WorkerResponse(_))
            )),
        }
    }
}
