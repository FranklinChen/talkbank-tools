//! Rust-owned worker-protocol V2 ASR executor control plane.
//!
//! **See also:** [INTERFACE_MAP.md](../../../INTERFACE_MAP.md) section "2. ASR Execution V2" for:
//! - Python caller: `batchalign/worker/_asr_v2.py::execute_asr_request_v2()`
//! - Full Rust/Python responsibility split and input/output contracts.

use batchalign_types::api::{DurationSeconds, LanguageCode3};
use batchalign_types::worker_v2::{
    AsrBackendV2, AsrElementKindV2, AsrElementV2, AsrInputV2, AsrModelIdentityV2, AsrMonologueV2,
    AsrRequestV2, ExecuteRequestV2, MonologueAsrResultV2, ProviderDiarizationV2,
    ProviderSpeakerLabelV2, SpeakerAttributionV2, TaskRequestV2, TaskResultV2,
    WhisperChunkResultV2,
};
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use crate::py_json_bridge::json_value_to_py;
use crate::worker_artifacts::require_mono_prepared_audio;
use crate::worker_execute::{
    ExecuteFailure, ValidatedRequestV2, execute_request_v2, extract_task_payload,
    parse_host_output, require_runner,
};

/// What a provider adapter said about one monologue's speaker.
///
/// A tagged value, mirroring [`SpeakerAttributionV2`] on the wire. It used to
/// be an untagged `i64 | u64 | String` that `stringify_speaker` flattened into
/// a string, so "the provider named speaker 0" and "this engine names nobody"
/// arrived identical and the second was invented by the adapters writing `0`.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ProviderSpeakerInput {
    /// The provider named a speaker, with its own label.
    Attributed { label: String },
    /// The provider separates no speakers and named none.
    Undiarized,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct ProviderAsrElementInput {
    value: String,
    #[serde(default)]
    ts: Option<f64>,
    #[serde(default)]
    end_ts: Option<f64>,
    #[serde(default = "default_provider_element_type", rename = "type")]
    type_name: String,
    #[serde(default)]
    confidence: Option<f64>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct ProviderAsrMonologueInput {
    speaker: ProviderSpeakerInput,
    #[serde(default)]
    elements: Vec<ProviderAsrElementInput>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct ProviderAsrResponseInput {
    lang: LanguageCode3,
    #[serde(default)]
    monologues: Vec<ProviderAsrMonologueInput>,
}

fn default_provider_element_type() -> String {
    "text".to_owned()
}

fn validate_non_negative(label: &str, value: f64) -> Result<(), ExecuteFailure> {
    if value < 0.0 {
        return Err(ExecuteFailure::Runtime(format!(
            "invalid ASR host output: {label} must be >= 0"
        )));
    }
    Ok(())
}

/// The identity this worker recorded when it loaded its ASR engine.
///
/// Read from worker state rather than threaded through the request because the
/// loader and this bridge are the SAME process: a provider response describes
/// speech and never the model behind it, so the load record is the only
/// witness in reach, and a second copy travelling beside the request could only
/// disagree with the first.
fn loaded_worker_identity(
    py: Python<'_>,
    backend: AsrBackendV2,
) -> Result<AsrModelIdentityV2, ExecuteFailure> {
    let state = PyModule::import(py, "batchalign.worker._types")
        .and_then(|module| module.getattr("_state"))
        .map_err(|error| ExecuteFailure::Runtime(error.to_string()))?;
    let identity = state
        .getattr("asr_model_identity")
        .map_err(|error| ExecuteFailure::Runtime(error.to_string()))?;
    if identity.is_none() {
        // Named per provider, so an operator reads WHICH loader failed to
        // record itself rather than a message true of any of five engines.
        //
        // Refused, never filled in. The request is sitting right here and its
        // `models` field would make a plausible-looking substitute, which is
        // exactly why it must not be used: writing the requested revisions into
        // the observed half would record what was ASKED FOR as though it had
        // been SEEN, and that substitution is the whole defect this workstream
        // removes. An engine that cannot say what it loaded has nothing to
        // report, and a transcript nobody can attribute is worse than a failed
        // job.
        return Err(ExecuteFailure::ModelUnavailable(format!(
            "the {backend:?} ASR engine served a request in this worker without having \
             recorded which models it loaded. Its loader must set `asr_model_identity` \
             when it loads, as the Hugging Face Whisper handle already does."
        )));
    }
    parse_host_output(&identity, "ASR model identity")
}

/// Refuse a response whose models are not the ones the plan pinned.
///
/// This is the refusal point named in the workstream: the worker may load only
/// what the request asked for, and a disagreement is reported by name (which
/// role, which id, requested against observed) rather than silently producing a
/// transcript from another checkpoint.
fn admit_identity(
    model: &AsrModelIdentityV2,
    asr_request: &AsrRequestV2,
) -> Result<(), ExecuteFailure> {
    model
        .admit(&asr_request.models, asr_request.backend)
        .map_err(|mismatch| ExecuteFailure::Runtime(mismatch.to_string()))
}

fn parse_whisper_result(
    response: &Bound<'_, PyAny>,
) -> Result<WhisperChunkResultV2, ExecuteFailure> {
    let parsed: WhisperChunkResultV2 = parse_host_output(response, "ASR")?;

    for chunk in &parsed.chunks {
        validate_non_negative("Whisper chunk start_s", chunk.start_s.0)?;
        validate_non_negative("Whisper chunk end_s", chunk.end_s.0)?;
        if chunk.end_s < chunk.start_s {
            return Err(ExecuteFailure::Runtime(
                "invalid ASR host output: Whisper chunk end_s must be >= start_s".to_owned(),
            ));
        }
    }

    Ok(parsed)
}

/// Admit one monologue's speaker against what the REQUEST asked for.
///
/// The refusal this makes possible is the point: when separation was requested
/// and the provider named nobody, that is a provider or configuration failure,
/// and it used to become speaker 0, which is a legal track. Reading the request
/// rather than guessing is what tells the two apart, and the request now
/// carries the question in a type ([`ProviderDiarizationV2`]) instead of a
/// bare count that could not express "do not separate".
fn admit_speaker(
    speaker: ProviderSpeakerInput,
    diarization: ProviderDiarizationV2,
    monologue_index: usize,
) -> Result<SpeakerAttributionV2, ExecuteFailure> {
    match speaker {
        // A named speaker is admitted whether or not separation was asked for:
        // a provider that volunteers labels is telling the truth about its own
        // output, and discarding them would lose real attribution.
        ProviderSpeakerInput::Attributed { label } => ProviderSpeakerLabelV2::try_from(label)
            .map(|label| SpeakerAttributionV2::Attributed { label })
            .map_err(|error| {
                ExecuteFailure::Runtime(format!(
                    "invalid ASR host output: monologue {monologue_index} {error}"
                ))
            }),
        ProviderSpeakerInput::Undiarized if diarization.is_requested() => {
            Err(ExecuteFailure::Runtime(format!(
                "invalid ASR host output: monologue {monologue_index} carries no speaker, \
                 but this request asked the provider to separate speakers. A transcript \
                 whose speakers cannot be attributed is not silently collapsed onto one \
                 track; check the provider's diarization settings or credentials."
            )))
        }
        ProviderSpeakerInput::Undiarized => Ok(SpeakerAttributionV2::Undiarized),
    }
}

fn parse_provider_result(
    response: &Bound<'_, PyAny>,
    model: AsrModelIdentityV2,
    diarization: ProviderDiarizationV2,
) -> Result<MonologueAsrResultV2, ExecuteFailure> {
    let parsed: ProviderAsrResponseInput = parse_host_output(response, "ASR")?;

    let mut monologues = Vec::with_capacity(parsed.monologues.len());
    for (monologue_index, monologue) in parsed.monologues.into_iter().enumerate() {
        let mut elements = Vec::with_capacity(monologue.elements.len());
        for element in monologue.elements {
            if let Some(start_s) = element.ts {
                validate_non_negative("ASR element start_s", start_s)?;
            }
            if let Some(end_s) = element.end_ts {
                validate_non_negative("ASR element end_s", end_s)?;
            }
            if let (Some(start_s), Some(end_s)) = (element.ts, element.end_ts)
                && end_s < start_s
            {
                return Err(ExecuteFailure::Runtime(
                    "invalid ASR host output: ASR element end_s must be >= start_s".to_owned(),
                ));
            }

            let kind = if element.type_name == "punctuation" {
                AsrElementKindV2::Punctuation
            } else {
                AsrElementKindV2::Text
            };

            elements.push(AsrElementV2 {
                value: element.value,
                start_s: element.ts.map(DurationSeconds),
                end_s: element.end_ts.map(DurationSeconds),
                kind,
                confidence: element.confidence,
            });
        }

        monologues.push(AsrMonologueV2 {
            speaker: admit_speaker(monologue.speaker, diarization, monologue_index)?,
            elements,
        });
    }

    Ok(MonologueAsrResultV2 {
        lang: parsed.lang,
        monologues,
        model,
    })
}

fn load_local_whisper_audio(
    request: &ExecuteRequestV2,
    asr_request: &AsrRequestV2,
) -> Result<Vec<f32>, ExecuteFailure> {
    let audio_ref_id = match &asr_request.input {
        AsrInputV2::PreparedAudio(value) => value.audio_ref_id.as_ref(),
        _ => {
            return Err(ExecuteFailure::InvalidPayload(
                "ASR backend expected prepared_audio input".to_owned(),
            ));
        }
    };

    require_mono_prepared_audio(&request.attachments, audio_ref_id, "worker protocol V2 ASR")?
        .samples()
}

fn build_provider_media_item<'py>(
    py: Python<'py>,
    asr_request: &AsrRequestV2,
) -> Result<Bound<'py, PyAny>, ExecuteFailure> {
    let provider_input = match &asr_request.input {
        AsrInputV2::ProviderMedia(value) => value,
        _ => {
            return Err(ExecuteFailure::InvalidPayload(
                "ASR backend expected provider_media input".to_owned(),
            ));
        }
    };

    let asr_module = PyModule::import(py, "batchalign.inference.asr")
        .map_err(|error| ExecuteFailure::Runtime(error.to_string()))?;
    let asr_batch_item = asr_module
        .getattr("AsrBatchItem")
        .map_err(|error| ExecuteFailure::Runtime(error.to_string()))?;
    let kwargs = PyDict::new(py);
    kwargs
        .set_item("audio_path", provider_input.media_path.as_ref())
        .map_err(|error| ExecuteFailure::Runtime(error.to_string()))?;
    kwargs
        .set_item("lang", asr_request.lang.as_worker_arg())
        .map_err(|error| ExecuteFailure::Runtime(error.to_string()))?;
    // The request's own typed answer to "separate speakers, and into how
    // many?", carried across as the tagged value it is and parsed once by
    // `AsrBatchItem`. It used to cross as an integer whose ZERO meant "do not
    // separate", a spelling only this function and the Tencent recognizer knew,
    // beside a Python default of 1 that meant the contradiction submission
    // refuses. One meaning now has one representation on both sides.
    //
    // Serialized rather than assembled here, so the tags are serde's and the
    // generated schema's, not a third copy written out at this call site.
    let diarization = serde_json::to_value(provider_input.diarization).map_err(|error| {
        ExecuteFailure::Runtime(format!(
            "worker protocol V2 ASR request could not serialize its diarization: {error}"
        ))
    })?;
    kwargs
        .set_item(
            "diarization",
            json_value_to_py(py, &diarization)
                .map_err(|error| ExecuteFailure::Runtime(error.to_string()))?,
        )
        .map_err(|error| ExecuteFailure::Runtime(error.to_string()))?;
    // Forward the request's own decode budget verbatim: `None` when Rust
    // could not derive one (see `AsrRequestV2::decode_budget_seconds`),
    // never a fabricated value. The native Qwen3-ASR engine is the only
    // consumer today; other providers ignore the extra kwarg.
    kwargs
        .set_item(
            "decode_budget_seconds",
            asr_request
                .decode_budget_seconds
                .map(|budget| budget.as_seconds()),
        )
        .map_err(|error| ExecuteFailure::Runtime(error.to_string()))?;
    asr_batch_item
        .call((), Some(&kwargs))
        .map_err(|error| ExecuteFailure::Runtime(error.to_string()))
}

fn run_local_whisper(
    py: Python<'_>,
    request: &ExecuteRequestV2,
    asr_request: &AsrRequestV2,
    local_whisper_runner: Option<Py<PyAny>>,
) -> Result<TaskResultV2, ExecuteFailure> {
    let audio = load_local_whisper_audio(request, asr_request)?;
    let runner = require_runner(
        local_whisper_runner,
        "no local Whisper ASR host loaded for worker protocol V2",
    )?;
    let audio_array = audio.into_pyarray(py);
    let response = runner
        .bind(py)
        .call1((audio_array, asr_request.lang.as_worker_arg()))
        .map_err(|error| ExecuteFailure::Runtime(error.to_string()))?;
    let parsed = parse_whisper_result(&response)?;
    // The Python handle stamps its own identity onto the payload, because it is
    // the object the snapshot was resolved into. Checking it here is what makes
    // that claim load-bearing rather than decorative.
    admit_identity(&parsed.model, asr_request)?;
    Ok(TaskResultV2::WhisperChunkResult(parsed))
}

fn run_provider_backend(
    py: Python<'_>,
    asr_request: &AsrRequestV2,
    provider_runner: Option<Py<PyAny>>,
    unavailable_message: &'static str,
) -> Result<TaskResultV2, ExecuteFailure> {
    // Read from the request, once, so the admission below judges what was
    // ASKED FOR rather than what came back.
    let diarization = match &asr_request.input {
        AsrInputV2::ProviderMedia(value) => value.diarization,
        _ => {
            return Err(ExecuteFailure::InvalidPayload(
                "ASR backend expected provider_media input".to_owned(),
            ));
        }
    };
    let item = build_provider_media_item(py, asr_request)?;
    let runner = require_runner(provider_runner, unavailable_message)?;
    // Admitted BEFORE the provider runs: these backends are paid cloud calls,
    // and a plan/worker disagreement is knowable without spending one.
    let model = loaded_worker_identity(py, asr_request.backend)?;
    admit_identity(&model, asr_request)?;
    let response = runner
        .bind(py)
        .call1((item,))
        .map_err(|error| ExecuteFailure::Runtime(error.to_string()))?;
    Ok(TaskResultV2::MonologueAsrResult(parse_provider_result(
        &response,
        model,
        diarization,
    )?))
}

fn run_asr(
    py: Python<'_>,
    request: ValidatedRequestV2<'_>,
    local_whisper_runner: Option<Py<PyAny>>,
    hk_tencent_runner: Option<Py<PyAny>>,
    hk_aliyun_runner: Option<Py<PyAny>>,
    hk_funaudio_runner: Option<Py<PyAny>>,
    hk_qwen_runner: Option<Py<PyAny>>,
) -> Result<TaskResultV2, ExecuteFailure> {
    let asr_request = extract_task_payload(
        &request,
        |payload| match payload {
            TaskRequestV2::Asr(value) => Some(value),
            _ => None,
        },
        "ASR",
    )?;
    match asr_request.backend {
        // ``WhisperHub`` shares the worker-side runtime shape with
        // ``LocalWhisper``, both host a ``WhisperASRHandle`` loaded at
        // worker bootstrap and receive prepared mono audio as the request
        // input. The distinction lives at load time (which checkpoint got
        // loaded) and at the worker-pool key (so separate workers serve
        // each variant); the PyO3 dispatch treats them identically.
        AsrBackendV2::LocalWhisper | AsrBackendV2::WhisperHub => {
            run_local_whisper(py, &request, asr_request, local_whisper_runner)
        }
        AsrBackendV2::HkTencent => run_provider_backend(
            py,
            asr_request,
            hk_tencent_runner,
            "no Tencent ASR host loaded for worker protocol V2",
        ),
        AsrBackendV2::HkAliyun => run_provider_backend(
            py,
            asr_request,
            hk_aliyun_runner,
            "no Aliyun ASR host loaded for worker protocol V2",
        ),
        AsrBackendV2::HkFunaudio => run_provider_backend(
            py,
            asr_request,
            hk_funaudio_runner,
            "no FunAudio ASR host loaded for worker protocol V2",
        ),
        AsrBackendV2::HkQwen => run_provider_backend(
            py,
            asr_request,
            hk_qwen_runner,
            "no Qwen3-ASR host loaded for worker protocol V2",
        ),
        AsrBackendV2::Revai => Err(ExecuteFailure::ModelUnavailable(
            "Rev.AI is handled directly by the Rust control plane, not the Python worker"
                .to_owned(),
        )),
    }
}

#[pyfunction]
#[pyo3(signature = (
    request,
    local_whisper_runner=None,
    hk_tencent_runner=None,
    hk_aliyun_runner=None,
    hk_funaudio_runner=None,
    hk_qwen_runner=None
))]
pub(crate) fn execute_asr_request_v2(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
    local_whisper_runner: Option<Py<PyAny>>,
    hk_tencent_runner: Option<Py<PyAny>>,
    hk_aliyun_runner: Option<Py<PyAny>>,
    hk_funaudio_runner: Option<Py<PyAny>>,
    hk_qwen_runner: Option<Py<PyAny>>,
) -> PyResult<String> {
    execute_request_v2(request, |request| {
        run_asr(
            py,
            request,
            local_whisper_runner,
            hk_tencent_runner,
            hk_aliyun_runner,
            hk_funaudio_runner,
            hk_qwen_runner,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use batchalign_types::api::NumSpeakers;

    /// A request that asked the provider to separate speakers, with a real
    /// count. One is not a shape this can take: submission refuses it.
    fn separation_requested() -> ProviderDiarizationV2 {
        ProviderDiarizationV2::for_backend(AsrBackendV2::HkTencent, NumSpeakers(2))
    }

    /// The failure an admission refused with, or a panic naming what it
    /// returned instead. `ExecuteFailure` carries no `Debug`, so neither
    /// `expect_err` nor `expect` is available on these results.
    fn refusal_message(result: Result<SpeakerAttributionV2, ExecuteFailure>) -> String {
        match result {
            Ok(admitted) => panic!("expected a refusal, got {admitted:?}"),
            Err(ExecuteFailure::Runtime(message)) => message,
            Err(_) => panic!("an unattributable monologue is a runtime failure"),
        }
    }

    /// The attribution an admission accepted, or a panic. The mirror of
    /// [`refusal_message`], and needed for the same reason.
    fn admitted(result: Result<SpeakerAttributionV2, ExecuteFailure>) -> SpeakerAttributionV2 {
        match result {
            Ok(attribution) => attribution,
            Err(_) => panic!("expected an admission, got a refusal"),
        }
    }

    #[test]
    fn a_named_speaker_is_admitted_whether_or_not_separation_was_requested() {
        // A provider that volunteers a label is telling the truth about its
        // own output, so the label survives either request shape.
        for diarization in [separation_requested(), ProviderDiarizationV2::NotRequested] {
            let attribution = admitted(admit_speaker(
                ProviderSpeakerInput::Attributed {
                    label: "2".to_owned(),
                },
                diarization,
                0,
            ));

            assert_eq!(
                attribution,
                SpeakerAttributionV2::Attributed {
                    label: ProviderSpeakerLabelV2::try_from("2").expect("a non-blank label"),
                }
            );
        }
    }

    #[test]
    fn an_unnamed_speaker_is_refused_when_separation_was_requested() {
        // The refusal this whole tagged value exists for. Collapsing these
        // onto one track is what produced a single PAR0 tier for a recording
        // the caller had asked to have separated.
        let message = refusal_message(admit_speaker(
            ProviderSpeakerInput::Undiarized,
            separation_requested(),
            3,
        ));

        assert!(message.contains("monologue 3"), "{message}");
        assert!(message.contains("carries no speaker"), "{message}");
    }

    #[test]
    fn an_unnamed_speaker_is_undiarized_when_no_separation_was_requested() {
        // Aliyun, FunASR, Whisper and Qwen all report this, and it is not an
        // error: nobody asked them to separate anyone.
        let attribution = admitted(admit_speaker(
            ProviderSpeakerInput::Undiarized,
            ProviderDiarizationV2::NotRequested,
            0,
        ));

        assert_eq!(attribution, SpeakerAttributionV2::Undiarized);
    }

    #[test]
    fn a_blank_provider_label_is_refused_with_its_monologue_index() {
        // A blank label distinguishes nothing while looking like an answer.
        let message = refusal_message(admit_speaker(
            ProviderSpeakerInput::Attributed {
                label: "   ".to_owned(),
            },
            ProviderDiarizationV2::NotRequested,
            1,
        ));

        assert!(message.contains("monologue 1"), "{message}");
    }
}
