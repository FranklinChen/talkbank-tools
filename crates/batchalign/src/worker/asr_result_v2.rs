//! Rust-side adapters for live worker-protocol V2 ASR results.
//!
//! The live V2 ASR worker path now returns typed `ExecuteResponseV2` payloads,
//! but the transcription pipeline still expects the established Rust
//! `AsrResponse` domain. This module keeps that normalization in Rust.

use crate::api::LanguageCode3;
use crate::transcribe::{AsrResponse, AsrToken};
use crate::types::worker_v2::{
    AsrBackendV2, AsrElementKindV2, AsrIdentityMismatchV2, AsrRequestedModelsV2, ExecuteResponseV2,
    SpeakerAttributionV2, TaskResultV2, WhisperChunkResultV2,
};
use crate::worker::execute_result_v2::{ExecuteFailureRead, require_success_result};
use batchalign_transform::asr_postprocess::{
    AsrElement, AsrElementKind, AsrMonologue, AsrOutput, AsrRawText, AsrTimestampSecs, SpeakerIndex,
};
use tracing::warn;

/// Why a live V2 ASR response could not become an [`AsrResponse`].
///
/// Two different facts, kept apart: a response this build cannot lower, and a
/// response describing models the plan did not pin. The second is not a parse
/// failure at all, and flattening it into a message is what let it travel as a
/// string nobody matched on.
#[derive(Debug, thiserror::Error)]
pub enum AsrResponseAdmissionError {
    /// The worker answered with a failure rather than a result.
    ///
    /// Carried whole, not flattened into a message: it holds the protocol
    /// code, which is the part a caller can route on.
    #[error("{0}")]
    Failure(ExecuteFailureRead),
    /// The response could not be lowered into the established ASR domain.
    #[error("{0}")]
    Lowering(String),
    /// The worker reported models that are not the ones the plan pinned.
    #[error(transparent)]
    Identity(#[from] AsrIdentityMismatchV2),
}

impl From<ExecuteFailureRead> for AsrResponseAdmissionError {
    fn from(failure: ExecuteFailureRead) -> Self {
        Self::Failure(failure)
    }
}

impl From<String> for AsrResponseAdmissionError {
    fn from(message: String) -> Self {
        Self::Lowering(message)
    }
}

impl From<&str> for AsrResponseAdmissionError {
    fn from(message: &str) -> Self {
        Self::Lowering(message.to_owned())
    }
}

/// Parse one live V2 ASR execute response into the established Rust ASR
/// domain, refusing a response whose models are not the ones `pinned` names.
///
/// `fallback_lang` is consulted only when the worker's response carries
/// an empty language string. Pass `Some(code)` for `Resolved(code)` jobs;
/// pass `None` for `Auto` jobs, in which case an empty worker response
/// becomes a typed error rather than a silent eng substitution.
///
/// # Why the admission is here and not only in the worker
///
/// The worker admits the same two halves inside the process that loaded the
/// models, and that check is worth keeping: it is the one that can refuse
/// before a paid cloud call is spent. But it cannot be the only one. It runs
/// in the process whose report is the thing in question, while THIS side holds
/// the requested half independently and writes the reported half into the
/// transcript's `asr_model=` stamp. Until this call existed the server copied
/// that identity through unread, so provenance said "pinned and verified" on
/// the strength of a check no server-side code had made. Taking `pinned` as a
/// parameter rather than reading it back off the response is the point: the
/// requested half must come from the plan, never from the answer.
pub fn parse_asr_response_v2(
    response: &ExecuteResponseV2,
    fallback_lang: Option<&LanguageCode3>,
    pinned: &AsrRequestedModelsV2,
    backend: AsrBackendV2,
) -> Result<AsrResponse, AsrResponseAdmissionError> {
    let result = require_success_result(response, "ASR")?;

    match result {
        TaskResultV2::WhisperChunkResult(result) => {
            result.model.admit(pinned, backend)?;
            Ok(whisper_chunk_result_to_asr_response(result, fallback_lang)?)
        }
        TaskResultV2::MonologueAsrResult(result) => {
            result.model.admit(pinned, backend)?;
            Ok(AsrResponse {
                lang: resolve_worker_lang(&result.lang, fallback_lang)?,
                // Carried from the wire result only after the admission above,
                // so what lands here, and in `asr_model=`, is what the worker
                // reported loading AND what this side asked for, rather than
                // what either of them hoped for.
                model: Some(result.model.clone()),
                tokens: result
                    .monologues
                    .iter()
                    .flat_map(|monologue| {
                        monologue.elements.iter().filter_map(|element| {
                            if element.kind != AsrElementKindV2::Text {
                                return None;
                            }

                            let text = element.value.trim();
                            if text.is_empty() {
                                return None;
                            }

                            Some(AsrToken {
                                text: text.to_string(),
                                start_s: element.start_s,
                                end_s: element.end_s,
                                // `None` is this field's own spelling of "no
                                // speaker label", which is exactly what an
                                // undiarized engine reports. It used to receive
                                // the string "0", indistinguishable from a
                                // provider's real first speaker.
                                speaker: speaker_label(&monologue.speaker),
                                confidence: element.confidence,
                            })
                        })
                    })
                    .collect(),
                source_monologues: Some(
                    AsrOutput {
                        monologues: result
                            .monologues
                            .iter()
                            .map(|monologue| AsrMonologue {
                                speaker: speaker_index(&monologue.speaker),
                                elements: monologue
                                    .elements
                                    .iter()
                                    .filter_map(|element| {
                                        let text = element.value.trim();
                                        if text.is_empty() {
                                            return None;
                                        }
                                        Some(AsrElement {
                                            value: AsrRawText::new(text),
                                            ts: AsrTimestampSecs::from(
                                                element.start_s.map(|ts| ts.0),
                                            ),
                                            end_ts: AsrTimestampSecs::from(
                                                element.end_s.map(|ts| ts.0),
                                            ),
                                            kind: match element.kind {
                                                AsrElementKindV2::Text => AsrElementKind::Text,
                                                AsrElementKindV2::Punctuation => {
                                                    AsrElementKind::Punctuation
                                                }
                                            },
                                        })
                                    })
                                    .collect(),
                            })
                            .collect(),
                    }
                    .monologues,
                ),
            })
        }
        TaskResultV2::WhisperTokenTimingResult(_) => {
            Err("worker protocol V2 ASR response returned forced-alignment token data".into())
        }
        TaskResultV2::IndexedWordTimingResult(_) => {
            Err("worker protocol V2 ASR response returned indexed timing data".into())
        }
        TaskResultV2::MorphosyntaxResult(_) => {
            Err("worker protocol V2 ASR response returned morphosyntax data".into())
        }
        TaskResultV2::UtsegResult(_) => {
            Err("worker protocol V2 ASR response returned utterance-segmentation data".into())
        }
        TaskResultV2::TranslationResult(_) => {
            Err("worker protocol V2 ASR response returned translation data".into())
        }
        TaskResultV2::CorefResult(_) => {
            Err("worker protocol V2 ASR response returned coreference data".into())
        }
        TaskResultV2::SpeakerResult(_) => {
            Err("worker protocol V2 ASR response returned speaker diarization data".into())
        }
        TaskResultV2::SpeakerEmbeddingResult(_) => {
            Err("worker protocol V2 ASR response returned speaker embedding data".into())
        }
        TaskResultV2::OpensmileResult(_) => {
            Err("worker protocol V2 ASR response returned openSMILE feature data".into())
        }
        TaskResultV2::AvqiResult(_) => {
            Err("worker protocol V2 ASR response returned AVQI feature data".into())
        }
    }
}

/// The provider's own label for a monologue's speaker, when it named one.
///
/// `None` for an undiarized engine, which is what [`AsrToken::speaker`] has
/// always meant by absence. The wire used to carry the string `"0"` there,
/// because it had no way to say "this engine separates nobody".
fn speaker_label(attribution: &SpeakerAttributionV2) -> Option<String> {
    match attribution {
        SpeakerAttributionV2::Attributed { label } => Some(label.as_str().to_owned()),
        SpeakerAttributionV2::Undiarized => None,
    }
}

/// The track a monologue occupies in the transcript.
///
/// Total, and both arms are deliberate:
///
/// - **Attributed**: the provider's label is read as its own speaker number.
///   Every provider that separates speakers numbers them (Tencent's
///   `SpeakerId`, Rev's `speaker`), so this preserves the numbering a
///   transcript has always had. A label that is NOT a number is a provider
///   contract change rather than something to guess at, and it lands on track
///   zero only after saying so; it cannot be silently folded into another
///   speaker's tier, which is what `rsplit('_')` did when it turned
///   `A_0` and `B_0` into the same track.
/// - **Undiarized**: the single track of a recording nobody separated. This is
///   the ONE place that number is chosen, and it is chosen because the engine
///   said it separates nobody, not because a label was missing.
fn speaker_index(attribution: &SpeakerAttributionV2) -> SpeakerIndex {
    match attribution {
        SpeakerAttributionV2::Attributed { label } => {
            SpeakerIndex(label.as_str().parse::<usize>().unwrap_or_else(|_| {
                warn!(
                    speaker = label.as_str(),
                    "provider speaker label is not a speaker number; this transcript's \
                     tiers cannot be numbered from it, so it takes the first track"
                );
                0
            }))
        }
        SpeakerAttributionV2::Undiarized => SpeakerIndex(0),
    }
}

/// Resolve a worker-provided language against the control-plane fallback.
///
/// Lower a raw Whisper chunk result into the established `AsrResponse` domain.
///
/// Shared by the Python-worker V2 path (`parse_asr_response_v2`) and the
/// Rust-native whisper.cpp path (`transcribe::infer::infer_whisper_rs_asr`) so
/// both engines emit an identical `AsrResponse` from identical chunks. Each
/// non-empty chunk becomes one timed `AsrToken`; there is no speaker or
/// confidence at the chunk granularity.
pub(crate) fn whisper_chunk_result_to_asr_response(
    result: &WhisperChunkResultV2,
    fallback_lang: Option<&LanguageCode3>,
) -> Result<AsrResponse, String> {
    Ok(AsrResponse {
        lang: resolve_worker_lang(&result.lang, fallback_lang)?,
        // Shared by the Python worker path and the in-process whisper.cpp path,
        // so both engines report their weights the same way.
        model: Some(result.model.clone()),
        tokens: result
            .chunks
            .iter()
            .filter_map(|chunk| {
                let text = chunk.text.trim();
                if text.is_empty() {
                    return None;
                }

                Some(AsrToken {
                    text: text.to_string(),
                    start_s: Some(chunk.start_s),
                    end_s: Some(chunk.end_s),
                    speaker: None,
                    confidence: None,
                })
            })
            .collect(),
        source_monologues: None,
    })
}

/// When the worker's response carries an empty language string, fall
/// back to `fallback_lang` if the caller supplied one. If the caller
/// supplied `None` (an `Auto` job with no resolved language yet) and
/// the worker also returned nothing, surface a typed error so the
/// caller writes nothing to the output's `@Languages:` rather than
/// silently stamping English.
fn resolve_worker_lang(
    worker_lang: &LanguageCode3,
    fallback_lang: Option<&LanguageCode3>,
) -> Result<LanguageCode3, String> {
    if worker_lang.trim().is_empty() {
        fallback_lang.cloned().ok_or_else(|| {
            "ASR worker returned an empty language and the job has no resolved \
             fallback (`--lang auto` with no Rev.AI/Whisper-detected language). \
             Re-run with an explicit `--lang <iso3>` instead of `--lang auto`."
                .to_string()
        })
    } else {
        Ok(worker_lang.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{DurationSeconds, LanguageCode3};
    use crate::types::worker_v2::{
        AsrElementKindV2, AsrElementV2, AsrModelIdentityV2, AsrMonologueV2, ExecuteResponseV2,
        HubCommitV2, LoadedModelV2, ModelIdV2, MonologueAsrResultV2, ObservedRevisionV2,
        RequestedModelV2, RequestedRevisionV2, TaskResultV2, WhisperChunkResultV2,
        WhisperChunkSpanV2, WorkerRequestIdV2,
    };

    /// The commit the plan pins in these fixtures.
    const PINNED: &str = "06f233fe06e710322aca913c1bc4249a0d71fce1";
    /// Another valid commit, for the worker that loaded the wrong one.
    const OTHER: &str = "c07281df297b9905d24a508279258cccf987a064";

    fn commit(hex: &str) -> HubCommitV2 {
        HubCommitV2::try_from(hex).expect("a 40 hex character commit is valid")
    }

    fn whisper_id() -> ModelIdV2 {
        ModelIdV2::try_from("openai/whisper-large-v3").expect("a hub model id is stamp safe")
    }

    /// A loaded identity for tests whose subject is LOWERING, not which
    /// checkpoint ran. Pinned and observed to the same commit, so it is a value
    /// [`test_requested`] admits: these fixtures must agree, or every test here
    /// would be measuring the refusal instead of the lowering.
    fn test_identity() -> AsrModelIdentityV2 {
        AsrModelIdentityV2::Whisper {
            asr: LoadedModelV2 {
                id: whisper_id(),
                requested: RequestedRevisionV2::Commit {
                    commit: commit(PINNED),
                },
                observed: ObservedRevisionV2::Commit {
                    commit: commit(PINNED),
                },
            },
        }
    }

    /// The pin [`test_identity`] answers: the same model at the same commit.
    fn test_requested() -> AsrRequestedModelsV2 {
        AsrRequestedModelsV2::Whisper {
            asr: RequestedModelV2 {
                id: whisper_id(),
                revision: RequestedRevisionV2::Commit {
                    commit: commit(PINNED),
                },
            },
        }
    }

    #[test]
    fn parses_whisper_chunk_result_into_established_asr_domain() {
        let response = ExecuteResponseV2::success(
            WorkerRequestIdV2::from("req-asr-v2-1"),
            TaskResultV2::WhisperChunkResult(WhisperChunkResultV2 {
                lang: LanguageCode3::eng(),
                text: "hello world".into(),
                chunks: vec![
                    WhisperChunkSpanV2 {
                        text: "hello".into(),
                        start_s: DurationSeconds(0.0),
                        end_s: DurationSeconds(0.5),
                    },
                    WhisperChunkSpanV2 {
                        text: "world".into(),
                        start_s: DurationSeconds(0.5),
                        end_s: DurationSeconds(1.0),
                    },
                ],
                model: test_identity(),
            }),
            DurationSeconds(0.01),
        );

        let parsed = parse_asr_response_v2(
            &response,
            Some(&LanguageCode3::eng()),
            &test_requested(),
            AsrBackendV2::LocalWhisper,
        )
        .expect("V2 ASR response should parse");

        assert_eq!(parsed.lang, "eng");
        assert_eq!(parsed.tokens.len(), 2);
        assert_eq!(parsed.tokens[0].text, "hello");
        assert_eq!(parsed.tokens[1].end_s, Some(DurationSeconds(1.0)));
    }

    #[test]
    fn parses_monologue_result_into_established_asr_domain() {
        let response = ExecuteResponseV2::success(
            WorkerRequestIdV2::from("req-asr-v2-provider"),
            TaskResultV2::MonologueAsrResult(MonologueAsrResultV2 {
                lang: LanguageCode3::yue(),
                monologues: vec![AsrMonologueV2 {
                    speaker: SpeakerAttributionV2::Attributed {
                        label: "1".try_into().expect("a provider label names somebody"),
                    },
                    elements: vec![
                        AsrElementV2 {
                            value: "nei5".into(),
                            start_s: Some(DurationSeconds(0.1)),
                            end_s: Some(DurationSeconds(0.4)),
                            kind: AsrElementKindV2::Text,
                            confidence: Some(0.9),
                        },
                        AsrElementV2 {
                            value: ",".into(),
                            start_s: None,
                            end_s: None,
                            kind: AsrElementKindV2::Punctuation,
                            confidence: None,
                        },
                        AsrElementV2 {
                            value: "hou2".into(),
                            start_s: Some(DurationSeconds(0.5)),
                            end_s: Some(DurationSeconds(0.8)),
                            kind: AsrElementKindV2::Text,
                            confidence: None,
                        },
                    ],
                }],
                model: test_identity(),
            }),
            DurationSeconds(0.01),
        );

        let parsed = parse_asr_response_v2(
            &response,
            Some(&LanguageCode3::eng()),
            &test_requested(),
            AsrBackendV2::LocalWhisper,
        )
        .expect("V2 monologue response should parse");

        assert_eq!(parsed.lang, "yue");
        assert_eq!(parsed.tokens.len(), 2);
        assert_eq!(parsed.tokens[0].speaker.as_deref(), Some("1"));
        assert_eq!(parsed.tokens[0].confidence, Some(0.9));
        assert_eq!(parsed.tokens[1].text, "hou2");
        let monologues = parsed.source_monologues.unwrap();
        assert_eq!(monologues[0].elements[1].ts, AsrTimestampSecs::Absent);
        assert_eq!(monologues[0].elements[1].end_ts, AsrTimestampSecs::Absent);
    }

    /// The server's own admission. A worker that loaded a commit the plan did
    /// not pin is refused HERE, whatever its own in-process check concluded,
    /// because this is the side that writes the identity into `asr_model=`.
    #[test]
    fn a_response_naming_another_commit_is_refused_by_the_server() {
        let response = ExecuteResponseV2::success(
            WorkerRequestIdV2::from("req-asr-v2-drift"),
            TaskResultV2::WhisperChunkResult(WhisperChunkResultV2 {
                lang: LanguageCode3::eng(),
                text: "hello".into(),
                chunks: vec![WhisperChunkSpanV2 {
                    text: "hello".into(),
                    start_s: DurationSeconds(0.0),
                    end_s: DurationSeconds(0.5),
                }],
                model: AsrModelIdentityV2::Whisper {
                    asr: LoadedModelV2 {
                        id: whisper_id(),
                        // The worker echoes the request it was sent, and
                        // reports loading something else: the shape a moved hub
                        // repository produces, and the one the stamp used to
                        // record as though it had been verified.
                        requested: RequestedRevisionV2::Commit {
                            commit: commit(PINNED),
                        },
                        observed: ObservedRevisionV2::Commit {
                            commit: commit(OTHER),
                        },
                    },
                },
            }),
            DurationSeconds(0.01),
        );

        let Err(refusal) = parse_asr_response_v2(
            &response,
            Some(&LanguageCode3::eng()),
            &test_requested(),
            AsrBackendV2::LocalWhisper,
        ) else {
            panic!("a commit the plan did not pin must be refused");
        };

        assert!(
            matches!(
                refusal,
                AsrResponseAdmissionError::Identity(AsrIdentityMismatchV2::Observation { .. })
            ),
            "the refusal must stay typed rather than becoming a message: {refusal}"
        );
        let rendered = refusal.to_string();
        assert!(rendered.contains(OTHER), "{rendered}");
        assert!(rendered.contains(PINNED), "{rendered}");
    }
}
