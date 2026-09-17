//! Rust-owned Rev.AI ASR inference for server-mode transcription and UTR.
//!
//! This path exists so `transcribe` and `benchmark` do not need to route the
//! Rev.AI provider through the Python worker at all. The only engines that
//! should stay in Python are the ones that genuinely require Python-hosted
//! model libraries.

use crate::revai::FetchedRevAsrEvidence;

use crate::revai::{RevAiClient, SubmitOptions, Transcript, TranscriptResult};
use batchalign_transform::asr_postprocess::{
    AsrElement, AsrElementKind, AsrMonologue, AsrOutput, AsrRawText, AsrTimestampSecs, SpeakerIndex,
};
use tracing::info;

use crate::api::{DurationSeconds, LanguageCode3, NumSpeakers};
use crate::error::ServerError;
use crate::transcribe::{AsrResponse, AsrToken};

use super::{
    AuthorizedRevEvidenceRun, CompletedRevAsrEvidence, RejectedRevLanguageEvidence,
    RevAsrEvidenceInference, RevAsrInferenceOutcome, VerifiedRevProviderMedia, load_revai_api_key,
};
use crate::types::revai_language::{RevLanguage, RevOptionSupport};

/// Run Rev.AI ASR directly from Rust and map the transcript into the shared
/// `AsrResponse` domain used by the transcribe pipeline.
///
/// For detection, runs Rev.AI Language Identification first and submits the
/// language it names; when that fails or names nothing we map, submits
/// `"auto"` and reads the detected language from the completed job.
async fn infer_revai_evidence(
    run: AuthorizedRevEvidenceRun,
) -> Result<RevAsrInferenceOutcome, ServerError> {
    let api_key =
        load_revai_api_key().map_err(|error| ServerError::Validation(error.to_string()))?;
    tokio::task::spawn_blocking(move || {
        let media = run
            .provider_media
            .verify()
            .map_err(|error| ServerError::Persistence(error.to_string()))?;
        let lang = run.requested_language;
        let num_speakers = run.expected_speakers;
        // When auto-detecting, run Rev.AI Language Identification first.
        // This is a separate API (~5-30s) that identifies the spoken language
        // from audio features: far more accurate than text-based trigram
        // detection, especially for code-switched bilingual audio.
        let effective_lang = match lang.request() {
            crate::api::AsrLanguageRequest::One(_) | crate::api::AsrLanguageRequest::Pair(_) => {
                lang.clone()
            }
            crate::api::AsrLanguageRequest::Detect => {
                let client = RevAiClient::new(api_key.as_str());
                match client.identify_language_bytes_blocking(
                    &media.bytes,
                    &media.upload_file_name,
                    media.upload_mime,
                    30,
                ) {
                    Ok(langid_result) => {
                        let detected = &langid_result.top_language;
                        info!(
                            detected_language = %detected,
                            confidence = langid_result.language_confidences.first()
                                .map(|c| c.confidence).unwrap_or(0.0),
                            "Rev.AI Language ID detected language"
                        );
                        match RevLanguage::identified(detected) {
                            Some(identified) => identified,
                            None => {
                                tracing::warn!(
                                    detected = %detected,
                                    "Rev.AI Language ID named a language BA3 does not map; \
                                     submitting for transcription-level detection"
                                );
                                lang.clone()
                            }
                        }
                    }
                    Err(error) => {
                        tracing::warn!(
                            %error,
                            "Rev.AI Language ID failed; falling back to transcription-level auto"
                        );
                        lang.clone()
                    }
                }
            }
        };

        // NOT `Validation(error.to_string())`. A provider failure is neither a
        // malformed request nor a string: the typed conversion keeps the
        // retryable/terminal verdict the control plane needs.
        let result = fetch_revai_transcript(&api_key, &media, &effective_lang, num_speakers)
            .map_err(ServerError::from)?;
        let (transcript_evidence, detected_language) = result.into_parts();

        Ok(classify_language_response(
            transcript_evidence,
            lang,
            effective_lang,
            detected_language,
        ))
    })
    .await
    .map_err(|error| ServerError::Validation(format!("Rev.AI task join error: {error}")))?
}

/// Language rejection must return the fetched bytes to the durable boundary.
/// This does not infer a language from transcript contents or default English.
///
/// One language or a pair is what the transcript is in, as requested.
/// Detection resolves only to a language Rev.AI itself reported.
pub(super) fn classify_language_response(
    transcript_evidence: super::types::RevTranscriptEvidence,
    requested_language: RevLanguage,
    effective_language: RevLanguage,
    detected_language: Option<String>,
) -> RevAsrInferenceOutcome {
    match effective_language.resolve(detected_language.as_deref()) {
        Some(resolved_language) => RevAsrInferenceOutcome::Fetched(FetchedRevAsrEvidence {
            transcript_evidence,
            resolved_language,
        }),
        None => RevAsrInferenceOutcome::UnresolvedLanguage(RejectedRevLanguageEvidence {
            transcript_evidence,
            requested_language,
            effective_language,
            detected_language,
        }),
    }
}

/// Production Rev.AI boundary carrying all inputs authorized by an evidence
/// cache miss or explicit refresh.
pub(crate) struct RevAsrService;

impl RevAsrService {
    pub(crate) fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl RevAsrEvidenceInference for RevAsrService {
    async fn infer(
        &self,
        run: AuthorizedRevEvidenceRun,
    ) -> Result<RevAsrInferenceOutcome, ServerError> {
        infer_revai_evidence(run).await
    }
}

/// Project admitted evidence into the shared single-language response.
///
/// `AsrResponse.lang` is one code, so a code-switched transcript reports its
/// primary language there. Transcribe reads a pair's languages from the
/// request it admitted, never from this field.
pub(crate) fn rev_evidence_to_asr_response(evidence: &CompletedRevAsrEvidence) -> AsrResponse {
    transcript_to_asr_response(
        evidence.transcript_evidence().transcript(),
        evidence.resolved_language().primary(),
    )
}

/// Submit the verified provider-media artifact and wait for its transcript.
///
/// Under detection, `speakers_count` and `skip_postprocessing` are not sent,
/// because the language's characteristics are not known ahead of time.
///
/// Which languages take `speakers_count` and `skip_postprocessing` is
/// [`RevLanguage::options`], from the option column of the Rev.AI language
/// table: English and Spanish take both. `speakers_count` lets Rev.AI's own
/// diarization use the expected count. `skip_postprocessing` turns off Rev.AI's
/// Inverse Text Normalization (ITN), which converts spoken form (what the
/// speaker said) into written form (`"eighty percent"` → `"80%"`,
/// `"seventeen year old"` → `"17-year-old"`); CHAT records spoken form.
///
/// Rev.AI's multilingual English/Spanish model takes neither: the live API
/// refused each of them for `en/es` with HTTP 400 on 2026-09-16. Its output
/// therefore keeps Rev.AI's written forms, and post-processing keeps a
/// code-switched transcript's numerals as digits.
///
/// Production can reach this paid boundary only after the durable evidence
/// cache authorizes a miss or an explicit refresh. The verified artifact keeps
/// the bytes hashed for that decision identical to the bytes uploaded here.
pub(super) fn fetch_revai_transcript(
    api_key: &super::RevAiApiKey,
    media: &VerifiedRevProviderMedia,
    lang: &RevLanguage,
    num_speakers: Option<NumSpeakers>,
) -> crate::revai::Result<TranscriptResult> {
    let client = RevAiClient::new(api_key.as_str());
    let options = rev_submit_options(lang, num_speakers, &media.metadata);
    client.transcribe_bytes_blocking(
        &media.bytes,
        &media.upload_file_name,
        media.upload_mime,
        &options,
        30,
    )
}

fn rev_submit_options(
    lang: &RevLanguage,
    num_speakers: Option<NumSpeakers>,
    metadata: &str,
) -> SubmitOptions {
    let (speakers_count, skip_postprocessing) = match lang.options() {
        RevOptionSupport::SpeakerCountAndSpokenForm => {
            (num_speakers.map(|count| count.0), Some(true))
        }
        RevOptionSupport::Neither => (None, None),
    };

    SubmitOptions {
        language: lang.provider_code().to_owned(),
        speakers_count,
        skip_postprocessing,
        metadata: Some(metadata.to_owned()),
    }
}

#[test]
fn auto_speakers_omits_provider_count_but_preserves_postprocessing() {
    let lang = RevLanguage::admit(&crate::api::AsrLanguageRequest::One(LanguageCode3::eng()))
        .expect("Rev.AI recognizes English");
    let automatic =
        serde_json::to_value(rev_submit_options(&lang, None, "fixture")).expect("provider JSON");
    assert!(automatic.get("speakers_count").is_none());
    assert_eq!(automatic["skip_postprocessing"], true);
    let exact = serde_json::to_value(rev_submit_options(&lang, Some(NumSpeakers(3)), "fixture"))
        .expect("provider JSON");
    assert_eq!(exact["speakers_count"], 3);
}

/// An English/Spanish pair, in either order, is submitted to Rev.AI's
/// multilingual model with neither the speaker count nor the spoken-form
/// switch: the live API refuses both for `en/es` (HTTP 400), so sending them
/// would fail every code-switched job.
#[test]
fn an_english_spanish_pair_submits_the_multilingual_model_without_refused_options() {
    for (primary, secondary) in [
        (LanguageCode3::eng(), LanguageCode3::spa()),
        (LanguageCode3::spa(), LanguageCode3::eng()),
    ] {
        let pair = crate::api::LanguagePair::new(primary, secondary).expect("two languages");
        let lang = RevLanguage::admit(&crate::api::AsrLanguageRequest::Pair(pair))
            .expect("Rev.AI has an English/Spanish model");
        let options =
            serde_json::to_value(rev_submit_options(&lang, Some(NumSpeakers(2)), "fixture"))
                .expect("provider JSON");
        assert_eq!(options["language"], "en/es");
        assert!(options.get("skip_postprocessing").is_none(), "{options}");
        assert!(options.get("speakers_count").is_none(), "{options}");
    }
}

fn transcript_to_asr_response(transcript: &Transcript, lang: &LanguageCode3) -> AsrResponse {
    let mut tokens = Vec::new();

    for monologue in &transcript.monologues {
        let speaker = monologue.speaker.to_string();
        for element in &monologue.elements {
            if element.element_type != "text" {
                continue;
            }

            let text = element.value.trim();
            if text.is_empty() {
                continue;
            }

            tokens.push(AsrToken {
                text: text.to_string(),
                start_s: element.ts.map(DurationSeconds),
                end_s: element.end_ts.map(DurationSeconds),
                speaker: Some(speaker.clone()),
                confidence: element.confidence,
            });
        }
    }

    AsrResponse {
        tokens,
        lang: lang.clone(),
        source_monologues: Some(transcript_to_asr_output(transcript).monologues),
        model: Some(crate::model_manifest::rev_loaded_identity()),
    }
}

fn transcript_to_asr_output(transcript: &Transcript) -> AsrOutput {
    AsrOutput {
        monologues: transcript
            .monologues
            .iter()
            .map(|monologue| AsrMonologue {
                speaker: SpeakerIndex(monologue.speaker as usize),
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
                            ts: AsrTimestampSecs::from(element.ts),
                            end_ts: AsrTimestampSecs::from(element.end_ts),
                            kind: if element.element_type == "text" {
                                AsrElementKind::Text
                            } else {
                                AsrElementKind::Punctuation
                            },
                        })
                    })
                    .collect(),
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::LanguageCode3;
    use crate::revai::Transcript;

    #[test]
    fn transcript_projection_keeps_flat_text_tokens_for_legacy_paths() {
        let transcript: Transcript = serde_json::from_str(
            r#"{
            "monologues": [{
                "speaker": 3,
                "elements": [
                    {"type": "text", "value": "hello", "ts": 0.5, "end_ts": 0.9, "confidence": 0.75},
                    {"type": "punct", "value": ","},
                    {"type": "text", "value": "world", "ts": 1.0, "end_ts": 1.4}
                ]
            }]
        }"#,
        )
        .unwrap();

        let response = transcript_to_asr_response(&transcript, &LanguageCode3::eng());
        assert_eq!(response.lang, "eng");
        assert_eq!(response.tokens.len(), 2);
        assert_eq!(response.tokens[0].text, "hello");
        assert_eq!(response.tokens[0].speaker.as_deref(), Some("3"));
        assert_eq!(response.tokens[0].confidence, Some(0.75));
        assert_eq!(response.tokens[1].text, "world");
    }

    #[test]
    fn transcript_projection_preserves_punctuation_and_monologue_boundaries() {
        let transcript: Transcript = serde_json::from_str(
            r#"{
            "monologues": [
                {
                    "speaker": 3,
                    "elements": [
                        {"type": "text", "value": "hello", "ts": 0.5, "end_ts": 0.9, "confidence": 0.75},
                        {"type": "punct", "value": ","},
                        {"type": "text", "value": "world", "ts": 1.0, "end_ts": 1.4}
                    ]
                },
                {
                    "speaker": 3,
                    "elements": [
                        {"type": "text", "value": "again", "ts": 2.0, "end_ts": 2.4},
                        {"type": "punct", "value": "?"}
                    ]
                }
            ]
        }"#,
        )
        .unwrap();

        let response = transcript_to_asr_response(&transcript, &LanguageCode3::eng());
        let monologues = response
            .source_monologues
            .expect("Rev projection should preserve provider-shaped monologues");

        assert_eq!(monologues.len(), 2);
        assert_eq!(monologues[0].speaker, SpeakerIndex(3));
        assert_eq!(monologues[0].elements.len(), 3);
        assert_eq!(monologues[0].elements[1].value, ",");
        assert_eq!(monologues[0].elements[1].kind, AsrElementKind::Punctuation);
        assert_eq!(monologues[0].elements[1].ts, AsrTimestampSecs::Absent);
        assert_eq!(monologues[0].elements[1].end_ts, AsrTimestampSecs::Absent);
        assert_eq!(monologues[1].speaker, SpeakerIndex(3));
        assert_eq!(monologues[1].elements.len(), 2);
        assert_eq!(monologues[1].elements[1].value, "?");
        assert_eq!(monologues[1].elements[1].kind, AsrElementKind::Punctuation);
    }

    /// A pair request resolves to the pair as requested, whatever language
    /// Rev.AI reports for the job; detection resolves only to what Rev.AI
    /// reports.
    #[test]
    fn a_pair_request_resolves_to_the_pair() {
        use crate::api::{AsrLanguageRequest, LanguagePair};
        let evidence = || {
            super::super::RevTranscriptEvidence::from_provider_json(
                r#"{"monologues":[]}"#.to_owned(),
            )
            .expect("provider JSON")
        };
        let pair = RevLanguage::admit(&AsrLanguageRequest::Pair(
            LanguagePair::new(LanguageCode3::spa(), LanguageCode3::eng()).expect("two languages"),
        ))
        .expect("Rev.AI takes English/Spanish");
        match classify_language_response(evidence(), pair.clone(), pair, Some("en/es".into())) {
            RevAsrInferenceOutcome::Fetched(completed) => assert_eq!(
                completed.resolved_language.to_string(),
                "spa,eng",
                "the requested order is the transcript's order"
            ),
            RevAsrInferenceOutcome::UnresolvedLanguage(_) => panic!("a pair is always resolved"),
        }
        match classify_language_response(
            evidence(),
            RevLanguage::detect(),
            RevLanguage::detect(),
            Some("es".into()),
        ) {
            RevAsrInferenceOutcome::Fetched(completed) => {
                assert_eq!(completed.resolved_language.to_string(), "spa");
            }
            RevAsrInferenceOutcome::UnresolvedLanguage(_) => panic!("Rev.AI reported Spanish"),
        }
    }

    #[test]
    fn skip_postprocessing_is_enabled_only_where_rev_supports_it() {
        use crate::api::AsrLanguageRequest;
        let skip = |request: AsrLanguageRequest| {
            let lang = RevLanguage::admit(&request).expect("Rev.AI takes the request");
            rev_submit_options(&lang, None, "fixture").skip_postprocessing
        };
        assert_eq!(
            skip(AsrLanguageRequest::One(LanguageCode3::eng())),
            Some(true)
        );
        assert_eq!(
            skip(AsrLanguageRequest::One(LanguageCode3::spa())),
            Some(true)
        );
        assert_eq!(skip(AsrLanguageRequest::One(LanguageCode3::zho())), None);
        assert_eq!(skip(AsrLanguageRequest::Detect), None);
    }
}
