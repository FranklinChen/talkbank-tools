//! Pure projection from durable Rev.AI evidence into the UTR cache shape.
//!
//! Provider I/O is deliberately absent from this module. The caller must first
//! obtain [`CompletedRevAsrEvidence`] through the typed evidence resolver, so a
//! projection change can be replayed without another paid request.

use crate::api::DurationSeconds;
use crate::transcribe::{AsrResponse, AsrToken};

use super::{CompletedRevAsrEvidence, extract_timed_words};

/// Project provider evidence into the normalized response consumed by UTR.
pub(crate) fn rev_evidence_to_utr_asr_response(evidence: &CompletedRevAsrEvidence) -> AsrResponse {
    AsrResponse {
        tokens: extract_timed_words(evidence.transcript_evidence.transcript())
            .into_iter()
            .map(|word| AsrToken {
                text: word.word,
                start_s: Some(DurationSeconds(word.start_ms as f64 / 1000.0)),
                end_s: Some(DurationSeconds(word.end_ms as f64 / 1000.0)),
                speaker: None,
                confidence: None,
            })
            .collect(),
        lang: evidence.resolved_language.clone(),
        source_monologues: None,
        model: Some(crate::model_manifest::rev_loaded_identity()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::LanguageCode3;

    #[test]
    fn transcript_projection_discards_blank_rev_tokens() {
        let transcript: crate::revai::Transcript = serde_json::from_str(
            r#"{
                "monologues": [{
                    "speaker": 0,
                    "elements": [
                        {"type": "text", "value": "hello", "ts": 0.1, "end_ts": 0.4},
                        {"type": "text", "value": "   ", "ts": 0.5, "end_ts": 0.8},
                        {"type": "text", "value": "world", "ts": 0.9, "end_ts": 1.2}
                    ]
                }]
            }"#,
        )
        .unwrap();

        let response = rev_evidence_to_utr_asr_response(&CompletedRevAsrEvidence {
            transcript_evidence: crate::revai::RevTranscriptEvidence::from_legacy_transcript(
                transcript,
            ),
            resolved_language: LanguageCode3::eng(),
        });
        assert_eq!(response.tokens.len(), 2);
        assert_eq!(response.tokens[0].text, "hello");
        assert_eq!(response.tokens[0].start_s, Some(DurationSeconds(0.1)));
        assert_eq!(response.tokens[1].text, "world");
        assert_eq!(response.tokens[1].end_s, Some(DurationSeconds(1.2)));
    }
}
