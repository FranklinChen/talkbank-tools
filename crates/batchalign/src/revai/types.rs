//! Rev.AI API request and response types.
//!
//! These records mirror the subset of the Rev.AI Speech-to-Text v1 API used by
//! batchalign3. They intentionally model the wire format closely so that the
//! shared client can be reused by both the server control plane and the PyO3
//! bridge without a second translation layer.

use serde::{Deserialize, Serialize};

use crate::api::AudioPositionSeconds;

/// Status of a Rev.AI transcription job.
///
/// Rev.AI jobs move in one direction:
/// `InProgress` -> `Transcribed` | `Failed`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    /// The upload was accepted and Rev.AI is still processing the audio.
    InProgress,
    /// The transcript is ready to download.
    Transcribed,
    /// The job failed permanently and must be resubmitted from scratch.
    Failed,
}

/// A Rev.AI job returned by submission or polling endpoints.
#[derive(Debug, Clone, Deserialize)]
pub struct Job {
    /// Rev.AI-assigned job identifier.
    pub id: String,
    /// Current lifecycle state for the job.
    pub status: JobStatus,
    /// Human-readable failure detail for terminal failures.
    #[serde(default)]
    pub failure_detail: Option<String>,
    /// Language detected by Rev.AI when `language: "auto"` was used.
    ///
    /// Rev.AI returns this field as an ISO 639-1 code (e.g. `"es"`, `"en"`)
    /// once a job completes with auto-detection enabled.
    #[serde(default)]
    pub language: Option<String>,
}

/// A single word or punctuation element inside one Rev.AI monologue.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Element {
    /// Rev.AI element type, typically `"text"` or `"punct"`.
    #[serde(rename = "type")]
    pub element_type: String,
    /// Element text as returned by Rev.AI.
    pub value: String,
    /// Start of a timed token, as a position in the audio. Admitted when the
    /// provider's JSON is read, so a negative or non-finite time refuses the
    /// transcript there rather than at each consumer.
    #[serde(default)]
    pub ts: Option<AudioPositionSeconds>,
    /// End of a timed token, as a position in the audio; admitted as `ts` is.
    #[serde(default)]
    pub end_ts: Option<AudioPositionSeconds>,
    /// Optional confidence score emitted for text elements.
    #[serde(default)]
    pub confidence: Option<f64>,
}

/// A contiguous speaker turn in a Rev.AI transcript.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Monologue {
    /// Zero-based speaker index assigned by Rev.AI.
    pub speaker: i32,
    /// Sequence of timed words and punctuation for the turn.
    pub elements: Vec<Element>,
}

/// Full Rev.AI transcript payload returned by the transcript endpoint.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Transcript {
    /// Ordered monologues produced by Rev.AI.
    pub monologues: Vec<Monologue>,
}

/// Fidelity of the provider transcript retained as reusable evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RevTranscriptFidelity {
    /// Exact JSON response bytes returned by the transcript endpoint.
    ExactProviderJson,
    /// Typed projection migrated from the earlier cache schema.
    LegacyTypedProjection,
}

/// A parsed Rev transcript inseparably paired with the evidence retained for
/// future projections.
///
/// The private representation prevents a caller from pairing raw JSON from one
/// response with a parsed transcript from another. Provider construction parses
/// once at the HTTP boundary. Cache deserialization repeats that validation
/// before the value can re-enter the pipeline.
#[derive(Debug, Clone)]
pub(crate) struct RevTranscriptEvidence {
    retained: RetainedRevTranscript,
    transcript: Transcript,
}

#[derive(Debug, Clone)]
enum RetainedRevTranscript {
    ExactProviderJson(String),
    LegacyTypedProjection,
}

impl RevTranscriptEvidence {
    pub(crate) fn from_provider_json(raw_json: String) -> Result<Self, serde_json::Error> {
        let transcript = serde_json::from_str(&raw_json)?;
        Ok(Self {
            retained: RetainedRevTranscript::ExactProviderJson(raw_json),
            transcript,
        })
    }

    pub(crate) fn from_legacy_transcript(transcript: Transcript) -> Self {
        Self {
            retained: RetainedRevTranscript::LegacyTypedProjection,
            transcript,
        }
    }

    pub(crate) fn transcript(&self) -> &Transcript {
        &self.transcript
    }

    pub(crate) fn fidelity(&self) -> RevTranscriptFidelity {
        match self.retained {
            RetainedRevTranscript::ExactProviderJson(_) => RevTranscriptFidelity::ExactProviderJson,
            RetainedRevTranscript::LegacyTypedProjection => {
                RevTranscriptFidelity::LegacyTypedProjection
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn exact_provider_json(&self) -> Option<&str> {
        match &self.retained {
            RetainedRevTranscript::ExactProviderJson(raw_json) => Some(raw_json),
            RetainedRevTranscript::LegacyTypedProjection => None,
        }
    }
}

#[derive(Serialize)]
#[serde(tag = "fidelity", rename_all = "snake_case")]
enum RevTranscriptEvidenceRef<'a> {
    ExactProviderJson { raw_json: &'a str },
    LegacyTypedProjection { transcript: &'a Transcript },
}

impl Serialize for RevTranscriptEvidence {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match &self.retained {
            RetainedRevTranscript::ExactProviderJson(raw_json) => {
                RevTranscriptEvidenceRef::ExactProviderJson { raw_json }.serialize(serializer)
            }
            RetainedRevTranscript::LegacyTypedProjection => {
                RevTranscriptEvidenceRef::LegacyTypedProjection {
                    transcript: &self.transcript,
                }
                .serialize(serializer)
            }
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "fidelity", rename_all = "snake_case")]
enum RevTranscriptEvidenceWire {
    ExactProviderJson { raw_json: String },
    LegacyTypedProjection { transcript: Transcript },
}

impl<'de> Deserialize<'de> for RevTranscriptEvidence {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        match RevTranscriptEvidenceWire::deserialize(deserializer)? {
            RevTranscriptEvidenceWire::ExactProviderJson { raw_json } => {
                Self::from_provider_json(raw_json).map_err(serde::de::Error::custom)
            }
            RevTranscriptEvidenceWire::LegacyTypedProjection { transcript } => {
                Ok(Self::from_legacy_transcript(transcript))
            }
        }
    }
}

#[cfg(test)]
mod transcript_evidence_tests {
    use super::*;

    #[test]
    fn exact_provider_json_retains_unknown_fields_across_cache_serialization() {
        let raw_json = r#"{"monologues":[],"provider_extension":{"future":true}}"#.to_owned();
        let evidence = RevTranscriptEvidence::from_provider_json(raw_json.clone())
            .expect("provider response should parse");

        let stored = serde_json::to_value(&evidence).expect("cache serialization");
        let replayed: RevTranscriptEvidence =
            serde_json::from_value(stored).expect("cache deserialization");

        assert_eq!(replayed.exact_provider_json(), Some(raw_json.as_str()));
        assert_eq!(replayed.transcript().monologues.len(), 0);
    }
}

/// Submission options for `POST /speechtotext/v1/jobs`.
#[derive(Debug, Clone, Serialize)]
pub struct SubmitOptions {
    /// ISO 639-1 language code expected by Rev.AI.
    pub language: String,
    /// Optional speaker-count hint for languages where Rev.AI supports it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speakers_count: Option<u32>,
    /// Optional toggle for Rev.AI post-processing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skip_postprocessing: Option<bool>,
    /// Optional metadata string attached to the submitted job.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<String>,
}

// ---------------------------------------------------------------------------
// Language Identification API types
// ---------------------------------------------------------------------------

/// Status of a Rev.AI language identification job.
///
/// Language ID jobs share the same lifecycle as transcription jobs:
/// `InProgress` -> `Completed` | `Failed`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LangIdJobStatus {
    /// The job is still being processed.
    InProgress,
    /// The result is ready to download.
    Completed,
    /// The job failed permanently.
    Failed,
}

/// A Rev.AI language identification job returned by submission or polling.
#[derive(Debug, Clone, Deserialize)]
pub struct LangIdJob {
    /// Rev.AI-assigned job identifier.
    pub id: String,
    /// Current lifecycle state.
    pub status: LangIdJobStatus,
    /// Human-readable failure detail for terminal failures.
    #[serde(default)]
    pub failure_detail: Option<String>,
}

/// One language with its confidence score from the Language ID result.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LanguageConfidence {
    /// ISO 639-1 language code (e.g., `"en"`, `"es"`).
    pub language: String,
    /// Confidence score (0.0 -1.0).
    pub confidence: f64,
}

/// Result from `GET /languageid/v1/jobs/{id}/result`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LangIdResult {
    /// The most probable language (ISO 639-1 code, e.g., `"es"`).
    pub top_language: String,
    /// All detected languages ranked by confidence.
    pub language_confidences: Vec<LanguageConfidence>,
}

// ---------------------------------------------------------------------------
// Speech-to-Text supplementary types
// ---------------------------------------------------------------------------

/// Simplified timed-word projection used by the UTR path.
#[derive(Debug, Clone, Serialize)]
pub struct TimedWord {
    /// Trimmed token text. Empty/whitespace tokens are filtered out upstream.
    pub word: String,
    /// Absolute start time in milliseconds.
    pub start_ms: u64,
    /// Absolute end time in milliseconds.
    pub end_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The provider boundary is where a Rev.AI time becomes a position, so a
    /// transcript naming a negative one is refused as it is read rather than
    /// by each consumer afterwards.
    #[test]
    fn a_transcript_with_a_negative_element_time_is_refused_when_read() {
        let negative = r#"{"monologues":[{"speaker":0,"elements":[
            {"type":"text","value":"hello","ts":-0.1,"end_ts":0.4}]}]}"#;
        assert!(serde_json::from_str::<Transcript>(negative).is_err());
        let valid = r#"{"monologues":[{"speaker":0,"elements":[
            {"type":"text","value":"hello","ts":0.1,"end_ts":0.4}]}]}"#;
        let transcript: Transcript = serde_json::from_str(valid).expect("a valid transcript");
        assert_eq!(
            transcript.monologues[0].elements[0]
                .ts
                .map(AudioPositionSeconds::get),
            Some(0.1)
        );
    }
}
