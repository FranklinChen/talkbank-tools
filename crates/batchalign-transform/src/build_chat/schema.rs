use serde::Deserialize;
use talkbank_model::model::MediaStatus;

use crate::asr_postprocess;

/// Structured description of a transcript to be assembled into CHAT format.
///
/// Fields mirror the JSON format accepted by the PyO3 `build_chat()` function.
#[derive(Debug, Clone, Deserialize)]
pub struct TranscriptDescription {
    /// ISO 639-3 language codes (e.g. `["eng"]`). Defaults to `["eng"]` if empty.
    #[serde(default)]
    pub langs: Vec<String>,
    /// Participant entries. At least one is required.
    pub participants: Vec<ParticipantDesc>,
    /// Optional media filename (e.g. `"recording.mp3"`).
    pub media_name: Option<String>,
    /// Optional media type (`"audio"` or `"video"`). Defaults to `"audio"`.
    pub media_type: Option<String>,
    /// Optional relationship between the transcript and its media.
    ///
    /// In particular, `Unlinked` records that the media exists while the
    /// transcript has no timing evidence. Omitting the status asserts that
    /// the transcript is linked to the media.
    #[serde(default)]
    pub media_status: Option<MediaStatus>,
    /// Utterances to include in the transcript.
    #[serde(default)]
    pub utterances: Vec<UtteranceDesc>,
    /// Whether to generate `%wor` tiers when word-level timing is available.
    ///
    /// Defaults to `false` (BA2 parity: transcribe omits `%wor` unless
    /// explicitly requested via `--wor`). The JSON bridge (PyO3) defaults to
    /// `false` via serde; callers that want `%wor` must set this to `true`.
    #[serde(default)]
    pub write_wor: bool,
}

/// A participant in the transcript.
#[derive(Debug, Clone, Deserialize)]
pub struct ParticipantDesc {
    /// Speaker code (e.g. `"PAR"`, `"INV"`, `"CHI"`).
    pub id: String,
    /// Participant name for `@Participants` header. `None` omits the name
    /// field (output: `CODE Role`). `Some("...")` adds it (output: `CODE Name Role`).
    pub name: Option<String>,
    /// Participant role (e.g. `"Participant"`, `"Investigator"`, `"Target_Child"`).
    /// Callers should always set this, derive from speaker code via
    /// `role_for_speaker_code` if unknown. Defaults to `"Participant"` only
    /// for JSON backward compatibility.
    #[serde(default = "default_participant_role")]
    pub role: String,
    /// Corpus name for `@ID` header. Empty string if unknown.
    #[serde(default)]
    pub corpus: String,
}

/// An utterance in the transcript.
///
/// Either `words` (word-level with individual timings) or `text` (parse as
/// a single CHAT utterance line) should be provided. If both are present,
/// `words` takes precedence (when non-empty).
#[derive(Debug, Clone, Deserialize)]
pub struct UtteranceDesc {
    /// Speaker code for this utterance.
    pub speaker: String,
    /// Word-level tokens with optional per-word timing.
    pub words: Option<Vec<WordDesc>>,
    /// Full utterance text (alternative to word-level). Parsed via tree-sitter.
    ///
    /// This is a public API surface for callers who want to pass pre-formatted
    /// CHAT text rather than individual word tokens. The text is wrapped in a
    /// mini CHAT document and parsed by `build_text_utterance()`. Currently
    /// unused by the ASR pipeline (which always provides `words`), but
    /// preserved for external JSON API consumers.
    pub text: Option<String>,
    /// Utterance-level timing (used with `text` mode), admitted where it is
    /// described. JSON keeps its flat `start_ms`/`end_ms` fields.
    #[serde(flatten)]
    pub timing: DescribedTiming,
    /// Detected language for this utterance (ISO 639-3). When set and different
    /// from the primary language (`langs[0]`), a `[- lang]` precode is prepended.
    #[serde(default)]
    pub lang: Option<String>,
}

/// A single word token with optional timing.
#[derive(Debug, Clone, Deserialize)]
pub struct WordDesc {
    /// Word text (ready for CHAT assembly via TreeSitterParser).
    pub text: asr_postprocess::ChatWordText,
    /// The word's timing, admitted where it is described. JSON keeps its flat
    /// `start_ms`/`end_ms` fields.
    #[serde(flatten)]
    pub timing: DescribedTiming,
    /// What role this word plays (regular, retrace, etc.).
    #[serde(default)]
    pub kind: asr_postprocess::WordKind,
}

/// A described word's or utterance's timing, as the builder may use it: a
/// positive interval, or none with a named cause.
///
/// The description types carry this proof instead of two optional numbers, so
/// a generated bullet can only be built from a positive interval
/// ([`asr_postprocess::PositiveInterval::bullet`]). A zero-width, inverted or
/// out-of-range pair is admitted as untimed with its cause, never as a
/// bullet; the word itself is kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(from = "DescribedMillis")]
pub enum DescribedTiming {
    /// The described bounds enclose a positive duration.
    Positive(asr_postprocess::PositiveInterval),
    /// No usable timing, and why.
    Untimed(asr_postprocess::UntimedCause),
}

impl DescribedTiming {
    /// Admit optional millisecond bounds, as the post-processing word type
    /// stores them. Inadmissible bounds are untimed with their cause.
    pub fn admit_millis(start_ms: Option<i64>, end_ms: Option<i64>) -> Self {
        match asr_postprocess::WordTiming::from_millis(start_ms, end_ms) {
            Ok(timing) => Self::of(timing),
            Err(_) => Self::Untimed(asr_postprocess::UntimedCause::RefusedByAdmission),
        }
    }

    /// The description of an admitted ASR word timing.
    pub fn of(timing: asr_postprocess::WordTiming) -> Self {
        match timing.positive() {
            Ok(interval) => Self::Positive(interval),
            Err(cause) => Self::Untimed(cause),
        }
    }

    /// The positive interval, when there is one.
    pub fn positive(self) -> Option<asr_postprocess::PositiveInterval> {
        match self {
            Self::Positive(interval) => Some(interval),
            Self::Untimed(_) => None,
        }
    }
}

/// The JSON shape of a described timing: flat optional millisecond bounds.
/// Exists only to be admitted into [`DescribedTiming`] on the way in.
#[derive(Deserialize)]
struct DescribedMillis {
    #[serde(default)]
    start_ms: Option<u64>,
    #[serde(default)]
    end_ms: Option<u64>,
}

impl From<DescribedMillis> for DescribedTiming {
    fn from(raw: DescribedMillis) -> Self {
        // A bound beyond i64 is beyond any admissible range as well.
        let signed = |bound: Option<u64>| bound.map(i64::try_from).transpose();
        match (signed(raw.start_ms), signed(raw.end_ms)) {
            (Ok(start_ms), Ok(end_ms)) => Self::admit_millis(start_ms, end_ms),
            (Err(_), _) | (_, Err(_)) => {
                Self::Untimed(asr_postprocess::UntimedCause::RefusedByAdmission)
            }
        }
    }
}

/// Derive the default participant role used by the JSON bridge.
pub(super) fn default_participant_role() -> String {
    "Participant".to_string()
}
