use talkbank_model::model::{LanguageCode, MediaStatus};

use crate::asr_postprocess;

use super::TranscriptBuildError;

/// Structured description of a transcript to be assembled into CHAT format.
///
/// Built in Rust (the ASR bridge, `NamedAsrUtterances::into_transcript`) and
/// read by [`super::build_chat`]. There is no deserializer: every field is a
/// proof built where its value is born (languages parsed, words parsed).
#[derive(Debug, Clone)]
pub struct TranscriptDescription {
    /// The transcript's languages, primary first: never empty.
    pub langs: TranscriptLanguages,
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
    pub media_status: Option<MediaStatus>,
    /// Utterances to include in the transcript.
    pub utterances: Vec<UtteranceDesc>,
    /// Whether to generate `%wor` tiers when word-level timing is available.
    ///
    /// BA2 parity: transcribe omits `%wor` unless explicitly requested via
    /// `--wor`.
    pub write_wor: bool,
}

/// A transcript's languages, primary first, each a parsed CHAT code, and
/// never none: parsed once, where the raw codes enter
/// ([`Self::parse`]), so nothing downstream defaults, re-parses or checks
/// for emptiness.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptLanguages {
    primary: LanguageCode,
    secondary: Vec<LanguageCode>,
}

impl TranscriptLanguages {
    /// Parse raw ISO 639-3 codes, primary first.
    ///
    /// # Errors
    ///
    /// No code at all ([`TranscriptBuildError::MissingPrimaryLanguage`]), or
    /// one that is not a CHAT language code
    /// ([`TranscriptBuildError::InvalidLanguageCode`]).
    pub fn parse(raw: &[String]) -> Result<Self, TranscriptBuildError> {
        let parse = |code: &String| {
            LanguageCode::new(code).map_err(|source| TranscriptBuildError::InvalidLanguageCode {
                lang: code.clone(),
                source,
            })
        };
        let (primary, secondary) = raw
            .split_first()
            .ok_or(TranscriptBuildError::MissingPrimaryLanguage)?;
        Ok(Self {
            primary: parse(primary)?,
            secondary: secondary.iter().map(parse).collect::<Result<_, _>>()?,
        })
    }

    /// The primary language: `@ID` headers carry it, and a word-level
    /// utterance in another language gets a `[- lang]` precode.
    pub fn primary(&self) -> &LanguageCode {
        &self.primary
    }

    /// Every language, primary first, as `@Languages` lists them.
    pub fn all(&self) -> Vec<LanguageCode> {
        std::iter::once(self.primary.clone())
            .chain(self.secondary.iter().cloned())
            .collect()
    }
}

/// A participant in the transcript.
#[derive(Debug, Clone)]
pub struct ParticipantDesc {
    /// Speaker code (e.g. `"PAR"`, `"INV"`, `"CHI"`).
    pub id: String,
    /// Participant name for `@Participants` header. `None` omits the name
    /// field (output: `CODE Role`). `Some("...")` adds it (output: `CODE Name Role`).
    pub name: Option<String>,
    /// Participant role (e.g. `"Participant"`, `"Investigator"`, `"Target_Child"`).
    /// The ASR bridge derives it from the speaker code.
    pub role: String,
    /// Corpus name for `@ID` header. Empty string if unknown.
    pub corpus: String,
}

/// An utterance in the transcript.
#[derive(Debug, Clone)]
pub struct UtteranceDesc {
    /// Speaker code for this utterance.
    pub speaker: String,
    /// What the utterance is made of.
    pub body: UtteranceBody,
    /// Detected language for this utterance, parsed. When set and different
    /// from the primary language, a `[- lang]` precode is prepended.
    pub lang: Option<LanguageCode>,
}

/// An utterance's content: word-level tokens with how the utterance ends, or
/// a CHAT utterance line.
#[derive(Debug, Clone)]
pub enum UtteranceBody {
    /// Word-level tokens with per-word timing, and the utterance's end. The
    /// end is a field, never a word: the builder writes its terminator from
    /// it, and every token in `words` is content.
    Words {
        /// The content words.
        words: Vec<WordDesc>,
        /// How the utterance ends.
        end: asr_postprocess::UtteranceEnd,
    },
    /// A full utterance line (its terminator included), parsed via
    /// tree-sitter, with utterance-level timing.
    ///
    /// For callers who pass pre-formatted CHAT text rather than word
    /// tokens. Unused by the ASR pipeline (which always provides words).
    Text {
        /// The utterance's CHAT text.
        text: String,
        /// Utterance-level timing, admitted where it is described.
        timing: DescribedTiming,
    },
}

/// A single word token with optional timing.
#[derive(Debug, Clone)]
pub struct WordDesc {
    /// The token, parsed when it was admitted.
    pub text: asr_postprocess::ChatWordText,
    /// The word's timing, admitted where it is described.
    pub timing: DescribedTiming,
    /// What role this word plays (regular, retrace, etc.).
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
