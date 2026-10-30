//! Why a retokenize rebuild kept an utterance's transcript as written.
//!
//! Any refusal keeps the whole utterance as transcribed and reports it. The
//! refusals used to be strings, so the report could not say which one
//! happened, and the injection layer filled the gap by labelling every
//! retokenize refusal `mwt_reassembly_bug`: a token merged across a retrace
//! was reported to the user as a multi-word-token defect. Each refusal now
//! carries its kind, and its [`MisalignmentClass`] is the kind's, or
//! `Unknown` where no class names it.

use std::fmt;

use super::rebuild::ExpansionRefusal;
use crate::inject::MisalignmentClass;

/// One reason the rebuild could not rewrite the main tier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RebuildRefusal {
    /// A token joins word `first_word` to the next word across other
    /// main-tier content (a retrace, a pause, a bracket edge).
    MergeCrossesContent { token: usize, first_word: usize },
    /// A word owns no token in the mapping.
    WordUnmapped { word: usize },
    /// A scoped word (annotated or replacement) could not be expanded.
    ScopedExpansion(ExpansionRefusal),
    /// A replacement (`[: ...]`) has no words to expand.
    ReplacementWithoutWords,
    /// A model token is not valid CHAT word syntax.
    NotAWord { token: String },
    /// The model's terminator token differs from the transcript's.
    TerminatorDiffers { model: String, transcript: String },
}

impl RebuildRefusal {
    /// The misalignment class this refusal reports as.
    pub(crate) fn class(&self) -> MisalignmentClass {
        match self {
            Self::MergeCrossesContent { .. } => MisalignmentClass::MergeCrossesContent,
            Self::WordUnmapped { .. }
            | Self::ScopedExpansion(_)
            | Self::ReplacementWithoutWords
            | Self::NotAWord { .. }
            | Self::TerminatorDiffers { .. } => MisalignmentClass::Unknown,
        }
    }
}

impl fmt::Display for RebuildRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MergeCrossesContent { token, first_word } => write!(
                f,
                "token {token} merges words {first_word} and {} across other main-tier \
                 content; keeping the transcript as written",
                first_word + 1
            ),
            Self::WordUnmapped { word } => write!(
                f,
                "word {word} has no character-level match in Stanza tokens; keeping original"
            ),
            Self::ScopedExpansion(refused) => {
                write!(f, "retokenize scoped word: {refused}; keeping original")
            }
            Self::ReplacementWithoutWords => {
                write!(f, "replacement has no admitted words; keeping original")
            }
            Self::NotAWord { token } => write!(
                f,
                "token {token:?} is not valid CHAT word syntax; keeping original word"
            ),
            Self::TerminatorDiffers { model, transcript } => write!(
                f,
                "skipped Stanza terminator {model:?} does not match existing terminator \
                 {transcript:?}; keeping existing terminator"
            ),
        }
    }
}
