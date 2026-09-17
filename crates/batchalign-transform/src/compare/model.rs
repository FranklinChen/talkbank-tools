/// Status of a compared token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareStatus {
    /// Word matches between main and gold.
    Match,
    /// Word present in main but not in gold (insertion).
    ExtraMain,
    /// Word present in gold but not in main (deletion).
    ExtraGold,
}

/// A single token in the comparison output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompareToken {
    /// The word text.
    pub text: String,
    /// Uppercased part-of-speech tag when `%mor` data is available.
    pub pos: Option<String>,
    /// Match status.
    pub status: CompareStatus,
}

/// Per-utterance comparison result.
#[derive(Debug, Clone)]
pub struct UtteranceComparison {
    /// Zero-based utterance index in the main file.
    pub utterance_index: usize,
    /// Speaker code.
    pub speaker: String,
    /// Comparison tokens (matches, insertions, deletions).
    pub tokens: Vec<CompareToken>,
}

/// What a gold transcript claims to cover, stated by the caller.
///
/// Compare aligns the two transcripts word for word, and some main utterances
/// match no gold word at all. Whether their words are ERRORS is not a fact
/// compare can work out from the two files: it depends on what the gold was
/// made to be.
///
/// There is deliberately no `Default`. A wrong answer here moves the headline
/// WER in a direction nobody would notice, so the caller states it and the
/// compiler makes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoldCoverage {
    /// The gold is a full reference for this transcript.
    ///
    /// Main material the gold does not account for is material the system
    /// produced and the reference does not contain, so it is charged as
    /// insertions. This is the right answer for a gold companion that is a
    /// re-transcription of the same recording. Nothing is decided from main
    /// utterance boundaries, so the scores are the same however the main
    /// transcript is split into utterances.
    Complete,
    /// The gold covers only part of what the main transcript covers.
    ///
    /// A main utterance that matched no gold word is not scored at all,
    /// because the reference makes no claim about it. This is the right answer
    /// for a sampled slice, a single timepoint, or a single-speaker reference.
    /// Reported WER then describes the covered part only, and, unlike
    /// [`Self::Complete`], depends on where the main transcript puts its
    /// utterance boundaries, since that is how the covered part is found.
    ///
    /// Known limit: what is in scope is decided per main utterance from one
    /// alignment of the whole file, which maximizes matched words without
    /// regard to how spread out they are. With a single-speaker gold, one
    /// chance match of a common word in another speaker's utterance brings that
    /// whole utterance into scope, and its other words are charged as
    /// insertions. No production caller passes `Partial` today.
    Partial,
}

pub use super::metrics::CompareMetrics;

/// Per-POS compare counters.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PosErrorCounts {
    /// Number of matching tokens for this POS.
    pub matches: usize,
    /// Number of insertion tokens for this POS.
    pub insertions: usize,
    /// Number of deletion tokens for this POS.
    pub deletions: usize,
}

/// Full comparison bundle.
///
/// This is the internal workflow artifact produced by transcript comparison.
/// It can later support multiple materialization paths (main-annotated output,
/// gold-projected output, metrics sidecars, debugging views) without forcing
/// the compare stage itself to decide the final output shape.
#[derive(Debug, Clone)]
pub struct ComparisonBundle {
    /// Main-anchored per-utterance comparison annotations.
    pub main_utterances: Vec<UtteranceComparison>,
    /// Gold-anchored per-utterance comparison annotations.
    pub gold_utterances: Vec<UtteranceComparison>,
    /// Structural word matches from gold back to the matched main word.
    pub gold_word_matches: Vec<GoldWordMatch>,
    /// Aggregate metrics.
    pub metrics: CompareMetrics,
}

/// Compatibility alias retained while the compare pipeline is refactored toward
/// workflow bundles plus explicit materializers.
pub type CompareResult = ComparisonBundle;

/// A structural match between one gold word slot and one main word slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GoldWordMatch {
    /// Gold utterance containing the matched word.
    pub gold_utterance_index: usize,
    /// Zero-based compared-word position within the gold utterance.
    pub gold_word_position: usize,
    /// Main utterance supplying the matched word.
    pub main_utterance_index: usize,
    /// Zero-based compared-word position within the main utterance.
    pub main_word_position: usize,
}
