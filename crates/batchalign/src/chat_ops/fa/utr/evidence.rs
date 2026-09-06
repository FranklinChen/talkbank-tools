//! Typed evidence retained by utterance timing recovery.
//!
//! Constructors stay inside the parent UTR module. Public consumers can read
//! and serialize evidence, but cannot fabricate cross-domain addresses or an
//! empty matched-word population.

use batchalign_transform::decisions::{DecisionRecord, LineIdx};

/// Result summary from UTR injection.
#[derive(Debug, Clone, serde::Serialize)]
pub struct UtrResult {
    /// Utterances that received timing from ASR tokens.
    pub(super) injected: usize,
    /// Already-timed utterances left unchanged.
    pub(super) skipped: usize,
    /// Untimed utterances that could not be matched to ASR tokens.
    pub(super) unmatched: usize,
    /// Replayable alignment evidence retained independently from projection.
    pub(super) alignment: UtrAlignmentEvidence,
    /// Provenance records are not part of result equality or JSON evidence.
    #[serde(skip)]
    pub(super) decisions: Vec<DecisionRecord>,
}

impl PartialEq for UtrResult {
    fn eq(&self, other: &Self) -> bool {
        self.injected == other.injected
            && self.skipped == other.skipped
            && self.unmatched == other.unmatched
            && self.alignment == other.alignment
    }
}

impl Eq for UtrResult {}

impl UtrResult {
    /// Construct the only valid state in which UTR does not run.
    pub(crate) fn not_run_no_untimed(skipped: usize) -> Self {
        Self {
            injected: 0,
            skipped,
            unmatched: 0,
            alignment: UtrAlignmentEvidence::NotRunNoUntimed,
            decisions: Vec::new(),
        }
    }

    /// Number of utterances that received UTR timing.
    pub fn injected(&self) -> usize {
        self.injected
    }

    /// Number of already-timed utterances preserved by UTR.
    pub fn skipped(&self) -> usize {
        self.skipped
    }

    /// Number of untimed utterances left without UTR timing.
    pub fn unmatched(&self) -> usize {
        self.unmatched
    }

    /// Replayable alignment evidence retained independently from projection.
    pub fn alignment(&self) -> &UtrAlignmentEvidence {
        &self.alignment
    }

    /// Per-utterance decision records emitted by UTR.
    pub fn decisions(&self) -> &[DecisionRecord] {
        &self.decisions
    }

    /// Remove the obsolete pass-1 decision for a pass-2 recovered utterance.
    pub(super) fn discard_recovered_unmatched_decision(&mut self, line_idx: LineIdx) {
        self.decisions
            .retain(|decision| decision.line_idx != line_idx);
    }

    /// Attach local overlap recoveries to a completed global first pass.
    pub(super) fn with_overlap_recoveries(self, recoveries: Vec<UtrOverlapRecovery>) -> Self {
        let Self {
            injected,
            skipped,
            unmatched,
            alignment,
            decisions,
        } = self;
        let alignment = match alignment {
            UtrAlignmentEvidence::Global { plan } => UtrAlignmentEvidence::TwoPass {
                first_pass: plan,
                overlap_recoveries: recoveries,
            },
            UtrAlignmentEvidence::NotRunNoUntimed => UtrAlignmentEvidence::NotRunNoUntimed,
            UtrAlignmentEvidence::TwoPass { .. } => alignment,
        };
        Self {
            injected,
            skipped,
            unmatched,
            alignment,
            decisions,
        }
    }
}

/// Which alignment strategy produced the per-utterance token ranges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UtrAlignmentStrategy {
    /// The transcript was a unique exact monotonic ASR subsequence.
    UniqueExactSubsequence,
    /// The full-file Hirschberg alignment remained necessary.
    GlobalDp,
}

/// Zero-based main-tier utterance ordinal in one UTR plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
#[serde(transparent)]
pub struct UtrUtteranceOrdinal(pub(super) usize);

impl UtrUtteranceOrdinal {
    /// Return the zero-based ordinal for indexing the same main-tier set.
    pub fn index(self) -> usize {
        self.0
    }
}

/// Zero-based alignable-word ordinal within one CHAT utterance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
#[serde(transparent)]
pub(super) struct UtrWordOrdinal(pub(super) usize);

/// Zero-based token ordinal in the admitted ASR timing stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
#[serde(transparent)]
pub(super) struct UtrAsrTokenOrdinal(pub(super) usize);

/// Zero-based whitespace-delimited word ordinal within one provider token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
#[serde(transparent)]
pub(super) struct UtrAsrWordOrdinal(pub(super) usize);

impl UtrAsrTokenOrdinal {
    pub(super) fn index(self) -> usize {
        self.0
    }
}

/// Stable address of one alignable CHAT word in the UTR payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct UtrWordAddress {
    /// Zero-based utterance ordinal among main tiers.
    pub(super) utterance_index: UtrUtteranceOrdinal,
    /// Zero-based ordinal among alignable words in that utterance.
    pub(super) word_index: UtrWordOrdinal,
}

impl UtrWordAddress {
    /// Main-tier utterance containing this word.
    pub fn utterance_index(self) -> usize {
        self.utterance_index.index()
    }

    /// Alignable-word ordinal within the containing utterance.
    pub fn word_index(self) -> usize {
        self.word_index.0
    }
}

/// Stable address of one token in the admitted ASR timing stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct UtrAsrTokenAddress {
    /// Zero-based token ordinal in the exact ASR stream given to UTR.
    pub(super) token_index: UtrAsrTokenOrdinal,
    /// Word position within that token's original text, without inferred timing.
    pub(super) word_index: UtrAsrWordOrdinal,
}

impl UtrAsrTokenAddress {
    /// Token ordinal within the admitted ASR timing stream.
    pub fn token_index(self) -> usize {
        self.token_index.index()
    }

    /// Word ordinal after splitting the original token on Unicode whitespace.
    pub fn word_index(self) -> usize {
        self.word_index.0
    }
}

/// Why UTR treated two words as a match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UtrLexicalRelation {
    /// Byte-for-byte equality.
    Exact,
    /// ASCII case-folded equality.
    CaseInsensitive,
    /// Jaro-Winkler similarity admitted by the configured fuzzy threshold.
    Fuzzy {
        /// Similarity rounded to integer millionths.
        similarity_millionths: u32,
    },
}

/// One admitted CHAT-word to ASR-token match.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct UtrWordMatch {
    /// Address of the CHAT word.
    pub(super) word: UtrWordAddress,
    /// Address of the ASR token.
    pub(super) token: UtrAsrTokenAddress,
    /// CHAT word as presented to the aligner.
    pub(super) chat_text: String,
    /// ASR token text as presented to the aligner.
    pub(super) asr_text: String,
    /// Lexical relation that admitted the match.
    pub(super) relation: UtrLexicalRelation,
}

/// A non-empty collection of word matches.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct NonEmptyUtrWordMatches {
    /// First match in CHAT word order.
    pub(super) first: UtrWordMatch,
    /// Remaining matches in CHAT word order.
    pub(super) rest: Vec<UtrWordMatch>,
}

impl NonEmptyUtrWordMatches {
    pub(super) fn from_vec(matches: Vec<UtrWordMatch>) -> Option<Self> {
        let mut matches = matches.into_iter();
        let first = matches.next()?;
        Some(Self {
            first,
            rest: matches.collect(),
        })
    }

    /// The lowest and highest matched ASR token ordinals: the extent the
    /// proposal was built from, computed here and nowhere else.
    pub(super) fn token_extent(&self) -> (UtrAsrTokenOrdinal, UtrAsrTokenOrdinal) {
        let first = self.first.token.token_index;
        self.rest
            .iter()
            .fold((first, first), |(minimum, maximum), item| {
                let token = item.token.token_index;
                (minimum.min(token), maximum.max(token))
            })
    }
}

/// Timing geometry implied by one utterance's matched ASR tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum UtrTimingProposal {
    /// The matched tokens imply a usable positive-duration span.
    Positive {
        /// Start of the first matched ASR token.
        start_ms: u64,
        /// End of the last matched ASR token.
        end_ms: u64,
    },
    /// The matched tokens imply a zero- or negative-duration span.
    NonPositive {
        /// Start of the first matched ASR token.
        start_ms: u64,
        /// End of the last matched ASR token.
        end_ms: u64,
    },
}

impl UtrTimingProposal {
    /// The span from the first matched token's start to the last matched
    /// token's end, classified once at the moment the plan learns it. This
    /// is the ONLY place the positive/non-positive fact is decided;
    /// projection reads it and never recomputes it from tokens.
    pub(super) fn spanning(first: &super::AsrTimingToken, last: &super::AsrTimingToken) -> Self {
        if first.start_ms < last.end_ms {
            Self::Positive {
                start_ms: first.start_ms,
                end_ms: last.end_ms,
            }
        } else {
            Self::NonPositive {
                start_ms: first.start_ms,
                end_ms: last.end_ms,
            }
        }
    }
}

/// Complete evidence state for one CHAT utterance in a global UTR plan.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum UtrUtteranceAlignmentEvidence {
    /// One or more CHAT words matched ASR tokens.
    Matched {
        /// Zero-based utterance ordinal among main tiers.
        utterance_index: UtrUtteranceOrdinal,
        /// Number of alignable CHAT words in the utterance.
        alignable_words: usize,
        /// Non-empty matched word population.
        matches: NonEmptyUtrWordMatches,
        /// Span implied by the first and last matched ASR tokens.
        proposal: UtrTimingProposal,
    },
    /// The utterance had words, but none matched an ASR token.
    Unmatched {
        /// Zero-based utterance ordinal among main tiers.
        utterance_index: UtrUtteranceOrdinal,
        /// Number of alignable CHAT words in the utterance.
        alignable_words: usize,
    },
    /// Deliberately omitted from a global pass for local overlap recovery.
    ExcludedMarkedOverlap {
        /// Zero-based utterance ordinal among main tiers.
        utterance_index: UtrUtteranceOrdinal,
        /// Number of alignable CHAT words in the excluded utterance.
        alignable_words: usize,
    },
    /// The utterance contained no words eligible for UTR matching.
    NoAlignableWords {
        /// Zero-based utterance ordinal among main tiers.
        utterance_index: UtrUtteranceOrdinal,
    },
}

/// Complete replayable evidence for one global UTR alignment pass.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct UtrAlignmentPlan {
    /// Strategy used to build the matched token ranges.
    pub(super) strategy: UtrAlignmentStrategy,
    /// Exhaustive evidence in CHAT utterance order.
    pub(super) utterances: Vec<UtrUtteranceAlignmentEvidence>,
}

impl UtrAlignmentPlan {
    /// Per-utterance matched token extents, for tests that pin which tokens
    /// an alignment chose. Production never reads token ranges: projection
    /// consumes each utterance's `UtrTimingProposal` instead.
    #[cfg(test)]
    pub(super) fn token_extents(&self) -> Vec<Option<(usize, usize)>> {
        self.utterances
            .iter()
            .map(|utterance| match utterance {
                UtrUtteranceAlignmentEvidence::Matched { matches, .. } => {
                    let (minimum, maximum) = matches.token_extent();
                    Some((minimum.index(), maximum.index()))
                }
                UtrUtteranceAlignmentEvidence::Unmatched { .. }
                | UtrUtteranceAlignmentEvidence::ExcludedMarkedOverlap { .. }
                | UtrUtteranceAlignmentEvidence::NoAlignableWords { .. } => None,
            })
            .collect()
    }
}

/// Local overlap recovery retained by the two-pass UTR strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct UtrOverlapRecovery {
    /// Zero-based utterance ordinal among main tiers.
    pub(super) utterance_index: UtrUtteranceOrdinal,
    /// Start of the locally recovered timing span.
    pub(super) start_ms: u64,
    /// End of the locally recovered timing span.
    pub(super) end_ms: u64,
}

/// Which replayable alignment evidence a UTR invocation produced.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "strategy", rename_all = "snake_case")]
pub enum UtrAlignmentEvidence {
    /// UTR was skipped because every utterance was already timed.
    NotRunNoUntimed,
    /// One global alignment plan was selected.
    Global {
        /// Exact word-to-token evidence consumed by global projection.
        plan: UtrAlignmentPlan,
    },
    /// A global first pass plus local marked-overlap recoveries was selected.
    TwoPass {
        /// Exact word-to-token evidence from the global first pass.
        first_pass: UtrAlignmentPlan,
        /// Timing-only recoveries from the local overlap pass.
        overlap_recoveries: Vec<UtrOverlapRecovery>,
    },
}
