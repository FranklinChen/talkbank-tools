//! Typed evidence retained by utterance timing recovery.
//!
//! Constructors stay inside the parent UTR module. Public consumers can read
//! and serialize evidence, but cannot fabricate cross-domain addresses or an
//! empty matched-word population.

use batchalign_transform::decisions::{DecisionRecord, LineIdx};

use super::anchors::AnchorIndex;

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
    /// Word anchors this pass observed: transcript words matched exactly (or
    /// case-insensitively) to a timed single-word ASR token, which forced
    /// alignment may cut an over-budget utterance at.
    ///
    /// Derived from `alignment` and the token stream that produced it, in the
    /// source-bound prepared pass that holds both, so the plan's JSON is
    /// already its replayable form; like `decisions`, it is excluded from
    /// equality and serialization.
    #[serde(skip)]
    pub(super) anchors: AnchorIndex,
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
            // Nothing was matched, so nothing was anchored: forced alignment
            // then groups exactly as it does without recovery.
            anchors: AnchorIndex::not_observed(),
        }
    }

    /// Whether recovery ran at all. Only the no-untimed-utterances result did
    /// not; every other result came from matching ASR tokens.
    pub fn ran(&self) -> bool {
        match self.alignment {
            UtrAlignmentEvidence::NotRunNoUntimed => false,
            UtrAlignmentEvidence::Global { .. } | UtrAlignmentEvidence::TwoPass { .. } => true,
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

    /// Take the word anchors this pass observed, for forced-alignment
    /// grouping, leaving the result's counts and decisions to be read.
    /// The result then reports no observation.
    pub fn take_anchors(&mut self) -> AnchorIndex {
        std::mem::replace(&mut self.anchors, AnchorIndex::not_observed())
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
            anchors,
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
            // Anchors come from the global pass only. Pass 2 does match the
            // excluded overlap utterances' words against the stream, within
            // local windows, but only to recover their bullets; those local
            // matches are not read for anchors.
            anchors,
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
    /// Joint singleton/adjacent different-speaker-run episode composition.
    LocalInterleaving,
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

    /// THE crossing from a region's local census into the file's: a region
    /// is a contiguous census slice starting at `region_start`, so a local
    /// ordinal names the file utterance that far past it. Only plan assembly
    /// (`UtrAlignmentPlan::assemble`) reaches this, once per published address.
    pub(super) fn placed_after(self, region_start: Self) -> Self {
        Self(region_start.0 + self.0)
    }

    /// THE conversion into forced alignment's utterance space.
    ///
    /// Both count `Line::Utterance` entries in document order, so the value
    /// carries over unchanged; what this function adds is that the crossing
    /// happens in one named place, never as a `.0` read beside a constructor.
    pub(super) fn fa_utterance(self) -> talkbank_model::UtteranceIdx {
        talkbank_model::UtteranceIdx::new(self.0)
    }
}

/// Zero-based alignable-word ordinal within one CHAT utterance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
#[serde(transparent)]
pub(super) struct UtrWordOrdinal(pub(super) usize);

impl UtrWordOrdinal {
    /// THE conversion into forced alignment's word space.
    ///
    /// UTR and FA grouping both extract an utterance's words through
    /// `collect_fa_words`, so a UTR word ordinal and an FA `WordIdx` address
    /// the same alignable word. That shared extraction is the whole proof, and
    /// this is the only place it is relied on: word anchors cross from UTR's
    /// plan into FA grouping through here and nowhere else.
    pub(super) fn fa_word(self) -> talkbank_model::WordIdx {
        talkbank_model::WordIdx::new(self.0)
    }
}

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
    /// The same word addressed in the file's census instead of a region's.
    pub(super) fn placed_after(self, region_start: UtrUtteranceOrdinal) -> Self {
        let Self {
            utterance_index,
            word_index,
        } = self;
        Self {
            utterance_index: utterance_index.placed_after(region_start),
            word_index,
        }
    }

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
    /// Equal lexical words after removing terminal ASR sentence punctuation;
    /// original provider text and coordinates remain in the match evidence.
    TerminalPunctuation,
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

/// One selected CHAT-word to ASR-token match, not proof of unique correspondence.
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

impl UtrWordMatch {
    /// The same match with its CHAT word addressed in the file's census.
    pub(super) fn placed_after(self, region_start: UtrUtteranceOrdinal) -> Self {
        let Self {
            word,
            token,
            chat_text,
            asr_text,
            relation,
        } = self;
        Self {
            word: word.placed_after(region_start),
            token,
            chat_text,
            asr_text,
            relation,
        }
    }
}

impl NonEmptyUtrWordMatches {
    /// Every match placed in the file's census.
    pub(super) fn placed_after(self, region_start: UtrUtteranceOrdinal) -> Self {
        let Self { first, rest } = self;
        Self {
            first: first.placed_after(region_start),
            rest: rest
                .into_iter()
                .map(|matched| matched.placed_after(region_start))
                .collect(),
        }
    }

    pub(super) fn from_vec(matches: Vec<UtrWordMatch>) -> Option<Self> {
        let mut matches = matches.into_iter();
        let first = matches.next()?;
        Some(Self {
            first,
            rest: matches.collect(),
        })
    }

    /// The lowest and highest matched ASR token ordinals for addressing and
    /// diagnostics. Ordinal order does not establish temporal extrema.
    #[cfg(test)]
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

/// A positive UTR interval admitted by its timing producer.
///
/// This is a provisional hint, not proof of recording containment or final
/// CHAT validity. Consumers can read it but cannot create an empty interval.
///
/// ```
/// use batchalign::chat_ops::fa::utr::PositiveUtrInterval;
/// fn duration(interval: PositiveUtrInterval) -> u64 {
///     interval.end_ms() - interval.start_ms()
/// }
/// ```
///
/// ```compile_fail
/// use batchalign::chat_ops::fa::utr::PositiveUtrInterval;
/// let invalid = PositiveUtrInterval { start_ms: 100, end_ms: 100 };
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct PositiveUtrInterval {
    start_ms: u64,
    end_ms: u64,
}

impl PositiveUtrInterval {
    /// Admit the interval where provider timing becomes a proposed hint.
    pub(super) fn admit(start_ms: u64, end_ms: u64) -> Option<Self> {
        (start_ms < end_ms).then_some(Self { start_ms, end_ms })
    }

    /// Start of the admitted interval, in recording milliseconds.
    pub fn start_ms(self) -> u64 {
        self.start_ms
    }

    /// End of the admitted interval, in recording milliseconds.
    pub fn end_ms(self) -> u64 {
        self.end_ms
    }

    /// Publish admitted recovery geometry as a provisional hint, never as an
    /// original transcript boundary. FA must derive its final span from words.
    pub(super) fn into_hint(self) -> talkbank_model::model::Bullet {
        talkbank_model::model::Bullet::utr_hint(self.start_ms, self.end_ms)
    }

    /// Keep only the observed interval after a preceding non-overlap end.
    /// Exhaustion leaves no hint; this operation never extends an end.
    pub(super) fn after(self, floor_end_ms: u64) -> Option<Self> {
        Self::admit(self.start_ms.max(floor_end_ms), self.end_ms)
    }

    /// Cover two admitted observations without inventing either endpoint.
    fn covering(self, other: Self) -> Self {
        Self {
            start_ms: self.start_ms.min(other.start_ms),
            end_ms: self.end_ms.max(other.end_ms),
        }
    }
}

/// Timing geometry implied by one utterance's matched ASR tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum UtrTimingProposal {
    /// The matched tokens imply a usable positive-duration span.
    Positive {
        /// Producer-admitted interval; serialization retains start/end fields.
        #[serde(flatten)]
        interval: PositiveUtrInterval,
    },
    /// A matched provider token has zero or negative duration.
    NonPositive {
        /// Start of the unusable matched ASR token.
        start_ms: u64,
        /// End of the unusable matched ASR token.
        end_ms: u64,
    },
}

impl UtrTimingProposal {
    /// Admit every matched provider interval before combining its evidence.
    /// The first token proves nonemptiness; unrelated tokens cannot widen the
    /// proposal. Projection consumes this classification without rematching.
    pub(super) fn from_observations<'a>(
        first: &'a super::lexical::ObservedWordTiming,
        rest: impl IntoIterator<Item = &'a super::lexical::ObservedWordTiming>,
    ) -> Self {
        let admit = |token: &super::lexical::ObservedWordTiming| {
            PositiveUtrInterval::admit(token.start_ms, token.end_ms).ok_or(Self::NonPositive {
                start_ms: token.start_ms,
                end_ms: token.end_ms,
            })
        };
        let mut interval = match admit(first) {
            Ok(interval) => interval,
            Err(refusal) => return refusal,
        };
        for token in rest {
            let next = match admit(token) {
                Ok(interval) => interval,
                Err(refusal) => return refusal,
            };
            interval = interval.covering(next);
        }
        Self::Positive { interval }
    }
}

#[cfg(test)]
mod interval_tests {
    use super::*;

    #[test]
    fn interval_admission_and_clipping_never_extend_observed_evidence() {
        assert_eq!(PositiveUtrInterval::admit(100, 100), None);
        assert_eq!(PositiveUtrInterval::admit(200, 100), None);
        let interval = PositiveUtrInterval::admit(100, 200).expect("positive interval");
        assert_eq!(interval.after(0), Some(interval));
        let clipped = interval.after(150).expect("positive residual interval");
        assert_eq!((clipped.start_ms(), clipped.end_ms()), (150, 200));
        assert_eq!(interval.after(200), None);
        assert_eq!(interval.after(300), None);
        let ceiling = PositiveUtrInterval::admit(u64::MAX - 1, u64::MAX)
            .expect("positive interval at integer ceiling");
        assert_eq!(ceiling.after(u64::MAX - 1), Some(ceiling));
        assert_eq!(ceiling.after(u64::MAX), None);
    }

    #[test]
    fn checked_positive_payload_preserves_the_proposal_wire_shape() {
        let proposal = UtrTimingProposal::Positive {
            interval: PositiveUtrInterval::admit(100, 900).expect("positive interval"),
        };
        insta::assert_json_snapshot!(proposal, @r###"
        {
          "status": "positive",
          "start_ms": 100,
          "end_ms": 900
        }
        "###);
    }
}

/// Complete evidence state for one CHAT utterance in a global UTR plan.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum UtrUtteranceAlignmentEvidence {
    /// Both utterance endpoints have producer-admitted correspondences.
    Matched {
        /// Zero-based utterance ordinal among main tiers.
        utterance_index: UtrUtteranceOrdinal,
        /// Number of alignable CHAT words in the utterance.
        alignable_words: usize,
        /// Non-empty matched word population.
        matches: NonEmptyUtrWordMatches,
        /// Common-to-every-optimum matches admitted by the lexical producer.
        admitted_matches: super::lexical::EndpointBoundUtrWordMatches,
        /// Timing hull implied only by admitted correspondences.
        proposal: UtrTimingProposal,
    },
    /// Proved words remain usable as anchors, but cannot bound the utterance.
    InteriorOnly {
        /// Source utterance ordinal.
        utterance_index: UtrUtteranceOrdinal,
        /// Number of alignable CHAT words.
        alignable_words: usize,
        /// Selected optimum retained for inspection, not timing authority.
        matches: NonEmptyUtrWordMatches,
        /// Common correspondences retained even without endpoint proof.
        admitted_matches: super::lexical::NonEmptyAdmittedUtrWordMatches,
        /// Precisely which endpoint proof is missing.
        missing_endpoints: super::lexical::MissingUtrEndpoints,
    },
    /// The selected path remains inspectable but cannot authorize timing.
    SelectedOnly {
        /// Source utterance ordinal.
        utterance_index: UtrUtteranceOrdinal,
        /// Number of alignable CHAT words.
        alignable_words: usize,
        /// Arbitrarily selected optimum, retained for inspection.
        matches: NonEmptyUtrWordMatches,
        /// Ambiguity is distinct from bounded proof work not being completed.
        reason: UtrCorrespondenceRefusal,
    },
    /// Proof was refused before a selected match population was available.
    Refused {
        /// Source utterance ordinal.
        utterance_index: UtrUtteranceOrdinal,
        /// Number of alignable CHAT words.
        alignable_words: usize,
        /// Refusal does not claim lexical absence or acoustic ambiguity.
        reason: UtrCorrespondenceRefusal,
    },
    /// Already timed and kept unchanged. Its region's correspondence search
    /// exceeded its budget, so no word correspondence (and no forced-alignment
    /// word anchor) was observed for it. Timing is never refused.
    RetainedUnsearched {
        /// Source utterance ordinal.
        utterance_index: UtrUtteranceOrdinal,
        /// Number of alignable CHAT words.
        alignable_words: usize,
        /// The region and the budget it exceeded.
        unsearched: UtrBudgetRefusal,
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

impl UtrUtteranceAlignmentEvidence {
    /// The same evidence with every utterance and word address placed in the
    /// file's census. Exhaustive, with every field named, so a new address
    /// cannot be left in a region's local numbering.
    fn placed_after(self, region_start: UtrUtteranceOrdinal) -> Self {
        match self {
            Self::Matched {
                utterance_index,
                alignable_words,
                matches,
                admitted_matches,
                proposal,
            } => Self::Matched {
                utterance_index: utterance_index.placed_after(region_start),
                alignable_words,
                matches: matches.placed_after(region_start),
                admitted_matches: admitted_matches.placed_after(region_start),
                proposal,
            },
            Self::InteriorOnly {
                utterance_index,
                alignable_words,
                matches,
                admitted_matches,
                missing_endpoints,
            } => Self::InteriorOnly {
                utterance_index: utterance_index.placed_after(region_start),
                alignable_words,
                matches: matches.placed_after(region_start),
                admitted_matches: admitted_matches.placed_after(region_start),
                missing_endpoints,
            },
            Self::SelectedOnly {
                utterance_index,
                alignable_words,
                matches,
                reason,
            } => Self::SelectedOnly {
                utterance_index: utterance_index.placed_after(region_start),
                alignable_words,
                matches: matches.placed_after(region_start),
                reason,
            },
            Self::Refused {
                utterance_index,
                alignable_words,
                reason,
            } => Self::Refused {
                utterance_index: utterance_index.placed_after(region_start),
                alignable_words,
                reason,
            },
            Self::RetainedUnsearched {
                utterance_index,
                alignable_words,
                unsearched,
            } => Self::RetainedUnsearched {
                utterance_index: utterance_index.placed_after(region_start),
                alignable_words,
                unsearched,
            },
            Self::Unmatched {
                utterance_index,
                alignable_words,
            } => Self::Unmatched {
                utterance_index: utterance_index.placed_after(region_start),
                alignable_words,
            },
            Self::ExcludedMarkedOverlap {
                utterance_index,
                alignable_words,
            } => Self::ExcludedMarkedOverlap {
                utterance_index: utterance_index.placed_after(region_start),
                alignable_words,
            },
            Self::NoAlignableWords { utterance_index } => Self::NoAlignableWords {
                utterance_index: utterance_index.placed_after(region_start),
            },
        }
    }
}

/// One region's plan in the region's own census numbering: what a planner
/// returns before the region is placed in the file. It cannot be published;
/// [`UtrAlignmentPlan::assemble`] is the only route into a file plan, and it
/// places every region through [`UtrUtteranceOrdinal::placed_after`].
///
/// Built only by [`Self::from_searched`], from one value per searched
/// utterance of one region: exactly one entry per searched utterance, each
/// pairing that utterance's evidence with its search envelope, and the span
/// of the region it was searched for. The planner cannot hand `assemble` a
/// plan shorter than its region, evidence and envelopes of different
/// lengths, or a plan under another region's span.
pub(super) struct LocalRegionPlan<'c> {
    /// Algorithm the region's planner used (or attempted, if refused).
    strategy: UtrAlignmentStrategy,
    /// The region this plan searched.
    span: super::regions::UtrRegionSpan,
    /// One entry per searched utterance, owned or trailing context.
    utterances: Vec<PlannedUtterance<'c>>,
    /// Tokens whose admitted correspondences were withdrawn as contested.
    withdrawn_claims: Vec<UtrAsrTokenOrdinal>,
}

/// One searched utterance's plan: the utterance, its correspondence
/// evidence, and the search envelope forced alignment may use for it.
pub(super) struct PlannedUtterance<'c> {
    /// The utterance as the census records it.
    pub(super) info: &'c super::UtrUtteranceInfo,
    /// What correspondence established for it.
    pub(super) evidence: UtrUtteranceAlignmentEvidence,
    /// Where forced alignment may search for its words, if anywhere.
    pub(super) envelope: Option<super::search::FaSearchEnvelope>,
}

impl<'c> LocalRegionPlan<'c> {
    /// Plan every searched utterance of one region: `plan` turns each
    /// utterance and its value into its evidence and envelope.
    pub(super) fn from_searched<T>(
        strategy: UtrAlignmentStrategy,
        withdrawn_claims: Vec<UtrAsrTokenOrdinal>,
        per_utterance: super::regions::PerSearched<'c, T>,
        mut plan: impl FnMut(
            usize,
            &'c super::UtrUtteranceInfo,
            T,
        ) -> (
            UtrUtteranceAlignmentEvidence,
            Option<super::search::FaSearchEnvelope>,
        ),
    ) -> Self {
        let span = per_utterance.searched().span();
        let utterances = per_utterance
            .into_pairs()
            .enumerate()
            .map(|(ordinal, (info, value))| {
                let (evidence, envelope) = plan(ordinal, info, value);
                PlannedUtterance {
                    info,
                    evidence,
                    envelope,
                }
            })
            .collect();
        Self {
            strategy,
            span,
            utterances,
            withdrawn_claims,
        }
    }

    /// Every searched utterance's plan, in order. A slice: entries may be
    /// refined (an envelope added), never added or removed.
    pub(super) fn utterances_mut(&mut self) -> &mut [PlannedUtterance<'c>] {
        &mut self.utterances
    }

    /// The provider tokens this region's OWNED utterances admitted as
    /// correspondences: the claims the file plan will publish. Trailing
    /// context is excluded; its own region publishes it.
    pub(super) fn owned_claims(&self) -> impl Iterator<Item = UtrAsrTokenOrdinal> + '_ {
        self.utterances
            .iter()
            .take(self.span.owned())
            .flat_map(|planned| {
                // The two admitting states hold different proof types; one of
                // the two options is filled, so nothing is allocated.
                let (bound, interior) = match &planned.evidence {
                    UtrUtteranceAlignmentEvidence::Matched {
                        admitted_matches, ..
                    } => (Some(admitted_matches), None),
                    UtrUtteranceAlignmentEvidence::InteriorOnly {
                        admitted_matches, ..
                    } => (None, Some(admitted_matches)),
                    UtrUtteranceAlignmentEvidence::SelectedOnly { .. }
                    | UtrUtteranceAlignmentEvidence::Refused { .. }
                    | UtrUtteranceAlignmentEvidence::RetainedUnsearched { .. }
                    | UtrUtteranceAlignmentEvidence::Unmatched { .. }
                    | UtrUtteranceAlignmentEvidence::ExcludedMarkedOverlap { .. }
                    | UtrUtteranceAlignmentEvidence::NoAlignableWords { .. } => (None, None),
                };
                bound
                    .into_iter()
                    .flat_map(|matches| matches.iter())
                    .chain(interior.into_iter().flat_map(|matches| matches.iter()))
                    .map(|matched| matched.matched().token.token_index)
            })
    }
}

/// Which word-order model a pass's census implies. Chosen once per file,
/// exactly as before regions existed; regions bound the search, not the
/// model, so hint projection reads this and nothing per region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UtrOrderModel {
    /// One monotonic word order: no two adjacent participating turns belong
    /// to different speakers.
    Monotonic,
    /// Adjacent different-speaker turns may interleave (joint composition);
    /// each speaker's own order is preserved.
    Interleaved,
}

/// What one anchored region did, for inspection: which utterances it owned,
/// which ASR onsets it searched and which algorithm ran. Refusals are
/// recorded on the utterances themselves, each naming its region.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct UtrRegionSummary {
    /// Owned utterances and searched onset window.
    pub(super) span: super::regions::UtrRegionSpan,
    /// Algorithm used for this region (attempted, if refused).
    pub(super) strategy: UtrAlignmentStrategy,
    /// Provider tokens whose correspondences this region withdrew because
    /// a neighbouring region's owned words claimed them too: the regional
    /// search cannot tell which claim the whole file would admit.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(super) withdrawn_claims: Vec<UtrAsrTokenOrdinal>,
}

impl UtrRegionSummary {
    /// Owned utterances and searched onset window.
    pub fn span(&self) -> super::regions::UtrRegionSpan {
        self.span
    }

    /// Algorithm used for this region.
    pub fn strategy(&self) -> UtrAlignmentStrategy {
        self.strategy
    }
}

/// Complete replayable evidence for one UTR pass.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct UtrAlignmentPlan {
    /// Order model of the whole pass; governs hint projection.
    pub(super) strategy: UtrOrderModel,
    /// Anchored regions in document order; together they own every
    /// utterance exactly once.
    pub(super) regions: Vec<UtrRegionSummary>,
    /// Exhaustive evidence in CHAT utterance order.
    pub(super) utterances: Vec<UtrUtteranceAlignmentEvidence>,
    /// Complete exact candidate ranges: search authority only, not timing.
    pub(super) search_envelopes: Vec<Option<super::search::FaSearchEnvelope>>,
}

impl UtrAlignmentPlan {
    /// Place each region's local plan in the file, in document order.
    ///
    /// Keeps the evidence for the utterances each region owns, dropping its
    /// trailing context anchor (whose evidence its own region publishes),
    /// and addresses it in the file's census. Regions arrive in document
    /// order and partition the census, so concatenation is file order.
    pub(super) fn assemble<'c>(
        strategy: UtrOrderModel,
        regions: impl IntoIterator<Item = LocalRegionPlan<'c>>,
    ) -> Self {
        let mut plan = Self {
            strategy,
            regions: Vec::new(),
            utterances: Vec::new(),
            search_envelopes: Vec::new(),
        };
        for local in regions {
            let LocalRegionPlan {
                strategy,
                span,
                utterances,
                withdrawn_claims,
            } = local;
            let region_start = span.first_utterance();
            plan.regions.push(UtrRegionSummary {
                span,
                strategy,
                withdrawn_claims,
            });
            // A region plan has one entry per searched utterance, and a
            // region searches every utterance it owns first, so the first
            // `owned` entries are exactly the owned utterances.
            for planned in utterances.into_iter().take(span.owned()) {
                plan.utterances
                    .push(planned.evidence.placed_after(region_start));
                plan.search_envelopes.push(planned.envelope);
            }
        }
        plan
    }

    /// Order model of the whole pass.
    pub fn order_model(&self) -> UtrOrderModel {
        self.strategy
    }

    /// Anchored regions in document order.
    pub fn regions(&self) -> &[UtrRegionSummary] {
        &self.regions
    }

    /// Every utterance's evidence in CHAT order.
    pub fn utterances(&self) -> &[UtrUtteranceAlignmentEvidence] {
        &self.utterances
    }

    /// Per-utterance matched token extents, for tests that pin which tokens
    /// an alignment chose. Production never reads token ranges: projection
    /// consumes each utterance's `UtrTimingProposal` instead.
    #[cfg(test)]
    pub(super) fn token_extents(&self) -> Vec<Option<(usize, usize)>> {
        self.utterances
            .iter()
            .map(|utterance| match utterance {
                UtrUtteranceAlignmentEvidence::Matched { matches, .. }
                | UtrUtteranceAlignmentEvidence::InteriorOnly { matches, .. }
                | UtrUtteranceAlignmentEvidence::SelectedOnly { matches, .. } => {
                    let (minimum, maximum) = matches.token_extent();
                    Some((minimum.index(), maximum.index()))
                }
                UtrUtteranceAlignmentEvidence::Refused { .. }
                | UtrUtteranceAlignmentEvidence::RetainedUnsearched { .. }
                | UtrUtteranceAlignmentEvidence::Unmatched { .. }
                | UtrUtteranceAlignmentEvidence::ExcludedMarkedOverlap { .. }
                | UtrUtteranceAlignmentEvidence::NoAlignableWords { .. } => None,
            })
            .collect()
    }
}

/// Why a selected path did not establish correspondence authority.
///
/// Externally tagged on the wire: `"ambiguous"`, or
/// `{"budget_exhausted": {"region": ..., "budget": ...}}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UtrCorrespondenceRefusal {
    /// None of this utterance's matches occurs in every optimal alignment.
    Ambiguous,
    /// The utterance's region exceeded a fixed search budget. Says nothing
    /// about lexical absence or ambiguity, and nothing about other regions.
    BudgetExhausted(UtrBudgetRefusal),
}

/// One region's correspondence search exceeding one fixed budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct UtrBudgetRefusal {
    /// The refused region; no other region is affected.
    pub(super) region: super::regions::UtrRegionSpan,
    /// The budget it exceeded.
    pub(super) budget: UtrSearchBudget,
}

impl UtrBudgetRefusal {
    /// The refused region.
    pub fn region(&self) -> super::regions::UtrRegionSpan {
        self.region
    }

    /// The budget it exceeded.
    pub fn budget(&self) -> UtrSearchBudget {
        self.budget
    }
}

/// Which fixed correspondence-search budget a region exceeded, with the
/// limit in that budget's own unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UtrSearchBudget {
    /// Monotonic proof: candidate (word, token) match edges.
    CandidateEdges {
        /// Edge limit.
        limit: usize,
    },
    /// Monotonic proof under a fuzzy relation: word-pair comparisons.
    FuzzyComparisons {
        /// Comparison limit.
        limit: usize,
    },
    /// Interleaved proof: source-graph nodes times reference positions.
    InterleavingWork {
        /// Work-cell limit.
        limit: usize,
    },
    /// Interleaved proof: retained score cells.
    InterleavingMemory {
        /// Score-cell limit.
        limit: usize,
    },
    /// Interleaved proof: distinct addressed candidate matches.
    InterleavingCandidates {
        /// Candidate limit.
        limit: usize,
    },
}

impl UtrSearchBudget {
    /// The monotonic correspondence producer's named budget.
    pub(super) fn monotonic(budget: batchalign_transform::dp_align::CorrespondenceBudget) -> Self {
        use batchalign_transform::dp_align::CorrespondenceBudget;
        let limit = budget.limit();
        match budget {
            CorrespondenceBudget::CandidateEdges => Self::CandidateEdges { limit },
            CorrespondenceBudget::FuzzyComparisons => Self::FuzzyComparisons { limit },
        }
    }

    /// The interleaved correspondence producer's named budget.
    pub(super) fn interleaved(
        refusal: batchalign_transform::dp_align::interleaving::LocalInterleavingRefusal,
    ) -> Self {
        use batchalign_transform::dp_align::interleaving::LocalInterleavingRefusal;
        let limit = refusal.limit();
        match refusal {
            LocalInterleavingRefusal::WorkBudgetExceeded => Self::InterleavingWork { limit },
            LocalInterleavingRefusal::MemoryBudgetExceeded => Self::InterleavingMemory { limit },
            LocalInterleavingRefusal::CandidateBudgetExceeded => {
                Self::InterleavingCandidates { limit }
            }
        }
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
