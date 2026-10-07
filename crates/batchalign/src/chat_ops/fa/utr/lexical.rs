//! Word matching retains the provider token that owns each timing interval.

use super::evidence::{UtrAsrTokenOrdinal, UtrAsrWordOrdinal};
use super::{
    AsrTimingToken, UtrAsrTokenAddress, UtrLexicalRelation, UtrTimingProposal, UtrWordAddress,
    UtrWordMatch, lexical_relation,
};
use batchalign_transform::dp_align::CommonCorrespondences;

/// A lexical projection of one retained ASR stream.
/// Raw provider segments cannot be passed directly to word alignment.
pub(super) struct UtrLexicalStream<'source> {
    tokens: &'source [AsrTimingToken],
    words: Vec<UtrLexicalWord<'source>>,
}

/// A nonempty whitespace-delimited word borrowed from its provider token.
/// Only projection constructs the word and its original-stream address.
struct UtrLexicalWord<'source> {
    text: &'source str,
    surface: &'source str,
    address: UtrAsrTokenAddress,
}

impl<'source> UtrLexicalWord<'source> {
    /// Remove provider sentence punctuation, never internal punctuation,
    /// apostrophes or hyphens. Keep the source surface and original address.
    fn from_provider_word(surface: &'source str, address: UtrAsrTokenAddress) -> Option<Self> {
        let text = surface.trim_end_matches(['.', ',', '!', '?', ';', ':']);
        (!text.is_empty()).then_some(Self {
            text,
            surface,
            address,
        })
    }

    fn relation(&self, chat_text: &str) -> UtrLexicalRelation {
        if self.text != self.surface && chat_text.eq_ignore_ascii_case(self.text) {
            UtrLexicalRelation::TerminalPunctuation
        } else {
            lexical_relation(chat_text, self.surface)
        }
    }
}

/// A selected lexical match independently proved common to every optimum.
/// Only the lexical producer constructs this; selection alone is insufficient.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(transparent)]
pub struct AdmittedUtrWordMatch {
    matched: UtrWordMatch,
    #[serde(skip)]
    word_timing: Option<ObservedWordTiming>,
    #[serde(skip)]
    timing: ObservedWordTiming,
}

/// Original provider-token timing, bound where correspondence is born.
/// `word_timing` is present only for a single-word token; `timing` also retains
/// coarse segments without inventing within-segment word boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ObservedWordTiming {
    pub(super) start_ms: u64,
    pub(super) end_ms: u64,
}

/// Outer provider endpoints of every candidate for an obligatory boundary
/// word. These are search limits, not an interval measured for a source word.
struct RequiredWordSearchBounds {
    preceding_floor_ms: u64,
    following_ceiling_ms: u64,
}

impl AdmittedUtrWordMatch {
    /// Inspect the underlying addressed correspondence without creating proof.
    pub fn matched(&self) -> &UtrWordMatch {
        &self.matched
    }

    pub(super) fn word_timing(&self) -> Option<ObservedWordTiming> {
        self.word_timing
    }

    /// The provider token interval bound into this proof (a segment's whole
    /// interval for a multi-word token). Search bounds only, never a word
    /// boundary of its own.
    pub(super) fn token_timing(&self) -> ObservedWordTiming {
        self.timing
    }

    /// The same proof with its CHAT word addressed in the file's census.
    fn placed_after(self, region_start: super::UtrUtteranceOrdinal) -> Self {
        let Self {
            matched,
            word_timing,
            timing,
        } = self;
        Self {
            matched: matched.placed_after(region_start),
            word_timing,
            timing,
        }
    }
}

/// Nonempty producer-admitted correspondences. Interior evidence permits
/// anchors, but does not by itself authorize an utterance crop window.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct NonEmptyAdmittedUtrWordMatches {
    first: AdmittedUtrWordMatch,
    rest: Vec<AdmittedUtrWordMatch>,
}

impl NonEmptyAdmittedUtrWordMatches {
    pub(super) fn from_vec(matches: Vec<AdmittedUtrWordMatch>) -> Option<Self> {
        let mut matches = matches.into_iter();
        Some(Self {
            first: matches.next()?,
            rest: matches.collect(),
        })
    }

    /// Read-only access to admitted correspondence, never arbitrary matches.
    pub fn iter(&self) -> impl Iterator<Item = &AdmittedUtrWordMatch> {
        std::iter::once(&self.first).chain(&self.rest)
    }

    /// Every proof placed in the file's census.
    pub(super) fn placed_after(self, region_start: super::UtrUtteranceOrdinal) -> Self {
        let Self { first, rest } = self;
        Self {
            first: first.placed_after(region_start),
            rest: rest
                .into_iter()
                .map(|matched| matched.placed_after(region_start))
                .collect(),
        }
    }

    pub(super) fn token_extent(&self) -> (UtrAsrTokenOrdinal, UtrAsrTokenOrdinal) {
        let first = self.first.matched.token.token_index;
        self.rest.iter().fold((first, first), |(lo, hi), matched| {
            let token = matched.matched.token.token_index;
            (lo.min(token), hi.max(token))
        })
    }

    /// Admit utterance boundaries against the producer's lexical census.
    /// Missing interior words do not invalidate proved first/last endpoints.
    ///
    /// An endpoint is bounded when its correspondence is common to every
    /// optimum, or, failing that, when `extents` says it is matched in every
    /// optimum (it can never be missing) and gives the extent of every token
    /// it may match: the endpoint lies inside that extent whichever optimum
    /// is true, so a timing hull including it excludes none of its speech.
    /// An endpoint some optimum leaves unmatched bounds nothing.
    pub(super) fn admit_endpoints(
        self,
        word_count: usize,
        extents: EndpointExtents,
    ) -> BoundaryAdmission {
        let proved = |index: usize| {
            self.iter()
                .any(|word| word.matched.word.word_index.0 == index)
        };
        let first = proved(0) || extents.first.is_some();
        let last = word_count
            .checked_sub(1)
            .is_some_and(|index| proved(index) || extents.last.is_some());
        let missing = match (first, last) {
            (true, true) => {
                return BoundaryAdmission::Bound(EndpointBoundUtrWordMatches {
                    matches: self,
                    extents,
                });
            }
            (false, true) => MissingUtrEndpoints::First,
            (true, false) => MissingUtrEndpoints::Last,
            (false, false) => MissingUtrEndpoints::Both,
        };
        BoundaryAdmission::Interior {
            matches: self,
            missing,
        }
    }
}

/// Where an utterance's first and last words can lie, when the producer
/// knows: for each endpoint word matched in every optimum (never missing),
/// the extent of every provider token it may match. Absent for a word some
/// optimum leaves unmatched, or one whose candidates are not positive
/// single-word observations of its own text. Built only by the interleaving
/// producer ([`UtrLexicalStream::plan_interleaving`]), which observes every
/// optimum; the monotonic producers observe common correspondences only and
/// pass [`Self::UNOBSERVED`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct EndpointExtents {
    first: Option<ObservedWordTiming>,
    last: Option<ObservedWordTiming>,
}

impl EndpointExtents {
    /// The producer did not observe candidate sets.
    pub(super) const UNOBSERVED: Self = Self {
        first: None,
        last: None,
    };

    fn observations(&self) -> impl Iterator<Item = &ObservedWordTiming> {
        self.first.iter().chain(self.last.iter())
    }
}

/// Nonempty correspondence with both source utterance endpoints bounded:
/// proved common to every optimum, or matched in every optimum within a
/// known extent. Only census-bound admission constructs this timing
/// authority.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(transparent)]
pub struct EndpointBoundUtrWordMatches {
    matches: NonEmptyAdmittedUtrWordMatches,
    /// Extents of ambiguous endpoints; they widen the timing hull, and are
    /// recorded in it (the proposal), not as matches.
    #[serde(skip)]
    extents: EndpointExtents,
}

impl EndpointBoundUtrWordMatches {
    /// Endpoint authority survives placement: the same words, file-addressed.
    pub(super) fn placed_after(self, region_start: super::UtrUtteranceOrdinal) -> Self {
        Self {
            matches: self.matches.placed_after(region_start),
            extents: self.extents,
        }
    }

    /// Read the admitted words without discarding endpoint authority.
    pub fn iter(&self) -> impl Iterator<Item = &AdmittedUtrWordMatch> {
        self.matches.iter()
    }

    pub(super) fn token_extent(&self) -> (UtrAsrTokenOrdinal, UtrAsrTokenOrdinal) {
        self.matches.token_extent()
    }

    /// Timing remains bound to its original provider observations, and is
    /// available only after first/last correspondence has been bounded: the
    /// hull of the admitted matches and of any ambiguous endpoint's extent.
    pub(super) fn proposal(&self) -> UtrTimingProposal {
        UtrTimingProposal::from_observations(
            &self.matches.first.timing,
            self.matches
                .rest
                .iter()
                .map(|word| &word.timing)
                .chain(self.extents.observations()),
        )
    }
}

/// Which utterance endpoints lack common-to-every-optimum correspondence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MissingUtrEndpoints {
    /// The first alignable word is not proved.
    First,
    /// The last alignable word is not proved.
    Last,
    /// Neither endpoint is proved.
    Both,
}

pub(super) enum BoundaryAdmission {
    Bound(EndpointBoundUtrWordMatches),
    Interior {
        matches: NonEmptyAdmittedUtrWordMatches,
        missing: MissingUtrEndpoints,
    },
}

impl<'source> UtrLexicalStream<'source> {
    pub(super) fn from_tokens(tokens: &'source [AsrTimingToken]) -> Self {
        Self::project(tokens, |_| true)
    }

    /// The tokens an anchored region may match: onsets inside its window,
    /// projected with their original stream ordinals.
    pub(super) fn within_region(
        tokens: &'source [AsrTimingToken],
        region: &super::regions::UtrRegion,
    ) -> Self {
        Self::project(tokens, |token| region.admits(token))
    }

    /// Filter in provider coordinates before projecting words, preserving
    /// original token ordinals even in a local overlap-recovery window.
    pub(super) fn within_window(
        tokens: &'source [AsrTimingToken],
        start_ms: u64,
        end_ms: u64,
    ) -> Self {
        Self::project(tokens, |token| {
            token.start_ms < end_ms && token.end_ms > start_ms
        })
    }

    fn project(
        tokens: &'source [AsrTimingToken],
        include: impl Fn(&AsrTimingToken) -> bool,
    ) -> Self {
        let words = tokens
            .iter()
            .enumerate()
            .filter(|(_, token)| include(token))
            .flat_map(|(token_index, token)| {
                token.text.split_whitespace().enumerate().filter_map(
                    move |(word_index, surface)| {
                        UtrLexicalWord::from_provider_word(
                            surface,
                            UtrAsrTokenAddress {
                                token_index: UtrAsrTokenOrdinal(token_index),
                                word_index: UtrAsrWordOrdinal(word_index),
                            },
                        )
                    },
                )
            })
            .collect();
        Self { tokens, words }
    }

    pub(super) fn texts(&self) -> Vec<String> {
        self.words.iter().map(|word| word.text.to_owned()).collect()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// Cover every provider interval between extremal embeddings. Provider
    /// segments may bound a search but never become individual word anchors.
    /// Inverted, empty or reordered timing is not a search capability.
    pub(super) fn ordered_search(&self) -> Option<OrderedLexicalSearch<'_, 'source>> {
        let observed = self.onset_ordered_search()?;
        let mut previous = None;
        for word in &observed.0.words {
            let token = &observed.0.tokens[word.address.token_index()];
            if previous.is_some_and(|end| token.end_ms < end) {
                return None;
            }
            previous = Some(token.end_ms);
        }
        Some(OrderedLexicalSearch(self))
    }

    /// Overlapping words may have nested ends. Their onset order is sufficient
    /// for candidate-extrema corridors, but not for O(1) endpoint-only hulls.
    fn onset_ordered_search(&self) -> Option<OnsetOrderedLexicalSearch<'_, 'source>> {
        let mut previous = None;
        for word in &self.words {
            let token = &self.tokens[word.address.token_index()];
            if token.start_ms >= token.end_ms
                || previous.is_some_and(|start| token.start_ms < start)
            {
                return None;
            }
            previous = Some(token.start_ms);
        }
        Some(OnsetOrderedLexicalSearch(self))
    }

    /// Convert a matched lexical position into original provider evidence.
    pub(super) fn matched_word(
        &self,
        lexical_index: usize,
        word: UtrWordAddress,
        chat_text: &str,
    ) -> UtrWordMatch {
        let lexical = &self.words[lexical_index];
        UtrWordMatch {
            word,
            token: lexical.address,
            chat_text: chat_text.to_owned(),
            asr_text: lexical.surface.to_owned(),
            relation: lexical.relation(chat_text),
        }
    }

    /// Project the exact all-optima proof onto the original provider owners.
    pub(super) fn common_matches(
        &self,
        common: &CommonCorrespondences,
        payload: &[super::UtrPayloadWord],
    ) -> Vec<AdmittedUtrWordMatch> {
        common
            .pairs()
            .map(|(p, r)| {
                let word = &payload[p];
                self.admitted_word(r, word.address, &word.text)
            })
            .collect()
    }

    /// The earliest/latest embedding equality is the fast path's proof.
    pub(super) fn unique_embedding_matches(
        &self,
        payload: &[super::UtrPayloadWord],
        embedding: &super::UniqueEmbedding,
    ) -> Vec<AdmittedUtrWordMatch> {
        payload
            .iter()
            .zip(&embedding.indices)
            .map(|(word, &r)| self.admitted_word(r, word.address, &word.text))
            .collect()
    }

    fn admitted_word(
        &self,
        index: usize,
        word: UtrWordAddress,
        text: &str,
    ) -> AdmittedUtrWordMatch {
        let matched = self.matched_word(index, word, text);
        let token = &self.tokens[matched.token.token_index()];
        let timing = ObservedWordTiming {
            start_ms: token.start_ms,
            end_ms: token.end_ms,
        };
        let word_timing = (token.text.split_whitespace().count() == 1).then_some(timing);
        AdmittedUtrWordMatch {
            matched,
            word_timing,
            timing,
        }
    }

    /// The lexical producer owns reference projection, joint proof and its
    /// materialization together. Consumers cannot pair a proof with another
    /// provider stream or authorize hints from the diagnostic local relaxation.
    pub(super) fn plan_interleaving<'c>(
        &self,
        searched: super::regions::SearchedCensus<'c>,
        mode: batchalign_transform::dp_align::MatchMode,
        participation: super::GlobalUtrParticipation,
        contested: &super::ContestedTokens,
    ) -> super::evidence::LocalRegionPlan<'c> {
        use batchalign_transform::dp_align::interleaving::{LocalInterleaving, SpeakerTurn};
        let source = searched.infos();
        let region = searched.span();
        let payload = source
            .iter()
            .enumerate()
            .filter(|(_, info)| !info.excluded_from(participation))
            .flat_map(|(utterance, info)| {
                info.words
                    .iter()
                    .enumerate()
                    .map(move |(word, text)| super::UtrPayloadWord {
                        text: text.clone(),
                        address: UtrWordAddress {
                            utterance_index: super::UtrUtteranceOrdinal(utterance),
                            word_index: super::UtrWordOrdinal(word),
                        },
                    })
            })
            .collect::<Vec<_>>();
        let reference = self.texts();
        let turns: Vec<_> = source
            .iter()
            .map(|info| {
                let words = if info.excluded_from(participation) {
                    &[][..]
                } else {
                    &info.words[..]
                };
                if info.has_lazy_overlap || info.has_ca_overlap {
                    SpeakerTurn::overlapping_continuation(&info.speaker, words)
                } else {
                    SpeakerTurn::new(&info.speaker, words)
                }
            })
            .collect();
        let analysis = match LocalInterleaving::observe(&turns, &reference, mode) {
            Ok(analysis) => analysis,
            Err(refusal) => {
                let refusal = super::UtrBudgetRefusal {
                    region,
                    budget: super::UtrSearchBudget::interleaved(refusal),
                };
                tracing::warn!(
                    region = %region.describe(),
                    budget = ?refusal.budget,
                    "joint UTR correspondence proof refused for this region; \
                     other regions are unaffected and no monotonic fallback is taken"
                );
                let selection = super::UtrMatchSelection {
                    selected: Vec::new(),
                    admission: super::CorrespondenceProof::BudgetExhausted(refusal),
                    per_utterance: searched.map(|_| super::UtteranceSelection::UNSEARCHED),
                };
                return super::build_alignment_plan(
                    super::UtrAlignmentStrategy::LocalInterleaving,
                    &payload,
                    self,
                    selection,
                    participation,
                    contested,
                );
            }
        };
        let mut positions = source
            .iter()
            .map(|info| vec![None; info.words.len()])
            .collect::<Vec<_>>();
        for (index, word) in payload.iter().enumerate() {
            positions[word.address.utterance_index.index()][word.address.word_index.0] =
                Some(index);
        }
        let selected = analysis
            .selected()
            .filter_map(|matched| {
                positions[matched.utterance_index()][matched.word_index()]
                    .map(|index| (index, matched.reference_index()))
            })
            .collect();
        let mut admitted = Vec::new();
        let mut ranges = vec![None::<ObservedWordTiming>; source.len()];
        let mut complete = vec![true; source.len()];
        let mut seen = vec![0usize; source.len()];
        let mut extents = vec![EndpointExtents::UNOBSERVED; source.len()];
        for (evidence, word) in analysis.words().zip(&payload) {
            let ordinal = word.address.utterance_index.index();
            seen[ordinal] += 1;
            // An endpoint matched in every optimum, if not to one token, is
            // still bounded: by the extent of every token it may match. A
            // contested token (claimed across an anchor by another region) is
            // withdrawn here too: it bounds nothing, so the endpoint does not
            // either, and the withdrawal cannot come back through the hull.
            let index = word.address.word_index.0;
            let last = source[ordinal].words.len().saturating_sub(1);
            if (index == 0 || index == last) && !evidence.can_be_missing() {
                let extent = evidence.candidates().try_fold(None, |extent, matched| {
                    let lexical = &self.words[matched.reference_index()];
                    let token = &self.tokens[lexical.address.token_index()];
                    let usable = token.start_ms < token.end_ms
                        && token.text.split_whitespace().count() == 1
                        && matched.source_text() == word.text
                        && !contested.contains(lexical.address.token_index);
                    usable.then(|| {
                        Some(extent.map_or(
                            ObservedWordTiming {
                                start_ms: token.start_ms,
                                end_ms: token.end_ms,
                            },
                            |old: ObservedWordTiming| ObservedWordTiming {
                                start_ms: old.start_ms.min(token.start_ms),
                                end_ms: old.end_ms.max(token.end_ms),
                            },
                        ))
                    })
                });
                if let Some(Some(extent)) = extent {
                    if index == 0 {
                        extents[ordinal].first = Some(extent);
                    }
                    if index == last {
                        extents[ordinal].last = Some(extent);
                    }
                }
            }
            if let Some(common) = evidence.common() {
                let matched = common.matched();
                admitted.push(self.admitted_word(
                    matched.reference_index(),
                    word.address,
                    matched.source_text(),
                ));
            }
            let mut candidates = evidence.candidates().peekable();
            if evidence.can_be_missing() || candidates.peek().is_none() {
                complete[ordinal] = false;
                continue;
            }
            for matched in candidates {
                let lexical = &self.words[matched.reference_index()];
                let token = &self.tokens[lexical.address.token_index()];
                if token.start_ms >= token.end_ms || matched.source_text() != word.text {
                    complete[ordinal] = false;
                    continue;
                }
                let interval = ObservedWordTiming {
                    start_ms: token.start_ms,
                    end_ms: token.end_ms,
                };
                ranges[ordinal] =
                    Some(ranges[ordinal].map_or(interval, |old| ObservedWordTiming {
                        start_ms: old.start_ms.min(interval.start_ms),
                        end_ms: old.end_ms.max(interval.end_ms),
                    }));
            }
        }
        let mut floors = vec![0; source.len()];
        let mut previous = std::collections::BTreeMap::<&str, u64>::new();
        for (ordinal, info) in source.iter().enumerate() {
            floors[ordinal] = previous.get(info.speaker.as_str()).copied().unwrap_or(0);
            if !info.has_lazy_overlap
                && !info.has_ca_overlap
                && let Some(timing) = info.retained_timing
            {
                previous
                    .entry(info.speaker.as_str())
                    .and_modify(|end| *end = (*end).max(timing.end_ms))
                    .or_insert(timing.end_ms);
            }
        }
        let mut ceilings = vec![None; source.len()];
        let mut following = std::collections::BTreeMap::<&str, u64>::new();
        for (ordinal, info) in source.iter().enumerate().rev() {
            ceilings[ordinal] = following.get(info.speaker.as_str()).copied();
            if !info.has_lazy_overlap
                && !info.has_ca_overlap
                && let Some(timing) = info.retained_timing
            {
                following
                    .entry(info.speaker.as_str())
                    .and_modify(|start| *start = (*start).min(timing.start_ms))
                    .or_insert(timing.start_ms);
            }
        }
        let onset_ordered_search = self.onset_ordered_search();
        let envelope = |ordinal: usize, info: &super::UtrUtteranceInfo| {
            if seen[ordinal] != info.words.len()
                || info.words.is_empty()
                || info.retained_timing.is_some()
                || info.excluded_from(participation)
            {
                return None;
            }
            let (floor, ceiling) = if info.has_lazy_overlap || info.has_ca_overlap {
                (0, None)
            } else {
                (floors[ordinal], ceilings[ordinal])
            };
            if !complete[ordinal] {
                // Missing source words cannot inherit the crop of adjacent
                // document-order turns: those may interleave with this one.
                // Only the joint producer knows which obligatory words
                // precede/follow this source in EVERY legal composition,
                // including preserved order on the same speaker's chain.
                let ordered = onset_ordered_search.as_ref()?;
                let corridor = analysis.search_corridor(ordinal)?;
                let candidate_bounds = |required| ordered.required_bounds(required);
                let before = corridor
                    .preceding_required()
                    .filter_map(candidate_bounds)
                    .map(|b| b.preceding_floor_ms)
                    .max()
                    .unwrap_or(0);
                let after = corridor
                    .following_required()
                    .filter_map(candidate_bounds)
                    .map(|b| b.following_ceiling_ms)
                    .min();
                let ceiling = match (ceiling, after) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (a, b) => a.or(b),
                };
                return Some(super::search::FaSearchEnvelope::order_corridor(
                    info.words.clone(),
                    floor.max(before),
                    ceiling,
                ));
            }
            let range = ranges[ordinal]?;
            super::search::FaSearchEnvelope::interleaved(
                info.words.clone(),
                range.start_ms,
                range.end_ms,
                floor,
                ceiling,
            )
        };
        let selection = super::UtrMatchSelection {
            selected,
            admission: super::CorrespondenceProof::Complete(admitted),
            per_utterance: searched.map(|(ordinal, info)| super::UtteranceSelection {
                envelope: envelope(ordinal, info),
                endpoints: extents[ordinal],
            }),
        };
        super::build_alignment_plan(
            super::UtrAlignmentStrategy::LocalInterleaving,
            &payload,
            self,
            selection,
            participation,
            contested,
        )
    }
}

/// Producer-admitted provider ordering; each envelope lookup is constant time.
/// Validating once avoids rescanning overlapping ambiguous ranges per turn.
pub(super) struct OrderedLexicalSearch<'stream, 'source>(&'stream UtrLexicalStream<'source>);

/// Positive provider intervals in onset order, retaining legitimate nested
/// ends. Constructed only by the provider observer, not the FA consumer.
struct OnsetOrderedLexicalSearch<'stream, 'source>(&'stream UtrLexicalStream<'source>);

impl OnsetOrderedLexicalSearch<'_, '_> {
    fn required_bounds(
        &self,
        required: batchalign_transform::dp_align::interleaving::RequiredLocalWord<'_, '_>,
    ) -> Option<RequiredWordSearchBounds> {
        required
            .candidates()
            .try_fold(None::<RequiredWordSearchBounds>, |range, matched| {
                let lexical = self.0.words.get(matched.reference_index())?;
                let token = &self.0.tokens[lexical.address.token_index()];
                if token.text.split_whitespace().count() != 1 {
                    return None;
                }
                let bounds = RequiredWordSearchBounds {
                    preceding_floor_ms: token.start_ms,
                    following_ceiling_ms: token.end_ms,
                };
                Some(Some(range.map_or(bounds, |old| RequiredWordSearchBounds {
                    preceding_floor_ms: old.preceding_floor_ms.min(token.start_ms),
                    following_ceiling_ms: old.following_ceiling_ms.max(token.end_ms),
                })))
            })
            .flatten()
    }
}

impl OrderedLexicalSearch<'_, '_> {
    pub(super) fn interval(&self, first: usize, last: usize) -> Option<(u64, u64)> {
        if first > last {
            return None;
        }
        let start = &self.0.tokens[self.0.words.get(first)?.address.token_index()];
        let end = &self.0.tokens[self.0.words.get(last)?.address.token_index()];
        Some((start.start_ms, end.end_ms))
    }
}
