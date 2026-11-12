//! Dispatch units: how FA groups become worker requests, and how the
//! requests' timings become group timings again.
//!
//! # Why groups and requests are different things
//!
//! A group (`chat_ops::fa::FaGroup`) is the unit of INJECTION: its timings
//! are written back by walking its words with one cursor. A request is the
//! unit of EXECUTION: one window of audio, within the engine budget, and the
//! words aligned against it. For an ordinary group the two coincide. For an
//! anchored group (one over-budget utterance split at recovered word anchors)
//! they do not: it is one group and requests only for admitted pieces.
//! Unavailable pieces own their original words but have no request or cache key.
//!
//! So everything that is about a REQUEST (cache keys, cache admission, worker
//! dispatch, raw evidence, fallback) is per [`DispatchUnit`], and everything
//! that is about the TRANSCRIPT (`%wor` reuse, injection, post-processing)
//! stays per group. [`UnitLedger::assemble`] is the one place unit timings
//! become group timings: full contextual replies are admitted against their
//! request, then only owned words are concatenated, with explicit untimed slots
//! for unavailable ranges. Context words are never injected twice.
//!
//! ```mermaid
//! flowchart LR
//!     groups["groups (FaGroup)"] -->|"DispatchPlan::build"| plan["one PlannedGroup per group,<br/>its units by shape"]
//!     plan -->|"UnitLedger::open"| ledger["per group: Reused, or<br/>one receiving slot per request"]
//!     plan -->|"mint once with slot"| ticket["unique ReplyTicket"]
//!     ticket -->|"cache hit, or prepared worker reply"| retained["consume ticket; complete its slot"]
//!     retained --> ledger
//!     ledger -->|"assemble"| timings["group timings, word order"]
//!     timings --> inject["injection (unchanged)"]
//! ```
//!
//! # The shape is the index
//!
//! The plan holds each group's units in the group's own shape (one unit, or
//! the pieces), and the ledger holds each group's resolution in the same
//! shape: a group reused from `%wor` is `Reused` and has no request slots at
//! all, and every request slot is paired with the unit it awaits. Assembly
//! therefore walks pairs, never parallel arrays. Each request's unique ticket
//! owns its receiving destination through cache lookup and inference. No free
//! ledger/unit pair, ordinal lookup, or duplicate insertion is available.

mod replies;
mod resolution;
pub(in crate::fa) use replies::ReplyTicket;
pub(in crate::fa) use replies::UnitResolution;
use replies::{ReplyProgress, ReplySlot};
pub(in crate::fa) use resolution::UnitAnswer;

mod inference;
use inference::RetainUnitResults;

use std::collections::HashSet;

mod cache;
pub(super) use cache::{
    CandidateCacheAvailability, CandidateCacheReceipts, CandidateCacheState,
    collect_candidate_cache,
};

use talkbank_model::WordIdx;
use tracing::{info, warn};

use super::preparation::{FaPreparationContext, PreparedFaPopulation};
use super::transport::{FaInferencePlan, FaWorkerBatch, FaWorkerTransport, plan_fa_inference};
use super::{
    AdmittedFaCacheGroup, FaCacheUnitAdmission, incremental::collect_preserved_group_timings,
};
use crate::api::DurationMs;
use crate::chat_ops::CacheKey;
use crate::chat_ops::fa::coordinates::FaWindow;
use crate::chat_ops::fa::resource_partition::{ResourcePart, ResourceUnavailable};
use crate::chat_ops::fa::{
    AnchoredPart, AnchoredPiece, FaGroup, FaWord, GroupUnit, GroupUnits, UnavailablePiece,
    WordTiming, cache_key,
};
use crate::error::ServerError;
use crate::runner::util::{FileStage, ProgressSender, ProgressUpdate};
use crate::types::results::{FaGroupContents, FaGroupEvidence};
use crate::types::traces::{
    FaEvidenceSourceTrace, FaFallbackEventTrace, FaGroupSpanTrace, FaPieceTrace, TimingTrace,
};

/// A group's position in grouping's output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct GroupOrdinal(usize);

impl GroupOrdinal {
    /// The ordinal as evidence and logs spell it.
    pub(super) fn index(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for GroupOrdinal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A request's position in one file's [`DispatchPlan`], for request ids,
/// logs and evidence. Minted only by [`DispatchPlan::build`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct UnitOrdinal(usize);

impl UnitOrdinal {
    /// The ordinal as evidence, request ids and logs spell it.
    pub(crate) fn index(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for UnitOrdinal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Source-owned request capability. A piece keeps context, owned output range
/// and window together, so dispatch cannot combine independently chosen parts.
#[derive(Debug, Clone, Copy)]
enum UnitRequest<'g> {
    Whole(GroupUnit<'g>),
    Piece(&'g AnchoredPiece),
}

impl<'g> UnitRequest<'g> {
    fn words(self) -> &'g [FaWord] {
        match self {
            Self::Whole(request) => request.words(),
            Self::Piece(piece) => piece.request_words(),
        }
    }

    fn window(self) -> FaWindow {
        match self {
            Self::Whole(request) => request.window(),
            Self::Piece(piece) => piece.window(),
        }
    }
}

/// One FA request: some consecutive words of one group, the window they are
/// aligned against, and the request's cache identity.
#[derive(Debug)]
pub(crate) struct DispatchUnit<'g> {
    ordinal: UnitOrdinal,
    group: GroupOrdinal,
    request: UnitRequest<'g>,
    cache_key: CacheKey,
}

impl<'g> DispatchUnit<'g> {
    pub(super) fn supplied(
        source: &'g crate::chat_ops::fa::SuppliedRetrySource,
        context: FaPreparationContext<'_>,
    ) -> Self {
        Self::keyed(
            UnitOrdinal(0),
            GroupOrdinal(0),
            UnitRequest::Whole(source.request()),
            &RequestKeying::of(&context),
        )
    }
    /// Key one request and place it in the plan.
    fn keyed(
        ordinal: UnitOrdinal,
        group: GroupOrdinal,
        request: UnitRequest<'g>,
        key: &RequestKeying<'_>,
    ) -> Self {
        let texts: Vec<String> = request
            .words()
            .iter()
            .map(|word| word.text.clone())
            .collect();
        let cache_key = key.key(&texts, request.window());
        Self {
            ordinal,
            group,
            request,
            cache_key,
        }
    }

    /// This unit's position in its plan.
    pub(crate) fn ordinal(&self) -> UnitOrdinal {
        self.ordinal
    }

    /// The group whose words this unit aligns.
    pub(super) fn group(&self) -> GroupOrdinal {
        self.group
    }

    /// The words this request aligns, in order.
    pub(crate) fn words(&self) -> &'g [FaWord] {
        self.request.words()
    }

    pub(super) fn input(&self) -> GroupUnit<'g> {
        match self.request {
            UnitRequest::Whole(request) => request,
            UnitRequest::Piece(piece) => GroupUnit::from_piece(piece),
        }
    }

    /// The window this request aligns against, within the engine budget.
    pub(crate) fn window(&self) -> FaWindow {
        self.request.window()
    }

    /// The request's content-addressed cache identity.
    pub(crate) fn cache_key(&self) -> &CacheKey {
        &self.cache_key
    }
}

/// The run facts every request's cache key is computed from.
/// Everything an FA request's cache key depends on besides its words and
/// window: the one place a key is computed ([`Self::key`]), so the key
/// formula cannot drift between the preparation, candidate and dispatch
/// sites that each used to spell it out.
pub(super) struct RequestKeying<'a> {
    audio_identity: &'a crate::chat_ops::fa::AudioIdentity,
    whisper_text: crate::chat_ops::fa::WhisperTextMode,
    engine: crate::types::engines::FaEngineName,
    language: crate::chat_ops::fa::FaKeyLanguage,
}

impl<'a> RequestKeying<'a> {
    /// The keying for `engine` aligning `audio_identity` in `language`.
    pub(super) fn new(
        audio_identity: &'a crate::chat_ops::fa::AudioIdentity,
        whisper_text: crate::chat_ops::fa::WhisperTextMode,
        engine: crate::types::engines::FaEngineName,
        language: &crate::api::LanguageCode3,
    ) -> Self {
        Self {
            audio_identity,
            whisper_text,
            engine,
            language: crate::chat_ops::fa::FaKeyLanguage::for_engine(engine, language),
        }
    }

    /// The keying of one preparation context.
    pub(super) fn of(context: &FaPreparationContext<'a>) -> Self {
        Self::new(
            context.audio.audio_identity,
            context.params.whisper_text,
            context.params.engine,
            context.worker_lang,
        )
    }

    /// The cache key of `words` aligned in `window`.
    pub(super) fn key(
        &self,
        words: &[String],
        window: crate::chat_ops::fa::coordinates::FaWindow,
    ) -> CacheKey {
        cache_key(
            words,
            self.audio_identity,
            window.audio_start().get(),
            window.end().get(),
            self.whisper_text,
            self.engine,
            &self.language,
        )
    }
}

/// One piece's request and the words it starts and ends at, for evidence.
#[derive(Debug)]
struct PlannedPiece<'g> {
    unit: DispatchUnit<'g>,
    first_word: WordIdx,
    last_word: WordIdx,
}

/// One original word range, either an actual request or an explicit non-request.
#[derive(Debug)]
enum PlannedPart<'g> {
    Request(PlannedPiece<'g>),
    Unavailable(Unrequested<'g>),
}

/// Historical declared-capacity evidence and actual resource refusals remain
/// distinct. Neither variant owns a request slot or an execution capability.
#[derive(Debug, Clone, Copy)]
enum Unrequested<'g> {
    Declared(&'g UnavailablePiece),
    Resource(&'g ResourceUnavailable),
}

impl Unrequested<'_> {
    fn words(&self) -> &[FaWord] {
        match self {
            Self::Declared(piece) => piece.words(),
            Self::Resource(piece) => piece.words(),
        }
    }
    fn first_word(&self) -> WordIdx {
        match self {
            Self::Declared(piece) => piece.first_word(),
            Self::Resource(piece) => piece.first_word(),
        }
    }
    fn last_word(&self) -> WordIdx {
        match self {
            Self::Declared(piece) => piece.last_word(),
            Self::Resource(piece) => piece.last_word(),
        }
    }
    fn window_trace(&self) -> crate::types::traces::RefusedWindowTrace {
        match self {
            Self::Declared(piece) => piece.refusal().into(),
            Self::Resource(piece) => crate::types::traces::RefusedWindowTrace::Resource {
                start_ms: piece.window().audio_start().get(),
                end_ms: piece.window().end().get(),
                constraint: piece.constraint().clone(),
            },
        }
    }
}

enum PartInput<'g> {
    Request(&'g AnchoredPiece),
    Unavailable(Unrequested<'g>),
}

fn plan_parts<'g>(
    parts: impl Iterator<Item = PartInput<'g>>,
    mut mint: impl FnMut() -> UnitOrdinal,
    group: GroupOrdinal,
    key: &RequestKeying<'_>,
) -> PlannedUnits<'g> {
    PlannedUnits::Pieces(
        parts
            .map(|part| match part {
                PartInput::Request(piece) => PlannedPart::Request(PlannedPiece {
                    unit: DispatchUnit::keyed(mint(), group, UnitRequest::Piece(piece), key),
                    first_word: piece.first_word(),
                    last_word: piece.last_word(),
                }),
                PartInput::Unavailable(piece) => PlannedPart::Unavailable(piece),
            })
            .collect(),
    )
}

/// A group's requests, in the group's own shape.
#[derive(Debug)]
enum PlannedUnits<'g> {
    /// A single group's one request.
    Whole(DispatchUnit<'g>),
    /// An anchored group's pieces, in word order.
    Pieces(Vec<PlannedPart<'g>>),
}

/// One group and its requests.
#[derive(Debug)]
struct PlannedGroup<'g> {
    group: &'g FaGroup,
    units: PlannedUnits<'g>,
}

impl<'g> PlannedGroup<'g> {
    /// The group's requests, in word order.
    fn units(&self) -> impl Iterator<Item = &DispatchUnit<'g>> {
        let (whole, pieces) = match &self.units {
            PlannedUnits::Whole(unit) => (Some(unit), [].iter()),
            PlannedUnits::Pieces(pieces) => (None, pieces.iter()),
        };
        whole
            .into_iter()
            .chain(pieces.filter_map(|part| match part {
                PlannedPart::Request(piece) => Some(&piece.unit),
                PlannedPart::Unavailable(_) => None,
            }))
    }
}

/// Every request one file's groups are executed as, held per group in each
/// group's own shape.
#[derive(Debug)]
pub(crate) struct DispatchPlan<'g> {
    groups: Vec<PlannedGroup<'g>>,
}

impl<'g> DispatchPlan<'g> {
    /// Lay out every group's requests and key each one.
    ///
    /// A single group is keyed by its own words and window, the same inputs
    /// its cache entries are written under, so they stay reachable.
    pub(super) fn build(groups: &'g [FaGroup], key: RequestKeying<'_>) -> Self {
        let mut next_ordinal = 0usize;
        let mut mint = || {
            let ordinal = UnitOrdinal(next_ordinal);
            next_ordinal += 1;
            ordinal
        };
        let groups = groups
            .iter()
            .enumerate()
            .map(|(index, group)| {
                let ordinal = GroupOrdinal(index);
                let units = match group.units() {
                    GroupUnits::Whole(request) => PlannedUnits::Whole(DispatchUnit::keyed(
                        mint(),
                        ordinal,
                        UnitRequest::Whole(request),
                        &key,
                    )),
                    GroupUnits::Pieces(pieces) => plan_parts(
                        pieces.map(|part| match part {
                            AnchoredPart::Request(piece) => PartInput::Request(piece),
                            AnchoredPart::Unavailable(piece) => {
                                PartInput::Unavailable(Unrequested::Declared(piece))
                            }
                        }),
                        &mut mint,
                        ordinal,
                        &key,
                    ),
                    GroupUnits::Resource(pieces) => plan_parts(
                        pieces.map(|part| match part {
                            ResourcePart::Request(piece) => PartInput::Request(piece),
                            ResourcePart::Unavailable(piece) => {
                                PartInput::Unavailable(Unrequested::Resource(piece))
                            }
                        }),
                        &mut mint,
                        ordinal,
                        &key,
                    ),
                };
                PlannedGroup { group, units }
            })
            .collect();
        Self { groups }
    }

    /// Every unit, in plan order.
    pub(crate) fn units(&self) -> impl Iterator<Item = &DispatchUnit<'g>> {
        self.groups.iter().flat_map(PlannedGroup::units)
    }
}

/// How one group is being resolved, in the group's own shape.
enum GroupResolution<'p, 'g> {
    /// Reused whole from `%wor`: no request is awaited.
    Reused(Vec<Option<WordTiming>>),
    /// A single group's one request.
    Whole(ReplySlot<'p, 'g>),
    /// An anchored group's pieces, each paired with its bounds.
    Pieces(Vec<PartResolution<'p, 'g>>),
}

/// An unavailable part has no reply slot and cannot receive a worker result.
enum PartResolution<'p, 'g> {
    Request {
        slot: ReplySlot<'p, 'g>,
        piece: &'p PlannedPiece<'g>,
    },
    Unavailable(&'p Unrequested<'g>),
}

/// What has been resolved so far, one entry per group in plan order, each
/// paired with the planned group it resolves. Consumed by
/// [`UnitLedger::assemble`].
pub(super) struct UnitLedger<'p, 'g> {
    groups: Vec<(&'p PlannedGroup<'g>, GroupResolution<'p, 'g>)>,
}

pub(super) struct OpenedLedger<'p, 'g> {
    pub(super) ledger: UnitLedger<'p, 'g>,
    pub(super) replies: Vec<ReplyTicket<'p>>,
    progress: ReplyProgress,
}

impl<'p, 'g> UnitLedger<'p, 'g> {
    /// Decide whole-group reuse and mint each awaited slot with its only
    /// completion ticket. Reused and unavailable populations mint none.
    pub(super) fn open(
        plan: &'p DispatchPlan<'g>,
        mut reuse: impl FnMut(&FaGroup) -> Option<Vec<Option<WordTiming>>>,
    ) -> OpenedLedger<'p, 'g> {
        let progress = ReplyProgress::for_plan(plan);
        let mut replies = Vec::new();
        let groups = plan
            .groups
            .iter()
            .map(|planned| {
                let resolution = match reuse(planned.group) {
                    Some(timings) => {
                        progress.reused(planned);
                        GroupResolution::Reused(timings)
                    }
                    None => match &planned.units {
                        PlannedUnits::Whole(unit) => {
                            let (slot, reply) = ReplySlot::open(unit, progress.clone());
                            replies.push(reply);
                            GroupResolution::Whole(slot)
                        }
                        PlannedUnits::Pieces(pieces) => GroupResolution::Pieces(
                            pieces
                                .iter()
                                .map(|part| match part {
                                    PlannedPart::Request(piece) => {
                                        let (slot, reply) =
                                            ReplySlot::open(&piece.unit, progress.clone());
                                        replies.push(reply);
                                        PartResolution::Request { slot, piece }
                                    }
                                    PlannedPart::Unavailable(piece) => {
                                        PartResolution::Unavailable(piece)
                                    }
                                })
                                .collect(),
                        ),
                    },
                };
                (planned, resolution)
            })
            .collect();
        OpenedLedger {
            ledger: Self { groups },
            replies,
            progress,
        }
    }

    /// How many requests belong to groups reused from `%wor`.
    fn reused_units(&self) -> usize {
        self.groups
            .iter()
            .map(|(planned, resolution)| match resolution {
                GroupResolution::Reused(_) => planned.units().count(),
                GroupResolution::Whole(_) | GroupResolution::Pieces(_) => 0,
            })
            .sum()
    }

    /// Turn every group's resolution into its timings, in word order, with
    /// its evidence. THE place units become groups.
    ///
    /// A reused group takes its timings whole, and each of its requests is
    /// recorded as `wor_reuse`; any other group concatenates its requests'
    /// timings in piece order, which is word order. Every awaited request
    /// must have been resolved, and must have answered for exactly its own
    /// words, since injection walks the group's words with one cursor: a
    /// short or missing piece would shift every timing after it.
    pub(super) fn assemble(self, context: &str) -> Result<Vec<ResolvedGroup>, ServerError> {
        let mut missing: Vec<UnitOrdinal> = Vec::new();
        let mut resolved = Vec::with_capacity(self.groups.len());
        for (planned, resolution) in self.groups {
            let (timings, span) = match resolution {
                GroupResolution::Reused(timings) => {
                    let span = match &planned.units {
                        PlannedUnits::Whole(unit) => {
                            single_span(unit, FaEvidenceSourceTrace::WorReuse)
                        }
                        PlannedUnits::Pieces(pieces) => FaGroupSpanTrace::Anchored {
                            pieces: pieces
                                .iter()
                                .filter_map(|part| match part {
                                    PlannedPart::Request(piece) => {
                                        Some(piece_trace(piece, FaEvidenceSourceTrace::WorReuse))
                                    }
                                    PlannedPart::Unavailable(_) => None,
                                })
                                .collect(),
                            unrequested: pieces
                                .iter()
                                .filter_map(|part| match part {
                                    PlannedPart::Request(_) => None,
                                    PlannedPart::Unavailable(piece) => Some(unrequested_trace(
                                        piece,
                                        crate::types::traces::UnrequestedRangeSource::WorReuse,
                                    )),
                                })
                                .collect(),
                        },
                    };
                    (timings, span)
                }
                GroupResolution::Whole(slot) => {
                    let (unit, answer) = slot.receive();
                    let Some(answer) = answer else {
                        missing.push(unit.ordinal);
                        continue;
                    };
                    let (timings, source) = owned_timings(unit, answer);
                    (timings, single_span(unit, source))
                }
                GroupResolution::Pieces(pieces) => {
                    let mut timings = Vec::new();
                    let mut traces = Vec::with_capacity(pieces.len());
                    let mut unrequested = Vec::new();
                    let missing_before = missing.len();
                    for part in pieces {
                        match part {
                            PartResolution::Unavailable(piece) => {
                                timings.extend(std::iter::repeat_n(None, piece.words().len()));
                                unrequested.push(unrequested_trace(
                                    piece,
                                    crate::types::traces::UnrequestedRangeSource::Unaligned,
                                ));
                            }
                            PartResolution::Request { slot, piece } => {
                                let (unit, answer) = slot.receive();
                                match answer {
                                    None => missing.push(unit.ordinal),
                                    Some(answer) => {
                                        let (owned, source) = owned_timings(unit, answer);
                                        traces.push(piece_trace(piece, source));
                                        timings.extend(owned);
                                    }
                                }
                            }
                        }
                    }
                    // A group with an unresolved request has nothing to
                    // assemble; the missing list below names every such
                    // request and fails the file.
                    if missing.len() > missing_before {
                        continue;
                    }
                    (
                        timings,
                        FaGroupSpanTrace::Anchored {
                            pieces: traces,
                            unrequested,
                        },
                    )
                }
            };
            resolved.push(ResolvedGroup {
                group: planned.group.clone(),
                contents: group_contents(planned.group, span),
                timings,
            });
        }
        match missing.is_empty() {
            true => Ok(resolved),
            false => Err(ServerError::Validation(format!(
                "{context} completed without timings for request(s): {:?}",
                missing.iter().map(|unit| unit.0).collect::<Vec<_>>()
            ))),
        }
    }
}

/// The timings a resolution contributes to its group: all of a whole
/// group's, or only a piece's owned words (its borrowed context words are
/// another piece's). The resolution was bound to this request when it was
/// made, so there is nothing left to refuse.
fn owned_timings(
    unit: &DispatchUnit<'_>,
    answer: UnitResolution,
) -> (Vec<Option<WordTiming>>, FaEvidenceSourceTrace) {
    let (timings, source) = answer.into_parts();
    let owned = match unit.request {
        UnitRequest::Whole(_) => timings,
        UnitRequest::Piece(piece) => {
            let owned = piece.owned_range();
            timings
                .into_iter()
                .skip(owned.start)
                .take(owned.len())
                .collect()
        }
    };
    (owned, source)
}

/// A single group's evidence span.
fn single_span(unit: &DispatchUnit<'_>, source: FaEvidenceSourceTrace) -> FaGroupSpanTrace {
    FaGroupSpanTrace::Single {
        source,
        cache_key: unit.cache_key.as_str().to_owned(),
    }
}

/// One piece's evidence.
fn piece_trace(piece: &PlannedPiece<'_>, source: FaEvidenceSourceTrace) -> FaPieceTrace {
    FaPieceTrace {
        start_ms: piece.unit.window().audio_start().get(),
        end_ms: piece.unit.window().end().get(),
        first_word: piece.first_word.raw(),
        last_word: piece.last_word.raw(),
        source,
        cache_key: piece.unit.cache_key.as_str().to_owned(),
        context: match piece.unit.request {
            UnitRequest::Piece(source) => Some(crate::types::traces::FaPieceContextTrace {
                first_word: source.request_words()[0].utterance_word_index.raw(),
                last_word: source.request_words()[source.request_words().len() - 1]
                    .utterance_word_index
                    .raw(),
                ownership_start_ms: source.ownership_window().audio_start().get(),
                ownership_end_ms: source.ownership_window().end().get(),
            }),
            UnitRequest::Whole(_) => None,
        },
    }
}

fn unrequested_trace(
    piece: &Unrequested<'_>,
    source: crate::types::traces::UnrequestedRangeSource,
) -> crate::types::traces::FaUnrequestedPieceTrace {
    crate::types::traces::FaUnrequestedPieceTrace {
        first_word: piece.first_word().raw(),
        last_word: piece.last_word().raw(),
        window: piece.window_trace(),
        source,
    }
}

/// The evidence of one group: what it holds and how it was executed.
fn group_contents(group: &FaGroup, span: FaGroupSpanTrace) -> FaGroupContents {
    FaGroupContents {
        audio_start_ms: DurationMs(group.audio_start_ms()),
        audio_end_ms: DurationMs(group.audio_end_ms()),
        utterances: group.utterance_indices(),
        words: group.words().map(|w| w.text.clone()).collect(),
        word_ids: group.words().map(|word| word.stable_id()).collect(),
        source_windows: group.source_windows(),
        span,
    }
}

/// One group's assembled timings and its evidence.
#[derive(Debug)]
pub(super) struct ResolvedGroup {
    group: FaGroup,
    timings: Vec<Option<WordTiming>>,
    contents: FaGroupContents,
}

/// Everything resolution produced for a file's groups.
pub(super) struct ResolvedGroups {
    groups: Vec<ResolvedGroup>,
    /// Engine fallbacks taken by any request.
    pub(super) fallback_events: Vec<FaFallbackEventTrace>,
}

struct ProjectionParts {
    groups: Vec<FaGroup>,
    timings: Vec<Vec<Option<WordTiming>>>,
    evidence: Vec<FaGroupEvidence>,
    fallback_events: Vec<FaFallbackEventTrace>,
}

impl ResolvedGroups {
    /// Split into the timings injection consumes (one list per group, in
    /// group order) and the evidence the trace records, whose pre-injection
    /// snapshot is taken from those very timings.
    fn into_parts(self) -> ProjectionParts {
        let mut groups = Vec::with_capacity(self.groups.len());
        let mut timings = Vec::with_capacity(self.groups.len());
        let mut evidence = Vec::with_capacity(self.groups.len());
        for resolved in self.groups {
            let pre_injection_timings = resolved
                .timings
                .iter()
                .map(|timing| timing.as_ref().map(TimingTrace::from_word_timing))
                .collect();
            groups.push(resolved.group);
            timings.push(resolved.timings);
            evidence.push(FaGroupEvidence {
                group: resolved.contents,
                pre_injection_timings,
            });
        }
        ProjectionParts {
            groups,
            timings,
            evidence,
            fallback_events: self.fallback_events,
        }
    }
}

/// Everything resolution needs about the file and the run.
pub(super) struct FaDispatchInputs<'a, 'document> {
    /// Resource-planned groups and their matching immutable worker inputs.
    pub(super) population: PreparedFaPopulation<'a>,
    /// The document the groups were built from, for `%wor` reuse.
    pub(super) chat_file: &'document crate::chat_ops::ChatFile,
    /// Utterances whose existing `%wor` timing may be reused; a group is
    /// reused only when every one of its utterances is in this set.
    pub(super) reusable_utterances: &'document HashSet<usize>,
    pub(super) recovery: &'a [std::sync::Arc<super::input::RetainedUtrPass>],
    /// Progress sink.
    pub(super) progress: Option<&'document ProgressSender>,
    /// Which path is resolving, for errors and logs.
    pub(super) context: &'static str,
}

/// Source geometry cannot be paired with another population's timing arrays.
pub(super) struct ResolvedFaPopulation<'a> {
    resolved: ResolvedGroups,
    retry: super::supplied_retry::FaRetryInputs<'a>,
    recovery: &'a [std::sync::Arc<super::input::RetainedUtrPass>],
}

pub(super) struct InjectedFaPopulation<'a> {
    pub(super) applied: crate::chat_ops::fa::FaApplied,
    pub(super) evidence: Vec<FaGroupEvidence>,
    pub(super) fallback_events: Vec<FaFallbackEventTrace>,
    pub(super) retry: super::supplied_retry::FaRetryInputs<'a>,
    pub(super) recovery: Vec<std::sync::Arc<super::input::RetainedUtrPass>>,
}

impl<'a> ResolvedFaPopulation<'a> {
    pub(super) fn inject(
        self,
        chat: &mut crate::chat_ops::ChatFile,
        projection: crate::chat_ops::fa::FaProjection,
        write_wor: bool,
    ) -> Result<InjectedFaPopulation<'a>, ServerError> {
        let ProjectionParts {
            groups,
            timings,
            evidence,
            fallback_events,
        } = self.resolved.into_parts();
        let applied = crate::chat_ops::fa::apply_fa_results_with_projection_policy(
            chat, &groups, &timings, projection, write_wor,
        )
        .map_err(super::kept_bullets_failed)?;
        Ok(InjectedFaPopulation {
            applied,
            evidence,
            fallback_events,
            retry: self.retry,
            recovery: self.recovery.to_vec(),
        })
    }
}

/// Resolve every group's timings: `%wor` reuse for whole groups, then cache
/// and worker inference per request, then assembly. Shared by the full and
/// incremental FA paths.
pub(super) async fn resolve_group_timings<'a>(
    inputs: FaDispatchInputs<'a, '_>,
) -> Result<ResolvedFaPopulation<'a>, ServerError> {
    let FaDispatchInputs {
        population,
        chat_file,
        reusable_utterances,
        recovery,
        progress,
        context,
    } = inputs;
    let PreparedFaPopulation {
        context: preparation,
        groups,
        requests,
        cached: candidate_cache,
        mut retry,
    } = population;
    let FaPreparationContext {
        audio,
        worker_lang,
        services,
        params: fa_params,
    } = preparation;
    let plan = DispatchPlan::build(
        &groups,
        RequestKeying::new(
            audio.audio_identity,
            fa_params.whisper_text,
            fa_params.engine,
            worker_lang,
        ),
    );
    let total_units = plan.units().count();
    for unit in plan.units() {
        retry.settle(super::preparation::RequestScope::for_unit(unit));
    }
    if let Some(tx) = progress {
        let _ = tx.send(ProgressUpdate::new(
            FileStage::CheckingCache,
            Some(0),
            Some(total_units as i64),
        ));
    }

    // Tier 1: a group whose every utterance already carries clean `%wor`
    // timing is reused whole. Group-level on purpose: reuse reads whole
    // utterances back, which is what a group is made of.
    let OpenedLedger {
        ledger,
        replies,
        progress: reply_progress,
    } = UnitLedger::open(&plan, |group| {
        match group.is_reusable_from(reusable_utterances) {
            true => collect_preserved_group_timings(chat_file, group),
            false => None,
        }
    });
    let reused_units = ledger.reused_units();

    // Tiers 2 and 3: admitted cache evidence for every awaited request.
    let key_strings: Vec<String> = replies
        .iter()
        .map(ReplyTicket::unit)
        .filter(|unit| {
            matches!(
                candidate_cache.state(unit),
                cache::CandidateCacheState::Unexamined
            )
        })
        .map(|unit| unit.cache_key.as_str().to_owned())
        .collect();
    let (cached, cached_raw) = match fa_params.cache_policy {
        crate::params::CachePolicy::SkipCache => (
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
        ),
        crate::params::CachePolicy::UseCache | crate::params::CachePolicy::RequireCache => {
            super::cache_storage::read_values(&key_strings, services).await
        }
    };

    let mut misses = Vec::new();
    let mut fallback_events = Vec::new();
    for reply in replies {
        let unit = reply.unit();
        let answer = match candidate_cache.state(unit) {
            cache::CandidateCacheState::Admitted(answer) => Some(answer.clone()),
            cache::CandidateCacheState::Miss => None,
            cache::CandidateCacheState::Unexamined => {
                cache::resolve(
                    unit,
                    preparation,
                    cached_raw.get(unit.cache_key().as_str()),
                    cached.get(unit.cache_key().as_str()),
                )
                .await
            }
        };
        match answer {
            Some(answer) => match reply.cached(answer)? {
                replies::CacheDelivery::Delivered => {}
                // Corrupt evidence for this request: aligned again.
                replies::CacheDelivery::Refused(reply) => misses.push(reply),
            },
            None => misses.push(reply),
        }
    }

    {
        let sink = RetainUnitResults::new(reply_progress, services, &mut fallback_events, progress);
        let resolved_without_inference = sink.completed();
        info!(
            context,
            groups = groups.len(),
            requests = sink.total(),
            reused = reused_units,
            cache_hits = resolved_without_inference - reused_units,
            misses = misses.len(),
            "FA partition (reused from %wor / cache hits / misses)"
        );
        sink.report();

        // Inference consumes the same missed units and retained inputs. The
        // sink alone records each result and releases the next dispatch.
        if let FaInferencePlan::Authorized(authorization) =
            plan_fa_inference(fa_params.cache_policy, misses)?
        {
            FaWorkerTransport::production(services)
                .infer_units(
                    FaWorkerBatch {
                        authorization,
                        worker_lang: worker_lang.into(),
                        engine: fa_params.engine,
                    },
                    requests,
                    sink,
                )
                .await?
                .into_sink();
        }
    }

    if let Some(tx) = progress {
        let _ = tx.send(ProgressUpdate::new(
            FileStage::ApplyingResults,
            Some(total_units as i64),
            Some(total_units as i64),
        ));
    }

    let resolved = ResolvedGroups {
        groups: ledger.assemble(context)?,
        fallback_events,
    };
    drop(plan);
    Ok(ResolvedFaPopulation {
        resolved,
        recovery,
        retry: super::supplied_retry::FaRetryInputs {
            preparation,
            retry,
            cached: candidate_cache,
        },
    })
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use crate::chat_ops::fa::coordinates::{FileMs, Ms, Recording};
    use crate::chat_ops::fa::{AnchorIndex, AnchoredSplit, UtteranceAnchors, WordAnchor};
    use crate::chat_ops::{UtteranceIdx, WordIdx};

    /// A plan over `groups` with fixed run parameters, for unit tests of the
    /// transport and assembly.
    pub(crate) fn plan_for(groups: &[FaGroup]) -> DispatchPlan<'_> {
        let identity = crate::chat_ops::fa::AudioIdentity::from_metadata("/tmp/test.wav", 0, 0);
        DispatchPlan::build(
            groups,
            RequestKeying::new(
                &identity,
                crate::chat_ops::fa::WhisperTextMode::CharSpaced,
                crate::types::engines::FaEngineName::Wave2Vec,
                &crate::api::LanguageCode3::eng(),
            ),
        )
    }

    /// An anchored group of five words over 0..40 s, cut after word 1 and
    /// word 3 into three pieces.
    pub(crate) fn anchored_group() -> FaGroup {
        let words: Vec<FaWord> = (0..5)
            .map(|i| FaWord {
                utterance_index: UtteranceIdx::new(1),
                utterance_word_index: WordIdx::new(i),
                text: format!("w{i}"),
            })
            .collect();
        let recording = Recording::of_duration(Ms(100_000)).expect("non-empty");
        let window =
            FaWindow::within(&recording, FileMs::new(0), FileMs::new(40_000)).expect("inside");
        let anchors = AnchorIndex::fixture([(
            UtteranceIdx::new(1),
            UtteranceAnchors::fixture(
                5,
                vec![
                    WordAnchor::fixture(1, 10_000, 13_000),
                    WordAnchor::fixture(3, 24_000, 26_000),
                ],
            ),
        )]);
        let split = AnchoredSplit::plan_for_test(
            window,
            Ms(15_000),
            words,
            anchors.lookup(UtteranceIdx::new(1)),
        );
        FaGroup::anchored_test_fixture(split)
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{anchored_group, plan_for};
    use super::*;
    use crate::chat_ops::fa::TimeSpan;
    use crate::chat_ops::{UtteranceIdx, WordIdx};

    fn word(utterance: usize, index: usize, text: &str) -> FaWord {
        FaWord {
            utterance_index: UtteranceIdx::new(utterance),
            utterance_word_index: WordIdx::new(index),
            text: text.to_owned(),
        }
    }

    fn timing(start: u64, end: u64) -> Option<WordTiming> {
        WordTiming::fixture(start, end)
    }

    /// A fixture answer, offered the way the cache tier offers one: the
    /// ticket binds it to its own request.
    fn deliver(
        ticket: ReplyTicket<'_>,
        timings: Vec<Option<WordTiming>>,
        source: FaEvidenceSourceTrace,
    ) {
        match ticket
            .cached(UnitAnswer::new(timings, source))
            .expect("receiving ledger remains owned")
        {
            replies::CacheDelivery::Delivered => {}
            replies::CacheDelivery::Refused(_) => panic!("fixture answers its request"),
        }
    }

    fn bound_timing(unit: &DispatchUnit<'_>, start: u64, end: u64) -> Option<WordTiming> {
        timing(start, end).map(|timing| {
            timing
                .bind_request(unit.window())
                .expect("fixture fits request")
        })
    }

    fn split_parts(
        resolved: Vec<ResolvedGroup>,
    ) -> (Vec<Vec<Option<WordTiming>>>, Vec<FaGroupEvidence>) {
        let ProjectionParts {
            timings, evidence, ..
        } = ResolvedGroups {
            groups: resolved,
            fallback_events: Vec::new(),
        }
        .into_parts();
        (timings, evidence)
    }

    #[test]
    fn source_window_trace_does_not_invent_provenance_for_older_groups() {
        let legacy = serde_json::json!({
            "audio_start_ms": 0,
            "audio_end_ms": 1000,
            "utterance_indices": [0],
            "words": ["hello"],
            "word_ids": ["u0:w0"],
            "span": {
                "kind": "single",
                "source": "inference",
                "cache_key": "key"
            }
        });
        let trace: crate::types::traces::FaGroupTrace = serde_json::from_value(legacy).unwrap();
        assert!(trace.source_windows.is_none());
    }

    /// Single and anchored groups side by side: the plan lays out one unit
    /// for the single group and one per piece, and assembly concatenates
    /// each group's units back into word order, tagging every piece's
    /// evidence on the anchored span.
    #[test]
    fn assemble_reassembles_unit_timings_in_word_order() {
        let groups = vec![
            FaGroup::test_fixture(
                TimeSpan::new(0, 900),
                vec![word(0, 0, "hello"), word(0, 1, "world")],
                vec![UtteranceIdx::new(0)],
            ),
            anchored_group(),
        ];
        let plan = plan_for(&groups);
        let units: Vec<&DispatchUnit<'_>> = plan.units().collect();
        let shape: Vec<(usize, usize, usize)> = units
            .iter()
            .map(|unit| {
                (
                    unit.ordinal().index(),
                    unit.group().index(),
                    unit.words().len(),
                )
            })
            .collect();
        assert_eq!(shape, vec![(0, 0, 2), (1, 1, 2), (2, 1, 2), (3, 1, 1)]);

        // Resolve out of order, as cache hits and inference interleave.
        let OpenedLedger {
            ledger,
            replies,
            progress,
        } = UnitLedger::open(&plan, |_| None);
        assert_eq!((progress.completed(), progress.total()), (0, 4));
        let mut tickets: std::collections::HashMap<_, _> = replies
            .into_iter()
            .map(|reply| (reply.unit().ordinal().index(), reply))
            .collect();
        let replies = [
            (
                3,
                vec![timing(30_000, 31_000)],
                FaEvidenceSourceTrace::Inference,
            ),
            (
                1,
                vec![timing(1_000, 2_000), timing(11_000, 13_000)],
                FaEvidenceSourceTrace::Cache,
            ),
            (
                0,
                vec![timing(100, 400), timing(500, 800)],
                FaEvidenceSourceTrace::Inference,
            ),
            (
                2,
                vec![timing(14_000, 15_000), None],
                FaEvidenceSourceTrace::RawEvidenceReplay,
            ),
        ];
        for (index, timings, source) in replies {
            deliver(
                tickets
                    .remove(&index)
                    .expect("one completion right per request"),
                timings,
                source,
            );
        }

        assert_eq!((progress.completed(), progress.total()), (4, 4));
        let (timings, evidence) = split_parts(
            ledger
                .assemble("test")
                .expect("every unit resolved for its own words"),
        );
        assert_eq!(
            timings,
            vec![
                vec![
                    bound_timing(units[0], 100, 400),
                    bound_timing(units[0], 500, 800)
                ],
                vec![
                    bound_timing(units[1], 1_000, 2_000),
                    bound_timing(units[1], 11_000, 13_000),
                    bound_timing(units[2], 14_000, 15_000),
                    None,
                    bound_timing(units[3], 30_000, 31_000),
                ],
            ]
        );
        match &evidence[1].group.span {
            FaGroupSpanTrace::Anchored {
                pieces,
                unrequested,
            } => {
                assert!(unrequested.is_empty());
                let summary: Vec<_> = pieces
                    .iter()
                    .map(|p| (p.start_ms, p.end_ms, p.first_word, p.last_word))
                    .collect();
                assert_eq!(
                    summary,
                    vec![
                        (0, 13_000, 0, 1),
                        (13_000, 26_000, 2, 3),
                        (26_000, 40_000, 4, 4),
                    ]
                );
                assert!(matches!(pieces[0].source, FaEvidenceSourceTrace::Cache));
                assert!(matches!(pieces[2].source, FaEvidenceSourceTrace::Inference));
            }
            other => panic!("an anchored group records its pieces, got {other:?}"),
        }
        assert!(matches!(
            evidence[0].group.span,
            FaGroupSpanTrace::Single {
                source: FaEvidenceSourceTrace::Inference,
                ..
            }
        ));
        assert_eq!(evidence[1].pre_injection_timings.len(), 5);
    }

    /// A missing piece fails the file rather than shifting every later
    /// timing onto the wrong word.
    #[test]
    fn assemble_refuses_a_group_with_an_unresolved_piece() {
        let groups = vec![anchored_group()];
        let plan = plan_for(&groups);
        let OpenedLedger {
            ledger, replies, ..
        } = UnitLedger::open(&plan, |_| None);
        for reply in replies {
            if reply.unit().ordinal().index() == 1 {
                drop(reply); // An actual abandoned response, not a foreign or duplicate write.
            } else {
                let timings = vec![None; reply.unit().words().len()];
                deliver(reply, timings, FaEvidenceSourceTrace::Unaligned);
            }
        }
        let error = ledger
            .assemble("test")
            .expect_err("piece 1 was never resolved");
        assert!(
            error
                .to_string()
                .contains("completed without timings for request(s): [1]")
        );
    }

    /// Binding is the one boundary every answer crosses (worker JSON,
    /// replayed raw evidence, the derived cache), so its refusals are
    /// asserted here; nothing downstream can receive an unbound answer.
    #[test]
    fn binding_admits_only_an_answer_for_exactly_its_request() {
        let groups = vec![anchored_group()];
        let plan = plan_for(&groups);
        let unit = plan.units().next().unwrap();
        let (start, end) = (unit.window().audio_start().get(), unit.window().end().get());
        let words = unit.words().len();
        let bind = |timings| resolution::bind_timings(unit.input(), timings);

        let mut inside = vec![None; words];
        inside[0] = timing(start, end);
        bind(inside).expect("an answer for its request");

        assert_eq!(
            bind(vec![None; words - 1]).unwrap_err(),
            resolution::ReplyRefusal::WordCount {
                answered: words - 1,
                expected: words
            }
        );
        let mut outside = vec![None; words];
        outside[0] = timing(end - 10, end + 10);
        assert_eq!(
            bind(outside).unwrap_err(),
            resolution::ReplyRefusal::OutsideWindow {
                start_ms: end - 10,
                end_ms: end + 10,
                window_start_ms: start,
                window_end_ms: end,
            }
        );
    }

    /// A group reused from `%wor` takes its timings whole, records every one
    /// of its requests as reused, and mints no reply capability.
    #[test]
    fn a_reused_group_takes_its_timings_whole_and_awaits_no_reply() {
        let groups = vec![anchored_group()];
        let plan = plan_for(&groups);
        let whole = vec![timing(1, 2), None, None, None, timing(3, 4)];
        let OpenedLedger {
            ledger, replies, ..
        } = UnitLedger::open(&plan, |_| Some(whole.clone()));
        assert!(replies.is_empty(), "reused groups mint no completion right");
        let (timings, evidence) =
            split_parts(ledger.assemble("test").expect("reuse covers the group"));
        assert_eq!(timings, vec![whole]);
        match &evidence[0].group.span {
            FaGroupSpanTrace::Anchored { pieces, .. } => assert!(
                pieces
                    .iter()
                    .all(|piece| matches!(piece.source, FaEvidenceSourceTrace::WorReuse))
            ),
            other => panic!("expected an anchored span, got {other:?}"),
        }
    }

    #[test]
    fn unavailable_parts_have_no_request_slots_and_keep_their_original_offsets() {
        use crate::chat_ops::fa::coordinates::{FileMs, Ms, Recording};
        use crate::chat_ops::fa::{AnchorIndex, AnchoredSplit, UtteranceAnchors, WordAnchor};
        let recording = Recording::of_duration(Ms(40_000)).expect("recording");
        let anchors = AnchorIndex::fixture([(
            UtteranceIdx::new(0),
            UtteranceAnchors::fixture(
                5,
                vec![
                    WordAnchor::fixture(1, 3_000, 4_000),
                    WordAnchor::fixture(2, 29_000, 30_000),
                ],
            ),
        )]);
        let split = AnchoredSplit::plan_for_test(
            FaWindow::within(&recording, FileMs::new(0), FileMs::new(40_000)).expect("window"),
            Ms(15_000),
            (0..5).map(|i| word(0, i, "word")).collect(),
            anchors.lookup(UtteranceIdx::new(0)),
        );
        let groups = vec![FaGroup::anchored_test_fixture(split)];
        let plan = plan_for(&groups);
        let units: Vec<_> = plan.units().collect();
        assert_eq!(units.len(), 2, "no request for original word 2");
        let OpenedLedger {
            ledger,
            mut replies,
            ..
        } = UnitLedger::open(&plan, |_| None);
        assert_eq!(replies.len(), 2);
        deliver(
            replies.pop().unwrap(),
            vec![timing(31_000, 32_000), timing(33_000, 34_000)],
            FaEvidenceSourceTrace::Cache,
        );
        deliver(
            replies.pop().unwrap(),
            vec![timing(100, 200), timing(300, 400)],
            FaEvidenceSourceTrace::Cache,
        );
        let (timings, evidence) =
            split_parts(ledger.assemble("partial").expect("all requests answered"));
        assert_eq!(
            timings[0],
            vec![
                bound_timing(units[0], 100, 200),
                bound_timing(units[0], 300, 400),
                None,
                bound_timing(units[1], 31_000, 32_000),
                bound_timing(units[1], 33_000, 34_000)
            ]
        );
        let FaGroupSpanTrace::Anchored { unrequested, .. } = &evidence[0].group.span else {
            panic!("anchored")
        };
        insta::assert_json_snapshot!(unrequested, @r#"
        [
          {
            "first_word": 2,
            "last_word": 2,
            "window": {
              "cause": "anchor_gap",
              "start_ms": 4000,
              "end_ms": 30000,
              "budget_ms": 15000,
              "gap_start_ms": 4000,
              "gap_end_ms": 30000
            },
            "source": "unaligned"
          }
        ]
        "#);
        // Reuse remains whole-group evidence, including timing on the range
        // that this run could not freshly request. No request slot is opened.
        let whole = vec![timing(100, 200); 5];
        let OpenedLedger {
            ledger: reused,
            replies,
            ..
        } = UnitLedger::open(&plan, |_| Some(whole.clone()));
        assert!(replies.is_empty());
        let (timings, evidence) = split_parts(reused.assemble("reuse").expect("retained evidence"));
        assert_eq!(timings[0], whole);
        let FaGroupSpanTrace::Anchored { unrequested, .. } = &evidence[0].group.span else {
            panic!("anchored")
        };
        assert_eq!(
            unrequested[0].source,
            crate::types::traces::UnrequestedRangeSource::WorReuse
        );
    }

    #[test]
    fn old_complete_anchored_trace_remains_readable_without_unrequested_ranges() {
        let old = serde_json::json!({"kind":"anchored","pieces":[]});
        let trace: FaGroupSpanTrace =
            serde_json::from_value(old.clone()).expect("schema 6 anchored span");
        assert_eq!(serde_json::to_value(trace).expect("roundtrip"), old);
    }
}
