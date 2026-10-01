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
//! they do not: it is one group and several requests, one per piece.
//!
//! So everything that is about a REQUEST (cache keys, cache admission, worker
//! dispatch, raw evidence, fallback) is per [`DispatchUnit`], and everything
//! that is about the TRANSCRIPT (`%wor` reuse, injection, post-processing)
//! stays per group. [`UnitLedger::assemble`] is the one place unit timings
//! become group timings, concatenated in word order.
//!
//! ```mermaid
//! flowchart LR
//!     groups["groups (FaGroup)"] -->|"DispatchPlan::build"| plan["one PlannedGroup per group,<br/>its units by shape"]
//!     plan -->|"UnitLedger::open"| ledger["per group: Reused, or<br/>one slot per request"]
//!     ledger -->|"cache / worker"| ledger
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
//! therefore walks pairs, never parallel arrays; the one lookup by ordinal is
//! where a worker reply comes back, and it fails loudly if the reply names a
//! request the ledger does not await.

use std::collections::HashSet;

use talkbank_model::WordIdx;
use tracing::{info, warn};

use super::transport::{FaInferencePlan, FaWorkerBatch, FaWorkerTransport, plan_fa_inference};
use super::{
    AdmittedCachedFaTimings, AdmittedFaCacheGroup, FaCacheUnitAdmission, FaServices,
    incremental::collect_preserved_group_timings,
};
use crate::api::DurationMs;
use crate::cache::tasks::{FORCED_ALIGNMENT, FORCED_ALIGNMENT_RAW_EVIDENCE};
use crate::chat_ops::CacheKey;
use crate::chat_ops::fa::coordinates::FaWindow;
use crate::chat_ops::fa::{
    AnchoredPiece, FaGroup, FaWord, GroupUnit, GroupUnits, WordTiming, cache_key,
};
use crate::error::ServerError;
use crate::params::{AudioContext, FaParams};
use crate::runner::util::{FileStage, ProgressSender, ProgressUpdate};
use crate::types::results::FaGroupEvidence;
use crate::types::traces::{
    FaEvidenceSourceTrace, FaFallbackEventTrace, FaGroupSpanTrace, FaGroupTrace, FaPieceTrace,
    TimingTrace,
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

    /// An ordinal minted outside any plan, for transport tests of request ids.
    #[cfg(test)]
    pub(crate) fn fixture(index: usize) -> Self {
        Self(index)
    }
}

impl std::fmt::Display for UnitOrdinal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A piece's position within its anchored group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PieceOrdinal(usize);

/// Which share of its group a request is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum UnitRole {
    /// A single group's one request, over all its words.
    Whole,
    /// One piece of an anchored group.
    Piece(PieceOrdinal),
}

/// One FA request: some consecutive words of one group, the window they are
/// aligned against, and the request's cache identity.
#[derive(Debug)]
pub(crate) struct DispatchUnit<'g> {
    ordinal: UnitOrdinal,
    group: GroupOrdinal,
    role: UnitRole,
    request: GroupUnit<'g>,
    cache_key: CacheKey,
}

impl<'g> DispatchUnit<'g> {
    /// Key one request and place it in the plan.
    fn keyed(
        ordinal: UnitOrdinal,
        group: GroupOrdinal,
        role: UnitRole,
        request: GroupUnit<'g>,
        key: &RequestKeying<'_>,
    ) -> Self {
        let texts: Vec<String> = request.words.iter().map(|word| word.text.clone()).collect();
        let cache_key = cache_key(
            &texts,
            key.audio_identity,
            request.window.audio_start().get(),
            request.window.end().get(),
            key.gap_healing,
            key.engine,
        );
        Self {
            ordinal,
            group,
            role,
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
        self.request.words
    }

    /// The window this request aligns against, within the engine budget.
    pub(crate) fn window(&self) -> FaWindow {
        self.request.window
    }

    /// The request's content-addressed cache identity.
    pub(crate) fn cache_key(&self) -> &CacheKey {
        &self.cache_key
    }

    /// The words' cleaned texts, as the aligner receives them.
    pub(crate) fn texts(&self) -> Vec<String> {
        self.request
            .words
            .iter()
            .map(|word| word.text.clone())
            .collect()
    }
}

/// The run facts every request's cache key is computed from.
struct RequestKeying<'a> {
    audio_identity: &'a crate::chat_ops::fa::AudioIdentity,
    gap_healing: crate::chat_ops::fa::WordGapHealing,
    engine: crate::types::engines::FaEngineName,
}

/// One piece's request and the words it starts and ends at, for evidence.
#[derive(Debug)]
struct PlannedPiece<'g> {
    unit: DispatchUnit<'g>,
    first_word: WordIdx,
    last_word: WordIdx,
}

/// A group's requests, in the group's own shape.
#[derive(Debug)]
enum PlannedUnits<'g> {
    /// A single group's one request.
    Whole(DispatchUnit<'g>),
    /// An anchored group's pieces, in word order.
    Pieces(Vec<PlannedPiece<'g>>),
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
        whole.into_iter().chain(pieces.map(|piece| &piece.unit))
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
    pub(super) fn build(
        groups: &'g [FaGroup],
        audio_identity: &crate::chat_ops::fa::AudioIdentity,
        gap_healing: crate::chat_ops::fa::WordGapHealing,
        engine: crate::types::engines::FaEngineName,
    ) -> Self {
        let key = RequestKeying {
            audio_identity,
            gap_healing,
            engine,
        };
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
                        UnitRole::Whole,
                        request,
                        &key,
                    )),
                    GroupUnits::Pieces(pieces) => PlannedUnits::Pieces(
                        pieces
                            .enumerate()
                            .map(
                                |(position, piece): (usize, &'g AnchoredPiece)| PlannedPiece {
                                    unit: DispatchUnit::keyed(
                                        mint(),
                                        ordinal,
                                        UnitRole::Piece(PieceOrdinal(position)),
                                        GroupUnit {
                                            words: piece.words(),
                                            window: piece.window(),
                                        },
                                        &key,
                                    ),
                                    first_word: piece.first_word(),
                                    last_word: piece.last_word(),
                                },
                            )
                            .collect(),
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

/// One unit's resolved timings and where they came from.
#[derive(Debug)]
struct UnitResolution {
    timings: Vec<Option<WordTiming>>,
    source: FaEvidenceSourceTrace,
}

/// One awaited request and its resolution, once it has one.
type Slot<'p, 'g> = (&'p DispatchUnit<'g>, Option<UnitResolution>);

/// How one group is being resolved, in the group's own shape.
enum GroupResolution<'p, 'g> {
    /// Reused whole from `%wor`: no request is awaited.
    Reused(Vec<Option<WordTiming>>),
    /// A single group's one request.
    Whole(Slot<'p, 'g>),
    /// An anchored group's pieces, each paired with its bounds.
    Pieces(Vec<(Slot<'p, 'g>, &'p PlannedPiece<'g>)>),
}

/// What has been resolved so far, one entry per group in plan order, each
/// paired with the planned group it resolves. Consumed by
/// [`UnitLedger::assemble`].
pub(super) struct UnitLedger<'p, 'g> {
    groups: Vec<(&'p PlannedGroup<'g>, GroupResolution<'p, 'g>)>,
}

impl<'p, 'g> UnitLedger<'p, 'g> {
    /// Open the ledger, deciding `%wor` reuse per group: `reuse` returns a
    /// group's preserved timings when the whole group may be reused.
    fn open(
        plan: &'p DispatchPlan<'g>,
        mut reuse: impl FnMut(&FaGroup) -> Option<Vec<Option<WordTiming>>>,
    ) -> Self {
        let groups = plan
            .groups
            .iter()
            .map(|planned| {
                let resolution = match reuse(planned.group) {
                    Some(timings) => GroupResolution::Reused(timings),
                    None => match &planned.units {
                        PlannedUnits::Whole(unit) => GroupResolution::Whole((unit, None)),
                        PlannedUnits::Pieces(pieces) => GroupResolution::Pieces(
                            pieces
                                .iter()
                                .map(|piece| ((&piece.unit, None), piece))
                                .collect(),
                        ),
                    },
                };
                (planned, resolution)
            })
            .collect();
        Self { groups }
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

    /// Every awaited request's slot, in plan order.
    fn open_slots(&mut self) -> Vec<&mut Slot<'p, 'g>> {
        let mut slots = Vec::new();
        for (_, resolution) in &mut self.groups {
            match resolution {
                GroupResolution::Reused(_) => {}
                GroupResolution::Whole(slot) => slots.push(slot),
                GroupResolution::Pieces(pieces) => {
                    slots.extend(pieces.iter_mut().map(|(slot, _)| slot));
                }
            }
        }
        slots
    }

    /// Record a worker reply for `unit`. Refused, loudly, when the ledger
    /// does not await that request: a reply for a reused group, for a shape
    /// the group does not have, or from another file's plan.
    fn record(
        &mut self,
        unit: &DispatchUnit<'_>,
        resolution: UnitResolution,
    ) -> Result<(), ServerError> {
        let not_awaited = || {
            ServerError::Validation(format!(
                "FA reply for request {} (group {}) answers no awaited request",
                unit.ordinal, unit.group
            ))
        };
        let (_, group) = self.groups.get_mut(unit.group.0).ok_or_else(not_awaited)?;
        let slot = match (group, unit.role) {
            (GroupResolution::Whole((_, slot)), UnitRole::Whole) => slot,
            (GroupResolution::Pieces(pieces), UnitRole::Piece(index)) => {
                let ((_, slot), _) = pieces.get_mut(index.0).ok_or_else(not_awaited)?;
                slot
            }
            (GroupResolution::Reused(_), UnitRole::Whole | UnitRole::Piece(_))
            | (GroupResolution::Whole(_), UnitRole::Piece(_))
            | (GroupResolution::Pieces(_), UnitRole::Whole) => return Err(not_awaited()),
        };
        *slot = Some(resolution);
        Ok(())
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
    fn assemble(self, context: &str) -> Result<Vec<ResolvedGroup>, ServerError> {
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
                                .map(|piece| piece_trace(piece, FaEvidenceSourceTrace::WorReuse))
                                .collect(),
                        },
                    };
                    (timings, span)
                }
                GroupResolution::Whole((unit, None)) => {
                    missing.push(unit.ordinal);
                    continue;
                }
                GroupResolution::Whole((unit, Some(answer))) => {
                    let span = single_span(unit, answer.source);
                    (answered(unit, answer, context)?, span)
                }
                GroupResolution::Pieces(pieces) => {
                    let mut timings = Vec::new();
                    let mut traces = Vec::with_capacity(pieces.len());
                    let missing_before = missing.len();
                    for ((unit, answer), piece) in pieces {
                        match answer {
                            None => missing.push(unit.ordinal),
                            Some(answer) => {
                                traces.push(piece_trace(piece, answer.source));
                                timings.extend(answered(unit, answer, context)?);
                            }
                        }
                    }
                    // A group with an unresolved request has nothing to
                    // assemble; the missing list below names every such
                    // request and fails the file.
                    if missing.len() > missing_before {
                        continue;
                    }
                    (timings, FaGroupSpanTrace::Anchored { pieces: traces })
                }
            };
            resolved.push(ResolvedGroup {
                trace: group_trace(planned.group, span),
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

/// A request's timings, refused when they do not answer for exactly its
/// words.
fn answered(
    unit: &DispatchUnit<'_>,
    answer: UnitResolution,
    context: &str,
) -> Result<Vec<Option<WordTiming>>, ServerError> {
    match answer.timings.len() == unit.words().len() {
        true => Ok(answer.timings),
        false => Err(ServerError::Validation(format!(
            "{context}: request {} of group {} answered for {} words, not its {}",
            unit.ordinal,
            unit.group,
            answer.timings.len(),
            unit.words().len()
        ))),
    }
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
    }
}

/// The evidence trace of one group: what it holds and how it was executed.
fn group_trace(group: &FaGroup, span: FaGroupSpanTrace) -> FaGroupTrace {
    FaGroupTrace {
        audio_start_ms: DurationMs(group.audio_start_ms()),
        audio_end_ms: DurationMs(group.audio_end_ms()),
        utterance_indices: group
            .utterance_indices()
            .iter()
            .map(|idx| idx.raw())
            .collect(),
        words: group.words().map(|w| w.text.clone()).collect(),
        word_ids: group.words().map(|word| word.stable_id()).collect(),
        span,
    }
}

/// One group's assembled timings and its evidence trace.
#[derive(Debug)]
pub(super) struct ResolvedGroup {
    timings: Vec<Option<WordTiming>>,
    trace: FaGroupTrace,
}

/// Everything resolution produced for a file's groups.
pub(super) struct ResolvedGroups {
    groups: Vec<ResolvedGroup>,
    /// Engine fallbacks taken by any request.
    pub(super) fallback_events: Vec<FaFallbackEventTrace>,
}

impl ResolvedGroups {
    /// Split into the timings injection consumes (one list per group, in
    /// group order) and the evidence the trace records, whose pre-injection
    /// snapshot is taken from those very timings.
    pub(super) fn into_parts(self) -> (Vec<Vec<Option<WordTiming>>>, Vec<FaGroupEvidence>) {
        self.groups
            .into_iter()
            .map(|group| {
                let pre_injection_timings = group
                    .timings
                    .iter()
                    .map(|timing| timing.as_ref().map(TimingTrace::from_word_timing))
                    .collect();
                (
                    group.timings,
                    FaGroupEvidence {
                        group: group.trace,
                        pre_injection_timings,
                    },
                )
            })
            .unzip()
    }
}

/// Everything resolution needs about the file and the run.
pub(super) struct FaDispatchInputs<'a> {
    /// The groups to resolve, in order.
    pub(super) groups: &'a [FaGroup],
    /// The document the groups were built from, for `%wor` reuse.
    pub(super) chat_file: &'a crate::chat_ops::ChatFile,
    /// Utterances whose existing `%wor` timing may be reused; a group is
    /// reused only when every one of its utterances is in this set.
    pub(super) reusable_utterances: &'a HashSet<usize>,
    /// The audio being aligned.
    pub(super) audio: &'a AudioContext<'a>,
    /// Worker language for model bootstrap.
    pub(super) worker_lang: &'a crate::api::LanguageCode3,
    /// Pool, cache and FA namespace.
    pub(super) services: FaServices<'a>,
    /// The run's FA parameters.
    pub(super) fa_params: &'a FaParams,
    /// Progress sink.
    pub(super) progress: Option<&'a ProgressSender>,
    /// Which path is resolving, for errors and logs.
    pub(super) context: &'static str,
}

/// Resolve every group's timings: `%wor` reuse for whole groups, then cache
/// and worker inference per request, then assembly. Shared by the full and
/// incremental FA paths.
pub(super) async fn resolve_group_timings(
    inputs: FaDispatchInputs<'_>,
) -> Result<ResolvedGroups, ServerError> {
    let FaDispatchInputs {
        groups,
        chat_file,
        reusable_utterances,
        audio,
        worker_lang,
        services,
        fa_params,
        progress,
        context,
    } = inputs;
    let plan = DispatchPlan::build(
        groups,
        audio.audio_identity,
        fa_params.gap_healing,
        fa_params.engine,
    );
    let total_units = plan.units().count();
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
    let mut ledger = UnitLedger::open(&plan, |group| {
        match group.is_reusable_from(reusable_utterances) {
            true => collect_preserved_group_timings(chat_file, group),
            false => None,
        }
    });
    let reused_units = ledger.reused_units();

    // Tiers 2 and 3: admitted cache evidence for every awaited request.
    let mut slots = ledger.open_slots();
    let key_strings: Vec<String> = slots
        .iter()
        .map(|(unit, _)| unit.cache_key.as_str().to_owned())
        .collect();
    let (cached, cached_raw) = match fa_params.cache_policy {
        crate::params::CachePolicy::SkipCache => (
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
        ),
        crate::params::CachePolicy::UseCache | crate::params::CachePolicy::RequireCache => {
            let cached = match services
                .pipeline
                .cache
                .get_batch(&key_strings, FORCED_ALIGNMENT, services.cache_namespace)
                .await
            {
                Ok(map) => map,
                Err(error) => {
                    warn!(error = %error, "FA cache batch lookup failed (treating all as misses)");
                    std::collections::HashMap::new()
                }
            };
            let cached_raw = match services
                .pipeline
                .cache
                .get_batch(
                    &key_strings,
                    FORCED_ALIGNMENT_RAW_EVIDENCE,
                    services.cache_namespace,
                )
                .await
            {
                Ok(map) => map,
                Err(error) => {
                    warn!(error = %error, "Raw FA evidence cache batch lookup failed");
                    std::collections::HashMap::new()
                }
            };
            (cached, cached_raw)
        }
    };

    let mut misses: Vec<&DispatchUnit<'_>> = Vec::new();
    let mut fallback_events = Vec::new();
    for (unit, slot) in slots.iter_mut().map(|slot| (slot.0, &mut slot.1)) {
        // Tier 2: replay immutable worker evidence through current Rust logic.
        // Tier 3: fall back to the admitted derived timing cache when raw
        // evidence is unavailable. This ordering is what makes local algorithm
        // experiments inference-free.
        let key = unit.cache_key.as_str();
        let resolution =
            FaCacheUnitAdmission::new(unit, fa_params.engine, services.cache_namespace)
                .resolve(cached_raw.get(key), cached.get(key));
        for refusal in resolution.refusals {
            warn!(
                error = %refusal.error,
                cache_layer = refusal.layer,
                request = unit.ordinal.0,
                group = unit.group.0,
                "Cached FA evidence was refused"
            );
        }
        match resolution.admitted {
            AdmittedFaCacheGroup::RawEvidence(evidence) => {
                let evidence = *evidence;
                if let Some(event) = evidence.fallback_event {
                    fallback_events.push(event);
                }
                *slot = Some(UnitResolution {
                    timings: evidence.timings,
                    source: FaEvidenceSourceTrace::RawEvidenceReplay,
                });
            }
            AdmittedFaCacheGroup::DerivedTimings(timings) => {
                *slot = Some(UnitResolution {
                    timings: timings.into_timings(),
                    source: FaEvidenceSourceTrace::Cache,
                });
            }
            // Tier 4: no admitted cache evidence, so inference is needed.
            AdmittedFaCacheGroup::Miss => misses.push(unit),
        }
    }
    drop(slots);

    let resolved_without_inference = total_units - misses.len();
    info!(
        context,
        groups = groups.len(),
        requests = total_units,
        reused = reused_units,
        cache_hits = resolved_without_inference - reused_units,
        misses = misses.len(),
        "FA partition (reused from %wor / cache hits / misses)"
    );
    if let Some(tx) = progress {
        let _ = tx.send(ProgressUpdate::new(
            FileStage::Aligning,
            Some(resolved_without_inference as i64),
            Some(total_units as i64),
        ));
    }

    // Tier 4: worker inference for the misses, one request per unit. The
    // authorization holds the missed units themselves, so the transport never
    // looks a request up by number.
    if let FaInferencePlan::Authorized(authorization) =
        plan_fa_inference(fa_params.cache_policy, misses)?
    {
        let transport = FaWorkerTransport::production(services);
        let parsed_results = transport
            .infer_units(FaWorkerBatch {
                authorization,
                audio_path: audio.audio_path,
                worker_lang: worker_lang.into(),
                engine: fa_params.engine,
                gap_healing: fa_params.gap_healing,
            })
            .await?;

        for (parsed_idx, parsed_result) in parsed_results.into_iter().enumerate() {
            let projection = parsed_result.into_projection();
            let source = projection.source();
            if let Some(event) = projection.fallback_event {
                fallback_events.push(event);
            }
            // Only direct, version-identified evidence can enter either cache
            // layer. Fallback and unaligned results remain valid for this run
            // but are deliberately recomputed later: the fallback model is
            // outside the primary request's version namespace.
            if let Some(raw_evidence) = projection.raw_evidence {
                store_unit_evidence(
                    services,
                    projection.unit.cache_key.as_str(),
                    &projection.timings,
                    raw_evidence,
                )
                .await;
            }
            ledger.record(
                projection.unit,
                UnitResolution {
                    timings: projection.timings,
                    source,
                },
            )?;

            if let Some(tx) = progress {
                let _ = tx.send(ProgressUpdate::new(
                    FileStage::Aligning,
                    Some((resolved_without_inference + parsed_idx + 1) as i64),
                    Some(total_units as i64),
                ));
            }
        }
    }

    if let Some(tx) = progress {
        let _ = tx.send(ProgressUpdate::new(
            FileStage::ApplyingResults,
            Some(total_units as i64),
            Some(total_units as i64),
        ));
    }

    Ok(ResolvedGroups {
        groups: ledger.assemble(context)?,
        fallback_events,
    })
}

/// Write one request's direct evidence to both cache layers. Failures are
/// logged and non-fatal: the timings are valid for this run either way.
async fn store_unit_evidence(
    services: FaServices<'_>,
    key: &str,
    timings: &[Option<WordTiming>],
    raw_evidence: super::raw_evidence::ReplayableFaRawEvidence,
) {
    match AdmittedCachedFaTimings::encode_from_raw(timings.to_vec(), &raw_evidence) {
        Ok(cache_data) => {
            if let Err(error) = services
                .pipeline
                .cache
                .put_batch(
                    &[(key.to_owned(), cache_data)],
                    FORCED_ALIGNMENT,
                    services.cache_namespace,
                )
                .await
            {
                warn!(error = %error, "Failed to cache derived FA evidence (non-fatal)");
            }
        }
        Err(error) => {
            warn!(error = %error, "Failed to encode derived FA evidence (non-fatal)");
        }
    }
    match serde_json::to_value(raw_evidence) {
        Ok(cache_data) => {
            if let Err(error) = services
                .pipeline
                .cache
                .put_batch(
                    &[(key.to_owned(), cache_data)],
                    FORCED_ALIGNMENT_RAW_EVIDENCE,
                    services.cache_namespace,
                )
                .await
            {
                warn!(error = %error, "Failed to cache raw FA evidence (non-fatal)");
            }
        }
        Err(error) => {
            warn!(error = %error, "Failed to serialize raw FA evidence (non-fatal)");
        }
    }
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
            &identity,
            crate::chat_ops::fa::WordGapHealing::PreserveMeasured,
            crate::types::engines::FaEngineName::Wave2Vec,
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

    fn answer(timings: Vec<Option<WordTiming>>, source: FaEvidenceSourceTrace) -> UnitResolution {
        UnitResolution { timings, source }
    }

    fn split_parts(
        resolved: Vec<ResolvedGroup>,
    ) -> (Vec<Vec<Option<WordTiming>>>, Vec<FaGroupEvidence>) {
        ResolvedGroups {
            groups: resolved,
            fallback_events: Vec::new(),
        }
        .into_parts()
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
        let mut ledger = UnitLedger::open(&plan, |_| None);
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
            ledger
                .record(units[index], answer(timings, source))
                .expect("every request is awaited");
        }

        let (timings, evidence) = split_parts(
            ledger
                .assemble("test")
                .expect("every unit resolved for its own words"),
        );
        assert_eq!(
            timings,
            vec![
                vec![timing(100, 400), timing(500, 800)],
                vec![
                    timing(1_000, 2_000),
                    timing(11_000, 13_000),
                    timing(14_000, 15_000),
                    None,
                    timing(30_000, 31_000),
                ],
            ]
        );
        match &evidence[1].group.span {
            FaGroupSpanTrace::Anchored { pieces } => {
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
        let units: Vec<&DispatchUnit<'_>> = plan.units().collect();
        let mut ledger = UnitLedger::open(&plan, |_| None);
        for (index, timings) in [(0, vec![None, None]), (2, vec![None])] {
            ledger
                .record(
                    units[index],
                    answer(timings, FaEvidenceSourceTrace::Unaligned),
                )
                .expect("awaited");
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

    /// A piece answering for the wrong number of words is refused: the
    /// injection cursor would otherwise drift.
    #[test]
    fn assemble_refuses_a_piece_answering_for_other_words() {
        let groups = vec![anchored_group()];
        let plan = plan_for(&groups);
        let units: Vec<&DispatchUnit<'_>> = plan.units().collect();
        let mut ledger = UnitLedger::open(&plan, |_| None);
        for unit in &units {
            ledger
                .record(unit, answer(vec![None], FaEvidenceSourceTrace::Cache))
                .expect("awaited");
        }
        let error = ledger
            .assemble("test")
            .expect_err("pieces 0 and 1 hold two words each");
        assert!(
            error
                .to_string()
                .contains("answered for 1 words, not its 2")
        );
    }

    /// A group reused from `%wor` takes its timings whole, records every one
    /// of its requests as reused, and awaits no reply: one is refused.
    #[test]
    fn a_reused_group_takes_its_timings_whole_and_awaits_no_reply() {
        let groups = vec![anchored_group()];
        let plan = plan_for(&groups);
        let whole = vec![timing(1, 2), None, None, None, timing(3, 4)];
        let mut ledger = UnitLedger::open(&plan, |_| Some(whole.clone()));
        let stray = plan.units().next().expect("the group has requests");
        assert!(
            ledger
                .record(
                    stray,
                    answer(vec![None, None], FaEvidenceSourceTrace::Cache)
                )
                .is_err(),
            "a reused group holds no request slot"
        );
        let (timings, evidence) =
            split_parts(ledger.assemble("test").expect("reuse covers the group"));
        assert_eq!(timings, vec![whole]);
        match &evidence[0].group.span {
            FaGroupSpanTrace::Anchored { pieces } => assert!(
                pieces
                    .iter()
                    .all(|piece| matches!(piece.source, FaEvidenceSourceTrace::WorReuse))
            ),
            other => panic!("expected an anchored span, got {other:?}"),
        }
    }
}
