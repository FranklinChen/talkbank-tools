//! Source-bound partitions admitted against actual request resources.
//!
//! A failed whole request does not imply its prefixes fail or fit. Probe real
//! source-supported prefixes, in descending source order, and retain an explicit
//! unavailable atom when none fits. No duration threshold authorizes a request.

use std::future::Future;

use talkbank_model::{UtteranceIdx, WordIdx};

use super::FaWord;
use super::coordinates::{FaWindow, FileMs};
use super::grouping::SourceUtterance;
use super::split::{AnchoredPiece, NonEmptyWords};
use super::utr::AnchorLookup;
use super::{FaGroup, GroupSpan, GroupUnit};
use crate::types::worker_v2::FaResourceConstraintV2;

mod context;

/// Immutable source words and the complete window proposed for them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResourceRequest {
    utterance: UtteranceIdx,
    words: NonEmptyWords,
    window: FaWindow,
}

impl ResourceRequest {
    pub(crate) fn from_recorded(source: SourceUtterance<'_>) -> Self {
        Self {
            utterance: source.utterance(),
            window: source.window(),
            words: NonEmptyWords::from_recorded(source),
        }
    }

    pub(crate) fn words(&self) -> &[FaWord] {
        self.words.as_slice()
    }
    pub(crate) fn window(&self) -> FaWindow {
        self.window
    }
    pub(crate) fn from_anchored(split: &super::AnchoredSplit) -> Self {
        Self {
            utterance: *split.utterance(),
            words: split.source_words(),
            window: split.window(),
        }
    }

    /// Source/context selection is the sole edge admitting the coupled word
    /// population and audio window. It never fabricates a missing word.
    fn selection(
        &self,
        first: WordIdx,
        last: WordIdx,
        start: FileMs,
        end: FileMs,
    ) -> Result<Self, InvalidResourceCut> {
        if first < self.words.first_word()
            || last > self.words.last_word()
            || first > last
            || start < self.window.audio_start()
            || end > self.window.end()
        {
            return Err(InvalidResourceCut);
        }
        let window = FaWindow::within(&self.window.recording(), start, end)
            .map_err(|_| InvalidResourceCut)?;
        let words = NonEmptyWords::from_vec(
            self.words
                .as_slice()
                .iter()
                .filter(|word| {
                    first <= word.utterance_word_index && word.utterance_word_index <= last
                })
                .cloned()
                .collect(),
        )
        .ok_or(InvalidResourceCut)?;
        Ok(Self {
            utterance: self.utterance,
            words,
            window,
        })
    }

    fn split(self, cut: SourceCut) -> Result<(Self, Self), InvalidResourceCut> {
        let (before_window, after_window) = self
            .window
            .split_at(cut.at)
            .map_err(|_| InvalidResourceCut)?;
        let (before_words, after_words) = self
            .words
            .split_after(cut.word)
            .map_err(|_| InvalidResourceCut)?;
        Ok((
            Self {
                utterance: self.utterance,
                words: before_words,
                window: before_window,
            },
            Self {
                utterance: self.utterance,
                words: after_words,
                window: after_window,
            },
        ))
    }

    async fn prepare<P: ResourceProbe>(
        self,
        probe: &mut P,
    ) -> Result<Admission<P::Prepared>, ResourcePlanError<P::Error>> {
        match probe
            .prepare(GroupUnit::from_resource(&self))
            .await
            .map_err(ResourcePlanError::Probe)?
        {
            ResourceProbeOutcome::Prepared(prepared) => Ok(Admission::Prepared(PreparedPiece {
                source: AnchoredPiece::new(self.words, self.window),
                prepared,
            })),
            ResourceProbeOutcome::Limited(constraint) => Ok(Admission::Limited(LimitedRequest {
                request: self,
                constraint,
            })),
        }
    }
}

/// A context request whose owned answer range was admitted with its window.
/// The aligner sees all request words; injection sees only this owned range.
pub(super) struct ResourceContext {
    request: ResourceRequest,
    owned: std::ops::Range<usize>,
    ownership_window: FaWindow,
}

impl ResourceContext {
    fn admit(
        request: ResourceRequest,
        first: WordIdx,
        last: WordIdx,
        ownership_window: FaWindow,
    ) -> Result<Self, InvalidResourceCut> {
        if request.window.audio_start() > ownership_window.audio_start()
            || request.window.end() < ownership_window.end()
        {
            return Err(InvalidResourceCut);
        }
        let words = request.words();
        let start = words
            .iter()
            .position(|word| word.utterance_word_index == first)
            .ok_or(InvalidResourceCut)?;
        let end = words
            .iter()
            .position(|word| word.utterance_word_index == last)
            .ok_or(InvalidResourceCut)?;
        if start > end {
            return Err(InvalidResourceCut);
        }
        Ok(Self {
            request,
            owned: start..end + 1,
            ownership_window,
        })
    }

    pub(super) fn into_parts(self) -> (NonEmptyWords, std::ops::Range<usize>, FaWindow, FaWindow) {
        (
            self.request.words,
            self.owned,
            self.ownership_window,
            self.request.window,
        )
    }

    async fn prepare<P: ResourceProbe>(
        self,
        probe: &mut P,
    ) -> Result<ResourceProbeOutcome<PreparedPiece<P::Prepared>>, ResourcePlanError<P::Error>> {
        match probe
            .prepare(GroupUnit::from_resource(&self.request))
            .await
            .map_err(ResourcePlanError::Probe)?
        {
            ResourceProbeOutcome::Prepared(prepared) => {
                Ok(ResourceProbeOutcome::Prepared(PreparedPiece {
                    source: AnchoredPiece::from_resource_context(self),
                    prepared,
                }))
            }
            ResourceProbeOutcome::Limited(cause) => Ok(ResourceProbeOutcome::Limited(cause)),
        }
    }
}

/// A model/host refusal is separate from transport or preparation failure.
pub(crate) enum ResourceProbeOutcome<P> {
    Prepared(P),
    Limited(FaResourceConstraintV2),
}

/// Implementations retain their actual prepared artifacts in `Prepared`.
pub(crate) trait ResourceProbe: Send {
    type Prepared: Send;
    type Error: Send;
    fn prepare(
        &mut self,
        request: GroupUnit<'_>,
    ) -> impl Future<Output = Result<ResourceProbeOutcome<Self::Prepared>, Self::Error>> + Send;
}

/// A source-cut construction defect cannot masquerade as a model refusal.
#[derive(Debug)]
pub(crate) struct InvalidResourceCut;

#[derive(Debug)]
pub(crate) enum ResourcePlanError<E> {
    Probe(E),
    InvalidCut(InvalidResourceCut),
}

#[derive(Clone, Copy)]
struct SourceCut {
    word: WordIdx,
    at: FileMs,
}

/// Only this admission edge can turn recovered anchors into cut proposals.
struct SourceCuts {
    cuts: Vec<SourceCut>,
}

impl SourceCuts {
    fn from_source(request: &ResourceRequest, lookup: AnchorLookup<'_>) -> Self {
        let AnchorLookup::Anchored(anchors) = lookup else {
            return Self { cuts: Vec::new() };
        };
        if anchors.alignable_words() != request.words.count() {
            return Self { cuts: Vec::new() };
        }
        let mut cuts: Vec<SourceCut> = Vec::new();
        for anchor in anchors.anchors() {
            if anchor.word() >= request.words.last_word()
                || anchor.end() <= request.window.audio_start()
                || anchor.end() >= request.window.end()
            {
                continue;
            }
            let cut = SourceCut {
                word: anchor.word(),
                at: anchor.end(),
            };
            match cuts.last_mut() {
                Some(previous) if previous.at == cut.at => *previous = cut,
                Some(_) | None => cuts.push(cut),
            }
        }
        Self { cuts }
    }

    fn inside(&self, request: &ResourceRequest) -> Vec<SourceCut> {
        self.cuts
            .iter()
            .copied()
            .filter(|cut| {
                request.words.first_word() <= cut.word
                    && cut.word < request.words.last_word()
                    && request.window.audio_start() < cut.at
                    && cut.at < request.window.end()
            })
            .collect()
    }
}

enum Admission<P> {
    Prepared(PreparedPiece<P>),
    Limited(LimitedRequest),
}
struct LimitedRequest {
    request: ResourceRequest,
    constraint: FaResourceConstraintV2,
}

/// Request words may later borrow context; output ownership remains disjoint.
pub(crate) struct PreparedPiece<P> {
    pub(crate) source: AnchoredPiece,
    pub(crate) prepared: P,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceUnavailable {
    words: NonEmptyWords,
    window: FaWindow,
    constraint: FaResourceConstraintV2,
}

impl ResourceUnavailable {
    pub(crate) fn words(&self) -> &[FaWord] {
        self.words.as_slice()
    }
    pub(crate) fn window(&self) -> FaWindow {
        self.window
    }
    pub(crate) fn first_word(&self) -> WordIdx {
        self.words.first_word()
    }
    pub(crate) fn last_word(&self) -> WordIdx {
        self.words.last_word()
    }
    pub(crate) fn constraint(&self) -> &FaResourceConstraintV2 {
        &self.constraint
    }
}

impl From<LimitedRequest> for ResourceUnavailable {
    fn from(limited: LimitedRequest) -> Self {
        Self {
            words: limited.request.words,
            window: limited.request.window,
            constraint: limited.constraint,
        }
    }
}

pub(crate) enum PreparedResourcePart<P> {
    Request(PreparedPiece<P>),
    Unavailable(ResourceUnavailable),
}

/// Final source geometry. Execution authority remains in the retained artifacts
/// and the subsequent lease-owning pool transition, never in this shape alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourcePart {
    Request(AnchoredPiece),
    Unavailable(ResourceUnavailable),
}

impl ResourcePart {
    #[cfg(test)]
    pub fn words(&self) -> &[FaWord] {
        match self {
            Self::Request(piece) => piece.words(),
            Self::Unavailable(piece) => piece.words(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceSplit {
    source: ResourceRequest,
    envelope: FaWindow,
    earlier: Vec<ResourcePart>,
    last: ResourcePart,
}

impl ResourceSplit {
    pub(crate) fn utterance(&self) -> &UtteranceIdx {
        &self.source.utterance
    }
    pub(crate) fn window(&self) -> FaWindow {
        self.envelope
    }
    pub(crate) fn words(&self) -> &[FaWord] {
        self.source.words()
    }
    pub(crate) fn source(&self) -> &ResourceRequest {
        &self.source
    }
    pub(crate) fn pieces(&self) -> ResourceParts<'_> {
        self.earlier.iter().chain(std::iter::once(&self.last))
    }
}

pub(crate) type ResourceParts<'a> =
    std::iter::Chain<std::slice::Iter<'a, ResourcePart>, std::iter::Once<&'a ResourcePart>>;

/// Nonempty by construction: the first planning step owns a part before this
/// accumulator exists. No word disappears when a request is unavailable.
pub(crate) struct PreparedResourceSplit<P> {
    source: ResourceRequest,
    earlier: Vec<PreparedResourcePart<P>>,
    last: PreparedResourcePart<P>,
}

impl<P> PreparedResourceSplit<P> {
    pub(crate) fn finish(self, mut retain: impl FnMut(&AnchoredPiece, P)) -> ResourceSplit {
        let end = match &self.last {
            PreparedResourcePart::Request(piece) => piece.source.ownership_window().end(),
            PreparedResourcePart::Unavailable(piece) => piece.window().end(),
        };
        let envelope = self
            .source
            .window
            .extend_by(end.since(self.source.window.end()));
        let mut finish = |part| match part {
            PreparedResourcePart::Request(piece) => {
                retain(&piece.source, piece.prepared);
                ResourcePart::Request(piece.source)
            }
            PreparedResourcePart::Unavailable(piece) => ResourcePart::Unavailable(piece),
        };
        ResourceSplit {
            source: self.source,
            envelope,
            earlier: self.earlier.into_iter().map(&mut finish).collect(),
            last: finish(self.last),
        }
    }

    fn push(&mut self, part: PreparedResourcePart<P>) {
        self.earlier.push(std::mem::replace(&mut self.last, part));
    }
}

enum Remaining {
    Done,
    More(ResourceRequest),
}
struct Step<P> {
    part: PreparedResourcePart<P>,
    remaining: Remaining,
}

async fn step<P: ResourceProbe>(
    request: ResourceRequest,
    cuts: &SourceCuts,
    probe: &mut P,
) -> Result<Step<P::Prepared>, ResourcePlanError<P::Error>> {
    let limited = match request.prepare(probe).await? {
        Admission::Prepared(piece) => {
            return Ok(Step {
                part: PreparedResourcePart::Request(piece),
                remaining: Remaining::Done,
            });
        }
        Admission::Limited(limited) => limited,
    };
    match &limited.constraint {
        FaResourceConstraintV2::EmptyDecodedAudio { .. } => {
            return Ok(Step {
                part: PreparedResourcePart::Unavailable(limited.into()),
                remaining: Remaining::Done,
            });
        }
        FaResourceConstraintV2::Model {
            cause:
                crate::types::worker_v2::FaPreparationRefusalV2::InsufficientInput { .. }
                | crate::types::worker_v2::FaPreparationRefusalV2::ProjectionUnavailable { .. },
        } => {
            return Ok(Step {
                part: PreparedResourcePart::Unavailable(limited.into()),
                remaining: Remaining::Done,
            });
        }
        FaResourceConstraintV2::Model {
            cause: crate::types::worker_v2::FaPreparationRefusalV2::ModelConstraint { .. },
        }
        | FaResourceConstraintV2::HostMemory { .. } => {}
    }
    let inside = cuts.inside(&limited.request);
    let Some((last, before)) = inside.split_last() else {
        return Ok(Step {
            part: PreparedResourcePart::Unavailable(limited.into()),
            remaining: Remaining::Done,
        });
    };
    // Probe every proposed prefix with its actual words and window. In
    // particular, failure at one cut does not prove failure at another.
    let (prefix, mut after) = limited
        .request
        .clone()
        .split(*last)
        .map_err(ResourcePlanError::InvalidCut)?;
    let mut refused = match prefix.prepare(probe).await? {
        Admission::Prepared(piece) => {
            return Ok(Step {
                part: PreparedResourcePart::Request(piece),
                remaining: Remaining::More(after),
            });
        }
        Admission::Limited(limited) => limited,
    };
    for cut in before.iter().rev() {
        let (prefix, remainder) = limited
            .request
            .clone()
            .split(*cut)
            .map_err(ResourcePlanError::InvalidCut)?;
        match prefix.prepare(probe).await? {
            Admission::Prepared(piece) => {
                return Ok(Step {
                    part: PreparedResourcePart::Request(piece),
                    remaining: Remaining::More(remainder),
                });
            }
            Admission::Limited(limited) => {
                refused = limited;
                after = remainder;
            }
        }
    }
    Ok(Step {
        part: PreparedResourcePart::Unavailable(refused.into()),
        remaining: Remaining::More(after),
    })
}

async fn plan_core<P: ResourceProbe>(
    request: ResourceRequest,
    anchors: AnchorLookup<'_>,
    probe: &mut P,
) -> Result<PreparedResourceSplit<P::Prepared>, ResourcePlanError<P::Error>> {
    let cuts = SourceCuts::from_source(&request, anchors);
    let original = request.clone();
    let first = step(request, &cuts, probe).await?;
    let mut split = PreparedResourceSplit {
        source: original.clone(),
        earlier: Vec::new(),
        last: first.part,
    };
    let mut remaining = first.remaining;
    while let Remaining::More(request) = remaining {
        let next = step(request, &cuts, probe).await?;
        split.push(next.part);
        remaining = next.remaining;
    }
    match anchors {
        AnchorLookup::Anchored(anchors) if anchors.alignable_words() == original.words.count() => {
            context::prepare(split, &original, anchors, probe).await
        }
        AnchorLookup::Anchored(_)
        | AnchorLookup::NotRecovered
        | AnchorLookup::NoReliableMatch
        | AnchorLookup::Refused(_) => Ok(split),
    }
}

async fn plan_before<P: ResourceProbe>(
    request: ResourceRequest,
    anchors: AnchorLookup<'_>,
    following: FileMs,
    probe: &mut P,
) -> Result<PreparedResourceSplit<P::Prepared>, ResourcePlanError<P::Error>> {
    let mut split = plan_core(request, anchors, probe).await?;
    if let PreparedResourcePart::Request(last) = &split.last
        && let Some(proposal) = last.source.trailing_proposal(following)
    {
        match probe
            .prepare(GroupUnit::from_piece(&proposal))
            .await
            .map_err(ResourcePlanError::Probe)?
        {
            ResourceProbeOutcome::Prepared(prepared) => {
                split.last = PreparedResourcePart::Request(PreparedPiece {
                    source: proposal,
                    prepared,
                })
            }
            // The unextended request already has its matching proof.
            ResourceProbeOutcome::Limited(_) => {}
        }
    }
    Ok(split)
}

#[cfg(test)]
pub(crate) async fn plan<P: ResourceProbe>(
    request: ResourceRequest,
    anchors: AnchorLookup<'_>,
    probe: &mut P,
) -> Result<PreparedResourceSplit<P::Prepared>, ResourcePlanError<P::Error>> {
    let following = FileMs::new(request.window.recording().duration().get());
    plan_before(request, anchors, following, probe).await
}

/// The queue alone binds each original group to the following source edge.
/// Whole feasible groups retain their geometry; only a constrained partition
/// uses this opportunity to propose separately admitted trailing context.
pub(crate) struct GroupPlanningInput {
    group: FaGroup,
    following: FileMs,
}

impl GroupPlanningInput {
    pub(crate) fn group(&self) -> &FaGroup {
        &self.group
    }
    pub(crate) fn into_group(self) -> FaGroup {
        self.group
    }
}

pub(crate) fn planning_queue(groups: Vec<FaGroup>) -> impl Iterator<Item = GroupPlanningInput> {
    let mut groups = groups.into_iter().peekable();
    std::iter::from_fn(move || {
        let group = groups.next()?;
        let recording_end = group.window().recording().duration().get();
        let following = groups.peek().map_or(recording_end, |next| {
            next.window().audio_start().get().min(recording_end)
        });
        Some(GroupPlanningInput {
            group,
            following: FileMs::new(following),
        })
    })
}

/// Production and declared-model controls share the exact geometry producer.
/// Every retained proof stays paired with the view the probe actually saw.
pub(crate) async fn prepare_group<P: ResourceProbe>(
    input: GroupPlanningInput,
    anchors: &super::AnchorIndex,
    probe: &mut P,
    mut retain: impl FnMut(GroupUnit<'_>, P::Prepared) + Send,
) -> Result<Vec<FaGroup>, ResourcePlanError<P::Error>> {
    let mut groups = Vec::new();
    let GroupPlanningInput { group, following } = input;
    match group.span() {
        GroupSpan::Single(single) => match probe
            .prepare(single.request())
            .await
            .map_err(ResourcePlanError::Probe)?
        {
            ResourceProbeOutcome::Prepared(prepared) => {
                retain(single.request(), prepared);
                groups.push(group);
            }
            ResourceProbeOutcome::Limited(_) => {
                let mut sources = single.source_utterances().peekable();
                while let Some(source) = sources.next() {
                    let utterance = source.utterance();
                    let following = sources
                        .peek()
                        .map_or(following, |next| next.window().audio_start());
                    let prepared = plan_before(
                        ResourceRequest::from_recorded(source),
                        anchors.lookup(utterance),
                        following,
                        probe,
                    )
                    .await?;
                    groups.push(FaGroup::from_resource_split(
                        prepared.finish(|piece, proof| retain(GroupUnit::from_piece(piece), proof)),
                    ));
                }
            }
        },
        GroupSpan::Anchored(split) => {
            // Historical geometry already includes its recorded trailing pad.
            let prepared = plan_before(
                ResourceRequest::from_anchored(split),
                anchors.lookup(*split.utterance()),
                split.window().end(),
                probe,
            )
            .await?;
            groups.push(FaGroup::from_resource_split(
                prepared.finish(|piece, proof| retain(GroupUnit::from_piece(piece), proof)),
            ));
        }
        GroupSpan::Resource(split) => {
            let prepared = plan_before(
                split.source().clone(),
                anchors.lookup(*split.utterance()),
                following,
                probe,
            )
            .await?;
            groups.push(FaGroup::from_resource_split(
                prepared.finish(|piece, proof| retain(GroupUnit::from_piece(piece), proof)),
            ));
        }
    }
    Ok(groups)
}

#[cfg(test)]
pub(crate) mod test_support;
#[cfg(test)]
mod tests;
