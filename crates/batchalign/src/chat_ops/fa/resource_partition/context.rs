//! Resource-checked tail recovery and acoustic/lexical context borrowing.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::*;
use crate::chat_ops::fa::utr::UtteranceAnchors;

pub(super) async fn prepare<P: ResourceProbe>(
    split: PreparedResourceSplit<P::Prepared>,
    original: &ResourceRequest,
    anchors: &UtteranceAnchors,
    probe: &mut P,
) -> Result<PreparedResourceSplit<P::Prepared>, ResourcePlanError<P::Error>> {
    let mut supported = Vec::new();
    for part in split.earlier.into_iter().chain(std::iter::once(split.last)) {
        match (supported.pop(), part) {
            (
                Some(PreparedResourcePart::Unavailable(before)),
                PreparedResourcePart::Request(after),
            ) => {
                let (before, after) = recover(before, after, original, anchors, probe).await?;
                supported.push(before);
                supported.push(PreparedResourcePart::Request(after));
            }
            (Some(before), next) => {
                supported.push(before);
                supported.push(next);
            }
            (None, next) => supported.push(next),
        }
    }
    // A minimum-input refusal can gain context from supported neighbours; it
    // cannot improve by being split into still smaller requests.
    recover_minimums(&mut supported, original, anchors, probe).await?;
    let mut segments: Vec<ContextSegment<P::Prepared>> = Vec::new();
    for part in supported {
        match (segments.last_mut(), part) {
            (Some(ContextSegment::Supported(run)), PreparedResourcePart::Request(piece)) => {
                run.rest.push(piece)
            }
            (_, PreparedResourcePart::Request(first)) => {
                segments.push(ContextSegment::Supported(SupportedRun {
                    first,
                    rest: Vec::new(),
                }))
            }
            (_, PreparedResourcePart::Unavailable(piece)) => {
                segments.push(ContextSegment::Unavailable(piece))
            }
        }
    }
    let mut supported = Vec::new();
    for segment in segments {
        match segment {
            ContextSegment::Supported(mut run) => {
                expand_run(&mut run, original, anchors, probe).await?;
                supported.extend(
                    std::iter::once(run.first)
                        .chain(run.rest)
                        .map(PreparedResourcePart::Request),
                );
            }
            ContextSegment::Unavailable(piece) => {
                supported.push(PreparedResourcePart::Unavailable(piece))
            }
        }
    }
    let last = supported
        .pop()
        .ok_or(ResourcePlanError::InvalidCut(InvalidResourceCut))?;
    Ok(PreparedResourceSplit {
        source: split.source,
        earlier: supported,
        last,
    })
}

/// Both candidates exist only when a nonempty unknown prefix can retain the
/// original leading window. A singleton cannot become a cropped tail request.
struct TailRecovery {
    prefix: ResourceRequest,
    joined: ResourceRequest,
}

impl TailRecovery {
    fn admit(
        before: &ResourceUnavailable,
        after: &AnchoredPiece,
        original: &ResourceRequest,
        anchors: &UtteranceAnchors,
    ) -> Result<Option<Self>, InvalidResourceCut> {
        if before.window.end() != after.ownership_window().audio_start()
            || before.last_word().raw() + 1 != after.first_word().raw()
        {
            return Ok(None);
        }
        let Some(previous) = before.words.as_slice().iter().rev().nth(1) else {
            return Ok(None);
        };
        let Some(anchor) = anchors.anchors().iter().find(|anchor| {
            anchor.word() == before.last_word()
                && anchor.end() == before.window.end()
                && anchor.start() < anchor.end()
                && before.window.audio_start() < anchor.start()
        }) else {
            return Ok(None);
        };
        Ok(Some(Self {
            prefix: original.selection(
                before.first_word(),
                previous.utterance_word_index,
                before.window.audio_start(),
                anchor.start(),
            )?,
            joined: original.selection(
                before.last_word(),
                after.last_word(),
                anchor.start(),
                after.ownership_window().end(),
            )?,
        }))
    }
}

async fn recover<P: ResourceProbe>(
    before: ResourceUnavailable,
    after: PreparedPiece<P::Prepared>,
    original: &ResourceRequest,
    anchors: &UtteranceAnchors,
    probe: &mut P,
) -> Result<
    (
        PreparedResourcePart<P::Prepared>,
        PreparedPiece<P::Prepared>,
    ),
    ResourcePlanError<P::Error>,
> {
    let Some(recovery) = TailRecovery::admit(&before, &after.source, original, anchors)
        .map_err(ResourcePlanError::InvalidCut)?
    else {
        return Ok((PreparedResourcePart::Unavailable(before), after));
    };
    let joined = match recovery.joined.prepare(probe).await? {
        Admission::Prepared(joined) => joined,
        Admission::Limited(_) => return Ok((PreparedResourcePart::Unavailable(before), after)),
    };
    let prefix = match recovery.prefix.prepare(probe).await? {
        Admission::Prepared(prefix) => PreparedResourcePart::Request(prefix),
        Admission::Limited(limited) => PreparedResourcePart::Unavailable(limited.into()),
    };
    Ok((prefix, joined))
}

#[derive(Clone, Copy)]
struct Edge {
    word: WordIdx,
    at: FileMs,
}

struct Ownership {
    first: WordIdx,
    last: WordIdx,
    window: FaWindow,
}

impl Ownership {
    fn of_piece(piece: &AnchoredPiece) -> Self {
        Self {
            first: piece.first_word(),
            last: piece.last_word(),
            window: piece.ownership_window(),
        }
    }
    fn of_unavailable(piece: &ResourceUnavailable) -> Self {
        Self {
            first: piece.first_word(),
            last: piece.last_word(),
            window: piece.window(),
        }
    }
}

/// Largest geometric proposals first, without a monotonicity assumption about
/// tokenization or allocation. One heap entry per start avoids materializing
/// the quadratic cross product of source-bound context edges.
struct ContextProposals {
    starts: Vec<Edge>,
    ends: Vec<Edge>,
    heap: BinaryHeap<(u64, Reverse<usize>, usize)>,
}

impl ContextProposals {
    fn new(first: Edge, last: Edge, owned: &Ownership, anchors: &UtteranceAnchors) -> Self {
        let mut starts = vec![
            first,
            Edge {
                word: owned.first,
                at: owned.window.audio_start(),
            },
        ];
        let mut ends = vec![
            last,
            Edge {
                word: owned.last,
                at: owned.window.end(),
            },
        ];
        for anchor in anchors.anchors() {
            if anchor.word() < first.word
                || anchor.word() > last.word
                || anchor.start() >= anchor.end()
                || anchor.start() < first.at
                || anchor.end() > last.at
            {
                continue;
            }
            if anchor.word() <= owned.first && anchor.start() <= owned.window.audio_start() {
                starts.push(Edge {
                    word: anchor.word(),
                    at: anchor.start(),
                });
            }
            if anchor.word() >= owned.last && anchor.end() >= owned.window.end() {
                ends.push(Edge {
                    word: anchor.word(),
                    at: anchor.end(),
                });
            }
        }
        starts.sort_by_key(|edge| (edge.at, edge.word));
        starts.dedup_by_key(|edge| (edge.at, edge.word));
        ends.sort_by_key(|edge| (edge.at, edge.word));
        ends.dedup_by_key(|edge| (edge.at, edge.word));
        let final_end = ends.len() - 1; // The supplied run endpoint makes this nonempty.
        let heap = starts
            .iter()
            .enumerate()
            .map(|(index, start)| {
                (
                    ends[final_end].at.since(start.at).0,
                    Reverse(index),
                    final_end,
                )
            })
            .collect();
        Self { starts, ends, heap }
    }
}

impl Iterator for ContextProposals {
    type Item = (Edge, Edge);
    fn next(&mut self) -> Option<Self::Item> {
        let (_, Reverse(start), end) = self.heap.pop()?;
        let selection = (self.starts[start], self.ends[end]);
        if end > 0 {
            self.heap.push((
                self.ends[end - 1].at.since(self.starts[start].at).0,
                Reverse(start),
                end - 1,
            ));
        }
        Some(selection)
    }
}

async fn context_for<P: ResourceProbe>(
    owned: &Ownership,
    first: Edge,
    last: Edge,
    original: &ResourceRequest,
    anchors: &UtteranceAnchors,
    probe: &mut P,
) -> Result<Option<PreparedPiece<P::Prepared>>, ResourcePlanError<P::Error>> {
    for (start, end) in ContextProposals::new(first, last, owned, anchors) {
        let candidate = original
            .selection(start.word, end.word, start.at, end.at)
            .map_err(ResourcePlanError::InvalidCut)?;
        let context = ResourceContext::admit(candidate, owned.first, owned.last, owned.window)
            .map_err(ResourcePlanError::InvalidCut)?;
        match context.prepare(probe).await? {
            ResourceProbeOutcome::Prepared(piece) => return Ok(Some(piece)),
            ResourceProbeOutcome::Limited(_) => {}
        }
    }
    Ok(None)
}

/// A context run is nonempty and contains only prepared requests. An
/// unavailable range ends the run instead of entering its expansion loop.
struct SupportedRun<P> {
    first: PreparedPiece<P>,
    rest: Vec<PreparedPiece<P>>,
}
enum ContextSegment<P> {
    Supported(SupportedRun<P>),
    Unavailable(ResourceUnavailable),
}

async fn expand_run<P: ResourceProbe>(
    run: &mut SupportedRun<P::Prepared>,
    original: &ResourceRequest,
    anchors: &UtteranceAnchors,
    probe: &mut P,
) -> Result<(), ResourcePlanError<P::Error>> {
    let Some(last) = run.rest.last() else {
        return Ok(());
    };
    let first = Edge {
        word: run.first.source.first_word(),
        at: run.first.source.ownership_window().audio_start(),
    };
    let last = Edge {
        word: last.source.last_word(),
        at: last.source.ownership_window().end(),
    };
    for piece in std::iter::once(&mut run.first).chain(run.rest.iter_mut()) {
        if let Some(context) = context_for(
            &Ownership::of_piece(&piece.source),
            first,
            last,
            original,
            anchors,
            probe,
        )
        .await?
        {
            *piece = context;
        }
    }
    Ok(())
}

async fn recover_minimums<P: ResourceProbe>(
    parts: &mut [PreparedResourcePart<P::Prepared>],
    original: &ResourceRequest,
    anchors: &UtteranceAnchors,
    probe: &mut P,
) -> Result<(), ResourcePlanError<P::Error>> {
    use crate::types::worker_v2::FaPreparationRefusalV2;
    for index in 0..parts.len() {
        let PreparedResourcePart::Unavailable(piece) = &parts[index] else {
            continue;
        };
        let FaResourceConstraintV2::Model {
            cause: FaPreparationRefusalV2::InsufficientInput { .. },
        } = piece.constraint()
        else {
            continue;
        };
        let owned = Ownership::of_unavailable(piece);
        let mut first = Edge {
            word: owned.first,
            at: owned.window.audio_start(),
        };
        let mut last = Edge {
            word: owned.last,
            at: owned.window.end(),
        };
        for before in parts[..index].iter().rev() {
            let PreparedResourcePart::Request(before) = before else {
                break;
            };
            first = Edge {
                word: before.source.first_word(),
                at: before.source.ownership_window().audio_start(),
            };
        }
        for after in &parts[index + 1..] {
            let PreparedResourcePart::Request(after) = after else {
                break;
            };
            last = Edge {
                word: after.source.last_word(),
                at: after.source.ownership_window().end(),
            };
        }
        if let Some(context) = context_for(&owned, first, last, original, anchors, probe).await? {
            parts[index] = PreparedResourcePart::Request(context);
        }
    }
    Ok(())
}
