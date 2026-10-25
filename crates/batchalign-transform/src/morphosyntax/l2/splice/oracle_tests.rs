//! An oracle for the L2 splice of one span: a symbolic graph model of what
//! the splice must write, compared with what [`host_redirects`],
//! [`splice_with_primary_root_policy`] and chatter's coordinated splice
//! actually write.
//!
//! The model never computes a chunk number. Every chunk is a node named by
//! where it came from ([`Node::Host`] by its pre-splice chunk, [`Node::Block`]
//! by its block chunk), every relation is an edge between nodes, and numbers
//! appear only at the end, by enumerating the nodes in their final order. The
//! production path translates numbers (offsets, a growth delta, redirects by
//! chunk position), so an arithmetic slip there cannot be mirrored here.
//!
//! Host and block are random trees, decoded from Prüfer sequences, over
//! items of one or two chunks (a word, or a word with a clitic), so spans
//! grow, shrink and change shape; the span, its attachment, and the source
//! word are random too. The model also predicts every refusal the policy
//! cannot recover from.

use std::collections::BTreeMap;

use proptest::prelude::*;

use super::*;
use crate::morphosyntax::alignment::HeadTarget;
use crate::morphosyntax::l2::extract::PrimaryStructuralInfo;
use crate::morphosyntax::l2::plan::ExternalRelation;
use talkbank_model::model::dependent_tier::GrammaticalRelation;
use talkbank_model::model::dependent_tier::mor::{Mor, MorStem, MorWord, PosCategory};
use talkbank_model::{Location, Terminator};

/// A chunk, named by where it came from; never by a number it is written at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Node {
    /// The utterance root (`%gra` head 0).
    Root,
    /// A host chunk, by its 1-based number before the splice.
    Host(usize),
    /// A block chunk, by its 1-based number in the block.
    Block(usize),
}

/// The decoded tree of a Prüfer sequence over nodes `1..=n`, rooted at
/// `root`: each node's parent, 0 for the root.
fn tree_from_pruefer(sequence: &[usize], n: usize, root: usize) -> Vec<usize> {
    let mut edges: Vec<(usize, usize)> = Vec::new();
    if n >= 2 {
        let mut degree = vec![1usize; n + 1];
        for &node in sequence {
            degree[node] += 1;
        }
        for &node in sequence {
            let leaf = (1..=n)
                .find(|&candidate| degree[candidate] == 1)
                .unwrap_or_else(|| panic!("a Prüfer sequence always has a leaf"));
            edges.push((leaf, node));
            degree[leaf] -= 1;
            degree[node] -= 1;
        }
        let rest: Vec<usize> = (1..=n).filter(|&node| degree[node] == 1).collect();
        edges.push((rest[0], rest[1]));
    }
    // Orient every edge away from the root.
    // `None` until a node is reached; the root's parent is 0.
    let mut parent: Vec<Option<usize>> = vec![None; n + 1];
    parent[root] = Some(0);
    let mut frontier = vec![root];
    while let Some(node) = frontier.pop() {
        for &(a, b) in &edges {
            let next = match (a == node, b == node) {
                (true, _) => b,
                (_, true) => a,
                _ => continue,
            };
            if parent[next].is_none() {
                parent[next] = Some(node);
                frontier.push(next);
            }
        }
    }
    parent[1..]
        .iter()
        .map(|parent| parent.unwrap_or_else(|| panic!("a tree reaches every node")))
        .collect()
}

/// A random tree over `n` nodes: a Prüfer sequence and a root.
fn tree(n: usize) -> impl Strategy<Value = Vec<usize>> {
    let sequence = prop::collection::vec(1..=n, n.saturating_sub(2));
    (sequence, 1..=n).prop_map(move |(sequence, root)| tree_from_pruefer(&sequence, n, root))
}

/// How the span root's attachment was planned.
#[derive(Debug, Clone, Copy)]
enum Planned {
    /// No span word attaches outside: no source word.
    Internal,
    /// The source word decided the span root is the utterance root.
    UtteranceRoot { source: usize },
    /// Under the host governor of the source word.
    HostGovernor { source: usize },
}

/// One generated case.
#[derive(Debug, Clone)]
struct Case {
    /// Chunks per host item (1 or 2).
    host_items: Vec<usize>,
    /// The host tree over its word chunks: each chunk's parent, 0 for root.
    host_parents: Vec<usize>,
    /// The replaced items, `start..end`.
    start: usize,
    end: usize,
    /// Chunks per block item (1 or 2).
    block_items: Vec<usize>,
    /// The block tree over its chunks.
    block_parents: Vec<usize>,
    planned: Planned,
}

fn case() -> impl Strategy<Value = Case> {
    (
        prop::collection::vec(1usize..=2, 1..=5),
        prop::collection::vec(1usize..=2, 1..=4),
    )
        .prop_flat_map(|(host_items, block_items)| {
            let host_chunks: usize = host_items.iter().sum();
            let block_chunks: usize = block_items.iter().sum();
            let items = host_items.len();
            let span = (0..items).prop_flat_map(move |start| (Just(start), start + 1..=items));
            (
                Just(host_items),
                tree(host_chunks),
                span,
                Just(block_items),
                tree(block_chunks),
                0..3usize,
                any::<prop::sample::Index>(),
            )
        })
        .prop_map(
            |(host_items, host_parents, (start, end), block_items, block_parents, kind, source)| {
                let source = source.index(end - start);
                let planned = match kind {
                    0 => Planned::Internal,
                    1 => Planned::UtteranceRoot { source },
                    _ => Planned::HostGovernor { source },
                };
                Case {
                    host_items,
                    host_parents,
                    start,
                    end,
                    block_items,
                    block_parents,
                    planned,
                }
            },
        )
}

/// A `%mor` item of `chunks` chunks.
fn item(stem: String, chunks: usize) -> Mor {
    let word = Mor::new(MorWord::new(PosCategory::new("noun"), MorStem::new(stem)));
    match chunks {
        1 => word,
        _ => word.with_post_clitic(MorWord::new(PosCategory::new("pron"), MorStem::new("cl"))),
    }
}

/// The relation label a tree edge gets.
fn label(parent: usize) -> &'static str {
    match parent {
        0 => "ROOT",
        _ => "DEP",
    }
}

/// The relation the planned host attachment carries: the source word's
/// `nmod`, as CHAT writes it. A literal, not the production renderer, so a
/// rendering defect cannot be mirrored here.
fn governor_relation() -> String {
    "NMOD".to_owned()
}

/// Why the splice refuses a case, as both sides name it. Comparing the
/// reason, not only "both refused", is what makes a refusal half of the
/// oracle: about a third of generated cases refuse, and a production path
/// refusing one of them for the wrong reason passed as agreement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Refusal {
    /// No root for the span can be applied: the planned one, nor the
    /// policy's fallback.
    RootUnavailable,
    /// With no source word, the span and the block hold different numbers
    /// of items, so no item has a counterpart by position.
    ItemCountsDiffer,
    /// A replaced item has no counterpart item in the block.
    NoCounterpart,
    /// A host dependent of a replaced item cannot be redirected: its
    /// counterpart has no unique chunk headed outside it.
    NoUniqueHeadChunk,
}

impl Refusal {
    /// The production path's refusal, by its typed error. A refusal the
    /// model never predicts fails the test: the model has to name every
    /// reason the production path can give on a generated case.
    fn of_production(error: AnchorError) -> Self {
        let AnchorError::Unreadable(error) = error else {
            panic!("the oracle generates no case refused as {error}");
        };
        match error {
            CoordinatedMutationError::UtteranceRootTaken { .. }
            | CoordinatedMutationError::SpanRootDependsOnSpan { .. }
            | CoordinatedMutationError::SpanRootInReplacedRange { .. }
            | CoordinatedMutationError::SpanRootOutOfHost { .. } => Self::RootUnavailable,
            CoordinatedMutationError::RedirectItemCountsDiffer { .. } => Self::ItemCountsDiffer,
            CoordinatedMutationError::NoCounterpart { .. } => Self::NoCounterpart,
            CoordinatedMutationError::NoUniqueHeadChunk { .. } => Self::NoUniqueHeadChunk,
            other => panic!("the oracle generates no case refused as {other}"),
        }
    }
}

/// What the splice writes: the attachment applied and the whole `%gra`, or
/// why it refused.
type Written = Result<(AppliedAttachment, Vec<(usize, usize, String)>), Refusal>;

/// The production path over one case.
fn production(case: &Case) -> Written {
    let host_chunks: usize = case.host_items.iter().sum();
    let mors: Vec<Mor> = case
        .host_items
        .iter()
        .enumerate()
        .map(|(at, &chunks)| item(format!("h{at}"), chunks))
        .collect();
    // A tree has exactly one root.
    let host_root = case
        .host_parents
        .iter()
        .position(|&p| p == 0)
        .map(|c| c + 1)
        .unwrap_or_else(|| panic!("a generated host tree has a root"));
    let mut relations: Vec<GrammaticalRelation> = case
        .host_parents
        .iter()
        .enumerate()
        .map(|(at, &parent)| {
            GrammaticalRelation::new(
                at + 1,
                parent,
                talkbank_model::GrammaticalRelationType::new(label(parent))
                    .expect("valid fixture relation"),
            )
        })
        .collect();
    relations.push(GrammaticalRelation::new(
        host_chunks + 1,
        host_root,
        talkbank_model::GrammaticalRelationType::new("PUNCT").expect("valid fixture relation"),
    ));
    let mut mor = MorTier::new_mor(
        Location::Unlocated,
        mors,
        Terminator::Period {
            location: Location::Unlocated,
        },
    );
    let mut gra = GraTier::new_gra(Location::Unlocated, relations);

    let source_word = |offset: usize| MorItemIndex::new(case.start + offset);
    let attachment = match case.planned {
        Planned::Internal => L2Attachment::InternalRoot,
        Planned::UtteranceRoot { source } => L2Attachment::UtteranceRoot {
            source_word: source_word(source),
        },
        Planned::HostGovernor { source } => L2Attachment::HostGovernor {
            source_word: source_word(source),
            source: PrimaryStructuralInfo::for_test("nmod", HeadTarget::Root, None),
            relation: ExternalRelation::Primary,
        },
    };
    let block_mors: Vec<Mor> = case
        .block_items
        .iter()
        .enumerate()
        .map(|(at, &chunks)| item(format!("b{at}"), chunks))
        .collect();
    let block_relations: Vec<GrammaticalRelation> = case
        .block_parents
        .iter()
        .enumerate()
        .map(|(at, &parent)| {
            GrammaticalRelation::new(
                at + 1,
                parent,
                talkbank_model::GrammaticalRelationType::new(label(parent))
                    .expect("valid fixture relation"),
            )
        })
        .collect();

    let anchor = current_root_anchor_for_attachment(&mor, &gra, &attachment)
        .map_err(Refusal::of_production)?;
    let block = SplicedBlock::new(block_mors, block_relations)
        .unwrap_or_else(|error| panic!("a generated block is a tree: {error}"));
    let range = case.start..case.end;
    let redirects =
        host_redirects(&attachment, &range, block.root_chunk()).map_err(Refusal::of_production)?;
    let applied =
        splice_with_primary_root_policy(&mut mor, &mut gra, range, block, anchor, redirects)
            .map_err(Refusal::of_production)?;
    Ok((
        applied,
        gra.relations()
            .iter()
            .map(|r| (r.index, r.head, r.relation.as_str().to_owned()))
            .collect(),
    ))
}

/// Where the span root attaches, as the model resolves it.
enum RootSpec {
    Utterance,
    Under(Node, String),
}

/// Why an attachment cannot be applied, as the model resolves it.
enum RootRefusal {
    /// A kept host chunk is already the root.
    Taken(Node),
    /// The anchor is replaced, or its chain of heads reaches the span.
    Unusable,
}

/// The symbolic model of the same splice.
fn model(case: &Case) -> Written {
    // The host graph: word chunks, then the terminator under the root word.
    let host_chunks: usize = case.host_items.iter().sum();
    let mut host: BTreeMap<Node, (Node, String)> = BTreeMap::new();
    for (at, &parent) in case.host_parents.iter().enumerate() {
        let head = match parent {
            0 => Node::Root,
            p => Node::Host(p),
        };
        host.insert(Node::Host(at + 1), (head, label(parent).to_owned()));
    }
    let root_word = case
        .host_parents
        .iter()
        .position(|&p| p == 0)
        .map(|c| c + 1);
    host.insert(
        Node::Host(host_chunks + 1),
        (root_word.map_or(Node::Root, Node::Host), "PUNCT".to_owned()),
    );
    // Which host chunks each host item owns, and which items are replaced.
    let mut host_item_chunks: Vec<Vec<Node>> = Vec::new();
    let mut next = 1;
    for &chunks in &case.host_items {
        host_item_chunks.push((next..next + chunks).map(Node::Host).collect());
        next += chunks;
    }
    let replaced: Vec<Node> = host_item_chunks[case.start..case.end].concat();
    let is_replaced = |node: Node| replaced.contains(&node);

    // The block graph, and which block chunks each block item owns.
    let mut block: BTreeMap<Node, Option<Node>> = BTreeMap::new();
    for (at, &parent) in case.block_parents.iter().enumerate() {
        block.insert(
            Node::Block(at + 1),
            (parent != 0).then_some(Node::Block(parent)),
        );
    }
    let block_root = block
        .iter()
        .find(|(_, head)| head.is_none())
        .map(|(node, _)| *node)
        .unwrap_or_else(|| panic!("a generated block tree has a root"));
    let mut block_item_chunks: Vec<Vec<Node>> = Vec::new();
    let mut next = 1;
    for &chunks in &case.block_items {
        block_item_chunks.push((next..next + chunks).map(Node::Block).collect());
        next += chunks;
    }

    // Where the span root goes. A host anchor is read off the source item's
    // first chunk, as the production path reads it.
    let planned = match case.planned {
        Planned::Internal | Planned::UtteranceRoot { .. } => RootSpec::Utterance,
        Planned::HostGovernor { source } => {
            match host[&host_item_chunks[case.start + source][0]].0 {
                Node::Root => RootSpec::Utterance,
                anchor => RootSpec::Under(anchor, governor_relation()),
            }
        }
    };
    let resolve = |spec: &RootSpec| -> Result<(Node, String), RootRefusal> {
        match spec {
            RootSpec::Utterance => match host
                .iter()
                .find(|(node, (head, _))| *head == Node::Root && !is_replaced(**node))
            {
                Some((taken, _)) => Err(RootRefusal::Taken(*taken)),
                None => Ok((Node::Root, "ROOT".to_owned())),
            },
            RootSpec::Under(anchor, relation) => {
                if is_replaced(*anchor) {
                    return Err(RootRefusal::Unusable);
                }
                // Its chain of heads must not reach the span.
                let mut current = *anchor;
                for _ in 0..host.len() {
                    match host[&current].0 {
                        Node::Root => break,
                        head if is_replaced(head) => return Err(RootRefusal::Unusable),
                        head => current = head,
                    }
                }
                Ok((*anchor, relation.clone()))
            }
        }
    };
    let (attached_to, root_relation, applied) = match resolve(&planned) {
        Ok((node, relation)) => {
            let applied = match planned {
                RootSpec::Utterance => AppliedAttachment::UtteranceRoot,
                RootSpec::Under(..) => AppliedAttachment::PlannedHost,
            };
            (node, relation, applied)
        }
        Err(RootRefusal::Taken(host_root)) => {
            let (node, relation) = resolve(&RootSpec::Under(host_root, "DEP".to_owned()))
                .map_err(|_| Refusal::RootUnavailable)?;
            (node, relation, AppliedAttachment::PrimaryRootFallback)
        }
        Err(RootRefusal::Unusable) => {
            let (node, relation) =
                resolve(&RootSpec::Utterance).map_err(|_| Refusal::RootUnavailable)?;
            (node, relation, AppliedAttachment::UtteranceRoot)
        }
    };

    // Where a dependent of each replaced host chunk goes: the span root for
    // the source item, otherwise the counterpart item's matching chunk (same
    // shape) or its one chunk headed outside it (`None` when not unique).
    let source = match case.planned {
        Planned::Internal => None,
        Planned::UtteranceRoot { source } | Planned::HostGovernor { source } => Some(source),
    };
    if source.is_none() && case.end - case.start != case.block_items.len() {
        return Err(Refusal::ItemCountsDiffer);
    }
    let mut redirect: BTreeMap<Node, Option<Node>> = BTreeMap::new();
    for offset in 0..case.end - case.start {
        let old = &host_item_chunks[case.start + offset];
        if Some(offset) == source {
            for chunk in old {
                redirect.insert(*chunk, Some(block_root));
            }
            continue;
        }
        let new = block_item_chunks
            .get(offset)
            .ok_or(Refusal::NoCounterpart)?;
        match old.len() == new.len() {
            true => {
                for (chunk, counterpart) in old.iter().zip(new) {
                    redirect.insert(*chunk, Some(*counterpart));
                }
            }
            false => {
                let outside: Vec<Node> = new
                    .iter()
                    .copied()
                    .filter(|chunk| match block[chunk] {
                        None => true,
                        Some(head) => !new.contains(&head),
                    })
                    .collect();
                let target = match outside[..] {
                    [one] => Some(one),
                    _ => None,
                };
                for chunk in old {
                    redirect.insert(*chunk, target);
                }
            }
        }
    }

    // The relations after the splice, as edges between nodes, in final
    // order: kept host chunks before the span, the block, the rest.
    let mut edges: Vec<(Node, Node, String)> = Vec::new();
    let mut block_written = false;
    for (node, (head, relation)) in &host {
        if is_replaced(*node) {
            if !block_written {
                block_written = true;
                for (chunk, head) in &block {
                    match head {
                        None => edges.push((*chunk, attached_to, root_relation.clone())),
                        Some(head) => edges.push((*chunk, *head, "DEP".to_owned())),
                    }
                }
            }
            continue;
        }
        let head = match is_replaced(*head) {
            true => redirect[head].ok_or(Refusal::NoUniqueHeadChunk)?,
            false => *head,
        };
        edges.push((*node, head, relation.clone()));
    }
    // Numbers, at last: each node's position in the final order.
    let number: BTreeMap<Node, usize> = edges
        .iter()
        .enumerate()
        .map(|(at, (node, _, _))| (*node, at + 1))
        .chain(std::iter::once((Node::Root, 0)))
        .collect();
    Ok((
        applied,
        edges
            .iter()
            .map(|(node, head, relation)| (number[node], number[head], relation.clone()))
            .collect(),
    ))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4096))]

    /// The splice writes what the symbolic model writes, or refuses where
    /// and why the model refuses.
    #[test]
    fn the_splice_agrees_with_a_symbolic_graph_model(case in case()) {
        prop_assert_eq!(production(&case), model(&case), "{:?}", case);
    }
}
