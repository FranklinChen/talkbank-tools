//! Transform-layer planning for secondary L2 dispatch.
//!
//! `batchalign` should only obtain primary/secondary UD analyses. The stable
//! CHAT-specific planning seam lives here: contiguous span grouping and host
//! attachment planning.
//!
//! The plan is a step in a chain of owned types, each made from the one
//! before: deferred positions ([`L2DeferredPosition`]) become planned spans
//! ([`L2SpanPlan`], which own their positions), a planned span and its
//! secondary analysis become a merged span (`MergedL2Span`), and the splice
//! consumes merged spans. No step refers back to an earlier one by a bare
//! index.

use talkbank_model::alignment::MorItemIndex;
use talkbank_model::model::LanguageCode;

use super::deprel::UdDeprel;
use super::extract::{L2DeferredPosition, PrimaryStructuralInfo};
use crate::morphosyntax::alignment::HeadTarget;

/// How a secondary span's root attaches to the host utterance.
///
/// Decided from the primary analysis of the span's words: the first word
/// whose primary head lies outside the span is the ATTACHMENT SOURCE. The
/// span root takes its place in the host tree.
///
/// `R` is the stage of the external relation: [`AsPlanned`] in a plan (the
/// source's primary relation, with no field for anything else), and
/// [`ExternalRelation`] once merged (the primary's, or the merge's
/// correction). So a plan cannot carry a correction, and a merged span says
/// which one it has.
#[derive(Debug, Clone, PartialEq)]
pub enum L2Attachment<R = AsPlanned> {
    /// No span word attaches outside the span (only a cyclic primary
    /// analysis gets here); the secondary root stays a root.
    InternalRoot,
    /// The span root attaches under the host word the source's primary head
    /// is, with the relation `relation` gives.
    HostGovernor {
        /// The span word whose primary head lies outside the span.
        source_word: MorItemIndex,
        /// The primary's structure for that word.
        source: PrimaryStructuralInfo,
        /// The external relation at this stage.
        relation: R,
    },
    /// The source is the primary's utterance root, so the span root is the
    /// utterance root. Its relation is `root` by definition: there is no
    /// field to hold any other.
    UtteranceRoot {
        /// The span word the primary made the utterance root.
        source_word: MorItemIndex,
    },
}

/// The stage of a host-governed span's external relation: which relation
/// the span root takes, given the attachment source's primary structure.
pub trait RelationStage: Clone + std::fmt::Debug + PartialEq {
    /// The relation, read with the source it qualifies.
    fn relation<'a>(&'a self, source: &'a PrimaryStructuralInfo) -> &'a UdDeprel;
}

/// A planned attachment's relation: the source's primary relation, as the
/// primary gave it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AsPlanned;

impl RelationStage for AsPlanned {
    fn relation<'a>(&'a self, source: &'a PrimaryStructuralInfo) -> &'a UdDeprel {
        source.deprel()
    }
}

/// A merged attachment's relation.
#[derive(Debug, Clone, PartialEq)]
pub enum ExternalRelation {
    /// The source's primary relation stands.
    Primary,
    /// The merge corrected it against the secondary root's category.
    Corrected(UdDeprel),
}

impl RelationStage for ExternalRelation {
    fn relation<'a>(&'a self, source: &'a PrimaryStructuralInfo) -> &'a UdDeprel {
        match self {
            Self::Primary => source.deprel(),
            Self::Corrected(deprel) => deprel,
        }
    }
}

impl<R: RelationStage> L2Attachment<R> {
    /// The relation the span root gets in the host utterance, if it
    /// attaches there.
    pub fn external_root_deprel(&self) -> Option<UdDeprel> {
        match self {
            Self::InternalRoot => None,
            Self::HostGovernor {
                relation, source, ..
            } => Some(relation.relation(source).clone()),
            Self::UtteranceRoot { .. } => Some(UdDeprel::new("root")),
        }
    }

    /// Whether the span root attaches to the host utterance.
    pub fn is_external_root(&self) -> bool {
        match self {
            Self::InternalRoot => false,
            Self::HostGovernor { .. } | Self::UtteranceRoot { .. } => true,
        }
    }

    /// The span word that decided the attachment.
    pub fn source_word(&self) -> Option<MorItemIndex> {
        match self {
            Self::InternalRoot => None,
            Self::HostGovernor { source_word, .. } | Self::UtteranceRoot { source_word } => {
                Some(*source_word)
            }
        }
    }
}

impl L2Attachment<ExternalRelation> {
    /// The merge's correction of the external relation, if it made one.
    pub fn corrected_deprel(&self) -> Option<&UdDeprel> {
        match self {
            Self::HostGovernor {
                relation: ExternalRelation::Corrected(deprel),
                ..
            } => Some(deprel),
            Self::HostGovernor {
                relation: ExternalRelation::Primary,
                ..
            }
            | Self::InternalRoot
            | Self::UtteranceRoot { .. } => None,
        }
    }
}

/// One contiguous secondary-dispatch span plus its planned host attachment.
///
/// Owns its deferred positions: consecutive host words of one utterance,
/// one target language, at least one word. Built only by
/// [`plan_dispatch_spans`].
#[derive(Debug, Clone, PartialEq)]
pub struct L2SpanPlan {
    pub(super) line_idx: usize,
    pub(super) target_lang: LanguageCode,
    pub(super) terminator: talkbank_model::Terminator,
    /// The host word of `positions[0]`, stored so no reader handles an
    /// empty span.
    pub(super) first_word: MorItemIndex,
    pub(super) positions: Vec<L2DeferredPosition>,
    pub(super) attachment: L2Attachment,
}

impl L2SpanPlan {
    /// Owning line in the `ChatFile`.
    pub fn line_idx(&self) -> usize {
        self.line_idx
    }

    /// Target language to dispatch this span to.
    pub fn target_lang(&self) -> &LanguageCode {
        &self.target_lang
    }

    /// Terminator of the utterance this span came from.
    ///
    /// An INPUT to the secondary Stanza model, which changes its analysis
    /// when sentence-final punctuation is absent or wrong.
    pub fn terminator(&self) -> &talkbank_model::Terminator {
        &self.terminator
    }

    /// The span's deferred positions, in host order.
    pub fn positions(&self) -> &[L2DeferredPosition] {
        &self.positions
    }

    /// Provenance-sealed word texts to send to the secondary model.
    pub fn words(&self) -> impl Iterator<Item = &talkbank_model::ChatCleanedText> {
        self.positions.iter().map(L2DeferredPosition::word)
    }

    /// Number of words in the span (at least one).
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether the span has no words; producer-admitted plans are nonempty.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// The host word the span starts at.
    pub fn first_word(&self) -> MorItemIndex {
        self.first_word
    }

    /// Explicit host-attachment plan for the span's secondary root.
    pub fn attachment(&self) -> &L2Attachment {
        &self.attachment
    }
}

/// Full secondary-dispatch plan for one utterance batch.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct L2DispatchPlan {
    /// Planned spans in transcript order.
    pub spans: Vec<L2SpanPlan>,
}

/// The attachment of a span, from its words' primary heads.
fn planned_attachment(positions: &[L2DeferredPosition]) -> L2Attachment {
    let covers = |head: MorItemIndex| positions.iter().any(|p| p.word_idx() == head);
    let mut utterance_root = None;

    for current in positions {
        match current.primary().head() {
            // The primary attached this word to a host word outside the
            // span: it is the span's attachment source.
            HeadTarget::Word(head) if !covers(head) => {
                return L2Attachment::HostGovernor {
                    source_word: current.word_idx(),
                    source: current.primary().clone(),
                    relation: AsPlanned,
                };
            }
            // Attached inside the span: the secondary parse governs it.
            HeadTarget::Word(_) => {}
            // The primary's utterance root. A host-governed word later in
            // the span still wins, so only the first root is remembered.
            HeadTarget::Root => {
                utterance_root.get_or_insert(current.word_idx());
            }
        }
    }

    match utterance_root {
        Some(source_word) => L2Attachment::UtteranceRoot { source_word },
        None => L2Attachment::InternalRoot,
    }
}

/// Whether host word `next` immediately follows host word `previous`.
pub(super) fn follows(previous: MorItemIndex, next: MorItemIndex) -> bool {
    previous.as_usize() + 1 == next.as_usize()
}

/// Plan contiguous secondary-dispatch spans from deferred positions,
/// consuming them: each position moves into exactly one span.
///
/// A span is a run of positions on one line, in one target language, at
/// consecutive host words.
pub fn plan_dispatch_spans(deferred: Vec<L2DeferredPosition>) -> L2DispatchPlan {
    let mut runs: Vec<Vec<L2DeferredPosition>> = Vec::new();
    for position in deferred {
        let extends = |run: &Vec<L2DeferredPosition>| {
            run.last().is_some_and(|previous| {
                previous.line_idx() == position.line_idx()
                    && previous.target_lang() == position.target_lang()
                    && follows(previous.word_idx(), position.word_idx())
            })
        };
        match runs.last_mut() {
            Some(run) if extends(run) => run.push(position),
            Some(_) | None => runs.push(vec![position]),
        }
    }

    let spans = runs
        .into_iter()
        .filter_map(|positions| {
            let first = positions.first()?;
            Some(L2SpanPlan {
                line_idx: first.line_idx(),
                target_lang: first.target_lang().clone(),
                terminator: first.terminator().clone(),
                first_word: first.word_idx(),
                attachment: planned_attachment(&positions),
                positions,
            })
        })
        .collect();

    L2DispatchPlan { spans }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphosyntax::UniversalPos;
    use crate::morphosyntax::l2::PrimaryStructuralInfo;

    /// A deferred position at host word `word_idx`, attached by the
    /// primary to host word `head` (`None` for the utterance root).
    fn make_deferred(
        line_idx: usize,
        word_idx: usize,
        lang: &str,
        deprel: &str,
        head: Option<usize>,
        word: &str,
    ) -> L2DeferredPosition {
        L2DeferredPosition::for_test(
            line_idx,
            MorItemIndex::new(word_idx),
            lang,
            word,
            PrimaryStructuralInfo::for_test(
                deprel,
                head.map_or(HeadTarget::Root, |word| {
                    HeadTarget::Word(MorItemIndex::new(word))
                }),
                Some(UniversalPos::Verb),
            ),
        )
    }

    /// The span's words, as text.
    fn words(span: &L2SpanPlan) -> Vec<&str> {
        span.words().map(|word| word.as_str()).collect()
    }

    #[test]
    fn plan_dispatch_spans_tracks_external_attachment_for_contiguous_span() {
        let deferred = vec![
            make_deferred(5, 1, "spa", "obj", Some(0), "los"),
            make_deferred(5, 2, "spa", "obl", Some(0), "ninos"),
        ];

        let plan = plan_dispatch_spans(deferred);
        assert_eq!(plan.spans.len(), 1);
        let span = &plan.spans[0];
        assert_eq!(span.line_idx(), 5);
        assert_eq!(span.first_word(), MorItemIndex::new(1));
        assert_eq!(words(span), ["los", "ninos"]);
        assert!(matches!(
            span.attachment(),
            L2Attachment::HostGovernor { source_word, source, relation: AsPlanned }
                if *source_word == MorItemIndex::new(1) && source.deprel().as_str() == "obj"
        ));
    }

    #[test]
    fn plan_dispatch_spans_separates_noncontiguous_same_language_words() {
        let deferred = vec![
            make_deferred(5, 1, "spa", "obj", Some(0), "uno"),
            make_deferred(5, 3, "spa", "obl", Some(0), "dos"),
        ];

        let plan = plan_dispatch_spans(deferred);
        assert_eq!(plan.spans.len(), 2);
        assert_eq!(words(&plan.spans[0]), ["uno"]);
        assert_eq!(words(&plan.spans[1]), ["dos"]);
    }

    /// A span whose earlier word is the primary's utterance root and whose
    /// later word attaches to a host word attaches through the later word.
    #[test]
    fn plan_dispatch_spans_prefers_real_host_attachment_over_earlier_primary_root_noise() {
        let deferred = vec![
            make_deferred(5, 1, "spa", "root", None, "uno"),
            make_deferred(5, 2, "spa", "obj", Some(5), "dos"),
        ];

        let plan = plan_dispatch_spans(deferred);
        assert_eq!(plan.spans.len(), 1);
        assert!(matches!(
            plan.spans[0].attachment(),
            L2Attachment::HostGovernor { source_word, source, relation: AsPlanned }
                if *source_word == MorItemIndex::new(2) && source.deprel().as_str() == "obj"
        ));
    }

    #[test]
    fn plan_dispatch_spans_preserves_utterance_root_when_no_host_governor_exists() {
        let deferred = vec![
            make_deferred(5, 1, "spa", "dep", Some(2), "uno"),
            make_deferred(5, 2, "spa", "root", None, "dos"),
        ];

        let plan = plan_dispatch_spans(deferred);
        assert_eq!(plan.spans.len(), 1);
        assert_eq!(
            plan.spans[0].attachment(),
            &L2Attachment::UtteranceRoot {
                source_word: MorItemIndex::new(2)
            }
        );
    }

    /// A word the primary made the utterance root plans as `UtteranceRoot`
    /// whatever label the primary gave it, and that attachment's relation
    /// is `root`: the variant has no field for another one. (Copying the
    /// primary's label here produced `head=0` with `DET`/`NMOD`, E722.)
    #[test]
    fn a_primary_root_word_plans_as_utterance_root_with_relation_root() {
        let plan = plan_dispatch_spans(vec![make_deferred(5, 1, "spa", "det", None, "el")]);

        let attachment = plan.spans[0].attachment();
        assert_eq!(
            attachment,
            &L2Attachment::UtteranceRoot {
                source_word: MorItemIndex::new(1)
            }
        );
        assert_eq!(
            attachment.external_root_deprel(),
            Some(UdDeprel::new("root"))
        );
    }

    /// A planned span carries the terminator of the positions it groups.
    ///
    /// The utterance-level source of that terminator is tested where it
    /// enters, in `extract`; here it only has to survive grouping.
    #[test]
    fn planned_span_carries_the_positions_terminator() {
        let def = make_deferred(5, 1, "spa", "obj", Some(0), "camino").with_terminator(
            talkbank_model::Terminator::Question {
                span: talkbank_model::Span::DUMMY,
            },
        );

        let plan = plan_dispatch_spans(vec![def]);

        assert!(
            matches!(
                plan.spans[0].terminator(),
                talkbank_model::Terminator::Question { .. }
            ),
            "grouping must not substitute a period; got {:?}",
            plan.spans[0].terminator()
        );
    }

    /// Positions on different lines are different spans.
    #[test]
    fn plan_dispatch_spans_separates_utterances() {
        let plan = plan_dispatch_spans(vec![
            make_deferred(3, 5, "eng", "obj", Some(0), "film"),
            make_deferred(7, 2, "eng", "obj", Some(0), "studies"),
        ]);
        assert_eq!(plan.spans.len(), 2);
    }

    /// Adjacent positions in different target languages are different
    /// spans, in transcript order.
    #[test]
    fn plan_dispatch_spans_separates_languages() {
        let plan = plan_dispatch_spans(vec![
            make_deferred(5, 2, "spa", "obj", Some(0), "tienda"),
            make_deferred(5, 3, "fra", "obj", Some(0), "bonjour"),
        ]);
        let languages: Vec<&str> = plan
            .spans
            .iter()
            .map(|span| span.target_lang().as_str())
            .collect();
        assert_eq!(languages, ["spa", "fra"]);
    }

    /// Three consecutive positions are one span of three words.
    #[test]
    fn plan_dispatch_spans_groups_three_consecutive_words() {
        let plan = plan_dispatch_spans(vec![
            make_deferred(10, 4, "eng", "amod", Some(6), "full"),
            make_deferred(10, 5, "eng", "amod", Some(6), "English"),
            make_deferred(10, 6, "eng", "obj", Some(0), "breakfast"),
        ]);
        assert_eq!(plan.spans.len(), 1);
        assert_eq!(words(&plan.spans[0]), ["full", "English", "breakfast"]);
        assert_eq!(plan.spans[0].len(), 3);
        assert!(!plan.spans[0].is_empty());
    }
}
