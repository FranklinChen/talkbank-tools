//! Structural merge: combine the primary model's attachment with the
//! secondary model's analysis of an `@s` span.
//!
//! Ownership is split by what each model can know. The secondary model
//! knows the language: it owns every word's lexical category, lemma and
//! features. The primary model knows the host utterance: it owns where the
//! span attaches and with which relation. Where the primary's relation
//! contradicts the secondary's category, the RELATION is corrected (and
//! the correction recorded); the category is never changed to fit it.
//! The `%mor` items are the `%mor` mapper's rendering of the secondary
//! analysis (with its language-specific overrides, such as Italian's);
//! the one category the merge itself writes is [`ModelAssignedPos`], which
//! can be built only from the secondary analysis.

use talkbank_model::alignment::MorItemIndex;
use talkbank_model::model::LanguageCode;
use talkbank_model::model::dependent_tier::GrammaticalRelation;
use talkbank_model::model::dependent_tier::mor::{Mor, PosCategory};

use super::deprel::{deprel_to_pos_constraint, infer_deprel_from_pos};
use super::plan::{AsPlanned, ExternalRelation, L2Attachment, L2SpanPlan};
use crate::morphosyntax::alignment::{
    AlignedUd, AlignedWord, SpanWordPosition, UdAlignment, UdAlignmentError,
};
use crate::morphosyntax::{
    ItemLayout, MappingContext, MappingError, UdPunctable, UdSentence, UdWord, UniversalPos,
    map_tokens,
};

/// Sentence-level context from the secondary model for a single `@s`
/// word being merged.
///
/// The mapped `Mor` of a span word carries the secondary's UPOS and lemma
/// for that word, but not the structural evidence that identifies a
/// phrasal-verb particle (the `compound:prt` relation). This context is the word's place in the
/// secondary span's [`UdAlignment`], so it is available for every span
/// word: it is built from the alignment, by UD id, and never from a
/// position in `UdSentence::words`.
#[derive(Debug, Clone, Copy)]
pub struct SecondaryUdContext<'a, 's> {
    word: AlignedWord<'a, 's, SpanWordPosition>,
}

impl<'a, 's> SecondaryUdContext<'a, 's> {
    /// The context of one aligned span word.
    pub fn of(word: AlignedWord<'a, 's, SpanWordPosition>) -> Self {
        Self { word }
    }

    /// Whether the current word is the particle of a phrasal verb: the
    /// secondary attached it `compound:prt` to a word it tagged VERB.
    ///
    /// `compound:prt` is Stanza's signal for a verb + particle construction
    /// (`wake up`, `give up`, `figure out`). Recognised by the relation, not
    /// the tag: Stanza tags the English particle `ADP`. The head must be a
    /// verb: Stanza also attaches `out` `compound:prt` to the NOUN `time` in
    /// `time out`, which is a compound noun, and `out` keeps its ADP. A
    /// multi-word token is never a particle: its `%mor` item is the
    /// assembled clitic group.
    pub fn is_phrasal_verb_particle(&self) -> bool {
        match self.word.ud() {
            AlignedUd::Word(word) => {
                is_compound_prt(&word.deprel)
                    && self.word.head_word().is_some_and(|head| {
                        matches!(head.upos, UdPunctable::Value(UniversalPos::Verb))
                    })
            }
            AlignedUd::Mwt { .. } => false,
        }
    }

    /// The category the secondary model gave this word: PART for a
    /// phrasal-verb particle, otherwise the tag of the word (for a
    /// multi-word token, of its representative).
    pub fn model_pos(&self) -> ModelAssignedPos {
        if self.is_phrasal_verb_particle() {
            ModelAssignedPos {
                pos: UniversalPos::Part,
                source: PosSource::PhrasalParticle,
            }
        } else {
            ModelAssignedPos::tag_of(self.word.representative())
        }
    }
}

/// A part of speech a model assigned to an `@s` word.
///
/// The only category the merge writes or reasons with (the `%mor` mapper
/// renders the rest from the same analysis). Both constructors
/// read the secondary model's own analysis of the span: the tag of a word
/// ([`PosSource::SecondaryTag`]), or PART for a word it attached as a
/// phrasal-verb particle ([`PosSource::PhrasalParticle`]; Stanza tags the
/// English particle ADP). No rule over the primary's relation can build
/// one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelAssignedPos {
    pos: UniversalPos,
    source: PosSource,
}

/// Which part of the secondary analysis a [`ModelAssignedPos`] came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PosSource {
    /// The UPOS column of the word.
    SecondaryTag,
    /// The `compound:prt` relation of the word.
    PhrasalParticle,
}

impl ModelAssignedPos {
    /// The secondary's tag of one UD word. A raw punctuation tag reads as
    /// PUNCT, as the `%mor` mapper reads it.
    fn tag_of(word: &UdWord) -> Self {
        let pos = match word.upos {
            UdPunctable::Value(pos) => pos,
            UdPunctable::Punct(_) => UniversalPos::Punct,
        };
        Self {
            pos,
            source: PosSource::SecondaryTag,
        }
    }

    /// The category.
    pub fn pos(self) -> UniversalPos {
        self.pos
    }

    /// Where in the secondary analysis it came from.
    pub fn source(self) -> PosSource {
        self.source
    }
}

/// Why a planned secondary span cannot be merged. The span's words fall
/// back to `L2|xxx`, and the caller reports this error.
#[derive(Debug, thiserror::Error)]
pub enum L2MergeError {
    /// The secondary analysis does not align to the span's words.
    #[error("secondary analysis does not align to the span: {0}")]
    Alignment(#[from] UdAlignmentError),
    /// The secondary analysis could not be mapped to `%mor`/`%gra`.
    #[error("secondary analysis does not map: {0}")]
    Mapping(#[from] MappingError),
    /// The secondary analysis has no root word.
    #[error("secondary analysis has no root word")]
    NoSecondaryRoot,
}

/// Deprel-base comparison against the UD phrasal-verb particle
/// relation. Matches `compound:prt` exactly; the only other
/// `compound:*` subtype we have seen in Stanza output is `compound:svc`
/// (serial verb construction), which has a different semantics and
/// must not trigger phrasal-verb promotion.
fn is_compound_prt(deprel: &str) -> bool {
    deprel == "compound:prt"
}

/// A planned span merged with its secondary analysis, ready to splice.
///
/// Made from an [`L2SpanPlan`] by [`merge_planned_secondary_span`], which
/// consumes the plan, and consumed by the splice.
#[derive(Debug, Clone)]
pub struct MergedL2Span {
    /// Owning line in the `ChatFile`.
    pub(super) line_idx: usize,
    /// The language the span was analysed in (for reports).
    pub(super) target_lang: LanguageCode,
    /// The host word the span starts at; the span covers
    /// `first_word .. first_word + mors.len()`.
    pub(super) first_word: MorItemIndex,
    /// One `%mor` item per span word, in order: the secondary's items as
    /// `map_ud_sentence` mapped them (`it's` is `pron|it~aux|be`), with only
    /// a phrasal particle's category rewritten to PART.
    pub(super) mors: Vec<Mor>,
    /// The secondary's relations for the chunks of `mors`, unchanged: heads
    /// are 1-based within the span, `0` for the secondary root. Inside the
    /// span these are authoritative.
    pub(super) gras: Vec<GrammaticalRelation>,
    /// How the secondary root attaches to the host, with the span's one
    /// external relation (the primary's, or its correction).
    pub(super) attachment: L2Attachment<ExternalRelation>,
}

impl MergedL2Span {
    /// The span's `%mor` items.
    pub fn mors(&self) -> &[Mor] {
        &self.mors
    }

    /// The span's secondary relations (span-relative heads).
    pub fn gras(&self) -> &[GrammaticalRelation] {
        &self.gras
    }

    /// How the span root attaches to the host.
    pub fn attachment(&self) -> &L2Attachment<ExternalRelation> {
        &self.attachment
    }

    /// The host word the span starts at.
    pub fn first_word(&self) -> MorItemIndex {
        self.first_word
    }

    /// A span built directly, for splice tests that feed shapes no
    /// secondary analysis produces.
    #[cfg(test)]
    pub(super) fn for_splice_test(
        line_idx: usize,
        first_word: MorItemIndex,
        mors: Vec<Mor>,
        gras: Vec<GrammaticalRelation>,
        attachment: L2Attachment<ExternalRelation>,
    ) -> Self {
        Self {
            line_idx,
            target_lang: LanguageCode::new("spa").expect("valid language code"),
            first_word,
            mors,
            gras,
            attachment,
        }
    }
}

/// Whether the entry at `start` reaches a `head=0` row by following
/// head pointers within `chunk_count + 1` hops. Returns false on
/// cycles and self-loops: the conditions
/// [`repair_secondary_gras`]'s pass 4 needs to detect.
fn entry_reaches_root_via_heads(
    gras: &[GrammaticalRelation],
    start: usize,
    chunk_count: usize,
) -> bool {
    let mut current = start;
    for _ in 0..=chunk_count {
        let head = gras[current].head;
        if head == 0 {
            return true;
        }
        let next = head - 1; // 1-indexed → 0-indexed
        if next == current {
            return false; // self-loop
        }
        current = next;
    }
    false
}

/// Repair a span-aggregated secondary gras slice so the merged
/// result can be admitted as a `SplicedBlock`. Admission, not repair,
/// supplies the proof of a one-rooted tree and its block-relative root.
///
/// Runs at the splice, once per span, over the span's whole relation list:
/// heads are span-relative, so one word's relation may point at another
/// word of the span.
///
/// Four passes, each reading and updating the one root found so far:
/// 1. Clamp OOB (`head > chunk_count` becomes `head=0`).
///    Catches `secondary_head_oob`.
/// 2. The first `head=0` row is the root (labelled `ROOT`); every later
///    one attaches to it as `DEP`. Catches `secondary_multi_root`.
/// 3. With no root and an [`L2Attachment::InternalRoot`] attachment, the
///    first row becomes the root. No-op for an external attachment;
///    subsequent block admission refuses any remaining rootless block. Catches
///    `secondary_no_root`.
/// 4. A row that does not reach the root (a cycle) attaches to the root,
///    or becomes it when there is none yet. Catches `secondary_cycle`.
pub(super) fn repair_secondary_gras<R>(
    gras: &mut [GrammaticalRelation],
    attachment: &L2Attachment<R>,
) {
    let chunk_count = gras.len();

    // Pass 1.
    for rel in gras.iter_mut() {
        if rel.head > chunk_count {
            rel.head = 0;
        }
    }

    // Pass 2.
    let mut root: Option<usize> = None;
    for (i, relation) in gras.iter_mut().enumerate() {
        if relation.head != 0 {
            continue;
        }
        match root {
            None => {
                relation.relation = "ROOT".into();
                root = Some(i);
            }
            Some(first) => {
                relation.head = first + 1; // 1-indexed
                relation.relation = "DEP".into();
            }
        }
    }

    // Pass 3.
    if root.is_none()
        && chunk_count > 0
        && let L2Attachment::InternalRoot = attachment
    {
        gras[0].head = 0;
        gras[0].relation = "ROOT".into();
        root = Some(0);
    }

    // Pass 4.
    for i in 0..chunk_count {
        if gras[i].head == 0 || entry_reaches_root_via_heads(gras, i, chunk_count) {
            continue;
        }
        match root {
            Some(root) => {
                gras[i].head = root + 1;
                gras[i].relation = "DEP".into();
            }
            None => {
                gras[i].head = 0;
                gras[i].relation = "ROOT".into();
                root = Some(i);
            }
        }
    }
}

/// Merge one planned span with the secondary model's analysis of it.
///
/// Consumes the plan. The secondary sentence is aligned to the span's words
/// first ([`UdAlignment`]); the merge then takes, from the secondary, every
/// word's `%mor` item and every relation inside the span, unchanged except
/// that a phrasal particle is written PART; and from the plan, the one
/// external relation, checked against the category of the secondary root
/// (the word that carries it) and corrected where it contradicts it and a
/// correction is implied (`infer_deprel_from_pos`); where none is implied
/// the primary's relation stands.
///
/// A sentence that does not align, does not map, or has no root is an
/// error for the whole span; its words stay `L2|xxx`.
pub fn merge_planned_secondary_span(
    span: L2SpanPlan,
    sentence: &UdSentence,
) -> Result<MergedL2Span, L2MergeError> {
    let words = span.len();
    let alignment = UdAlignment::<SpanWordPosition>::new(sentence, words)?;
    let mapping_ctx = MappingContext {
        lang: span.target_lang.clone(),
    };
    // The mapper consumes the alignment's own walk: one item per aligned
    // span word, so the two cannot disagree on which UD words a word covers.
    let (mors, mut gras) =
        map_tokens(alignment.tokens(), &mapping_ctx, ItemLayout::PerToken)?.into_parts();
    let total_chunks: usize = mors.iter().map(Mor::count_chunks).sum();
    // The mapper appends the secondary terminator's relation; the span's
    // words end before it.
    gras.truncate(total_chunks);

    let root = alignment
        .root()
        .map(|(_, word)| ModelAssignedPos::tag_of(word))
        .ok_or(L2MergeError::NoSecondaryRoot)?;

    // Two sequences of one length (one item per aligned token), walked
    // together rather than indexed.
    let mors = alignment
        .words()
        .zip(mors)
        .map(|(word, mor)| with_model_pos(mor, SecondaryUdContext::of(word).model_pos()))
        .collect();

    let attachment = external_relation(span.attachment, root);

    Ok(MergedL2Span {
        line_idx: span.line_idx,
        target_lang: span.target_lang,
        first_word: span.first_word,
        mors,
        gras,
        attachment,
    })
}

/// A span word's `%mor` item with its model-assigned category.
///
/// The mapped item already carries the secondary's tag; only a phrasal
/// particle is rewritten (to PART).
fn with_model_pos(mut mor: Mor, pos: ModelAssignedPos) -> Mor {
    match pos.source() {
        PosSource::SecondaryTag => {}
        PosSource::PhrasalParticle => {
            mor.main.pos = PosCategory::new(pos.pos().to_chat_pos_name());
        }
    }
    mor
}

/// The span's external relation, corrected where the primary's relation
/// contradicts the category of the secondary root that carries it.
///
/// Only a host-governed attachment has a relation from the primary to
/// check: an utterance-root attachment's relation is `root`, and an
/// internal root has none. The correction is recorded, never applied to
/// the category. A contradiction no rule corrects (a VERB root under
/// `obj`) keeps the primary's relation; the book lists this as a
/// limitation.
fn external_relation(
    attachment: L2Attachment<AsPlanned>,
    root: ModelAssignedPos,
) -> L2Attachment<ExternalRelation> {
    match attachment {
        L2Attachment::HostGovernor {
            source_word,
            source,
            relation: AsPlanned,
        } => {
            let primary = source.deprel();
            let contradicts = primary.base() == "flat"
                || !deprel_to_pos_constraint(primary).contains(&root.pos());
            let relation = match contradicts {
                true => infer_deprel_from_pos(
                    root.pos(),
                    source.head_upos(),
                    source.has_case_dependent(),
                )
                .map_or(ExternalRelation::Primary, ExternalRelation::Corrected),
                false => ExternalRelation::Primary,
            };
            L2Attachment::HostGovernor {
                source_word,
                source,
                relation,
            }
        }
        L2Attachment::UtteranceRoot { source_word } => L2Attachment::UtteranceRoot { source_word },
        L2Attachment::InternalRoot => L2Attachment::InternalRoot,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphosyntax::alignment::HeadTarget;
    use crate::morphosyntax::l2::PrimaryStructuralInfo;
    use crate::morphosyntax::l2::deprel::UdDeprel;
    use crate::morphosyntax::l2::pipeline_tests::ud_sentence;
    use crate::morphosyntax::l2::{L2DeferredPosition, plan_dispatch_spans};

    /// One planned span of `@s` words at host words `first..`, each
    /// attached by the primary as `(deprel, head)` (`None` for the root).
    fn planned_span(first: usize, words: &[(&str, &str, Option<usize>)]) -> L2SpanPlan {
        let deferred = words
            .iter()
            .enumerate()
            .map(|(offset, (word, deprel, head))| {
                L2DeferredPosition::for_test(
                    3,
                    MorItemIndex::new(first + offset),
                    "spa",
                    word,
                    PrimaryStructuralInfo::for_test(
                        deprel,
                        head.map_or(HeadTarget::Root, |word| {
                            HeadTarget::Word(MorItemIndex::new(word))
                        }),
                        Some(UniversalPos::Verb),
                    ),
                )
            })
            .collect();
        let mut plan = plan_dispatch_spans(deferred);
        assert_eq!(plan.spans.len(), 1, "one contiguous span");
        plan.spans.remove(0)
    }

    /// The merge keeps the secondary's relations inside the span, writes a
    /// phrasal particle PART, and carries the planned attachment.
    #[test]
    fn merge_keeps_secondary_relations_and_carries_the_attachment() {
        let span = planned_span(1, &[("wake", "obj", Some(0)), ("up", "obj", Some(0))]);
        let sentence = ud_sentence(
            "1 wake wake VERB _ 0 root
             2 up up ADP _ 1 compound:prt
             3 . . PUNCT _ 1 punct",
        );

        let merged = merge_planned_secondary_span(span, &sentence).expect("the span merges");

        let pos: Vec<&str> = merged.mors().iter().map(|m| m.main.pos.as_str()).collect();
        assert_eq!(pos, ["verb", "part"]);
        let gras: Vec<String> = merged.gras().iter().map(ToString::to_string).collect();
        assert_eq!(gras, ["1|0|ROOT", "2|1|COMPOUND-PRT"]);
        assert!(matches!(
            merged.attachment(),
            L2Attachment::HostGovernor { source_word, source, relation: ExternalRelation::Primary }
                if *source_word == MorItemIndex::new(1) && source.deprel().as_str() == "obj"
        ));
        assert_eq!(merged.first_word(), MorItemIndex::new(1));
    }

    /// The external relation is corrected against the secondary root's
    /// category, and the correction is recorded.
    #[test]
    fn a_contradicting_external_relation_is_corrected_and_recorded() {
        // The primary says `advmod`; the secondary root is a NOUN under a
        // verb, so the relation becomes `obj`.
        let span = planned_span(1, &[("camino", "advmod", Some(0))]);
        let sentence = ud_sentence(
            "1 camino camino NOUN _ 0 root
             2 . . PUNCT _ 1 punct",
        );

        let merged = merge_planned_secondary_span(span, &sentence).expect("the span merges");

        assert_eq!(merged.mors()[0].main.pos.as_str(), "noun");
        assert_eq!(
            merged.attachment().corrected_deprel(),
            Some(&UdDeprel::new("obj"))
        );
        assert_eq!(
            merged.attachment().external_root_deprel(),
            Some(UdDeprel::new("obj"))
        );
    }

    /// A secondary analysis with more words than the span is an error for
    /// the span, not a partial merge.
    #[test]
    fn a_secondary_analysis_that_does_not_align_is_an_error() {
        let span = planned_span(1, &[("extranjero", "obj", Some(0))]);
        let sentence = ud_sentence(
            "1 extran extran NOUN _ 0 root
             2 jero jero NOUN _ 1 flat",
        );

        assert!(matches!(
            merge_planned_secondary_span(span, &sentence),
            Err(L2MergeError::Alignment(
                UdAlignmentError::WordCountMismatch {
                    chat_words: 1,
                    ud_tokens: 2
                }
            ))
        ));
    }

    /// A span word's category is the secondary's tag, PART for a word the
    /// secondary attached as a phrasal particle, and a contraction's is its
    /// representative's. Nothing else can produce one.
    #[test]
    fn model_pos_reads_only_the_secondary_analysis() {
        let sentence = ud_sentence(
            "1-2 it's _ X _ 0 dep
             1 it it PRON _ 3 nsubj
             2 's be AUX _ 3 aux
             3 wake wake VERB _ 0 root
             4 up up ADP _ 3 compound:prt",
        );
        let alignment =
            UdAlignment::<SpanWordPosition>::new(&sentence, 3).expect("the sentence aligns");
        let pos: Vec<(UniversalPos, PosSource)> = alignment
            .words()
            .map(|word| {
                let pos = SecondaryUdContext::of(word).model_pos();
                (pos.pos(), pos.source())
            })
            .collect();
        assert_eq!(
            pos,
            [
                (UniversalPos::Pron, PosSource::SecondaryTag),
                (UniversalPos::Verb, PosSource::SecondaryTag),
                (UniversalPos::Part, PosSource::PhrasalParticle),
            ]
        );
    }

    /// A `compound:prt` word whose head is not a verb is no phrasal-verb
    /// particle: in `time out` (Stanza: `out` `compound:prt` under the NOUN
    /// `time`) `out` keeps the secondary's ADP.
    #[test]
    fn a_particle_of_a_noun_keeps_its_tag() {
        let sentence = ud_sentence(
            "1 time time NOUN _ 0 root
             2 out out ADP _ 1 compound:prt
             3 . . PUNCT _ 1 punct",
        );
        let alignment =
            UdAlignment::<SpanWordPosition>::new(&sentence, 2).expect("the sentence aligns");
        let out = alignment.words().nth(1).expect("span word");
        let pos = SecondaryUdContext::of(out).model_pos();
        assert_eq!(
            (pos.pos(), pos.source()),
            (UniversalPos::Adp, PosSource::SecondaryTag)
        );
    }

    /// Repair leaves at most one root: with two separate cycles and no root,
    /// the first cycle's promoted root is the root the second attaches to.
    #[test]
    fn repair_leaves_at_most_one_root() {
        let mut gras = vec![
            GrammaticalRelation::new(1, 2, "DEP"),
            GrammaticalRelation::new(2, 1, "DEP"),
            GrammaticalRelation::new(3, 4, "DEP"),
            GrammaticalRelation::new(4, 3, "DEP"),
        ];
        let attachment: L2Attachment = L2Attachment::UtteranceRoot {
            source_word: MorItemIndex::new(0),
        };
        repair_secondary_gras(&mut gras, &attachment);
        let roots = gras.iter().filter(|rel| rel.head == 0).count();
        assert_eq!(roots, 1, "{gras:?}");
        assert_eq!(gras[0].head, 0);
    }
}
