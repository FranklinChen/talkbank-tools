//! Splice merged L2 morphology back into a ChatFile.
//!
//! Overwrites `L2|xxx` MOR items with pre-mapped `Mor` items from the
//! structural merge algorithm, and optionally corrects GRA deprels.

use super::merge::MergedL2Span;
use super::plan::{L2Attachment, RelationStage};
use talkbank_model::alignment::{GraHeadRef, MorItemIndex};
use talkbank_model::model::dependent_tier::mor::tier::CoordinatedMutationError;
use talkbank_model::model::dependent_tier::{
    AttachmentRelation, BlockChunk, GraTier, HostRedirects, ItemTarget, MorTier,
    RootRelationUnderHost, SpanRoot, SplicedBlock, SplicedBlockError,
};

/// Why the host anchor of a span could not be read.
#[derive(Debug, thiserror::Error)]
enum AnchorError {
    /// The host tiers do not hold the attachment source's relation.
    #[error("the attachment source's host relation cannot be read: {0}")]
    Unreadable(#[from] CoordinatedMutationError),
    /// A root label cannot attach a span under another chunk.
    #[error(transparent)]
    RootRelation(#[from] RootRelationUnderHost),
    /// The secondary relations do not form an admitted tree.
    #[error(transparent)]
    Block(#[from] SplicedBlockError),
    /// The attachment source must belong to the span it describes.
    #[error("attachment source {source_word} is outside span {start}..{end}")]
    SourceOutsideSpan {
        source_word: usize,
        start: usize,
        end: usize,
    },
}

/// Outcome of splicing L2 results into a `ChatFile`.
#[derive(Debug, Default)]
pub struct SpliceOutcome {
    /// Number of @s positions successfully spliced with real morphology.
    pub spliced: usize,
    /// Number of words of the spliced spans that fell back to `L2|xxx`
    /// (rolled back, or no `%mor` to place them in). Words whose span was
    /// never merged are not counted here; the caller reports them.
    pub fallback: usize,
    /// Number of corrected external relations written into `%gra`. A
    /// correction is written only on a span root that ends attached to a
    /// host governor. Utterance-root replacements and generic primary-root
    /// fallback attachments are not external corrections and are not counted.
    pub gra_upgraded: usize,
}

/// Reason an L2 splice rolled back to `L2|xxx` for one or more
/// positions. Each variant doubles as a TODO bucket: a category is
/// a candidate for smarter merge logic that would *recover*
/// secondary morphology instead of falling back to `L2|xxx`.
/// `Display` emits the lower-case snake_case tag used in
/// `tracing::warn!` `category` fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpliceFallbackCategory {
    SecondaryMultiRoot,
    SecondaryNoRoot,
    SecondaryCycle,
    SecondaryHeadOob,
    SpliceInvariantOther,
    HostAnchorUnreadable,
}

impl SpliceFallbackCategory {
    /// Classify the splice's post-validation error into a fallback
    /// bucket. Adding a new variant to `MappingError` should add a
    /// new arm here.
    fn from_mapping_error(err: &crate::morphosyntax::MappingError) -> Self {
        use crate::morphosyntax::MappingError;
        match err {
            MappingError::InvalidRoot { details } if details.contains("multiple") => {
                Self::SecondaryMultiRoot
            }
            MappingError::InvalidRoot { .. } => Self::SecondaryNoRoot,
            MappingError::CircularDependency { .. } => Self::SecondaryCycle,
            MappingError::InvalidHeadReference { .. } => Self::SecondaryHeadOob,
            MappingError::EmptyStem { .. }
            | MappingError::InvalidDeprel { .. }
            | MappingError::Sentence(_)
            | MappingError::EmptyRangeComponents => Self::SpliceInvariantOther,
        }
    }
}

impl std::fmt::Display for SpliceFallbackCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::SecondaryMultiRoot => "secondary_multi_root",
            Self::SecondaryNoRoot => "secondary_no_root",
            Self::SecondaryCycle => "secondary_cycle",
            Self::SecondaryHeadOob => "secondary_head_oob",
            Self::SpliceInvariantOther => "splice_invariant_other",
            Self::HostAnchorUnreadable => "host_anchor_unreadable",
        };
        f.write_str(s)
    }
}

/// Render a slice of `GrammaticalRelation` as the `%gra` body
/// (`index|head|relation` per relation, space-separated). Used in
/// fallback `tracing::warn!` messages on the rollback path.
fn join_relations(relations: &[talkbank_model::model::GrammaticalRelation]) -> String {
    relations
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Outcome of the post-splice invariant gate; `splice_span` updates its
/// `outcome` counters off this result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpliceValidationResult {
    /// Splice output passed `validate_generated_gra`; commit.
    Valid,
    /// Splice output failed validation; tiers were restored from the
    /// snapshots and the affected words were reset to `L2|xxx`.
    RolledBack,
}

/// Position descriptor for a splice fallback's `tracing::warn!`
/// payload. A one-word span reports its host word; a longer span
/// reports its first host word and size, matching the shape of
/// `splice_range_coordinated`'s `item_range` argument.
#[derive(Debug, Clone, Copy)]
enum SplicePositionDescriptor {
    SingleWord { word_idx: MorItemIndex },
    Span { range_start: usize, size: usize },
}

/// Bundle of warn-fields that don't depend on the splice outcome.
/// Keeps the `validate_or_rollback_splice` signature compact.
struct SpliceFallbackContext<'a> {
    line_idx: usize,
    target_lang: &'a talkbank_model::model::LanguageCode,
    /// Word indices in the host utterance that get reset to `L2|xxx`
    /// when the splice rolls back. For single-position splices this
    /// is a one-element slice; for multi-position contiguous spans,
    /// every word in the span.
    word_indices_to_reset: &'a [MorItemIndex],
    position: SplicePositionDescriptor,
}

/// Validate the post-splice gra against the structural invariants
/// (`validate_generated_gra`); on failure, restore the pre-splice
/// snapshots, reset the affected `word_indices_to_reset` to
/// `L2|xxx`, and emit a categorized `tracing::warn!`.
///
/// The `secondary_gras_summary` closure is only invoked on the
/// rollback path so the success path pays no diagnostic-building
/// cost. Callers update their `outcome.spliced` / `outcome.fallback`
/// counters off the returned [`SpliceValidationResult`].
///
/// This is the single chokepoint for the "no invalid CHAT shipped
/// downstream" guarantee: any future change to the rollback shape
/// (categories, warn fields, fallback policy) lands here once
/// instead of being mirrored in both branches.
fn validate_or_rollback_splice(
    mor: &mut talkbank_model::model::dependent_tier::MorTier,
    gra: &mut talkbank_model::model::dependent_tier::GraTier,
    mor_snapshot: talkbank_model::model::dependent_tier::MorTier,
    gra_snapshot: talkbank_model::model::dependent_tier::GraTier,
    secondary_gras_summary: impl FnOnce() -> Vec<String>,
    ctx: SpliceFallbackContext<'_>,
) -> SpliceValidationResult {
    let Err(invariant_err) =
        crate::morphosyntax::gra_validate::validate_generated_gra(gra.relations())
    else {
        return SpliceValidationResult::Valid;
    };

    let category = SpliceFallbackCategory::from_mapping_error(&invariant_err);
    let pre_splice_gra = join_relations(gra_snapshot.relations());
    let post_splice_gra = join_relations(gra.relations());
    let secondary_gras = secondary_gras_summary();

    *mor = mor_snapshot;
    *gra = gra_snapshot;
    for &word_idx in ctx.word_indices_to_reset {
        if let Some(mor_item) = mor.items_mut().get_mut(word_idx.as_usize()) {
            mor_item.main.reset_to_l2_placeholder();
        }
    }

    match ctx.position {
        SplicePositionDescriptor::SingleWord { word_idx } => {
            tracing::warn!(
                line_idx = ctx.line_idx,
                word_idx = word_idx.as_usize(),
                target_lang = %ctx.target_lang,
                category = %category,
                invariant_error = %invariant_err,
                secondary_gras = ?secondary_gras,
                host_pre_splice_gra = %pre_splice_gra,
                host_post_splice_gra = %post_splice_gra,
                "L2 splice fell back to L2|xxx because secondary input \
                 would have produced invalid CHAT (post-splice gra fails \
                 structural invariants); see this warning's category for \
                 the smarter-merge TODO bucket"
            );
        }
        SplicePositionDescriptor::Span { range_start, size } => {
            tracing::warn!(
                line_idx = ctx.line_idx,
                span_word_start = range_start,
                span_size = size,
                target_lang = %ctx.target_lang,
                category = %category,
                invariant_error = %invariant_err,
                secondary_gras_per_position = ?secondary_gras,
                host_pre_splice_gra = %pre_splice_gra,
                host_post_splice_gra = %post_splice_gra,
                "L2 multi-position splice fell back to L2|xxx for the \
                 whole span because secondary input would have produced \
                 invalid CHAT (post-splice gra fails structural \
                 invariants); see this warning's category for the \
                 smarter-merge TODO bucket"
            );
        }
    }

    SpliceValidationResult::RolledBack
}

/// Read the span attachment in the host's pre-splice numbering. Chatter
/// owns translation and refuses an anchor inside or dependent on the span.
fn current_root_anchor_for_attachment(
    mor: &MorTier,
    gra: &GraTier,
    attachment: &L2Attachment<super::plan::ExternalRelation>,
) -> Result<SpanRoot, AnchorError> {
    match attachment {
        L2Attachment::InternalRoot | L2Attachment::UtteranceRoot { .. } => {
            Ok(SpanRoot::UtteranceRoot)
        }
        L2Attachment::HostGovernor {
            source_word,
            source,
            relation,
        } => {
            let head = mor.governing_head_for_item(gra, *source_word)?;
            match head {
                GraHeadRef::Root => Ok(SpanRoot::UtteranceRoot),
                GraHeadRef::Word(chunk) => Ok(SpanRoot::HostChunk {
                    chunk,
                    relation: AttachmentRelation::new(relation.relation(source).to_chat_gra())?,
                }),
            }
        }
    }
}

/// Host dependents of the primary attachment source follow the secondary
/// span root. Other words retain their own counterpart's head chunk.
fn host_redirects(
    attachment: &L2Attachment<super::plan::ExternalRelation>,
    range: &std::ops::Range<usize>,
    root: BlockChunk,
) -> Result<HostRedirects, AnchorError> {
    let Some(source) = attachment.source_word() else {
        return Ok(HostRedirects::ByItem);
    };
    let source_word = source.as_usize();
    if !range.contains(&source_word) {
        return Err(AnchorError::SourceOutsideSpan {
            source_word,
            start: range.start,
            end: range.end,
        });
    }
    Ok(HostRedirects::PerItem(
        range
            .clone()
            .map(|item| {
                if item == source_word {
                    ItemTarget::Chunk(root)
                } else {
                    ItemTarget::Counterpart
                }
            })
            .collect(),
    ))
}

/// The attachment actually admitted, distinct from the requested plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AppliedAttachment {
    PlannedHost,
    UtteranceRoot,
    PrimaryRootFallback,
}

/// Retain the primary root when the secondary requests the utterance root
/// but a host root survives outside the span. This is the explicit generic
/// DEP policy; the splice itself never invents that relation.
fn splice_with_primary_root_policy(
    mor: &mut MorTier,
    gra: &mut GraTier,
    range: std::ops::Range<usize>,
    block: SplicedBlock,
    root: SpanRoot,
    redirects: HostRedirects,
) -> Result<AppliedAttachment, AnchorError> {
    let planned = match &root {
        SpanRoot::HostChunk { .. } => AppliedAttachment::PlannedHost,
        SpanRoot::UtteranceRoot => AppliedAttachment::UtteranceRoot,
    };
    match mor.splice_range_coordinated(gra, range.clone(), block.clone(), root, redirects.clone()) {
        Ok(()) => Ok(planned),
        Err(CoordinatedMutationError::UtteranceRootTaken { host_root }) => {
            mor.splice_range_coordinated(
                gra,
                range,
                block,
                SpanRoot::HostChunk {
                    chunk: host_root,
                    relation: AttachmentRelation::new("DEP")?,
                },
                redirects,
            )?;
            Ok(AppliedAttachment::PrimaryRootFallback)
        }
        Err(
            CoordinatedMutationError::SpanRootDependsOnSpan { .. }
            | CoordinatedMutationError::SpanRootInReplacedRange { .. }
            | CoordinatedMutationError::SpanRootOutOfHost { .. },
        ) => {
            // Retry as the utterance root only if admission proves the
            // primary root is being replaced, not surviving outside it.
            mor.splice_range_coordinated(gra, range, block, SpanRoot::UtteranceRoot, redirects)?;
            Ok(AppliedAttachment::UtteranceRoot)
        }
        Err(error) => Err(AnchorError::Unreadable(error)),
    }
}

/// Overwrite `L2|xxx` MOR items with merged spans, consuming them.
///
/// Each [`MergedL2Span`] replaces the `%mor` items of its host words with
/// the secondary's items, and the matching `%gra` relations with the
/// secondary's span-relative relations, the span root reattached to the
/// host as its attachment says. Every span goes through chatter's
/// range splice ([`MorTier::splice_range_coordinated`]), which keeps the
/// span's cross-word heads (`la@s fecha@s bien@s`: `la` and `bien` both
/// under `fecha`) in one block.
///
/// Every span is validated after splicing and rolled back to `L2|xxx` if
/// the result breaks a `%gra` invariant.
///
/// Must be called AFTER `inject_results` has set `L2|xxx` on every `@s`
/// position.
///
/// [`MorTier::splice_range_coordinated`]: talkbank_model::model::MorTier::splice_range_coordinated
pub fn splice_l2_into_chat(
    chat_file: &mut talkbank_model::model::ChatFile,
    mut merged: Vec<MergedL2Span>,
) -> SpliceOutcome {
    // Transcript order, whatever order the caller merged in (it batches by
    // language): each splice reads the %gra the previous ones left, so the
    // order is part of the result.
    merged.sort_by_key(|span| (span.line_idx, span.first_word));
    let mut outcome = SpliceOutcome::default();
    for span in merged {
        splice_span(chat_file, span, &mut outcome);
    }
    outcome
}

/// Splice one merged span into its host utterance.
fn splice_span(
    chat_file: &mut talkbank_model::model::ChatFile,
    span: MergedL2Span,
    outcome: &mut SpliceOutcome,
) {
    use talkbank_model::model::DependentTier;
    use talkbank_model::model::Line;

    let words = span.mors.len();
    let Some(Line::Utterance(utt)) = chat_file.lines.as_mut_slice().get_mut(span.line_idx) else {
        outcome.fallback += words;
        return;
    };
    let mut mor_tier = None;
    let mut gra_tier = None;
    for tier in &mut utt.dependent_tiers {
        match &mut tier.tier {
            DependentTier::Mor(m) => mor_tier = Some(m),
            DependentTier::Gra(g) => gra_tier = Some(g),
            _ => {}
        }
    }
    let Some(mor) = mor_tier else {
        outcome.fallback += words;
        return;
    };
    let start = span.first_word.as_usize();
    let item_range = start..start + words;
    if item_range.end > mor.items().len() {
        outcome.fallback += words;
        return;
    }

    let Some(gra) = gra_tier else {
        // No %gra tier: only the %mor items change.
        for (slot, item) in mor.items_mut()[item_range].iter_mut().zip(span.mors) {
            *slot = item;
        }
        outcome.spliced += words;
        return;
    };

    let mut new_gras = span.gras;
    crate::morphosyntax::l2::merge::repair_secondary_gras(&mut new_gras, &span.attachment);
    let anchor = match current_root_anchor_for_attachment(mor, gra, &span.attachment) {
        Ok(anchor) => anchor,
        Err(error) => {
            tracing::warn!(
                line_idx = span.line_idx,
                span_word_start = start,
                span_size = words,
                target_lang = %span.target_lang,
                category = %SpliceFallbackCategory::HostAnchorUnreadable,
                %error,
                "L2 splice fell back to L2|xxx: the host anchor of the span \
                 root cannot be read"
            );
            outcome.fallback += words;
            return;
        }
    };

    // Whole-tier snapshot for rollback: chatter's splices re-index and
    // re-head relations across the WHOLE host %gra, not just the block.
    let mor_snapshot = mor.clone();
    let gra_snapshot = gra.clone();
    let secondary_gras = join_relations(&new_gras);

    let position = match words {
        1 => SplicePositionDescriptor::SingleWord {
            word_idx: span.first_word,
        },
        _ => SplicePositionDescriptor::Span {
            range_start: start,
            size: words,
        },
    };
    let prepared = (|| {
        let block = SplicedBlock::new(span.mors, new_gras)?;
        let redirects = host_redirects(&span.attachment, &item_range, block.root_chunk())?;
        splice_with_primary_root_policy(mor, gra, item_range.clone(), block, anchor, redirects)
    })();
    let applied_attachment = match prepared {
        Ok(applied_attachment) => applied_attachment,
        Err(error) => {
            tracing::warn!(
                line_idx = span.line_idx,
                span_word_start = start,
                target_lang = %span.target_lang,
                %error,
                "L2 splice refused; the primary placeholders are retained"
            );
            outcome.fallback += words;
            return;
        }
    };

    let word_indices: Vec<MorItemIndex> = item_range.map(MorItemIndex::new).collect();
    match validate_or_rollback_splice(
        mor,
        gra,
        mor_snapshot,
        gra_snapshot,
        || vec![secondary_gras],
        SpliceFallbackContext {
            line_idx: span.line_idx,
            target_lang: &span.target_lang,
            word_indices_to_reset: &word_indices,
            position,
        },
    ) {
        SpliceValidationResult::Valid => {
            outcome.spliced += words;
            // A correction counts where it was written.
            if applied_attachment == AppliedAttachment::PlannedHost
                && span.attachment.corrected_deprel().is_some()
            {
                outcome.gra_upgraded += 1;
            }
        }
        SpliceValidationResult::RolledBack => {
            outcome.fallback += words;
        }
    }
}

#[cfg(test)]
mod cardinality_tests {
    use super::*;
    use crate::morphosyntax::l2::deprel::UdDeprel;
    use crate::morphosyntax::l2::extract::{L2DeferredPosition, PrimaryStructuralInfo};
    use crate::morphosyntax::l2::merge::MergedL2Span;
    use crate::morphosyntax::l2::plan::ExternalRelation;
    use crate::parse::parse_lenient;
    use talkbank_model::ParseValidateOptions;
    use talkbank_model::WriteChat;
    use talkbank_model::model::dependent_tier::GrammaticalRelation;
    use talkbank_model::model::dependent_tier::mor::{Mor, MorStem, MorWord, PosCategory};
    use talkbank_parser::TreeSitterParser;

    /// Splice replaces a 1-chunk `L2|xxx` slot with an N-chunk merged Mor.
    /// The output `%mor` chunk count grows but `%gra` count stays the same;
    /// the resulting ChatFile must still validate.
    ///
    /// Fixture note (2026-05-06): the original fixture was
    /// `*PAR: yellow@s .`: a whole-utterance `@s` pattern that
    /// validator E255 (BUG-023, GREEN 2026-05-05) correctly rejects
    /// (whole-utterance language switches must use `[- LANG]` precode,
    /// not per-word `@s`). The test now embeds the L2 word among native
    /// French words so the fixture validates pre-splice while still
    /// exercising the multi-chunk MWT splice path.
    #[test]
    fn multi_chunk_merged_mor_keeps_chat_valid() {
        let chat_text = "@UTF8\n\
                         @Begin\n\
                         @Languages:\tfra, ara\n\
                         @Participants:\tPAR Participant\n\
                         @ID:\tfra|test|PAR|||||Participant|||\n\
                         *PAR:\tvoici yellow@s .\n\
                         %mor:\tintj|voici L2|xxx .\n\
                         %gra:\t1|2|DISCOURSE 2|0|ROOT 3|2|PUNCT\n\
                         @End\n";
        let parser = TreeSitterParser::new().unwrap();
        let (mut chat_file, _errors) = parse_lenient(&parser, chat_text);

        let mut precondition = chat_file.clone();
        let opts = ParseValidateOptions::default().with_alignment();
        assert!(
            talkbank_model::validate_chat_file_with_options(&mut precondition, &opts).is_ok(),
            "fixture precondition: input must validate before splice",
        );

        let merged_mor = Mor::new(MorWord::new(PosCategory::new("verb"), MorStem::new("yel")))
            .with_post_clitic(MorWord::new(PosCategory::new("part"), MorStem::new("lo")));

        let mut utt_idx = None;
        for (i, line) in chat_file.lines.iter().enumerate() {
            if let talkbank_model::model::Line::Utterance(_) = line {
                utt_idx = Some(i);
                break;
            }
        }
        let line_idx = utt_idx.expect("utterance present");

        // The L2 placeholder is at word_idx 1 (after "voici" at idx 0).
        // Host primary said the L2 word is utterance root (head=0,
        // deprel=root): chunk 2 in the host gra.
        let deferred = vec![L2DeferredPosition::for_test(
            line_idx,
            MorItemIndex::new(1),
            "ara",
            "mrhba",
            PrimaryStructuralInfo::for_test(
                "root",
                crate::morphosyntax::alignment::HeadTarget::Root,
                None,
            ),
        )];
        let merged = vec![Some(PositionResult {
            mor: merged_mor,
            gras: vec![
                GrammaticalRelation::new(1, 0, "ROOT"),
                GrammaticalRelation::new(2, 1, "DEP"),
            ],
            attachment: TestAttachment::Internal,
        })];

        let outcome = splice_positions(&mut chat_file, &deferred, merged);
        assert_eq!(
            outcome.spliced, 1,
            "splice must report success for the slot"
        );
        assert_eq!(outcome.fallback, 0);

        validate_morphosyntax(&mut chat_file);
    }

    /// Helper: build a one-utterance ChatFile with N L2 placeholders.
    /// Returns (ChatFile, line_idx). The %mor and %gra are pre-validated
    /// to ensure the precondition is met before splicing.
    fn build_l2_fixture(
        languages_header: &str,
        word_text: &str,
        n_l2_words: usize,
    ) -> (talkbank_model::model::ChatFile, usize) {
        // %mor: one `L2|xxx` per word, then terminator.
        let mor_line = (0..n_l2_words)
            .map(|_| "L2|xxx")
            .collect::<Vec<_>>()
            .join(" ")
            + " .";
        // %gra: one ROOT then n_l2_words-1 DEPs back to root, then PUNCT.
        // For n=1: `1|0|ROOT 2|1|PUNCT`
        // For n=3: `1|0|ROOT 2|1|DEP 3|1|DEP 4|1|PUNCT`
        let mut gra_parts: Vec<String> = Vec::with_capacity(n_l2_words + 1);
        gra_parts.push("1|0|ROOT".to_string());
        for i in 2..=n_l2_words {
            gra_parts.push(format!("{}|1|DEP", i));
        }
        gra_parts.push(format!("{}|1|PUNCT", n_l2_words + 1));
        let gra_line = gra_parts.join(" ");

        let chat_text = format!(
            "@UTF8\n\
             @Begin\n\
             @Languages:\t{lang}\n\
             @Participants:\tPAR Participant\n\
             @ID:\t{lang_first}|test|PAR|||||Participant|||\n\
             *PAR:\t{words} .\n\
             %mor:\t{mor}\n\
             %gra:\t{gra}\n\
             @End\n",
            lang = languages_header,
            lang_first = languages_header.split(',').next().unwrap().trim(),
            words = word_text,
            mor = mor_line,
            gra = gra_line,
        );
        let parser = TreeSitterParser::new().unwrap();
        let (chat_file, _errors) = parse_lenient(&parser, &chat_text);
        let line_idx = chat_file
            .lines
            .iter()
            .position(|l| matches!(l, talkbank_model::model::Line::Utterance(_)))
            .expect("fixture has an utterance");
        (chat_file, line_idx)
    }

    /// Helper: construct a deferred position with no head/no head_upos
    /// other than what the caller provides. Models the primary analysis
    /// of a single `@s` word.
    fn deferred_position(
        line_idx: usize,
        word_idx: usize,
        target_lang: &str,
        primary_deprel: &str,
        primary_head: usize,
    ) -> L2DeferredPosition {
        L2DeferredPosition::for_test(
            line_idx,
            MorItemIndex::new(word_idx),
            target_lang,
            "word",
            PrimaryStructuralInfo::for_test(
                primary_deprel,
                // The UD head id the primary gave, as the host word it
                // lies in (fixtures have no contractions, so id `n` is
                // word `n - 1`).
                match primary_head.checked_sub(1) {
                    None => crate::morphosyntax::alignment::HeadTarget::Root,
                    Some(word) => {
                        crate::morphosyntax::alignment::HeadTarget::Word(MorItemIndex::new(word))
                    }
                },
                None,
            ),
        )
    }

    /// **RED test 1** (l2.md §6 / postmortem §6 Step 1): three contiguous
    /// `@s` words in one utterance, the minimal multi-position L2 span.
    /// Mirrors the wild bad-case shape from
    /// `<corpus-root>/biling-data/Bangor/Patagonia/07.cha:394`
    /// (`la@s fecha@s bien@s`). Stanza secondary returns one sentence
    /// covering all three with cross-position heads. Per-position splicing
    /// of the resulting per-position gras is expected to fail because
    /// `splice_coordinated`'s "internal reference within new block" branch
    /// remaps heads with the WRONG position's `chunk_offset`.
    #[test]
    fn multi_position_mwt_in_one_utterance() {
        let (mut chat_file, line_idx) = build_l2_fixture("spa, eng", "la fecha bien", 3);

        // Three contiguous L2 positions, primary analysis from (host) Welsh
        // says all three head to position 1 (the conventional ROOT in the
        // fixture's %gra). Targeting Spanish secondary.
        let deferred = vec![
            deferred_position(line_idx, 0, "spa", "det", 0),
            deferred_position(line_idx, 1, "spa", "root", 0),
            deferred_position(line_idx, 2, "spa", "advmod", 0),
        ];

        // Per-position merged_results, the way `dispatch_secondary_l2`
        // currently slices `gra_relations` from the secondary Stanza call.
        // Heads use SECONDARY-SENTENCE 1-indexed values: la→head=2 means
        // "fecha" (the second word in the secondary sentence), bien→head=2
        // means "fecha" too.
        let merged = vec![
            Some(PositionResult {
                mor: Mor::new(MorWord::new(PosCategory::new("det"), MorStem::new("la"))),
                gras: vec![GrammaticalRelation::new(1, 2, "DET")],
                attachment: TestAttachment::Internal,
            }),
            Some(PositionResult {
                mor: Mor::new(MorWord::new(
                    PosCategory::new("noun"),
                    MorStem::new("fecha"),
                )),
                gras: vec![GrammaticalRelation::new(1, 0, "ROOT")],
                attachment: TestAttachment::Internal,
            }),
            Some(PositionResult {
                mor: Mor::new(MorWord::new(PosCategory::new("adv"), MorStem::new("bien"))),
                gras: vec![GrammaticalRelation::new(1, 2, "ADVMOD")],
                attachment: TestAttachment::Internal,
            }),
        ];

        let outcome = splice_positions(&mut chat_file, &deferred, merged);
        assert_eq!(outcome.spliced, 3, "all three positions must splice");
        assert_eq!(outcome.fallback, 0);
        validate_morphosyntax(&mut chat_file);
    }

    /// **RED test 2**, one `@s` word that Stanza expands to 3 chunks
    /// (verb + two clitics, e.g. `verb|x~part|y~part|z`). Same cardinality
    /// assertion as the existing 1→2 chunk test. Catches off-by-one
    /// breakage in `splice_coordinated`'s chunk-count delta arithmetic
    /// when delta = +2 instead of the +1 the existing test exercises.
    #[test]
    fn multi_clitic_mwt() {
        let (mut chat_file, line_idx) = build_l2_fixture("fra, ara", "verb", 1);

        let merged_mor = Mor::new(MorWord::new(PosCategory::new("verb"), MorStem::new("v")))
            .with_post_clitic(MorWord::new(PosCategory::new("part"), MorStem::new("c1")))
            .with_post_clitic(MorWord::new(PosCategory::new("part"), MorStem::new("c2")));

        let deferred = vec![deferred_position(line_idx, 0, "fra", "root", 0)];
        let merged = vec![Some(PositionResult {
            mor: merged_mor,
            gras: vec![
                GrammaticalRelation::new(1, 0, "ROOT"),
                GrammaticalRelation::new(2, 1, "DEP"),
                GrammaticalRelation::new(3, 1, "DEP"),
            ],
            attachment: TestAttachment::Internal,
        })];

        let outcome = splice_positions(&mut chat_file, &deferred, merged);
        assert_eq!(outcome.spliced, 1);
        assert_eq!(outcome.fallback, 0);
        validate_morphosyntax(&mut chat_file);
    }

    /// **RED test 3**: phrasal-verb particle: an `@s` word whose merge
    /// corrects the external relation to `compound:prt`. Verifies that
    /// (a) the splice still reports success and (b) the host's terminator
    /// gra is preserved (not silently overwritten by the corrected deprel
    /// path). The existing single-position test does not exercise the
    /// corrected-relation branch.
    #[test]
    fn phrasal_verb_particle_preserves_terminator_gra() {
        // Host: `wake up .` where `up@s` is the L2 particle. Primary says
        // "up" is advmod of "wake".
        let chat_text = "@UTF8\n\
                         @Begin\n\
                         @Languages:\teng, fra\n\
                         @Participants:\tPAR Participant\n\
                         @ID:\teng|test|PAR|||||Participant|||\n\
                         *PAR:\twake up .\n\
                         %mor:\tverb|wake L2|xxx .\n\
                         %gra:\t1|0|ROOT 2|1|DEP 3|1|PUNCT\n\
                         @End\n";
        let parser = TreeSitterParser::new().unwrap();
        let (mut chat_file, _errors) = parse_lenient(&parser, chat_text);
        let line_idx = chat_file
            .lines
            .iter()
            .position(|l| matches!(l, talkbank_model::model::Line::Utterance(_)))
            .unwrap();

        let deferred = vec![deferred_position(line_idx, 1, "fra", "advmod", 1)];
        let merged = vec![Some(PositionResult {
            mor: Mor::new(MorWord::new(PosCategory::new("part"), MorStem::new("up"))),
            // The secondary's own analysis of the one-word span: its root.
            // The splice writes the corrected relation on it.
            gras: vec![GrammaticalRelation::new(1, 0, "ROOT")],
            attachment: host_attachment(0, "compound:prt"),
        })];

        let outcome = splice_positions(&mut chat_file, &deferred, merged);
        assert_eq!(outcome.spliced, 1);
        assert_eq!(outcome.gra_upgraded, 1);
        validate_morphosyntax(&mut chat_file);

        // Explicit: terminator gra (originally `3|1|PUNCT`) must still be
        // a PUNCT relation pointing at the verb.
        let utt = match &chat_file.lines[line_idx] {
            talkbank_model::model::Line::Utterance(u) => u,
            _ => unreachable!(),
        };
        let gra_tier = utt
            .dependent_tiers
            .iter()
            .find_map(|t| match &t.tier {
                talkbank_model::model::DependentTier::Gra(g) => Some(g),
                _ => None,
            })
            .expect("gra tier must exist after splice");
        let last = gra_tier
            .relations()
            .last()
            .expect("terminator gra must remain");
        assert_eq!(
            last.relation.as_str().to_ascii_uppercase(),
            "PUNCT",
            "terminator gra deprel must still be PUNCT after splice"
        );
    }

    /// **RED test 4**: Italian range-override path: `del` is the MWT of
    /// `di` + `il` (prep + det). The fixture exercises the
    /// `try_handle_italian_range_override` codepath in `sentence_mapping`.
    /// Asserts cardinality is preserved post-splice.
    #[test]
    fn italian_range_override_preserves_cardinality() {
        let (mut chat_file, line_idx) = build_l2_fixture("ita, eng", "del", 1);

        let merged_mor = Mor::new(MorWord::new(PosCategory::new("prep"), MorStem::new("di")))
            .with_post_clitic(MorWord::new(PosCategory::new("det"), MorStem::new("il")));

        let deferred = vec![deferred_position(line_idx, 0, "ita", "root", 0)];
        let merged = vec![Some(PositionResult {
            mor: merged_mor,
            gras: vec![
                GrammaticalRelation::new(1, 0, "ROOT"),
                GrammaticalRelation::new(2, 1, "DET"),
            ],
            attachment: TestAttachment::Internal,
        })];

        let outcome = splice_positions(&mut chat_file, &deferred, merged);
        assert_eq!(outcome.spliced, 1);
        assert_eq!(outcome.fallback, 0);
        validate_morphosyntax(&mut chat_file);
    }

    /// **RED test 5**: end-to-end serialize / re-parse / re-validate.
    /// The internal `validate_chat_file_with_options` used by the other
    /// tests catches some but not all cardinality issues (per postmortem
    /// §4b: "the 235 wild files passed `validate_chat_file_with_options`
    /// at write time but fail `chatter validate` afterwards"). This test
    /// closes that gap by serializing the spliced ChatFile and re-validating
    /// the round-tripped form, which is what `chatter validate` itself does.
    #[test]
    fn chatter_validate_passes_after_splice() {
        use crate::serialize::to_chat_string;

        let (mut chat_file, line_idx) = build_l2_fixture("fra, ara", "yellow", 1);

        let merged_mor = Mor::new(MorWord::new(PosCategory::new("verb"), MorStem::new("yel")))
            .with_post_clitic(MorWord::new(PosCategory::new("part"), MorStem::new("lo")));
        let deferred = vec![deferred_position(line_idx, 0, "ara", "root", 0)];
        let merged = vec![Some(PositionResult {
            mor: merged_mor,
            gras: vec![
                GrammaticalRelation::new(1, 0, "ROOT"),
                GrammaticalRelation::new(2, 1, "DEP"),
            ],
            attachment: TestAttachment::Internal,
        })];

        splice_positions(&mut chat_file, &deferred, merged);

        // Round-trip: serialize → re-parse → validate the re-parsed file.
        let parser = TreeSitterParser::new().unwrap();
        let serialized = to_chat_string(&chat_file);
        let (mut reparsed, _errors) = parse_lenient(&parser, &serialized);
        validate_morphosyntax(&mut reparsed);
    }

    #[test]
    fn single_position_root_remap_does_not_leave_root_deprel_on_non_root_head() {
        let chat_text = "@UTF8\n\
                         @Begin\n\
                         @Languages:\teng, spa\n\
                         @Participants:\tPAR Participant\n\
                         @ID:\teng|test|PAR|||||Participant|||\n\
                         *PAR:\thost foreign tail .\n\
                         %mor:\tverb|host L2|xxx noun|tail .\n\
                         %gra:\t1|0|ROOT 2|1|DEP 3|1|OBJ 4|1|PUNCT\n\
                         @End\n";
        let parser = TreeSitterParser::new().unwrap();
        let (mut chat_file, _errors) = parse_lenient(&parser, chat_text);
        let line_idx = chat_file
            .lines
            .iter()
            .position(|l| matches!(l, talkbank_model::model::Line::Utterance(_)))
            .unwrap();

        let deferred = vec![deferred_position(line_idx, 1, "spa", "obj", 1)];
        let merged = vec![Some(PositionResult {
            mor: Mor::new(MorWord::new(
                PosCategory::new("noun"),
                MorStem::new("extranjero"),
            )),
            gras: vec![GrammaticalRelation::new(1, 0, "ROOT")],
            attachment: host_attachment(0, "obj"),
        })];

        let outcome = splice_positions(&mut chat_file, &deferred, merged);
        assert_eq!(outcome.spliced, 1);
        assert_eq!(outcome.fallback, 0);
        validate_morphosyntax(&mut chat_file);

        let utt = match &chat_file.lines[line_idx] {
            talkbank_model::model::Line::Utterance(u) => u,
            _ => unreachable!(),
        };
        let gra_tier = utt
            .dependent_tiers
            .iter()
            .find_map(|t| match &t.tier {
                talkbank_model::model::DependentTier::Gra(g) => Some(g),
                _ => None,
            })
            .expect("gra tier");
        let foreign_rel = &gra_tier.relations()[1];
        assert_eq!(foreign_rel.head, 1);
        assert_ne!(
            foreign_rel.relation.as_str().to_ascii_uppercase(),
            "ROOT",
            "a remapped external dependency must not keep deprel ROOT"
        );
    }

    #[test]
    fn single_position_root_remap_does_not_create_two_cycle() {
        let chat_text = "@UTF8\n\
                         @Begin\n\
                         @Languages:\teng, spa\n\
                         @Participants:\tPAR Participant\n\
                         @ID:\teng|test|PAR|||||Participant|||\n\
                         *PAR:\thost foreign .\n\
                         %mor:\tnoun|host L2|xxx .\n\
                         %gra:\t1|2|DEP 2|0|ROOT 3|2|PUNCT\n\
                         @End\n";
        let parser = TreeSitterParser::new().unwrap();
        let (mut chat_file, _errors) = parse_lenient(&parser, chat_text);
        let line_idx = chat_file
            .lines
            .iter()
            .position(|l| matches!(l, talkbank_model::model::Line::Utterance(_)))
            .unwrap();

        let deferred = vec![deferred_position(line_idx, 1, "spa", "case", 1)];
        let merged = vec![Some(PositionResult {
            mor: Mor::new(MorWord::new(
                PosCategory::new("noun"),
                MorStem::new("extranjero"),
            )),
            gras: vec![GrammaticalRelation::new(1, 0, "ROOT")],
            attachment: host_attachment(0, "case"),
        })];

        let outcome = splice_positions(&mut chat_file, &deferred, merged);
        assert_eq!(outcome.spliced, 1);
        assert_eq!(outcome.fallback, 0);
        validate_morphosyntax(&mut chat_file);

        let utt = match &chat_file.lines[line_idx] {
            talkbank_model::model::Line::Utterance(u) => u,
            _ => unreachable!(),
        };
        let gra_tier = utt
            .dependent_tiers
            .iter()
            .find_map(|t| match &t.tier {
                talkbank_model::model::DependentTier::Gra(g) => Some(g),
                _ => None,
            })
            .expect("gra tier");
        let host_rel = &gra_tier.relations()[0];
        let foreign_rel = &gra_tier.relations()[1];
        assert!(
            !(host_rel.head == foreign_rel.index && foreign_rel.head == host_rel.index),
            "root remap must not create a direct host<->foreign cycle"
        );
    }

    #[test]
    fn single_position_root_remap_does_not_self_anchor_non_root_relation() {
        let chat_text = "@UTF8\n\
                         @Begin\n\
                         @Languages:\teng, spa\n\
                         @Participants:\tPAR Participant\n\
                         @ID:\teng|test|PAR|||||Participant|||\n\
                         *PAR:\thost foreign .\n\
                         %mor:\tverb|host L2|xxx .\n\
                         %gra:\t1|0|ROOT 2|1|DEP 3|1|PUNCT\n\
                         @End\n";
        let parser = TreeSitterParser::new().unwrap();
        let (mut chat_file, _errors) = parse_lenient(&parser, chat_text);
        let line_idx = chat_file
            .lines
            .iter()
            .position(|l| matches!(l, talkbank_model::model::Line::Utterance(_)))
            .unwrap();

        let deferred = vec![deferred_position(line_idx, 1, "spa", "case", 2)];
        let merged = vec![Some(PositionResult {
            mor: Mor::new(MorWord::new(
                PosCategory::new("noun"),
                MorStem::new("extranjero"),
            )),
            gras: vec![GrammaticalRelation::new(1, 0, "ROOT")],
            attachment: host_attachment(0, "case"),
        })];

        let outcome = splice_positions(&mut chat_file, &deferred, merged);
        assert_eq!(outcome.spliced, 1);
        assert_eq!(outcome.fallback, 0);

        let utt = match &chat_file.lines[line_idx] {
            talkbank_model::model::Line::Utterance(u) => u,
            _ => unreachable!(),
        };
        let gra_tier = utt
            .dependent_tiers
            .iter()
            .find_map(|t| match &t.tier {
                talkbank_model::model::DependentTier::Gra(g) => Some(g),
                _ => None,
            })
            .expect("gra tier");
        let foreign_rel = &gra_tier.relations()[1];
        assert_ne!(
            foreign_rel.head, foreign_rel.index,
            "root remap must not create a self-headed non-ROOT relation"
        );
    }

    #[test]
    fn single_position_root_remap_does_not_use_out_of_bounds_anchor() {
        let chat_text = "@UTF8\n\
                         @Begin\n\
                         @Languages:\teng, spa\n\
                         @Participants:\tPAR Participant\n\
                         @ID:\teng|test|PAR|||||Participant|||\n\
                         *PAR:\thost foreign .\n\
                         %mor:\tverb|host L2|xxx .\n\
                         %gra:\t1|0|ROOT 2|1|DEP 3|1|PUNCT\n\
                         @End\n";
        let parser = TreeSitterParser::new().unwrap();
        let (mut chat_file, _errors) = parse_lenient(&parser, chat_text);
        let line_idx = chat_file
            .lines
            .iter()
            .position(|l| matches!(l, talkbank_model::model::Line::Utterance(_)))
            .unwrap();

        let deferred = vec![deferred_position(line_idx, 1, "spa", "compound", 5)];
        let merged = vec![Some(PositionResult {
            mor: Mor::new(MorWord::new(
                PosCategory::new("noun"),
                MorStem::new("extranjero"),
            )),
            gras: vec![GrammaticalRelation::new(1, 0, "ROOT")],
            attachment: host_attachment(0, "compound"),
        })];

        let outcome = splice_positions(&mut chat_file, &deferred, merged);
        assert_eq!(outcome.spliced, 1);
        assert_eq!(outcome.fallback, 0);
        validate_morphosyntax(&mut chat_file);

        let utt = match &chat_file.lines[line_idx] {
            talkbank_model::model::Line::Utterance(u) => u,
            _ => unreachable!(),
        };
        let gra_tier = utt
            .dependent_tiers
            .iter()
            .find_map(|t| match &t.tier {
                talkbank_model::model::DependentTier::Gra(g) => Some(g),
                _ => None,
            })
            .expect("gra tier");
        let foreign_rel = &gra_tier.relations()[1];
        assert!(
            foreign_rel.head <= gra_tier.relations().len(),
            "root remap must not emit a head outside the final %gra length"
        );
    }

    #[test]
    fn single_position_root_remap_uses_host_chunk_anchor_not_word_index() {
        let chat_text = "@UTF8\n\
                         @Begin\n\
                         @Languages:\teng, spa\n\
                         @Participants:\tPAR Participant\n\
                         @ID:\teng|test|PAR|||||Participant|||\n\
                         *PAR:\tit@s:eng foreign .\n\
                         %mor:\tpron|it~aux|be L2|xxx .\n\
                         %gra:\t1|2|EXPL 2|0|ROOT 3|2|DEP 4|2|PUNCT\n\
                         @End\n";
        let parser = TreeSitterParser::new().unwrap();
        let (mut chat_file, _errors) = parse_lenient(&parser, chat_text);
        let line_idx = chat_file
            .lines
            .iter()
            .position(|l| matches!(l, talkbank_model::model::Line::Utterance(_)))
            .unwrap();

        let deferred = vec![deferred_position(line_idx, 1, "spa", "obj", 1)];
        let merged = vec![Some(PositionResult {
            mor: Mor::new(MorWord::new(
                PosCategory::new("noun"),
                MorStem::new("extranjero"),
            )),
            gras: vec![GrammaticalRelation::new(1, 0, "ROOT")],
            attachment: host_attachment(0, "obj"),
        })];

        let outcome = splice_positions(&mut chat_file, &deferred, merged);
        assert_eq!(outcome.spliced, 1);
        assert_eq!(outcome.fallback, 0);
        validate_morphosyntax(&mut chat_file);

        let utt = match &chat_file.lines[line_idx] {
            talkbank_model::model::Line::Utterance(u) => u,
            _ => unreachable!(),
        };
        let gra_tier = utt
            .dependent_tiers
            .iter()
            .find_map(|t| match &t.tier {
                talkbank_model::model::DependentTier::Gra(g) => Some(g),
                _ => None,
            })
            .expect("gra tier");
        let foreign_rel = &gra_tier.relations()[2];
        assert_eq!(
            foreign_rel.head, 2,
            "L2 root remap must target the host's governing chunk, not the raw word index"
        );
    }

    #[test]
    fn multiple_noncontiguous_single_word_l2_spans_preserve_single_root() {
        let chat_text = "@UTF8\n\
                         @Begin\n\
                         @Languages:\teng, spa\n\
                         @Participants:\tPAR Participant\n\
                         @ID:\teng|test|PAR|||||Participant|||\n\
                         *PAR:\thost uno bridge dos tail .\n\
                         %mor:\tverb|host L2|xxx noun|bridge L2|xxx noun|tail .\n\
                         %gra:\t1|0|ROOT 2|1|DEP 3|1|OBJ 4|1|DEP 5|1|OBL 6|1|PUNCT\n\
                         @End\n";
        let parser = TreeSitterParser::new().unwrap();
        let (mut chat_file, _errors) = parse_lenient(&parser, chat_text);
        let line_idx = chat_file
            .lines
            .iter()
            .position(|l| matches!(l, talkbank_model::model::Line::Utterance(_)))
            .unwrap();

        let deferred = vec![
            deferred_position(line_idx, 1, "spa", "obj", 1),
            deferred_position(line_idx, 3, "spa", "obl", 1),
        ];
        let merged = vec![
            Some(PositionResult {
                mor: Mor::new(MorWord::new(PosCategory::new("noun"), MorStem::new("uno"))),
                gras: vec![GrammaticalRelation::new(1, 0, "ROOT")],
                attachment: host_attachment(0, "obj"),
            }),
            Some(PositionResult {
                mor: Mor::new(MorWord::new(PosCategory::new("noun"), MorStem::new("dos"))),
                gras: vec![GrammaticalRelation::new(1, 0, "ROOT")],
                attachment: host_attachment(1, "obl"),
            }),
        ];

        let outcome = splice_positions(&mut chat_file, &deferred, merged);
        assert_eq!(outcome.spliced, 2);
        assert_eq!(outcome.fallback, 0);
        validate_morphosyntax(&mut chat_file);

        let utt = match &chat_file.lines[line_idx] {
            talkbank_model::model::Line::Utterance(u) => u,
            _ => unreachable!(),
        };
        let gra_tier = utt
            .dependent_tiers
            .iter()
            .find_map(|t| match &t.tier {
                talkbank_model::model::DependentTier::Gra(g) => Some(g),
                _ => None,
            })
            .expect("gra tier");
        let root_count = gra_tier
            .relations()
            .iter()
            .filter(|rel| rel.head == 0)
            .count();
        assert_eq!(root_count, 1, "exactly one ROOT head must remain");
        let non_root_root_labels = gra_tier
            .relations()
            .iter()
            .filter(|rel| rel.head != 0 && rel.relation.as_str().eq_ignore_ascii_case("ROOT"))
            .count();
        assert_eq!(
            non_root_root_labels, 0,
            "non-contiguous L2 spans must not leave stray ROOT labels"
        );
    }

    #[test]
    fn multiword_span_with_later_root_source_preserves_that_root_anchor() {
        let chat_text = "@UTF8\n\
                         @Begin\n\
                         @Languages:\teng, spa\n\
                         @Participants:\tPAR Participant\n\
                         @ID:\teng|test|PAR|||||Participant|||\n\
                         *PAR:\thost uno dos .\n\
                         %mor:\tverb|host L2|xxx L2|xxx .\n\
                         %gra:\t1|3|DEP 2|3|DEP 3|0|ROOT 4|3|PUNCT\n\
                         @End\n";
        let parser = TreeSitterParser::new().unwrap();
        let (mut chat_file, _errors) = parse_lenient(&parser, chat_text);
        let line_idx = chat_file
            .lines
            .iter()
            .position(|l| matches!(l, talkbank_model::model::Line::Utterance(_)))
            .unwrap();

        let deferred = vec![
            deferred_position(line_idx, 1, "spa", "dep", 3),
            deferred_position(line_idx, 2, "spa", "root", 0),
        ];
        let merged = vec![
            Some(PositionResult {
                mor: Mor::new(MorWord::new(PosCategory::new("noun"), MorStem::new("uno"))),
                gras: vec![GrammaticalRelation::new(1, 2, "DEP")],
                attachment: utterance_root_attachment(1),
            }),
            Some(PositionResult {
                mor: Mor::new(MorWord::new(PosCategory::new("noun"), MorStem::new("dos"))),
                gras: vec![GrammaticalRelation::new(1, 0, "ROOT")],
                attachment: utterance_root_attachment(1),
            }),
        ];

        let outcome = splice_positions(&mut chat_file, &deferred, merged);
        assert_eq!(outcome.spliced, 2);
        assert_eq!(outcome.fallback, 0);
        validate_morphosyntax(&mut chat_file);

        let utt = match &chat_file.lines[line_idx] {
            talkbank_model::model::Line::Utterance(u) => u,
            _ => unreachable!(),
        };
        let gra_tier = utt
            .dependent_tiers
            .iter()
            .find_map(|t| match &t.tier {
                talkbank_model::model::DependentTier::Gra(g) => Some(g),
                _ => None,
            })
            .expect("gra tier");
        assert_eq!(
            gra_tier.relations()[2].head,
            0,
            "when the later item in a contiguous L2 span is the root-bearing \
             source, the spliced root must stay anchored to utterance ROOT \
             instead of inheriting the first replaced item's old head"
        );
    }

    fn validate_morphosyntax(chat: &mut talkbank_model::model::ChatFile) {
        use talkbank_model::ParseValidateOptions;
        let opts = ParseValidateOptions::default().with_alignment();
        if let Err(e) = talkbank_model::validate_chat_file_with_options(chat, &opts) {
            panic!("Morphosyntax validation failed: {:#?}", e);
        }
    }

    // ========================================================================
    // Family B: L2 splice integrity (joint-invariant RED tests).
    //
    // Pure-unit pinning for the Family B partition; see the L2
    // architectural-reassessment notes (§5).
    //
    // Wild evidence (from a 2026-05-06 wild-corpus error classification
    // log):
    //
    // - sastre03.cha:843 `+" yo@s soy@s el@s lieutenant .` produces
    //   `%gra: 1|3|NSUBJ 2|3|COP 3|0|DET 4|1|FLAT 5|1|PUNCT`, chunk 3
    //   has head=0 with deprel="DET" instead of "ROOT" (E722).
    // - herring09.cha:2570 `... el@s camino@s .` produces
    //   `%gra: 1|6|CC 2|6|NSUBJ 3|6|AUX 4|6|COP 5|6|CASE 6|7|DET 7|0|NMOD 8|6|PUNCT`
    //: chunk 7 has head=0 with deprel="NMOD" instead of "ROOT" (E722).
    // - sastre03.cha:2823 `al@s lado@s de@s Smith .` produces
    //   `%gra: 1|3|CASE 2|3|DET 3|0|NMOD 4|1|FIXED 5|1|FLAT 6|1|PUNCT`
    //: chunk 3 has head=0 with deprel="NMOD" instead of "ROOT" (E722).
    //
    // The unifying invariant the splice must enforce after writing its
    // output to the host `%gra`:
    //
    //     For every relation r:
    //         (r.head == 0) ⟺ (r.relation.eq_ignore_ascii_case("ROOT"))
    //
    // BUG-025's `single_position_root_remap_does_not_leave_root_deprel_on_non_root_head`
    // covered the right-to-left direction (deprel="ROOT" but head ≠ 0).
    // The wild patterns above are the left-to-right direction
    // (head = 0 but deprel ≠ "ROOT"), symmetric and equally broken.
    // ========================================================================

    /// Walk a GraTier's relations and assert the joint root invariant
    /// holds. Reports the offending relation(s) with the full %gra body
    /// for debuggability. Use from any Family B test.
    fn assert_joint_root_invariant(
        chat: &talkbank_model::model::ChatFile,
        line_idx: usize,
        scenario_label: &str,
    ) {
        let utt = match &chat.lines[line_idx] {
            talkbank_model::model::Line::Utterance(u) => u,
            _ => panic!("expected Line::Utterance at idx {line_idx}"),
        };
        let gra_tier = utt
            .dependent_tiers
            .iter()
            .find_map(|t| match &t.tier {
                talkbank_model::model::DependentTier::Gra(g) => Some(g),
                _ => None,
            })
            .expect("utterance must have %gra after splice");
        let body: String = join_relations(gra_tier.relations());

        let mut head_zero_count = 0usize;
        for rel in gra_tier.relations() {
            let head_zero = rel.head == 0;
            let labelled_root = rel.relation.as_str().eq_ignore_ascii_case("ROOT");
            assert!(
                head_zero == labelled_root,
                "Family B joint invariant violated [{scenario_label}]: \
                 chunk {} has head={} relation={:?}, head=0 must pair \
                 EXACTLY with deprel=ROOT (no head=0/non-ROOT, no \
                 ROOT-deprel/head!=0); got %gra: {body}",
                rel.index,
                rel.head,
                rel.relation.as_str()
            );
            if head_zero {
                head_zero_count += 1;
            }
        }
        assert_eq!(
            head_zero_count, 1,
            "Family B invariant [{scenario_label}]: utterance must have \
             exactly one head=0 relation; got %gra: {body}"
        );
    }

    /// **Family B, B-WILD-1**: three-`@s` cluster as the host's
    /// utterance root, mirroring the sastre03.cha:843 shape
    /// (`+" yo@s soy@s el@s lieutenant .`).
    ///
    /// Setup: host primary `%gra: 1|0|ROOT 2|0|ROOT 3|0|ROOT 4|3|FLAT
    /// 5|3|PUNCT` would be invalid (multiple ROOTs); the realistic shape
    /// is one of the L2 positions carrying head=0/ROOT in the primary
    /// (here: chunk 3 = "el") and the others as in-cluster dependents.
    /// Three deferred L2 positions; one secondary response of three
    /// chunks where the secondary's own root is the second chunk
    /// ("soy": Spanish copula); the splice must promote the
    /// secondary's root to the host's root anchor and emit
    /// `head=0/deprel=ROOT` (NOT `head=0/deprel=DET` or any other
    /// label inherited from the host's primary deprel for the L2
    /// position).
    ///
    /// EXPECTED on current build: FAILS via the joint-invariant walker.
    #[test]
    fn family_b_three_at_s_cluster_at_host_root_keeps_root_deprel() {
        let chat_text = "@UTF8\n\
                         @Begin\n\
                         @Languages:\teng, spa\n\
                         @Participants:\tPAR Participant\n\
                         @ID:\teng|test|PAR|||||Participant|||\n\
                         *PAR:\tyo soy el lieutenant .\n\
                         %mor:\tL2|xxx L2|xxx L2|xxx x|lieutenant .\n\
                         %gra:\t1|3|DEP 2|3|DEP 3|0|ROOT 4|3|FLAT 5|3|PUNCT\n\
                         @End\n";
        let parser = TreeSitterParser::new().unwrap();
        let (mut chat_file, _errors) = parse_lenient(&parser, chat_text);
        let line_idx = chat_file
            .lines
            .iter()
            .position(|l| matches!(l, talkbank_model::model::Line::Utterance(_)))
            .unwrap();

        // Three deferred positions for "yo", "soy", "el".
        // Deferred index 2 ("el") is the host primary's root, that's
        // the position that carries the utterance-root anchor.
        let deferred = vec![
            deferred_position(line_idx, 0, "spa", "dep", 3),
            deferred_position(line_idx, 1, "spa", "dep", 3),
            deferred_position(line_idx, 2, "spa", "root", 0),
        ];

        // Secondary Stanza-Spanish parse of "yo soy el":
        //   yo  → head=2 (subject of soy), deprel=nsubj
        //   soy → head=0 (root), deprel=root
        //   el  → head=2 (det, but secondary's "el" has nothing to
        //         determine since "lieutenant" is outside the secondary
        //         input: Stanza in practice may attach el→soy with
        //         deprel=det or similar)
        // Each position's result carries one chunk.
        let merged = vec![
            Some(PositionResult {
                mor: Mor::new(MorWord::new(PosCategory::new("pron"), MorStem::new("yo"))),
                gras: vec![GrammaticalRelation::new(1, 2, "NSUBJ")],
                attachment: utterance_root_attachment(2),
            }),
            Some(PositionResult {
                mor: Mor::new(MorWord::new(PosCategory::new("aux"), MorStem::new("ser"))),
                gras: vec![GrammaticalRelation::new(1, 0, "ROOT")],
                attachment: utterance_root_attachment(2),
            }),
            Some(PositionResult {
                mor: Mor::new(MorWord::new(PosCategory::new("det"), MorStem::new("el"))),
                gras: vec![GrammaticalRelation::new(1, 2, "DET")],
                attachment: utterance_root_attachment(2),
            }),
        ];

        let outcome = splice_positions(&mut chat_file, &deferred, merged);
        assert_eq!(outcome.spliced, 3);
        assert_eq!(outcome.fallback, 0);

        assert_joint_root_invariant(
            &chat_file,
            line_idx,
            "three-@s-cluster-at-host-root (sastre03 shape)",
        );
    }

    /// **Family B, B-WILD-2**: two-`@s` Spanish noun phrase whose
    /// internal root is the second chunk, mirroring herring09.cha:2570
    /// (`... el@s camino@s .`). Host primary attaches the cluster to
    /// chunk K (a host word) with deprel=NMOD. Secondary parses the
    /// 2-chunk cluster with the second chunk (`camino`) as its root.
    ///
    /// EXPECTED on current build: FAILS, the wild output had
    /// `7|0|NMOD` (head=0, deprel=NMOD) instead of one consistent
    /// pairing.
    #[test]
    fn family_b_two_at_s_np_anchored_externally_keeps_root_deprel_consistent() {
        let chat_text = "@UTF8\n\
                         @Begin\n\
                         @Languages:\teng, spa\n\
                         @Participants:\tPAR Participant\n\
                         @ID:\teng|test|PAR|||||Participant|||\n\
                         *PAR:\thost foreign1 foreign2 .\n\
                         %mor:\tverb|host L2|xxx L2|xxx .\n\
                         %gra:\t1|0|ROOT 2|1|OBL 3|2|FLAT 4|1|PUNCT\n\
                         @End\n";
        let parser = TreeSitterParser::new().unwrap();
        let (mut chat_file, _errors) = parse_lenient(&parser, chat_text);
        let line_idx = chat_file
            .lines
            .iter()
            .position(|l| matches!(l, talkbank_model::model::Line::Utterance(_)))
            .unwrap();

        // Host primary said "foreign1 foreign2" attaches to host chunk
        // 1 with deprel=OBL. Both deferred indices share the same
        // host attachment.
        let deferred = vec![
            deferred_position(line_idx, 1, "spa", "obl", 1),
            deferred_position(line_idx, 2, "spa", "flat", 2),
        ];
        let merged = vec![
            Some(PositionResult {
                mor: Mor::new(MorWord::new(PosCategory::new("det"), MorStem::new("el"))),
                gras: vec![GrammaticalRelation::new(1, 2, "DET")],
                attachment: host_attachment(0, "obl"),
            }),
            Some(PositionResult {
                mor: Mor::new(MorWord::new(
                    PosCategory::new("noun"),
                    MorStem::new("camino"),
                )),
                gras: vec![GrammaticalRelation::new(1, 0, "ROOT")],
                attachment: host_attachment(0, "obl"),
            }),
        ];

        let outcome = splice_positions(&mut chat_file, &deferred, merged);
        assert_eq!(outcome.spliced, 2);
        assert_eq!(outcome.fallback, 0);

        assert_joint_root_invariant(
            &chat_file,
            line_idx,
            "two-@s-NP-anchored-externally (herring09 shape)",
        );
    }

    /// **Family B, B-WILD-3**: joint-invariant guard for the
    /// already-fixed BUG-025 direction. Re-uses the existing
    /// `single_position_root_remap_does_not_leave_root_deprel_on_non_root_head`
    /// scenario but applies the symmetric joint-invariant walker, so
    /// any future regression in EITHER direction trips this test.
    ///
    /// EXPECTED on current build: PASSES (BUG-025 was fixed). Locks the
    /// fix in via the unifying invariant rather than a one-off
    /// "deprel != ROOT" assertion.
    #[test]
    fn family_b_joint_invariant_holds_after_bug025_remap() {
        let chat_text = "@UTF8\n\
                         @Begin\n\
                         @Languages:\teng, spa\n\
                         @Participants:\tPAR Participant\n\
                         @ID:\teng|test|PAR|||||Participant|||\n\
                         *PAR:\thost foreign tail .\n\
                         %mor:\tverb|host L2|xxx noun|tail .\n\
                         %gra:\t1|0|ROOT 2|1|DEP 3|1|OBJ 4|1|PUNCT\n\
                         @End\n";
        let parser = TreeSitterParser::new().unwrap();
        let (mut chat_file, _errors) = parse_lenient(&parser, chat_text);
        let line_idx = chat_file
            .lines
            .iter()
            .position(|l| matches!(l, talkbank_model::model::Line::Utterance(_)))
            .unwrap();

        let deferred = vec![deferred_position(line_idx, 1, "spa", "obj", 1)];
        let merged = vec![Some(PositionResult {
            mor: Mor::new(MorWord::new(
                PosCategory::new("noun"),
                MorStem::new("extranjero"),
            )),
            gras: vec![GrammaticalRelation::new(1, 0, "ROOT")],
            attachment: host_attachment(0, "obj"),
        })];

        let outcome = splice_positions(&mut chat_file, &deferred, merged);
        assert_eq!(outcome.spliced, 1);
        assert_eq!(outcome.fallback, 0);

        assert_joint_root_invariant(
            &chat_file,
            line_idx,
            "BUG-025 single-position remap (regression guard)",
        );
    }

    // ========================================================================
    // Family C: Post-splice gra invariant under adversarial secondary
    // input. RED tests pinning the load-bearing rule:
    //
    //     splice_l2_into_chat MUST never write a `%gra` that violates
    //     the structural invariants checked by `validate_generated_gra`
    //     (`crates/talkbank-transform/src/morphosyntax/gra_validate.rs`):
    //     - exactly one head=0 ROOT relation
    //     - acyclic head graph
    //     - all heads in 0..=N
    //
    // When secondary Stanza dispatch produces noisy output (cycles,
    // out-of-bounds heads, multiple roots, terminator-punct as root),
    // OR when our merge slicing math is wrong, the splice must either
    // correctly normalize or fall back to `L2|xxx` for the affected
    // position. Silent passthrough produced 756 wild errors on
    // 2026-05-06 (E724=539, E713=109, E723=108) in one production
    // run's validation log.
    //
    // Each fallback in production must emit a structured warning so
    // every fallback becomes an actionable TODO toward smarter merge
    // logic. (This is enforced separately at the splice's fallback
    // sites; tests here only assert post-splice gra is valid.)
    //
    // EXPECTED on current build: every test in this section FAILS
    // because the splice does not validate its own output.
    // ========================================================================

    /// Helper: assert post-splice ChatFile passes ALL the L2 splice
    /// integrity invariants. Reports the offending utterance and the
    /// validation diagnostics for debuggability. Use from any Family C
    /// test.
    fn assert_post_splice_gra_valid(chat: &mut talkbank_model::model::ChatFile, scenario: &str) {
        use talkbank_model::ParseValidateOptions;
        let opts = ParseValidateOptions::default().with_alignment();
        let result = talkbank_model::validate_chat_file_with_options(chat, &opts);
        if let Err(e) = result {
            let mut gra_summary = String::new();
            for line in &chat.lines {
                if let talkbank_model::model::Line::Utterance(u) = line {
                    for tier in &u.dependent_tiers {
                        if let talkbank_model::model::DependentTier::Gra(g) = &tier.tier {
                            gra_summary.push_str(&join_relations(g.relations()));
                            gra_summary.push_str(" | ");
                        }
                    }
                }
            }
            panic!(
                "Family C invariant violation [{scenario}]: post-splice \
                 ChatFile fails validation. Splice must reject and fall \
                 back to L2|xxx (with a structured warning) when secondary \
                 input would produce invalid gra.\n  post-splice gra: \
                 {gra_summary}\n  validation: {e:#?}"
            );
        }
    }

    /// **Family C, C1**: secondary's gra has head pointing at a
    /// nonexistent chunk (out-of-bounds). Mirrors the wild E713
    /// pattern at `asd-data/Croatian/ROGPOP/ASD/46.cha:970` where the
    /// post-splice %gra was `1|2|ADVMOD 2|0|ROOT 3|5|COMPOUND
    /// 4|2|PUNCT`: head=5 in a 4-chunk gra.
    ///
    /// Adversarial input: secondary merged result with one chunk but
    /// a gra relation pointing at chunk index 5 (which doesn't exist
    /// within the secondary's chunk count). The splice must not
    /// propagate the broken head into the host gra.
    ///
    /// EXPECTED on current build: FAILS, splice silently propagates
    /// the bogus head index.
    #[test]
    fn family_c_secondary_head_out_of_bounds_falls_back_or_normalizes() {
        let chat_text = "@UTF8\n\
                         @Begin\n\
                         @Languages:\teng, spa\n\
                         @Participants:\tPAR Participant\n\
                         @ID:\teng|test|PAR|||||Participant|||\n\
                         *PAR:\tone foreign three .\n\
                         %mor:\tnum|one L2|xxx num|three .\n\
                         %gra:\t1|2|NUMMOD 2|0|ROOT 3|2|FLAT 4|2|PUNCT\n\
                         @End\n";
        let parser = TreeSitterParser::new().unwrap();
        let (mut chat_file, _errors) = parse_lenient(&parser, chat_text);
        let line_idx = chat_file
            .lines
            .iter()
            .position(|l| matches!(l, talkbank_model::model::Line::Utterance(_)))
            .unwrap();

        let deferred = vec![deferred_position(line_idx, 1, "spa", "root", 0)];
        // Adversarial: head=5 references a nonexistent chunk.
        let merged = vec![Some(PositionResult {
            mor: Mor::new(MorWord::new(PosCategory::new("noun"), MorStem::new("dos"))),
            gras: vec![GrammaticalRelation::new(1, 5, "COMPOUND")],
            attachment: utterance_root_attachment(0),
        })];

        splice_positions(&mut chat_file, &deferred, merged);
        assert_post_splice_gra_valid(&mut chat_file, "C1: secondary head OOB (E713 wild shape)");
    }

    /// **Family C, C2**: secondary's gras form a 2-cycle.
    /// Mirrors the wild E724 dominant pattern
    /// `1|2|DET 2|3|NMOD 3|2|PUNCT` (18 occurrences across the corpus).
    ///
    /// Adversarial input: 2-chunk MWT secondary where chunk 1 → 2 and
    /// chunk 2 → 1 (mutual reference cycle within the secondary slice).
    ///
    /// EXPECTED on current build: FAILS, splice propagates the cycle.
    #[test]
    fn family_c_secondary_cycle_falls_back_or_normalizes() {
        let chat_text = "@UTF8\n\
                         @Begin\n\
                         @Languages:\tfra, eng\n\
                         @Participants:\tPAR Participant\n\
                         @ID:\tfra|test|PAR|||||Participant|||\n\
                         *PAR:\tle foreign .\n\
                         %mor:\tdet|le L2|xxx .\n\
                         %gra:\t1|2|DET 2|0|ROOT 3|2|PUNCT\n\
                         @End\n";
        let parser = TreeSitterParser::new().unwrap();
        let (mut chat_file, _errors) = parse_lenient(&parser, chat_text);
        let line_idx = chat_file
            .lines
            .iter()
            .position(|l| matches!(l, talkbank_model::model::Line::Utterance(_)))
            .unwrap();

        let deferred = vec![deferred_position(line_idx, 1, "eng", "root", 0)];
        // Adversarial: secondary returns a 2-chunk MWT result with an
        // internal 2-cycle.
        let merged = vec![Some(PositionResult {
            mor: Mor::new(MorWord::new(PosCategory::new("noun"), MorStem::new("foo")))
                .with_post_clitic(MorWord::new(PosCategory::new("part"), MorStem::new("bar"))),
            gras: vec![
                GrammaticalRelation::new(1, 2, "FLAT"),
                GrammaticalRelation::new(2, 1, "FLAT"),
            ],
            attachment: utterance_root_attachment(0),
        })];

        splice_positions(&mut chat_file, &deferred, merged);
        assert_post_splice_gra_valid(&mut chat_file, "C2: secondary 2-cycle (E724 wild shape)");
    }

    /// **Family C, C3**: secondary's gras have multiple head=0
    /// relations. Mirrors wild E723 patterns like
    /// `1|0|ROOT 2|3|COP 3|0|ROOT 4|1|PUNCT`.
    ///
    /// Adversarial input: 2-chunk MWT secondary where BOTH chunks
    /// have head=0 / deprel=ROOT (Stanza emitting two roots, which
    /// is malformed UD but does happen).
    ///
    /// EXPECTED on current build: FAILS, splice's merge inserts both
    /// secondary roots, leaving the host with two head=0 relations.
    #[test]
    fn family_c_secondary_multi_root_falls_back_or_normalizes() {
        let chat_text = "@UTF8\n\
                         @Begin\n\
                         @Languages:\teng, spa\n\
                         @Participants:\tPAR Participant\n\
                         @ID:\teng|test|PAR|||||Participant|||\n\
                         *PAR:\thost foreign .\n\
                         %mor:\tverb|host L2|xxx .\n\
                         %gra:\t1|0|ROOT 2|1|OBJ 3|1|PUNCT\n\
                         @End\n";
        let parser = TreeSitterParser::new().unwrap();
        let (mut chat_file, _errors) = parse_lenient(&parser, chat_text);
        let line_idx = chat_file
            .lines
            .iter()
            .position(|l| matches!(l, talkbank_model::model::Line::Utterance(_)))
            .unwrap();

        let deferred = vec![deferred_position(line_idx, 1, "spa", "obj", 1)];
        // Adversarial: secondary returns 2 chunks, BOTH labelled ROOT
        // with head=0.
        let merged = vec![Some(PositionResult {
            mor: Mor::new(MorWord::new(PosCategory::new("noun"), MorStem::new("a")))
                .with_post_clitic(MorWord::new(PosCategory::new("noun"), MorStem::new("b"))),
            gras: vec![
                GrammaticalRelation::new(1, 0, "ROOT"),
                GrammaticalRelation::new(2, 0, "ROOT"),
            ],
            attachment: host_attachment(0, "obj"),
        })];

        splice_positions(&mut chat_file, &deferred, merged);
        assert_post_splice_gra_valid(&mut chat_file, "C3: secondary multi-root (E723 wild shape)");
    }

    /// **Family C, C4**: secondary's per-relation head points at a
    /// host-side terminator chunk after remap. Mirrors the
    /// `bougzers@s` cycle pattern at
    /// `biling-data/MLE-MPF/09.cha:2425`. Wild output:
    /// `1|2|DET 2|3|NMOD 3|2|PUNCT`: chunk 2 (the L2 word) has
    /// head=3 (the period). Cycle 2↔3.
    ///
    /// Adversarial shape: single L2 word as host's utterance root,
    /// secondary returns one chunk whose gra has head=2 within the
    /// secondary's local index space (e.g., Stanza pointed
    /// `bougzers` at the period that followed it in the secondary
    /// dispatch input).
    ///
    /// EXPECTED on current build: FAILS, splice propagates the bogus
    /// head into the host instead of honoring the planner's
    /// `UtteranceRoot` attachment.
    #[test]
    fn family_c_secondary_head_into_host_terminator_falls_back() {
        let chat_text = "@UTF8\n\
                         @Begin\n\
                         @Languages:\tfra, eng\n\
                         @Participants:\tPAR Participant\n\
                         @ID:\tfra|test|PAR|||||Participant|||\n\
                         *PAR:\tle foreign .\n\
                         %mor:\tdet|le L2|xxx .\n\
                         %gra:\t1|2|DET 2|0|ROOT 3|2|PUNCT\n\
                         @End\n";
        let parser = TreeSitterParser::new().unwrap();
        let (mut chat_file, _errors) = parse_lenient(&parser, chat_text);
        let line_idx = chat_file
            .lines
            .iter()
            .position(|l| matches!(l, talkbank_model::model::Line::Utterance(_)))
            .unwrap();

        let deferred = vec![deferred_position(line_idx, 1, "eng", "root", 0)];
        // Adversarial: Stanza English treats placeholder as a dependent
        // of a phantom token at secondary local index 2 (which doesn't
        // exist in this 1-chunk merged response). After the splice's
        // remap, head=2 maps to host chunk 3 (the period), creating a
        // 2-cycle (chunk 2 ↔ chunk 3).
        let merged = vec![Some(PositionResult {
            mor: Mor::new(MorWord::new(
                PosCategory::new("noun"),
                MorStem::new("foreign"),
            )),
            gras: vec![GrammaticalRelation::new(1, 2, "NMOD")],
            attachment: utterance_root_attachment(0),
        })];

        splice_positions(&mut chat_file, &deferred, merged);
        assert_post_splice_gra_valid(
            &mut chat_file,
            "C4: secondary head into host terminator (bougzers wild shape)",
        );
    }

    /// **Family C, C6**: multi-position contiguous span where
    /// `splice_range_coordinated` succeeds with invalid output.
    /// Mirrors the wild `por@s favor@s` shape at
    /// `biling-data/Bangor/Miami/eng/maria/maria20.cha:559-562`:
    ///
    /// ```text
    /// *MAR: they look normal Jackie por@s favor@s .
    /// %mor: pron|they verb|look adj|normal propn|Jackie adp|por noun|favor .
    /// %gra: 1|2|NSUBJ 2|0|ROOT 3|5|AMOD 4|5|COMPOUND 5|4|FLAT 6|5|FIXED 7|2|PUNCT
    /// ```
    ///
    /// Cycle: chunk 4 → chunk 5 (COMPOUND), chunk 5 → chunk 4 (FLAT).
    /// E724 fires.
    ///
    /// Adversarial scenario for the unit test: two contiguous `@s`
    /// positions whose secondaries supply gras that, when concatenated
    ///: encode a 2-cycle within the spliced span (chunk-1 → chunk-2,
    /// chunk-2 → chunk-1). After the host-side index remap this
    /// produces a cycle in the host gra that `splice_range_coordinated`
    /// doesn't reject (the mor chunk count balances; the gra count
    /// balances; only the structural acyclic invariant is violated).
    ///
    /// EXPECTED on current build: FAILS, multi-position branch has
    /// no post-splice validation.
    #[test]
    fn family_c_multi_position_contiguous_internal_cycle_falls_back() {
        let chat_text = "@UTF8\n\
                         @Begin\n\
                         @Languages:\teng, spa\n\
                         @Participants:\tPAR Participant\n\
                         @ID:\teng|test|PAR|||||Participant|||\n\
                         *PAR:\thost a b .\n\
                         %mor:\tverb|host L2|xxx L2|xxx .\n\
                         %gra:\t1|0|ROOT 2|1|DEP 3|1|DEP 4|1|PUNCT\n\
                         @End\n";
        let parser = TreeSitterParser::new().unwrap();
        let (mut chat_file, _errors) = parse_lenient(&parser, chat_text);
        let line_idx = chat_file
            .lines
            .iter()
            .position(|l| matches!(l, talkbank_model::model::Line::Utterance(_)))
            .unwrap();

        // Two CONTIGUOUS positions, same target_lang, exercises the
        // multi-position branch (span_size > 1) of splice_l2_into_chat.
        let deferred = vec![
            deferred_position(line_idx, 1, "spa", "dep", 1),
            deferred_position(line_idx, 2, "spa", "dep", 1),
        ];
        // Adversarial shape: secondary's gras for the 2-position span
        // form an in-span 2-cycle.
        //
        // Secondary returns a 2-chunk block where:
        //   - chunk 1's gra has head=2 (points at chunk 2 within span)
        //   - chunk 2's gra has head=1 (points back at chunk 1)
        //
        // Chunk counts and gra counts balance, so
        // `splice_range_coordinated` accepts the input. Only the cycle
        // invariant is violated.
        let merged = vec![
            Some(PositionResult {
                mor: Mor::new(MorWord::new(PosCategory::new("adp"), MorStem::new("a"))),
                gras: vec![GrammaticalRelation::new(1, 2, "FIXED")],
                attachment: host_attachment(0, "dep"),
            }),
            Some(PositionResult {
                mor: Mor::new(MorWord::new(PosCategory::new("noun"), MorStem::new("b"))),
                gras: vec![GrammaticalRelation::new(1, 1, "FLAT")],
                attachment: host_attachment(1, "dep"),
            }),
        ];

        splice_positions(&mut chat_file, &deferred, merged);
        assert_post_splice_gra_valid(
            &mut chat_file,
            "C6: multi-position contiguous internal cycle (maria20 wild shape)",
        );
    }

    /// **Family C, C5**: joint sweep with two L2 positions in one
    /// utterance: one position has adversarial OOB input, the other
    /// has clean input. Forces the splice to handle multi-position
    /// fallback consistently: the bad position must not poison the
    /// good one, and vice versa.
    ///
    /// EXPECTED on current build: FAILS at the OOB position.
    #[test]
    fn family_c_multi_position_adversarial_sweep_falls_back() {
        let chat_text = "@UTF8\n\
                         @Begin\n\
                         @Languages:\teng, spa\n\
                         @Participants:\tPAR Participant\n\
                         @ID:\teng|test|PAR|||||Participant|||\n\
                         *PAR:\thost a b .\n\
                         %mor:\tverb|host L2|xxx L2|xxx .\n\
                         %gra:\t1|0|ROOT 2|1|DEP 3|1|DEP 4|1|PUNCT\n\
                         @End\n";
        let parser = TreeSitterParser::new().unwrap();
        let (mut chat_file, _errors) = parse_lenient(&parser, chat_text);
        let line_idx = chat_file
            .lines
            .iter()
            .position(|l| matches!(l, talkbank_model::model::Line::Utterance(_)))
            .unwrap();

        let deferred = vec![
            deferred_position(line_idx, 1, "spa", "dep", 1),
            deferred_position(line_idx, 2, "spa", "dep", 1),
        ];
        let merged = vec![
            // Position 1: adversarial OOB head=99
            Some(PositionResult {
                mor: Mor::new(MorWord::new(PosCategory::new("noun"), MorStem::new("a"))),
                gras: vec![GrammaticalRelation::new(1, 99, "COMPOUND")],
                attachment: host_attachment(0, "dep"),
            }),
            // Position 2: clean: should succeed normally and not be
            // disturbed by the failure at position 1.
            Some(PositionResult {
                mor: Mor::new(MorWord::new(PosCategory::new("noun"), MorStem::new("b"))),
                gras: vec![GrammaticalRelation::new(1, 0, "ROOT")],
                attachment: host_attachment(1, "dep"),
            }),
        ];

        splice_positions(&mut chat_file, &deferred, merged);
        assert_post_splice_gra_valid(&mut chat_file, "C5: multi-position adversarial sweep");
    }

    // ─────────────────────────────────────────────────────────────────────
    // L2 redesign 2026-05-07: constructive merge tests.
    //
    // These four tests pin the dominant rollback variants observed in the
    // wild (net captured-tracing run 8b461fee-df9, 750 sample files):
    //
    //   secondary_no_root        40.4%  → constructed away (Tests 1, 5)
    //   secondary_multi_root     38.5%  → constructed away (Test 2)
    //   secondary_head_oob       11.0%  → constructed away (Test 3)
    //   secondary_cycle          10.1%  → genuine fallback (Test 4)
    //
    // Each test fixture is grounded in a real warn-line shape from
    // a captured-tracing morphotag run's `server.log`. Root-cause
    // walk-through is in the private workspace.
    //
    // PRE-FIX: tests 1-3 fail (rollback, fallback==1). Test 4 passes
    // (cycle is irreducible; rollback is the right answer).
    // POST-FIX: tests 1-3 pass (splice succeeds; tree invariants
    // maintained by construction). Test 4 still passes (regression pin).
    // ─────────────────────────────────────────────────────────────────────

    /// Test 1 (no_root): single-position L2 span where the L2 word IS the
    /// host root, and secondary's per-position relation has NO `head=0,
    /// ROOT`. Real-world shape from the warn log: host `4|0|ROOT` at the
    /// L2 position; secondary returns `1|2|NMOD`; current splice replaces
    /// `4|0|ROOT` with `4|5|NMOD`, killing the only root. After fix:
    /// merge ensures the L2 position becomes `head=0, ROOT` because the
    /// `InternalRoot` attachment promises this position owns the host's
    /// root.
    #[test]
    fn merge_constructs_root_when_l2_span_owns_host_root() {
        // Host: `voici yellow@s .`, `yellow` is at word_idx=1 and is the
        // host's primary root (matches the 2026-05-07 warn-line family
        // where the L2 word at word_idx=N IS at host root position).
        let chat_text = "@UTF8\n\
                         @Begin\n\
                         @Languages:\tfra, ara\n\
                         @Participants:\tPAR Participant\n\
                         @ID:\tfra|test|PAR|||||Participant|||\n\
                         *PAR:\tvoici yellow@s .\n\
                         %mor:\tintj|voici L2|xxx .\n\
                         %gra:\t1|2|DISCOURSE 2|0|ROOT 3|2|PUNCT\n\
                         @End\n";
        let parser = TreeSitterParser::new().unwrap();
        let (mut chat_file, _errors) = parse_lenient(&parser, chat_text);
        let line_idx = chat_file
            .lines
            .iter()
            .position(|l| matches!(l, talkbank_model::model::Line::Utterance(_)))
            .unwrap();

        let deferred = vec![deferred_position(line_idx, 1, "ara", "root", 0)];
        // Secondary's relation lacks head=0/ROOT, head=2 mimics the
        // wild warn-line `secondary_gras=["1|2|NMOD"]`. Single-chunk
        // merged Mor (one gra entry).
        let merged = vec![Some(PositionResult {
            mor: Mor::new(MorWord::new(
                PosCategory::new("noun"),
                MorStem::new("yellow"),
            )),
            gras: vec![GrammaticalRelation::new(1, 2, "NMOD")],
            attachment: TestAttachment::Internal,
        })];

        let outcome = splice_positions(&mut chat_file, &deferred, merged);
        assert_eq!(
            outcome.spliced, 1,
            "InternalRoot span must splice successfully; merge must \
             ensure the position carries head=0/ROOT regardless of \
             secondary's relation. Got fallback={}, spliced={}.",
            outcome.fallback, outcome.spliced
        );
        assert_eq!(outcome.fallback, 0);
        assert_post_splice_gra_valid(
            &mut chat_file,
            "Test 1: InternalRoot, no head=0/ROOT in secondary",
        );
        // Stronger: assert the L2 position is the post-splice root.
        let utt = match &chat_file.lines[line_idx] {
            talkbank_model::model::Line::Utterance(u) => u,
            _ => unreachable!(),
        };
        let gra = utt
            .dependent_tiers
            .iter()
            .find_map(|t| match &t.tier {
                talkbank_model::model::DependentTier::Gra(g) => Some(g),
                _ => None,
            })
            .expect("post-splice gra present");
        let roots: Vec<_> = gra
            .relations()
            .iter()
            .filter(|r| r.head == 0 && r.relation.eq_ignore_ascii_case("ROOT"))
            .collect();
        assert_eq!(
            roots.len(),
            1,
            "post-splice gra must have exactly one ROOT; got {} ({:?})",
            roots.len(),
            gra.relations()
        );
    }

    /// Test 2 (multi_root): L2 word is NOT the host root, and secondary
    /// returned `1|0|ROOT` for the L2 word (parsed it as its local root).
    /// Splice must rewrite secondary's `head=0/ROOT` to attach to the
    /// host anchor with the host's deprel, not preserve a second root.
    /// Real-world shape from warn log: host_pre had `16|0|ROOT, 18|...`;
    /// L2 at host pos 18; secondary returned `1|0|ROOT`; current splice
    /// produces both `16|0|ROOT` and `18|0|ROOT`.
    #[test]
    fn merge_does_not_double_root_when_secondary_returns_local_root() {
        // Host: `voici yellow@s .`, `voici` is host root; `yellow` is OBJ.
        let chat_text = "@UTF8\n\
                         @Begin\n\
                         @Languages:\tfra, ara\n\
                         @Participants:\tPAR Participant\n\
                         @ID:\tfra|test|PAR|||||Participant|||\n\
                         *PAR:\tvoici yellow@s .\n\
                         %mor:\tintj|voici L2|xxx .\n\
                         %gra:\t1|0|ROOT 2|1|OBJ 3|1|PUNCT\n\
                         @End\n";
        let parser = TreeSitterParser::new().unwrap();
        let (mut chat_file, _errors) = parse_lenient(&parser, chat_text);
        let line_idx = chat_file
            .lines
            .iter()
            .position(|l| matches!(l, talkbank_model::model::Line::Utterance(_)))
            .unwrap();

        let deferred = vec![deferred_position(line_idx, 1, "ara", "obj", 1)];
        let merged = vec![Some(PositionResult {
            mor: Mor::new(MorWord::new(
                PosCategory::new("noun"),
                MorStem::new("yellow"),
            )),
            gras: vec![GrammaticalRelation::new(1, 0, "ROOT")],
            attachment: host_attachment(0, "obj"),
        })];

        let outcome = splice_positions(&mut chat_file, &deferred, merged);
        assert_eq!(
            outcome.spliced, 1,
            "ExternalRoot span must splice; secondary's `1|0|ROOT` must \
             be rewritten to attach to host anchor. \
             fallback={}, spliced={}",
            outcome.fallback, outcome.spliced
        );
        assert_eq!(outcome.fallback, 0);
        assert_post_splice_gra_valid(
            &mut chat_file,
            "Test 2: ExternalRoot, secondary returns head=0/ROOT",
        );
        let utt = match &chat_file.lines[line_idx] {
            talkbank_model::model::Line::Utterance(u) => u,
            _ => unreachable!(),
        };
        let gra = utt
            .dependent_tiers
            .iter()
            .find_map(|t| match &t.tier {
                talkbank_model::model::DependentTier::Gra(g) => Some(g),
                _ => None,
            })
            .expect("post-splice gra present");
        let roots: Vec<_> = gra
            .relations()
            .iter()
            .filter(|r| r.head == 0 && r.relation.eq_ignore_ascii_case("ROOT"))
            .collect();
        assert_eq!(
            roots.len(),
            1,
            "post-splice gra must have exactly one ROOT (the host's \
             original root, not a duplicate from secondary); got {} ({:?})",
            roots.len(),
            gra.relations()
        );
    }

    /// Test 3 (head_oob): single-position L2 with secondary's head index
    /// way out of bounds. Real-world shape: secondary returned
    /// `1|11|NMOD` when the host has only 10 positions; current splice's
    /// `splice_coordinated` translation produces `9|11|NMOD` and the
    /// validator rejects it. After fix: merge clamps OOB heads to attach
    /// at the host anchor (or treats them as external attachments).
    #[test]
    fn merge_clamps_secondary_head_to_anchor_when_local_index_oob() {
        // Host: `voici yellow@s .`, `voici` is host root; `yellow` is OBJ.
        let chat_text = "@UTF8\n\
                         @Begin\n\
                         @Languages:\tfra, ara\n\
                         @Participants:\tPAR Participant\n\
                         @ID:\tfra|test|PAR|||||Participant|||\n\
                         *PAR:\tvoici yellow@s .\n\
                         %mor:\tintj|voici L2|xxx .\n\
                         %gra:\t1|0|ROOT 2|1|OBJ 3|1|PUNCT\n\
                         @End\n";
        let parser = TreeSitterParser::new().unwrap();
        let (mut chat_file, _errors) = parse_lenient(&parser, chat_text);
        let line_idx = chat_file
            .lines
            .iter()
            .position(|l| matches!(l, talkbank_model::model::Line::Utterance(_)))
            .unwrap();

        let deferred = vec![deferred_position(line_idx, 1, "ara", "obj", 1)];
        // Secondary returns `1|99|NMOD`: head=99 has no host preimage.
        // Today: splice_coordinated maps 99 to a host index out of bounds
        // → secondary_head_oob. After fix: head clamped to anchor (host
        // pos 1, the verb), relation rewritten to host's `obj`.
        let merged = vec![Some(PositionResult {
            mor: Mor::new(MorWord::new(
                PosCategory::new("noun"),
                MorStem::new("yellow"),
            )),
            gras: vec![GrammaticalRelation::new(1, 99, "NMOD")],
            attachment: host_attachment(0, "obj"),
        })];

        let outcome = splice_positions(&mut chat_file, &deferred, merged);
        assert_eq!(
            outcome.spliced, 1,
            "OOB-head span must splice with head clamped to anchor. \
             fallback={}, spliced={}",
            outcome.fallback, outcome.spliced
        );
        assert_eq!(outcome.fallback, 0);
        assert_post_splice_gra_valid(&mut chat_file, "Test 3: ExternalRoot, secondary head OOB");
    }

    /// Test 4 (cycle, REPAIRED): two-position L2 span where the
    /// per-position secondary relations form a cycle within the span.
    ///
    /// **Originally drafted as a regression pin for "cycles are
    /// irreducible".** After implementing the constructive merge, it
    /// turns out cycles ARE repairable: pass 4 of `repair_secondary_gras`
    /// detects a cycle by walking head chains and breaks it by
    /// re-rooting one of the cycle members. The result is a valid tree
    /// (with one position deterministically picked as the root) rather
    /// than a rollback to L2|xxx.
    ///
    /// This is strictly better than rollback: morphology (POS, lemma,
    /// features) is preserved at every position; only the structural
    /// arc that participated in the cycle gets adjusted. The "wrong
    /// attachment within span" cost is bounded to one edge per cycle.
    ///
    /// What stays the same: Test 4 still pins the cycle code path
    /// it now pins that the cycle gets *repaired* rather than that it
    /// rolls back. The regression class this defends against is "the
    /// cycle-detection pass silently regresses to letting the cyclic
    /// gras through to splice, where it fails the post-splice
    /// validator".
    /// Test 5 (multi_root via UtteranceRoot mismatch): the L2 plan
    /// picked `UtteranceRoot` for an L2 word (because the primary
    /// host parse said primary.head=0 for that position at extract
    /// time), but the host's final gra has the root at a DIFFERENT
    /// position. Pre-fix: splice keeps `head=0/ROOT` at the L2
    /// position and the host's existing root, producing two
    /// head=0/ROOT entries → `secondary_multi_root` rollback to
    /// L2|xxx. Post-fix: post-splice repair detects the duplicate
    /// root and demotes the L2 contribution to attach to the host
    /// root with a generic DEP relation.
    ///
    /// Background: this is the 43-rollbacks-per-750-files variant
    /// the constructive merge didn't address.
    #[test]
    fn merge_demotes_duplicate_root_when_host_already_has_one() {
        let chat_text = "@UTF8\n\
                         @Begin\n\
                         @Languages:\teng, fra\n\
                         @Participants:\tPAR Participant\n\
                         @ID:\teng|test|PAR|||||Participant|||\n\
                         *PAR:\tshe said yellow@s .\n\
                         %mor:\tpron|she verb|say-Past L2|xxx .\n\
                         %gra:\t1|2|NSUBJ 2|0|ROOT 3|2|OBJ 4|2|PUNCT\n\
                         @End\n";
        let parser = TreeSitterParser::new().unwrap();
        let (mut chat_file, _) = parse_lenient(&parser, chat_text);
        let line_idx = chat_file
            .lines
            .iter()
            .position(|l| matches!(l, talkbank_model::model::Line::Utterance(_)))
            .unwrap();

        // L2 word at word_idx=2 ("yellow"). Plan thinks utterance
        // root (UtteranceRoot attachment), but host's final gra has
        // the verb at position 2 as root.
        let deferred = vec![deferred_position(line_idx, 2, "fra", "root", 0)];
        let merged = vec![Some(PositionResult {
            mor: Mor::new(MorWord::new(
                PosCategory::new("noun"),
                MorStem::new("yellow"),
            )),
            gras: vec![GrammaticalRelation::new(1, 0, "ROOT")],
            attachment: utterance_root_attachment(0),
        })];

        let outcome = splice_positions(&mut chat_file, &deferred, merged);
        assert_eq!(
            outcome.spliced, 1,
            "L2 with UtteranceRoot conflicting with host root must \
             splice (with demotion) rather than rolling back. \
             fallback={}, spliced={}",
            outcome.fallback, outcome.spliced
        );
        assert_eq!(outcome.fallback, 0);
        assert_post_splice_gra_valid(
            &mut chat_file,
            "Test 5: UtteranceRoot vs host root conflict",
        );

        // Verify post-splice has exactly ONE head=0/ROOT.
        let utt = match &chat_file.lines[line_idx] {
            talkbank_model::model::Line::Utterance(u) => u,
            _ => unreachable!(),
        };
        let gra = utt
            .dependent_tiers
            .iter()
            .find_map(|t| match &t.tier {
                talkbank_model::model::DependentTier::Gra(g) => Some(g),
                _ => None,
            })
            .expect("post-splice gra present");
        let roots: Vec<_> = gra
            .relations()
            .iter()
            .filter(|r| r.head == 0 && r.relation.eq_ignore_ascii_case("ROOT"))
            .collect();
        assert_eq!(
            roots.len(),
            1,
            "exactly one head=0/ROOT must remain (host's original); \
             got {} ({:?})",
            roots.len(),
            gra.relations()
        );
        // The host's root at position 2 should be the surviving one;
        // the L2 word at position 3 (its chunk after the verb) should
        // be demoted to attach to it.
        let host_root = roots[0];
        assert_eq!(host_root.index, 2, "host root preserved at position 2");
    }

    #[test]
    fn merge_falls_back_only_on_genuine_secondary_cycle() {
        let (mut chat_file, line_idx) = build_l2_fixture("eng, fra", "x y", 2);

        let deferred = vec![
            deferred_position(line_idx, 0, "fra", "dep", 0),
            deferred_position(line_idx, 1, "fra", "dep", 0),
        ];
        // Each position's secondary parse points at the OTHER position
        // (in span-local indexing): pos 0's relation says head=2 (= span
        // pos 1), pos 1's relation says head=1 (= span pos 0). Cycle.
        let merged = vec![
            Some(PositionResult {
                mor: Mor::new(MorWord::new(PosCategory::new("noun"), MorStem::new("x"))),
                gras: vec![GrammaticalRelation::new(1, 2, "DEP")],
                attachment: TestAttachment::Internal,
            }),
            Some(PositionResult {
                mor: Mor::new(MorWord::new(PosCategory::new("noun"), MorStem::new("y"))),
                gras: vec![GrammaticalRelation::new(1, 1, "DEP")],
                attachment: TestAttachment::Internal,
            }),
        ];

        let outcome = splice_positions(&mut chat_file, &deferred, merged);
        assert_eq!(
            outcome.spliced, 2,
            "cycle must be repaired into a valid tree; \
             fallback={}, spliced={}",
            outcome.fallback, outcome.spliced
        );
        assert_eq!(outcome.fallback, 0);
        assert_post_splice_gra_valid(
            &mut chat_file,
            "Test 4: cycle in secondary, repaired via cycle-detection pass",
        );
    }

    /// Spans splice in transcript order whatever order the caller merged
    /// them in (the dispatcher batches by language). Each splice reads the
    /// %gra the previous ones left, so the order is part of the result.
    #[test]
    fn spans_splice_in_transcript_order_whatever_the_input_order() {
        let chat_text = "@UTF8\n\
                         @Begin\n\
                         @Languages:\teng, spa, fra\n\
                         @Participants:\tPAR Participant\n\
                         @ID:\teng|test|PAR|||||Participant|||\n\
                         *PAR:\thost uno bridge dos tail .\n\
                         %mor:\tverb|host L2|xxx noun|bridge L2|xxx noun|tail .\n\
                         %gra:\t1|0|ROOT 2|1|DEP 3|1|OBJ 4|5|DEP 5|1|OBL 6|1|PUNCT\n\
                         @End\n";
        let parser = TreeSitterParser::new().unwrap();
        let line_idx = parse_lenient(&parser, chat_text)
            .0
            .lines
            .iter()
            .position(|l| matches!(l, talkbank_model::model::Line::Utterance(_)))
            .expect("fixture has an utterance");
        let spans = |source: &L2DeferredPosition, other: &L2DeferredPosition| {
            vec![
                MergedL2Span::for_splice_test(
                    line_idx,
                    source.word_idx(),
                    vec![
                        Mor::new(MorWord::new(PosCategory::new("noun"), MorStem::new("uno")))
                            .with_post_clitic(MorWord::new(
                                PosCategory::new("pron"),
                                MorStem::new("lo"),
                            )),
                    ],
                    vec![
                        GrammaticalRelation::new(1, 0, "ROOT"),
                        GrammaticalRelation::new(2, 1, "OBJ"),
                    ],
                    L2Attachment::HostGovernor {
                        source_word: source.word_idx(),
                        source: source.primary().clone(),
                        relation: ExternalRelation::Primary,
                    },
                ),
                MergedL2Span::for_splice_test(
                    line_idx,
                    other.word_idx(),
                    vec![Mor::new(MorWord::new(
                        PosCategory::new("noun"),
                        MorStem::new("dos"),
                    ))],
                    vec![GrammaticalRelation::new(1, 0, "ROOT")],
                    L2Attachment::HostGovernor {
                        source_word: other.word_idx(),
                        source: other.primary().clone(),
                        relation: ExternalRelation::Primary,
                    },
                ),
            ]
        };
        let uno = deferred_position(line_idx, 1, "spa", "obj", 1);
        let dos = deferred_position(line_idx, 3, "fra", "nmod", 5);
        let render = |reverse: bool| {
            let (mut chat_file, _errors) = parse_lenient(&parser, chat_text);
            let mut merged = spans(&uno, &dos);
            if reverse {
                merged.reverse();
            }
            let outcome = splice_l2_into_chat(&mut chat_file, merged);
            assert_eq!(outcome.spliced, 2, "{outcome:?}");
            let talkbank_model::model::Line::Utterance(utt) = &chat_file.lines[line_idx] else {
                unreachable!()
            };
            utt.gra_tier().map(|gra| gra.to_chat_string())
        };
        assert_eq!(render(true), render(false));
    }

    /// Where a test position's span root attaches, by index into the
    /// test's `deferred` list.
    enum TestAttachment {
        Internal,
        Host(usize, &'static str),
        UtteranceRoot(usize),
    }

    /// One word's result as these tests state it. [`splice_positions`]
    /// groups consecutive results into spans the way the plan groups
    /// words, each span taking its first position's attachment.
    struct PositionResult {
        mor: Mor,
        gras: Vec<GrammaticalRelation>,
        attachment: TestAttachment,
    }

    /// Group per-word results into merged spans (a `None` breaks a span and
    /// counts as a fallback) and splice them.
    fn splice_positions(
        chat_file: &mut talkbank_model::model::ChatFile,
        deferred: &[L2DeferredPosition],
        merged: Vec<Option<PositionResult>>,
    ) -> SpliceOutcome {
        let attachment = |test: &TestAttachment| match *test {
            TestAttachment::Internal => L2Attachment::InternalRoot,
            // The test's relation is the primary's when it matches it, and
            // otherwise the merge's correction.
            TestAttachment::Host(source, deprel) => {
                let primary = deferred[source].primary().clone();
                let relation = match primary.deprel().as_str() == deprel {
                    true => ExternalRelation::Primary,
                    false => ExternalRelation::Corrected(UdDeprel::new(deprel)),
                };
                L2Attachment::HostGovernor {
                    source_word: deferred[source].word_idx(),
                    source: primary,
                    relation,
                }
            }
            TestAttachment::UtteranceRoot(source) => L2Attachment::UtteranceRoot {
                source_word: deferred[source].word_idx(),
            },
        };
        let mut runs: Vec<Vec<(&L2DeferredPosition, PositionResult)>> = Vec::new();
        let mut unmerged = 0;
        let mut previous: Option<&L2DeferredPosition> = None;
        for (position, result) in deferred.iter().zip(merged) {
            let Some(result) = result else {
                unmerged += 1;
                previous = None;
                continue;
            };
            let extends = previous.is_some_and(|previous| {
                previous.line_idx() == position.line_idx()
                    && previous.target_lang() == position.target_lang()
                    && crate::morphosyntax::l2::plan::follows(
                        previous.word_idx(),
                        position.word_idx(),
                    )
            });
            match runs.last_mut() {
                Some(run) if extends => run.push((position, result)),
                Some(_) | None => runs.push(vec![(position, result)]),
            }
            previous = Some(position);
        }
        let spans = runs
            .into_iter()
            .map(|run| {
                let (first, first_result) = &run[0];
                let span_attachment = attachment(&first_result.attachment);
                let line_idx = first.line_idx();
                let first_word = first.word_idx();
                let (mors, gras): (Vec<Mor>, Vec<Vec<GrammaticalRelation>>) = run
                    .into_iter()
                    .map(|(_, result)| (result.mor, result.gras))
                    .unzip();
                MergedL2Span::for_splice_test(
                    line_idx,
                    first_word,
                    mors,
                    gras.into_iter()
                        .flatten()
                        .enumerate()
                        .map(|(position, mut relation)| {
                            relation.index = position + 1;
                            relation
                        })
                        .collect(),
                    span_attachment,
                )
            })
            .collect();
        let mut outcome = splice_l2_into_chat(chat_file, spans);
        outcome.fallback += unmerged;
        outcome
    }

    /// Test helper: a span root attached under the host governor of the
    /// position at `source`, with `deprel`.
    fn host_attachment(source: usize, deprel: &'static str) -> TestAttachment {
        TestAttachment::Host(source, deprel)
    }

    /// Test helper: a span root that is the utterance root, decided by the
    /// position at `source`.
    fn utterance_root_attachment(source: usize) -> TestAttachment {
        TestAttachment::UtteranceRoot(source)
    }

    /// A one-utterance file with the given main tier and host tiers.
    fn host_file(main: &str, mor: &str, gra: &str) -> (talkbank_model::model::ChatFile, usize) {
        let chat_text = format!(
            "@UTF8\n@Begin\n@Languages:\teng, spa\n@Participants:\tPAR Participant\n\
             @ID:\teng|test|PAR|||||Participant|||\n*PAR:\t{main}\n%mor:\t{mor}\n\
             %gra:\t{gra}\n@End\n"
        );
        let parser = TreeSitterParser::new().expect("parser");
        let (chat_file, _errors) = parse_lenient(&parser, &chat_text);
        let line_idx = chat_file
            .lines
            .iter()
            .position(|l| matches!(l, talkbank_model::model::Line::Utterance(_)))
            .expect("fixture has an utterance");
        (chat_file, line_idx)
    }

    /// The `%gra` line of the fixture's utterance.
    fn gra_line(chat_file: &talkbank_model::model::ChatFile, line_idx: usize) -> String {
        let talkbank_model::model::Line::Utterance(utt) = &chat_file.lines[line_idx] else {
            unreachable!("the fixture line is an utterance")
        };
        utt.gra_tier()
            .map(|gra| gra.to_chat_string())
            .unwrap_or_default()
    }

    /// A corrected relation is counted only where it was written.
    ///
    /// `dos` attaches outside the span to `casa`, but `casa` depends on the
    /// span (`uno`), so attaching the span root to it would make a cycle:
    /// the span root stays the utterance root, and the corrected relation
    /// is written nowhere. `gra_upgraded` used to count it anyway, one per
    /// span with a correction.
    #[test]
    fn an_unwritten_correction_is_not_counted() {
        let (mut chat_file, line_idx) = host_file(
            "uno@s:spa dos@s:spa casa .",
            "L2|xxx L2|xxx noun|casa .",
            "1|0|ROOT 2|3|OBJ 3|1|NMOD 4|1|PUNCT",
        );
        let deferred = vec![
            deferred_position(line_idx, 0, "spa", "root", 0),
            deferred_position(line_idx, 1, "spa", "obj", 3),
        ];
        let merged = vec![
            Some(PositionResult {
                mor: Mor::new(MorWord::new(PosCategory::new("num"), MorStem::new("uno"))),
                gras: vec![GrammaticalRelation::new(1, 2, "NUMMOD")],
                attachment: host_attachment(1, "nmod"),
            }),
            Some(PositionResult {
                mor: Mor::new(MorWord::new(PosCategory::new("noun"), MorStem::new("dos"))),
                gras: vec![GrammaticalRelation::new(2, 0, "ROOT")],
                attachment: TestAttachment::Internal,
            }),
        ];

        let outcome = splice_positions(&mut chat_file, &deferred, merged);

        assert_eq!(outcome.spliced, 2, "{outcome:?}");
        let gra = gra_line(&chat_file, line_idx);
        assert!(
            gra.contains(" 2|0|ROOT "),
            "the span root stays the root: {gra}"
        );
        assert_eq!(outcome.gra_upgraded, 0, "no corrected relation was written");
    }

    /// A host anchor that cannot be read is an error with its reason, not
    /// "no anchor": the latter used to send the splice on with the span
    /// root's old head.
    #[test]
    fn an_unreadable_host_anchor_is_an_error() {
        let (chat_file, line_idx) = host_file(
            "dos@s:spa casa .",
            "L2|xxx noun|casa .",
            "1|2|NMOD 2|0|ROOT 3|2|PUNCT",
        );
        let talkbank_model::model::Line::Utterance(utt) = &chat_file.lines[line_idx] else {
            unreachable!("the fixture line is an utterance")
        };
        let mor = utt.mor_tier().expect("%mor");
        let gra = utt.gra_tier().expect("%gra");
        let source = deferred_position(line_idx, 5, "spa", "nmod", 2);
        let attachment = L2Attachment::HostGovernor {
            source_word: source.word_idx(),
            source: source.primary().clone(),
            relation: ExternalRelation::Primary,
        };
        assert!(
            current_root_anchor_for_attachment(mor, gra, &attachment).is_err(),
            "item 5 has no %gra relation to read"
        );
    }

    /// The span root attaches to the host word it named, after a span that
    /// grew past it.
    ///
    /// `dos` (one chunk) becomes `verb|dar~pron|lo` (two), and its host
    /// governor `casa` lies after it, at chunk 2 before the splice and 3
    /// after. chatter v0.27.0 applies the anchor as given, after the splice,
    /// so the root lands on the span's own clitic and the cycle rolls the
    /// splice back. chatter v0.28's `SpanRoot::HostChunk` names the chunk
    /// before the splice and translates it.
    #[test]
    fn a_span_root_after_a_growing_span_attaches_to_its_host_word() {
        let (mut chat_file, line_idx) = host_file(
            "dos@s:spa casa .",
            "L2|xxx noun|casa .",
            "1|2|NMOD 2|0|ROOT 3|2|PUNCT",
        );
        let deferred = vec![deferred_position(line_idx, 0, "spa", "nmod", 2)];
        let merged = vec![Some(PositionResult {
            mor: Mor::new(MorWord::new(PosCategory::new("verb"), MorStem::new("dar")))
                .with_post_clitic(MorWord::new(PosCategory::new("pron"), MorStem::new("lo"))),
            gras: vec![
                GrammaticalRelation::new(1, 0, "ROOT"),
                GrammaticalRelation::new(2, 1, "OBJ"),
            ],
            attachment: host_attachment(0, "nmod"),
        })];

        let outcome = splice_positions(&mut chat_file, &deferred, merged);

        assert_eq!(outcome.spliced, 1, "{outcome:?}");
        assert_eq!(
            gra_line(&chat_file, line_idx),
            "%gra:\t1|3|NMOD 2|1|OBJ 3|0|ROOT 4|3|PUNCT"
        );
    }

    /// A farther governor must follow its word when a preceding span grows.
    #[test]
    fn a_growing_span_keeps_its_farther_host_governor() {
        let (mut chat_file, line_idx) = host_file(
            "dos@s:spa muy casa .",
            "L2|xxx adv|muy noun|casa .",
            "1|3|NMOD 2|3|ADVMOD 3|0|ROOT 4|3|PUNCT",
        );
        let deferred = vec![deferred_position(line_idx, 0, "spa", "nmod", 3)];
        let merged = vec![Some(PositionResult {
            mor: Mor::new(MorWord::new(PosCategory::new("verb"), MorStem::new("dar")))
                .with_post_clitic(MorWord::new(PosCategory::new("pron"), MorStem::new("lo"))),
            gras: vec![
                GrammaticalRelation::new(1, 0, "ROOT"),
                GrammaticalRelation::new(2, 1, "OBJ"),
            ],
            attachment: host_attachment(0, "nmod"),
        })];
        let outcome = splice_positions(&mut chat_file, &deferred, merged);
        assert_eq!(outcome.spliced, 1, "{outcome:?}");
        assert_eq!(
            gra_line(&chat_file, line_idx),
            "%gra:\t1|4|NMOD 2|1|OBJ 3|4|ADVMOD 4|0|ROOT 5|4|PUNCT"
        );
    }

    /// A host dependent of the primary representative follows the secondary root.
    #[test]
    fn host_dependents_of_the_attachment_source_follow_the_span_root() {
        let (mut chat_file, line_idx) = host_file(
            "uno@s:spa dos@s:spa muy casa .",
            "L2|xxx L2|xxx adv|muy noun|casa .",
            "1|4|NMOD 2|1|OBJ 3|1|ADVMOD 4|0|ROOT 5|4|PUNCT",
        );
        let deferred = vec![
            deferred_position(line_idx, 0, "spa", "nmod", 4),
            deferred_position(line_idx, 1, "spa", "obj", 1),
        ];
        let merged = vec![
            Some(PositionResult {
                mor: Mor::new(MorWord::new(PosCategory::new("num"), MorStem::new("uno"))),
                gras: vec![GrammaticalRelation::new(1, 2, "NUMMOD")],
                attachment: host_attachment(0, "nmod"),
            }),
            Some(PositionResult {
                mor: Mor::new(MorWord::new(PosCategory::new("noun"), MorStem::new("dos"))),
                gras: vec![GrammaticalRelation::new(2, 0, "ROOT")],
                attachment: TestAttachment::Internal,
            }),
        ];
        let outcome = splice_positions(&mut chat_file, &deferred, merged);
        assert_eq!(outcome.spliced, 2, "{outcome:?}");
        assert_eq!(
            gra_line(&chat_file, line_idx),
            "%gra:\t1|2|NUMMOD 2|4|NMOD 3|2|ADVMOD 4|0|ROOT 5|4|PUNCT"
        );
    }
}
