//! Inject morphosyntax data into a CHAT AST utterance.
//!
//! After parsing %mor and %gra strings into typed Rust structures,
//! this module adds full tier structs to the utterance's dependent tiers.
//!
//! [`inject_morphosyntax`] validates alignment before injecting:
//! - MOR item count must match alignable word count (Mor domain).
//! - GRA relation count must match MOR chunk count.
//!
//! A count mismatch is always a bug in one of the three stages that
//! together make the 1-to-1 invariant hold (CHAT extraction via
//! `counts_for_tier`, Stanza tokenizer realignment via
//! `tok_ctx.original_words`, UD→Mor mapping with MWT reassembly).
//! The typed [`MisalignmentDiagnostic`] error carries the input words
//! and the expected/actual counts so the caller can triage without
//! re-running. See `morphosyntax/outcome.rs` for the full outcome model.

use talkbank_model::Location;
use talkbank_model::WriteChat;
use talkbank_model::alignment::helpers::{
    MorAlignableWordCount, MorItemCount, PositionalDomain, count_tier_positions,
};
use talkbank_model::model::{
    DependentTier, GraTier, GrammaticalRelation, Mor, MorTier, TierContentItems, Utterance,
    UtteranceContent,
};

use crate::extract::collect_utterance_content;

pub use crate::dependent_tiers::replace_or_add_tier;

/// Diagnostic data for a misalignment bug, enough to triage the failing stage
/// without re-running the pipeline.
#[derive(Debug, Clone, thiserror::Error)]
#[error(
    "morphotag misalignment: class={} expected={expected} actual={actual} \
     chat_words={chat_words:?} stanza_tokens={stanza_tokens_after_mapping:?}",
    suspected_class.as_str()
)]
pub struct MisalignmentDiagnostic {
    /// The Mor-alignable words extracted from the CHAT main tier.
    pub chat_words: Vec<String>,
    /// The tokens produced after tokenizer realignment and mapping.
    pub stanza_tokens_after_mapping: Vec<String>,
    /// Expected count: CHAT Mor-alignable words.
    pub expected: MorAlignableWordCount,
    /// Actual count: `%mor` items produced by mapping.
    pub actual: MorItemCount,
    /// Best-effort classification of the likely failing stage.
    pub suspected_class: MisalignmentClass,
}

impl MisalignmentDiagnostic {
    /// The model's tokens could not be placed on the transcript's words: no
    /// word owns them, so the words and the tokens are reported side by side.
    pub fn unplaced(words: &[crate::extract::ExtractedWord], tokens: Vec<String>) -> Self {
        Self {
            chat_words: words
                .iter()
                .map(|word| word.text.as_str().to_owned())
                .collect(),
            expected: MorAlignableWordCount::new(words.len()),
            actual: MorItemCount::new(tokens.len()),
            stanza_tokens_after_mapping: tokens,
            suspected_class: MisalignmentClass::TokensDivergeFromWords,
        }
    }
}

/// Which stage a morphotag misalignment came from, set by the code that
/// observed it. Only classes some producer sets exist: a class reached by
/// guessing named the wrong stage (every retokenize refusal was once
/// "mwt_reassembly_bug").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MisalignmentClass {
    /// `%mor` chunk count and `%gra` relation count disagreed at injection
    /// time. Upstream sentence-mapping likely failed to append a terminator
    /// PUNCT relation (or appended one for an utterance that doesn't have a
    /// terminator chunk). dona@s bug class.
    MorGraCountMismatch,
    /// Retokenization: a model token joins words that other main-tier
    /// content (a retrace, a pause, a bracket edge) separates.
    MergeCrossesContent,
    /// The model's tokens do not spell the transcript's words, so no word
    /// can be said to own any token.
    TokensDivergeFromWords,
    /// Diagnostic data is insufficient to classify more precisely.
    Unknown,
}

impl MisalignmentClass {
    /// Stable label for tracing and structured decision evidence.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::MorGraCountMismatch => "mor_gra_count_mismatch",
            Self::MergeCrossesContent => "merge_crosses_content",
            Self::TokensDivergeFromWords => "tokens_diverge_from_words",
            Self::Unknown => "unknown",
        }
    }
}

/// Inject parsed Mor items and GRA relations into an utterance: admit them
/// against its content ([`admit_morphosyntax`]), then write them. Atomic by
/// construction: on a refusal the utterance is unchanged.
///
/// # Errors
///
/// The [`MisalignmentDiagnostic`] [`admit_morphosyntax`] returns.
#[tracing::instrument(
    level = "info",
    skip_all,
    fields(utterance = %utterance.main.to_chat_string()),
)]
pub fn inject_morphosyntax(
    utterance: &mut Utterance,
    mors: Vec<Mor>,
    terminator: talkbank_model::Terminator,
    gra_relations: Vec<GrammaticalRelation>,
) -> Result<(), MisalignmentDiagnostic> {
    admit_morphosyntax(utterance, mors, terminator, gra_relations)?.write();
    Ok(())
}

/// Morphology admitted for one utterance, holding that utterance: the
/// `%mor` item count equals the Mor-alignable word count of the main-tier
/// content it will annotate, and a `%gra` tier, when there is one, pairs one
/// relation with every `%mor` chunk.
///
/// Built only by [`admit_morphosyntax`] (against the utterance's own content)
/// or [`admit_rewrite`] (against the content that will replace it), neither of
/// which modifies anything, and written only by [`AdmittedMorphology::write`],
/// which cannot fail and writes the utterance it was admitted for. Every
/// refusal therefore happens before the one write, admitted tiers cannot be
/// written to another utterance, and rewritten content and its tiers are
/// installed together: a caller that declines one utterance leaves it exactly
/// as it was.
#[must_use = "admitted morphology does nothing until it is written"]
pub struct AdmittedMorphology<'u> {
    utterance: &'u mut Utterance,
    rewrite: Option<Vec<UtteranceContent>>,
    tiers: Admitted,
}

/// What admission produced. Private, so the only route to a variant is
/// admission itself.
enum Admitted {
    /// No items, for content with no Mor-alignable word.
    Nothing,
    /// A `%mor` tier alone. Some paths legitimately produce `%mor` without a
    /// paired `%gra` (e.g. utseg).
    Mor(MorTier),
    /// A `%mor` and `%gra` pair, co-constructed so their alignment holds.
    MorGra(MorTier, GraTier),
}

impl AdmittedMorphology<'_> {
    /// Install the rewritten content, if any, and the admitted tiers. A tier
    /// the admission did not produce is removed rather than left beside new
    /// ones: a `%gra` from an earlier run indexes the earlier `%mor`'s items.
    pub fn write(self) {
        let Self {
            utterance,
            rewrite,
            tiers,
        } = self;
        if let Some(content) = rewrite {
            utterance.main.content.content = TierContentItems::new(content);
        }
        let tiers_of = &mut utterance.dependent_tiers;
        match tiers {
            Admitted::Nothing => {
                tiers_of.retain(|entry| {
                    !matches!(entry.tier, DependentTier::Mor(_) | DependentTier::Gra(_))
                });
            }
            Admitted::Mor(mor) => {
                tiers_of.retain(|entry| !matches!(entry.tier, DependentTier::Gra(_)));
                replace_or_add_tier(tiers_of, DependentTier::Mor(mor));
            }
            Admitted::MorGra(mor, gra) => {
                replace_or_add_tier(tiers_of, DependentTier::Mor(mor));
                replace_or_add_tier(tiers_of, DependentTier::Gra(gra));
            }
        }
    }
}

/// Admit Mor items and GRA relations against the utterance's own main-tier
/// content, without modifying anything.
///
/// # Errors
///
/// The [`MisalignmentDiagnostic`] of [`admit_tiers`].
pub fn admit_morphosyntax(
    utterance: &mut Utterance,
    mors: Vec<Mor>,
    terminator: talkbank_model::Terminator,
    gra_relations: Vec<GrammaticalRelation>,
) -> Result<AdmittedMorphology<'_>, MisalignmentDiagnostic> {
    let tiers = admit_tiers(
        &utterance.main.content.content,
        mors,
        terminator,
        gra_relations,
    )?;
    Ok(AdmittedMorphology {
        utterance,
        rewrite: None,
        tiers,
    })
}

/// Admit Mor items and GRA relations against `content`, the main-tier content
/// that will replace the utterance's own when the admission is written.
///
/// # Errors
///
/// The [`MisalignmentDiagnostic`] of [`admit_tiers`].
pub fn admit_rewrite(
    utterance: &mut Utterance,
    content: Vec<UtteranceContent>,
    mors: Vec<Mor>,
    terminator: talkbank_model::Terminator,
    gra_relations: Vec<GrammaticalRelation>,
) -> Result<AdmittedMorphology<'_>, MisalignmentDiagnostic> {
    let tiers = admit_tiers(&content, mors, terminator, gra_relations)?;
    Ok(AdmittedMorphology {
        utterance,
        rewrite: Some(content),
        tiers,
    })
}

/// Admit parsed Mor items and GRA relations against the main-tier content they
/// will annotate. Private: the public entries bind the result to its utterance.
///
/// Validates alignment at the point of corruption rather than deferring to
/// pre-serialization validation.
///
/// # Errors
///
/// Returns a [`MisalignmentDiagnostic`] when the 1-to-1 invariant is
/// violated (Mor count differs from the Mor-alignable CHAT word count, or the
/// `%gra` relations do not pair with the `%mor` chunks). The diagnostic
/// carries `chat_words`, `expected`, `actual`, and a `suspected_class`;
/// callers with more context (the model's tokens) may enrich it.
///
/// # Instrumentation contract
///
/// Every call emits at least one `tracing` event so an operator running with
/// `--verbose` (filter at least info) can reconstruct the path taken for any
/// utterance:
///
/// | Branch                                   | Level | Target message                        |
/// |------------------------------------------|-------|---------------------------------------|
/// | Entry (always)                           | info  | `inject_morphosyntax: enter`          |
/// | mors empty (skip)                        | info  | `inject_morphosyntax: skip empty`     |
/// | word/mor count mismatch (returns Err)    | warn  | `MOR count mismatch ...`              |
/// | %mor-only (no %gra) install              | info  | `inject_morphosyntax: mor-only`       |
/// | %mor/%gra count mismatch (returns Err)   | warn  | `%mor/%gra count mismatch ...`        |
/// | aligned (success)                        | info  | `inject_morphosyntax: aligned`        |
///
/// At the daemon's default filter (`warn`), only the two error events fire.
fn admit_tiers(
    content: &[UtteranceContent],
    mors: Vec<Mor>,
    terminator: talkbank_model::Terminator,
    gra_relations: Vec<GrammaticalRelation>,
) -> Result<Admitted, MisalignmentDiagnostic> {
    tracing::info!(
        mor_count = mors.len(),
        gra_count = gra_relations.len(),
        "inject_morphosyntax: enter",
    );

    // The canonical N applies the CHAT manual's alignment rules
    // (`talkbank-model`'s `count_tier_positions` over the Mor domain, which
    // `Utterance::mor_alignable_word_count` also reads). Typed counts keep
    // the two sides from being swapped at this boundary.
    let word_count =
        MorAlignableWordCount::new(count_tier_positions(content, PositionalDomain::Mor));
    let mor_count = MorItemCount::new(mors.len());
    tracing::debug!(
        word_count = word_count.get(),
        mor_count = mor_count.get(),
        "inject_morphosyntax: alignment check"
    );
    if word_count.get() != mor_count.get() {
        // The 1-to-1 invariant (CHAT extraction, tokenizer realignment and
        // UD-to-Mor mapping each keep N) is violated: always a bug in one of
        // them, never an expected divergence. The word list is built only for
        // the diagnostic.
        let mut extracted = Vec::new();
        collect_utterance_content(content, PositionalDomain::Mor, &mut extracted);
        tracing::warn!(
            word_count = word_count.get(),
            mor_count = mor_count.get(),
            "MOR count mismatch: returning MisalignmentDiagnostic"
        );
        return Err(MisalignmentDiagnostic {
            chat_words: extracted
                .iter()
                .map(|w| w.text.as_ref().to_string())
                .collect(),
            // The injector sees only the mapped Mors, not the model's tokens;
            // the caller enriches with those when it has them.
            stanza_tokens_after_mapping: Vec::new(),
            expected: word_count,
            actual: mor_count,
            suspected_class: MisalignmentClass::Unknown,
        });
    }

    // Counts agree, so no items means no Mor-alignable words: content with
    // nothing to annotate, never an analysed utterance reported complete.
    if mors.is_empty() {
        tracing::info!("inject_morphosyntax: skip empty");
        return Ok(Admitted::Nothing);
    }

    // The terminator's PUNCT relation is the last of `gra_relations` (the
    // upstream AppendTrailingPunct policy); it fills the typed terminator
    // slot. No relations at all means a `%mor` alone.
    let mut item_relations = gra_relations;
    let Some(terminator_relation) = item_relations.pop() else {
        tracing::info!(mor_count = mors.len(), "inject_morphosyntax: mor-only");
        return Ok(Admitted::Mor(MorTier::new_mor(
            Location::Unlocated,
            mors,
            terminator,
        )));
    };
    let slot = talkbank_model::alignment::MorGraTerminatorSlot {
        terminator,
        relation: terminator_relation,
    };
    match talkbank_model::alignment::try_align_mor_gra(
        mors,
        item_relations,
        slot,
        Location::Unlocated,
    ) {
        Ok((mor_tier, gra_tier)) => {
            tracing::info!(
                mor_chunks = mor_tier.count_chunks(),
                gra_relations = gra_tier.relations().len(),
                "inject_morphosyntax: aligned",
            );
            Ok(Admitted::MorGra(mor_tier, gra_tier))
        }
        Err(talkbank_model::alignment::MorGraConstructionError::CountMismatch {
            mor_chunks,
            gra_relations,
        }) => {
            // dona@s bug class: upstream produced misaligned counts. The
            // chunk and relation counts travel in the typed count fields.
            tracing::warn!(
                mor_chunks,
                gra_relations,
                "%mor/%gra count mismatch: returning MisalignmentDiagnostic",
            );
            Err(MisalignmentDiagnostic {
                chat_words: Vec::new(),
                stanza_tokens_after_mapping: Vec::new(),
                expected: MorAlignableWordCount::new(mor_chunks),
                actual: MorItemCount::new(gra_relations),
                suspected_class: MisalignmentClass::MorGraCountMismatch,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use talkbank_model::model::{ChatFile, Line, WriteChat};
    use talkbank_parser::TreeSitterParser;
    use talkbank_parser_tests::test_error::ParseProductTestExt;

    const HELLO_CHAT: &str = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI||female|||Target_Child|||\n*CHI:\thello .\n@End\n";

    fn parse_chat(text: &str) -> ChatFile {
        let parser = TreeSitterParser::new().unwrap();
        parser.parse_chat_file(text).expect_built()
    }

    fn get_utterance(chat: &mut ChatFile, idx: usize) -> &mut Utterance {
        let mut utt_idx = 0;
        for line in &mut chat.lines {
            if let Line::Utterance(utt) = line {
                if utt_idx == idx {
                    return utt;
                }
                utt_idx += 1;
            }
        }
        panic!("Utterance {idx} not found");
    }

    #[test]
    fn test_replace_or_add_tier_user_defined() {
        use talkbank_model::model::{NonEmptyString, UserDefinedDependentTier};

        let mut tiers = smallvec::smallvec![];

        // Add %xtra
        let xtra1 = DependentTier::UserDefined(UserDefinedDependentTier {
            label: talkbank_model::UserDefinedDependentTierLabel::new("xtra").unwrap(),
            content: Some(NonEmptyString::new("first").unwrap().into()),
            location: Location::Unlocated,
        });
        replace_or_add_tier(&mut tiers, xtra1);
        assert_eq!(tiers.len(), 1);

        // Replace %xtra with new content
        let xtra2 = DependentTier::UserDefined(UserDefinedDependentTier {
            label: talkbank_model::UserDefinedDependentTierLabel::new("xtra").unwrap(),
            content: Some(NonEmptyString::new("second").unwrap().into()),
            location: Location::Unlocated,
        });
        replace_or_add_tier(&mut tiers, xtra2);
        assert_eq!(tiers.len(), 1); // replaced, not appended

        // Verify content was replaced
        if let DependentTier::UserDefined(ud) = &tiers[0].tier {
            assert_eq!(ud.content.as_ref().map(|c| c.as_str()), Some("second"));
        } else {
            panic!("Expected UserDefined tier");
        }

        // Add %xcod (different label), should NOT replace %xtra
        let xcod = DependentTier::UserDefined(UserDefinedDependentTier {
            label: talkbank_model::UserDefinedDependentTierLabel::new("xcod").unwrap(),
            content: Some(NonEmptyString::new("code").unwrap().into()),
            location: Location::Unlocated,
        });
        replace_or_add_tier(&mut tiers, xcod);
        assert_eq!(tiers.len(), 2); // appended, not replaced
    }

    #[test]
    fn test_replace_or_add_tier_replaces_existing_wor() {
        use talkbank_model::model::WorTier;

        let mut tiers = smallvec::smallvec![talkbank_model::model::DependentTierEntry::new(
            DependentTier::Wor(WorTier::from_words(Location::Unlocated, Vec::new()))
        )];
        let replacement = DependentTier::Wor(WorTier::from_words(
            Location::Unlocated,
            vec![talkbank_model::model::Word::new(
                Location::Unlocated,
                talkbank_model::WordText::new("hello").expect("nonempty fixture word"),
            )],
        ));

        replace_or_add_tier(&mut tiers, replacement);

        assert_eq!(tiers.len(), 1);
        let DependentTier::Wor(wor) = &tiers[0].tier else {
            panic!("expected %wor tier");
        };
        assert_eq!(wor.words().count(), 1);
    }

    /// SURVIVES: boundary. No items for an utterance that has words is a
    /// misalignment, refused with the utterance untouched: it used to be
    /// admitted as "nothing to write", so an analysed utterance could report
    /// completed morphology while keeping none, or a stale `%mor`.
    #[test]
    fn no_items_for_an_utterance_with_words_is_refused_and_leaves_it_untouched() {
        let mut chat = parse_chat(HELLO_CHAT);
        let output_before = chat.to_chat_string();

        let utt = get_utterance(&mut chat, 0);
        let refused = inject_morphosyntax(
            utt,
            Vec::new(),
            talkbank_model::Terminator::Period {
                location: Location::Unlocated,
            },
            Vec::new(),
        )
        .expect_err("one word, no items");
        assert_eq!((refused.expected.get(), refused.actual.get()), (1, 0));
        assert_eq!(output_before, chat.to_chat_string());
    }

    /// A Mor-count / main-tier word-count mismatch must surface as `Err`
    /// warning-and-continuing hides mapping bugs silently.
    #[test]
    fn inject_morphosyntax_count_mismatch_returns_err() {
        use talkbank_model::model::dependent_tier::mor::{MorStem, MorWord, PosCategory};

        // eng_hello_female.cha's first utterance is `hello .`, one
        // alignable word on the main tier. Supply TWO Mor items to force
        // a mismatch (2 vs 1).
        let mut chat = parse_chat(HELLO_CHAT);
        let utt = get_utterance(&mut chat, 0);

        let mor = |pos: &str, lemma: &str| {
            Mor::new(MorWord::new(PosCategory::new(pos), MorStem::new(lemma)))
        };
        let too_many_mors = vec![mor("intj", "hello"), mor("intj", "extra")];

        let result = inject_morphosyntax(
            utt,
            too_many_mors,
            talkbank_model::Terminator::Period {
                location: Location::Unlocated,
            },
            Vec::new(),
        );

        assert!(
            result.is_err(),
            "expected Err on count mismatch; got Ok (silent-skip regression)"
        );
        let diag = result.unwrap_err();
        // Typed diagnostic should carry the expected/actual counts and
        // the CHAT words, not a stringly message.
        assert_eq!(
            diag.expected.get(),
            1,
            "CHAT has 1 alignable word (hello); got {}",
            diag.expected
        );
        assert_eq!(
            diag.actual.get(),
            2,
            "test passed 2 Mor items; got {}",
            diag.actual
        );
        assert_eq!(
            diag.chat_words,
            vec!["hello".to_string()],
            "chat_words must reflect what CHAT extraction actually found"
        );
    }

    /// Mor/Gra count mismatch (mors_total_chunks + 1 != gra_relations.len())
    /// must surface as an Err, silent emission of a misaligned pair was the
    /// dona@s bug class.
    ///
    /// Setup: HELLO_CHAT's first utterance is `hello .` (1 alignable word).
    /// Supply 1 Mor (matching word count, so the upstream check passes) and
    /// gra_relations of length 1, short by one (missing the terminator's
    /// PUNCT entry that AppendTrailingPunct should have produced upstream).
    /// MorTier with terminator counts 2 chunks; GraTier has 1 relation;
    /// inject must reject.
    #[test]
    fn inject_morphosyntax_gra_count_mismatch_returns_err() {
        use talkbank_model::model::dependent_tier::mor::{MorStem, MorWord, PosCategory};

        let mut chat = parse_chat(HELLO_CHAT);
        let utt = get_utterance(&mut chat, 0);

        let mor = |pos: &str, lemma: &str| {
            Mor::new(MorWord::new(PosCategory::new(pos), MorStem::new(lemma)))
        };
        let mors = vec![mor("intj", "hello")];
        // Upstream produced gras for items only, forgot to append the
        // terminator's PUNCT relation. Mor count_chunks() == 2 (1 item +
        // sidecar terminator), gras.len() == 1. Inject must reject.
        let item_only_gras = vec![GrammaticalRelation::new(
            1,
            0,
            talkbank_model::GrammaticalRelationType::new("ROOT").expect("valid fixture relation"),
        )];

        let result = inject_morphosyntax(
            utt,
            mors,
            talkbank_model::Terminator::Period {
                location: Location::Unlocated,
            },
            item_only_gras,
        );

        assert!(
            result.is_err(),
            "expected Err on mor/gra count mismatch; got Ok (silent-misalignment regression \
             - this is the dona@s bug class)"
        );
    }
}
