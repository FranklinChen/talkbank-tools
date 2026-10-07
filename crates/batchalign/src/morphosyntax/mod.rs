//! Server-side morphosyntax orchestrator.
//!
//! Owns the full CHAT lifecycle for morphotag jobs:
//! parse → clear → collect → infer → inject → serialize.
//!
//! Python workers receive only `(words, lang) → UdResponse` via the infer protocol
//! pure Stanza inference with zero CHAT awareness.
//!
//! # Call path
//!
//! `batchalign-cli`/API submission
//! → `runner::routing` (name-matched `morphotag` arm)
//! → `execution::dispatch_morphotag_job` (recipe-owned stack)
//! → `execution::worker_gateway`
//! → [`process_morphosyntax`] for single-file processing
//! → `crate::chat_ops::morphosyntax_ops::{collect_payloads, inject_results}`
//! → worker `execute_v2` (batched), via
//!   `infer_retry::dispatch_execute_v2_with_retry_and_progress`
//! → validation + serialization.
//!
//! This line said `batch_infer(task="morphosyntax")` until 2026-08-13, which
//! had stopped being true: the production path moved to `execute_v2` (see
//! `worker.rs`, "Send batch items to workers for NLP inference via batched
//! `execute_v2`"), and `batch_infer` survives only for the PyO3 bridge, the
//! worker tests and `doctor`. The stale line matters because it is what a
//! reader consults to decide which seam a change belongs at.
//!
//! # Invariants for contributors
//!
//! - `line_idx`/utterance positions from payload collection must still address
//!   utterances at injection time.
//! - `TokenizationMode::StanzaRetokenize` changes main-tier token boundaries
//!   during injection; post-injection alignment checks must still pass.
//! - Workers must stay CHAT-agnostic: only structured NLP payloads/responses
//!   cross the Rust/Python boundary.

mod batch;
mod capability;
pub(crate) mod identity;
mod worker;

use crate::chat_ops::LanguageCode;
use crate::chat_ops::morphosyntax_ops::{
    TokenizationMode, apply_pos_hint_evidence, collect_payloads, collect_pos_hints,
    declared_languages, validate_mor_alignment,
};
use crate::error::ServerError;
use crate::params::MorphosyntaxParams;
use crate::pipeline::PipelineServices;
use crate::pipeline::morphosyntax::{
    ParsedFile, run_admitted_morphosyntax, run_morphosyntax_pipeline,
};
use crate::pipeline::post_validate::PostValidated;
use batchalign_transform::morphosyntax::{InjectionError, MatchedMorphosyntaxResponses};
#[cfg(test)]
use batchalign_transform::parse::{is_dummy, parse_lenient};
use tracing::{info, warn};

pub(crate) use batch::dispatch_secondary_l2;
pub use capability::{AnalysisLanguageRequirement, AnalysisUnavailable, AnalysisUnavailableReason};
pub(crate) use worker::infer_batch;

// ---------------------------------------------------------------------------
// Per-file morphosyntax processing
// ---------------------------------------------------------------------------

/// Process a single CHAT file through the morphosyntax pipeline.
///
/// Returns the serialized CHAT text with %mor/%gra tiers injected.
///
/// Algorithm outline:
/// 1. Admit all retained CHAT, removing source-bound `%mor/%gra` only when
///    this file will be analyzed. Pass-through keeps and validates every tier.
/// 2. Consume admission into a prepared analysis.
/// 3. Collect per-utterance payloads with language/special-form metadata.
/// 4. Infer all utterances (no caching for text NLP, see
///    `batchalign3/CLAUDE.md` "Utterance Cache" for the rationale).
/// 5. Inject results.
/// 6. Validate light alignment checks.
/// 7. Run full post-validation and serialize.
pub(crate) async fn process_morphosyntax(
    chat_text: &str,
    services: PipelineServices<'_>,
    params: &MorphosyntaxParams<'_>,
) -> Result<PostValidated, ServerError> {
    run_morphosyntax_impl(chat_text, services, params).await
}

pub(crate) async fn run_morphosyntax_impl(
    chat_text: &str,
    services: PipelineServices<'_>,
    params: &MorphosyntaxParams<'_>,
) -> Result<PostValidated, ServerError> {
    run_morphosyntax_pipeline(chat_text, services, params).await
}

// ---------------------------------------------------------------------------
// Incremental morphosyntax processing
// ---------------------------------------------------------------------------

/// Process a CHAT file incrementally by diffing against a "before" version.
///
/// Compares `before_text` (previous file with existing %mor/%gra) against
/// `after_text` (user-edited version) and only reprocesses utterances whose
/// words changed or whose prior analysis is incomplete. Unchanged utterances
/// reuse a complete %mor/%gra pair from the admitted "before" version;
/// unchanged text alone is not proof that analysis can be reused.
///
/// Returns the serialized CHAT text with %mor/%gra tiers on all utterances.
///
/// # When to use
///
/// Use this when reprocessing a file the user has edited (e.g., fixing
/// words, splitting/merging utterances). The "before" is the file as it
/// was before editing (with %mor/%gra from a previous run), and "after"
/// is the edited version needing updated %mor/%gra.
///
/// Falls back to full processing if no "before" is available (first run).
pub(crate) async fn process_morphosyntax_incremental(
    before_text: &str,
    after_text: &str,
    services: PipelineServices<'_>,
    params: &MorphosyntaxParams<'_>,
) -> Result<PostValidated, ServerError> {
    use batchalign_transform::diff::{DiffSummary, diff_chat};

    let parser = crate::chat_parser();
    let parsed = ParsedFile::parse(after_text, params.policy.ca_policy)?;
    let after = match parsed {
        ParsedFile::PassThrough(_) => {
            return run_admitted_morphosyntax(parsed, services, params).await;
        }
        ParsedFile::Analyze(after) => after,
    };
    // Every reused tier originates in this fully admitted document. The
    // after-file's MOR/GRA were physically discarded by source-bound admission;
    // all of them will be replaced by admitted prior tiers or fresh inference.
    let before = crate::pipeline::text_infer::admit_retained_text(
        &parser,
        before_text,
        talkbank_model::model::TranscriptName::Anonymous,
    )?;
    let before_file = before.document();
    let deltas = diff_chat(before_file, after.document());
    let summary = DiffSummary::from_deltas(&deltas);
    if summary.unchanged == 0 && summary.speaker_changed == 0 && summary.timing_only == 0 {
        return run_admitted_morphosyntax(ParsedFile::Analyze(after), services, params).await;
    }
    let input = after.into_incremental();
    let mut after_file = input.chat;
    let primary_lang = input.language;
    let primary_api_language = input.api_language;
    // The retained input already passed complete admission. Decision tiers
    // produced by an earlier run are stripped only once analysis is selected.
    batchalign_transform::decisions::strip_decision_tiers(&mut after_file);

    info!(
        unchanged = summary.unchanged,
        words_changed = summary.words_changed,
        inserted = summary.inserted,
        deleted = summary.deleted,
        timing_only = summary.timing_only,
        speaker_changed = summary.speaker_changed,
        "Incremental morphosyntax diff"
    );

    // Only complete source-admitted pairs can satisfy reuse. Copying one tier,
    // or copying nothing from an unchanged utterance, leaves analysis owed.
    let preserved_count =
        preserve_complete_prior_morphology(&before, &mut after_file, &deltas, &primary_lang);
    info!(preserved_count, "Preserved %mor/%gra from before file");

    // Admission already removed all after-file morphology. Only complete pairs
    // were installed above. The canonical collector therefore selects exactly
    // the remaining work, including unchanged utterances with missing analysis.
    // No second text-diff filter may drop those obligations.
    let langs = declared_languages(&after_file, &primary_lang);
    let filtered_payloads = collect_payloads(
        &after_file,
        &primary_lang,
        &langs,
        params.multilingual_policy,
    )
    .batch_items;

    if filtered_payloads.is_empty() {
        return gate_incremental_output(after_file);
    }

    info!(
        total_utterances = summary.total(),
        reprocessing = filtered_payloads.len(),
        "Incremental morphosyntax: sending utterances requiring analysis to worker"
    );

    // Warn when Cantonese input appears to be per-character without --retokenize.
    let retokenize = params.tokenization_mode == TokenizationMode::StanzaRetokenize;
    if !retokenize && primary_api_language.as_ref() == "yue" {
        let per_char_count = filtered_payloads
            .iter()
            .flat_map(|collected| collected.item().words().iter().map(|word| word.text()))
            .filter(|w| w.chars().count() == 1 && w.chars().all(|c| c > '\u{2E80}'))
            .count();
        let total_words: usize = filtered_payloads
            .iter()
            .map(|collected| collected.item().words().len())
            .sum();
        if total_words > 0 && per_char_count * 100 / total_words > 80 {
            warn!(
                "Cantonese input appears to be per-character tokens ({per_char_count}/{total_words} single-CJK words). \
                 Consider --retokenize for word-level analysis."
            );
        }
    }

    // Infer for the outstanding payloads.
    //
    // The stamp is written only on this path, where a model actually ran. The
    // earlier no-work exit reanalyzed nothing, so it leaves whatever stamp the
    // document already carries (it names the models behind the tiers they
    // preserved) instead of replacing it with one that names no engine.
    let mut applied = identity::AppliedAnalyses::none();
    if !filtered_payloads.is_empty() {
        let pos_hint_evidence = params
            .policy
            .pos_hints
            .should_apply()
            .then(|| collect_pos_hints(&after_file));
        let misses = filtered_payloads;
        match infer_batch(
            services.pool,
            &misses,
            &primary_api_language,
            params.mwt,
            retokenize,
            params.progress,
            params.cancellation,
        )
        .await
        {
            Ok(admitted) => {
                // The models behind the reanalyzed utterances (and the L2
                // models below) are what this run's stamp names, and the
                // relations they repaired are what it counts.
                let (responses, reanalyzed) = identity::AppliedAnalyses::take_applied(admitted);
                applied.extend(reanalyzed);
                let injection = MatchedMorphosyntaxResponses::from_admitted(misses, responses)
                    .map_err(InjectionError::from)
                    .and_then(|batch| {
                        batch.inject(
                            &parser,
                            &mut after_file,
                            params.tokenization_mode,
                            params.mwt,
                        )
                    });
                match injection {
                    Ok(injection_result) => {
                        // Secondary L2 dispatch for @s words, at the
                        // positions read from the analysis injection mapped.
                        let (retokenization_traces, l2) = injection_result.into_parts();
                        let l2_deferred = if params.policy.l2.should_analyze() {
                            l2.into_reported_positions()
                        } else {
                            Vec::new()
                        };
                        if !l2_deferred.is_empty() {
                            applied.extend(
                                dispatch_secondary_l2(
                                    &mut after_file,
                                    l2_deferred,
                                    services,
                                    "incremental",
                                    params.cancellation,
                                )
                                .await?,
                            );
                        }
                        if let Some(evidence) = pos_hint_evidence {
                            let outcome = apply_pos_hint_evidence(
                                &mut after_file,
                                evidence,
                                &retokenization_traces,
                            );
                            tracing::debug!(?outcome, "Applied transcriber POS hints");
                        }
                    }
                    Err(e) => {
                        return Err(ServerError::MorphosyntaxInjection(e));
                    }
                }

                let alignment_warnings = validate_mor_alignment(&after_file);
                for w in &alignment_warnings {
                    warn!(warning = %w, "Morphosyntax alignment mismatch");
                }
            }
            Err(e) => {
                return Err(e);
            }
        }
    }

    // A run whose reanalysis ran no model (every changed utterance was
    // wordless) leaves the stamp the document
    // already carries. One that did names the models behind the tiers kept
    // from `before_file` together with the ones that ran now.
    if let Some(provenance) = crate::provenance::incremental_morphotag_provenance(
        before_file,
        &primary_api_language,
        &applied,
        retokenize,
    )? {
        crate::provenance::inject_provenance(&mut after_file, &provenance);
    }
    gate_incremental_output(after_file)
}

/// A complete analysis pair borrowed from a fully admitted prior transcript.
/// Private fields prevent an isolated MOR or GRA tier from satisfying reuse.
struct ReusablePriorMorphology<'source> {
    mor: &'source talkbank_model::model::MorTier,
    gra: &'source talkbank_model::model::GraTier,
    input: batchalign_transform::morphosyntax::MorphologyInput<'source>,
}

impl<'source> ReusablePriorMorphology<'source> {
    fn from_admitted(
        prior: &'source batchalign_transform::AdmittedSourceChat<'_>,
        primary_lang: &LanguageCode,
    ) -> Vec<Option<Self>> {
        use talkbank_model::model::{DependentTier, Line};
        let languages = declared_languages(prior.document(), primary_lang);
        prior
            .document()
            .lines
            .iter()
            .filter_map(|line| {
                let Line::Utterance(utterance) = line else {
                    return None;
                };
                let mor = utterance.dependent_tiers.iter().find_map(|entry| {
                    if let DependentTier::Mor(tier) = &entry.tier {
                        Some(tier)
                    } else {
                        None
                    }
                });
                let gra = utterance.dependent_tiers.iter().find_map(|entry| {
                    if let DependentTier::Gra(tier) = &entry.tier {
                        Some(tier)
                    } else {
                        None
                    }
                });
                Some(mor.zip(gra).and_then(|(mor, gra)| {
                    batchalign_transform::morphosyntax::MorphologyInput::from_utterance(
                        utterance,
                        primary_lang,
                        &languages,
                    )
                    .map(|input| Self { mor, gra, input })
                }))
            })
            .collect()
    }

    fn bind_to<'destination>(
        &self,
        destination: &'destination mut talkbank_model::model::Utterance,
        primary_lang: &LanguageCode,
        languages: &[LanguageCode],
    ) -> Option<CompatiblePriorMorphology<'source, 'destination>> {
        let input = batchalign_transform::morphosyntax::MorphologyInput::from_utterance(
            destination,
            primary_lang,
            languages,
        )?;
        if !self.input.equivalent_to(&input) {
            return None;
        }
        Some(CompatiblePriorMorphology {
            mor: self.mor,
            gra: self.gra,
            destination,
        })
    }
}

/// Transfer permission owns the matching destination borrow. It cannot be
/// reused after mutation or redirected to another utterance by the caller.
struct CompatiblePriorMorphology<'source, 'destination> {
    mor: &'source talkbank_model::model::MorTier,
    gra: &'source talkbank_model::model::GraTier,
    destination: &'destination mut talkbank_model::model::Utterance,
}

impl CompatiblePriorMorphology<'_, '_> {
    fn install(self) {
        use batchalign_transform::dependent_tiers::replace_or_add_tier;
        use talkbank_model::model::DependentTier;
        replace_or_add_tier(
            &mut self.destination.dependent_tiers,
            DependentTier::Mor(self.mor.clone()),
        );
        replace_or_add_tier(
            &mut self.destination.dependent_tiers,
            DependentTier::Gra(self.gra.clone()),
        );
    }
}

/// Diff eligibility selects a position, never proof of complete prior analysis.
/// Borrowed source/destination indexes are built once, not rescanned per tier.
fn preserve_complete_prior_morphology(
    prior: &batchalign_transform::AdmittedSourceChat<'_>,
    after: &mut crate::chat_ops::ChatFile,
    deltas: &[batchalign_transform::diff::UtteranceDelta],
    primary_lang: &LanguageCode,
) -> usize {
    use batchalign_transform::diff::UtteranceDelta;
    use talkbank_model::model::Line;
    let analyses = ReusablePriorMorphology::from_admitted(prior, primary_lang);
    let languages = declared_languages(after, primary_lang);
    let mut destinations: Vec<_> = after
        .lines
        .as_mut_slice()
        .iter_mut()
        .filter_map(|line| {
            if let Line::Utterance(utterance) = line {
                Some(utterance)
            } else {
                None
            }
        })
        .collect();
    let mut preserved = 0;
    for delta in deltas {
        let (before_idx, after_idx) = match delta {
            UtteranceDelta::Unchanged {
                before_idx,
                after_idx,
            }
            | UtteranceDelta::SpeakerChanged {
                before_idx,
                after_idx,
            }
            | UtteranceDelta::TimingOnly {
                before_idx,
                after_idx,
            } => (before_idx, after_idx),
            UtteranceDelta::WordsChanged { .. }
            | UtteranceDelta::Inserted { .. }
            | UtteranceDelta::Deleted { .. } => continue,
        };
        if let (Some(Some(analysis)), Some(destination)) = (
            analyses.get(before_idx.raw()),
            destinations.get_mut(after_idx.raw()),
        ) && let Some(compatible) = analysis.bind_to(destination, primary_lang, &languages)
        {
            compatible.install();
            preserved += 1;
        }
    }
    preserved
}

/// Run the post-validation gate for the incremental morphotag path.
///
/// One owner for the three exits of `process_morphosyntax_incremental`, so a
/// further exit cannot quietly skip complete checked-construction admission.
fn gate_incremental_output(
    after_file: crate::chat_ops::ChatFile,
) -> Result<PostValidated, ServerError> {
    // BY VALUE: both exits drop `after_file` immediately, and the
    // borrowing gate would clone the whole document straight back.
    PostValidated::gate_owned(after_file, crate::api::ReleasedCommand::Morphotag)
        .map_err(|failure| failure.into_server_error())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_ops::morphosyntax_ops::MultilingualPolicy;
    use batchalign_transform::parse::TreeSitterParser;

    const INCREMENTAL_MAIN: &str = "@UTF8\n@Begin\n@Languages:\teng\n\
@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI||female|||Target_Child|||\n\
*CHI:\tI eat cookies .\n@End\n";
    const INCREMENTAL_MOR: &str =
        "%mor:\tpron|I-Prs-Nom-S1 verb|eat-Fin-Ind-Pres-S1 noun|cookie-Plur .\n";
    const INCREMENTAL_GRA: &str = "%gra:\t1|2|NSUBJ 2|0|ROOT 3|2|OBJ 4|2|PUNCT\n";

    fn incremental_selection(before_text: &str, after_text: &str) -> (usize, Vec<usize>, String) {
        let parser = crate::chat_parser();
        let before = crate::pipeline::text_infer::admit_retained_text(
            &parser,
            before_text,
            talkbank_model::model::TranscriptName::Anonymous,
        )
        .expect("prior fixture must be fully valid CHAT");
        let ParsedFile::Analyze(after) =
            ParsedFile::parse(after_text, crate::options::CaMorphotagPolicy::Honor)
                .expect("working fixture must pass command admission")
        else {
            panic!("test requires actual analysis");
        };
        let deltas = batchalign_transform::diff::diff_chat(before.document(), after.document());
        let input = after.into_incremental();
        let mut chat = input.chat;
        let preserved =
            preserve_complete_prior_morphology(&before, &mut chat, &deltas, &input.language);
        let languages = declared_languages(&chat, &input.language);
        let outstanding = collect_payloads(
            &chat,
            &input.language,
            &languages,
            MultilingualPolicy::ProcessAll,
        )
        .batch_items
        .into_iter()
        .map(|item| item.utt_ordinal())
        .collect();
        (
            preserved,
            outstanding,
            batchalign_transform::serialize::to_chat_string(&chat),
        )
    }

    #[test]
    fn incremental_unchanged_missing_or_partial_analysis_remains_owed() {
        for tiers in [String::new(), INCREMENTAL_MOR.to_owned()] {
            let before = INCREMENTAL_MAIN.replace("@End", &(tiers + "@End"));
            let (preserved, pending, selected) = incremental_selection(&before, INCREMENTAL_MAIN);
            assert_eq!(preserved, 0);
            assert_eq!(pending, vec![0]);
            assert_eq!(
                selected, INCREMENTAL_MAIN,
                "partial prior must not suppress collection"
            );
        }
    }

    #[test]
    fn incremental_complete_pair_alone_discharge_reuse_obligation() {
        let before =
            INCREMENTAL_MAIN.replace("@End", &format!("{INCREMENTAL_MOR}{INCREMENTAL_GRA}@End"));
        let (preserved, pending, selected) = incremental_selection(&before, INCREMENTAL_MAIN);
        assert_eq!(preserved, 1);
        assert!(pending.is_empty());
        assert_eq!(selected, before);
    }

    #[test]
    fn incremental_mixed_reuse_keeps_missing_utterance_and_contributor_comment() {
        let before = INCREMENTAL_MAIN.replace("@End", &format!(
            "{INCREMENTAL_MOR}{INCREMENTAL_GRA}%com:\tContributor content.\n*CHI:\tYou eat cookies .\n@End",
        ));
        let after = INCREMENTAL_MAIN.replace(
            "@End",
            "%com:\tContributor content.\n*CHI:\tYou eat cookies .\n@End",
        );
        let (preserved, pending, selected) = incremental_selection(&before, &after);
        assert_eq!(preserved, 1);
        assert_eq!(pending, vec![1]);
        insta::assert_snapshot!(selected, @"
        @UTF8
        @Begin
        @Languages:\teng
        @Participants:\tCHI Target_Child
        @ID:\teng|test|CHI||female|||Target_Child|||
        *CHI:\tI eat cookies .
        %com:\tContributor content.
        %mor:\tpron|I-Prs-Nom-S1 verb|eat-Fin-Ind-Pres-S1 noun|cookie-Plur .
        %gra:\t1|2|NSUBJ 2|0|ROOT 3|2|OBJ 4|2|PUNCT
        *CHI:\tYou eat cookies .
        @End
        ");
    }

    #[test]
    fn incremental_changed_words_cannot_reuse_complete_prior_pair() {
        let before =
            INCREMENTAL_MAIN.replace("@End", &format!("{INCREMENTAL_MOR}{INCREMENTAL_GRA}@End"));
        let after = INCREMENTAL_MAIN.replace("I eat cookies", "You eat cookies");
        let (preserved, pending, selected) = incremental_selection(&before, &after);
        assert_eq!(preserved, 0);
        assert_eq!(pending, vec![0]);
        assert_eq!(selected, after);
    }

    #[test]
    fn incremental_same_cleaned_words_do_not_certify_morphology_context() {
        let main = INCREMENTAL_MAIN
            .replace("@Languages:\teng", "@Languages:\teng, spa")
            .replace("I eat cookies", "no here");
        let before = main.replace(
            "@End",
            "%mor:\tintj|no adv|here .\n%gra:\t1|0|ROOT 2|1|ADVMOD 3|1|PUNCT\n@End",
        );
        for (case, after) in [
            ("question", main.replace("no here .", "no here ?")),
            (
                "utterance language",
                main.replace("no here", "[- spa] no here"),
            ),
            ("inline language", main.replace("no here", "no@s:spa here")),
            (
                "span language",
                main.replace("no here", "<no> [@s:spa] here"),
            ),
            ("special form", main.replace("no here", "no@o here")),
            ("part-of-speech hint", main.replace("no here", "no$n here")),
            ("pause evidence", main.replace("no here", "no (.) here")),
            (
                "declared primary language",
                main.replace("@Languages:\teng, spa", "@Languages:\tspa, eng"),
            ),
        ] {
            let (preserved, pending, selected) = incremental_selection(&before, &after);
            assert_eq!(preserved, 0, "{case}");
            assert_eq!(pending, vec![0], "{case}");
            assert_eq!(selected, after, "{case}");
        }
    }

    #[test]
    fn incremental_selected_replacement_hints_require_fresh_analysis() {
        for (before_main, after_main) in [
            ("I ate [: eat] cookies", "I ate [: eat$n] cookies"),
            ("I ate [: eat$v] cookies", "I ate [: eat$n] cookies"),
            ("I eat$v [: eat] cookies", "I eat$v [: eat$n] cookies"),
        ] {
            let before = INCREMENTAL_MAIN
                .replace("I eat cookies", before_main)
                .replace("@End", &format!("{INCREMENTAL_MOR}{INCREMENTAL_GRA}@End"));
            let after = INCREMENTAL_MAIN.replace("I eat cookies", after_main);
            let (preserved, pending, selected) = incremental_selection(&before, &after);
            assert_eq!(preserved, 0, "{after_main}");
            assert_eq!(pending, vec![0], "{after_main}");
            assert_eq!(selected, after, "{after_main}");
        }
    }

    #[test]
    fn incremental_context_compatibility_preserves_speaker_and_timing_fast_paths() {
        let main = INCREMENTAL_MAIN
            .replace(
                "@Participants:\tCHI Target_Child",
                "@Participants:\tCHI Target_Child, MOT Mother",
            )
            .replace("*CHI:", "@ID:\teng|test|MOT||female|||Mother|||\n*CHI:");
        let before = main.replace("@End", &format!("{INCREMENTAL_MOR}{INCREMENTAL_GRA}@End"));
        for after in [
            main.replace("*CHI:", "*MOT:"),
            main.replace("cookies .", "cookies . \u{15}0_1000\u{15}")
                .replace("*CHI:", "@Media:\tsample, audio\n*CHI:"),
        ] {
            let (preserved, pending, selected) = incremental_selection(&before, &after);
            assert_eq!(preserved, 1);
            assert!(pending.is_empty());
            assert!(selected.contains(INCREMENTAL_MOR));
            assert!(selected.contains(INCREMENTAL_GRA));
        }
    }

    #[test]
    fn test_declared_languages_via_chat_ops() {
        let parser = TreeSitterParser::new().unwrap();
        let chat = include_str!("../../../../test-fixtures/eng_hello_male.cha");
        let (chat_file, _) = parse_lenient(&parser, chat);
        let primary = LanguageCode::new("eng").expect("valid test language code");
        let langs = declared_languages(&chat_file, &primary);
        assert!(!langs.is_empty());
    }

    #[test]
    fn test_collect_payloads_skip_non_primary_skips_non_primary() {
        let parser = TreeSitterParser::new().unwrap();
        // File with @Languages: eng, spa and a [- spa] code-switched utterance
        let chat = include_str!("../../../../test-fixtures/eng_spa_bilingual_code_switch.cha");
        let (chat_file, _) = parse_lenient(&parser, chat);
        let primary = LanguageCode::new("eng").expect("valid test language code");
        let langs = declared_languages(&chat_file, &primary);

        // With SkipNonPrimary, the Spanish utterance should be skipped
        let items_skip = collect_payloads(
            &chat_file,
            &primary,
            &langs,
            MultilingualPolicy::SkipNonPrimary,
        )
        .batch_items;
        // With ProcessAll, all utterances should be included
        let items_all =
            collect_payloads(&chat_file, &primary, &langs, MultilingualPolicy::ProcessAll)
                .batch_items;

        // SkipNonPrimary should produce fewer items than ProcessAll
        assert!(
            items_skip.len() < items_all.len(),
            "SkipNonPrimary should skip non-primary-language utterances: \
             got {} with SkipNonPrimary vs {} with ProcessAll",
            items_skip.len(),
            items_all.len()
        );
    }

    #[test]
    fn test_collect_payloads_process_all_includes_all() {
        let parser = TreeSitterParser::new().unwrap();
        let chat = include_str!("../../../../test-fixtures/eng_spa_bilingual_code_switch.cha");
        let (chat_file, _) = parse_lenient(&parser, chat);
        let primary = LanguageCode::new("eng").expect("valid test language code");
        let langs = declared_languages(&chat_file, &primary);

        let items = collect_payloads(&chat_file, &primary, &langs, MultilingualPolicy::ProcessAll)
            .batch_items;
        // Both utterances should be included
        assert_eq!(items.len(), 2, "ProcessAll should include all utterances");
    }

    #[test]
    fn legacy_dummy_detection_does_not_establish_input_admission() {
        let parser = TreeSitterParser::new().unwrap();
        let chat = include_str!("../../../../test-fixtures/eng_hello_world_dummy.cha");
        let (chat_file, _) = parse_lenient(&parser, chat);
        assert!(is_dummy(&chat_file), "@Options: dummy should be detected");

        // Legacy detection and payload collection do not certify valid input.
        let primary = LanguageCode::new("eng").expect("valid test language code");
        let langs = declared_languages(&chat_file, &primary);
        let items = collect_payloads(&chat_file, &primary, &langs, MultilingualPolicy::ProcessAll)
            .batch_items;
        assert!(!items.is_empty());
        assert!(ParsedFile::parse(chat, crate::options::CaMorphotagPolicy::Honor).is_err());
    }

    #[test]
    fn test_non_dummy_file_not_detected() {
        let parser = TreeSitterParser::new().unwrap();
        let chat = include_str!("../../../../test-fixtures/eng_hello_world.cha");
        let (chat_file, _) = parse_lenient(&parser, chat);
        assert!(!is_dummy(&chat_file));
    }
}
