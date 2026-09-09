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
mod worker;

use crate::chat_ops::LanguageCode;
use crate::chat_ops::morphosyntax_ops::{
    BatchItemWithPosition, TokenizationMode, apply_pos_hint_evidence, clear_morphosyntax_selective,
    collect_payloads, collect_pos_hints, declared_languages, inject_results,
    validate_mor_alignment,
};
use crate::error::ServerError;
use crate::params::MorphosyntaxParams;
use crate::pipeline::PipelineServices;
use crate::pipeline::morphosyntax::run_morphosyntax_pipeline;
use crate::pipeline::post_validate::PostValidated;
use batchalign_transform::parse::{is_ca, is_dummy, parse_lenient};
use batchalign_transform::validate::{ValidityLevel, validate_to_level};
use tracing::{info, warn};

pub(crate) use batch::dispatch_secondary_l2;
pub(crate) use worker::infer_batch;

/// Per-file decision produced from the submitted policy and parsed CHAT
/// header. Morphology mechanisms consume this decision rather than
/// reinterpreting `@Options` themselves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MorphotagDisposition {
    Analyze,
    PassThroughCa,
}

impl crate::options::CaMorphotagPolicy {
    pub(crate) fn disposition_for(
        self,
        chat_file: &crate::chat_ops::ChatFile,
    ) -> MorphotagDisposition {
        match (self, is_ca(chat_file)) {
            (Self::Honor, true) => MorphotagDisposition::PassThroughCa,
            (Self::Honor | Self::Analyze, false) | (Self::Analyze, true) => {
                MorphotagDisposition::Analyze
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-file morphosyntax processing
// ---------------------------------------------------------------------------

/// Process a single CHAT file through the morphosyntax pipeline.
///
/// Returns the serialized CHAT text with %mor/%gra tiers injected.
///
/// Algorithm outline:
/// 1. Parse and pre-validate to `MainTierValid`.
/// 2. Clear existing `%mor/%gra`.
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
/// words changed. Unchanged utterances preserve their existing %mor/%gra
/// tiers from the "before" version.
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
    use batchalign_transform::diff::preserve::TierKind;
    use batchalign_transform::diff::{
        DiffSummary, UtteranceDelta, copy_dependent_tiers, diff_chat,
    };

    // Parse the job-level language ONCE at this boundary (chatter 0.3.0 made
    // `LanguageCode` construction fallible); downstream code threads the
    // typed value. The error is stringified because chatter v0.3.0 does not
    // re-export `LanguageCodeError` (upstream defect, reported).
    let primary_lang = LanguageCode::new(params.lang.as_ref()).map_err(|e| {
        ServerError::Validation(format!(
            "morphotag: invalid language code {:?}: {e}",
            params.lang.as_ref()
        ))
    })?;
    let parser = crate::chat_parser();
    let (before_file, _) = parse_lenient(&parser, before_text);
    let (mut after_file, parse_errors) = parse_lenient(&parser, after_text);
    // This strip is the ANALYZE path's: it is the counterpart of the one
    // `pipeline::morphosyntax::states::Analysis::<Applied>::postcheck` runs on
    // the non-incremental path, and everything below (pre-validation, the
    // diff, the tier copies) sees the stripped document. It stays HERE, ahead
    // of the two declining exits, so that ordering is unchanged; the declining
    // exits no longer depend on it, because their constructor strips too.
    batchalign_transform::decisions::strip_decision_tiers(&mut after_file);

    if !parse_errors.is_empty() {
        warn!(
            num_errors = parse_errors.len(),
            "Parse errors in 'after' file (continuing with recovery)"
        );
    }

    // Neither of these analyses the document; both DO leave it with the
    // decision tiers an earlier run wrote removed, so the input's own bytes
    // are not the answer and `PostValidated::pass_through` (which carries them
    // verbatim) would be the wrong route.
    // `declined_stripping_decision_tiers` names exactly what was applied and
    // why it is not gated, and it PERFORMS the strip rather than trusting this
    // call site to have run it first.
    //
    // Both predicates read only `ChatFile::options` (`is_dummy` looks for the
    // `dummy` option flag; `disposition_for` delegates to `is_ca`, which looks
    // for the `CA` flag), and `strip_decision_tiers` touches only utterances'
    // dependent tiers. So the strip above cannot change either verdict, and
    // the redundant strip inside the constructor cannot either.
    if is_dummy(&after_file) {
        return Ok(PostValidated::declined_stripping_decision_tiers(
            after_file,
            crate::api::ReleasedCommand::Morphotag,
        ));
    }

    if matches!(
        params.policy.ca_policy.disposition_for(&after_file),
        MorphotagDisposition::PassThroughCa
    ) {
        return Ok(PostValidated::declined_stripping_decision_tiers(
            after_file,
            crate::api::ReleasedCommand::Morphotag,
        ));
    }

    // Pre-validation
    if let Err(errors) = validate_to_level(&after_file, &parse_errors, ValidityLevel::MainTierValid)
    {
        let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
        return Err(ServerError::Validation(format!(
            "morphotag pre-validation failed: {}",
            msgs.join("; ")
        )));
    }

    // Diff before vs after
    let deltas = diff_chat(&before_file, &after_file);
    let summary = DiffSummary::from_deltas(&deltas);

    info!(
        unchanged = summary.unchanged,
        words_changed = summary.words_changed,
        inserted = summary.inserted,
        deleted = summary.deleted,
        timing_only = summary.timing_only,
        speaker_changed = summary.speaker_changed,
        "Incremental morphosyntax diff"
    );

    // If everything changed, fall back to full processing
    if summary.unchanged == 0 && summary.speaker_changed == 0 && summary.timing_only == 0 {
        return process_morphosyntax(after_text, services, params).await;
    }

    // Step 1: Copy %mor/%gra from "before" for unchanged/speaker-only/timing-only utterances
    let tier_kinds = &[TierKind::Mor, TierKind::Gra];
    let mut preserved_count = 0usize;
    for delta in &deltas {
        match delta {
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
            } => {
                let n = copy_dependent_tiers(
                    &before_file,
                    *before_idx,
                    &mut after_file,
                    *after_idx,
                    tier_kinds,
                );
                if n > 0 {
                    preserved_count += 1;
                }
            }
            _ => {}
        }
    }
    info!(preserved_count, "Preserved %mor/%gra from before file");

    // Step 2: Clear %mor/%gra on utterances that need reprocessing
    // (WordsChanged and Inserted utterances)
    let needs_processing: Vec<usize> = deltas
        .iter()
        .filter(|d| d.needs_nlp_reprocessing())
        .filter_map(|d| d.after_idx())
        .map(|idx| idx.raw())
        .collect();

    if needs_processing.is_empty() {
        // Nothing to reprocess: every utterance was preserved from "before".
        // The tiers were still COPIED in, so this is real output and is
        // gated, not passed through.
        return gate_incremental_output(after_file);
    }

    // Build a set of utterance ordinals that need processing
    let needs_set: std::collections::HashSet<usize> = needs_processing.iter().copied().collect();

    // Clear %mor/%gra only on utterances that need reprocessing
    clear_morphosyntax_selective(&mut after_file, &needs_set);

    // Step 3: Collect payloads only for utterances that need reprocessing
    let langs = declared_languages(&after_file, &primary_lang);
    let all_payloads = collect_payloads(
        &after_file,
        &primary_lang,
        &langs,
        params.multilingual_policy,
    )
    .batch_items;

    // Filter to only the utterances that need reprocessing
    let filtered_payloads: Vec<BatchItemWithPosition> = all_payloads
        .into_iter()
        .filter(|(_, utt_ordinal, _, _)| needs_set.contains(utt_ordinal))
        .collect();

    if filtered_payloads.is_empty() {
        return gate_incremental_output(after_file);
    }

    info!(
        total_utterances = summary.total(),
        reprocessing = filtered_payloads.len(),
        "Incremental morphosyntax: sending only changed utterances to worker"
    );

    // Warn when Cantonese input appears to be per-character without --retokenize.
    let retokenize = params.tokenization_mode == TokenizationMode::StanzaRetokenize;
    if !retokenize && params.lang.as_ref() == "yue" {
        let per_char_count = filtered_payloads
            .iter()
            .flat_map(|(_, _, item, _)| item.words.iter())
            .filter(|w| w.chars().count() == 1 && w.chars().all(|c| c > '\u{2E80}'))
            .count();
        let total_words: usize = filtered_payloads
            .iter()
            .map(|(_, _, item, _)| item.words.len())
            .sum();
        if total_words > 0 && per_char_count * 100 / total_words > 80 {
            warn!(
                "Cantonese input appears to be per-character tokens ({per_char_count}/{total_words} single-CJK words). \
                 Consider --retokenize for word-level analysis."
            );
        }
    }

    // Step 4: Infer for the filtered payloads.
    //
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
            params.lang,
            params.mwt,
            retokenize,
            params.progress,
            params.cancellation,
        )
        .await
        {
            Ok(responses) => {
                // Extract L2 deferred positions before inject_results
                // takes ownership of misses/responses.
                let l2_deferred = if params.policy.l2.should_analyze() {
                    crate::chat_ops::morphosyntax_ops::l2::extract_l2_deferred_positions(
                        &misses, &responses,
                    )
                } else {
                    Vec::new()
                };

                match inject_results(
                    &parser,
                    &mut after_file,
                    misses,
                    responses,
                    &primary_lang,
                    params.tokenization_mode,
                    params.mwt,
                ) {
                    Ok(injection_result) => {
                        // Secondary L2 dispatch for @s words.
                        if !l2_deferred.is_empty() {
                            dispatch_secondary_l2(
                                &mut after_file,
                                &l2_deferred,
                                services,
                                "incremental",
                            )
                            .await;
                        }
                        if let Some(evidence) = pos_hint_evidence {
                            let outcome = apply_pos_hint_evidence(
                                &mut after_file,
                                evidence,
                                &injection_result.retokenization_traces,
                            );
                            tracing::debug!(?outcome, "Applied transcriber POS hints");
                        }
                    }
                    Err(e) => {
                        return Err(ServerError::Validation(format!(
                            "Result injection failed: {e}"
                        )));
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

    gate_incremental_output(after_file)
}

/// Run the post-validation gate for the incremental morphotag path.
///
/// One owner for the three exits of `process_morphosyntax_incremental`, so a
/// fourth exit cannot quietly skip the gate. `MainTierValid` is the level this
/// path's own pre-validation admitted the input at.
fn gate_incremental_output(
    after_file: crate::chat_ops::ChatFile,
) -> Result<PostValidated, ServerError> {
    // BY VALUE: all three exits drop `after_file` immediately, and the
    // borrowing gate would clone the whole document straight back.
    PostValidated::gate_owned(
        after_file,
        ValidityLevel::MainTierValid,
        crate::api::ReleasedCommand::Morphotag,
    )
    .map_err(|failure| ServerError::Validation(failure.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_ops::morphosyntax_ops::MultilingualPolicy;
    use batchalign_transform::parse::TreeSitterParser;

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
    fn test_dummy_file_skipped_by_is_dummy() {
        let parser = TreeSitterParser::new().unwrap();
        let chat = include_str!("../../../../test-fixtures/eng_hello_world_dummy.cha");
        let (chat_file, _) = parse_lenient(&parser, chat);
        assert!(is_dummy(&chat_file), "@Options: dummy should be detected");

        // Collect payloads should return items for non-dummy file
        let primary = LanguageCode::new("eng").expect("valid test language code");
        let langs = declared_languages(&chat_file, &primary);
        let items = collect_payloads(&chat_file, &primary, &langs, MultilingualPolicy::ProcessAll)
            .batch_items;
        // The file has an utterance, but is_dummy tells the orchestrator to skip it
        assert!(
            !items.is_empty(),
            "collect_payloads still collects from dummy files, the orchestrator is what gates on is_dummy"
        );
    }

    #[test]
    fn test_non_dummy_file_not_detected() {
        let parser = TreeSitterParser::new().unwrap();
        let chat = include_str!("../../../../test-fixtures/eng_hello_world.cha");
        let (chat_file, _) = parse_lenient(&parser, chat);
        assert!(!is_dummy(&chat_file));
    }
}
