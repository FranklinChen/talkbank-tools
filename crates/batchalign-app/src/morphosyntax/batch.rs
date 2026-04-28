//! Cross-file batch morphosyntax processing.

use std::collections::HashMap;

use crate::error::ServerError;
use crate::params::MorphosyntaxParams;
use crate::pipeline::PipelineServices;
use crate::text_batch::{TextBatchFileInput, TextBatchFileResult, TextBatchFileResults};
use batchalign_chat_ops::morphosyntax::l2;
use batchalign_chat_ops::morphosyntax::{
    BatchItemWithPosition, TokenizationMode, clear_morphosyntax, collect_payloads,
    declared_languages, inject_results, remove_empty_morphosyntax_placeholders,
    validate_mor_alignment,
};
use batchalign_chat_ops::nlp::UdResponse;
use batchalign_chat_ops::parse::{is_dummy, parse_lenient};
use batchalign_chat_ops::serialize::to_chat_string;
use batchalign_chat_ops::validate::{ValidityLevel, validate_output, validate_to_level};
use batchalign_chat_ops::{ChatFile, LanguageCode};
use tracing::warn;

use super::outcomes::{
    FileInjectionDecision, LanguageGroupFailure, LanguageGroupOutcome,
    aggregate_language_group_outcomes, classify_file_for_injection,
};
use super::worker::infer_batch;

// ---------------------------------------------------------------------------
// Cross-file batch morphosyntax processing
// ---------------------------------------------------------------------------

/// Process multiple CHAT files, pooling all payloads into a single
/// `batch_infer` call for maximum throughput.
///
/// Returns `(filename, Ok(output_text) | Err(error_msg))` for each file.
///
/// This function preserves per-file correctness boundaries while sharing one
/// model call: parse/collect per file, aggregate payloads globally, then
/// repartition responses back by file before injection and validation.
pub(crate) async fn run_morphosyntax_batch_impl(
    files: &[TextBatchFileInput],
    services: PipelineServices<'_>,
    dispatcher: &dyn super::LanguageGroupDispatcher,
    params: &MorphosyntaxParams<'_>,
    progress_tx: Option<tokio::sync::mpsc::Sender<crate::types::worker_v2::ProgressEventV2>>,
    group_timeout: std::time::Duration,
) -> TextBatchFileResults {
    let parser = crate::chat_parser();
    let primary_lang = LanguageCode::new(params.lang.as_ref());
    let mut results: TextBatchFileResults = Vec::with_capacity(files.len());

    // 1. Parse all files
    let parse_start = tokio::time::Instant::now();
    let mut parsed_files: Vec<ChatFile> = Vec::with_capacity(files.len());
    let mut dummy_flags: Vec<bool> = Vec::with_capacity(files.len());
    let mut validation_errors: Vec<Option<String>> = Vec::with_capacity(files.len());
    for file in files {
        let filename = file.filename.as_ref();
        let (mut chat_file, parse_errors) = parse_lenient(&parser, file.chat_text.as_ref());
        if !parse_errors.is_empty() {
            warn!(
                filename = %filename,
                num_errors = parse_errors.len(),
                "Parse errors (continuing with recovery)"
            );
        }
        let dummy = is_dummy(&chat_file);
        if !dummy {
            // Pre-validation gate (L2: MainTierValid)
            if let Err(errors) =
                validate_to_level(&chat_file, &parse_errors, ValidityLevel::MainTierValid)
            {
                let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
                validation_errors.push(Some(format!(
                    "morphotag pre-validation failed: {}",
                    msgs.join("; ")
                )));
                dummy_flags.push(true); // treat as skip
                parsed_files.push(chat_file);
                continue;
            }
            clear_morphosyntax(&mut chat_file);
        }
        validation_errors.push(None);
        dummy_flags.push(dummy);
        parsed_files.push(chat_file);
    }

    let num_files = files.len();
    let total_utterances: usize = parsed_files
        .iter()
        .map(|f| {
            f.lines
                .iter()
                .filter(|l| matches!(l, batchalign_chat_ops::Line::Utterance(_)))
                .count()
        })
        .sum();
    tracing::warn!(
        num_files,
        total_utterances,
        parse_ms = parse_start.elapsed().as_millis() as u64,
        "Pipeline timing: parse phase"
    );

    // 2. Collect payloads from each file, tracking provenance for the
    //    cross-file response redistribution below.
    let collect_start = tokio::time::Instant::now();
    struct PerFileBatch {
        item_count: usize,
        global_start: usize,
    }

    let mut all_items: Vec<BatchItemWithPosition> = Vec::new();
    let mut per_file_info: Vec<Option<PerFileBatch>> = Vec::with_capacity(files.len());

    for file_idx in 0..parsed_files.len() {
        // Skip dummy files entirely — they pass through unchanged
        if dummy_flags[file_idx] {
            per_file_info.push(None);
            continue;
        }

        let langs = declared_languages(&parsed_files[file_idx], &primary_lang);
        let collected = collect_payloads(
            &parsed_files[file_idx],
            &primary_lang,
            &langs,
            params.multilingual_policy,
        );
        let batch_items = collected.batch_items;
        // Note: collected.not_applicable is currently discarded here;
        // Wave 5 in the morphotag-reconciliation plan threads it into
        // per-file outcome reporting. For now, the payload-gate behavior
        // is unchanged — NotApplicable utterances simply aren't dispatched.

        // Debug: dump extracted payloads
        let filename = &files[file_idx].filename;
        services.debug_dumper.dump_morphosyntax_extracted(
            filename,
            &batch_items.iter().map(|(li, uo, item, words)| {
                serde_json::json!({
                    "line_idx": li,
                    "utt_ordinal": uo,
                    "item_words": &item.words,
                    "extracted_words": words.iter().map(|w| w.text.as_ref()).collect::<Vec<_>>(),
                    "word_count": words.len(),
                })
            }).collect::<Vec<_>>(),
        );

        if batch_items.is_empty() {
            per_file_info.push(None);
            continue;
        }

        // Warn when Cantonese input appears to be per-character without --retokenize.
        let retokenize = params.tokenization_mode == TokenizationMode::StanzaRetokenize;
        if !retokenize && params.lang.as_ref() == "yue" {
            let per_char_count = batch_items
                .iter()
                .flat_map(|(_, _, item, _)| item.words.iter())
                .filter(|w| w.chars().count() == 1 && w.chars().all(|c| c > '\u{2E80}'))
                .count();
            let total_words: usize = batch_items
                .iter()
                .map(|(_, _, item, _)| item.words.len())
                .sum();
            if total_words > 0 && per_char_count * 100 / total_words > 80 {
                warn!(
                    "Cantonese input appears to be per-character tokens \
                     ({per_char_count}/{total_words} single-CJK words). \
                     Consider --retokenize for word-level analysis."
                );
            }
        }

        let global_start = all_items.len();
        let item_count = batch_items.len();
        per_file_info.push(Some(PerFileBatch {
            item_count,
            global_start,
        }));
        all_items.extend(batch_items);
    }

    let collect_ms = collect_start.elapsed().as_millis() as u64;
    tracing::warn!(
        batch_items = all_items.len(),
        collect_ms,
        "Pipeline timing: collect phase"
    );

    // 3. Batch infer grouped by per-item language — all languages in parallel.
    //
    // Multilingual CHAT files (e.g., @Languages: fra, eng) produce batch
    // items with different per-item languages. Each language group must be
    // dispatched to a worker loaded with the correct Stanza model — sending
    // French text to an English MWT pipeline produces corrupt Range tokens.
    //
    // Language groups are dispatched **concurrently** since each language uses
    // a separate worker process. This is the primary throughput lever for
    // multilingual batches: a 5-language batch runs ~5x faster than serial.
    //
    // BA2 parity: BA2 used stanza.MultilingualPipeline for this. BA3 groups
    // by language and dispatches each group to a separate single-language
    // worker, which achieves the same correctness without MultilingualPipeline.
    // `language_group_failure` is `Some` when any language group bailed
    // out of dispatch (deadlock-prevention, timeout, worker crash).
    // Files whose utterance ranges intersect the failed global indices
    // must be marked as per-file errors downstream instead of injected
    // with empty UdResponses (2026-04-14 silent-corruption regression).
    let (all_ud_responses, language_group_failure): (
        Vec<UdResponse>,
        Option<LanguageGroupFailure>,
    ) = if all_items.is_empty() {
        (Vec::new(), None)
    } else {
        let retokenize = params.tokenization_mode == TokenizationMode::StanzaRetokenize;

        // Group items by their per-item language, preserving original indices.
        let mut by_lang: HashMap<LanguageCode, Vec<(usize, &BatchItemWithPosition)>> =
            HashMap::new();
        for (global_idx, item) in all_items.iter().enumerate() {
            let item_lang = &item.2.lang;
            by_lang
                .entry(item_lang.clone())
                .or_default()
                .push((global_idx, item));
        }

        // Prepare per-language dispatch inputs (owned data for the async tasks).
        struct LangDispatch {
            lang3: crate::api::LanguageCode3,
            items: Vec<BatchItemWithPosition>,
            /// Original global indices so results can be placed back.
            global_indices: Vec<usize>,
        }

        // Partition language groups into supported (dispatch to workers)
        // and unsupported (skip with warning).  This prevents spawning
        // workers for languages Stanza cannot process, which would either
        // crash the worker or deadlock the pool.
        let mut dispatches: Vec<LangDispatch> = Vec::new();
        let mut skipped_indices: Vec<(usize, String)> = Vec::new();

        for (lang, lang_items) in &by_lang {
            let lang3 = crate::api::LanguageCode3::try_new(lang.as_ref())
                .unwrap_or_else(|_| params.lang.clone());

            // Check language support via the capability registry (populated
            // from worker's resources.json), falling back to the hardcoded
            // table when the registry hasn't been populated yet.
            let lang_supported = if let Some(reg) = services.pool.stanza_registry() {
                reg.supports_morphosyntax(lang.as_ref())
            } else {
                batchalign_chat_ops::morphosyntax::stanza_languages::is_stanza_supported(lang)
            };
            if !lang_supported {
                tracing::warn!(
                    lang = %lang3,
                    items = lang_items.len(),
                    "Skipping unsupported language — utterances will have empty morphosyntax"
                );
                for (global_idx, _) in lang_items {
                    skipped_indices.push((*global_idx, lang3.to_string()));
                }
                continue;
            }

            let items: Vec<BatchItemWithPosition> =
                lang_items.iter().map(|(_, item)| (*item).clone()).collect();
            let global_indices: Vec<usize> = lang_items.iter().map(|(idx, _)| *idx).collect();
            dispatches.push(LangDispatch {
                lang3,
                items,
                global_indices,
            });
        }

        if !skipped_indices.is_empty() {
            tracing::info!(
                skipped = skipped_indices.len(),
                "Skipped utterances with unsupported languages"
            );
        }

        // Dispatch language groups with bounded concurrency.
        //
        // Each language group needs up to `max_workers_per_key` workers.
        // Unbounded `join_all` would try to spawn workers for all languages
        // simultaneously, exceeding `max_total_workers` and deadlocking.
        //
        // Instead, we use a semaphore to limit the number of concurrent
        // language groups to `max_total_workers / max_workers_per_key`.
        // When a group finishes and releases its workers, the next group
        // starts — no deadlock, full utilization, all groups eventually
        // process.  This is the same pattern FA pipeline uses for per-file
        // concurrency (JoinSet + Semaphore).
        let max_per_key = services.pool.max_workers_per_key().max(1);
        let max_total = services.pool.effective_max_total_workers().max(1);
        let max_concurrent_groups = (max_total / max_per_key).max(1);

        tracing::info!(
            language_groups = dispatches.len(),
            max_concurrent_groups,
            max_total_workers = max_total,
            max_workers_per_key = max_per_key,
            "Dispatching morphosyntax language groups with bounded concurrency"
        );

        let lang_sem = std::sync::Arc::new(tokio::sync::Semaphore::new(max_concurrent_groups));

        // Build futures that each acquire a semaphore permit before dispatching.
        let futures: Vec<_> = dispatches
            .iter()
            .map(|d| {
                let sem = lang_sem.clone();
                let ptx = progress_tx.clone();
                async move {
                    let _permit = sem.acquire().await.map_err(|_| {
                        ServerError::Validation("language group semaphore closed".into())
                    })?;
                    tracing::info!(
                        lang = %d.lang3,
                        items = d.items.len(),
                        available_permits = sem.available_permits(),
                        "Acquired semaphore for language group"
                    );
                    match tokio::time::timeout(
                        group_timeout,
                        dispatcher.dispatch(
                            &d.lang3,
                            &d.items,
                            params.mwt,
                            retokenize,
                            ptx.as_ref(),
                        ),
                    )
                    .await
                    {
                        Ok(result) => result,
                        Err(_) => {
                            tracing::error!(
                                lang = %d.lang3,
                                items = d.items.len(),
                                timeout_s = group_timeout.as_secs(),
                                "Language group timed out"
                            );
                            Err(ServerError::Validation(format!(
                                "language group {} timed out after {}s",
                                d.lang3,
                                group_timeout.as_secs()
                            )))
                        }
                    }
                }
            })
            .collect();

        let raw_outcomes = futures::future::join_all(futures).await;

        // Dump post-mortem batch payloads for any failure BEFORE we hand
        // the outcomes to the aggregator. The aggregator is pure and
        // does not know about the debug dumper; writing failure dumps
        // here preserves the existing post-mortem artifacts without
        // making the aggregator I/O-bound.
        for (dispatch, outcome) in dispatches.iter().zip(raw_outcomes.iter()) {
            if let Err(e) = outcome {
                tracing::warn!(
                    lang = %dispatch.lang3,
                    error = %e,
                    "Batch infer failed for language group"
                );
                let dump_items: Vec<_> = dispatch
                    .items
                    .iter()
                    .map(|(li, uo, item, _)| {
                        serde_json::json!({
                            "line_idx": li,
                            "utt_ordinal": uo,
                            "words": &item.words,
                            "lang": item.lang.as_ref(),
                        })
                    })
                    .collect();
                services.debug_dumper.dump_morphosyntax_failed_batch(
                    &format!("batch_failure_{}", dispatch.lang3),
                    &dump_items,
                    e,
                );
            }
        }

        // Turn raw outcomes into typed `LanguageGroupOutcome` values
        // and aggregate. The aggregator separates partial-success
        // responses (still useful for files whose languages all
        // succeeded) from a typed `LanguageGroupFailure` that the
        // caller must surface as per-file errors.
        let typed_outcomes: Vec<LanguageGroupOutcome> = dispatches
            .iter()
            .zip(raw_outcomes)
            .map(|(dispatch, result)| LanguageGroupOutcome {
                lang3: dispatch.lang3.as_ref().to_string(),
                global_indices: dispatch.global_indices.clone(),
                result,
            })
            .collect();
        let aggregate = aggregate_language_group_outcomes(typed_outcomes, all_items.len());

        if let Some(ref failure) = aggregate.failure {
            tracing::error!(
                failed_groups = failure.num_failed,
                languages = %failure.languages,
                "Language-group failure will be propagated as per-file errors \
                 instead of silently emitting empty %mor (2026-04-14 corruption \
                 regression guard)"
            );
        }

        services
            .debug_dumper
            .dump_morphosyntax_ud_responses("batch", &aggregate.responses);
        (aggregate.responses, aggregate.failure)
    };

    // 4. Distribute responses back to files and inject
    let inject_start = tokio::time::Instant::now();
    for (file_idx, file) in files.iter().enumerate() {
        let filename = file.filename.as_ref();
        // Skip files that failed pre-validation
        if let Some(ref err) = validation_errors[file_idx] {
            results.push(TextBatchFileResult::err(file.filename.clone(), err.clone()));
            continue;
        }

        let chat_file = &mut parsed_files[file_idx];

        if let Some(ref fm) = per_file_info[file_idx] {
            let global_start = fm.global_start;
            let count = fm.item_count;

            if let FileInjectionDecision::SkipFailed { failed_languages } =
                classify_file_for_injection(global_start, count, language_group_failure.as_ref())
            {
                results.push(TextBatchFileResult::err(
                    file.filename.clone(),
                    format!("morphosyntax language-group dispatch failed for {failed_languages}"),
                ));
                continue;
            }

            let file_responses: Vec<UdResponse> =
                all_ud_responses[global_start..global_start + count].to_vec();
            let file_items: Vec<BatchItemWithPosition> =
                all_items[global_start..global_start + count].to_vec();

            // Extract L2 deferred positions BEFORE inject_results consumes
            // the items/responses (the function reads but does not take ownership
            // of the slices; we pass the slices from the originals).
            let l2_deferred = if params.l2_morphotag {
                l2::extract_l2_deferred_positions(
                    &all_items[global_start..global_start + count],
                    &all_ud_responses[global_start..global_start + count],
                )
            } else {
                Vec::new()
            };

            let file_inject_start = tokio::time::Instant::now();
            match inject_results(
                &parser,
                chat_file,
                file_items,
                file_responses,
                &primary_lang,
                params.tokenization_mode,
                params.mwt,
            ) {
                Ok(_injection_result) => {
                    let inject_ms = file_inject_start.elapsed().as_millis() as u64;
                    if inject_ms > 1000 {
                        warn!(
                            filename = %filename,
                            inject_ms,
                            items = count,
                            "Slow injection (>1s)"
                        );
                    }

                    // ── Secondary L2 dispatch (experimental) ──
                    //
                    // After primary injection set L2|xxx on @s positions,
                    // dispatch those words to secondary language workers
                    // and splice real morphology back in.
                    if !l2_deferred.is_empty() {
                        dispatch_secondary_l2(chat_file, &l2_deferred, services, filename).await;
                    }

                    // ── Transcriber `$POS`-hint post-pass (default on) ──
                    //
                    // Walk the ChatFile and override `%mor` POS
                    // categories that disagree with transcriber
                    // `$POS` annotations. Lemma and features from
                    // Stanza are preserved. Opt out via
                    // `--no-pos-hints` at the CLI. See
                    // `batchalign-chat-ops::morphosyntax::pos_hints`
                    // and `talkbank-model::...::clan_to_ud_upos`.
                    if params.respect_pos_hints {
                        let hint_outcome =
                            batchalign_chat_ops::morphosyntax::apply_pos_hints(chat_file);
                        tracing::info!(
                            filename = %filename,
                            considered = hint_outcome.hints_considered,
                            agreed = hint_outcome.hints_agreed,
                            overridden = hint_outcome.hints_overridden,
                            unmapped = hint_outcome.hints_unmapped,
                            no_mor = hint_outcome.hints_skipped_no_mor,
                            "$POS hint post-pass applied"
                        );
                    }
                }
                Err(e) => {
                    results.push(TextBatchFileResult::err(
                        file.filename.clone(),
                        format!("Result injection failed: {e}"),
                    ));
                    continue;
                }
            }

            // Validate alignment
            let alignment_warnings = validate_mor_alignment(chat_file);
            for w in &alignment_warnings {
                warn!(filename = %filename, warning = %w, "Morphosyntax alignment mismatch");
            }
        }

        // Post-validation check (warn only — always serialize output so it can
        // be inspected for debugging).
        if !dummy_flags[file_idx]
            && let Err(errors) = validate_output(chat_file, "morphotag")
        {
            let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
            warn!(filename = %filename, errors = ?msgs, "morphotag post-validation warnings (non-fatal)");
        }

        // Sweep any unfilled %mor/%gra placeholders left by `clear_morphosyntax`
        // for utterances whose response produced no UD sentence. Keeping
        // empty stubs in `dependent_tiers` preserves tier order during
        // injection; removing them here prevents empty `%mor:` / `%gra:`
        // lines from leaking into the final output.
        remove_empty_morphosyntax_placeholders(chat_file);

        results.push(TextBatchFileResult::ok(
            file.filename.clone(),
            to_chat_string(chat_file),
        ));
    }

    tracing::warn!(
        num_files,
        inject_ms = inject_start.elapsed().as_millis() as u64,
        "Pipeline timing: injection + serialization phase"
    );

    results
}

// ---------------------------------------------------------------------------
// Experimental: secondary L2 dispatch for @s words
// ---------------------------------------------------------------------------

/// Dispatch @s words to secondary language workers and splice results back.
///
/// This function:
/// 1. Groups deferred positions into contiguous spans by target language
/// 2. For each supported target language, builds a minimal `BatchItemWithPosition`
///    and dispatches it to a secondary Stanza worker via `infer_batch`
/// 3. Runs the structural merge algorithm (primary structural + secondary lexical)
/// 4. Splices merged morphology back into the ChatFile, replacing L2|xxx
/// 5. Falls back to L2|xxx for unsupported languages or dispatch failures
pub(crate) async fn dispatch_secondary_l2(
    chat_file: &mut ChatFile,
    deferred: &[l2::L2DeferredPosition],
    services: PipelineServices<'_>,
    filename: &str,
) {
    use batchalign_chat_ops::morphosyntax::MorphosyntaxBatchItem;

    // Pre-extract word texts once per unique utterance.
    let word_cache = build_word_text_cache(chat_file, deferred);

    // Group into per-utterance contiguous spans. Each span becomes one
    // "sentence" for the secondary Stanza model, preserving within-span
    // context (e.g., "los niños" stays together, not sent as two isolated words).
    let dispatch_spans = l2::group_deferred_into_dispatch_spans(deferred, &word_cache);

    // Group spans by target language for batched dispatch.
    let mut by_lang: HashMap<LanguageCode, Vec<&l2::DispatchSpan>> = HashMap::new();
    for span in &dispatch_spans {
        by_lang
            .entry(span.target_lang.clone())
            .or_default()
            .push(span);
    }

    tracing::info!(
        filename = %filename,
        deferred = deferred.len(),
        spans = dispatch_spans.len(),
        languages = by_lang.len(),
        "L2 morphotag: dispatching @s words to secondary workers"
    );

    let mut merged_results: Vec<Option<l2::MergedL2Morphology>> = vec![None; deferred.len()];

    for (target_lang, lang_spans) in &by_lang {
        let lang3 = match crate::api::LanguageCode3::try_new(target_lang.as_ref()) {
            Ok(l) => l,
            Err(_) => {
                tracing::warn!(lang = %target_lang, "L2 morphotag: invalid language code");
                continue;
            }
        };

        let supported = if let Some(reg) = services.pool.stanza_registry() {
            reg.supports_morphosyntax(target_lang.as_ref())
        } else {
            batchalign_chat_ops::morphosyntax::stanza_languages::is_stanza_supported(target_lang)
        };

        if !supported {
            let total_words: usize = lang_spans.iter().map(|s| s.words.len()).sum();
            tracing::info!(lang = %lang3, words = total_words, "L2 morphotag: unsupported language");
            continue;
        }

        // Each span becomes one BatchItemWithPosition (one Stanza "sentence").
        let batch_items: Vec<BatchItemWithPosition> = lang_spans
            .iter()
            .map(|span| {
                let num_words = span.words.len();
                (
                    0, // line_idx placeholder
                    0, // utt_ordinal placeholder
                    MorphosyntaxBatchItem {
                        words: span.words.clone(),
                        terminator: ".".to_string(),
                        special_forms: vec![(None, None); num_words],
                        lang: target_lang.clone(),
                    },
                    Vec::new(), // no extracted words needed
                )
            })
            .collect();

        let empty_mwt: std::collections::BTreeMap<String, Vec<String>> =
            std::collections::BTreeMap::new();

        match infer_batch(services.pool, &batch_items, &lang3, &empty_mwt, true, None).await {
            Ok(responses) => {
                let mapping_ctx = batchalign_chat_ops::nlp::MappingContext {
                    lang: target_lang.clone(),
                };
                for (span, ud_resp) in lang_spans.iter().zip(responses.iter()) {
                    if let Some(sentence) = ud_resp.sentences.first() {
                        // Use map_ud_sentence to handle MWT Range tokens.
                        // This collapses contractions (it's → pron|it~aux|be)
                        // and produces 1 Mor per original word.
                        let mors =
                            match batchalign_chat_ops::nlp::map_ud_sentence(sentence, &mapping_ctx)
                            {
                                Ok((mors, _gras)) => mors,
                                Err(e) => {
                                    tracing::warn!(
                                        lang = %lang3,
                                        error = %e,
                                        "L2 morphotag: map_ud_sentence failed"
                                    );
                                    continue;
                                }
                            };

                        // Phrasal-verb recognition needs the full UD
                        // sentence, but only when the UD tokens line up
                        // 1:1 with the mapped Mors. MWT-collapsed cases
                        // (e.g., "it's" → pron|it~aux|be) do not line up,
                        // and phrasal verbs do not involve MWT, so we skip
                        // context for those spans.
                        let pass_context = sentence.words.len() == mors.len();
                        for (idx, global_idx) in span.global_indices.iter().enumerate() {
                            if let Some(mor) = mors.get(idx) {
                                let ctx = if pass_context {
                                    Some(l2::SecondaryUdContext {
                                        sentence,
                                        word_position: idx,
                                    })
                                } else {
                                    None
                                };
                                let merged = l2::merge_primary_secondary_with_context(
                                    &deferred[*global_idx].primary,
                                    mor.clone(),
                                    target_lang,
                                    ctx.as_ref(),
                                );
                                merged_results[*global_idx] = Some(merged);
                            }
                        }
                    }
                }
                let total_words: usize = lang_spans.iter().map(|s| s.words.len()).sum();
                tracing::info!(
                    lang = %lang3,
                    spans = lang_spans.len(),
                    words = total_words,
                    "L2 morphotag: secondary dispatch succeeded"
                );
            }
            Err(e) => {
                tracing::warn!(lang = %lang3, error = %e, "L2 morphotag: secondary dispatch failed");
            }
        }
    }

    let outcome = l2::splice_l2_into_chat(chat_file, deferred, &merged_results);
    tracing::info!(
        filename = %filename,
        spliced = outcome.spliced,
        fallback = outcome.fallback,
        gra_upgraded = outcome.gra_upgraded,
        "L2 morphotag: splice complete"
    );
}

/// Pre-extract word texts for all deferred positions, walking each utterance
/// at most once. Returns a map from `(line_idx, word_idx)` to word text.
fn build_word_text_cache(
    chat_file: &ChatFile,
    deferred: &[l2::L2DeferredPosition],
) -> HashMap<(usize, usize), String> {
    use batchalign_chat_ops::Line;
    use batchalign_chat_ops::extract;

    // Collect unique line indices to avoid re-walking the same utterance.
    let mut lines_needed: HashMap<usize, Vec<usize>> = HashMap::new();
    for def in deferred {
        lines_needed
            .entry(def.line_idx)
            .or_default()
            .push(def.word_idx);
    }

    let mut cache: HashMap<(usize, usize), String> = HashMap::new();
    for (line_idx, word_indices) in &lines_needed {
        let utt = match &chat_file.lines[*line_idx] {
            Line::Utterance(u) => u,
            _ => continue,
        };
        let mut words = Vec::new();
        extract::collect_utterance_content(
            &utt.main.content.content,
            batchalign_chat_ops::TierDomain::Mor,
            &mut words,
        );
        for &widx in word_indices {
            if let Some(w) = words.get(widx) {
                cache.insert((*line_idx, widx), w.text.as_str().to_string());
            }
        }
    }
    cache
}
