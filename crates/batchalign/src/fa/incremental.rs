//! Incremental forced alignment processing.
//!
//! Compares a "before" file (with existing timings) against an "after" file
//! (user-edited) and only re-aligns FA groups that still need worker or cache
//! work after stable `%wor` timing is copied forward from the old file.
//!
//! Like full-file FA, this module now depends on the transport-neutral FA
//! worker adapter instead of assembling a concrete worker payload inline. That
//! keeps the incremental path and full-file path on the same migration path as
//! the worker protocol evolves from V1 payloads to V2 prepared artifacts.

use crate::api::DurationMs;
use crate::cache::CacheBackend;
use crate::chat_ops::fa::{
    BulletRepairPolicy, FaGroup, WordTiming, apply_fa_results_with_projection_policy, cache_key,
    collect_existing_fa_word_timings, expand_bullets_for_edge_fillers, group_utterances,
    refresh_existing_alignment_for_utterance, strip_wor_from_monotonicity_stripped_utterances,
};
use crate::chat_ops::{CacheKey, ChatFile, Line, Utterance};
use crate::error::ServerError;
use crate::params::{AudioContext, FaParams};
use crate::pipeline::PipelineServices;
use crate::runner::util::{FileStage, ProgressSender, ProgressUpdate};
use crate::types::results::{FaOutput, FaResult};
use crate::types::traces::{FaEvidenceSourceTrace, FaGroupTrace, TimingTrace, ViolationTrace};
use batchalign_transform::diff::UtteranceDelta;
use batchalign_transform::diff::preserve::{TierKind, copy_dependent_tiers};
use batchalign_transform::parse::{is_dummy, is_no_align, parse_lenient};
use batchalign_transform::validate::{ValidityLevel, validate_output, validate_to_level};
use tracing::{info, warn};

use super::transport::{
    FaInferencePlan, FaWorkerTransport, UncheckedFaWorkerBatch, plan_fa_inference,
};
use super::{
    CACHE_TASK, RAW_EVIDENCE_CACHE_TASK, assemble_group_evidence, collect_evidence_sources,
    collect_final_timings, process_fa,
};
use crate::chat_ops::fa::Grouping;

/// Process a CHAT file through forced alignment incrementally.
///
/// Compares `before_text` (previous file with timings) against `after_text`
/// (user-edited version) and only re-aligns FA groups that contain changed
/// utterances. Unchanged groups preserve their existing timings.
///
/// Falls back to full processing if no "before" is available.
pub(crate) async fn process_fa_incremental(
    before_text: &str,
    after_text: &str,
    audio: &AudioContext<'_>,
    worker_lang: &crate::api::LanguageCode3,
    services: PipelineServices<'_>,
    fa_params: &FaParams,
    progress: Option<&ProgressSender>,
) -> Result<FaResult, ServerError> {
    use batchalign_transform::diff::{DiffSummary, diff_chat};

    let parser = crate::chat_parser();
    let (before_file, _) = parse_lenient(&parser, before_text);
    let (after_file, _) = parse_lenient(&parser, after_text);

    let deltas = diff_chat(&before_file, &after_file);
    let summary = DiffSummary::from_deltas(&deltas);

    info!(
        unchanged = summary.unchanged,
        words_changed = summary.words_changed,
        inserted = summary.inserted,
        deleted = summary.deleted,
        "Incremental FA diff"
    );

    // If there is no unchanged, speaker-only-changed, or timing-only region to
    // preserve from the previous file, the incremental path has nothing to
    // reuse and should fall back to the regular full-file align path.
    if summary.unchanged == 0 && summary.speaker_changed == 0 && summary.timing_only == 0 {
        return process_fa(
            after_text,
            audio,
            worker_lang,
            services,
            fa_params,
            progress,
        )
        .await;
    }

    // Group the "after" file's utterances
    let (mut chat_file, parse_errors) = parse_lenient(&parser, after_text);

    if is_dummy(&chat_file) || is_no_align(&chat_file) {
        return Ok(FaResult::pass_through(
            chat_file,
            fa_params.gap_healing,
            fa_params.engine.as_wire_name(),
            services.engine_version.as_ref(),
        ));
    }

    if let Err(errors) = validate_to_level(&chat_file, &parse_errors, ValidityLevel::MainTierValid)
    {
        let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
        return Err(ServerError::Validation(format!(
            "align pre-validation failed: {}",
            msgs.join("; ")
        )));
    }

    let reusable_after_indices =
        reuse_stable_wor_timing_from_before(&before_file, &mut chat_file, &deltas);
    let reusable_after_touched: Vec<crate::chat_ops::UtteranceIdx> = reusable_after_indices
        .iter()
        .map(|&idx| crate::chat_ops::UtteranceIdx::new(idx))
        .collect();

    expand_bullets_for_edge_fillers(&mut chat_file);

    // Resolved once, here, and used for BOTH grouping and the containment
    // checks on what the engine returns. Grouping used to take an
    // `Option<u64>` and invent its own behaviour when it was absent; there is
    // one recording and one answer.
    let recording = audio.recording().await?;
    let Grouping {
        groups,
        refusals: unplaceable_decisions,
        windows_clamped,
    } = group_utterances(&chat_file, fa_params.max_group_ms().0, &recording);
    if groups.is_empty() {
        // Every utterance reused by `reuse_stable_wor_timing_from_before` is
        // folded in here so `%wor` (when requested) is written after
        // monotonicity resolves, never by the refresh step itself
        // (2026-09-01 review, item 2).
        let finalized = crate::chat_ops::fa::projection_without_injection_with_touched(
            fa_params.projection_policy(),
            fa_params.wor_tier.should_write(),
            reusable_after_touched,
        )
        .then_finalize(
            &mut chat_file,
            BulletRepairPolicy::from(fa_params.bullet_repair),
        );
        if fa_params.bullet_repair {
            tracing::info!(stats = %finalized.repair_stats(), "bullet repair applied (incremental)");
        }
        strip_wor_from_monotonicity_stripped_utterances(&mut chat_file, finalized.monotonicity());
        let written = crate::chat_ops::fa::retain_decision_evidence(
            &mut chat_file,
            crate::chat_ops::fa::FaDecisions::without_injection(
                Vec::new(),
                unplaceable_decisions,
                finalized,
            ),
        );
        return Ok(FaResult::without_groups(
            chat_file,
            fa_params.gap_healing,
            fa_params.engine.as_wire_name(),
            services.engine_version.as_ref(),
        )?
        .with_written_decisions(written));
    }

    // Determine which groups still need re-alignment after stable `%wor`
    // regions from the "before" file were copied into the edited file.
    let group_needs_realign: Vec<bool> = groups
        .iter()
        .map(|group| {
            group
                .utterance_indices
                .iter()
                .any(|idx| !reusable_after_indices.contains(&idx.raw()))
        })
        .collect();
    let realign_count = group_needs_realign
        .iter()
        .filter(|needs_realign| **needs_realign)
        .count();
    let reused_group_count = group_needs_realign.len() - realign_count;

    info!(
        total_groups = groups.len(),
        realign_groups = realign_count,
        reused_groups = reused_group_count,
        // Same fact the full path reports, in the same place, so the two runs
        // stay comparable line for line.
        windows_clamped,
        "Incremental FA: selective group re-alignment with stable %wor reuse"
    );

    // Build cache keys and timing storage for all groups
    let word_texts: Vec<Vec<String>> = groups
        .iter()
        .map(|g| g.words.iter().map(|w| w.text.clone()).collect())
        .collect();

    let cache_keys: Vec<CacheKey> = groups
        .iter()
        .zip(word_texts.iter())
        .map(|(g, words)| {
            cache_key(
                words,
                audio.audio_identity,
                g.audio_start_ms(),
                g.audio_end_ms(),
                fa_params.gap_healing,
                fa_params.engine,
            )
        })
        .collect();

    let mut all_timings: Vec<Option<Vec<Option<WordTiming>>>> = vec![None; groups.len()];
    let mut evidence_sources: Vec<Option<FaEvidenceSourceTrace>> = vec![None; groups.len()];

    // Reused groups already have current main-tier word timing in `chat_file`.
    // Everything else still needs a cache lookup or worker call.
    let key_strings: Vec<String> = cache_keys.iter().map(|k| k.as_str().to_string()).collect();
    let cached = match fa_params.cache_policy {
        crate::params::CachePolicy::SkipCache => std::collections::HashMap::new(),
        crate::params::CachePolicy::UseCache | crate::params::CachePolicy::RequireCache => {
            match services
                .cache
                .get_batch(&key_strings, CACHE_TASK.as_str(), services.engine_version)
                .await
            {
                Ok(map) => map,
                Err(e) => {
                    warn!(error = %e, "FA cache batch lookup failed");
                    std::collections::HashMap::new()
                }
            }
        }
    };
    let cached_raw = match fa_params.cache_policy {
        crate::params::CachePolicy::SkipCache => std::collections::HashMap::new(),
        crate::params::CachePolicy::UseCache | crate::params::CachePolicy::RequireCache => {
            match services
                .cache
                .get_batch(
                    &key_strings,
                    RAW_EVIDENCE_CACHE_TASK.as_str(),
                    services.engine_version,
                )
                .await
            {
                Ok(map) => map,
                Err(error) => {
                    warn!(error = %error, "Raw FA evidence cache batch lookup failed");
                    std::collections::HashMap::new()
                }
            }
        }
    };

    // Populate reused groups and cache hits.
    let mut miss_indices: Vec<usize> = Vec::new();
    let mut fallback_events = Vec::new();
    for (i, key) in cache_keys.iter().enumerate() {
        if !group_needs_realign[i]
            && let Some(timings) = collect_preserved_group_timings(&chat_file, &groups[i])
        {
            all_timings[i] = Some(timings);
            evidence_sources[i] = Some(FaEvidenceSourceTrace::WorReuse);
            continue;
        }

        let resolution = super::FaCacheGroupAdmission::new(
            key,
            fa_params.engine,
            services.engine_version,
            i,
            &groups[i],
            &recording,
        )
        .resolve(cached_raw.get(key.as_str()), cached.get(key.as_str()));
        for refusal in resolution.refusals {
            warn!(
                error = %refusal.error,
                cache_layer = refusal.layer,
                group = i,
                "Cached FA evidence was refused"
            );
        }
        match resolution.admitted {
            super::AdmittedFaCacheGroup::RawEvidence(evidence) => {
                let evidence = *evidence;
                if let Some(event) = evidence.fallback_event {
                    fallback_events.push(event);
                }
                all_timings[i] = Some(evidence.timings);
                evidence_sources[i] = Some(FaEvidenceSourceTrace::RawEvidenceReplay);
            }
            super::AdmittedFaCacheGroup::DerivedTimings(timings) => {
                all_timings[i] = Some(timings.into_timings());
                evidence_sources[i] = Some(FaEvidenceSourceTrace::Cache);
            }
            super::AdmittedFaCacheGroup::Miss => miss_indices.push(i),
        }
    }

    let reused_or_cached_groups = groups.len() - miss_indices.len();
    if reused_or_cached_groups > 0 || !miss_indices.is_empty() {
        info!(
            reused_or_cached = reused_or_cached_groups,
            misses = miss_indices.len(),
            "FA incremental partition"
        );
    }

    if let Some(tx) = progress {
        let _ = tx.send(ProgressUpdate::new(
            FileStage::Aligning,
            Some(reused_or_cached_groups as i64),
            Some(groups.len() as i64),
        ));
    }

    let transport = FaWorkerTransport::production(services);

    // Send miss groups through the shared FA worker transport adapter.
    if let FaInferencePlan::Authorized(authorization) =
        plan_fa_inference(fa_params.cache_policy, &miss_indices)?
    {
        // Resolved before dispatch so every group's reply can be checked
        // against the audio it describes. Fails the file rather than running
        // unbounded: a pass that cannot state the recording's length cannot
        // tell a measurement from a moment that does not exist.
        let parsed_results = transport
            .infer_groups(
                UncheckedFaWorkerBatch {
                    word_texts: &word_texts,
                    groups: &groups,
                    cache_keys: &cache_keys,
                    authorization,
                    audio_path: audio.audio_path,
                    worker_lang: worker_lang.into(),
                    engine: fa_params.engine,
                    gap_healing: fa_params.gap_healing,
                    recording,
                }
                .admit()?,
            )
            .await?;

        for (parsed_idx, parsed_result) in parsed_results.into_iter().enumerate() {
            let projection = parsed_result.into_projection();
            let miss_idx = projection.group_index;
            evidence_sources[miss_idx] = Some(projection.source());
            let timings = projection.timings;
            let raw_evidence = projection.raw_evidence;
            let fallback_event = projection.fallback_event;
            if let Some(event) = fallback_event {
                fallback_events.push(event);
            }

            let ba_version = env!("CARGO_PKG_VERSION");
            if let Some(raw_evidence) = raw_evidence {
                match super::AdmittedCachedFaTimings::encode_from_raw(
                    timings.clone(),
                    &raw_evidence,
                ) {
                    Ok(cache_data) => {
                        if let Err(error) = services
                            .cache
                            .put_batch(
                                &[(cache_keys[miss_idx].as_str().to_string(), cache_data)],
                                CACHE_TASK.as_str(),
                                services.engine_version,
                                ba_version,
                            )
                            .await
                        {
                            warn!(error = %error, "Failed to cache derived FA evidence (non-fatal)");
                        }
                    }
                    Err(error) => {
                        warn!(error = %error, "Failed to encode derived FA evidence (non-fatal)");
                    }
                }
                match serde_json::to_value(raw_evidence) {
                    Ok(cache_data) => {
                        if let Err(error) = services
                            .cache
                            .put_batch(
                                &[(cache_keys[miss_idx].as_str().to_string(), cache_data)],
                                RAW_EVIDENCE_CACHE_TASK.as_str(),
                                services.engine_version,
                                ba_version,
                            )
                            .await
                        {
                            warn!(error = %error, "Failed to cache raw FA evidence (non-fatal)");
                        }
                    }
                    Err(error) => {
                        warn!(error = %error, "Failed to serialize raw FA evidence (non-fatal)");
                    }
                }
            }

            all_timings[miss_idx] = Some(timings);

            if let Some(tx) = progress {
                let done = reused_or_cached_groups + parsed_idx + 1;
                let _ = tx.send(ProgressUpdate::new(
                    FileStage::Aligning,
                    Some(done as i64),
                    Some(groups.len() as i64),
                ));
            }
        }
    }

    // Apply all results
    let final_timings = collect_final_timings(all_timings, "incremental forced alignment")?;
    let evidence_sources =
        collect_evidence_sources(evidence_sources, "incremental forced alignment")?;

    let pre_injection_timings: Vec<Vec<Option<TimingTrace>>> = final_timings
        .iter()
        .map(|group| {
            group
                .iter()
                .map(|t| t.as_ref().map(TimingTrace::from_word_timing))
                .collect()
        })
        .collect();

    // Injection, optional repair, then monotonicity enforcement, as ONE typed
    // transition. The sequence used
    // to be two statements plus a comment saying the second must follow the
    // first, and this path shipped without it: UTR anchor drift survived into
    // the output (APROCSA 2256_T4.cha, 2026-04-09). Consuming `FaApplied` is
    // now the only way to reach the injection records, so the comment is a
    // signature.
    let finalized = apply_fa_results_with_projection_policy(
        &mut chat_file,
        &groups,
        &final_timings,
        fa_params.projection_policy(),
        fa_params.wor_tier.should_write(),
    )
    // Utterances reused from the "before" file's `%wor` (2026-09-01 review,
    // item 2): their `%wor` (if requested) is written by this SAME phase,
    // after monotonicity resolves, not by the refresh step above.
    .also_touched(reusable_after_touched)
    .then_finalize(
        &mut chat_file,
        BulletRepairPolicy::from(fa_params.bullet_repair),
    );
    if fa_params.bullet_repair {
        tracing::info!(stats = %finalized.repair_stats(), "bullet repair applied (incremental)");
    }

    // The same owner the full path uses, so the ORDER lives in one place and a
    // new source cannot reach one path only. The strip and the two guards this
    // block used to carry (one on `review_level`, one on emptiness) disappeared
    // when CHAT-tier generation was removed from the reachable API.
    //
    // `rescue` is stated EMPTY rather than omitted: this path never runs
    // narrow-bullet rescue, and until 2026-08-15 that legitimate difference was
    // hidden behind a comment claiming the two lists were the same. Saying it
    // outright is what makes the next divergence visible.
    let written_decisions = crate::chat_ops::fa::retain_decision_evidence(
        &mut chat_file,
        crate::chat_ops::fa::FaDecisions {
            rescue: Vec::new(),
            unplaceable: unplaceable_decisions,
            finalized,
        },
    );
    let (decision_records, timing_effects) = written_decisions.into_evidence();
    let decision_traces = decision_records.into_iter().map(Into::into).collect();
    let timing_decisions = timing_effects.into_iter().map(Into::into).collect();

    let output = FaOutput::processed(chat_file)?;
    let violations = if let Err(errors) = validate_output(output.as_chat_file(), "align") {
        let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
        warn!(errors = ?msgs, "align post-validation warnings (non-fatal)");
        errors
            .iter()
            .map(|e| ViolationTrace {
                code: format!("L{}", e.level as u8),
                message: e.message.clone(),
                utterance_index: None,
            })
            .collect()
    } else {
        Vec::new()
    };

    let group_traces: Vec<FaGroupTrace> = groups
        .iter()
        .map(|g| FaGroupTrace {
            audio_start_ms: DurationMs(g.audio_start_ms()),
            audio_end_ms: DurationMs(g.audio_end_ms()),
            utterance_indices: g.utterance_indices.iter().map(|idx| idx.raw()).collect(),
            words: g.words.iter().map(|w| w.text.clone()).collect(),
            word_ids: g.words.iter().map(|word| word.stable_id()).collect(),
        })
        .collect();

    let group_evidence = assemble_group_evidence(
        group_traces,
        evidence_sources,
        cache_keys
            .iter()
            .map(|key| key.as_str().to_owned())
            .collect(),
        pre_injection_timings,
    )?;

    Ok(FaResult {
        output,
        group_evidence,
        engine: fa_params.engine.as_wire_name().to_owned(),
        engine_version: services.engine_version.as_ref().to_owned(),
        decisions: decision_traces,
        timing_decisions,
        gap_healing: fa_params.gap_healing,
        violations,
        fallback_events,
    })
}

/// Copy reusable `%wor` timing from the "before" file into the edited file.
///
/// Only utterances whose words are unchanged are candidates. That includes
/// plain unchanged utterances, speaker-only changes, and timing-only edits
/// where a rerun should restore timing from the durable `%wor` layer instead of
/// trusting the edited utterance bullet. Each reused utterance receives the
/// `%wor` tier from the "before" file and is then refreshed back onto the main
/// tier so later grouping sees current utterance bullets and word timings.
/// Mechanical only: never writes a fresh `%wor` tier itself. Callers fold
/// the returned indices into whichever `FaApplied` write phase runs next
/// (`also_touched` for the fresh-groups path, `projection_without_injection_with_touched`
/// for the no-groups path), so `%wor` is always written after monotonicity
/// resolves, never here (2026-09-01 review, item 2).
fn reuse_stable_wor_timing_from_before(
    before_file: &ChatFile,
    after_file: &mut ChatFile,
    deltas: &[UtteranceDelta],
) -> std::collections::HashSet<usize> {
    let mut reused = std::collections::HashSet::new();

    for delta in deltas {
        let (before_idx, after_idx) = match delta {
            UtteranceDelta::Unchanged {
                before_idx,
                after_idx,
            }
            | UtteranceDelta::TimingOnly {
                before_idx,
                after_idx,
            }
            | UtteranceDelta::SpeakerChanged {
                before_idx,
                after_idx,
            } => (*before_idx, *after_idx),
            _ => continue,
        };

        copy_dependent_tiers(
            before_file,
            before_idx,
            after_file,
            after_idx,
            &[TierKind::Wor],
        );

        let Some(utterance) = get_utterance_mut(after_file, after_idx.raw()) else {
            continue;
        };
        if refresh_existing_alignment_for_utterance(utterance) {
            reused.insert(after_idx.raw());
        }
    }

    reused
}

/// Collect current timings for a preserved FA group from the CHAT AST.
///
/// The caller should use this only for groups whose utterances have already
/// been refreshed from stable `%wor` timing. The returned vector matches the
/// same word order used by FA extraction and injection.
pub(super) fn collect_preserved_group_timings(
    chat_file: &ChatFile,
    group: &FaGroup,
) -> Option<Vec<Option<WordTiming>>> {
    let mut timings = Vec::new();

    for utt_idx in &group.utterance_indices {
        let utterance = get_utterance(chat_file, utt_idx.raw())?;
        timings.extend(collect_existing_fa_word_timings(utterance));
    }

    if timings.len() != group.words.len() {
        return None;
    }

    Some(timings)
}

/// Borrow one utterance immutably by utterance ordinal.
pub(super) fn get_utterance(chat_file: &ChatFile, idx: usize) -> Option<&Utterance> {
    let mut current = 0usize;
    for line in &chat_file.lines {
        if let Line::Utterance(utterance) = line {
            if current == idx {
                return Some(utterance);
            }
            current += 1;
        }
    }
    None
}

/// Borrow one utterance mutably by utterance ordinal.
fn get_utterance_mut(chat_file: &mut ChatFile, idx: usize) -> Option<&mut Utterance> {
    let mut current = 0usize;
    for line in &mut chat_file.lines {
        if let Line::Utterance(utterance) = line {
            if current == idx {
                return Some(utterance);
            }
            current += 1;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_ops::fa::{FaWord, TimeSpan, WordEndPolicy, WordGapHealing, apply_fa_results};
    use crate::chat_ops::{UtteranceIdx, WordIdx};
    use batchalign_transform::diff::diff_chat;

    fn parse_chat(text: &str) -> ChatFile {
        let parser = batchalign_transform::parse::TreeSitterParser::new().unwrap();
        batchalign_transform::parse::parse_lenient(&parser, text).0
    }

    fn chat_with_wor(words0: &str, words1: &str) -> String {
        format!(
            "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\t{words0}\n%wor:\thello \u{15}100_500\u{15} world \u{15}600_1000\u{15} .\n*CHI:\t{words1}\n%wor:\tgoodbye \u{15}1500_2000\u{15} .\n@End\n"
        )
    }

    #[test]
    fn reuse_stable_wor_timing_from_before_only_marks_unchanged_utterances() {
        let before = parse_chat(&chat_with_wor("hello world .", "goodbye ."));
        let mut after = parse_chat(&chat_with_wor("hello world .", "farewell ."));
        let deltas = diff_chat(&before, &after);

        let reused = reuse_stable_wor_timing_from_before(&before, &mut after, &deltas);
        assert!(reused.contains(&0));
        assert!(!reused.contains(&1));

        let utt0 = get_utterance(&after, 0).expect("missing utterance 0");
        assert_eq!(collect_existing_fa_word_timings(utt0).len(), 2);
        assert!(utt0.main.content.bullet.is_some());
    }

    #[test]
    fn collect_preserved_group_timings_reads_refreshed_main_tier_timing() {
        let before = parse_chat(&chat_with_wor("hello world .", "goodbye ."));
        let mut after = parse_chat(&chat_with_wor("hello world .", "goodbye ."));
        let deltas = diff_chat(&before, &after);
        let reused = reuse_stable_wor_timing_from_before(&before, &mut after, &deltas);
        assert_eq!(reused.len(), 2);

        let groups = group_utterances(
            &after,
            20_000,
            &crate::chat_ops::fa::coordinates::Recording::of_duration(
                crate::chat_ops::fa::coordinates::Ms(4_000),
            )
            .expect("test recording is non-empty"),
        )
        .groups;
        let timings = collect_preserved_group_timings(&after, &groups[0])
            .expect("group timings should exist");
        assert_eq!(timings.len(), groups[0].words.len());
        assert!(timings.iter().all(|timing| timing.is_some()));
    }

    #[test]
    fn reuse_stable_wor_timing_from_before_marks_timing_only_utterances() {
        let mut before = parse_chat(&chat_with_wor("hello world .", "goodbye ."));
        crate::chat_ops::fa::refresh_existing_alignment(&mut before, true);
        let before_text = batchalign_transform::serialize::to_chat_string(&before);
        let before = parse_chat(&before_text);
        let mut after = parse_chat(&before_text);

        let utt0 = get_utterance_mut(&mut after, 0).expect("missing utterance 0");
        utt0.main.content.bullet = None;

        let deltas = diff_chat(&before, &after);
        assert!(matches!(deltas[0], UtteranceDelta::TimingOnly { .. }));

        let reused = reuse_stable_wor_timing_from_before(&before, &mut after, &deltas);
        assert!(
            reused.contains(&0),
            "timing-only utterance should be reused"
        );

        let utt0 = get_utterance(&after, 0).expect("missing utterance 0");
        assert!(utt0.main.content.bullet.is_some());
        assert!(
            collect_existing_fa_word_timings(utt0)
                .iter()
                .all(|timing| timing.is_some())
        );
    }

    // ---------------------------------------------------------------------------
    // What monotonicity enforcement DOES, which no type can hold
    // ---------------------------------------------------------------------------
    //
    // This was a regression test for the CALL BEING SKIPPED: the full FA path
    // ran `enforce_monotonicity` after `apply_fa_results` and the incremental
    // path omitted it, so backward timestamps survived. That scenario is no
    // longer writable. `apply_fa_results` returns `FaApplied`, whose records are
    // reachable by durable evidence only through `then_finalize`, so a path
    // that skips enforcement cannot obtain the records required by that sink.
    //
    // What survives here is the part a signature cannot express: that
    // enforcement strips a backward bullet and leaves the forward one alone.
    //
    // Incident (2026-04-09): 2256_T4.cha (APROCSA aphasia protocol) produced
    // •639095_640375• immediately after •731556_733418• because the global
    // Hirschberg UTR matched repeated scripted phrases to an earlier audio
    // window.  FA injected those backward timings and `enforce_monotonicity`
    // was never called to strip them.
    //
    // Fix: `process_fa_incremental` now calls `enforce_monotonicity` after
    // `apply_fa_results`, matching the full-path invariant.

    /// `enforce_monotonicity` strips a backward timestamp injected by
    /// `apply_fa_results` when FA receives out-of-order audio windows.
    ///
    /// Two consecutive INV utterances:
    ///   utt0 "alright"  → FA assigns 731556-733418 ms  (correct, forward)
    ///   utt1 "look"     → FA assigns 639095-639300 ms  (backward, earlier
    ///                      than utt0's end time of 733418 ms)
    ///
    /// After `apply_fa_results + enforce_monotonicity`, utt1's bullet must be
    /// `None` (the backward timestamp is stripped).  Without `enforce_monotonicity`
    /// the backward 639095 ms bullet persists and produces E362/E704 violations.
    ///
    /// This regression test verifies the fix added to `process_fa_incremental`.
    #[test]
    fn test_incremental_path_enforce_monotonicity_strips_backward_timestamp() {
        let chat_text = concat!(
            "@UTF8\n",
            "@Begin\n",
            "@Languages:\teng\n",
            "@Participants:\tINV Investigator Adult_Unrelated\n",
            "@ID:\teng|test|INV||female|||Adult_Unrelated|||\n",
            "@Media:\ttest, audio\n",
            "*INV:\talright .\n",
            "*INV:\tlook .\n",
            "@End\n",
        );
        let mut chat = parse_chat(chat_text);

        // Two single-word groups: one per utterance.
        // Group 0 is forward (731556 ms); group 1 is BACKWARD (639095 < 733418).
        let groups = vec![
            FaGroup {
                audio_span: TimeSpan::new(731000, 734000),
                words: vec![FaWord {
                    utterance_index: UtteranceIdx::new(0),
                    utterance_word_index: WordIdx::new(0),
                    text: "alright".into(),
                }],
                utterance_indices: vec![UtteranceIdx::new(0)],
            },
            FaGroup {
                audio_span: TimeSpan::new(639000, 641000),
                words: vec![FaWord {
                    utterance_index: UtteranceIdx::new(1),
                    utterance_word_index: WordIdx::new(0),
                    text: "look".into(),
                }],
                utterance_indices: vec![UtteranceIdx::new(1)],
            },
        ];

        // Group 0: forward timing (correct).
        // Group 1: backward timing, earlier than group 0's end time (639095 < 733418).
        let timings = vec![
            vec![crate::chat_ops::fa::WordTiming::fixture(731556, 733418)],
            vec![crate::chat_ops::fa::WordTiming::fixture(639095, 639300)],
        ];

        let applied = apply_fa_results(
            &mut chat,
            &groups,
            &timings,
            WordEndPolicy::measured(WordGapHealing::Heal),
            false,
        );
        // Through the same route production takes, rather than replicating the
        // sequence by hand.
        let _ = applied.then_finalize(&mut chat, crate::chat_ops::fa::BulletRepairPolicy::Disabled);

        let utt0 = get_utterance(&chat, 0).expect("utterance 0 must exist");
        let utt1 = get_utterance(&chat, 1).expect("utterance 1 must exist");

        // utt0 retains its forward bullet at 731556 ms.
        let b0 = utt0
            .main
            .content
            .bullet
            .as_ref()
            .expect("utt0 must retain its forward bullet");
        assert_eq!(
            b0.timing.start_ms, 731556,
            "utt0 start must be 731556 ms after enforcement; got {}",
            b0.timing.start_ms
        );

        // utt1's backward bullet (639095 < 733418) must be stripped.
        assert!(
            utt1.main.content.bullet.is_none(),
            "backward bullet at 639095ms (< utt0 end {}ms) must be stripped by \
             enforce_monotonicity; got {:?}",
            b0.timing.end_ms,
            utt1.main.content.bullet,
        );
    }
}
