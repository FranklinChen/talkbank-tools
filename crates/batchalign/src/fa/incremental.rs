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

use crate::chat_ops::fa::{
    BulletRepairPolicy, FaGroup, WordTiming, apply_fa_results_with_projection_policy,
    collect_existing_fa_word_timings, expand_bullets_for_edge_fillers, group_utterances,
    refresh_existing_alignment_for_utterance, strip_wor_from_monotonicity_stripped_utterances,
};
use crate::chat_ops::{ChatFile, Line, Utterance};
use crate::error::ServerError;
use crate::params::{AudioContext, FaParams};
use crate::runner::util::ProgressSender;
use crate::types::results::{FaOutput, FaResult};
use batchalign_transform::diff::UtteranceDelta;
use batchalign_transform::diff::preserve::{TierKind, copy_dependent_tiers};
use batchalign_transform::parse::{is_dummy, is_no_align, parse_lenient};
// `ValidityLevel` and `validate_to_level` are no longer named here: the level
// FA admits at, and the level it gates output at, both live on `FaAdmission`.
use tracing::info;

use super::units::{FaDispatchInputs, resolve_group_timings};
use super::{AdmittedFaResult, FaAdmission};
use crate::chat_ops::fa::Grouping;

/// Process a CHAT file through forced alignment incrementally.
///
/// Compares `before_text` (previous file with timings) against `after`
/// (the user-edited version) and only re-aligns FA groups that contain changed
/// utterances. Unchanged groups preserve their existing timings.
///
/// Falls back to full processing if no "before" is available.
///
/// # It takes the MODEL, not a serialization of one
///
/// The caller holds the "after" document as a parsed [`FaInputDocument`]. It
/// used to serialize that model, hand over the string, and this function
/// parsed it back TWICE: once for the diff (throwing the parse errors away
/// into a `_`) and once more for a mutable target, with a third parse inside
/// the `process_fa` fallback. Four parses and a serialization per incremental
/// file, all to recover a document the caller already had.
///
/// Two consequences beyond the cost, and both make this path agree with the
/// full-file one rather than diverge from it. The document's own parse errors
/// now reach [`FaAdmission::admit`] instead of a re-parse's (a round trip
/// through the serializer used to launder them away), and a `@Options: dummy`
/// or `NoAlign` document passes through as the bytes it was READ as rather
/// than as a re-serialization of its model, which is what align promises and
/// what `run_fa_from_ast` already did.
pub(crate) async fn process_fa_incremental(
    before_text: &str,
    after: super::FaInputDocument<'_>,
    audio: &AudioContext<'_>,
    worker_lang: &crate::api::LanguageCode3,
    services: super::FaServices<'_>,
    fa_params: &FaParams,
    progress: Option<&ProgressSender>,
) -> Result<AdmittedFaResult, ServerError> {
    use batchalign_transform::diff::{DiffSummary, diff_chat};

    let parser = crate::chat_parser();
    let (before_file, _) = parse_lenient(&parser, before_text);

    let deltas = diff_chat(&before_file, &after.chat_file);
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
        // The AST entry point, not `process_fa`: the document is already
        // parsed, and `process_fa` exists only to parse a string into one.
        return super::run_fa_from_ast(after, audio, worker_lang, services, fa_params, progress)
            .await;
    }

    // The "after" document's own model is the mutation target. It used to be a
    // third parse of its own serialization.
    let super::FaInputDocument {
        mut chat_file,
        parse_errors,
        text: after_text,
        main_bullets,
        anchors,
    } = after;

    if is_dummy(&chat_file) || is_no_align(&chat_file) {
        // `after_text` is the bytes the document was READ as, so the
        // pass-through is literally unchanged. It used to be a
        // re-serialization of the model, because that is all the caller had
        // handed over.
        return Ok(FaAdmission::pass_through(
            chat_file,
            after_text,
            fa_params.gap_healing,
            fa_params.engine.as_wire_name(),
            services.cache_namespace,
        ));
    }

    // Same owner as the full path: the level is stated on `FaAdmission`, and
    // the proof it returns is what every `Ok` return below is gated against.
    let admission = FaAdmission::admit(&chat_file, &parse_errors)?;

    // The same pairing the full path makes. The "after" document is this
    // run's input, so under `--main-bullets keep` its bullets, bound at the
    // parse, are the ones that stay, whatever copying the "before" file's
    // `%wor` in and refreshing it below does to the working model.
    let projection =
        crate::chat_ops::fa::FaProjection::new(fa_params.projection_policy(), main_bullets);

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
    // The anchors describe the "after" document's words, which the reuse
    // above copied `%wor` onto without changing a word.
    let Grouping {
        groups,
        decisions: grouping_decisions,
        windows_clamped,
    } = group_utterances(&chat_file, fa_params.max_group_ms().0, &recording, anchors);
    if groups.is_empty() {
        // Every utterance reused by `reuse_stable_wor_timing_from_before` is
        // folded in here so `%wor` (when requested) is written after
        // monotonicity resolves, never by the refresh step itself
        // (2026-09-01 review, item 2).
        let finalized = crate::chat_ops::fa::projection_without_injection_with_touched(
            projection,
            fa_params.wor_tier.should_write(),
            reusable_after_touched,
        )
        .then_finalize(
            &mut chat_file,
            BulletRepairPolicy::from(fa_params.bullet_repair),
        )
        .map_err(super::kept_bullets_failed)?;
        if fa_params.bullet_repair {
            tracing::info!(stats = %finalized.repair_stats(), "bullet repair applied (incremental)");
        }
        strip_wor_from_monotonicity_stripped_utterances(&mut chat_file, finalized.monotonicity());
        let written = crate::chat_ops::fa::retain_decision_evidence(
            &mut chat_file,
            crate::chat_ops::fa::FaDecisions::without_injection(
                Vec::new(),
                grouping_decisions,
                finalized,
            ),
        );
        return admission.finish(
            FaResult::without_groups(
                chat_file,
                fa_params.gap_healing,
                fa_params.engine.as_wire_name(),
                services.cache_namespace,
            )?
            .with_written_decisions(written),
        );
    }

    // Groups whose every utterance was reused from the "before" file are
    // resolved from `%wor`; everything else goes to cache or worker. Counted
    // here only for the log line.
    let reused_group_count = groups
        .iter()
        .filter(|group| group.is_reusable_from(&reusable_after_indices))
        .count();

    info!(
        total_groups = groups.len(),
        realign_groups = groups.len() - reused_group_count,
        reused_groups = reused_group_count,
        anchored_utterances = anchors.anchored_utterances(),
        // Same fact the full path reports, in the same place, so the two runs
        // stay comparable line for line.
        windows_clamped,
        "Incremental FA: selective group re-alignment with stable %wor reuse"
    );

    // The same resolution the full path runs.
    let mut resolved = resolve_group_timings(FaDispatchInputs {
        groups: &groups,
        chat_file: &chat_file,
        reusable_utterances: &reusable_after_indices,
        audio,
        worker_lang,
        services,
        fa_params,
        progress,
        context: "incremental forced alignment",
    })
    .await?;
    let fallback_events = std::mem::take(&mut resolved.fallback_events);
    let (final_timings, group_evidence) = resolved.into_parts();

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
        projection,
        fa_params.wor_tier.should_write(),
    )
    .map_err(super::kept_bullets_failed)?
    // Utterances reused from the "before" file's `%wor` (2026-09-01 review,
    // item 2): their `%wor` (if requested) is written by this SAME phase,
    // after monotonicity resolves, not by the refresh step above.
    .also_touched(reusable_after_touched)
    .then_finalize(
        &mut chat_file,
        BulletRepairPolicy::from(fa_params.bullet_repair),
    )
    .map_err(super::kept_bullets_failed)?;
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
            grouping: grouping_decisions,
            finalized,
        },
    );
    let (decision_records, timing_effects) = written_decisions.into_evidence();
    let decision_traces = decision_records.into_iter().map(Into::into).collect();
    let timing_decisions = timing_effects.into_iter().map(Into::into).collect();

    // Post-validation runs in `FaAdmission::finish`, below, at the level this
    // file was ADMITTED at, together with every other `Ok` return.
    let output = FaOutput::processed(chat_file)?;

    admission.finish(FaResult {
        output,
        group_evidence,
        engine: fa_params.engine.as_wire_name().to_owned(),
        cache_namespace: services.cache_namespace.clone(),
        decisions: decision_traces,
        timing_decisions,
        gap_healing: fa_params.gap_healing,
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

    for utt_idx in group.utterance_indices() {
        let utterance = get_utterance(chat_file, utt_idx.raw())?;
        timings.extend(collect_existing_fa_word_timings(utterance));
    }

    if timings.len() != group.word_count() {
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

    /// The incremental path copies the "before" file's `%wor` into the edited
    /// file and refreshes the bullet from it. Under `--main-bullets keep` the
    /// EDITED file's bullets are the given ones: a person narrowed the first
    /// bullet, and the refreshed `%wor` hull must not undo that.
    #[test]
    fn incremental_reuse_keeps_the_edited_files_given_bullets() {
        use crate::chat_ops::fa::{
            BulletRepairPolicy, EndOverlapPolicy, ExistingWorBoundaryPolicy, FaProjection,
            FaProjectionPolicy, MainBulletAuthority, MainBulletPolicy,
            projection_without_injection_with_touched, retain_decision_evidence,
        };
        let before = parse_chat(&chat_with_wor(
            "hello world . \u{15}100_1000\u{15}",
            "goodbye . \u{15}1500_2000\u{15}",
        ));
        let edited = chat_with_wor(
            "hello world . \u{15}300_900\u{15}",
            "goodbye . \u{15}1500_2000\u{15}",
        );
        let given = parse_chat(&edited);
        let mut after = parse_chat(&edited);
        let projection = FaProjection::new(
            FaProjectionPolicy::new(
                WordEndPolicy::measured(WordGapHealing::PreserveMeasured),
                ExistingWorBoundaryPolicy::Preserve,
                EndOverlapPolicy::PreserveCrossSpeaker,
            ),
            MainBulletAuthority::bind(MainBulletPolicy::KeepGiven, &after)
                .expect("forward bullets"),
        );

        let deltas = diff_chat(&before, &after);
        let reused = reuse_stable_wor_timing_from_before(&before, &mut after, &deltas);
        assert!(reused.contains(&0), "a timing-only edit is reused");
        let finalized = projection_without_injection_with_touched(
            projection,
            true,
            reused.iter().map(|&idx| UtteranceIdx::new(idx)).collect(),
        )
        .then_finalize(&mut after, BulletRepairPolicy::Disabled)
        .expect("every kept bullet holds");
        let _ = retain_decision_evidence(
            &mut after,
            crate::chat_ops::fa::FaDecisions::without_injection(Vec::new(), Vec::new(), finalized),
        )
        .into_evidence();

        crate::chat_ops::fa::tests::assert_given_bullets_held(&given, &after);
        let wor = get_utterance(&after, 0)
            .and_then(|utt| utt.wor_tier().cloned())
            .expect("%wor rewritten");
        let timings: Vec<Option<(u64, u64)>> = wor
            .items
            .iter()
            .filter_map(|item| match item {
                talkbank_model::model::dependent_tier::WorItem::Word(word) => Some(
                    word.inline_bullet
                        .as_ref()
                        .map(|b| (b.timing.start_ms, b.timing.end_ms)),
                ),
                talkbank_model::model::dependent_tier::WorItem::Separator { .. } => None,
            })
            .collect();
        assert_eq!(timings, vec![Some((300, 500)), Some((600, 900))]);
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
            &crate::chat_ops::fa::AnchorIndex::not_observed(),
        )
        .groups;
        let timings = collect_preserved_group_timings(&after, &groups[0])
            .expect("group timings should exist");
        assert_eq!(timings.len(), groups[0].word_count());
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
            FaGroup::test_fixture(
                TimeSpan::new(731000, 734000),
                vec![FaWord {
                    utterance_index: UtteranceIdx::new(0),
                    utterance_word_index: WordIdx::new(0),
                    text: "alright".into(),
                }],
                vec![UtteranceIdx::new(0)],
            ),
            FaGroup::test_fixture(
                TimeSpan::new(639000, 641000),
                vec![FaWord {
                    utterance_index: UtteranceIdx::new(1),
                    utterance_word_index: WordIdx::new(0),
                    text: "look".into(),
                }],
                vec![UtteranceIdx::new(1)],
            ),
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
        let _finalized = applied
            .then_finalize(&mut chat, crate::chat_ops::fa::BulletRepairPolicy::Disabled)
            .expect("finalization under the default policy holds");

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
