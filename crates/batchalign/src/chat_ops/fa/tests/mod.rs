//! Tests for the forced-alignment module, partitioned by feature so
//! each child file fits the workspace's ≤800 LOC hard limit.
//!
//! Shared helpers (`parse_chat`, the Utterance accessors, fixtures
//! like `wor_timed_chat` / `proof_chat`, `make_fa_words`, etc.) live
//! here as `pub(super) fn` so each child file imports via
//! `use super::*;` without redundant copies.

#![allow(unused_imports, dead_code, ambiguous_glob_reexports)]

/// A recording of a known length, for tests that only need a bound.
///
/// One copy, because this module's docstring above promises exactly that and
/// three children had grown their own. Two of them carried an identical
/// five-line paragraph explaining why these tests pass a long recording: the
/// bound used to be `Option<u64>`, where `None` meant "audio length unknown"
/// and grouping silently SKIPPED untimed utterances, so their words were never
/// aligned. `Recording` deleted that state. These tests assert grouping SHAPE,
/// so they take a recording long enough not to bound anything they check.
pub(super) fn test_recording(ms: u64) -> crate::chat_ops::fa::coordinates::Recording {
    crate::chat_ops::fa::coordinates::Recording::of_duration(crate::chat_ops::fa::coordinates::Ms(
        ms,
    ))
    .expect("test recordings are non-empty")
}

// Re-export `fa::*` into `fa::tests::*` so child files can `use super::*;`
// to pull in both fa internals (the implementation under test) and the
// shared helpers defined below. Without this re-export, child modules
// would have to disambiguate against sibling fa modules
// (`fa/postprocess.rs`, `fa/grouping.rs`, etc.) whose names match our
// local children.
pub(super) use super::*;

use talkbank_model::UtteranceIdx;
use talkbank_model::model::{Line, UtteranceContent, WriteChat};
use talkbank_parser::TreeSitterParser;

mod bullet_rerun;
mod dropped_timing_evidence;
mod end_overlap_resolution;
mod find_reusable;
mod grouping_and_wor;
mod inject_and_parse;
mod kept_main_bullets;
mod kept_main_bullets_ordering;
mod postprocess_continuous;
mod replaced_word_and_compound;
mod timed_utterance_gate;
mod token_label_remap;
mod two_pass_and_strategy;
mod update_bullet;
mod utr_and_monotonicity;

pub(super) fn parse_chat(text: &str) -> talkbank_model::model::ChatFile {
    let parser = TreeSitterParser::new().unwrap();
    parser.parse_chat_file(text).expect_built()
}

pub(super) fn get_test_utterance(
    chat: &mut talkbank_model::model::ChatFile,
    idx: usize,
) -> &mut talkbank_model::model::Utterance {
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

pub(super) fn get_utterance(
    chat: &talkbank_model::model::ChatFile,
    idx: usize,
) -> &talkbank_model::model::Utterance {
    let mut utt_idx = 0;
    for line in &chat.lines {
        if let Line::Utterance(utt) = line {
            if utt_idx == idx {
                return utt;
            }
            utt_idx += 1;
        }
    }
    panic!("Utterance {idx} not found");
}

pub(super) fn wor_timed_chat() -> String {
    "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world .\n%wor:\thello \u{15}100_500\u{15} world \u{15}600_1000\u{15} .\n@End\n".to_string()
}

pub(super) fn proof_chat(main: &str) -> String {
    format!(
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\t{main}\n@End\n"
    )
}

pub(super) fn collect_proof_fa_words(main: &str) -> Vec<String> {
    let chat = parse_chat(&proof_chat(main));
    let utt = get_utterance(&chat, 0);
    let mut out = Vec::new();
    collect_fa_words(&utt.main.content.content, &mut out);
    out
}

pub(super) fn generate_proof_wor_words(main: &str) -> Vec<String> {
    let chat = parse_chat(&proof_chat(main));
    let utt = get_utterance(&chat, 0);
    utt.main
        .generate_wor_tier()
        .words()
        .map(|word| word.cleaned_text().to_string())
        .collect()
}

pub(super) fn words(items: &[&str]) -> Vec<String> {
    items.iter().map(|item| (*item).to_string()).collect()
}

pub(super) fn make_fa_words(texts: &[&str]) -> Vec<FaWord> {
    texts
        .iter()
        .enumerate()
        .map(|(i, t)| FaWord {
            utterance_index: UtteranceIdx::new(0),
            utterance_word_index: WordIdx::new(i),
            text: t.to_string(),
        })
        .collect()
}

pub(super) fn make_utr_tokens(words_with_times: &[(&str, u64, u64)]) -> Vec<utr::AsrTimingToken> {
    words_with_times
        .iter()
        .map(|(text, start, end)| utr::AsrTimingToken {
            text: text.to_string(),
            start_ms: *start,
            end_ms: *end,
        })
        .collect()
}

pub(super) fn get_utterance_bullet(
    chat: &talkbank_model::model::ChatFile,
    idx: usize,
) -> Option<(u64, u64)> {
    let mut utt_idx = 0;
    for line in &chat.lines {
        if let Line::Utterance(utt) = line {
            if utt_idx == idx {
                return utt
                    .main
                    .content
                    .bullet
                    .as_ref()
                    .map(|b| (b.timing.start_ms, b.timing.end_ms));
            }
            utt_idx += 1;
        }
    }
    None
}

pub(super) fn count_internal_bullets(chat: &talkbank_model::model::ChatFile) -> usize {
    let mut count = 0;
    for line in &chat.lines {
        if let Line::Utterance(utt) = line {
            for item in utt.main.content.content.as_slice() {
                if matches!(item, UtteranceContent::InternalBullet(_)) {
                    count += 1;
                }
            }
        }
    }
    count
}

pub(super) fn count_double_bullet_lines(chat_text: &str) -> usize {
    chat_text
        .lines()
        .filter(|line| line.starts_with('*'))
        .filter(|line| {
            let bullet_count = line.matches('\x15').count() / 2; // each bullet is a pair
            bullet_count > 1
        })
        .count()
}

/// The (start_ms, end_ms) of the `n`th `%wor` word in `utterance`'s `%wor`
/// tier, or `None` when that slot has no timing (or the tier/index doesn't
/// exist).
pub(super) fn wor_word_timing(
    chat: &talkbank_model::model::ChatFile,
    utt_idx: usize,
    word_idx: usize,
) -> Option<(u64, u64)> {
    wor_timings(chat, utt_idx).get(word_idx).copied().flatten()
}

/// Every `%wor` word timing of one utterance, in order (`None` = untimed);
/// empty when the utterance has no `%wor` tier.
pub(super) fn wor_timings(
    chat: &talkbank_model::model::ChatFile,
    utt_idx: usize,
) -> Vec<Option<(u64, u64)>> {
    use talkbank_model::model::dependent_tier::WorItem;
    let Some(wor) = get_utterance(chat, utt_idx).wor_tier() else {
        return Vec::new();
    };
    wor.items
        .iter()
        .filter_map(|item| match item {
            WorItem::Word(word) => Some(
                word.inline_bullet
                    .as_ref()
                    .map(|b| (b.timing.start_ms, b.timing.end_ms)),
            ),
            WorItem::Separator { .. } => None,
        })
        .collect()
}

/// Header for synthetic two-speaker transcripts.
pub(super) const TWO_SPEAKER_HEADER: &str = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child, MOT Mother\n@ID:\teng|x|CHI|||||Child|||\n@ID:\teng|x|MOT|||||Mother|||\n";

/// A synthetic two-speaker transcript with the given utterance lines.
pub(super) fn two_speaker_transcript(utterances: &str) -> String {
    format!("{TWO_SPEAKER_HEADER}{utterances}@End\n")
}

/// One FA word at `(utterance, word)`.
pub(super) fn fa_word(utterance: usize, word: usize, text: &str) -> FaWord {
    FaWord {
        utterance_index: UtteranceIdx::new(utterance),
        utterance_word_index: WordIdx::new(word),
        text: text.into(),
    }
}

/// Finalized records and effects, through the production retention owner.
pub(super) fn retained_evidence(
    chat: &mut talkbank_model::model::ChatFile,
    finalized: FaFinalized,
) -> (
    Vec<batchalign_transform::decisions::DecisionRecord>,
    Vec<MonotonicityEffect>,
) {
    retain_decision_evidence(
        chat,
        FaDecisions {
            rescue: Vec::new(),
            unplaceable: Vec::new(),
            finalized,
        },
    )
    .into_evidence()
}

/// A projection bound the way production binds one: `MainBulletAuthority::bind`
/// on the document as parsed, before anything edits it. Measured word ends
/// and no gap healing, so a `%wor` timing is what the fixture supplied unless
/// the policy cut it.
pub(super) fn bound_projection(
    input: &talkbank_model::model::ChatFile,
    end_overlaps: EndOverlapPolicy,
    main_bullets: MainBulletPolicy,
) -> FaProjection {
    FaProjection::new(
        FaProjectionPolicy::new(
            WordEndPolicy::measured(WordGapHealing::PreserveMeasured),
            ExistingWorBoundaryPolicy::Preserve,
            end_overlaps,
        ),
        MainBulletAuthority::bind(main_bullets, input).expect("fixture bullets are not backward"),
    )
}

/// Assert that EVERY bullet `input` carried is exactly the bullet `output`
/// has, utterance for utterance, and that no utterance was added or lost.
/// Crate-visible so the incremental path's own tests use the same check.
pub(crate) fn assert_given_bullets_held(
    input: &talkbank_model::model::ChatFile,
    output: &talkbank_model::model::ChatFile,
) {
    let bullets = |chat: &talkbank_model::model::ChatFile| -> Vec<Option<(u64, u64)>> {
        chat.utterances()
            .map(|utterance| {
                utterance
                    .main
                    .content
                    .bullet
                    .as_ref()
                    .map(|b| (b.timing.start_ms, b.timing.end_ms))
            })
            .collect()
    };
    let (given, written) = (bullets(input), bullets(output));
    assert_eq!(given.len(), written.len(), "utterance count changed");
    for (idx, (given, written)) in given.iter().zip(&written).enumerate() {
        if let Some(given) = given {
            assert_eq!(
                Some(*given),
                *written,
                "utterance {idx}: the given bullet did not hold"
            );
        }
    }
}

/// One fixture's outcome under one policy.
pub(super) struct PolicyRun<T> {
    pub chat: talkbank_model::model::ChatFile,
    pub outcome: T,
}

/// Run one fixture under `derive` and under `keep`, so a keep test can show
/// that its fixture really reaches the mechanism `keep` switches off. The
/// keep run is checked with [`assert_given_bullets_held`] here, so no keep
/// test can forget to check every given bullet.
pub(super) fn run_under_both_policies<T>(
    input: &str,
    end_overlaps: EndOverlapPolicy,
    run: impl Fn(&mut talkbank_model::model::ChatFile, FaProjection) -> T,
) -> (PolicyRun<T>, PolicyRun<T>) {
    let given = parse_chat(input);
    let under = |policy: MainBulletPolicy| {
        let mut chat = parse_chat(input);
        let projection = bound_projection(&chat, end_overlaps, policy);
        let outcome = run(&mut chat, projection);
        PolicyRun { chat, outcome }
    };
    let derived = under(MainBulletPolicy::DeriveFromWords);
    let kept = under(MainBulletPolicy::KeepGiven);
    assert_given_bullets_held(&given, &kept.chat);
    (derived, kept)
}

/// Apply one group of word timings and finalize.
pub(super) fn align_one_group(
    chat: &mut talkbank_model::model::ChatFile,
    projection: FaProjection,
    words: Vec<FaWord>,
    utterances: &[usize],
    timings: &[(u64, u64)],
    write_wor: bool,
    repair: BulletRepairPolicy,
) -> (
    Vec<batchalign_transform::decisions::DecisionRecord>,
    Vec<MonotonicityEffect>,
) {
    let groups = vec![FaGroup::test_fixture(
        TimeSpan::new(0, 20_000),
        words,
        utterances.iter().map(|&u| UtteranceIdx::new(u)).collect(),
    )];
    let responses = vec![
        timings
            .iter()
            .map(|&(start, end)| WordTiming::fixture(start, end))
            .collect::<Vec<_>>(),
    ];
    let finalized =
        apply_fa_results_with_projection_policy(chat, &groups, &responses, projection, write_wor)
            .expect("fixture utterances are all in the input")
            .then_finalize(chat, repair)
            .expect("every kept bullet holds");
    retained_evidence(chat, finalized)
}

/// The records carrying one strategy.
pub(super) fn records_with(
    records: &[batchalign_transform::decisions::DecisionRecord],
    strategy: batchalign_transform::decisions::DecisionStrategy,
) -> Vec<&batchalign_transform::decisions::DecisionRecord> {
    records.iter().filter(|r| r.strategy == strategy).collect()
}
