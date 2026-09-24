//! `enforce_monotonicity_with_policy`'s same-speaker end-overlap classification:
//! `EndOverlapResolution` and the three `MonotonicityEffect::EndClamped*`
//! variants it produces. The defect these tests pin: a same-speaker overlap
//! used to be resolved on the bullet alone, after `%wor` had been written,
//! which left `%wor` and main-tier word timings stale relative to it.

#![allow(unused_imports, dead_code)]

use super::*;

use crate::chat_ops::fa::orchestrate::{EndOverlapResolution, classify_end_overlap};
use talkbank_model::model::dependent_tier::WorItem;

/// Case 1: the previous utterance's word hull already ends before the next
/// utterance's start. Only the bullet's coverage overshot; the bullet end
/// moves to the hull, and the word is untouched.
#[test]
fn coverage_only_clamps_bullet_to_hull_leaves_words_untouched() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n@ID:\teng|x|CHI|||||Child|||\n*CHI:\thello . \u{15}1000_5000\u{15}\n%wor:\thello \u{15}1000_1500\u{15} .\n*CHI:\tnext . \u{15}2000_6000\u{15}\n@End\n";
    let mut chat = parse_chat(input);

    let result = enforce_monotonicity(&mut chat);

    assert_eq!(
        get_utterance_bullet(&chat, 0),
        Some((1000, 1500)),
        "bullet end should be pulled back to the word hull, not the next start"
    );
    assert_eq!(
        wor_word_timing(&chat, 0, 0),
        Some((1000, 1500)),
        "the word itself is untouched by a coverage-only clamp"
    );
    assert!(matches!(
        result.effects(),
        [MonotonicityEffect::EndClampedCoverageOnly {
            clamped_to_ms: 1500,
            ..
        }]
    ));
    assert!(!result.records()[0].needs_review, "bookkeeping only");
}

/// Case 1 without any words at all: falls back to the old bookkeeping clamp
/// (bullet end to next start) exactly as before this fix.
#[test]
fn coverage_only_with_no_words_clamps_to_next_start() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n@ID:\teng|x|CHI|||||Child|||\n*CHI:\thello . \u{15}1000_5000\u{15}\n*CHI:\tnext . \u{15}4000_6000\u{15}\n@End\n";
    let mut chat = parse_chat(input);

    let result = enforce_monotonicity(&mut chat);

    assert_eq!(get_utterance_bullet(&chat, 0), Some((1000, 4000)));
    assert!(matches!(
        result.effects(),
        [MonotonicityEffect::EndClampedCoverageOnly {
            clamped_to_ms: 4000,
            ..
        }]
    ));
}

/// Case 2: the previous utterance's hull ends after the next start, but the
/// two utterances' words do not interleave. Both bullets take their
/// measured hull edges; neither word is touched.
#[test]
fn boundary_from_words_replaces_both_bullet_edges_leaves_words_untouched() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n@ID:\teng|x|CHI|||||Child|||\n*CHI:\thello . \u{15}1000_5000\u{15}\n%wor:\thello \u{15}1000_3000\u{15} .\n*CHI:\tworld . \u{15}2000_6000\u{15}\n%wor:\tworld \u{15}3500_6000\u{15} .\n@End\n";
    let mut chat = parse_chat(input);

    let result = enforce_monotonicity(&mut chat);

    assert_eq!(
        get_utterance_bullet(&chat, 0),
        Some((1000, 3000)),
        "previous bullet end replaced by its measured hull end"
    );
    assert_eq!(
        get_utterance_bullet(&chat, 1),
        Some((3500, 6000)),
        "next bullet start replaced by its measured hull start"
    );
    assert_eq!(
        wor_word_timing(&chat, 0, 0),
        Some((1000, 3000)),
        "the previous word is untouched"
    );
    assert_eq!(
        wor_word_timing(&chat, 1, 0),
        Some((3500, 6000)),
        "the next word is untouched"
    );
    assert!(matches!(
        result.effects(),
        [MonotonicityEffect::EndClampedBoundaryFromWords {
            prev_hull_end_ms: 3000,
            next_hull_start_ms: 3500,
            ..
        }]
    ));
    assert!(
        !result.records()[0].needs_review,
        "a measurable boundary needs no human review"
    );
}

/// Case 3: the two utterances' words genuinely overlap in time. The bullet
/// is clamped to the next start and every previous-utterance word past that
/// bound is clamped with it (here, on `%wor`, since these words carry no
/// main-tier inline bullet), flagged for review.
#[test]
fn interleaved_words_clamps_bullet_and_every_word_past_the_bound() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n@ID:\teng|x|CHI|||||Child|||\n*CHI:\thello world . \u{15}1000_5000\u{15}\n%wor:\thello \u{15}1000_3000\u{15} world \u{15}3000_5000\u{15} .\n*CHI:\tnext . \u{15}4000_6000\u{15}\n%wor:\tnext \u{15}4000_6000\u{15} .\n@End\n";
    let mut chat = parse_chat(input);

    let result = enforce_monotonicity(&mut chat);

    assert_eq!(
        get_utterance_bullet(&chat, 0),
        Some((1000, 4000)),
        "bullet clamped to the next utterance's start, as before this fix"
    );
    assert_eq!(
        wor_word_timing(&chat, 0, 0),
        Some((1000, 3000)),
        "a word entirely before the bound is untouched"
    );
    assert_eq!(
        wor_word_timing(&chat, 0, 1),
        Some((3000, 4000)),
        "a word straddling the bound is clamped to it, not dropped"
    );
    let clamp = result
        .records()
        .iter()
        .find(|d| d.strategy.strategy_name() == "end_clamped_interleaved_words")
        .expect("overlap should produce an end clamp");
    assert!(clamp.needs_review, "a real word conflict needs review");
    let [
        MonotonicityEffect::EndClampedInterleavedWords {
            words_trimmed,
            words_dropped,
            ..
        },
    ] = result.effects()
    else {
        panic!(
            "expected exactly one EndClampedInterleavedWords effect: {:?}",
            result.effects()
        );
    };
    assert_eq!(*words_trimmed, 1);
    assert!(
        words_dropped.is_empty(),
        "no word is wholly past the bound here: {words_dropped:?}"
    );
}

/// Case 3, split-count regression (2026-09-02): a MEASURED timing dropped
/// wholly past the bound is a different fact than one merely trimmed to fit,
/// and the effect, decision reason and trace must all say so separately
/// rather than folding both into one `words_clamped` total. `third` starts
/// (4200) and ends (4800) entirely past the bound (4000), so nothing of it
/// survives the cut; `world` straddles the bound and keeps a shorter extent.
#[test]
fn interleaved_words_reports_trimmed_and_dropped_separately() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n@ID:\teng|x|CHI|||||Child|||\n*CHI:\thello world third . \u{15}1000_5000\u{15}\n%wor:\thello \u{15}1000_3000\u{15} world \u{15}3000_5000\u{15} third \u{15}4200_4800\u{15} .\n*CHI:\tnext . \u{15}4000_6000\u{15}\n%wor:\tnext \u{15}4000_6000\u{15} .\n@End\n";
    let mut chat = parse_chat(input);

    let result = enforce_monotonicity(&mut chat);

    assert_eq!(
        wor_word_timing(&chat, 0, 0),
        Some((1000, 3000)),
        "a word entirely before the bound is untouched"
    );
    assert_eq!(
        wor_word_timing(&chat, 0, 1),
        Some((3000, 4000)),
        "a word straddling the bound is trimmed, not dropped"
    );
    assert_eq!(
        wor_word_timing(&chat, 0, 2),
        None,
        "a word wholly past the bound has no positive extent left: dropped"
    );

    let decision = result
        .records()
        .iter()
        .find(|d| d.strategy.strategy_name() == "end_clamped_interleaved_words")
        .expect("overlap should produce an end clamp");
    assert!(
        decision.reason.contains("words_trimmed=1"),
        "reason must name the trimmed count: {}",
        decision.reason
    );
    assert!(
        decision.reason.contains("words_dropped=1"),
        "reason must name the dropped count, separately: {}",
        decision.reason
    );

    // The whole point of this test: the dropped side is not just a count.
    // A reviewer (or the dashboard) must be able to see WHICH word and WHAT
    // was lost, not only how many words were dropped.
    let [
        MonotonicityEffect::EndClampedInterleavedWords {
            words_trimmed,
            words_dropped,
            ..
        },
    ] = result.effects()
    else {
        panic!(
            "expected exactly one EndClampedInterleavedWords effect: {:?}",
            result.effects()
        );
    };
    assert_eq!(*words_trimmed, 1);
    assert_eq!(words_dropped.len(), 1);
    let dropped = &words_dropped[0];
    assert_eq!(
        dropped.word_index, 2,
        "third is the third %wor word, index 2 (0-based)"
    );
    assert_eq!(dropped.tier, WordTier::Wor);
    assert_eq!(
        dropped.measured,
        TimeSpan::new(4_200, 4_800),
        "the measured extent that was lost must survive to the effect"
    );
}

/// Case 3, the second trigger: the next utterance has no measured word at
/// all to fix a boundary against, so the overlap cannot be resolved from
/// measurement alone even though the previous utterance's own words don't
/// extend past a plausible boundary on their own.
#[test]
fn interleaved_words_when_next_utterance_has_no_measured_words() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n@ID:\teng|x|CHI|||||Child|||\n*CHI:\thello world . \u{15}1000_5000\u{15}\n%wor:\thello \u{15}1000_3000\u{15} world \u{15}3000_5000\u{15} .\n*CHI:\tnext . \u{15}4000_6000\u{15}\n@End\n";
    let mut chat = parse_chat(input);

    let result = enforce_monotonicity(&mut chat);

    assert_eq!(get_utterance_bullet(&chat, 0), Some((1000, 4000)));
    assert_eq!(wor_word_timing(&chat, 0, 1), Some((3000, 4000)));
    assert!(matches!(
        result.effects(),
        [MonotonicityEffect::EndClampedInterleavedWords { .. }]
    ));
}

/// A cross-speaker pair under `PreserveCrossSpeaker` is left entirely
/// alone: no classification runs, no bullet or word moves, no decision is
/// recorded.
#[test]
fn cross_speaker_pair_is_untouched_under_preserve_policy() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child, MOT Mother\n@ID:\teng|x|CHI|||||Child|||\n@ID:\teng|x|MOT|||||Mother|||\n*CHI:\thello world . \u{15}1000_5000\u{15}\n%wor:\thello \u{15}1000_3000\u{15} world \u{15}3000_5000\u{15} .\n*MOT:\tnext . \u{15}4000_6000\u{15}\n@End\n";
    let mut chat = parse_chat(input);

    let result =
        enforce_monotonicity_with_policy(&mut chat, EndOverlapPolicy::PreserveCrossSpeaker);

    assert_eq!(get_utterance_bullet(&chat, 0), Some((1000, 5000)));
    assert_eq!(wor_word_timing(&chat, 0, 1), Some((3000, 5000)));
    assert!(
        result.records().is_empty(),
        "no decision should be recorded for a preserved cross-speaker pair"
    );
    assert!(result.effects().is_empty());
}

/// The Preserve policy's inherited coverage, wider than the word hull on
/// both sides, survives untouched when the bullet does not actually overlap
/// its neighbor: the pass never enters the classification branch at all.
#[test]
fn non_overlapping_bullet_keeps_its_full_preserve_coverage() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n@ID:\teng|x|CHI|||||Child|||\n*CHI:\thello . \u{15}500_2000\u{15}\n%wor:\thello \u{15}1000_1500\u{15} .\n*CHI:\tnext . \u{15}2000_3000\u{15}\n@End\n";
    let mut chat = parse_chat(input);

    let result = enforce_monotonicity(&mut chat);

    assert_eq!(
        get_utterance_bullet(&chat, 0),
        Some((500, 2000)),
        "coverage on both sides of the hull survives when nothing overlaps"
    );
    assert_eq!(wor_word_timing(&chat, 0, 0), Some((1000, 1500)));
    assert!(result.records().is_empty());
    assert!(result.effects().is_empty());
}

/// Finding 1 (2026-09-01 review). Direct unit test of `classify_end_overlap`'s
/// guard against `BoundaryFromWords` pushing `next`'s start past its OWN
/// successor's start. Without the guard this returns `BoundaryFromWords`
/// unconditionally whenever the hulls agree, regardless of what follows.
#[test]
fn boundary_from_words_never_pushes_next_start_past_its_successor() {
    // The successor starts before where the hull would place `next`: the
    // move is unsafe, so this must fall back to `InterleavedWords`.
    assert!(matches!(
        classify_end_overlap(Some(2900), 2000, Some(2950), Some(2400)),
        EndOverlapResolution::InterleavedWords
    ));

    // Same hulls, but the successor starts safely after: the move is fine.
    assert!(matches!(
        classify_end_overlap(Some(2900), 2000, Some(2950), Some(3000)),
        EndOverlapResolution::BoundaryFromWords {
            prev_hull_end_ms: 2900,
            next_hull_start_ms: 2950,
        }
    ));

    // No successor at all (last timed utterance in the file): nothing to
    // violate, so the move is fine.
    assert!(matches!(
        classify_end_overlap(Some(2900), 2000, Some(2950), None),
        EndOverlapResolution::BoundaryFromWords {
            prev_hull_end_ms: 2900,
            next_hull_start_ms: 2950,
        }
    ));
}

/// Finding 1 (2026-09-01 review), end to end: the exact failing construction.
/// Three same-speaker utterances. U0's hull (ends 2900) and U1's hull (starts
/// 2950) agree on a `BoundaryFromWords` boundary in isolation, but U1's own
/// successor U2 starts at 2400, before 2950. Applying the move anyway (the
/// pre-fix behavior) leaves U1 starting after U2, which the very next pair's
/// zero-duration guard then "resolved" by STRIPPING U1 entirely, destroying
/// an utterance that had just been correctly resolved with `needs_review =
/// false`. The fix must produce no strip anywhere and lose no utterance.
#[test]
fn three_utterance_boundary_conflict_is_not_stripped() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n@ID:\teng|x|CHI|||||Child|||\n*CHI:\thello world . \u{15}1000_3000\u{15}\n%wor:\thello \u{15}1000_1500\u{15} world \u{15}1500_2900\u{15} .\n*CHI:\tnext . \u{15}2000_3500\u{15}\n%wor:\tnext \u{15}2950_3200\u{15} .\n*CHI:\tthird . \u{15}2400_5000\u{15}\n@End\n";
    let mut chat = parse_chat(input);

    let result = enforce_monotonicity(&mut chat);

    // No strip anywhere: every utterance keeps a bullet.
    assert!(get_utterance_bullet(&chat, 0).is_some(), "U0 must survive");
    assert!(get_utterance_bullet(&chat, 1).is_some(), "U1 must survive");
    assert!(get_utterance_bullet(&chat, 2).is_some(), "U2 must survive");

    // U1's start was never moved to 2950 (the guard refused it), so pair
    // (U1, U2) never sees a degenerate start >= next_start and never strips.
    assert_eq!(
        get_utterance_bullet(&chat, 1),
        Some((2000, 2400)),
        "U1 keeps its own original start; its end is clamped by pair (U1,U2)"
    );
    // U0 was clamped to U1's ORIGINAL start (2000), not the hull boundary
    // (2900/2950), since the guard forced the InterleavedWords fallback.
    assert_eq!(get_utterance_bullet(&chat, 0), Some((1000, 2000)));
    assert_eq!(
        wor_word_timing(&chat, 0, 1),
        Some((1500, 2000)),
        "the word straddling the fallback bound is clamped, not dropped"
    );
    // U1's own word (2950-3200) is entirely past its bullet's new end (2400)
    // once pair (U1,U2) clamps U1: no valid extent remains, so it is
    // dropped, same as any other word entirely past a clamp bound.
    assert_eq!(wor_word_timing(&chat, 1, 0), None);

    assert!(
        result
            .records()
            .iter()
            .all(|d| d.strategy.strategy_name() != "timing_stripped"),
        "no timing_stripped record anywhere: {:?}",
        result.records()
    );
    assert_eq!(result.effects().len(), 2);
    assert!(
        result
            .effects()
            .iter()
            .all(|e| matches!(e, MonotonicityEffect::EndClampedInterleavedWords { .. }))
    );
}

/// 2026-09-01 review, item 15. The real-data regression shape: a CHI
/// utterance (88) overlaps a LATER CHI utterance (90) by 1303 ms, with a
/// PAR utterance (89) sitting between them in file order. Under
/// `PreserveCrossSpeaker` (the default), the file-adjacent pass alone can
/// never form the (88, 90) pair at all -- it only ever pairs (88, 89) and
/// (89, 90), both cross-speaker and both skipped. The per-speaker-stream
/// pass must pair 88 directly with 90 (skipping 89), resolve their overlap,
/// and leave 89 completely untouched.
#[test]
fn same_speaker_pair_resolves_across_an_intervening_other_speaker() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child, PAR Parent\n\
         @ID:\teng|x|CHI|||||Child|||\n@ID:\teng|x|PAR|||||Parent|||\n\
         *CHI:\tearlier . \u{15}208000_210496\u{15}\n%wor:\tearlier \u{15}208000_210000\u{15} .\n\
         *PAR:\tinterjection . \u{15}209000_209500\u{15}\n\
         *CHI:\tnow I have to make it over . \u{15}209193_211557\u{15}\n%wor:\tnow \u{15}210100_210300\u{15} I have to make it over .\n\
         @End\n";
    let mut chat = parse_chat(input);

    // The default policy: this is the scenario item 15 is about.
    let result =
        enforce_monotonicity_with_policy(&mut chat, EndOverlapPolicy::PreserveCrossSpeaker);

    // Utterance 1 (PAR, index 1) is completely untouched: never part of any
    // same-speaker pair, and `PreserveCrossSpeaker` never resolves a
    // cross-speaker pair, adjacent or not.
    assert_eq!(
        get_utterance_bullet(&chat, 1),
        Some((209000, 209500)),
        "the intervening PAR utterance must be untouched"
    );

    // The CHI pair (0, 2) resolves via measured hulls: BoundaryFromWords.
    assert_eq!(
        get_utterance_bullet(&chat, 0),
        Some((208000, 210000)),
        "utterance 0's end moves to its own measured hull"
    );
    assert_eq!(
        get_utterance_bullet(&chat, 2),
        Some((210100, 211557)),
        "utterance 2's start moves to its own measured hull"
    );
    assert!(
        result
            .effects()
            .iter()
            .any(|e| matches!(e, MonotonicityEffect::EndClampedBoundaryFromWords { edge, .. } if edge.utterance_idx.raw() == 0 && edge.next_utterance_idx.raw() == 2)),
        "the resolved pair must be (0, 2), the same speaker's own stream, not any file-adjacent pair: {:?}",
        result.effects()
    );
}

/// 2026-09-01 review, item 15: the three-utterance `BoundaryFromWords`
/// successor guard (finding 1) must hold across an intervening
/// other-speaker utterance too: the guard's "successor" is the SAME
/// speaker's own next-next utterance, not the file-adjacent one.
#[test]
fn boundary_from_words_successor_guard_holds_across_an_intervening_speaker() {
    // CHI0/CHI1 hulls would agree on a boundary (2900/2950), but CHI1's
    // own successor in ITS speaker stream, CHI2 (with a PAR utterance
    // between CHI1 and CHI2 in file order), starts at 2400 -- before that
    // boundary. The move must be refused, exactly like the file-adjacent
    // case, falling back to `InterleavedWords` so nothing is stripped.
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child, PAR Parent\n\
         @ID:\teng|x|CHI|||||Child|||\n@ID:\teng|x|PAR|||||Parent|||\n\
         *CHI:\thello world . \u{15}1000_3000\u{15}\n%wor:\thello \u{15}1000_1500\u{15} world \u{15}1500_2900\u{15} .\n\
         *CHI:\tnext . \u{15}2000_3500\u{15}\n%wor:\tnext \u{15}2950_3200\u{15} .\n\
         *PAR:\taside . \u{15}2100_2200\u{15}\n\
         *CHI:\tthird . \u{15}2400_5000\u{15}\n\
         @End\n";
    let mut chat = parse_chat(input);

    let result =
        enforce_monotonicity_with_policy(&mut chat, EndOverlapPolicy::PreserveCrossSpeaker);

    assert!(
        get_utterance_bullet(&chat, 0).is_some(),
        "utterance 0 must survive, not be stripped"
    );
    assert!(
        get_utterance_bullet(&chat, 1).is_some(),
        "utterance 1 must survive, not be stripped"
    );
    assert_eq!(
        get_utterance_bullet(&chat, 1),
        Some((2000, 2400)),
        "utterance 1 keeps its own original start; its end is clamped against utterance 3 (its own next-next same-speaker successor)"
    );
    assert_eq!(get_utterance_bullet(&chat, 0), Some((1000, 2000)));
    assert!(
        result
            .records()
            .iter()
            .all(|d| d.strategy.strategy_name() != "timing_stripped"),
        "no strip anywhere: {:?}",
        result.records()
    );
}

/// 2026-09-01 review, item 15: `ClampAllAdjacent` still resolves a
/// genuinely cross-speaker FILE-ADJACENT overlap, unaffected by the new
/// per-speaker-stream pass (which never touches a cross-speaker pair at
/// all).
#[test]
fn clamp_all_adjacent_still_resolves_cross_speaker_file_adjacency() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child, MOT Mother\n\
         @ID:\teng|x|CHI|||||Child|||\n@ID:\teng|x|MOT|||||Mother|||\n\
         *CHI:\tone . \u{15}1000_5000\u{15}\n*MOT:\ttwo . \u{15}4000_8000\u{15}\n@End\n";
    let mut chat = parse_chat(input);

    let result = enforce_monotonicity_with_policy(&mut chat, EndOverlapPolicy::ClampAllAdjacent);

    assert_eq!(
        get_utterance_bullet(&chat, 0),
        Some((1000, 4000)),
        "cross-speaker adjacency is still clamped under ClampAllAdjacent"
    );
    assert!(matches!(
        result.effects(),
        [MonotonicityEffect::EndClampedCoverageOnly { .. }]
    ));
}

/// 2026-09-01 review, item 16: the real-data E362 regression. CHI 88 and
/// CHI 90 are consecutive in CHI's own speaker stream (and here also file
/// adjacent); CHI 90 has no later same-speaker utterance at all, so the
/// item-15 speaker-stream successor guard alone would see no successor and
/// wrongly permit `BoundaryFromWords` to move CHI 90's start up to its
/// hull start (2950). PAR 89, immediately after CHI 90 in FILE order, only
/// starts at 2900 -- before that would-be new start. Pass 1 already
/// enforced file-order start monotonicity across every speaker before Pass
/// 2 ran; Pass 2 must not be able to undo it. The guard must therefore
/// read the FILE-order successor (any speaker), not the speaker's own.
///
/// The final assertion (no bullet start precedes its file predecessor's
/// start) restates Pass 1's own rule against Pass 2's output. That is
/// legitimate here specifically because Pass 1 and Pass 2 are two steps of
/// ONE phase (`enforce_monotonicity_with_policy`) with a single documented
/// obligation -- Pass 2 must preserve what Pass 1 established -- not a
/// general invariant check bolted onto an unrelated function.
#[test]
fn boundary_from_words_guard_reads_file_order_successor_not_speaker_stream() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child, PAR Parent\n\
         @ID:\teng|x|CHI|||||Child|||\n@ID:\teng|x|PAR|||||Parent|||\n\
         *CHI:\tearlier . \u{15}1000_3000\u{15}\n%wor:\tearlier \u{15}1000_2950\u{15} .\n\
         *CHI:\tlater . \u{15}2000_4000\u{15}\n%wor:\tlater \u{15}2950_3800\u{15} .\n\
         *PAR:\tresponse . \u{15}2900_3500\u{15}\n\
         @End\n";
    let mut chat = parse_chat(input);

    let result =
        enforce_monotonicity_with_policy(&mut chat, EndOverlapPolicy::PreserveCrossSpeaker);

    // CHI 90 ("later") must NOT have been moved to its hull start (2950):
    // that would land past PAR 89's start (2900), undoing Pass 1.
    assert_eq!(
        get_utterance_bullet(&chat, 1).unwrap().0,
        2000,
        "CHI 90's start must be untouched, not moved to 2950"
    );
    assert!(
        matches!(
            result.effects(),
            [MonotonicityEffect::EndClampedInterleavedWords { .. }]
        ),
        "must fall back to InterleavedWords, not BoundaryFromWords: {:?}",
        result.effects()
    );

    // Pass 1's own rule, restated against Pass 2's output (see this test's
    // doc comment for why that is legitimate here): no bulleted
    // utterance's start may precede its immediate FILE predecessor's start.
    let mut previous_start_ms: Option<u64> = None;
    for idx in 0..3 {
        let Some((start_ms, _)) = get_utterance_bullet(&chat, idx) else {
            continue;
        };
        if let Some(previous) = previous_start_ms {
            assert!(
                start_ms >= previous,
                "utterance {idx} starts at {start_ms}, before its file predecessor's {previous}"
            );
        }
        previous_start_ms = Some(start_ms);
    }
}
