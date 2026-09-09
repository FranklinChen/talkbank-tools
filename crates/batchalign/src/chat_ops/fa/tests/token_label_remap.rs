//! What happens when an onset-only engine's LABELS do not tokenize the way the
//! transcript does.
//!
//! # The behaviour these tests pin
//!
//! The deterministic stitch in `alignment::token_map` walks labels and words
//! together and only ever concatenates whole labels into a whole word. That is
//! exact when it works and it is brittle: one tokenizer disagreement (a split
//! whose pieces do not spell the word, a numeral the engine spelled out) used
//! to end the WHOLE group, so every later word lost its timing even though its
//! own onset was sitting unread in the response.
//!
//! The residue of a broken stitch now goes through the character-level
//! Hirschberg aligner the workspace already owns
//! (`batchalign_transform::dp_align`), which is why these tests exist: the
//! numbers still come from label onsets the engine measured, but WHICH labels
//! belong to which word is now decided by an edit-distance alignment rather
//! than by exact concatenation, and that difference is recorded per word.
//!
//! The control (`an_unbroken_stitch_never_reaches_the_dp`) is the important
//! one: the DP is a FALLBACK for the residue, so a group that stitches cleanly
//! must produce exactly what it produced before, and must say so in its own
//! outcome rather than leaving a reader to infer it.

use super::*;

use crate::chat_ops::fa::alignment::residue::{MAX_RESIDUE_ALIGN_CHARS, owners_are_monotone};
use crate::chat_ops::fa::alignment::token_map::{
    UntimedReason, WordSlot, WordTimingOutcome, map_labels_to_words, normalize_fa_alignment_unit,
};
use crate::chat_ops::fa::coordinates::{FaWindow, FileMs, Ms, Recording};
use crate::chat_ops::fa::origin::{CharEdits, EngineId, Origin, ProvenanceTally};
use crate::chat_ops::nlp::FaRawToken;

/// A window at the start of a recording far longer than any fixture here.
///
/// Long on purpose: `FaWindow::to_file` drops a label the engine placed past
/// the audio it was given, and none of these fixtures is about that rejection.
fn remap_window() -> (Recording, FaWindow) {
    let recording = Recording::of_duration(Ms(600_000)).expect("non-zero");
    let window = FaWindow::within(&recording, FileMs::new(0), FileMs::new(60_000))
        .expect("window inside the recording");
    (recording, window)
}

fn remap_engine() -> EngineId {
    EngineId::new("test-fa")
}

/// The labels an onset-only engine returns, as `(text, seconds)`.
fn labels(items: &[(&str, f64)]) -> Vec<FaRawToken> {
    items
        .iter()
        .map(|(text, time_s)| FaRawToken {
            text: (*text).to_string(),
            time_s: *time_s,
        })
        .collect()
}

/// The `(start_ms, end_ms)` of every word, so a whole group reads in one line.
fn spans(outcomes: &[WordTimingOutcome]) -> Vec<Option<(u64, u64)>> {
    outcomes
        .iter()
        .map(|outcome| outcome.timing().map(|t| (t.start_ms, t.end_ms)))
        .collect()
}

#[test]
fn a_split_label_that_does_not_spell_the_word_no_longer_ends_the_group() {
    // The engine split one transcript word across two labels AND spelled it
    // its own way ("col" + "or" for "colour"). Concatenation cannot recover
    // that, so the stitch fails on the FIRST word, and until 2026-09-07 the
    // whole group stopped there: both words came back untimed even though
    // "shirt" had an onset of its own that nothing disputed.
    let words = make_fa_words(&["colour", "shirt"]);
    let tokens = labels(&[("col", 0.1), ("or", 0.4), ("shirt", 0.9)]);
    let (_recording, window) = remap_window();

    let mapping = map_labels_to_words(&words, &tokens, &window, &remap_engine());

    // Every word is timed, and the numbers are label onsets: "colour" spans
    // from the onset of "col" to the onset of the next label the DP did not
    // give it, and "shirt" is the last word, so its end is the named fallback.
    assert_eq!(
        spans(mapping.outcomes()),
        vec![Some((100, 900)), Some((900, 1_400))]
    );
    // The outcome says how completely each word's labels accounted for it.
    // "colour" against "col" + "or" leaves the "u" unmatched; "shirt" and its
    // own label match one for one, so nothing about it was guessed even though
    // it was the stitch's residue.
    let WordTimingOutcome::Timed { edits, .. } = &mapping.outcomes()[0] else {
        panic!("the split word is timed: {:?}", mapping.outcomes()[0])
    };
    assert_eq!(
        *edits,
        CharEdits {
            transcript_only: 1,
            label_only: 0,
        }
    );
    let WordTimingOutcome::Timed { edits, .. } = &mapping.outcomes()[1] else {
        panic!("the trailing word is timed: {:?}", mapping.outcomes()[1])
    };
    assert_eq!(*edits, CharEdits::ZERO);
}

#[test]
fn a_numeral_takes_the_span_of_the_labels_that_spell_it_out() {
    // "1995" against "nineteen ninety five" shares not one character with its
    // labels, so no character alignment can anchor it. What the DP DOES settle
    // is its neighbours: "the" stitched before it and "election" matched after
    // it, which brackets the numeral between two label positions. The labels in
    // that bracket are its span.
    let words = make_fa_words(&["the", "1995", "election"]);
    let tokens = labels(&[
        ("the", 0.1),
        ("nineteen", 0.5),
        ("ninety", 1.0),
        ("five", 1.4),
        ("election", 1.8),
    ]);
    let (_recording, window) = remap_window();

    let mapping = map_labels_to_words(&words, &tokens, &window, &remap_engine());

    assert_eq!(
        spans(mapping.outcomes()),
        vec![
            // "the" stitched exactly, and ends where the next label begins.
            Some((100, 500)),
            // The numeral covers "nineteen ninety five": from the onset of
            // "nineteen" to the onset of "election", which is the first label
            // the DP gave to somebody else.
            Some((500, 1_800)),
            // "election" is last, so its end is assumed.
            Some((1_800, 2_300)),
        ]
    );
    let WordTimingOutcome::Timed { edits, .. } = &mapping.outcomes()[0] else {
        panic!("the stitched word is timed: {:?}", mapping.outcomes()[0])
    };
    assert_eq!(*edits, CharEdits::ZERO, "\"the\" stitched exactly");
    // The numeral's edit counts say plainly that nothing about it matched: all
    // four of its characters went unmatched, and it absorbed eighteen label
    // characters it does not contain. A reader can tell this word from one the
    // DP actually recognised.
    let WordTimingOutcome::Timed { edits, .. } = &mapping.outcomes()[1] else {
        panic!("the numeral is timed: {:?}", mapping.outcomes()[1])
    };
    assert!(
        !edits.is_exact(),
        "the numeral shares no character with its labels, so its fold is not exact"
    );
    assert_eq!(edits.transcript_only, 4);
    assert_eq!(edits.label_only, 18);
}

#[test]
fn an_unbroken_stitch_never_reaches_the_dp() {
    // The control. The DP is a fallback for the RESIDUE of a broken stitch, so
    // a group whose labels tokenize exactly as the transcript does must produce
    // what it always produced, down to both origins on both words. This is the
    // same fixture as `test_parse_fa_response_token_level`, asserted at the
    // mapping seam so the route is visible as well as the numbers.
    let words = make_fa_words(&["hello", "world"]);
    let tokens = labels(&[("hello", 0.1), ("world", 0.6)]);
    let (_recording, window) = remap_window();

    let mapping = map_labels_to_words(&words, &tokens, &window, &remap_engine());

    let WordTimingOutcome::Timed {
        timing: first,
        edits: first_edits,
    } = &mapping.outcomes()[0]
    else {
        panic!("an exact stitch must not be routed through the DP")
    };
    let WordTimingOutcome::Timed {
        timing: second,
        edits: second_edits,
    } = &mapping.outcomes()[1]
    else {
        panic!("an exact stitch must not be routed through the DP")
    };
    assert_eq!(*first_edits, CharEdits::ZERO);
    assert_eq!(*second_edits, CharEdits::ZERO);
    assert_eq!(
        Some(first.clone()),
        WordTiming::new(
            100,
            600,
            Origin::EngineMeasured {
                engine: remap_engine()
            },
            Origin::DerivedFromNextOnset
        )
    );
    assert_eq!(
        Some(second.clone()),
        WordTiming::new(
            600,
            1_100,
            Origin::EngineMeasured {
                engine: remap_engine()
            },
            Origin::FallbackDuration {
                assumed: Ms(LAST_WORD_FALLBACK_MS)
            }
        )
    );
}

#[test]
fn two_words_merged_into_one_label_time_the_one_the_dp_can_place() {
    // The other half of a tokenizer disagreement, and the honest limit of this
    // change. One label ("gonna") covers two transcript words ("going" "to"),
    // and a label is ONE measured interval: splitting it would invent a
    // boundary between the two words that nothing measured. So the word the
    // characters favour keeps the label and the other stays untimed, named as
    // such rather than silently absent.
    //
    // What the change still buys here is everything AFTER the disagreement:
    // "school" used to be lost with the rest of the group.
    let words = make_fa_words(&["going", "to", "school"]);
    let tokens = labels(&[("gonna", 0.2), ("school", 1.0)]);
    let (_recording, window) = remap_window();

    let mapping = map_labels_to_words(&words, &tokens, &window, &remap_engine());

    assert_eq!(spans(mapping.outcomes())[0], Some((200, 1_000)));
    assert_eq!(
        mapping.outcomes()[1],
        WordTimingOutcome::Untimed {
            reason: UntimedReason::NoLabelSpan
        }
    );
    assert_eq!(spans(mapping.outcomes())[2], Some((1_000, 1_500)));
}

/// The engine's own preamble is not the first word's speech.
///
/// Whisper hallucinates "thanks for watching" over silence, and it arrives as
/// labels like any other. When word 0 itself fails to stitch there is no
/// earlier claim at all, so the DP has settled nothing about where this word
/// begins: giving it every label up to the next claim reads as a measurement
/// of 6.2 seconds that nobody made. The rule is that an unplaced word is
/// placed only when a claim brackets it on BOTH sides.
#[test]
fn a_leading_word_the_dp_cannot_anchor_is_refused_not_given_the_preamble() {
    let words = make_fa_words(&["1995", "election"]);
    let tokens = labels(&[
        ("thanks", 0.0),
        ("for", 0.3),
        ("watching", 0.6),
        ("nineteen", 5.0),
        ("ninety", 5.4),
        ("five", 5.8),
        ("election", 6.2),
    ]);
    let (_recording, window) = remap_window();

    let mapping = map_labels_to_words(&words, &tokens, &window, &remap_engine());

    assert_eq!(
        mapping.outcomes()[0],
        WordTimingOutcome::Untimed {
            reason: UntimedReason::NoLabelSpan
        }
    );
    // The word the DP DID anchor keeps its timing: refusing the numeral is not
    // a reason to lose the rest of the group.
    assert_eq!(spans(mapping.outcomes())[1], Some((6_200, 6_700)));
}

/// The same rule at the other end, where the 500 ms fallback hid it.
///
/// A trailing unplaced word used to take every remaining label, and because it
/// has no successor its end came from the assumed duration, so the damage
/// showed up as a plausible-looking half-second rather than as an absurd span.
/// It is the same fabricated attribution: nothing says this word is where the
/// engine's tail began.
#[test]
fn a_trailing_word_the_dp_cannot_anchor_is_refused_not_given_the_tail() {
    let words = make_fa_words(&["election", "1995"]);
    let tokens = labels(&[
        ("election", 0.2),
        ("nineteen", 5.0),
        ("ninety", 5.4),
        ("five", 5.8),
        ("thanks", 6.2),
        ("for", 6.5),
        ("watching", 6.8),
    ]);
    let (_recording, window) = remap_window();

    let mapping = map_labels_to_words(&words, &tokens, &window, &remap_engine());

    assert_eq!(spans(mapping.outcomes())[0], Some((200, 5_000)));
    assert_eq!(
        mapping.outcomes()[1],
        WordTimingOutcome::Untimed {
            reason: UntimedReason::NoLabelSpan
        }
    );
}

/// An EXACTLY reconciled fold is not an attribution.
///
/// RED FIRST (2026-09-07 review, item 8): `Route` recorded which code ran, so
/// a residue word whose characters reconciled perfectly was wrapped in
/// `AttributedByCharAlignment { edits: ZERO }` purely because the DP is what
/// placed it. Zero unreconciled characters on both sides IS the fact the
/// stitch proves, so that wrapper claimed a guess where none was made and the
/// reviewer's tally counted the word as our own answer.
///
/// "shirt" here follows the group's one disagreement ("colour" against "col" +
/// "or"), so it never stitched; its own characters and its own label's
/// characters still match one for one.
#[test]
fn a_residue_word_reconciled_exactly_is_not_marked_as_attributed() {
    let words = make_fa_words(&["colour", "shirt"]);
    let tokens = labels(&[("col", 0.1), ("or", 0.4), ("shirt", 0.9)]);
    let (_recording, window) = remap_window();

    let mapping = map_labels_to_words(&words, &tokens, &window, &remap_engine());
    let WordTimingOutcome::Timed { timing, edits } = &mapping.outcomes()[1] else {
        panic!("the trailing word is timed: {:?}", mapping.outcomes()[1])
    };
    assert_eq!(
        *edits,
        CharEdits::ZERO,
        "precondition: every character of this word and of its label matched"
    );
    assert_eq!(
        timing.start_origin(),
        &Origin::EngineMeasured {
            engine: remap_engine()
        },
        "an exact fold leaves the engine's own measurement alone"
    );
    assert_eq!(
        timing.end_origin(),
        &Origin::FallbackDuration {
            assumed: Ms(LAST_WORD_FALLBACK_MS)
        },
        "and leaves the end exactly the fact the label track settled"
    );

    // The reviewer's summary agrees: nothing here is our own answer.
    let mut tally = ProvenanceTally::default();
    tally.record(timing.start_origin());
    assert_eq!(tally.observed, 1);
    assert_eq!(tally.assumed, 0);
}

/// A remapped word's own NUMBER must say it was remapped.
///
/// The outcome type dies at `into_timings`, which is the module edge, so a
/// route recorded only there would never reach a reviewer: a guessed timing
/// and a proved one would be byte-identical in the transcript, which is the
/// exact laundering `Origin` exists to prevent. The start therefore WRAPS the
/// engine's measurement rather than replacing it, so both facts survive.
#[test]
fn a_dp_remapped_start_wraps_the_engine_measurement_and_reads_as_assumed() {
    let words = make_fa_words(&["colour", "shirt"]);
    let tokens = labels(&[("col", 0.1), ("or", 0.4), ("shirt", 0.9)]);
    let (_recording, window) = remap_window();

    let mapping = map_labels_to_words(&words, &tokens, &window, &remap_engine());
    let timing = mapping.outcomes()[0]
        .timing()
        .expect("the split word is timed");

    assert_eq!(
        timing.start_origin(),
        &Origin::AttributedByCharAlignment {
            was: Box::new(Origin::EngineMeasured {
                engine: remap_engine()
            }),
            // "colour" against "col" + "or": one transcript character (the
            // "u") had nothing to match, and every label character matched.
            edits: CharEdits {
                transcript_only: 1,
                label_only: 0,
            },
        }
    );
    // The measurement is still reachable underneath, so "could re-running the
    // aligner help" and "is this number evidence" stay separate questions.
    assert_eq!(
        timing.start_origin().underlying(),
        &Origin::EngineMeasured {
            engine: remap_engine()
        }
    );
    assert!(!timing.start_origin().is_observation());

    // And the summary a reviewer actually reads counts it as our own answer.
    let mut tally = ProvenanceTally::default();
    tally.record(timing.start_origin());
    assert_eq!(tally.assumed, 1);
    assert_eq!(tally.observed, 0);
    assert!(tally.needs_review());
}

/// BOTH boundaries of a remapped word are attributed, not just the start.
///
/// RED FIRST (2026-09-07 review, item 4): only the start was wrapped, so a
/// remapped word's end reported `DerivedFromNextOnset`, which reads as "the
/// next word's measured onset capped this one" and hides that WHICH label ends
/// this word was the same edit-distance guess that decided which label begins
/// it. The tally then counted one assumed number per remapped word instead of
/// two, and a reviewer comparing the two ends saw a disagreement that does not
/// exist.
///
/// The origin underneath each end is unchanged, so what the label track
/// measured is still reachable through `underlying()`.
#[test]
fn both_boundaries_of_a_remapped_word_are_attributed() {
    let words = make_fa_words(&["colour", "shirt"]);
    let tokens = labels(&[("col", 0.1), ("or", 0.4), ("shirt", 0.9)]);
    let (_recording, window) = remap_window();

    let mapping = map_labels_to_words(&words, &tokens, &window, &remap_engine());
    let timing = mapping.outcomes()[0]
        .timing()
        .expect("the split word is timed");

    let edits = CharEdits {
        transcript_only: 1,
        label_only: 0,
    };
    assert_eq!(
        timing.start_origin(),
        &Origin::AttributedByCharAlignment {
            was: Box::new(Origin::EngineMeasured {
                engine: remap_engine()
            }),
            edits,
        }
    );
    assert_eq!(
        timing.end_origin(),
        &Origin::AttributedByCharAlignment {
            was: Box::new(Origin::DerivedFromNextOnset),
            edits,
        },
        "the end of a remapped word is the same guess as its start"
    );
    // What the label track settled is still underneath both.
    assert_eq!(
        timing.start_origin().underlying(),
        &Origin::EngineMeasured {
            engine: remap_engine()
        }
    );
    assert_eq!(
        timing.end_origin().underlying(),
        &Origin::DerivedFromNextOnset
    );

    // A reviewer's summary now counts both ends as our own answer.
    let mut tally = ProvenanceTally::default();
    tally.record(timing.start_origin());
    tally.record(timing.end_origin());
    assert_eq!(tally.assumed, 2);
    assert_eq!(tally.derived, 0);
}

/// Label ownership that runs backwards is refused, not documented.
///
/// A monotone alignment cannot produce it, and that fact used to be a comment.
/// A comment is gated by nothing, so the property is fed hand-built owner
/// sequences here: the ones a real alignment yields, and the one it cannot.
#[test]
fn label_ownership_that_runs_backwards_is_refused() {
    let slot = WordSlot::for_test;
    // Forward, with an unowned label in the middle and a word owning two
    // labels: everything a real residue alignment produces.
    assert!(owners_are_monotone(&[
        Some(slot(0)),
        Some(slot(0)),
        None,
        Some(slot(1)),
        Some(slot(3)),
    ]));
    assert!(owners_are_monotone(&[]));
    assert!(owners_are_monotone(&[None, None]));
    // Backwards, which would let two words claim interleaved label runs and so
    // be given overlapping spans.
    assert!(!owners_are_monotone(&[Some(slot(1)), Some(slot(0))]));
    // Backwards across an unowned label, which a windowed check would miss.
    assert!(!owners_are_monotone(&[
        Some(slot(2)),
        None,
        None,
        Some(slot(1)),
    ]));
}

/// A residue too long to align is refused, and says so.
///
/// Grouping caps a MERGE, never a single oversized utterance, so nothing
/// upstream bounds the DP's input. Refused rather than truncated: half a
/// comparison would attribute labels chosen without ever seeing the words they
/// belong to.
#[test]
fn a_residue_longer_than_the_budget_is_refused_rather_than_truncated() {
    // One word that cannot stitch, then enough transcript to overrun the
    // budget on the transcript side.
    let long_word = "a".repeat(64);
    let mut texts: Vec<String> = vec!["zzzz".to_string()];
    texts.extend(std::iter::repeat_n(long_word, MAX_RESIDUE_ALIGN_CHARS / 64));
    let borrowed: Vec<&str> = texts.iter().map(String::as_str).collect();
    let words = make_fa_words(&borrowed);
    let tokens = labels(&[("qqqq", 0.1), ("wwww", 0.5)]);
    let (_recording, window) = remap_window();

    let mapping = map_labels_to_words(&words, &tokens, &window, &remap_engine());

    let WordTimingOutcome::Untimed {
        reason:
            UntimedReason::ResidueTooLongToAlign {
                transcript_chars,
                budget,
                ..
            },
    } = &mapping.outcomes()[0]
    else {
        panic!(
            "an over-budget residue must be refused: {:?}",
            mapping.outcomes()[0]
        )
    };
    assert!(*transcript_chars > *budget);
    assert_eq!(*budget, MAX_RESIDUE_ALIGN_CHARS);
    // Every word of the residue carries the same stated cause, not the
    // per-word "no label span" that would send a reader after the wrong thing.
    assert!(mapping.outcomes().iter().all(|outcome| matches!(
        outcome,
        WordTimingOutcome::Untimed {
            reason: UntimedReason::ResidueTooLongToAlign { .. }
        }
    )));
}

/// A refusal's log line must state the refusal's OWN cause.
///
/// Every refusal used to open "labels did not tokenize as the transcript
/// does", including a group with no alphanumeric content and a group with no
/// surviving labels, neither of which ever reached the tokenization question.
/// A stated cause that is not the cause is worse than no cause, so the
/// group-level reasons name themselves and the per-word ones stay silent and
/// let the counts speak.
#[test]
fn a_group_refusal_states_its_own_cause_in_the_log_line() {
    for reason in [
        UntimedReason::NoLexicalContent,
        UntimedReason::NoUsableLabels,
        UntimedReason::NonMonotoneAttribution,
        UntimedReason::ResidueTooLongToAlign {
            transcript_chars: 9_000,
            label_chars: 12,
            budget: MAX_RESIDUE_ALIGN_CHARS,
        },
    ] {
        let headline = reason
            .group_headline()
            .unwrap_or_else(|| panic!("{reason:?} refuses a whole group and must say why"));
        assert!(
            !headline.contains("did not tokenize"),
            "{reason:?} borrowed the tokenization sentence: {headline}"
        );
    }
    for reason in [
        UntimedReason::NoLabelSpan,
        UntimedReason::TimingRefusedProvenSpan,
    ] {
        assert_eq!(reason.group_headline(), None);
    }
}

/// The same word spelled two legal ways must compare equal.
///
/// Unicode gives "e" with an acute accent two encodings: NFC, one code point
/// U+00E9, and NFD, "e" followed by the combining mark U+0301. A transcript
/// and an engine have no reason to agree on which, and the filter here keeps
/// only alphanumerics, which a combining mark is NOT. So the NFD spelling used
/// to lose its accent and normalize to "cafe" while the NFC spelling kept it
/// as "café": two different keys for one word, and the stitch would fail on a
/// difference that exists only in the encoding.
///
/// Composing first is what makes the filter safe: the mark folds INTO its base
/// character instead of being dropped beside it.
#[test]
fn the_two_unicode_spellings_of_one_accented_word_normalize_alike() {
    let nfd = "cafe\u{301}";
    let nfc = "caf\u{e9}";
    // The fixture is only interesting if the two really are different bytes.
    assert_ne!(nfd, nfc);
    assert_eq!(
        normalize_fa_alignment_unit(nfd),
        normalize_fa_alignment_unit(nfc)
    );
    // And the accent SURVIVES rather than both collapsing to "cafe", which
    // would make this pass by destroying the distinction instead of resolving
    // it: "café" and "cafe" are different words.
    assert_eq!(normalize_fa_alignment_unit(nfd), nfc);
}

/// Composing must not disturb the ordinary case.
///
/// Almost every word this pipeline sees is ASCII, and NFC is identity on it.
/// Pinned because the composition was added for a rare input and runs on every
/// word of every group: a change of behaviour here would be a change to
/// essentially the whole corpus.
#[test]
fn ascii_text_is_untouched_by_composition() {
    assert_eq!(normalize_fa_alignment_unit("Hello, World!"), "helloworld");
    assert_eq!(normalize_fa_alignment_unit("1995"), "1995");
    assert_eq!(normalize_fa_alignment_unit("don't"), "dont");
    assert_eq!(normalize_fa_alignment_unit(","), "");
}
