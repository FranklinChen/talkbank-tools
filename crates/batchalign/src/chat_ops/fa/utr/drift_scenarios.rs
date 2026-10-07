//! Drift-class regression scenarios for UTR.
//!
//! These unit tests exercise the public [`super::inject_utr_timing`] entry
//! point against procedurally synthesized CHAT + ASR inputs that deterministically
//! exercise document-order versus acoustic-order ambiguity. They are synthetic
//! algorithm counterexamples, not actual model runs or corpus accuracy evidence.
//! Complete CHAT admission precedes injection. Existing ignored long controls
//! remain a residual inventory; changing strategy is not itself a repair proof.
//!
//! # Invariants checked by [`run_and_check`]
//!
//! For every utterance that UTR assigned a bullet:
//!
//! 1. `end_ms > start_ms` (no zero- or negative-duration bullets).
//! 2. Adjacent **non-overlap** utterance bullets are strictly monotone in
//!    `start_ms`. Overlap-continuation utterances: those carrying a `+<`
//!    [`Linker::LazyOverlapPrecedes`] OR a `⌊` CA bottom-overlap marker
//!    legitimately share timing with a predecessor and are EXCLUDED from the
//!    monotonicity chain (they neither participate in the comparison nor
//!    advance the `prev_start` cursor). This mirrors the overlap-aware pattern
//!    in the regression-harness arm for `UtteranceBulletMonotonicityPreserved`
//!    shipped with Task 1.1.
//! 3. Each assigned bullet lands inside the utterance's **ground-truth audio
//!    window**, expanded by one 500 ms word-cadence slack. This is the
//!    invariant that actually catches DP drift, the DP can remain
//!    monotone-preserving while still matching utterance K's words to tokens
//!    belonging to utterance K-3, producing a "wrong audio region" bullet
//!    that monotonicity alone cannot detect. Ground-truth windows are
//!    recovered from the synthetic cadence used in [`build_scenario`].
//!
//! Recovery coverage is retained separately, including turns with no surviving
//! acoustic words. Only the complete observation type certifies recovery of
//! every expected turn; an empty violation list cannot establish that claim.
//!
//! # Scenario taxonomy
//!
//! See the segment-aware-UTR design notes (operator-local) Task 1.2.
//!
//! | Scenario                        | Convention        | Utts | Drives drift via                  |
//! |---------------------------------|-------------------|------|-----------------------------------|
//! | `drift_ca_overlap_long_file`    | `⌊⌋` + `⌈⌉`      | 500  | dense CA brackets + 10% missing   |
//! | `drift_lazy_overlap_long_file`  | `+<`              | 500  | transcript reordered vs ASR       |
//! | `drift_inline_backchannel_*`    | `&*SPK:`          | 500  | &* tokens absent from ASR stream  |
//! | `drift_mixed_conventions`       | all three         | 500  | combined                          |
//! | `short_file_baseline`           | `⌊⌋`             | 30   | sanity check: should align cleanly |
//! | `anchor_sparse_stopword_heavy`  | none              | 500  | transcript is ALL <4-char tokens  |

use crate::chat_ops::fa::utr::{inject_utr_timing, overlap_markers};
use batchalign_transform::parse_source_with_parser;
use talkbank_model::model::{ChatFile, Line, TranscriptName};
use talkbank_parser::TreeSitterParser;

mod assessment;
mod fixture;
mod search_recovery;

use assessment::DriftObservation;
use fixture::{AdmittedDriftScenario, DriftParams, OverlapConvention, build_scenario};

/// A single non-overlap utterance's witnessed start time, tagged with its
/// main-tier index so violation messages can point at the offending utterance.
#[derive(Debug, Clone, Copy)]
struct MonotoneProbe {
    utt_index: usize,
    start_ms: u64,
}

/// Consume an admitted scenario and retain both drift and recovery coverage.
fn run_and_check(scenario: AdmittedDriftScenario) -> DriftObservation {
    run_and_check_with(scenario, &super::GlobalUtr)
}

fn run_and_check_with(
    scenario: AdmittedDriftScenario,
    strategy: &dyn super::UtrStrategy,
) -> DriftObservation {
    let AdmittedDriftScenario {
        source,
        tokens,
        expected,
    } = scenario;
    let mut chat = source.into_valid_file().into_unchecked();
    let result = strategy.inject(&mut chat, &tokens);
    let mut violations = Vec::new();
    check_bullet_integrity(&chat, &mut violations);
    check_utterance_monotonicity(&chat, &mut violations);
    DriftObservation::observe(&chat, &expected, result, violations)
}

/// Invariant 1: every bullet has `end_ms > start_ms`.
///
/// `run_global_utr` already refuses to emit a bullet with `start >= end`
/// (zero-duration frames go to `unmatched`), so this serves as a belt-and-
/// braces check: if a future strategy regresses that rule, this fires.
fn check_bullet_integrity(chat: &ChatFile, violations: &mut Vec<String>) {
    for (i, line) in chat.lines.iter().enumerate() {
        let Line::Utterance(utt) = line else { continue };
        let Some(bullet) = utt.main.content.bullet.as_ref() else {
            continue;
        };
        let (s, e) = (bullet.timing.start_ms, bullet.timing.end_ms);
        if e <= s {
            violations.push(format!(
                "line {i}: non-positive bullet duration (start={s}ms, end={e}ms)"
            ));
        }
    }
}

/// Invariant 2: adjacent non-overlap utterance bullets are strictly monotone
/// in `start_ms`.
///
/// Overlap-continuation utterances (`+<` linker OR `⌊`-bearing text) are
/// excluded from BOTH sides of the comparison, matching the Task 1.1
/// regression-harness arm `UtteranceBulletMonotonicityPreserved`.
fn check_utterance_monotonicity(chat: &ChatFile, violations: &mut Vec<String>) {
    let mut prev: Option<MonotoneProbe> = None;
    let mut utt_ordinal: usize = 0;
    for (line_idx, line) in chat.lines.iter().enumerate() {
        let Line::Utterance(utt) = line else { continue };
        let this_ordinal = utt_ordinal;
        utt_ordinal += 1;
        let Some(bullet) = utt.main.content.bullet.as_ref() else {
            continue;
        };
        let is_overlap = utt
            .main
            .content
            .linkers
            .iter()
            .any(|l| l.kind == talkbank_model::model::LinkerKind::LazyOverlapPrecedes)
            || overlap_markers::extract_overlap_info(&utt.main.content.content)
                .has_bottom_overlap();
        if is_overlap {
            continue;
        }
        let this_start = bullet.timing.start_ms;
        if let Some(p) = prev
            && this_start <= p.start_ms
        {
            violations.push(format!(
                "line {line_idx} (utt #{this_ordinal}): non-monotonic start \
                 (this={this_start}ms <= prev={prev_ms}ms from utt #{prev_idx})",
                prev_ms = p.start_ms,
                prev_idx = p.utt_index,
            ));
        }
        prev = Some(MonotoneProbe {
            utt_index: this_ordinal,
            start_ms: this_start,
        });
    }
    debug_assert!(
        utt_ordinal
            == chat
                .lines
                .iter()
                .filter(|l| matches!(l, Line::Utterance(_)))
                .count(),
        "utt_ordinal desync with chat.lines utterance count: {}",
        utt_ordinal,
    );
}

// --------------------------------------------------------------------------
// Helper self-tests.
//
// Confirm `check_bullet_integrity` and `check_utterance_monotonicity` detect
// the shapes they claim to, on tiny hand-crafted documents. If these pass,
// the scenario assertions can be trusted.
// --------------------------------------------------------------------------

#[cfg(test)]
mod helper_self_tests {
    use super::*;
    use talkbank_model::model::Bullet;

    #[test]
    fn complete_abstention_cannot_claim_complete_recovery() {
        let observation = run_and_check(build_scenario(DriftParams {
            n_utts: 3,
            convention: OverlapConvention::None,
            overlap_density: 0.0,
            asr_missing_rate: 1.0,
            stopword_only: false,
        }));
        assert!(observation.violations().is_empty());
        assert!(
            observation
                .summary()
                .contains("unrecovered_without_words=3")
        );
        assert!(observation.try_complete().is_err());
    }

    #[test]
    fn intact_acoustic_control_establishes_complete_recovery() {
        let observation = run_and_check(build_scenario(DriftParams {
            n_utts: 3,
            convention: OverlapConvention::None,
            overlap_density: 0.0,
            asr_missing_rate: 0.0,
            stopword_only: false,
        }));
        let complete = observation.try_complete().unwrap();
        assert!(complete.summary().contains("windows=3 within=3 outside=0"));
    }

    #[test]
    fn scorer_retains_missing_and_misplaced_output_separately() {
        let AdmittedDriftScenario {
            source,
            tokens,
            expected,
        } = build_scenario(DriftParams {
            n_utts: 3,
            convention: OverlapConvention::None,
            overlap_density: 0.0,
            asr_missing_rate: 0.0,
            stopword_only: false,
        });
        assert_eq!(
            expected
                .iter()
                .map(|window| window.surviving_asr_words)
                .sum::<usize>(),
            tokens.len()
        );
        let mut chat = source.into_valid_file().into_unchecked();
        let result = inject_utr_timing(&mut chat, &tokens);
        // Deliberately corrupt the OUTPUT to test the scorer. The algorithm
        // above receives only completely admitted source and acoustic tokens.
        let mut utterances = chat
            .lines
            .as_mut_slice()
            .iter_mut()
            .filter_map(|line| match line {
                Line::Utterance(utterance) => Some(utterance),
                _ => None,
            });
        utterances.next().unwrap().main.content.bullet = None;
        utterances.next().unwrap().main.content.bullet = Some(Bullet::new(20_000, 21_000));
        let observation = DriftObservation::observe(&chat, &expected, result, Vec::new());
        assert_eq!(observation.violations().len(), 1);
        assert!(observation.summary().contains(
            "windows=3 within=1 outside=1 unrecovered_with_words=1 unrecovered_without_words=0"
        ));
        assert!(observation.try_complete().is_err());
    }

    /// Build a 3-utterance CHAT with the supplied bullet timings (None => untimed).
    /// Optionally mark utterance `i` as lazy-overlap (`+<`).
    fn make_three_utt_chat(
        bullets: [Option<(u64, u64)>; 3],
        lazy_overlap_mask: [bool; 3],
    ) -> ChatFile {
        let parser = TreeSitterParser::new().unwrap();
        let mut text = String::new();
        text.push_str("@UTF8\n@Begin\n");
        text.push_str("@Languages:\teng\n");
        text.push_str("@Participants:\tMAI Participant, OTH Participant\n");
        text.push_str("@ID:\teng|test|MAI|||||Participant|||\n");
        text.push_str("@ID:\teng|test|OTH|||||Participant|||\n");
        text.push_str("@Media:\tsynthetic, audio, unlinked\n");
        for (i, &is_lazy) in lazy_overlap_mask.iter().enumerate() {
            let prefix = if is_lazy { "+< " } else { "" };
            let label = ["alpha", "beta", "gamma"][i];
            text.push_str(&format!("*MAI:\t{prefix}hello world {label} .\n"));
        }
        text.push_str("@End\n");
        let errors = talkbank_model::ErrorCollector::new();
        let mut chat = parse_source_with_parser(&parser, &text)
            .admit(TranscriptName::Anonymous, &errors)
            .unwrap_or_else(|error| panic!("helper base failed complete admission before deliberate boundary corruption: {error}"))
            .into_valid_file().into_unchecked();
        // Intentional invalid bullets below test the checker itself. These
        // mutated documents never enter the UTR algorithm or model execution.
        let mut utt_i = 0;
        for line in chat.lines.as_mut_slice().iter_mut() {
            if let Line::Utterance(utt) = line {
                if let Some((s, e)) = bullets[utt_i] {
                    utt.main.content.bullet = Some(Bullet::new(s, e));
                }
                utt_i += 1;
                if utt_i == 3 {
                    break;
                }
            }
        }
        chat
    }

    #[test]
    fn bullet_integrity_detects_zero_duration() {
        let chat = make_three_utt_chat([Some((1000, 1000)), Some((2000, 3000)), None], [false; 3]);
        let mut v = Vec::new();
        check_bullet_integrity(&chat, &mut v);
        assert_eq!(v.len(), 1, "expected one violation, got: {v:?}");
        assert!(v[0].contains("non-positive"), "msg: {}", v[0]);
    }

    #[test]
    fn bullet_integrity_passes_clean_document() {
        let chat = make_three_utt_chat(
            [Some((0, 500)), Some((500, 1000)), Some((1000, 1500))],
            [false; 3],
        );
        let mut v = Vec::new();
        check_bullet_integrity(&chat, &mut v);
        assert!(v.is_empty(), "unexpected violations: {v:?}");
    }

    #[test]
    fn monotonicity_detects_backwards_start() {
        // Utt 1 starts BEFORE utt 0 → should flag.
        let chat = make_three_utt_chat(
            [Some((1000, 1500)), Some((500, 900)), Some((2000, 2500))],
            [false; 3],
        );
        let mut v = Vec::new();
        check_utterance_monotonicity(&chat, &mut v);
        assert_eq!(v.len(), 1, "expected one violation, got: {v:?}");
        assert!(v[0].contains("non-monotonic"), "msg: {}", v[0]);
    }

    #[test]
    fn monotonicity_skips_lazy_overlap_utterance() {
        // Utt 1 has +< AND starts before utt 0, legitimate overlap, should NOT flag.
        // Utt 2 must still be ahead of utt 0 (the last non-overlap anchor).
        let chat = make_three_utt_chat(
            [Some((1000, 1500)), Some((500, 900)), Some((2000, 2500))],
            [false, true, false],
        );
        let mut v = Vec::new();
        check_utterance_monotonicity(&chat, &mut v);
        assert!(v.is_empty(), "unexpected violations: {v:?}");
    }

    #[test]
    fn monotonicity_passes_strictly_monotone_document() {
        let chat = make_three_utt_chat(
            [Some((0, 500)), Some((500, 1000)), Some((1000, 1500))],
            [false; 3],
        );
        let mut v = Vec::new();
        check_utterance_monotonicity(&chat, &mut v);
        assert!(v.is_empty(), "unexpected violations: {v:?}");
    }
}

// --------------------------------------------------------------------------
// Scenario tests
//
// These exercise the direct global injector, not command strategy selection.
// No-drift and complete recovery are distinct claims. The short baseline
// requires a complete observation; long residuals retain their drift assertions.
// --------------------------------------------------------------------------

/// Observe the existing density policy and explicit exclusion on unchanged
/// admitted controls. This is a counterfactual inventory, not a quality pass.
#[test]
#[ignore = "manual policy counterfactual observation; does not establish recovery quality"]
fn overlap_exclusion_counterfactuals() {
    for (name, convention, density, missing) in [
        ("ca", OverlapConvention::CaBracket, 0.60, 0.25),
        ("lazy", OverlapConvention::LazyPrecedes, 0.45, 0.15),
        (
            "backchannel",
            OverlapConvention::InlineBackchannel,
            0.45,
            0.10,
        ),
        ("mixed", OverlapConvention::Mixed, 0.60, 0.15),
    ] {
        for (policy, max_exclusion_density) in [("default", "0.30"), ("exclude", "1.0")] {
            let strategy = super::TwoPassOverlapUtr::new().with_config(super::TwoPassConfig {
                max_exclusion_density: max_exclusion_density.parse().expect("checked density"),
                ..super::TwoPassConfig::default()
            });
            let observation = run_and_check_with(
                build_scenario(DriftParams {
                    n_utts: 500,
                    convention,
                    overlap_density: density,
                    asr_missing_rate: missing,
                    stopword_only: false,
                }),
                &strategy,
            );
            println!("counterfactual_{name}_{policy}: {}", observation.summary());
            for violation in observation.violations().iter().take(10) {
                println!("counterfactual_{name}_{policy} violation: {violation}");
            }
        }
    }
}

/// Sanity: a short file with CA brackets should align cleanly on the current
/// pipeline. This is the control: if this ever goes RED, the problem is with
/// the scenario builder, not the production code.
#[test]
fn short_file_baseline() {
    let scenario = build_scenario(DriftParams {
        n_utts: 30,
        convention: OverlapConvention::CaBracket,
        overlap_density: 0.2,
        asr_missing_rate: 0.0,
        stopword_only: false,
    });
    let observation = run_and_check(scenario);
    let complete = observation.try_complete().unwrap_or_else(|observation| {
        panic!(
            "short baseline did not establish complete recovery: {}\n{:?}",
            observation.summary(),
            observation.violations()
        )
    });
    println!("{}", complete.summary());
}

/// Long-file CA-bracket drift: mimics MICASE / samtale. Dense `⌈⌉/⌊⌋` pairs
/// plus ~10% missing ASR tokens force the monotonic DP to commit to the wrong
/// repeated-word match and accumulate offset over hundreds of utterances.
///
/// The direct global-injector control is independent of command dispatch.
/// Any alternate strategy needs its own evidence rather than an assumed fix.
///
/// `#[ignore]` because the test is currently RED on the default pipeline
/// this is by design, as the test codifies a bug we have not yet fixed.
/// Run explicitly with `cargo test … -- --ignored` to see the RED evidence.
#[test]
#[ignore = "manual long drift/recovery assessment; no-drift is not complete recovery"]
fn drift_ca_overlap_long_file() {
    // Retuned for temporal-interleaving model (2026-04-22): higher
    // asr_missing_rate forces the DP off the correct path under
    // CA-bracket-induced interleaving.
    let scenario = build_scenario(DriftParams {
        n_utts: 500,
        convention: OverlapConvention::CaBracket,
        overlap_density: 0.60,
        asr_missing_rate: 0.25,
        stopword_only: false,
    });
    let observation = run_and_check(scenario);
    let result = observation.summary();
    println!("drift_ca_overlap_long_file: {result}");
    let violations = observation.violations();
    assert!(
        violations.is_empty(),
        "drift_ca_overlap_long_file: {} violations (showing up to 10)\n\
         result={result:?}\n  {}",
        violations.len(),
        violations
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n  "),
    );
}

/// Long-file `+<` drift: mimics the biling class. Reordered transcript
/// against temporally-ordered ASR is hard for a single monotonic DP.
#[test]
#[ignore = "manual long drift/recovery assessment; no-drift is not complete recovery"]
fn drift_lazy_overlap_long_file() {
    // Retuned for temporal-interleaving model (2026-04-22).
    let scenario = build_scenario(DriftParams {
        n_utts: 500,
        convention: OverlapConvention::LazyPrecedes,
        overlap_density: 0.45,
        asr_missing_rate: 0.15,
        stopword_only: false,
    });
    let observation = run_and_check(scenario);
    let result = observation.summary();
    println!("drift_lazy_overlap_long_file: {result}");
    let violations = observation.violations();
    assert!(
        violations.is_empty(),
        "drift_lazy_overlap_long_file: {} violations (showing up to 10)\n\
         result={result:?}\n  {}",
        violations.len(),
        violations
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n  "),
    );
}

/// Long-file `&*SPK:` inline-backchannel drift, mimics rhd. ASR does not
/// emit a token for the interjected backchannel, so the DP aligns the
/// main-speaker's post-backchannel words one token too early, drifting
/// subsequent utterances.
#[test]
#[ignore = "manual long drift/recovery assessment; no-drift is not complete recovery"]
fn drift_inline_backchannel_long_file() {
    // Retuned for temporal-interleaving model (2026-04-22): nested backchannel
    // utts produce temporal interleave + frequent "unmatched 2-word" failures.
    let scenario = build_scenario(DriftParams {
        n_utts: 500,
        convention: OverlapConvention::InlineBackchannel,
        overlap_density: 0.45,
        asr_missing_rate: 0.10,
        stopword_only: false,
    });
    let observation = run_and_check(scenario);
    let result = observation.summary();
    println!("drift_inline_backchannel_long_file: {result}");
    let violations = observation.violations();
    assert!(
        violations.is_empty(),
        "drift_inline_backchannel_long_file: {} violations (showing up to 10)\n\
         result={result:?}\n  {}",
        violations.len(),
        violations
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n  "),
    );
}

/// All three conventions interleaved, mimics real MICASE files, which mix
/// CA brackets, `+<`, and inline backchannel tokens freely.
#[test]
#[ignore = "manual long drift/recovery assessment; no-drift is not complete recovery"]
fn drift_mixed_conventions() {
    // Retuned for temporal-interleaving model (2026-04-22).
    let scenario = build_scenario(DriftParams {
        n_utts: 500,
        convention: OverlapConvention::Mixed,
        overlap_density: 0.60,
        asr_missing_rate: 0.15,
        stopword_only: false,
    });
    let observation = run_and_check(scenario);
    let result = observation.summary();
    println!("drift_mixed_conventions: {result}");
    let violations = observation.violations();
    assert!(
        violations.is_empty(),
        "drift_mixed_conventions: {} violations (showing up to 10)\n\
         result={result:?}\n  {}",
        violations.len(),
        violations
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n  "),
    );
}

/// Diagnostic-only sweep that verifies drift scenarios fail robustly across a
/// small parameter grid. Run manually with
/// `cargo test -p batchalign-chat-ops --lib drift_margin_sweep_diagnostic -- --ignored --nocapture`.
///
/// Purpose: prove the drift RED-gate isn't knife-edge. If all grid cells
/// produce >= 20 violations on GlobalUtr, the gate is robust. If some produce
/// 1-2 violations, retune constants.
#[test]
#[ignore]
fn drift_margin_sweep_diagnostic() {
    use std::fmt::Write;
    // Widened (2026-04-22) to cover the production drift-test params, which
    // use density up to 0.60 and missing up to 0.25. This lets the diagnostic
    // demonstrate that those production points are not knife-edge.
    let densities = [0.25, 0.35, 0.45, 0.60];
    let missing_rates = [0.05, 0.10, 0.15, 0.20, 0.25];
    let mut report = String::new();
    writeln!(
        &mut report,
        "convention        density  missing  violations"
    )
    .unwrap();
    for convention in [
        OverlapConvention::CaBracket,
        OverlapConvention::LazyPrecedes,
        OverlapConvention::InlineBackchannel,
    ] {
        for &d in &densities {
            for &r in &missing_rates {
                let params = DriftParams {
                    n_utts: 500,
                    convention,
                    overlap_density: d,
                    asr_missing_rate: r,
                    stopword_only: false,
                };
                let scenario = build_scenario(params);
                let observation = run_and_check(scenario);
                let violations = observation.violations();
                writeln!(
                    &mut report,
                    "{:<18} {:>6.2}  {:>6.2}  {:>10}",
                    format!("{:?}", convention),
                    d,
                    r,
                    violations.len(),
                )
                .unwrap();
            }
        }
    }
    // Print to stdout so `--nocapture` shows it:
    println!("\n{report}");
}

/// Anchor-sparse transcript, no overlap markers, but every word is a
/// high-frequency stopword (< 4 chars). Tests the Option-4 fallback for
/// anchor-starved DP: with no unambiguous rare-word anchors, the global DP
/// has little signal to lock onto. Utterances UTR cannot place should land
/// in `UtrResult.unmatched`, not receive a wrong bullet.
#[test]
fn anchor_sparse_stopword_heavy() {
    let scenario = build_scenario(DriftParams {
        n_utts: 500,
        convention: OverlapConvention::None,
        overlap_density: 0.0,
        asr_missing_rate: 0.15,
        stopword_only: true,
    });
    let observation = run_and_check(scenario);
    let result = observation.summary();
    let violations = observation.violations();
    assert!(
        violations.is_empty(),
        "anchor_sparse_stopword_heavy: {} violations (showing up to 10)\n\
         result={result:?}\n  {}",
        violations.len(),
        violations
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n  "),
    );
}
