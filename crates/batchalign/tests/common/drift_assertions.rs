//! Shared evaluator for the output-structural drift invariant used by both the
//! regression-fixture harness
//! (`tests/ml_golden/regression_fixtures/harness.rs`) and the env-gated
//! real-file drift integration tests
//! (`tests/ml_golden/align/drift_integration.rs` + `drift_runner.rs`) dispatch
//! through the function exported here.
//!
//! Legacy checks that searched `%xalign` or `%xrev` for failure markers were
//! removed when BA3 made those tiers non-generating and strips old copies.
//! Keeping them would create false-green tests: absence of a deleted reporting
//! surface cannot prove absence of the underlying defect.

use batchalign::chat_ops::ChatFile;

use crate::common::regression_manifest::FixtureAssertion;

/// Dispatch the output-structural drift assertion.
pub fn evaluate_drift_assertion(
    parsed: &ChatFile,
    assertion: &FixtureAssertion,
) -> Result<(), String> {
    match assertion {
        FixtureAssertion::UtteranceBulletMonotonicityPreserved => {
            check_utterance_bullet_monotonicity_preserved(parsed)
        }
        other => Err(format!(
            "evaluate_drift_assertion: unsupported assertion variant {other:?}; drift helper \
             only covers UtteranceBulletMonotonicityPreserved"
        )),
    }
}

/// Walk every timed main-tier utterance in document order and verify strict
/// start-time monotonicity. Utterances without a bullet are skipped (they
/// carry no timing to compare).
///
/// Overlap-continuation utterances: those carrying a `+<` LazyOverlapPrecedes
/// linker or a ⌊ CA bottom-overlap marker, legitimately share start timing
/// with their predecessor by design. They are skipped from BOTH the comparison
/// (their start is not a violation) AND the `prev_start` baseline (their start
/// must not become the lower bound the next non-overlap utterance is compared
/// against). Detection mirrors `batchalign::chat_ops::fa::utr::select_strategy`
///: the canonical overlap-aware pattern.
pub fn check_utterance_bullet_monotonicity_preserved(parsed: &ChatFile) -> Result<(), String> {
    use batchalign::chat_ops::fa::utr::overlap_markers;
    let mut prev_start: Option<u64> = None;
    let mut prev_index: usize = 0;
    for (i, utt) in parsed.utterances().enumerate() {
        let bullet = match utt.main.content.bullet.as_ref() {
            None => continue,
            Some(talkbank_model::BulletState::Parsed(bullet)) => bullet,
            Some(talkbank_model::BulletState::Unparsed { unparsed }) => {
                return Err(format!(
                    "utterance #{i} has an unparsed bullet: {}",
                    unparsed.source()
                ));
            }
        };
        let is_overlap_continuation = utt
            .main
            .content
            .linkers
            .iter()
            .any(|l| l.kind == talkbank_model::model::LinkerKind::LazyOverlapPrecedes)
            || overlap_markers::extract_overlap_info(&utt.main.content.content)
                .has_bottom_overlap();
        if is_overlap_continuation {
            // Do not compare against prev_start, and do not update prev_start
            // from this utterance.
            continue;
        }
        let this_start = bullet.timing.start_ms;
        if let Some(p) = prev_start
            && this_start <= p
        {
            return Err(format!(
                "utterance_bullet_monotonicity_preserved: utterance #{} start {} ms \
                 does not advance past utterance #{} start {} ms",
                i, this_start, prev_index, p,
            ));
        }
        prev_start = Some(this_start);
        prev_index = i;
    }
    Ok(())
}
