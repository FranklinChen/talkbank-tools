//! Every word timing this run threw away must reach the run's evidence
//! artifact as a standalone, self-describing record.
//!
//! The defect these tests pin: `WordClampOutcome::DroppedPastBound` carries
//! the measured extent it lost, and that extent reached only the in-memory
//! effect stream, nested inside one variant of one tagged union. A reviewer
//! asking "what measurements did this run discard?" had to walk
//! `timing_decisions`, match a single tag, and join each dropped word back to
//! its parent for the speaker, the utterance and the bound it exceeded. A
//! measurement that is that hard to recover is, in practice, lost, which is
//! the concrete form of "preserve information; losing it is the defect".
//!
//! These are ARTIFACT tests, at the real seam: the pipeline runs, the
//! evidence file is written to disk, and the assertions read the JSON back.
//! No type can stand in for them, because the thing under test is a wire
//! format leaving the process.

use super::*;

use crate::runner::debug_dumper::{DebugDumper, FaEvidenceDumpOutcome};
use crate::types::results::FaResult;
use crate::types::traces::FaTimelineTrace;

/// Run the no-injection projection over `input` and lower the result through
/// the SAME production path a real align run uses: the finalize typestate
/// that owns monotonicity, the written-decisions proof, the `FaResult` that
/// carries it, and the timeline trace that is what gets serialized. Nothing
/// here hand-simulates a pipeline step; the point of these tests is that the
/// artifact a REAL run writes carries the drops.
fn timeline_trace_for(input: &str) -> FaTimelineTrace {
    let mut chat = parse_chat(input);
    let projection = FaProjection::new(
        FaProjectionPolicy::new(
            WordEndPolicy::measured(WordGapHealing::PreserveMeasured),
            ExistingWorBoundaryPolicy::Preserve,
            EndOverlapPolicy::ClampAllAdjacent,
        ),
        MainBulletAuthority::DeriveFromWords,
    );
    let finalized = finalize_without_injection(&mut chat, projection, BulletRepairPolicy::Disabled)
        .expect("finalization under the default policy holds");
    let written = retain_decision_evidence(
        &mut chat,
        FaDecisions {
            rescue: Vec::new(),
            unplaceable: Vec::new(),
            finalized,
        },
    );
    FaResult::without_groups(
        crate::chat_ops::ChatFile::new(Vec::new()),
        WordGapHealing::PreserveMeasured,
        "test_engine",
        &crate::engine_reports::FaCacheNamespace::for_test("test-build"),
    )
    .expect("an empty untimed CHAT model can be retained as evidence")
    .with_written_decisions(written)
    .into_timeline_trace()
}

/// Write `trace` as the run's evidence artifact and read the JSON back.
fn dumped_evidence_json(trace: &FaTimelineTrace) -> serde_json::Value {
    let dir = tempfile::tempdir().expect("tempdir");
    let dumper = DebugDumper::new(Some(dir.path()));
    let FaEvidenceDumpOutcome::Written(path) = dumper
        .dump_fa_evidence("sample.cha", trace)
        .expect("an enabled evidence dump must be durable")
    else {
        panic!("an enabled dumper must write the evidence artifact");
    };
    serde_json::from_str(&std::fs::read_to_string(&path).expect("read evidence artifact"))
        .expect("evidence artifact must be valid JSON")
}

/// `third` (4200..4800) lies wholly past the bound (4000), so the clamp
/// throws its whole measured extent away. That extent must appear in the
/// artifact as a record a reviewer can read on its own: whose word it was,
/// which utterance and tier and position, what was measured, and what bound
/// the measurement exceeded.
#[test]
fn a_dropped_word_reaches_the_artifact_with_its_measured_span() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n@ID:\teng|x|CHI|||||Child|||\n*CHI:\thello world third . \u{15}1000_5000\u{15}\n%wor:\thello \u{15}1000_3000\u{15} world \u{15}3000_5000\u{15} third \u{15}4200_4800\u{15} .\n*CHI:\tnext . \u{15}4000_6000\u{15}\n%wor:\tnext \u{15}4000_6000\u{15} .\n@End\n";

    let json = dumped_evidence_json(&timeline_trace_for(input));

    let dropped = json["dropped_word_timings"]
        .as_array()
        .expect("the artifact must carry a dropped_word_timings section");
    let wor = dropped
        .iter()
        .find(|entry| entry["tier"] == "wor")
        .expect("the dropped %wor slot must be listed");

    assert_eq!(wor["speaker"], "CHI");
    assert_eq!(wor["utterance_idx"], 0);
    assert_eq!(wor["word_index"], 2);
    assert_eq!(wor["start_ms"], 4200);
    assert_eq!(wor["end_ms"], 4800);
    assert_eq!(
        wor["bound_ms"], 4000,
        "the record must name the bound the measurement exceeded, so it is \
         readable without its parent effect"
    );
}

/// A run that discards nothing still writes the section, empty. An absent
/// key and an empty one read identically to a consumer that does not know
/// the schema version it is looking at, and "no drops" is a fact worth
/// stating rather than one to be inferred from silence.
#[test]
fn a_run_that_drops_nothing_still_writes_an_empty_section() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n@ID:\teng|x|CHI|||||Child|||\n*CHI:\thello . \u{15}1000_5000\u{15}\n%wor:\thello \u{15}1000_1500\u{15} .\n*CHI:\tnext . \u{15}2000_6000\u{15}\n%wor:\tnext \u{15}2000_6000\u{15} .\n@End\n";

    let json = dumped_evidence_json(&timeline_trace_for(input));

    assert_eq!(
        json["dropped_word_timings"].as_array().map(Vec::len),
        Some(0),
        "the section must be present and empty, never absent"
    );
}
