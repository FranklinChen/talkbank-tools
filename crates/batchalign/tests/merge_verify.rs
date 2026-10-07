//! RED-first integration tests for the `merge-verify` subcommand
//! (the calibrated verify-flag tier pass over merged drafts).
//!
//! Contract under test: the command consumes a merged draft directory
//! plus an engine-verdicts
//! JSON (produced upstream by the FA / pitch / machine-ear engines or
//! replayed from a cache) and emits a rewritten draft plus a review
//! queue. Corpus-specific flag vocabularies stay OUTSIDE this boundary:
//! the verdicts JSON carries each line's category, and the command
//! identifies verify flags by a prefix.
//!
//! Tier semantics (human-calibrated against blind listening verdicts;
//! measured 97.5% precision on the auto-trust tier):
//! - AUTO_TRUST (category in {other, medium, approx} AND ear YES AND
//!   pitch CHILD): the verify %com is REWRITTEN to a machine-verified
//!   provenance note (never deleted: maintainer ruling, provenance
//!   must survive in the transcript).
//! - REVIEW (trusted category failing a gate): flag unchanged, line
//!   exported in the review queue.
//! - HOLD (clock, region): untouched entirely.
//! - Demotion: a confident line with adverse verdicts GAINS a review
//!   flag; text and timing are never moved.
//! - Preservation invariant: main-tier structure survives normalized writing.
// Integration tests are exempt from the crate's deny-level panic lints,
// matching the src/lib.rs `#![cfg_attr(test, allow(...))]` pattern
// (see docs/panic-audit/).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::todo,
    clippy::unimplemented
)]

use crate::cli_common;

use cli_common::cli_cmd as cmd;

/// A miniature merged draft: four CHI/PAR utterances covering the four
/// tier outcomes. Utterance ordinals (0-based, main-tier lines only)
/// key the verdicts JSON.
const DRAFT_CHA: &str = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child, PAR Participant
@ID:\teng|test|CHI|||||Target_Child|||
@ID:\teng|test|PAR|||||Participant|||
@Media:\tS1, audio
*CHI:\tyes . \u{15}1000_2000\u{15}
%com:\tverify: placement (diarization mislabel)
*CHI:\tno . \u{15}3000_4000\u{15}
%com:\tverify: placement (diarization mislabel)
*CHI:\tmaybe . \u{15}5000_6000\u{15}
%com:\tverify: placement (interpolated)
*PAR:\tsure . \u{15}7000_8000\u{15}
*CHI:\tthis long utterance wraps onto a continuation line and keeps
\tgoing before it ends . \u{15}9000_9900\u{15}
*CHI:\tokay . \u{15}10000_11000\u{15}
%com:\tGONNA HAD ONE SYLLABLE . ; verify: placement (medium-confidence anchor)
*CHI:\tthis very long utterance could never fit inside its collapsed span . \u{15}12000_12001\u{15}
%com:\tverify: placement (diarization mislabel)
*CHI:\tI know how, see ? \u{15}13000_14000\u{15}
@End
";

/// Engine verdicts for the four utterances. The category vocabulary is
/// the calibrated tier taxonomy, supplied by the corpus seam.
const VERDICTS_JSON: &str = r#"{
 "sessions": [
  {
   "session": "S1",
   "lines": [
    {"utterance_index": 0, "category": "other",
     "fa_mean_score": 0.52, "pitch": "child", "ear": "yes"},
    {"utterance_index": 1, "category": "other",
     "fa_mean_score": 0.41, "pitch": "child", "ear": "no"},
    {"utterance_index": 2, "category": "clock",
     "fa_mean_score": 0.10, "pitch": "ambiguous", "ear": "no"},
    {"utterance_index": 3, "category": "confident",
     "fa_mean_score": 0.05, "pitch": "adult", "ear": "no"},
    {"utterance_index": 5, "category": "medium",
     "fa_mean_score": 0.61, "pitch": "child", "ear": "yes"},
    {"utterance_index": 6, "category": "other",
     "fa_mean_score": null, "pitch": "child", "ear": "yes"}
   ]
  }
 ]
}"#;

struct VerifyFixture {
    _dir: tempfile::TempDir,
    draft: std::path::PathBuf,
    verdicts: std::path::PathBuf,
    out: std::path::PathBuf,
}

fn fixture() -> VerifyFixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let draft = dir.path().join("draft");
    let out = dir.path().join("out");
    std::fs::create_dir_all(&draft).expect("draft dir");
    std::fs::write(draft.join("S1.cha"), DRAFT_CHA).expect("write draft");
    let verdicts = dir.path().join("verdicts.json");
    std::fs::write(&verdicts, VERDICTS_JSON).expect("write verdicts");
    VerifyFixture {
        _dir: dir,
        draft,
        verdicts,
        out,
    }
}

fn run_merge_verify(fx: &VerifyFixture) -> assert_cmd::assert::Assert {
    cmd()
        .args([
            "merge-verify",
            "--draft",
            fx.draft.to_str().expect("utf8 path"),
            "--verdicts",
            fx.verdicts.to_str().expect("utf8 path"),
            "--out",
            fx.out.to_str().expect("utf8 path"),
        ])
        .assert()
}

/// The command exists, succeeds on the fixture, and writes the outputs.
#[test]
fn merge_verify_runs_and_writes_outputs() {
    let fx = fixture();
    run_merge_verify(&fx).success();
    assert!(fx.out.join("S1.cha").is_file(), "rewritten draft missing");
    assert!(
        fx.out.join("review-queue.json").is_file(),
        "review queue missing"
    );
}

/// AUTO_TRUST: the verify flag is rewritten to a machine-verified
/// provenance note carrying the three signals; never deleted.
#[test]
fn auto_trust_rewrites_flag_to_provenance_note() {
    let fx = fixture();
    run_merge_verify(&fx).success();
    let out = std::fs::read_to_string(fx.out.join("S1.cha")).expect("read output");
    let first_com = out
        .lines()
        .find(|l| l.starts_with("%com:"))
        .expect("first %com present");
    assert!(
        first_com.contains("machine-verified"),
        "promoted flag must become a provenance note, got: {first_com}"
    );
    for needle in ["fa=0.52", "pitch=child", "ear=yes"] {
        assert!(
            first_com.contains(needle),
            "provenance note must carry {needle}, got: {first_com}"
        );
    }
    assert!(
        !first_com.contains("verify:"),
        "promoted line must not still read as a verify flag: {first_com}"
    );
}

/// A %com tier carrying a human transcriber note BEFORE the flag (the
/// merge writer appends its flag to an existing comment as
/// "<human> ; <prefix>: <flag>") still gets its flag rewritten on
/// AUTO_TRUST, and the human note survives verbatim. Regression: four
/// bundle utterances with exactly this shape were silently skipped by
/// a starts-with predicate (found 2026-07-17).
#[test]
fn human_prefixed_flag_rewritten_and_note_preserved() {
    let fx = fixture();
    run_merge_verify(&fx).success();
    let out = std::fs::read_to_string(fx.out.join("S1.cha")).expect("read output");
    let mixed = out
        .lines()
        .find(|l| l.contains("GONNA HAD ONE SYLLABLE"))
        .expect("human-prefixed %com present");
    assert!(
        mixed.contains("GONNA HAD ONE SYLLABLE . ; machine-verified"),
        "human note must survive verbatim ahead of the provenance note, got: {mixed}"
    );
    assert!(
        !mixed.contains("verify:"),
        "promoted flag segment must be rewritten, got: {mixed}"
    );
}

/// An UNSCORABLE line (FA could not align: fa_mean_score is null in
/// the verdicts) still promotes on the composed rule (FA is
/// ordering-only, never a gate) and its provenance note says fa=n/a
/// rather than fabricating a number.
#[test]
fn unscorable_fa_line_promotes_with_na_note() {
    let fx = fixture();
    run_merge_verify(&fx).success();
    let out = std::fs::read_to_string(fx.out.join("S1.cha")).expect("read output");
    let line = out
        .lines()
        .filter(|l| l.starts_with("%com:"))
        .find(|l| l.contains("fa=n/a"))
        .expect("unscorable line's provenance note with fa=n/a present");
    assert!(
        line.contains("machine-verified") && line.contains("ear=yes"),
        "unscorable line must still promote on pitch+ear: {line}"
    );
}

/// Attached commas and wrapped main tiers retain structure, not formatting.
#[test]
fn normalized_output_preserves_attached_comma_structure() {
    let fx = fixture();
    run_merge_verify(&fx).success();
    let out = std::fs::read_to_string(fx.out.join("S1.cha")).expect("read output");
    assert!(
        out.contains("I know how , see ?"),
        "the typed writer normalizes the comma, got: {}",
        out.lines()
            .find(|l| l.contains("I know how"))
            .unwrap_or("<line missing>")
    );
    assert_main_structure_preserved(DRAFT_CHA, &out);
}

/// REVIEW and HOLD flags pass through unchanged; the review queue
/// contains exactly the REVIEW line and the demoted line.
#[test]
fn review_and_hold_flags_unchanged_and_queue_is_exact() {
    let fx = fixture();
    run_merge_verify(&fx).success();
    let out = std::fs::read_to_string(fx.out.join("S1.cha")).expect("read output");
    let coms: Vec<&str> = out.lines().filter(|l| l.starts_with("%com:")).collect();
    assert!(
        coms.iter()
            .any(|l| l.contains("verify: placement (diarization mislabel)")),
        "the REVIEW line's flag must be unchanged; %com lines: {coms:?}"
    );
    assert!(
        coms.iter()
            .any(|l| l.contains("verify: placement (interpolated)")),
        "the HOLD (clock) line's flag must be unchanged; %com lines: {coms:?}"
    );

    let queue: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fx.out.join("review-queue.json")).expect("read queue"),
    )
    .expect("queue parses");
    let entries = queue["entries"].as_array().expect("entries array");
    let indices: Vec<u64> = entries
        .iter()
        .map(|e| e["utterance_index"].as_u64().expect("index"))
        .collect();
    assert!(
        indices.contains(&1),
        "REVIEW line (index 1) must be queued, got {indices:?}"
    );
    assert!(
        indices.contains(&3),
        "demoted line (index 3) must be queued, got {indices:?}"
    );
    assert!(
        !indices.contains(&0) && !indices.contains(&2),
        "promoted and HOLD lines must not be queued, got {indices:?}"
    );
}

/// Demotion: the previously-confident line gains a review flag; its
/// main tier is untouched.
#[test]
fn demotion_adds_review_flag_without_moving_text() {
    let fx = fixture();
    run_merge_verify(&fx).success();
    let out = std::fs::read_to_string(fx.out.join("S1.cha")).expect("read output");
    let par_pos = out.find("*PAR:\tsure .").expect("PAR main tier preserved");
    let tail = &out[par_pos..];
    let next_com = tail
        .lines()
        .skip(1)
        .find(|l| l.starts_with("%com:"))
        .expect("demoted line must gain a %com flag");
    assert!(
        next_com.contains("review"),
        "demotion flag must route to review, got: {next_com}"
    );
}

/// Use the grammar and model, never a second main-tier scanner.
#[test]
fn main_tiers_are_structurally_identical() {
    let fx = fixture();
    run_merge_verify(&fx).success();
    let out = std::fs::read_to_string(fx.out.join("S1.cha")).expect("read output");
    assert_main_structure_preserved(DRAFT_CHA, &out);
}

fn admitted(chat: &str) -> talkbank_model::validation::ValidChatFile {
    let parser = batchalign_transform::parse::TreeSitterParser::new().expect("parser");
    batchalign_transform::parse_source_with_parser(&parser, chat)
        .admit(
            talkbank_model::model::TranscriptName::Anonymous,
            &talkbank_model::NullErrorSink,
        )
        .expect("complete CHAT admission")
        .into_valid_file()
}

fn assert_main_structure_preserved(before: &str, after: &str) {
    use talkbank_model::SemanticEq;
    use talkbank_model::model::Line;
    let before = admitted(before);
    let after = admitted(after);
    let mains = |file: &talkbank_model::validation::ValidChatFile| {
        file.document()
            .lines
            .iter()
            .filter_map(|line| match line {
                Line::Utterance(u) => Some(u.main.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    assert!(
        mains(&before).semantic_eq(&mains(&after)),
        "typed main tiers changed"
    );
}

/// Valid parsing alone is not admission; every retained tier is checked.
#[test]
fn invalid_sources_are_refused_without_output() {
    let cases = [
        DRAFT_CHA.replace("@Languages:\teng\n", ""),
        DRAFT_CHA.replace("yes .", "yes"),
        DRAFT_CHA.replace(
            "%com:\tverify: placement (diarization mislabel)",
            "%wor:\t[ .",
        ),
        DRAFT_CHA.replacen(
            "%com:\tverify: placement (diarization mislabel)",
            "%mor:\tx|yes x|extra .",
            1,
        ),
        DRAFT_CHA.replacen(
            "%com:\tverify: placement (diarization mislabel)",
            "%gra:\t1|999|ROOT",
            1,
        ),
        DRAFT_CHA.replace("S1, audio", "different, audio"),
    ];
    for source in cases {
        let fx = fixture();
        std::fs::write(fx.draft.join("S1.cha"), source).expect("source");
        run_merge_verify(&fx).failure();
        assert!(!fx.out.exists(), "refusal must precede output creation");
    }
}

#[test]
fn conflicting_and_identical_duplicate_verdicts_are_refused() {
    for conflict in [false, true] {
        let fx = fixture();
        let mut doc: serde_json::Value = serde_json::from_str(VERDICTS_JSON).expect("wire");
        let lines = doc["sessions"][0]["lines"].as_array_mut().expect("lines");
        let mut duplicate = lines[0].clone();
        if conflict {
            duplicate["ear"] = serde_json::json!("no");
        }
        lines.push(duplicate);
        std::fs::write(&fx.verdicts, doc.to_string()).expect("verdicts");
        run_merge_verify(&fx)
            .failure()
            .stderr(predicates::str::contains("duplicate verdict"));
        assert!(!fx.out.exists());
    }
}

#[test]
fn duplicate_sessions_and_unsafe_stems_are_refused() {
    for (first, second) in [("S1", "S1"), ("S1", "s1"), ("é", "e\u{301}")] {
        let fx = fixture();
        if first != "S1" {
            std::fs::rename(
                fx.draft.join("S1.cha"),
                fx.draft.join(format!("{first}.cha")),
            )
            .expect("rename");
            std::fs::write(
                fx.draft.join(format!("{first}.cha")),
                DRAFT_CHA.replace("@Media:\tS1, audio", &format!("@Media:\t{first}, audio")),
            )
            .expect("source");
        }
        let mut doc: serde_json::Value = serde_json::from_str(VERDICTS_JSON).expect("wire");
        doc["sessions"][0]["session"] = serde_json::json!(first);
        let mut duplicate = doc["sessions"][0].clone();
        duplicate["session"] = serde_json::json!(second);
        doc["sessions"]
            .as_array_mut()
            .expect("sessions")
            .push(duplicate);
        std::fs::write(&fx.verdicts, doc.to_string()).expect("verdicts");
        run_merge_verify(&fx)
            .failure()
            .stderr(predicates::str::contains("duplicate session"));
        assert!(!fx.out.exists());
    }
    for stem in [
        "",
        ".",
        "..",
        "../S1",
        "/S1",
        "nested/S1",
        "nested\\S1",
        "C:S1",
    ] {
        let fx = fixture();
        let doc = serde_json::json!({"sessions": [{"session": stem, "lines": []}]});
        std::fs::write(&fx.verdicts, doc.to_string()).expect("verdicts");
        run_merge_verify(&fx)
            .failure()
            .stderr(predicates::str::contains("unsafe session"));
        assert!(!fx.out.exists());
    }
}

#[test]
fn ordinal_and_later_session_refusal_do_not_write_earlier_sessions() {
    let fx = fixture();
    let mut doc: serde_json::Value = serde_json::from_str(VERDICTS_JSON).expect("wire");
    doc["sessions"][0]["lines"][0]["utterance_index"] = serde_json::json!(8);
    std::fs::write(&fx.verdicts, doc.to_string()).expect("verdicts");
    run_merge_verify(&fx)
        .failure()
        .stderr(predicates::str::contains("only 8"));
    assert!(!fx.out.exists());

    let fx = fixture();
    let mut doc: serde_json::Value = serde_json::from_str(VERDICTS_JSON).expect("wire");
    doc["sessions"]
        .as_array_mut()
        .expect("sessions")
        .push(serde_json::json!({"session": "S2", "lines": []}));
    std::fs::write(
        fx.draft.join("S2.cha"),
        DRAFT_CHA.replace("@Languages:\teng\n", ""),
    )
    .expect("invalid later source");
    std::fs::write(&fx.verdicts, doc.to_string()).expect("verdicts");
    run_merge_verify(&fx).failure();
    assert!(
        !fx.out.exists(),
        "all sessions must be admitted before any writes"
    );
}

#[test]
fn promoted_comments_retain_structured_segments_and_plain_comments() {
    use talkbank_model::model::{BulletContentSegment, DependentTier, Line};
    let fx = fixture();
    let source = DRAFT_CHA.replacen("%com:\tverify: placement (diarization mislabel)",
        "%com:\tcontributor note \u{15}1000_2000\u{15} ; verify: placement\n\tcontinued flag text \u{15}%pic:\"image.jpg\"\u{15} trailing contributor note", 1);
    std::fs::write(fx.draft.join("S1.cha"), &source).expect("source");
    run_merge_verify(&fx).success();
    let output = std::fs::read_to_string(fx.out.join("S1.cha")).expect("output");
    assert_main_structure_preserved(&source, &output);
    let file = admitted(&output);
    let first = file
        .document()
        .lines
        .iter()
        .find_map(|line| match line {
            Line::Utterance(u) => Some(u),
            _ => None,
        })
        .expect("utterance");
    let comment = first
        .dependent_tiers
        .iter()
        .find_map(|entry| match &entry.tier {
            DependentTier::Com(com) => Some(com),
            _ => None,
        })
        .expect("comment");
    assert!(
        comment
            .content
            .segments
            .iter()
            .any(|s| matches!(s, BulletContentSegment::Bullet(_)))
    );
    assert!(
        comment
            .content
            .segments
            .iter()
            .any(|s| matches!(s, BulletContentSegment::Picture(_)))
    );
    assert!(output.contains("contributor note"));
    assert!(output.contains("trailing contributor note"));
    assert!(output.contains("machine-verified"));
}

#[test]
fn plain_contributor_comment_is_not_promoted_or_dropped() {
    let fx = fixture();
    let source = DRAFT_CHA.replacen(
        "%com:\tverify: placement (diarization mislabel)",
        "%com:\ta plain contributor comment",
        1,
    );
    std::fs::write(fx.draft.join("S1.cha"), &source).expect("source");
    run_merge_verify(&fx).success();
    let output = std::fs::read_to_string(fx.out.join("S1.cha")).expect("output");
    assert!(output.contains("%com:\ta plain contributor comment"));
    assert_main_structure_preserved(&source, &output);
}

#[test]
fn review_queue_wire_is_deterministic() {
    let fx = fixture();
    run_merge_verify(&fx).success();
    let queue = std::fs::read_to_string(fx.out.join("review-queue.json")).expect("queue");
    run_merge_verify(&fx).success();
    assert_eq!(
        queue,
        std::fs::read_to_string(fx.out.join("review-queue.json")).expect("second queue")
    );
    insta::assert_json_snapshot!(serde_json::from_str::<serde_json::Value>(&queue).expect("wire"), @r#"
    {
      "entries": [
        {
          "category": "other",
          "ear": "no",
          "fa_mean_score": 0.41,
          "pitch": "child",
          "session": "S1",
          "tier": "review",
          "utterance_index": 1
        },
        {
          "category": "confident",
          "ear": "no",
          "fa_mean_score": 0.05,
          "pitch": "adult",
          "session": "S1",
          "tier": "demote",
          "utterance_index": 3
        }
      ]
    }
    "#);
}
