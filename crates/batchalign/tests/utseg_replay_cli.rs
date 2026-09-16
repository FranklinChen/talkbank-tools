//! Binary-boundary tests for offline utterance-segmentation replay.
//!
//! These invoke the executable an operator uses. No server, model, media, or
//! provider credential is involved.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::todo,
    clippy::unimplemented
)]

use std::collections::HashMap;

use crate::cli_common;

use batchalign_transform::parse::{TreeSitterParser, parse_lenient};
use batchalign_transform::serialize::to_chat_string;
use batchalign_transform::utseg::apply_utseg_results;
use cli_common::CliHarness;

const INPUT: &str = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI|3;|male|||Target_Child|||\n*CHI:\thello there how are you .\n@End\n";

const WORDS: [&str; 5] = ["hello", "there", "how", "are", "you"];

/// Evidence for `INPUT` recording `assignments`, in the shape a run writes.
fn evidence_json(assignments: &[usize]) -> String {
    serde_json::json!({
        "schema_version": 4,
        "phase": "post_chat",
        "language": "eng",
        "items": [{
            "item_ordinal": 0,
            "words": WORDS,
            "text": WORDS.join(" "),
            "prediction": {
                "source": "unobserved_assignments",
                "assignments": assignments,
            }
        }]
    })
    .to_string()
}

/// The document a run applying `assignments` to `INPUT` would have written.
fn retained_output(assignments: Vec<usize>) -> String {
    let parser = TreeSitterParser::new().expect("CHAT grammar");
    let (mut chat, errors) = parse_lenient(&parser, INPUT);
    assert!(errors.is_empty(), "fixture must parse cleanly");
    let mut map = HashMap::new();
    map.insert(0, assignments);
    apply_utseg_results(&mut chat, &map);
    to_chat_string(&chat)
}

/// Write the three artifacts of one retained run into `harness`.
fn artifacts(
    harness: &CliHarness,
    evidence: &str,
    retained: &str,
) -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
    let input = harness.home_dir().join("input.cha");
    let evidence_path = harness
        .home_dir()
        .join("input_post_chat_utseg_evidence.json");
    let output = harness.home_dir().join("output.cha");
    std::fs::write(&input, INPUT).expect("write input");
    std::fs::write(&evidence_path, evidence).expect("write evidence");
    std::fs::write(&output, retained).expect("write retained output");
    (input, evidence_path, output)
}

#[test]
fn offline_utseg_replay_reports_a_reproduced_run_and_leaves_artifacts_alone() {
    let harness = CliHarness::new();
    let split = vec![0, 0, 1, 1, 1];
    let retained = retained_output(split.clone());
    let (input, evidence, output) = artifacts(&harness, &evidence_json(&split), &retained);

    let result = harness
        .cmd()
        .args(["eval", "utseg-replay", "post-chat"])
        .arg("--input-chat")
        .arg(&input)
        .arg("--evidence")
        .arg(&evidence)
        .arg("--output-chat")
        .arg(&output)
        .assert()
        .success();

    let stdout = String::from_utf8(result.get_output().stdout.clone()).expect("report is UTF-8");
    let report: serde_json::Value = serde_json::from_str(&stdout).expect("report JSON");
    assert_eq!(report["outcome"]["kind"], "reproduced");
    assert_eq!(report["bound_items"], 1);
    assert_eq!(report["pass"]["kind"], "post_chat");
    assert_eq!(report["pass"]["utterances_after"], 2);

    assert_eq!(
        std::fs::read_to_string(&input).expect("read input"),
        INPUT,
        "a replay must never modify the artifacts it reads"
    );
    assert_eq!(
        std::fs::read_to_string(&output).expect("read retained output"),
        retained
    );
}

/// A run that does not reproduce exits nonzero without claiming a usage or
/// environment failure, so a script can tell "the answer was no" from "the
/// command was wrong".
#[test]
fn offline_utseg_replay_exits_one_when_the_retained_output_is_not_reproduced() {
    let harness = CliHarness::new();
    let retained = retained_output(vec![0, 0, 1, 1, 1]);
    let (input, evidence, output) =
        artifacts(&harness, &evidence_json(&[0, 0, 0, 0, 0]), &retained);

    harness
        .cmd()
        .args(["eval", "utseg-replay", "post-chat"])
        .arg("--input-chat")
        .arg(&input)
        .arg("--evidence")
        .arg(&evidence)
        .arg("--output-chat")
        .arg(&output)
        .assert()
        .failure()
        .code(1);
}
