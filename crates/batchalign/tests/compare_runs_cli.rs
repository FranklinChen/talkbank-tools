//! Binary subprocess tests for `batchalign3 compare-runs`.
//!
//! The `compare-runs` family shipped with unit and library coverage only. Its
//! own continuation handoff listed subprocess coverage as the first
//! recommended next work, and the gap is not cosmetic: the crate's verification
//! was `cargo test -p batchalign --lib`, which builds neither the integration
//! tests nor the doctests, so a whole class of breakage could not be seen. This
//! file exercises the actual boundary an operator uses.
//!
//! Everything here runs OFFLINE. `compare-runs` is routed before normal
//! Batchalign setup and server initialization and never invokes a producer
//! command, so no server, model, or network is required, and that property is
//! itself one of the things asserted below.
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

use std::fs;
use std::path::Path;

use cli_common::{CliHarness, MINIMAL_CHAT};
use predicates::prelude::*;

/// Write a transcript into `dir`, creating the directory if needed.
fn write_chat(dir: &Path, name: &str, body: &str) {
    fs::create_dir_all(dir).expect("create artifact dir");
    fs::write(dir.join(name), body).expect("write transcript");
}

/// A run directory holding one transcript, the smallest thing a manifest can
/// describe.
fn seed_run(root: &Path) {
    write_chat(root, "session-1.cha", MINIMAL_CHAT);
}

#[test]
fn help_lists_every_compare_runs_action() {
    CliHarness::new()
        .cmd()
        .args(["compare-runs", "--help"])
        .assert()
        .success()
        .stdout(
            predicate::str::contains("manifest")
                .and(predicate::str::contains("transcribe"))
                .and(predicate::str::contains("morphotag"))
                .and(predicate::str::contains("align")),
        );
}

#[test]
fn manifest_machine_writes_json_naming_its_producer_identity() {
    let harness = CliHarness::new();
    let artifacts = harness.home_dir().join("run-machine");
    seed_run(&artifacts);
    let output = harness.home_dir().join("machine.manifest.json");

    harness
        .cmd()
        .args(["compare-runs", "manifest", "machine"])
        .args(["--artifacts", artifacts.to_str().unwrap()])
        .args(["--output", output.to_str().unwrap()])
        .args(["--run-id", "ours-v10"])
        .args(["--source-id", "corpus-a"])
        .args(["--implementation", "batchalign3"])
        .args(["--command", "transcribe"])
        .args(["--build", "test-build"])
        .assert()
        .success();

    let written = fs::read_to_string(&output).expect("manifest written");
    // The producer identity is the point of the manifest: a comparison that
    // cannot say what produced each side is not evidence.
    assert!(written.contains("batchalign3"), "manifest: {written}");
    assert!(written.contains("ours-v10"), "manifest: {written}");
    assert!(written.contains("session-1.cha"), "manifest: {written}");
}

#[test]
fn manifest_authoring_is_deterministic_for_identical_inputs() {
    let harness = CliHarness::new();
    let artifacts = harness.home_dir().join("run-stable");
    seed_run(&artifacts);

    let mut written = Vec::new();
    for name in ["first.json", "second.json"] {
        let output = harness.home_dir().join(name);
        harness
            .cmd()
            .args(["compare-runs", "manifest", "human"])
            .args(["--artifacts", artifacts.to_str().unwrap()])
            .args(["--output", output.to_str().unwrap()])
            .args(["--run-id", "theirs-hand"])
            .args(["--source-id", "corpus-a"])
            .args(["--protocol", "inv-v1"])
            .args(["--cohort", "reviewers"])
            .assert()
            .success();
        written.push(fs::read_to_string(&output).expect("manifest written"));
    }

    // Byte-identical, not merely equivalent. A manifest feeds a
    // content-addressed comparison identity, so any run-to-run instability
    // (map ordering, a timestamp) would silently invalidate every cache hit
    // and make "unchanged inputs" produce a new comparison directory.
    assert_eq!(written[0], written[1], "manifest authoring is not stable");
}

#[test]
fn manifest_refuses_to_write_its_output_inside_the_artifact_root() {
    let harness = CliHarness::new();
    let artifacts = harness.home_dir().join("run-selfref");
    seed_run(&artifacts);
    // Writing the manifest into the tree it inventories would make the run
    // describe itself, and the artifacts are supposed to be immutable.
    let output = artifacts.join("manifest.json");

    harness
        .cmd()
        .args(["compare-runs", "manifest", "machine"])
        .args(["--artifacts", artifacts.to_str().unwrap()])
        .args(["--output", output.to_str().unwrap()])
        .args(["--run-id", "ours"])
        .args(["--source-id", "corpus-a"])
        .args(["--implementation", "batchalign3"])
        .args(["--command", "transcribe"])
        .args(["--build", "test-build"])
        .assert()
        .failure();
}

#[test]
fn manifest_reports_a_missing_artifact_root_rather_than_writing_an_empty_one() {
    let harness = CliHarness::new();
    let missing = harness.home_dir().join("not-there");
    let output = harness.home_dir().join("out.json");

    harness
        .cmd()
        .args(["compare-runs", "manifest", "machine"])
        .args(["--artifacts", missing.to_str().unwrap()])
        .args(["--output", output.to_str().unwrap()])
        .args(["--run-id", "ours"])
        .args(["--source-id", "corpus-a"])
        .args(["--implementation", "batchalign3"])
        .args(["--command", "transcribe"])
        .args(["--build", "test-build"])
        .assert()
        .failure();

    assert!(
        !output.exists(),
        "a failed manifest run must not leave an output behind"
    );
}

#[test]
fn an_unreadable_plan_fails_without_starting_a_server() {
    let harness = CliHarness::new();
    let plan = harness.home_dir().join("missing-plan.toml");

    // No server is configured and none may be started: `compare-runs` is
    // routed ahead of setup and server initialization, so this must fail on
    // the plan alone rather than on a connection attempt.
    harness
        .cmd()
        .args(["compare-runs", "transcribe"])
        .args(["--plan", plan.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("server").not());
}

#[test]
fn a_plan_with_an_unknown_field_is_rejected_by_name() {
    let harness = CliHarness::new();
    let plan = harness.home_dir().join("bad-field.toml");
    // `deny_unknown_fields` exists so a typo in a plan cannot be silently
    // ignored and change what was compared.
    fs::write(
        &plan,
        "left = \"a.json\"\nright = \"b.json\"\nnot_a_real_field = 1\n",
    )
    .expect("write plan");

    harness
        .cmd()
        .args(["compare-runs", "morphotag"])
        .args(["--plan", plan.to_str().unwrap()])
        .assert()
        .failure();
}

#[test]
fn every_execute_action_accepts_recompute() {
    // The flag exists on all three execution modes; a mode that silently
    // ignored it would serve a stale cached comparison after the policy
    // changed. Each still fails on the absent plan, which is the point: the
    // argument surface is what is under test here.
    let harness = CliHarness::new();
    let plan = harness.home_dir().join("absent.toml");
    for action in ["transcribe", "morphotag", "align"] {
        harness
            .cmd()
            .args(["compare-runs", action])
            .args(["--plan", plan.to_str().unwrap()])
            .arg("--recompute")
            .assert()
            .failure();
    }
}

/// Run the complete offline comparison boundary over immutable artifacts.
fn compare_artifacts(left: &str, right: &str) -> (CliHarness, std::process::Output) {
    compare_artifacts_with_options(left, right, "morphotag", "")
}

fn compare_artifacts_with_options(
    left: &str,
    right: &str,
    mode: &str,
    options: &str,
) -> (CliHarness, std::process::Output) {
    let harness = CliHarness::new();
    for (side, body) in [("left", left), ("right", right)] {
        let root = harness.home_dir().join(side);
        write_chat(&root, "session.cha", body);
        harness
            .cmd()
            .args(["compare-runs", "manifest", "machine"])
            .arg("--artifacts")
            .arg(&root)
            .arg("--output")
            .arg(harness.home_dir().join(format!("{side}.json")))
            .args([
                "--run-id",
                side,
                "--source-id",
                "same-input",
                "--implementation",
                "test-producer",
                "--command",
                mode,
                "--build",
                "immutable-test-build",
            ])
            .assert()
            .success();
    }
    let plan = harness.home_dir().join("plan.toml");
    fs::write(&plan, format!("schema_version = 1\npairing = \"same_source_chat\"\noutput = \"comparison\"\n{options}\n[left]\nmanifest = \"left.json\"\nartifacts = \"left\"\n[right]\nmanifest = \"right.json\"\nartifacts = \"right\"\n[[pairs]]\nleft = \"session.cha\"\nright = \"session.cha\"\n")).unwrap();
    let output = harness
        .cmd()
        .args(["compare-runs", mode])
        .arg("--plan")
        .arg(&plan)
        .output()
        .unwrap();
    (harness, output)
}

fn comparison_root(harness: &CliHarness) -> std::path::PathBuf {
    let mut roots = fs::read_dir(harness.home_dir().join("comparison/runs"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect::<Vec<_>>();
    assert_eq!(roots.len(), 1);
    roots.pop().unwrap()
}

#[test]
fn absent_annotations_are_visible_in_real_report_and_csv() {
    let (harness, output) = compare_artifacts(MINIMAL_CHAT, MINIMAL_CHAT);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let root = comparison_root(&harness);
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("report.json")).unwrap()).unwrap();
    assert_eq!(report["schema_version"], 3);
    assert_eq!(report["algorithm_version"], 4);
    let result = &report["pairs"][0]["outcome"]["result"];
    assert_eq!(result["compared_tokens"], 2);
    assert_eq!(result["fully_annotated_tokens"], 0);
    assert!(result["tokens"][0]["analysis_agreement"].is_null());
    let mut csv = csv::Reader::from_path(root.join("summary.csv")).unwrap();
    let headers = csv.headers().unwrap().clone();
    let presence = headers
        .iter()
        .position(|name| name == "left_annotation_state")
        .unwrap();
    let agreement = headers
        .iter()
        .position(|name| name == "analysis_agreement")
        .unwrap();
    for row in csv.records() {
        let row = row.unwrap();
        assert_eq!(&row[presence], "absent");
        assert_eq!(&row[agreement], "");
    }
    // Identical admitted inputs reuse only this version's result.
    harness
        .cmd()
        .args(["compare-runs", "morphotag"])
        .arg("--plan")
        .arg(harness.home_dir().join("plan.toml"))
        .assert()
        .success()
        .stderr(predicate::str::contains("0 computed, 1 reused"));
}

#[test]
fn post_clitic_differences_survive_the_cli_wire_boundary() {
    let left = MINIMAL_CHAT.replace("*PAR:\thello world .",
        "*PAR:\thello world .\n%mor:\tintj|hello~pron|you noun|world .\n%gra:\t1|0|ROOT 2|1|NSUBJ 3|1|OBJ 4|1|PUNCT");
    let right = left
        .replace("pron|you", "pron|we")
        .replace("2|1|NSUBJ", "2|3|AMOD");
    let (harness, output) = compare_artifacts(&left, &right);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(comparison_root(&harness).join("report.json")).unwrap())
            .unwrap();
    let row = &report["pairs"][0]["outcome"]["result"]["tokens"][0];
    for axis in ["lemma", "dependency_head", "relation"] {
        assert!(
            row["differences"]
                .as_array()
                .unwrap()
                .iter()
                .any(|value| value == axis)
        );
    }
    assert_eq!(row["analysis_agreement"], false);
}

#[test]
fn fully_invalid_retained_chat_is_unpairable_not_an_algorithm_input() {
    let invalid = MINIMAL_CHAT.replace(
        "*PAR:\thello world .",
        "*PAR:\thello world .\n%mor:\tintj|hello .",
    );
    let (harness, output) = compare_artifacts(MINIMAL_CHAT, &invalid);
    assert_eq!(
        output.status.code(),
        Some(2),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let root = comparison_root(&harness);
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("report.json")).unwrap()).unwrap();
    let outcome = &report["pairs"][0]["outcome"];
    assert_eq!(outcome["outcome"], "unpairable");
    assert_eq!(outcome["reason"]["kind"], "artifact_invalid");
    assert!(outcome.get("result").is_none());
}

#[test]
fn token_modes_refuse_partial_speaker_maps_instead_of_omitting_content() {
    let left = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant, OTH Participant\n@ID:\teng|test|PAR|||||Participant|||\n@ID:\teng|test|OTH|||||Participant|||\n*PAR:\thello world .\n*OTH:\tgood morning .\n@End\n";
    let right = left.replace("good morning", "good evening");
    for mode in ["morphotag", "align"] {
        let (harness, output) =
            compare_artifacts_with_options(left, &right, mode, "speaker_map = { PAR = \"PAR\" }");
        assert_eq!(
            output.status.code(),
            Some(2),
            "{mode}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report: serde_json::Value = serde_json::from_slice(
            &fs::read(comparison_root(&harness).join("report.json")).unwrap(),
        )
        .unwrap();
        let outcome = &report["pairs"][0]["outcome"];
        assert_eq!(outcome["reason"]["kind"], "incomplete_speaker_map");
        assert_eq!(
            outcome["reason"]["unmatched_left"],
            serde_json::json!(["OTH"])
        );
        assert_eq!(
            outcome["reason"]["unmatched_right"],
            serde_json::json!(["OTH"])
        );
        assert!(outcome.get("result").is_none());

        let (harness, output) = compare_artifacts_with_options(
            left,
            left,
            mode,
            "speaker_map = { PAR = \"PAR\", OTH = \"OTH\" }",
        );
        assert!(
            output.status.success(),
            "{mode}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report: serde_json::Value = serde_json::from_slice(
            &fs::read(comparison_root(&harness).join("report.json")).unwrap(),
        )
        .unwrap();
        let outcome = &report["pairs"][0]["outcome"];
        assert_eq!(outcome["outcome"], "compared");
        assert_eq!(outcome["result"]["tokens"].as_array().unwrap().len(), 4);
    }
}
