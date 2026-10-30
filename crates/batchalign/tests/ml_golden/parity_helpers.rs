use crate::common::{
    assert_ba2_parity, assert_completed_without_errors, load_ba2_golden, load_parity_fixture,
    require_live_direct_warmed, submit_and_complete_direct,
};
use batchalign::api::{FilePayload, JobStatus, ReleasedCommand};
use batchalign::options::CommandOptions;
use batchalign::worker::InferTask;

/// What batchalign2's Jan 9 baseline offers to compare a parity run against.
///
/// Declared by each test rather than discovered from the filesystem: a golden
/// that a test expects and cannot find is a repository defect, and used to
/// make the test quietly assert less.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Ba2Baseline {
    /// A committed `ba2_reference` golden exists; the output must match it.
    Golden,
    /// The committed BA2 baseline has no output for this command (no
    /// `batchalign/tests/golden/ba2_reference/<command>/`), so only the
    /// structural checks apply.
    NoBa2Golden,
}

/// Submit a curated parity fixture through BA3, check it structurally, and
/// compare it to the BA2 Jan 9 golden when `baseline` says one exists.
pub(super) async fn run_parity_test(
    command: ReleasedCommand,
    task: InferTask,
    fixture_name: &str,
    lang: &str,
    options: CommandOptions,
    baseline: Ba2Baseline,
) {
    let session = require_live_direct_warmed(
        task,
        command,
        lang,
        &format!("Direct session does not support {task:?} infer"),
    )
    .await;

    let input = load_parity_fixture(fixture_name);

    let files = vec![FilePayload {
        filename: format!("{fixture_name}.cha").into(),
        content: input,
    }];

    let (info, results) = submit_and_complete_direct(&session, command, lang, files, options).await;

    if info.status == JobStatus::Failed {
        panic!(
            "PREREQUISITE MISSING: {command} {fixture_name} ({lang}) failed (model likely not downloaded)"
        )
    }

    assert_completed_without_errors(&format!("{command}_{fixture_name}"), &info, &results);
    assert_eq!(results.len(), 1);

    let output = results[0].content.as_text().expect("text command result");
    match baseline {
        Ba2Baseline::Golden => {
            let golden = load_ba2_golden(command.as_ref(), fixture_name);
            assert_ba2_parity(&format!("{command}_{fixture_name}"), output, &golden);
        }
        Ba2Baseline::NoBa2Golden => {}
    }
}
