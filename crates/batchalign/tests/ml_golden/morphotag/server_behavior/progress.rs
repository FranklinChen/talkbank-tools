use super::super::helpers::repeated_chat;
use crate::common::{LiveServerJobClient, require_live_server};
use batchalign::api::{FilePayload, FileProgressStage, JobStatus, LanguageSpec, ReleasedCommand};
use batchalign::options::{CommandOptions, CommonOptions, MorphotagOptions};
use batchalign::worker::InferTask;

/// Verify an operator can see a long morphotag file ADVANCING, not just
/// "Analyzing" with no numbers.
///
/// **Rewritten 2026-07-29, and the reason matters.** This test used to assert
/// that a `--batch-window` job reported progress on `FileProgressStage::Parsing`.
/// Two things it depended on do not exist:
///
/// 1. `FileProgressStage::Parsing` has NO producer anywhere in the crate. It is
///    declared in the API enum and referenced only by the TUI's colour map, so
///    the assertion could never have passed.
/// 2. `batch_window` was parsed from `--batch-window`, stored, serialized, and
///    read by NOTHING in the execution path: the windowing it named belonged to
///    the cross-file pooled batching that per-file fanout replaced. The flag was
///    DELETED on 2026-07-30 rather than left as an operator-visible no-op.
///
/// What now exists, and what this asserts: per-file utterance counts published
/// from the batch-progress ledger's per-source projection, on the ordinary
/// file-progress channel every other command uses. That is the signal an
/// operator watching one long file actually needs.
#[tokio::test]
async fn morphotag_job_reports_per_file_utterance_progress() {
    let server = require_live_server(
        InferTask::Morphosyntax,
        "Server does not support morphosyntax infer",
    )
    .await;
    let jobs = LiveServerJobClient::from_session(&server);

    // One file, big enough that its inference certainly spans several progress
    // events: the backend throttles to one per second per request and always
    // emits the final one, so a file finishing inside a second would report only
    // its completion. 1,500 utterances takes tens of seconds.
    let files = vec![FilePayload {
        filename: "long_file.cha".into(),
        content: repeated_chat("eng", "PAR", "the progress test runs", 1500),
    }];

    let initial = jobs
        .submit_content_job(
            ReleasedCommand::Morphotag,
            LanguageSpec::PerFile,
            files,
            CommandOptions::Morphotag(MorphotagOptions {
                common: CommonOptions {
                    override_media_cache: true,
                    ..CommonOptions::default()
                },

                ..Default::default()
            }),
        )
        .await;

    let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(180);
    let mut max_current = 0;
    let mut max_total = 0;

    loop {
        let info = jobs.job_info(&initial.job_id).await;
        for file in &info.file_statuses {
            if file.progress_stage == Some(FileProgressStage::Analyzing) {
                max_current = max_current.max(file.progress_current.unwrap_or(0));
                max_total = max_total.max(file.progress_total.unwrap_or(0));
            }
        }
        if info.status.is_terminal() || tokio::time::Instant::now() >= deadline {
            break;
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }

    let final_info = jobs.poll_done(&initial.job_id).await;
    assert_eq!(
        final_info.status,
        JobStatus::Completed,
        "morphotag job should complete"
    );
    // This is now the only end-to-end assertion for the feature, so it asserts
    // presence rather than just coherence. The deterministic half lives in
    // `execution::morphotag::progress`'s own tests.
    assert!(
        max_total > 0,
        "a file being analyzed must report a denominator, not just a stage label; \
         observed current={max_current} total={max_total}"
    );
    assert!(
        max_current > 0,
        "utterance progress must advance past zero; \
         observed current={max_current} total={max_total}"
    );
    assert!(
        max_current <= max_total,
        "completed utterances must never exceed the total (the 453/274 defect); \
         observed current={max_current} total={max_total}"
    );
}
