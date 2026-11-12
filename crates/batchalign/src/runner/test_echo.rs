//! Test-echo dispatch: reads input files and writes them back as output,
//! exercising the full file lifecycle (progress tracking, result recording)
//! without any ML inference. Used by worker integration tests.

use crate::api::ContentType;
use crate::recipe_runner::runtime::{
    ChatOutputTarget, result_display_path_for_command, write_text_output_artifact,
};
use crate::scheduling::{FailureCategory, WorkUnitKind};
use crate::store::RunnerJobSnapshot;

use super::util::{FileRunTracker, FileStage, RunnerEventSink};

/// Dispatch each file in test-echo mode: read input, write it back as output.
pub(super) async fn dispatch_test_echo_files(
    job: &RunnerJobSnapshot,
    sink: &dyn RunnerEventSink,
    file_list: &[crate::store::PendingJobFile],
    test_delay_ms: u64,
) {
    for file in file_list {
        if job.cancel_token.is_cancelled() {
            break;
        }

        // Honor the pool's test-only pacing knob (`PoolConfig::test_delay_ms`)
        // even though echo mode never reaches a worker: integration tests
        // use it to keep a job observably mid-run (e.g. the restart-handoff
        // race) where instant echo would close the window.
        if test_delay_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(test_delay_ms)).await;
        }

        let filename = file.filename().as_ref();
        let lifecycle = FileRunTracker::new(sink, filename);
        lifecycle
            .begin_first_attempt(WorkUnitKind::FileProcess, FileStage::Processing)
            .await;

        let result_display_path = match result_display_path_for_command(
            job.dispatch.command,
            &job.dispatch.options,
            filename,
        ) {
            Ok(path) => path,
            Err(error) => {
                lifecycle
                    .fail(
                        &error.to_string(),
                        crate::scheduling::FailureCategory::System,
                    )
                    .await;
                continue;
            }
        };

        let output_text = match file.kind() {
            crate::store::InputKind::Chat => {
                let read_path = file.input_path(&job.filesystem.staging_dir);
                match tokio::fs::read_to_string(&read_path).await {
                    Ok(content) => content,
                    Err(error) => {
                        let err_msg =
                            format!("Failed to read input for test-echo dispatch: {error}");
                        lifecycle
                            .fail(&err_msg, FailureCategory::InputMissing)
                            .await;
                        continue;
                    }
                }
            }
            crate::store::InputKind::Media => "@UTF8\n@Begin\n@End\n".to_string(),
        };

        let target = ChatOutputTarget::new(&job.filesystem, file.location(), &result_display_path);
        if let Err(error) = write_text_output_artifact(&target, &output_text).await {
            let err_msg = format!("Failed to write test-echo output: {error}");
            lifecycle.fail(&err_msg, FailureCategory::Validation).await;
            continue;
        }

        lifecycle
            .complete_with_result(result_display_path, ContentType::Chat)
            .await;
    }
}
