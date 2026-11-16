//! A consuming file queue bounds preparation, inference and writeback together.

use super::text_io::PreparedTextJob;
use super::worker_gateway::{MorphotagGateway, MorphotagRuntimeOptions};
use crate::runner::util::{FileRunTracker, FileStage};
use crate::scheduling::WorkUnitKind;
use crate::text_batch::TextBatchFileResult;
use std::sync::Arc;

mod options;
pub(crate) mod progress;
use progress::BatchProgressReporter;

pub(crate) async fn dispatch_morphotag_job(
    files: PreparedTextJob<MorphotagRuntimeOptions>,
    gateway: Arc<dyn MorphotagGateway>,
) -> Result<(), crate::error::ServerError> {
    let reporter = BatchProgressReporter::spawn(files.host().sink().clone());
    files
        .run_files(|file, options| {
            let gateway = Arc::clone(&gateway);
            let options = options.clone();
            let progress = reporter.port(file.input().filename.clone());
            async move {
                FileRunTracker::new(file.host().sink().as_ref(), file.input().filename.as_ref())
                    .begin_first_attempt(WorkUnitKind::BatchInfer, FileStage::Parsing)
                    .await;
                let result = crate::runner::util::observing_file_waits(
                    file.host().sink(),
                    [file.input().filename.to_string()],
                    gateway.morphotag_single(
                        &file.input().chat_text,
                        file.before(),
                        options,
                        progress.as_ref(),
                        file.cancellation(),
                    ),
                )
                .await;
                let result = match result {
                    Ok(output) => TextBatchFileResult::ok(file.input().filename.clone(), output),
                    Err(error) => TextBatchFileResult::err(
                        file.input().filename.clone(),
                        crate::text_batch::TextWorkflowFileError::from_server_error(&error),
                    ),
                };
                file.complete(vec![result]).await;
            }
        })
        .await;
    reporter.finish().await;
    Ok(())
}
