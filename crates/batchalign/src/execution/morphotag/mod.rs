//! A consuming file queue bounds preparation, inference and writeback together.

use super::text_io::PreparedTextJob;
use super::worker_gateway::{MorphotagGateway, MorphotagRuntimeOptions};
use std::sync::Arc;

mod options;
pub mod progress;
use progress::BatchProgressReporter;

pub async fn dispatch_morphotag_job(
    files: PreparedTextJob<MorphotagRuntimeOptions>,
    gateway: Arc<dyn MorphotagGateway>,
) -> Result<(), crate::error::ServerError> {
    let reporter = BatchProgressReporter::spawn(files.host().sink().clone());
    files
        .run_files(|file| {
            let gateway = Arc::clone(&gateway);
            let progress = reporter.port(file.filename().clone());
            async move { file.submit(gateway.as_ref(), progress).await }
        })
        .await;
    reporter.finish().await;
    Ok(())
}
