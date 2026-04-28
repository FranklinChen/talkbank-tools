use std::sync::Arc;
use std::time::Duration;

use tokio::task::JoinHandle;

use crate::api::JobId;
use crate::runner::util::RunnerEventSink;
use crate::runner::util::batch_progress::BatchInferProgress;

pub(super) struct BatchProgressReporter {
    tx: tokio::sync::mpsc::Sender<crate::types::worker_v2::ProgressEventV2>,
    drain_handle: JoinHandle<()>,
}

impl BatchProgressReporter {
    pub(super) fn spawn(job_id: JobId, sink: Arc<dyn RunnerEventSink>) -> Self {
        let (tx, mut rx) =
            tokio::sync::mpsc::channel::<crate::types::worker_v2::ProgressEventV2>(64);
        let drain_handle = tokio::spawn(async move {
            let mut progress = BatchInferProgress::new();
            let mut last_publish = tokio::time::Instant::now();

            loop {
                match tokio::time::timeout(Duration::from_secs(120), rx.recv()).await {
                    Ok(Some(event)) => {
                        let lang = &event.stage;
                        if !progress.language_groups.contains_key(lang) {
                            progress.register_group(lang, event.total as u64);
                        }
                        progress.update_group(lang, event.completed as u64);

                        let now = tokio::time::Instant::now();
                        if now.duration_since(last_publish).as_secs() >= 2 {
                            last_publish = now;
                            sink.set_batch_progress(&job_id, &progress).await;
                        }
                    }
                    Ok(None) => break,
                    Err(_) => {
                        sink.set_batch_progress(&job_id, &progress).await;
                    }
                }
            }

            sink.set_batch_progress(&job_id, &progress).await;
        });

        Self { tx, drain_handle }
    }

    pub(super) fn sender(
        &self,
    ) -> tokio::sync::mpsc::Sender<crate::types::worker_v2::ProgressEventV2> {
        self.tx.clone()
    }

    pub(super) async fn finish(self) {
        drop(self.tx);
        let _ = self.drain_handle.await;
    }
}
