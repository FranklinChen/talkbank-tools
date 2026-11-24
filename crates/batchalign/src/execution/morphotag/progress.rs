//! Per-file inference progress reporting for a morphotag job.
//!
//! One reporter per job owns the accounting and the publishing cadence; each
//! per-file task holds a cheap [`BackendProgressPort`] that stamps its own
//! provenance onto every report. Nothing here is shared mutable state: the
//! ledger lives in the drain task and reaches it only over a channel.
//!
//! ```text
//! dispatch_morphotag_job
//!   ├── BatchProgressReporter::spawn(sink) ── owns the ledger
//!   ├── per file: reporter.port(filename) ──▶ MorphosyntaxParams.progress
//!   │                                            └─▶ infer_batch ─▶ per chunk
//!   └── reporter.finish().await                    (drains, publishes, exits)
//! ```
//!
//! # History
//!
//! `15c88de2` (2026-04-28) had a working drain loop in the legacy batched-text
//! dispatch, and a 61-line `BatchProgressReporter` written to replace it that
//! nothing ever constructed. `e8235c13` (2026-05-03) deleted the working loop
//! and left the unused successor behind, so the feature died with a designed
//! replacement sitting dead on arrival: the "frozen mid-sentence" state the
//! rearchitecture charter describes. This is that successor, built for the
//! per-file-fanout architecture that replaced the pooled one, with the
//! aggregation defect fixed (see [`crate::runner::util::batch_progress`]).

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::api::{DisplayPath, LanguageCode3, UtteranceCount};
use crate::runner::util::batch_progress::{
    BackendProgress, BatchChunkIndex, BatchProgressLedger, ProgressReport, SourceProgress,
};
use crate::runner::util::{FileStage, RunnerEventSink};
use crate::types::worker_v2::ProgressEventV2;

/// How often a changed snapshot is published.
///
/// The Python backend already throttles to at most one event per second per
/// request, so this bounds the store writes when many files report at once
/// rather than the wire traffic.
const PUBLISH_INTERVAL: Duration = Duration::from_secs(2);

/// How long without any new report before the last snapshot is republished.
///
/// A republish is how an operator tells "stalled" from "finished": the numbers
/// stay put but keep arriving. This is the heartbeat the 2026-04 code called
/// its stall detector, kept at the same 120s.
const STALL_REPUBLISH_INTERVAL: Duration = Duration::from_secs(120);

/// Channel depth for backend reports.
///
/// Reports are throttled per request and a job's chunk count is bounded by the
/// worker pool, so this is generous; a full channel drops the least useful
/// event (see [`BackendProgressPort::report`]) rather than blocking inference.
const REPORT_CHANNEL_DEPTH: usize = 256;

/// Owns one job's batch-progress accounting and publishing.
pub struct BatchProgressReporter {
    /// Kept so [`Self::port`] can hand out clones.
    tx: Option<mpsc::Sender<ProgressReport>>,
    /// Explicit shutdown signal for the drain task.
    ///
    /// Shutdown is NOT inferred from the channel closing. Every port holds a
    /// sender clone, so "the channel closed" means "no port survives anywhere",
    /// and a caller holding one port one line too long would make `finish()`
    /// wait forever. A hang is the worst failure mode available here, so the
    /// drain gets told to stop rather than left to deduce it. Caught by
    /// `finish_does_not_wait_for_outstanding_ports`.
    shutdown: CancellationToken,
    drain: tokio::task::JoinHandle<()>,
}

impl BatchProgressReporter {
    /// Start a reporter for one job.
    pub(crate) fn spawn(sink: Arc<dyn RunnerEventSink>) -> Self {
        let (tx, rx) = mpsc::channel::<ProgressReport>(REPORT_CHANNEL_DEPTH);
        let shutdown = CancellationToken::new();
        let drain = tokio::spawn(drain_reports(sink, rx, shutdown.clone()));
        Self {
            tx: Some(tx),
            shutdown,
            drain,
        }
    }

    /// A port for one input file, or `None` once [`Self::finish`] has run.
    pub(crate) fn port(&self, source_id: DisplayPath) -> Option<BackendProgressPort> {
        self.tx.as_ref().map(|tx| BackendProgressPort {
            source_id,
            tx: tx.clone(),
        })
    }

    /// Close the channel and wait for the drain task to publish what is left.
    ///
    /// Awaited rather than detached: the final snapshot is the one showing every
    /// group complete, and a dropped task would race the job's own completion so
    /// the last thing an operator saw could be a partial count.
    pub(crate) async fn finish(mut self) {
        drop(self.tx.take());
        self.shutdown.cancel();
        if let Err(error) = self.drain.await {
            tracing::warn!(error = %error, "batch progress drain task panicked");
        }
    }
}

/// One input file's handle for reporting backend progress.
///
/// Cloneable and cheap: this is what threads down through
/// `MorphosyntaxParams` to the chunk dispatch, so it deliberately knows only
/// its own file identity and nothing about jobs, stores or sinks.
#[derive(Clone)]
pub struct BackendProgressPort {
    source_id: DisplayPath,
    tx: mpsc::Sender<ProgressReport>,
}

impl BackendProgressPort {
    /// Declare how many utterances one language group of this file will process.
    ///
    /// Called before the group's chunks are dispatched. This is what gives a
    /// file an exact denominator from the start; inferring it from the chunks
    /// that have reported makes a file whose first chunk finished look complete.
    pub(crate) fn declare_group_total(&self, group: &LanguageCode3, total: UtteranceCount) {
        self.send(ProgressReport::GroupTotal {
            source_id: self.source_id.clone(),
            group: group.clone(),
            total,
        });
    }

    /// Report one wire progress event for one chunk of one language group.
    ///
    /// Only a request-scoped event is utterance progress. A chunk's request
    /// can also carry a load's progress (a lazy load the request triggered: a
    /// download's growth or a phase boundary): that measures the load, not
    /// this file's utterances, so it is logged and never reaches the ledger,
    /// where a growth count in KiB would read as utterances done.
    ///
    /// Uses `try_send`: a progress report must never slow inference or, worse,
    /// deadlock it if the drain task is gone. A dropped report costs at most one
    /// stale second on a display, and the next report supersedes it, because the
    /// ledger stores the latest value per chunk rather than accumulating deltas.
    /// That is a property of the aggregation, not luck.
    pub(crate) fn report(
        &self,
        group: &LanguageCode3,
        chunk: BatchChunkIndex,
        event: &ProgressEventV2,
    ) {
        match event {
            ProgressEventV2::Request(request) => {
                self.send(ProgressReport::Chunk(BackendProgress::from_units(
                    self.source_id.clone(),
                    group.clone(),
                    chunk,
                    request.progress(),
                )));
            }
            ProgressEventV2::Load(load) => {
                tracing::debug!(
                    source_id = %self.source_id,
                    stage = %load.stage,
                    measure = ?load.measure,
                    "model-load progress during a batch request: not utterance progress"
                );
            }
        }
    }

    /// Queue one report, dropping it rather than ever blocking inference.
    fn send(&self, report: ProgressReport) {
        match self.tx.try_send(report) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                tracing::debug!(
                    source_id = %self.source_id,
                    "dropped a batch progress report: channel full"
                );
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                tracing::debug!(
                    source_id = %self.source_id,
                    "dropped a batch progress report: reporter already finished"
                );
            }
        }
    }
}

/// Accumulate reports and publish snapshots until the channel closes.
async fn drain_reports(
    sink: Arc<dyn RunnerEventSink>,
    mut rx: mpsc::Receiver<ProgressReport>,
    shutdown: CancellationToken,
) {
    let mut ledger = BatchProgressLedger::new();
    let mut unpublished_change = false;
    let mut last_publish = Instant::now();

    loop {
        let received = tokio::select! {
            // Bias toward draining queued reports, so a shutdown arriving with
            // reports still in the channel does not discard the newest numbers.
            biased;
            received = tokio::time::timeout(PUBLISH_INTERVAL, rx.recv()) => received,
            () = shutdown.cancelled() => {
                // Take whatever is already queued, then publish and stop.
                while let Ok(report) = rx.try_recv() {
                    unpublished_change |= ledger.record(report);
                }
                if unpublished_change {
                    publish(sink.as_ref(), &ledger).await;
                }
                return;
            }
        };

        match received {
            Ok(Some(report)) => {
                unpublished_change |= ledger.record(report);
            }
            Ok(None) => {
                // Every port is gone: publish the final state and stop.
                if unpublished_change {
                    publish(sink.as_ref(), &ledger).await;
                }
                return;
            }
            Err(_timed_out) => {}
        }

        if unpublished_change && last_publish.elapsed() >= PUBLISH_INTERVAL {
            publish(sink.as_ref(), &ledger).await;
            unpublished_change = false;
            last_publish = Instant::now();
        } else if !unpublished_change && last_publish.elapsed() >= STALL_REPUBLISH_INTERVAL {
            publish(sink.as_ref(), &ledger).await;
            last_publish = Instant::now();
        }
    }
}

/// Publish the ledger's per-file counts.
///
/// There was a second, job-level per-language projection here until
/// 2026-07-30; see `runner::util::batch_progress` for why an aggregate whose
/// denominator only covers the files currently in flight was retired rather
/// than kept.
async fn publish(sink: &dyn RunnerEventSink, ledger: &BatchProgressLedger) {
    for source in ledger.source_progress() {
        // Skip a file whose utterances are all accounted for. Its row is about
        // to go terminal (or already has), and the store refuses progress on a
        // terminal file anyway; not sending is clearer than being refused.
        if source.completed.0 >= source.total.0 {
            continue;
        }
        publish_source(sink, &source).await;
    }
}

/// Publish one file's utterance counts on the ordinary file-progress channel.
///
/// This is the projection that gives an operator watching one long file a
/// denominator, and it deliberately reuses `set_file_progress` rather than
/// inventing a second per-file surface: `FileStage::Analyzing` is already what
/// the CLI spinner, the TUI and the dashboard row render for this stage.
async fn publish_source(sink: &dyn RunnerEventSink, source: &SourceProgress) {
    sink.set_file_progress(
        source.source_id.as_ref(),
        FileStage::Analyzing,
        Some(i64::try_from(source.completed.0).unwrap_or(i64::MAX)),
        Some(i64::try_from(source.total.0).unwrap_or(i64::MAX)),
    )
    .await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::util::batch_progress::BatchChunkIndex;
    use crate::runner::util::test_sink::RecordingSink;
    use batchalign_types::worker_v2::{
        LoadMeasureV2, LoadProgressV2, PhaseEdgeV2, RequestProgressV2, UnitProgressV2,
        WorkerRequestIdV2,
    };

    fn load(measure: LoadMeasureV2) -> ProgressEventV2 {
        ProgressEventV2::Load(LoadProgressV2 {
            stage: "downloading_stanza_eng".to_string(),
            measure,
        })
    }

    fn event(completed: u32, total: u32) -> ProgressEventV2 {
        ProgressEventV2::Request(
            RequestProgressV2::new(
                WorkerRequestIdV2::from("req"),
                "stanza_processing".to_string(),
                UnitProgressV2::new(completed, total).expect("a test count within its total"),
            )
            .expect("a named request"),
        )
    }

    fn lang(code: &str) -> LanguageCode3 {
        LanguageCode3::try_new(code).expect("valid test language")
    }

    /// A file whose chunks are all finished must not get another per-file write.
    ///
    /// Its row is about to go terminal and the store refuses progress on a
    /// terminal file, so sending would be noise that only shows up as a refused
    /// write. The job-level snapshot still counts it.
    /// Guards: Complete chunk reports do not rewrite final file progress.
    /// Kind: contract
    /// Not a type: The reporter must act on real queued completion counts and its sink can observe an erroneous write.
    /// State: A private reporter task, typed chunk messages and recording sink exercise the drain.
    /// Concurrency: Finish joins the reporter before checking writes.
    #[tokio::test]
    async fn a_finished_file_gets_no_further_per_file_write() {
        let sink = Arc::new(RecordingSink::default());
        let reporter = BatchProgressReporter::spawn(sink.clone());

        let port = reporter
            .port(DisplayPath::from("done.cha"))
            .expect("port must exist");
        port.declare_group_total(&lang("eng"), UtteranceCount(100));
        port.report(&lang("eng"), BatchChunkIndex(0), &event(100, 100));
        reporter.finish().await;

        assert!(
            sink.progress().is_empty(),
            "a complete file must not receive a per-file progress write"
        );
    }

    /// Guards: One reporter publishes the correct independent totals for one and two chunks.
    /// Kind: contract
    /// Not a type: Asynchronous chunk reports can be summed under the wrong source or overwrite another chunk.
    /// State: A private reporter task receives two file populations and publishes to a recording sink.
    /// Concurrency: Finish drains the messages before checking each final total.
    #[tokio::test]
    async fn reporter_sums_single_and_multiple_chunks_per_file() {
        let sink = Arc::new(RecordingSink::default());
        let reporter = BatchProgressReporter::spawn(sink.clone());
        for (name, total, chunks) in [
            ("single.cha", 100, &[(40, 100)][..]),
            ("multiple.cha", 548, &[(150, 274), (120, 274)][..]),
        ] {
            let port = reporter.port(DisplayPath::from(name)).unwrap();
            port.declare_group_total(&lang("eng"), UtteranceCount(total));
            for (index, &(current, size)) in chunks.iter().enumerate() {
                port.report(
                    &lang("eng"),
                    BatchChunkIndex(index as u32),
                    &event(current, size),
                );
            }
        }
        reporter.finish().await;
        let writes = sink.progress();
        for (name, total, current) in [("single.cha", 100, 40), ("multiple.cha", 548, 270)] {
            let write = writes.iter().rfind(|entry| entry.filename == name).unwrap();
            assert_eq!(write.total, Some(total));
            assert_eq!(write.current, Some(current));
            assert_eq!(write.stage, FileStage::Analyzing);
        }
    }

    /// `finish()` must not wait for outstanding ports.
    ///
    /// This is the test that caught the original design: `finish()` dropped its
    /// own sender and awaited the drain, but a live port holds a sender clone, so
    /// the channel never closed and `finish()` hung forever. Production happened
    /// to be safe (the fanout is joined before `finish`), which is the worst kind
    /// of safe.
    /// Guards: Reporter finish drains queued messages while extant ports become inert.
    /// Kind: contract
    /// Not a type: The real channel can remain open or accept post-shutdown sends even though the reporter owns its finish handle.
    /// State: A live reporter actor, one held port and a recording sink witness shutdown and late input.
    /// Concurrency: A bounded join observes publication before sending through the retired port.
    #[tokio::test]
    async fn finish_does_not_wait_for_outstanding_ports() {
        let sink = Arc::new(RecordingSink::default());
        let reporter = BatchProgressReporter::spawn(sink.clone());
        let port = reporter
            .port(DisplayPath::from("held.cha"))
            .expect("port must exist");
        port.declare_group_total(&lang("eng"), UtteranceCount(50));
        port.report(&lang("eng"), BatchChunkIndex(0), &event(5, 50));

        // The port is deliberately still alive here.
        tokio::time::timeout(Duration::from_secs(5), reporter.finish())
            .await
            .expect("finish must not wait on an outstanding port");

        assert_eq!(
            sink.progress().len(),
            1,
            "the queued reports must still be published on shutdown"
        );
        port.report(&lang("eng"), BatchChunkIndex(0), &event(20, 50));
        assert_eq!(sink.progress().len(), 1, "a retired port has no effect");
    }

    /// A model load the request triggered is not utterance progress: neither
    /// its growth (KiB, which would read as utterances) nor its phases reach
    /// the file's counts.
    /// Guards: Model load growth and phases never overwrite utterance progress.
    /// Kind: wire
    /// Not a type: Worker events may interleave bytes and utterance quantities under the same active chunk.
    /// State: A private reporter receives one valid count and two typed worker load events.
    /// Concurrency: Finish drains all events before reading the sink.
    #[tokio::test]
    async fn model_load_progress_is_not_utterance_progress() {
        let sink = Arc::new(RecordingSink::default());
        let reporter = BatchProgressReporter::spawn(sink.clone());
        let port = reporter
            .port(DisplayPath::from("a.cha"))
            .expect("port must exist");
        port.declare_group_total(&lang("eng"), UtteranceCount(100));
        port.report(&lang("eng"), BatchChunkIndex(0), &event(10, 100));
        port.report(
            &lang("eng"),
            BatchChunkIndex(0),
            &load(LoadMeasureV2::DownloadGrowth { grown_kib: 4_096 }),
        );
        port.report(
            &lang("eng"),
            BatchChunkIndex(0),
            &load(LoadMeasureV2::Phase {
                edge: PhaseEdgeV2::Complete,
            }),
        );
        reporter.finish().await;

        let writes = sink.progress();
        let write = writes.last().expect("a.cha must get a write");
        assert_eq!(write.current, Some(10));
        assert_eq!(write.total, Some(100));
    }
}
