//! Server-side benchmark orchestrator.
//!
//! Benchmarking is conceptually "transcribe, then compare against a gold CHAT
//! companion". Neither half requires Python-side document orchestration:
//! - transcription already has a Rust-owned pipeline around raw ASR inference
//! - comparison is already Rust-owned morphosyntax + DP alignment
//!
//! This module composes those two existing Rust pipelines so the `benchmark`
//! command no longer depends on a fictitious Python worker benchmark path.

use std::future::Future;
use std::path::Path;
use std::pin::Pin;

use crate::chat_ops::morphosyntax_ops::MwtDict;

use crate::api::LanguageCode3;
use crate::error::ServerError;
use crate::pipeline::PipelineServices;
use crate::runner::util::ProgressSender;
use crate::transcribe::TranscribeOptions;

pub(crate) use crate::compare::MainAnnotatedCompareOutputs as BenchmarkOutputs;

/// The producer owns the composite state machine on the heap before callers
/// poll it. Its two heavy sub-pipelines are likewise separate pinned owners.
type BenchmarkFuture<'a> =
    Pin<Box<dyn Future<Output = Result<BenchmarkOutputs, ServerError>> + Send + 'a>>;

/// Borrowed request bundle for one benchmark execution.
pub(crate) struct BenchmarkRequest<'a> {
    /// Audio file to transcribe before comparison.
    pub audio_path: &'a Path,
    /// Gold-standard CHAT transcript to compare against.
    pub reference: crate::compare::AdmittedComparisonReference,
    /// Primary language used for comparison and downstream NLP shaping.
    pub lang: &'a LanguageCode3,
    /// Shared worker/cache services used by the transcribe and compare phases.
    pub services: PipelineServices<'a>,
    /// Typed transcription options for the Rust-owned transcribe pipeline.
    pub transcribe_options: &'a TranscribeOptions,
    /// Multi-word-token dictionary shared with the compare pipeline.
    pub mwt: &'a MwtDict,
    /// Optional progress sink for the transcribe sub-pipeline.
    pub progress: Option<&'a ProgressSender>,
}

/// Run the benchmark pipeline for one audio file and one gold CHAT transcript.
///
/// Returns a heap-owned future resolving to [`BenchmarkOutputs`] with CHAT and metrics.
pub(crate) fn process_benchmark(request: BenchmarkRequest<'_>) -> BenchmarkFuture<'_> {
    Box::pin(async move {
        let transcribed = crate::transcribe::process_transcribe(
            request.audio_path,
            request.services,
            request.transcribe_options,
            request.progress.cloned(),
            None,
        )
        .await?;
        // Benchmark's deliverable is the comparison, and comparing needs an
        // admitted transcript (it morphotags the main side). A transcript
        // written with diagnostics is transcribe's output, not benchmark's:
        // there is nothing admitted to compare, so benchmark reports the
        // diagnosis as the refusal it is for this command, exactly as it did
        // before transcription kept such output.
        let crate::pipeline::transcribe::TranscribeOutput {
            document,
            shortfalls,
        } = transcribed;
        let transcribed_chat = match document {
            crate::pipeline::post_validate::ProducedOutput::Admitted(admitted) => admitted,
            crate::pipeline::post_validate::ProducedOutput::Diagnosed(diagnosed) => {
                return Err(diagnosed.into_failure().into_server_error());
            }
        };
        // An admitted transcript whose optional stages did not apply is still
        // comparable: compare morphotags the main side itself. The shortfall
        // is not part of benchmark's own outputs, so it is logged here.
        for shortfall in &shortfalls {
            tracing::warn!(%shortfall, "benchmark's transcript does not carry a requested stage");
        }

        Box::pin(crate::compare::process_compare_constructed_main(
            transcribed_chat,
            request.reference,
            request.lang,
            request.services,
            request.mwt,
        ))
        .await
    })
}

/// Derive the companion gold CHAT path for one audio file.
///
/// Convention:
/// - `sample.wav` -> `sample.cha`
/// - `/dir/sample.mp3` -> `/dir/sample.cha`
#[cfg(test)]
pub(crate) fn gold_chat_path_for_audio(audio_path: &str) -> String {
    let path = Path::new(audio_path);
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    let parent = path.parent().unwrap_or_else(|| Path::new(""));
    parent
        .join(format!("{stem}.cha"))
        .to_string_lossy()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::{BenchmarkFuture, BenchmarkRequest, gold_chat_path_for_audio, process_benchmark};

    #[test]
    fn composite_future_is_heap_owned_at_its_producer_boundary() {
        fn require_boxed_producer(_: for<'a> fn(BenchmarkRequest<'a>) -> BenchmarkFuture<'a>) {}
        require_boxed_producer(process_benchmark);
        assert_eq!(
            std::mem::size_of::<BenchmarkFuture<'static>>(),
            2 * std::mem::size_of::<usize>(),
        );
    }

    #[test]
    fn derives_gold_chat_path_from_audio() {
        assert_eq!(gold_chat_path_for_audio("sample.wav"), "sample.cha");
        assert_eq!(
            gold_chat_path_for_audio("/data/interview.mp3"),
            "/data/interview.cha"
        );
    }
}
