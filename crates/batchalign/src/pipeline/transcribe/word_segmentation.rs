//! Word segmentation stage: group a language's one-character recognizer words
//! into words, for the languages `WordSegmenterV2::for_language` names.
//!
//! ```text
//! MonologueWords --segment_words--> SegmentedMonologues --chunk--> PreparedMonologueChunk
//! ```
//!
//! Runs on whole monologues after stage 4 (word forms; Han-script
//! normalization has already put the text in the lexicon's script) and before
//! chunking, so a long-turn split never cuts a run, and before speaker
//! projection, so a word is attributed whole. [`SegmentedMonologues`] is the
//! only route from monologue words to chunks in transcribe, so the stage
//! cannot be skipped. The merge and its proof live in
//! `batchalign_transform::asr_postprocess::word_segmentation`; this stage only
//! carries the runs to the worker and the token lists back.

use batchalign_transform::asr_postprocess::{
    AsrPipelineSnapshot, MonologueWords, PlannedSegmentation, PreparedMonologueChunk,
    chunk_monologues_with_snapshot,
};

use crate::api::{LanguageCode3, ReportedEngineName};
use crate::error::ServerError;
use crate::infer_retry::{Cancellation, dispatch_execute_v2_with_retry};
use crate::types::worker_v2::WordSegmenterV2;
use crate::worker::error::WorkerError;
use crate::worker::pool::WorkerPool;
use crate::worker::text_request_v2::{
    PreparedTextRequestIdsV2, build_word_segmentation_request_v2,
};
use crate::worker::text_result_v2::parse_word_segmentation_result_v2;

/// Monologue words that have been through word segmentation, whether or not
/// it changed them.
pub(crate) struct SegmentedMonologues {
    monologues: Vec<MonologueWords>,
    /// The segmenter that changed the words, as the worker reported it; `None`
    /// when the language has no segmenter or the file had nothing to segment.
    segmenter: Option<ReportedEngineName>,
}

impl SegmentedMonologues {
    /// Cut the monologues into chunks (stages 5-5b), recording the segmented
    /// words in the trace when a segmenter changed them. Returns the chunks
    /// and the segmenter a provenance stamp names.
    pub(crate) fn chunk(
        self,
        lang: &str,
        mut snapshot: Option<&mut AsrPipelineSnapshot>,
    ) -> (Vec<PreparedMonologueChunk>, Option<ReportedEngineName>) {
        if let (Some(snapshot), Some(_)) = (snapshot.as_deref_mut(), &self.segmenter) {
            snapshot.after_word_segmentation = Some(
                self.monologues
                    .iter()
                    .flat_map(|monologue| monologue.words().iter().cloned())
                    .collect(),
            );
        }
        (
            chunk_monologues_with_snapshot(self.monologues, lang, snapshot),
            self.segmenter,
        )
    }
}

fn segmentation_protocol_error(message: impl Into<String>) -> ServerError {
    ServerError::Worker(WorkerError::Protocol(message.into()))
}

/// Segment `monologues` for `language`, or pass them through unchanged.
pub(crate) async fn segment_words(
    pool: &WorkerPool,
    language: &LanguageCode3,
    monologues: Vec<MonologueWords>,
    cancellation: Cancellation<'_>,
) -> Result<SegmentedMonologues, ServerError> {
    let unchanged = |monologues| SegmentedMonologues {
        monologues,
        segmenter: None,
    };
    let Some(segmenter) = WordSegmenterV2::for_language(language) else {
        return Ok(unchanged(monologues));
    };
    let plan = match PlannedSegmentation::new(monologues) {
        PlannedSegmentation::NothingToSegment(monologues) => return Ok(unchanged(monologues)),
        PlannedSegmentation::Runs(plan) => plan,
    };
    let request = build_word_segmentation_request_v2(
        &PreparedTextRequestIdsV2::for_task("word_segmentation"),
        segmenter,
        plan.texts().map(str::to_owned).collect(),
    )
    .map_err(|error| {
        ServerError::Validation(format!(
            "could not build the word-segmentation request: {error}"
        ))
    })?;
    let response = dispatch_execute_v2_with_retry(pool, language, &request, cancellation).await?;
    let result = parse_word_segmentation_result_v2(response).map_err(|error| {
        segmentation_protocol_error(format!("invalid word-segmentation V2 result: {error}"))
    })?;
    let monologues = plan.apply(result.items).map_err(|refusal| {
        segmentation_protocol_error(format!(
            "word segmentation refused for this file: {refusal}"
        ))
    })?;
    Ok(SegmentedMonologues {
        monologues,
        segmenter: Some(result.segmenter),
    })
}
