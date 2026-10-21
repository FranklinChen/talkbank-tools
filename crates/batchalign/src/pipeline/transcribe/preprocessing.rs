//! Word preparation, speaker projection, and pre-CHAT segmentation.

use super::word_segmentation::segment_words;
use super::*;
use crate::infer_retry::Cancellation;

pub(super) async fn stage_asr_postprocess<'a, P: TranscribePlan>(
    mut ctx: TranscribePipelineContext<'a, Recognized, P>,
) -> Result<TranscribePipelineContext<'a, Postprocessed, P>, ServerError> {
    let response = &ctx.state.response;
    // Nothing to post-process means nothing to transcribe. This returned
    // `Ok(())` until 2026-09-16, and `stage_build_chat` then wrote a
    // headers-only CHAT file, so a run that recognized not one word
    // finished as a completed job carrying an empty transcript. Silence
    // reported as success is the defect; the job now says which stage came
    // up empty.
    if response.tokens.is_empty() {
        return Err(ServerError::EmptyTranscription(EmptyTranscription::Asr));
    }

    let asr_output = convert_asr_response(response);
    info!(
        num_tokens = response.tokens.len(),
        num_monologues = asr_output.monologues.len(),
        "ASR response received, starting post-processing"
    );

    let resolved_lang = ctx.state.language.clone();
    let PostprocessedSpeech {
        utterances,
        speaker_assignments,
        word_segmenter,
    } = process_asr_with_prechat_segmentation(&mut ctx, &asr_output, &resolved_lang).await?;
    // The engine returned words and post-processing kept no utterance from
    // them. Its own empty-chunk path returns an empty Vec here, so without
    // this the emptiness travelled on to CHAT assembly unremarked.
    if utterances.is_empty() {
        return Err(ServerError::EmptyTranscription(
            EmptyTranscription::Postprocess,
        ));
    }
    info!(
        num_utterances = utterances.len(),
        "Post-processing complete, building CHAT"
    );

    // Optional diagnostic: when `BA3_DUMP_ASR_PIPELINE=/path/to/file.json`
    // is set, write the per-stage snapshot to disk for inspection.
    if let (Ok(path), Some(snapshot)) = (
        std::env::var("BA3_DUMP_ASR_PIPELINE"),
        ctx.asr_pipeline_snapshot.as_ref(),
    ) {
        let trace = crate::types::results::snapshot_into_pipeline_trace(snapshot.clone());
        match write_pipeline_trace(Path::new(&path), &trace) {
            Ok(()) => tracing::info!(
                path = %path,
                "BA3_DUMP_ASR_PIPELINE wrote per-stage AsrPipelineTrace JSON",
            ),
            Err(error) => tracing::warn!(
                path = %path,
                %error,
                "BA3_DUMP_ASR_PIPELINE failed; continuing with transcription",
            ),
        }
    }

    Ok(ctx.map_state(|asr| Postprocessed {
        asr,
        utterances,
        speaker_assignments,
        word_segmenter,
    }))
}

/// Diagnostic persistence is separate from admission of recognized speech.
/// Only a successful filesystem write produces the success result.
fn write_pipeline_trace(
    path: &Path,
    trace: &crate::traces::AsrPipelineTrace,
) -> Result<(), PipelineTraceWriteError> {
    let json = serde_json::to_vec_pretty(trace)?;
    std::fs::write(path, json)?;
    Ok(())
}

#[derive(Debug, thiserror::Error)]
enum PipelineTraceWriteError {
    #[error("could not encode ASR pipeline trace: {0}")]
    Encode(#[from] serde_json::Error),
    #[error("could not write ASR pipeline trace: {0}")]
    Write(#[from] std::io::Error),
}

pub(crate) fn build_prechat_utseg_items(chunks: &[PreparedMonologueChunk]) -> Vec<UtsegBatchItem> {
    chunks
        .iter()
        .map(|chunk| {
            let words: Vec<String> = chunk
                .words
                .iter()
                .map(|word| word.text.as_str().to_string())
                .collect();
            UtsegBatchItem {
                text: words.join(" "),
                words,
            }
        })
        .collect()
}

pub(crate) fn apply_prechat_assignments(
    chunks: &[PreparedMonologueChunk],
    predictions: &[crate::utseg::AdmittedUtsegPrediction],
) -> Vec<PreparedMonologueChunk> {
    chunks
        .iter()
        .zip(predictions.iter())
        .flat_map(|(chunk, prediction)| {
            asr_postprocess::split_prepared_chunk_by_assignments(
                chunk,
                &prediction.response().assignments,
            )
        })
        .collect()
}

struct PostprocessedSpeech {
    utterances: Vec<Utterance>,
    speaker_assignments: SpeakerAssignmentOutcome,
    /// The segmenter that changed the words, for the provenance stamp.
    word_segmenter: Option<crate::api::ReportedEngineName>,
}

/// `TranscribePipelineContext` has no job cancellation token wired yet (see
/// `stage_run_morphosyntax`'s identical note); genuinely NotWired rather than
/// a fabricated stand-in.
const fn transcribe_cancellation() -> Cancellation<'static> {
    Cancellation::NotWired {
        reason: "TranscribePipelineContext has no job cancellation token wired yet",
    }
}

/// Punctuation retokenization and cleanup of projected chunks: what every
/// path without a pre-CHAT boundary model ends with.
fn punctuation_utterances(
    chunks: Vec<PreparedMonologueChunk>,
    lang: &str,
    snapshot: Option<&mut AsrPipelineSnapshot>,
) -> Vec<Utterance> {
    let mut utterances = asr_postprocess::utterances_from_prepared_chunks(chunks);
    asr_postprocess::finalize_utterances(&mut utterances, lang);
    if let Some(s) = snapshot {
        s.final_utterances = utterances.clone();
    }
    utterances
}

struct ProjectedSpeechChunks {
    chunks: Vec<PreparedMonologueChunk>,
    speaker_assignments: SpeakerAssignmentOutcome,
}

async fn process_asr_with_prechat_segmentation<P: TranscribePlan>(
    ctx: &mut TranscribePipelineContext<'_, Recognized, P>,
    asr_output: &batchalign_transform::asr_postprocess::AsrOutput,
    language: &TranscriptLanguage,
) -> Result<PostprocessedSpeech, ServerError> {
    // Segmentation and the structural cleanup rules run under the primary
    // language; for a pair, the steps that write words keep what was
    // recognized (see `AsrTextLanguage`).
    let resolved_lang = language.primary();
    let lang_str = resolved_lang.to_string();
    let text_language = match language {
        TranscriptLanguage::One(_) => asr_postprocess::AsrTextLanguage::One(&lang_str),
        TranscriptLanguage::Pair(_) => {
            asr_postprocess::AsrTextLanguage::CodeSwitched { primary: &lang_str }
        }
    };
    let project_speakers = |chunks| -> Result<ProjectedSpeechChunks, ServerError> {
        let Some(segments) = ctx.speaker_segments.as_deref() else {
            return Ok(ProjectedSpeechChunks {
                chunks,
                speaker_assignments: SpeakerAssignmentOutcome::AsrOnly,
            });
        };
        let segments: Vec<ChatSpeakerSegment> = segments
            .iter()
            .map(|segment| ChatSpeakerSegment {
                interval: segment.interval,
                speaker: segment.speaker.clone(),
            })
            .collect();
        let projection = project_speakers_onto_chunks(chunks, &segments);
        let stats = projection.stats();
        info!(
            contested_timed_words = stats.contested_timed_words,
            unattested_timed_words = stats.unattested_timed_words,
            speaker_boundaries = stats.speaker_boundaries,
            "Projected diarization onto timed ASR words"
        );
        let admitted = projection.admit(SpeakerProjectionPolicy::BestEffortWithReviewEvidenceV1);
        ctx.dumper
            .dump_speaker_projection(
                ctx.audio_path.to_string_lossy().as_ref(),
                admitted.evidence(),
            )
            .map_err(|error| {
                ServerError::Persistence(format!(
                    "could not retain requested speaker projection evidence: {error}"
                ))
            })?;
        let parts = admitted.into_parts();
        Ok(ProjectedSpeechChunks {
            chunks: parts.chunks,
            speaker_assignments: SpeakerAssignmentOutcome::Dedicated(parts.evidence),
        })
    };
    // Stages 1-4, word segmentation, stages 5-5b, then speaker projection:
    // the same for every path below. Segmentation runs on whole monologues,
    // before chunking, and its result is the only route to chunks.
    let monologues = asr_postprocess::prepare_monologue_words_with_snapshot(
        asr_output,
        text_language,
        ctx.asr_pipeline_snapshot.as_mut(),
    )
    .map_err(han_script_normalization_refused)?;
    let segmented = segment_words(
        ctx.services.pool,
        resolved_lang,
        monologues,
        transcribe_cancellation(),
    )
    .await?;
    let (prepared, word_segmenter) =
        segmented.chunk(text_language.rules(), ctx.asr_pipeline_snapshot.as_mut());
    let ProjectedSpeechChunks {
        chunks: prepared_chunks,
        speaker_assignments,
    } = project_speakers(prepared)?;

    let Some(pre_chat_policy) = ctx.utseg_execution.pre_chat_policy() else {
        return Ok(PostprocessedSpeech {
            utterances: punctuation_utterances(
                prepared_chunks,
                &lang_str,
                ctx.asr_pipeline_snapshot.as_mut(),
            ),
            speaker_assignments,
            word_segmenter,
        });
    };

    // The pre-CHAT pass runs the boundary model or nothing: there is no
    // pre-CHAT Stanza path, so an authorized Stanza fallback segments only
    // after CHAT is built, and this pass hands its chunks to punctuation
    // retokenization exactly as it always did for a language with no model.
    let route = ctx.state.segmentation.as_ref().ok_or_else(|| {
        ServerError::Validation("pre-CHAT segmentation requested from a disabled plan".into())
    })?;
    let Some(route) = route.model().filter(|model| model.uses_boundary_model()) else {
        return Ok(PostprocessedSpeech {
            utterances: punctuation_utterances(
                prepared_chunks,
                &lang_str,
                ctx.asr_pipeline_snapshot.as_mut(),
            ),
            speaker_assignments,
            word_segmenter,
        });
    };

    if prepared_chunks.is_empty() {
        return Ok(PostprocessedSpeech {
            utterances: Vec::new(),
            speaker_assignments,
            word_segmenter,
        });
    }

    let items = build_prechat_utseg_items(&prepared_chunks);
    let predictions = crate::utseg::infer_utseg_predictions_with_policy(
        ctx.services.pool,
        route.language(),
        &items,
        route.fallback().is_allowed(),
        pre_chat_policy,
        transcribe_cancellation(),
    )
    .await?;
    let split_chunks = apply_prechat_assignments(&prepared_chunks, &predictions);
    let indexed_items: Vec<_> = items.iter().cloned().enumerate().collect();
    let evidence = UtsegEvidenceTrace::from_predictions(
        UtsegEvidencePhase::PreChat,
        resolved_lang.as_ref(),
        &indexed_items,
        &predictions,
    )
    .map_err(|error| ServerError::Validation(error.to_string()))?;
    ctx.utseg_evidence_sink
        .write(ctx.audio_path.to_string_lossy().as_ref(), &evidence)
        .map_err(|error| {
            ServerError::Persistence(format!(
                "could not retain requested pre-CHAT utseg evidence for {}: {error}",
                ctx.audio_path.display()
            ))
        })?;
    let mut utterances = asr_postprocess::utterances_from_prepared_chunks(split_chunks);
    asr_postprocess::finalize_utterances(&mut utterances, &lang_str);
    if let Some(s) = ctx.asr_pipeline_snapshot.as_mut() {
        s.final_utterances = utterances.clone();
    }
    Ok(PostprocessedSpeech {
        utterances,
        speaker_assignments,
        word_segmenter,
    })
}

/// Prepare ASR chunks fully in Rust.
///
/// Stages 1-3 (compound merge, timed-word extraction, Han-script normalization,
/// multi-word split) run per monologue. Number expansion is then applied per
/// word via `asr_postprocess::expand_number`. After expansion a whitespace-split
/// pass widens multi-word expansions into separate tokens. Stages 5-5b
/// (long-turn and pause splitting) finalize per monologue.
///
/// The entry point for any caller that wants what transcribe produces without
/// a per-stage trace, including the offline `eval utseg-replay` pre-ASR pass.
/// A replay that prepared its chunks some other way would be comparing two
/// implementations rather than replaying one. The one stage it leaves out is
/// word segmentation, which needs a worker and applies only to the languages
/// `WordSegmenterV2::for_language` names; none of them has a pre-CHAT
/// boundary model, so no replay of one is affected.
pub(crate) fn prepare_asr_chunks<'a>(
    asr_output: &batchalign_transform::asr_postprocess::AsrOutput,
    language: impl Into<asr_postprocess::AsrTextLanguage<'a>>,
) -> Result<Vec<PreparedMonologueChunk>, ServerError> {
    prepare_asr_chunks_with_snapshot(asr_output, language.into(), None)
}

/// A Han-script normalization that changed a monologue's character count refuses
/// the file.
///
/// The alternative would be re-cutting the words around the new characters,
/// after which each word's timing would belong to characters it no longer
/// holds. A transcript mistimed that way looks right and reads wrong, which is
/// worse than a named failure.
fn han_script_normalization_refused(
    refusal: batchalign_transform::asr_postprocess::NormalizationChangedLength,
) -> ServerError {
    ServerError::Validation(format!("ASR post-processing refused this file: {refusal}"))
}

/// Snapshot-aware variant of [`prepare_asr_chunks`]: the transform's two
/// steps with the trace threaded through them, and nothing between them.
///
/// When `snapshot` is `Some`, populates per-stage trace fields:
/// `raw_elements`, `after_compound_merge`, `after_timing_extract`,
/// `after_multiword_split`, `after_number_expand`,
/// `after_han_script_norm` (`yue` and `nan` only), `after_long_turn_split`.
pub(super) fn prepare_asr_chunks_with_snapshot(
    asr_output: &batchalign_transform::asr_postprocess::AsrOutput,
    language: asr_postprocess::AsrTextLanguage<'_>,
    mut snapshot: Option<&mut AsrPipelineSnapshot>,
) -> Result<Vec<PreparedMonologueChunk>, ServerError> {
    let monologues = asr_postprocess::prepare_monologue_words_with_snapshot(
        asr_output,
        language,
        snapshot.as_deref_mut(),
    )
    .map_err(han_script_normalization_refused)?;
    Ok(asr_postprocess::chunk_monologues_with_snapshot(
        monologues,
        language.rules(),
        snapshot,
    ))
}

#[cfg(test)]
mod trace_write_tests {
    use super::*;

    #[test]
    fn snapshot_write_preserves_schema_and_reports_filesystem_failure() {
        let dir = tempfile::tempdir().expect("temporary directory");
        let trace =
            crate::types::results::snapshot_into_pipeline_trace(AsrPipelineSnapshot::default());
        // A directory is an unwritable file target even when tests run as root.
        assert!(matches!(
            write_pipeline_trace(dir.path(), &trace),
            Err(PipelineTraceWriteError::Write(_))
        ));
        let path = dir.path().join("trace.json");
        write_pipeline_trace(&path, &trace).expect("write trace");
        let actual: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).expect("read trace")).expect("trace JSON");
        assert_eq!(actual, serde_json::to_value(trace).expect("expected trace"));
    }
}
