//! Word preparation, speaker projection, and pre-CHAT segmentation.

use super::*;

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
        if let Ok(json) = serde_json::to_string_pretty(&trace) {
            let _ = std::fs::write(&path, json);
            tracing::warn!(
                path = %path,
                "BA3_DUMP_ASR_PIPELINE wrote per-stage AsrPipelineTrace JSON",
            );
        }
    }

    Ok(ctx.map_state(|asr| Postprocessed {
        asr,
        utterances,
        speaker_assignments,
    }))
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
    let Some(pre_chat_policy) = ctx.utseg_execution.pre_chat_policy() else {
        let ProjectedSpeechChunks {
            chunks,
            speaker_assignments,
        } = project_speakers(prepare_asr_chunks_with_snapshot(
            asr_output,
            text_language,
            ctx.asr_pipeline_snapshot.as_mut(),
        )?)?;
        let mut utterances = asr_postprocess::utterances_from_prepared_chunks(chunks);
        asr_postprocess::finalize_utterances(&mut utterances, &lang_str);
        if let Some(s) = ctx.asr_pipeline_snapshot.as_mut() {
            s.final_utterances = utterances.clone();
        }
        return Ok(PostprocessedSpeech {
            utterances,
            speaker_assignments,
        });
    };

    // The pre-CHAT pass runs the boundary model or nothing: there is no
    // pre-CHAT Stanza path, so an authorized Stanza fallback segments only
    // after CHAT is built, and this pass hands its chunks to punctuation
    // retokenization exactly as it always did for a language with no model.
    let route = ctx.state.segmentation.as_ref().ok_or_else(|| {
        ServerError::Validation("pre-CHAT segmentation requested from a disabled plan".into())
    })?;
    if !route.uses_boundary_model() {
        let ProjectedSpeechChunks {
            chunks,
            speaker_assignments,
        } = project_speakers(prepare_asr_chunks_with_snapshot(
            asr_output,
            text_language,
            ctx.asr_pipeline_snapshot.as_mut(),
        )?)?;
        let mut utterances = asr_postprocess::utterances_from_prepared_chunks(chunks);
        asr_postprocess::finalize_utterances(&mut utterances, &lang_str);
        if let Some(s) = ctx.asr_pipeline_snapshot.as_mut() {
            s.final_utterances = utterances.clone();
        }
        return Ok(PostprocessedSpeech {
            utterances,
            speaker_assignments,
        });
    }

    let ProjectedSpeechChunks {
        chunks: prepared_chunks,
        speaker_assignments,
    } = project_speakers(prepare_asr_chunks_with_snapshot(
        asr_output,
        text_language,
        ctx.asr_pipeline_snapshot.as_mut(),
    )?)?;
    if prepared_chunks.is_empty() {
        return Ok(PostprocessedSpeech {
            utterances: Vec::new(),
            speaker_assignments,
        });
    }

    let items = build_prechat_utseg_items(&prepared_chunks);
    let predictions = crate::utseg::infer_utseg_predictions_with_policy(
        ctx.services.pool,
        route.language(),
        &items,
        route.fallback().is_allowed(),
        pre_chat_policy,
        // `TranscribePipelineContext` has no job cancellation token wired
        // yet (see `stage_run_morphosyntax`'s identical note); genuinely
        // NotWired rather than a fabricated stand-in.
        crate::infer_retry::Cancellation::NotWired {
            reason: "TranscribePipelineContext has no job cancellation token wired yet",
        },
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
    })
}

/// Prepare ASR chunks fully in Rust.
///
/// Stages 1-3 (compound merge, timed-word extraction, Cantonese normalization,
/// multi-word split) run per monologue. Number expansion is then applied per
/// word via `asr_postprocess::expand_number`. After expansion a whitespace-split
/// pass widens multi-word expansions into separate tokens. Stages 5-5b
/// (long-turn and pause splitting) finalize per monologue.
///
/// The entry point for any caller that wants what transcribe produces without
/// a per-stage trace, including the offline `eval utseg-replay` pre-ASR pass.
/// A replay that prepared its chunks some other way would be comparing two
/// implementations rather than replaying one.
pub(crate) fn prepare_asr_chunks<'a>(
    asr_output: &batchalign_transform::asr_postprocess::AsrOutput,
    language: impl Into<asr_postprocess::AsrTextLanguage<'a>>,
) -> Result<Vec<PreparedMonologueChunk>, ServerError> {
    prepare_asr_chunks_with_snapshot(asr_output, language.into(), None)
}

/// A Cantonese normalization that changed a monologue's character count refuses
/// the file.
///
/// The alternative would be re-cutting the words around the new characters,
/// after which each word's timing would belong to characters it no longer
/// holds. A transcript mistimed that way looks right and reads wrong, which is
/// worse than a named failure.
fn cantonese_normalization_refused(
    refusal: batchalign_transform::asr_postprocess::NormalizationChangedLength,
) -> ServerError {
    ServerError::Validation(format!("ASR post-processing refused this file: {refusal}"))
}

/// Snapshot-aware variant of [`prepare_asr_chunks`].
///
/// When `snapshot` is `Some`, populates per-stage trace fields:
/// `raw_elements`, `after_compound_merge`, `after_timing_extract`,
/// `after_multiword_split`, `after_number_expand`,
/// `after_cantonese_norm` (yue only), `after_long_turn_split`. The
/// `final_utterances` field is filled by the caller after `retokenize`.
///
/// Multi-monologue inputs concatenate their stage outputs into the
/// snapshot's flat fields (other than `after_long_turn_split` which
/// is `Vec<Vec<...>>` and accumulates chunks in order).
pub(super) fn prepare_asr_chunks_with_snapshot(
    asr_output: &batchalign_transform::asr_postprocess::AsrOutput,
    language: asr_postprocess::AsrTextLanguage<'_>,
    mut snapshot: Option<&mut AsrPipelineSnapshot>,
) -> Result<Vec<PreparedMonologueChunk>, ServerError> {
    let lang = language.rules();
    let Some(_) = snapshot.as_ref() else {
        // One implementation of this step, not two. Without a trace to fill in
        // there is nothing this function adds over the transform's own
        // preparation, and keeping a second copy here is exactly the drift
        // `eval utseg-replay` exists to catch. The traced path below stays
        // separate only because it must record each stage as it goes;
        // `snapshot_and_plain_preparation_agree` holds the two to one answer.
        return asr_postprocess::prepare_asr_chunks(asr_output, language)
            .map_err(cantonese_normalization_refused);
    };

    if let Some(ref mut s) = snapshot {
        for m in &asr_output.monologues {
            s.raw_elements.extend_from_slice(&m.elements);
        }
    }

    let mut monologue_words: Vec<(asr_postprocess::SpeakerIndex, Vec<AsrWord>)> =
        Vec::with_capacity(asr_output.monologues.len());
    for m in &asr_output.monologues {
        let mut sub = AsrPipelineSnapshot::default();
        let cap = snapshot.is_some().then_some(&mut sub);
        let words =
            asr_postprocess::prepare_words_pre_expansion_with_snapshot(&m.elements, language, cap)
                .map_err(cantonese_normalization_refused)?;
        if let Some(ref mut s) = snapshot {
            s.after_compound_merge.extend(sub.after_compound_merge);
            s.after_timing_extract.extend(sub.after_timing_extract);
            // Cantonese normalization is captured here now: it runs inside
            // preparation, before the multi-word split, not after expansion.
            if let Some(yue) = sub.after_cantonese_norm {
                s.after_cantonese_norm
                    .get_or_insert_with(Vec::new)
                    .extend(yue);
            }
            s.after_multiword_split.extend(sub.after_multiword_split);
        }
        monologue_words.push((m.speaker, words));
    }

    // Stage 4, CHAT word forms: the one implementation the untraced path
    // runs too (it owns the code-switched policy as well).
    for (_speaker, words) in &mut monologue_words {
        *words = asr_postprocess::write_word_forms(std::mem::take(words), language);
    }

    if let Some(ref mut s) = snapshot {
        for (_, words) in &monologue_words {
            s.after_number_expand.extend_from_slice(words);
        }
    }

    let mut prepared = Vec::new();
    for (speaker, words) in monologue_words {
        let mut sub = AsrPipelineSnapshot::default();
        let cap = snapshot.is_some().then_some(&mut sub);
        prepared.extend(asr_postprocess::finalize_words_to_chunks_with_snapshot(
            words, speaker, lang, cap,
        ));
        if let Some(ref mut s) = snapshot {
            s.after_long_turn_split.extend(sub.after_long_turn_split);
        }
    }
    Ok(prepared)
}
