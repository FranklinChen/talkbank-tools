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
        mut utterances,
        speaker_assignments,
        word_segmenter,
        abbreviations,
    } = process_asr_with_prechat_segmentation(&mut ctx, &asr_output, &resolved_lang).await?;
    // The engine returned words and post-processing kept no utterance from
    // them. Its own empty-chunk path returns an empty Vec here, so without
    // this the emptiness travelled on to CHAT assembly unremarked. An
    // utterance that kept only the record of its dropped tokens is not a
    // kept utterance either: it has no words to write.
    if utterances
        .iter()
        .all(|utterance| utterance.words.is_empty())
    {
        return Err(ServerError::EmptyTranscription(
            EmptyTranscription::Postprocess,
        ));
    }
    // The recognizer's window seams it settled by rule go on the utterance
    // that keeps each copy, now that the utterances exist. Retokenization
    // rewrites words but keeps their timing, which is what the record is
    // placed by; see `asr_postprocess::seam`. Refused only with no utterance
    // at all, which the emptiness check above has already excluded.
    crate::transcribe::place_response_seams(&mut utterances, &ctx.state.response)
        .map_err(|unplaced| ServerError::Validation(unplaced.to_string()))?;
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
        abbreviations,
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

/// Cut each prepared chunk as its admitted prediction says, after analyzing
/// the chunk's exact repetitions: a cut that would separate the copies of
/// one is not made, so retrace detection, which runs per piece afterwards,
/// still sees both copies (`yeah I [/] I went .`, never `yeah I .` and
/// `I went .`).
///
/// The analysis is the only way to the split
/// ([`asr_postprocess::RetraceAnalyzedChunk::analyze`] mints the state
/// [`asr_postprocess::RetraceAnalyzedChunk::split`] consumes), so this path
/// cannot cut before it. Each piece carries the terminator its cut predicted.
pub(crate) fn apply_prechat_segmentations(
    chunks: Vec<PreparedMonologueChunk>,
    predictions: &[crate::utseg::AdmittedUtsegPrediction],
    lang: &str,
) -> Result<Vec<asr_postprocess::SegmentedChunk>, ServerError> {
    // One prediction per chunk, by position: a `zip` would drop whichever
    // side is longer without a word.
    if chunks.len() != predictions.len() {
        return Err(ServerError::Validation(format!(
            "pre-CHAT segmentation returned {} predictions for {} chunks",
            predictions.len(),
            chunks.len()
        )));
    }
    let mut pieces = Vec::new();
    for (chunk_index, (chunk, prediction)) in chunks.into_iter().zip(predictions.iter()).enumerate()
    {
        let split = asr_postprocess::RetraceAnalyzedChunk::analyze(chunk, lang)
            .split(prediction.segmentation())
            .map_err(|error| {
                ServerError::Validation(format!(
                    "pre-CHAT segmentation of chunk {chunk_index} is not the chunk's: {error}"
                ))
            })?;
        if !split.suppressed_cuts_before_words.is_empty() {
            info!(
                chunk = chunk_index,
                before_words = ?split.suppressed_cuts_before_words,
                "kept the copies of an exact repetition in one utterance against a model boundary"
            );
        }
        pieces.extend(split.pieces);
    }
    Ok(pieces)
}

struct PostprocessedSpeech {
    utterances: Vec<Utterance>,
    speaker_assignments: SpeakerAssignmentOutcome,
    /// The segmenter that changed the words, for the provenance stamp.
    word_segmenter: Option<crate::api::ReportedEngineName>,
    /// What the abbreviation step did, for the provenance stamp.
    abbreviations: AbbreviationMerging,
}

/// Apply `--merge-abbrev` to the projected chunks: merge single letters that
/// spell a known abbreviation into one word timed by the union of the letters
/// (see `asr_postprocess::merge_abbreviations` for the rules), and say what was
/// done, for the stamp. The stamp is read from this result rather than from the
/// request, so it cannot claim a merge a path did not run.
///
/// It runs HERE, on the recognizer's words, for three reasons. Each word still
/// carries the timing it was recognized with, so the merged word's interval is
/// the true union rather than a guess made after the fact. Speakers are final
/// (diarization has been projected), so a speaker change ends a run. And no
/// CHAT exists yet, so nothing is parsed or re-serialized to make the change.
pub(crate) fn resolve_abbreviations(
    policy: crate::params::MergeAbbrevPolicy,
    chunks: Vec<PreparedMonologueChunk>,
) -> (Vec<PreparedMonologueChunk>, AbbreviationMerging) {
    match policy {
        crate::params::MergeAbbrevPolicy::Keep => (chunks, AbbreviationMerging::NotRequested),
        crate::params::MergeAbbrevPolicy::Merge => {
            let (chunks, report) = asr_postprocess::merge_abbreviations(chunks);
            info!(
                merged = report.merged(),
                left_untimed = report.left_untimed(),
                "Merged single-letter abbreviations on the recognized words"
            );
            (chunks, AbbreviationMerging::Ran)
        }
    }
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
        chunks: projected_chunks,
        speaker_assignments,
    } = project_speakers(prepared)?;
    let (prepared_chunks, abbreviations) =
        resolve_abbreviations(ctx.opts.merge_abbrev, projected_chunks);

    let Some(pre_chat_policy) = ctx.utseg_execution.pre_chat_policy() else {
        return Ok(PostprocessedSpeech {
            utterances: punctuation_utterances(
                prepared_chunks,
                &lang_str,
                ctx.asr_pipeline_snapshot.as_mut(),
            ),
            speaker_assignments,
            word_segmenter,
            abbreviations,
        });
    };

    // The pre-CHAT pass runs the boundary model or nothing: there is no
    // pre-CHAT Stanza path, so an authorized Stanza fallback segments only
    // after CHAT is built, and this pass hands its chunks to punctuation
    // retokenization exactly as it always did for a language with no model.
    let route = ctx.state.segmentation.as_ref().ok_or_else(|| {
        ServerError::Validation("pre-CHAT segmentation requested from a disabled plan".into())
    })?;
    let Some(route) = route
        .model()
        .and_then(crate::utseg_route::UtsegRoute::boundary_model)
    else {
        return Ok(PostprocessedSpeech {
            utterances: punctuation_utterances(
                prepared_chunks,
                &lang_str,
                ctx.asr_pipeline_snapshot.as_mut(),
            ),
            speaker_assignments,
            word_segmenter,
            abbreviations,
        });
    };

    if prepared_chunks.is_empty() {
        return Ok(PostprocessedSpeech {
            utterances: Vec::new(),
            speaker_assignments,
            word_segmenter,
            abbreviations,
        });
    }

    let items = build_prechat_utseg_items(&prepared_chunks);
    let predictions = crate::utseg::infer_utseg_predictions_with_policy(
        ctx.services.pool,
        &route,
        &items,
        pre_chat_policy,
        transcribe_cancellation(),
    )
    .await?;
    let split_chunks = apply_prechat_segmentations(prepared_chunks, &predictions, &lang_str)?;
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
    let mut utterances = asr_postprocess::utterances_from_segmented_chunks(split_chunks);
    asr_postprocess::finalize_utterances(&mut utterances, &lang_str);
    if let Some(s) = ctx.asr_pipeline_snapshot.as_mut() {
        s.final_utterances = utterances.clone();
    }
    Ok(PostprocessedSpeech {
        utterances,
        speaker_assignments,
        word_segmenter,
        abbreviations,
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
/// boundary model, so no replay of one is affected. Abbreviation merging is
/// the caller's next step ([`resolve_abbreviations`]), under the policy the
/// run used: the replay reads it from the retained output's stamp.
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
mod prechat_segmentation_tests {
    use super::*;
    use crate::types::worker_v2::{
        BoundaryProbabilityMicrosV2, HubCommitV2, UtsegAdjacencyPolicyRevisionV2,
        UtsegBoundaryActionV2, UtsegBoundaryModelEvidenceV2, UtsegNormalizationRevisionV2,
        UtsegWordBoundaryEvidenceV2,
    };
    use crate::utseg::{UtsegPredictionOrigin, admit_prediction};
    use batchalign_transform::asr_postprocess::{AsrWord, SpeakerIndex};

    /// The chunk a recognizer's unpunctuated words make, timed in order.
    fn chunk(words: &[&str]) -> PreparedMonologueChunk {
        PreparedMonologueChunk {
            speaker: SpeakerIndex(0),
            words: words
                .iter()
                .enumerate()
                .map(|(i, w)| AsrWord::new(*w, Some(i as i64 * 100), Some(i as i64 * 100 + 90)))
                .collect(),
        }
    }

    /// An admitted boundary-model prediction for `words` whose model placed
    /// `action` after each word named in `boundaries`, exactly as the worker
    /// declares one: applied actions equal to raw ones, assignments derived
    /// from them, admitted through the one constructor.
    fn admitted(
        words: &[&str],
        boundaries: &[(usize, UtsegBoundaryActionV2)],
    ) -> crate::utseg::AdmittedUtsegPrediction {
        let word_evidence: Vec<_> = (0..words.len())
            .map(|index| {
                let action = boundaries
                    .iter()
                    .find(|(at, _)| *at == index)
                    .map_or(UtsegBoundaryActionV2::Ordinary, |(_, action)| *action);
                UtsegWordBoundaryEvidenceV2::Classified {
                    raw_action: action,
                    applied_action: action,
                    boundary_probability_micros: BoundaryProbabilityMicrosV2::try_from(900_000)
                        .expect("fixture probability"),
                }
            })
            .collect();
        // The assignments the worker derives from its applied actions.
        let assignments = batchalign_types::segmentation::UtteranceSegmentation::from_boundaries(
            word_evidence.iter().map(|evidence| match evidence {
                UtsegWordBoundaryEvidenceV2::Classified { applied_action, .. } => {
                    applied_action.boundary()
                }
                UtsegWordBoundaryEvidenceV2::NormalizationOmission
                | UtsegWordBoundaryEvidenceV2::ModelShortCircuit => None,
            }),
        )
        .into_assignments();
        let evidence = UtsegBoundaryModelEvidenceV2 {
            model_id: "talkbank/utterance-boundary".to_owned(),
            model_revision: HubCommitV2::try_from("0123456789abcdef0123456789abcdef01234567")
                .expect("fixture commit"),
            normalization_revision: UtsegNormalizationRevisionV2::LowerStripAsciiPunctuationV1,
            adjacency_policy_revision:
                UtsegAdjacencyPolicyRevisionV2::SuppressEarlierAdjacentNonordinaryV1,
            word_evidence,
        };
        let request = UtsegBatchItem {
            words: words.iter().map(|w| (*w).to_owned()).collect(),
            text: words.join(" "),
        };
        admit_prediction(
            &request,
            assignments,
            UtsegPredictionOrigin::BoundaryModel(&evidence),
        )
        .expect("a worker-declared prediction is admitted")
    }

    /// The written CHAT of the pre-CHAT pass over one chunk and its
    /// prediction: cut, retokenize, finalize, build, serialize.
    fn written_chat(
        chunk: &PreparedMonologueChunk,
        prediction: &crate::utseg::AdmittedUtsegPrediction,
    ) -> String {
        let pieces = apply_prechat_segmentations(
            vec![chunk.clone()],
            std::slice::from_ref(prediction),
            "eng",
        )
        .expect("the prediction is the chunk's");
        let mut utterances = asr_postprocess::utterances_from_segmented_chunks(pieces);
        asr_postprocess::finalize_utterances(&mut utterances, "eng");
        let transcript =
            batchalign_transform::build_chat::NamedAsrUtterances::numbered(&utterances)
                .into_transcript(&["eng".to_owned()], None, false)
                .expect("transcript");
        let built = batchalign_transform::build_chat::build_chat(transcript.description)
            .expect("CHAT builds")
            .into_file();
        batchalign_transform::serialize::to_chat_string(&built)
    }

    /// RED FIRST: a model boundary after the first `I` of `yeah I I went`
    /// cut the chunk before retrace detection ran, so the repetition's copies
    /// were in different utterances and no `[/]` was written. The split now
    /// consumes the retrace analysis and keeps the copies together.
    #[test]
    fn a_model_boundary_inside_a_repetition_does_not_strand_the_retrace() {
        let words = ["yeah", "I", "I", "went"];
        let chat = written_chat(
            &chunk(&words),
            &admitted(&words, &[(1, UtsegBoundaryActionV2::PeriodBoundary)]),
        );
        assert!(chat.contains("*PAR0:\tYeah I [/] I went ."), "{chat}");
    }

    /// A model boundary writes the terminator it predicted on the utterance
    /// it closes (`?`, `!`), and the words after the last prediction end
    /// with a period. Until 2026-11-03 every cut wrote `.`.
    #[test]
    fn a_model_boundary_writes_the_terminator_it_predicted() {
        let words = ["what", "did", "you", "do", "okay", "yeah", "fine"];
        let chat = written_chat(
            &chunk(&words),
            &admitted(
                &words,
                &[
                    (3, UtsegBoundaryActionV2::QuestionBoundary),
                    (5, UtsegBoundaryActionV2::ExclamationBoundary),
                ],
            ),
        );
        assert!(chat.contains("*PAR0:\tWhat did you do ?"), "{chat}");
        assert!(chat.contains("*PAR0:\tOkay yeah !"), "{chat}");
        assert!(chat.contains("*PAR0:\tFine ."), "{chat}");
    }
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

#[cfg(test)]
mod abbreviation_tests {
    use super::*;
    use crate::params::MergeAbbrevPolicy;
    use crate::pipeline::post_validate::{PostValidated, ProducedOutput};
    use batchalign_transform::asr_postprocess::{AsrWord, SpeakerIndex};

    fn word(text: &str, start_ms: i64, end_ms: i64) -> AsrWord {
        AsrWord::new(text, Some(start_ms), Some(end_ms))
    }

    /// One speaker saying "the F B I is here ." with every letter timed.
    fn spoken_letters() -> Vec<PreparedMonologueChunk> {
        vec![PreparedMonologueChunk {
            speaker: SpeakerIndex(0),
            words: vec![
                word("the", 0, 200),
                word("F", 300, 400),
                word("B", 450, 520),
                word("I", 600, 700),
                word("is", 800, 900),
                word("here", 950, 1200),
                word(".", 1200, 1200),
            ],
        }]
    }

    /// Build the transcript the way transcribe does after this stage, and
    /// judge it with the producer transition transcribe's writer uses.
    fn transcript(chunks: Vec<PreparedMonologueChunk>) -> ProducedOutput {
        let utterances = punctuation_utterances(chunks, "eng", None);
        let description = batchalign_transform::build_chat::transcript_from_asr_utterances(
            &utterances,
            &["PAR".to_string()],
            &["eng".to_string()],
            Some("clip"),
            true,
        )
        .expect("the transcript description builds")
        .description;
        let file = batchalign_transform::build_chat::build_chat(description)
            .expect("CHAT builds")
            .into_file();
        PostValidated::produced(file, crate::api::ReleasedCommand::Transcribe)
    }

    /// The merged word reaches CHAT as one word whose `%wor` bullet is the
    /// union of its letters' intervals, and the document is admitted by the
    /// same gate every transcribe output passes.
    #[test]
    fn merged_letters_become_one_admitted_word_timed_by_their_union() {
        let (chunks, merging) = resolve_abbreviations(MergeAbbrevPolicy::Merge, spoken_letters());
        assert_eq!(merging, AbbreviationMerging::Ran);

        let produced = transcript(chunks);
        let ProducedOutput::Admitted(admitted) = &produced else {
            panic!("a transcript with a merged abbreviation is admitted: {produced:?}");
        };
        let chat = admitted.as_str();
        assert!(chat.contains("*PAR:\tThe FBI is here ."), "{chat}");
        assert!(chat.contains("FBI \u{15}300_700\u{15}"), "{chat}");
    }

    /// Without the option the letters reach CHAT as recognized, each with
    /// its own timing, and the step records that it was not asked for.
    #[test]
    fn without_the_option_the_letters_are_written_as_recognized() {
        let (chunks, merging) = resolve_abbreviations(MergeAbbrevPolicy::Keep, spoken_letters());
        assert_eq!(merging, AbbreviationMerging::NotRequested);
        let produced = transcript(chunks);
        let ProducedOutput::Admitted(admitted) = &produced else {
            panic!("the unmerged transcript is admitted: {produced:?}");
        };
        let chat = admitted.as_str();
        assert!(chat.contains("*PAR:\tThe F B I is here ."), "{chat}");
        assert!(chat.contains("F \u{15}300_400\u{15}"), "{chat}");
    }
}
