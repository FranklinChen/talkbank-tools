//! Typed CHAT assembly and output admission (the producer transition).

use super::*;

pub(super) async fn stage_build_chat<'a, P: TranscribePlan>(
    mut ctx: TranscribePipelineContext<'a, Postprocessed, P>,
) -> Result<TranscribePipelineContext<'a, Built, P>, ServerError> {
    let resolved_lang = ctx.state.asr.language.clone();
    let utterances = &mut ctx.state.utterances;
    // Under detection, run per-utterance language detection for
    // code-switching markup and multi-language headers. A requested
    // language or pair is declared as requested.
    let langs: Vec<String> = match ctx.opts.language() {
        AsrLanguageRequest::Detect => {
            use batchalign_transform::asr_postprocess::lang_detect;

            // Concatenate each utterance's words for language detection
            let utt_texts: Vec<String> = utterances
                .iter()
                .map(|utt| {
                    utt.words
                        .iter()
                        .map(|w| w.text.as_str())
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .collect();
            let utt_text_refs: Vec<&str> = utt_texts.iter().map(String::as_str).collect();

            // One decision for the tags and the header that declares them:
            // an utterance is tagged only with a language the header lists.
            let (declared, tags) =
                lang_detect::UtteranceLanguages::detect(&utt_text_refs, resolved_lang.primary())
                    .into_parts();
            for (utt, tag) in utterances.iter_mut().zip(tags) {
                utt.lang = tag;
            }
            declared
        }
        AsrLanguageRequest::One(_) | AsrLanguageRequest::Pair(_) => resolved_lang
            .declared()
            .into_iter()
            .map(ToString::to_string)
            .collect(),
    };

    let transcript = build_chat::NamedAsrUtterances::numbered(utterances).into_transcript(
        &langs,
        ctx.opts.media_name.as_deref(),
        ctx.opts.write_wor,
    )?;

    // The gate hands back every token it refused and we emitted anyway.
    // Saying so here is the point: an operator reading this run now learns
    // that the transcript is knowingly invalid CHAT and why, instead of
    // finding out from `chatter validate` days later with no way to tell
    // an ASR artifact from a transcription error. The tokens themselves
    // stay verbatim, which is this pipeline's standing policy: what the
    // speaker actually said is a human's call, not the pipeline's.
    report_language_invalid_words(&ctx, &transcript.language_invalid);
    let desc = transcript.description;

    let built = build_chat::build_chat(desc).map_err(|error| match error {
        // Neither malformed input nor an internal fault: words reached CHAT
        // assembly and none of them was content, which is what a Cantonese
        // engine that recognized only punctuation produces. Carried as the
        // typed emptiness so the failure names the stage, rather than being
        // flattened into a validation message.
        build_chat::BuildChatError::NoUtterances => {
            ServerError::EmptyTranscription(EmptyTranscription::ChatBuild { described: 0 })
        }
        build_chat::BuildChatError::NoUtteranceSurvivedBuild { described } => {
            ServerError::EmptyTranscription(EmptyTranscription::ChatBuild { described })
        }
        other => ServerError::Validation(format!("Failed to build CHAT: {other}")),
    })?;
    report_synthesized_terminators(&ctx, &built.synthesized_ends);
    let synthesized_terminators = std::num::NonZeroUsize::new(built.synthesized_ends.len());
    let mut chat_file = built.into_file();
    // Inject processing provenance comment. It also counts the utterances
    // assembly closed with a terminator this pipeline added, which their `.`
    // cannot say itself; a later utseg split's cuts are utseg's to record.
    let asr = ctx.asr_identity();
    let provenance = crate::provenance::transcribe_provenance(
        &resolved_lang,
        &asr,
        ctx.opts.diarize,
        ctx.opts.write_wor,
        ctx.state.word_segmenter.as_ref(),
        synthesized_terminators,
    );
    crate::provenance::inject_provenance(&mut chat_file, &provenance);

    // Inject human-readable "unchecked ASR" warning (a user's workflow depends on this).
    crate::provenance::inject_unchecked_warning(&mut chat_file, &asr);

    // CHAT construction validity is not acoustic speaker accuracy. Retain the
    // producer-derived caveat even when no debug directory was requested.
    if let Some(warning) = ctx.state.speaker_assignments.review_warning() {
        let position = chat_file
            .lines
            .iter()
            .position(|line| matches!(line, crate::chat_ops::Line::Utterance(_)))
            .unwrap_or(chat_file.lines.len());
        chat_file.lines.insert(
            position,
            crate::chat_ops::Line::header(
                talkbank_model::Location::Unlocated,
                crate::chat_ops::Header::Comment {
                    content: crate::chat_ops::BulletContent::from_text(warning),
                },
            ),
        );
    }

    // The generated document is OUR output, so it is judged by the producer
    // transition: the same judgement as every gate, but a document that does
    // not pass is kept, with every finding, and written as diagnosed instead
    // of making the whole transcript vanish behind a file error. One ASR
    // token such as `b2` (E220) or `Www` (E241), kept verbatim above for
    // human review, is exactly that case.
    let document = crate::pipeline::post_validate::PostValidated::produced(
        chat_file,
        crate::api::ReleasedCommand::Transcribe,
    );
    let filename = ctx
        .audio_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");
    ctx.dumper.dump_post_asr_chat(filename, document.as_str());
    Ok(ctx.map_state(|state| match document {
        crate::pipeline::post_validate::ProducedOutput::Admitted(document) => {
            Built::Admitted(ChatReady {
                asr: state.asr,
                document,
                speaker_assignments: state.speaker_assignments,
                shortfalls: Vec::new(),
            })
        }
        crate::pipeline::post_validate::ProducedOutput::Diagnosed(document) => {
            let findings = document.findings();
            tracing::warn!(
                %filename,
                findings = findings.count(),
                first = %findings.first(),
                "generated transcript did not pass output admission; it is written \
                 with its diagnostics, and later stages run only where they can"
            );
            Built::Diagnosed(ChatDiagnosed {
                asr: state.asr,
                document,
                speaker_assignments: state.speaker_assignments,
            })
        }
    }))
}

/// One refused token, in the shape written to the run's debug artifacts.
#[derive(serde::Serialize)]
struct LanguageInvalidWordRecord<'a> {
    utterance: usize,
    word: usize,
    speaker: &'a str,
    /// The provider's surface, verbatim, exactly as it was emitted.
    text: &'a str,
    lang: &'a str,
    /// chatter error codes, e.g. `E220` (digits) or `E241` (reserved
    /// untranscribed marker).
    codes: Vec<&'a str>,
    messages: Vec<&'a str>,
}

/// Say how many utterances end with a terminator this pipeline added.
///
/// The `.` such an utterance is written with cannot say so itself: CHAT has
/// no notation for a terminator nobody heard. The durable record is the
/// count in the file's provenance stamp; this line names the first one by
/// its position among the written utterances, so a reader of the run can
/// find it.
fn report_synthesized_terminators<S, P: TranscribePlan>(
    ctx: &TranscribePipelineContext<'_, S, P>,
    synthesized: &[build_chat::SynthesizedEnd],
) {
    let Some(first) = synthesized.first() else {
        return;
    };
    tracing::info!(
        file = %ctx.audio_path.display(),
        synthesized = synthesized.len(),
        first_utterance = first.utterance,
        first_speaker = %first.speaker,
        "utterances closed with a synthesized terminator: the recognizer gave no ending \
         punctuation, and CHAT requires one"
    );
}

/// Narrate, and durably record, every word the CHAT-legality gate refused.
///
/// Deliberately not fatal: emitting the provider's surface for human review is
/// the pipeline's standing policy. What was missing until 2026-09-03 was any
/// record at all, so `Www` (E241) and `b2` (E220) reached a corpus with the
/// only evidence in a log line nobody was reading.
fn report_language_invalid_words<S, P: TranscribePlan>(
    ctx: &TranscribePipelineContext<'_, S, P>,
    refused: &[build_chat::LanguageInvalidWord],
) {
    if refused.is_empty() {
        return;
    }

    let records: Vec<LanguageInvalidWordRecord<'_>> = refused
        .iter()
        .map(|word| LanguageInvalidWordRecord {
            utterance: word.utt_idx,
            word: word.word_idx,
            speaker: word.speaker_id.as_str(),
            text: word.text.as_str(),
            lang: word.lang.as_str(),
            codes: word
                .parse_errors
                .iter()
                .map(|error| error.code.as_str())
                .collect(),
            messages: word
                .parse_errors
                .iter()
                .map(|error| error.message.as_str())
                .collect(),
        })
        .collect();

    let mut codes: Vec<&str> = records
        .iter()
        .flat_map(|r| r.codes.iter().copied())
        .collect();
    codes.sort_unstable();
    codes.dedup();

    let filename = ctx
        .audio_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("unknown");
    tracing::warn!(
        %filename,
        refused_words = records.len(),
        codes = %codes.join(","),
        "transcript emitted with words the CHAT language rules refuse; \
         surfaces kept verbatim for human review"
    );
    ctx.dumper.dump_language_invalid_words(filename, &records);
}
