//! ASR and dedicated speaker inference stages.

use super::*;

pub(super) async fn stage_asr_infer<'a>(
    mut ctx: TranscribePipelineContext<'a, PendingAsr>,
    rev_inference: &dyn RevAsrEvidenceInference,
) -> Result<TranscribePipelineContext<'a, Recognized>, ServerError> {
    info!(
        audio_path = %ctx.audio_path.display(),
                                lang = %ctx.opts.language(),
        expected_speakers = ?ctx.opts.expected_speakers(),
        "Starting ASR inference"
    );

    let opts = ctx.opts;
    let mut evidence_language = None;
    let response = match opts.plan.asr().inference() {
        crate::transcribe::AsrInference::NonRev {
            backend,
            speakers,
            language,
        } => {
            // Measured ONCE, here, and only when the request needs it: a
            // provider-media backend sizes its decode budget from the file's
            // length, while a prepared-audio backend derives the budget from
            // the audio it prepares. The request builder used to probe the file
            // itself (a whole-file walk for MP3 or ADTS), out of this
            // pipeline's sight. `probe_audio_duration` is reused rather than
            // probing here, so there is one owner of "measure the length and
            // log the failure".
            let decode_budget = match backend {
                NonRevAsrBackend::Worker(mode) if mode.reads_provider_media() => {
                    crate::runner::util::probe_audio_duration(ctx.audio_path)
                        .await
                        .map(|duration| DecodeBudgetSeconds::for_duration_ms(duration.length().0))
                }
                NonRevAsrBackend::Worker(_) | NonRevAsrBackend::RustWhisperRs => None,
            };
            infer_asr(
                ctx.services.pool,
                &AsrInferParams {
                    backend,
                    audio_path: ctx.audio_path,
                    lang: language,
                    num_speakers: NumSpeakers(speakers.get()),
                    extras: &ctx.opts.engine_extras,
                    decode_budget,
                },
            )
            .await?
        }
        crate::transcribe::AsrInference::RevAi { language } => {
            let provider_media = PreparedRevProviderMedia::from_source(ctx.audio_path)
                .await
                .map_err(|error| ServerError::Persistence(error.to_string()))?;
            let request = RevAsrEvidenceRequest::new(
                provider_media,
                language,
                ctx.opts.expected_speakers(),
                &RevAsrModelRevision::current(),
            )
            .map_err(|error| ServerError::Persistence(error.to_string()))?;
            let resolution = resolve_rev_asr_evidence(
                &request,
                ctx.services.cache,
                ctx.opts.cache_policies.rev_asr,
                rev_inference,
            )
            .await
            .map_err(rev_asr_resolution_error_to_server_error)?;
            let trace = resolution.trace(RevAsrProjectionRevision::AsrResponseV1);
            if ctx.dumper.is_enabled() {
                ctx.dumper
                    .dump_rev_evidence(ctx.audio_path.to_string_lossy().as_ref(), &trace)
                    .map_err(|error| ServerError::Persistence(error.to_string()))?;
            }
            ctx.rev_evidence = Some(trace);
            match resolution.source() {
                RevAsrEvidenceSource::Replayed => {
                    info!(
                        cache_key = %request.cache_key(),
                        "Replaying validated raw Rev.AI transcript evidence"
                    );
                }
                RevAsrEvidenceSource::Inferred(reason) => {
                    info!(
                        cache_key = %request.cache_key(),
                        reason = ?reason,
                        "Committed fresh raw Rev.AI transcript evidence"
                    );
                }
            }
            let evidence = resolution.into_evidence();
            // The evidence says what language its transcript is in; that is
            // the transcript's language, not a second reading of the request.
            evidence_language = Some(evidence.resolved_language().clone());
            rev_evidence_to_asr_response(&evidence)?
        }
    };
    let filename = ctx
        .audio_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");
    ctx.dumper.dump_asr_response(filename, &response);
    let language = match evidence_language {
        Some(language) => language,
        None => resolved_asr_language(opts, &response)?,
    };
    let planned = opts.plan.asr().identity();
    let identity = match response.model.clone() {
        Some(model) => planned.with_loaded_models(model),
        None => planned,
    };
    let recognized = Recognized::admit(response, language, identity, &opts.plan)?;
    Ok(ctx.map_state(|_| recognized))
}

pub(super) async fn stage_speaker_diarization<P: TranscribePlan>(
    ctx: &mut TranscribePipelineContext<'_, Recognized, P>,
) -> Result<(), ServerError> {
    if P::REPLAY {
        return Ok(());
    }
    let response = &ctx.state.response;
    if !should_run_dedicated_speaker_diarization(response, ctx.opts.speaker_backend) {
        return Ok(());
    }

    // Control-flow invariant: `should_run_dedicated_speaker_diarization`
    // immediately above returns false when `ctx.opts.speaker_backend`
    // is `None`, taking the early-return branch. Reaching this
    // line therefore guarantees `Some(...)`.
    #[allow(clippy::expect_used)]
    let speaker_backend = ctx
        .opts
        .speaker_backend
        .expect("speaker backend presence checked above");

    // Speaker workers require a concrete routing language. The request may
    // be detection even after ASR, so the value comes from the context's
    // one resolution: the transcript's primary language.
    let speaker_worker_lang = WorkerLanguage::from(ctx.state.language.primary());
    info!(
        audio_path = %ctx.audio_path.display(),
        speaker_backend = ?speaker_backend,
        expected_speakers = ?ctx.opts.expected_speakers(),
        "Running dedicated speaker diarization"
    );
    let expected_speakers = ctx.opts.expected_speakers();
    let cache_policy = ctx.opts.cache_policies.speaker;
    let resolution = resolve_speaker_evidence_for_audio(
        ctx.services.pool,
        ctx.services.cache,
        speaker_worker_lang,
        SpeakerEvidenceRunParams {
            audio_path: ctx.audio_path,
            backend: speaker_backend,
            expected_speakers,
            cache_policy,
        },
    )
    .await?;
    let evidence_identity = ctx.audio_path.to_string_lossy().into_owned();
    let trace = resolution.trace();
    ctx.dumper
        .dump_speaker_evidence(&evidence_identity, &trace)
        .map_err(|error| {
            ServerError::Persistence(format!(
                "could not retain requested speaker evidence trace for {evidence_identity}: {error}"
            ))
        })?;
    let num_segments = resolution.segments().len();
    match resolution.source() {
        SpeakerEvidenceSource::ReplayedDerived => {
            info!(
                cache_key = %resolution.cache_key(),
                num_segments,
                "Replaying validated speaker diarization evidence"
            );
        }
        SpeakerEvidenceSource::DerivedFromRaw => {
            info!(
                cache_key = %resolution.cache_key(),
                num_segments,
                "Derived speaker segments from retained raw evidence"
            );
        }
        SpeakerEvidenceSource::Inferred(reason) => {
            info!(
                cache_key = %resolution.cache_key(),
                reason = ?reason,
                num_segments,
                "Committed fresh speaker diarization evidence"
            );
        }
    }
    let segments = resolution.into_segments();
    info!(
        num_segments = segments.len(),
        "Speaker diarization complete"
    );
    ctx.dumper
            .dump_speaker_turns(&evidence_identity, speaker_backend, &segments)
            .map_err(|error| {
                ServerError::Persistence(format!(
                    "could not retain requested same-job diarization evidence for {evidence_identity}: {error}"
                ))
            })?;
    ctx.speaker_segments = Some(segments);
    Ok(())
}
