use super::*;

#[test]
fn dedicated_speaker_diarization_runs_when_backend_is_available_even_if_asr_has_labels() {
    let response = AsrResponse {
        tokens: vec![AsrToken {
            text: "hello".into(),
            start_s: at(0.0),
            end_s: at(0.5),
            speaker: Some("SPEAKER_1".into()),
            confidence: None,
        }],
        lang: LanguageCode3::eng(),
        model: None,
        source_monologues: None,
    };

    assert!(
        should_run_dedicated_speaker_diarization(&response, Some(SpeakerBackendV2::Pyannote)),
        "explicit diarization should still run even when ASR already carries first-pass speaker labels"
    );
}

#[test]
fn dedicated_speaker_diarization_skips_when_response_is_empty() {
    let response = AsrResponse {
        tokens: vec![],
        lang: LanguageCode3::eng(),
        model: None,
        source_monologues: None,
    };

    assert!(
        !should_run_dedicated_speaker_diarization(&response, Some(SpeakerBackendV2::Pyannote)),
        "empty ASR responses should not trigger dedicated speaker diarization"
    );
}

#[tokio::test]
async fn speaker_diarization_stage_skips_when_backend_is_unavailable() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let pool = WorkerPool::new(PoolConfig::default());
    let services = PipelineServices::new(&pool, &cache);
    let audio_path = tempdir.path().join("sample.wav");
    let opts = test_transcribe_options(None);
    let ctx = TranscribePipelineContext::new_with_rev_inference(
        &audio_path,
        services,
        &opts,
        DebugDumper::disabled(),
        UtsegEvidenceSink::Disabled,
        &PRODUCTION_REV_INFERENCE,
    );
    let mut ctx = ctx
        .admit_response(AsrResponse {
            tokens: vec![AsrToken {
                text: "hello".into(),
                start_s: at(0.0),
                end_s: at(0.5),
                speaker: None,
                confidence: None,
            }],
            lang: LanguageCode3::eng(),
            model: None,
            source_monologues: None,
        })
        .expect("resolved ASR response");

    stage_speaker_diarization(&mut ctx)
        .await
        .expect("speaker stage should succeed");

    assert!(
        ctx.speaker_segments.is_none(),
        "dedicated speaker inference should be skipped when no speaker backend is configured"
    );
}

/// `--no-utseg` controls the whole transcribe segmentation execution, not
/// only the optional CHAT-level stage. English is deliberately used here:
/// with segmentation enabled these two words require a worker request, and
/// this pool has no worker to satisfy one.
#[tokio::test]
async fn no_utseg_bypasses_the_pre_chat_worker_for_a_supported_language() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let pool = WorkerPool::new(PoolConfig::default());
    let services = PipelineServices::new(&pool, &cache);
    let audio_path = tempdir.path().join("sample.wav");
    let mut opts = test_transcribe_options(None);
    opts.diarize = false;
    let ctx = TranscribePipelineContext::new_with_rev_inference(
        &audio_path,
        services,
        &opts,
        DebugDumper::disabled(),
        UtsegEvidenceSink::Disabled,
        &PRODUCTION_REV_INFERENCE,
    );
    let ctx = ctx
        .admit_response(AsrResponse {
            tokens: vec![
                AsrToken {
                    text: "hello".into(),
                    start_s: at(0.0),
                    end_s: at(0.5),
                    speaker: None,
                    confidence: None,
                },
                AsrToken {
                    text: "there".into(),
                    start_s: at(0.5),
                    end_s: at(1.0),
                    speaker: None,
                    confidence: None,
                },
            ],
            lang: LanguageCode3::eng(),
            model: None,
            source_monologues: None,
        })
        .expect("resolved ASR response");

    let ctx = stage_asr_postprocess(ctx)
        .await
        .expect("disabled segmentation must not dispatch a worker");

    assert_eq!(Some(ctx.state.utterances.len()), Some(1));
}

/// Dedicated diarization is available while ASR words still carry their
/// observed timings. A speaker boundary between two words must therefore
/// constrain utterance segmentation before CHAT is built, rather than
/// relabeling the already-mixed utterance afterward.
#[tokio::test]
async fn diarization_boundary_splits_timed_asr_words_before_chat_build() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let pool = WorkerPool::new(PoolConfig::default());
    let services = PipelineServices::new(&pool, &cache);
    let audio_path = tempdir.path().join("sample.wav");
    let opts = test_transcribe_options_in(
        Some(SpeakerBackendV2::Pyannote),
        LanguageCode3::fra().into(),
    );

    let ctx = TranscribePipelineContext::new_with_rev_inference(
        &audio_path,
        services,
        &opts,
        DebugDumper::disabled(),
        UtsegEvidenceSink::Disabled,
        &PRODUCTION_REV_INFERENCE,
    );
    let mut ctx = ctx
        .admit_response(AsrResponse {
            tokens: vec![
                AsrToken {
                    text: "bonjour".into(),
                    start_s: at(0.0),
                    end_s: at(0.5),
                    speaker: Some("ASR_0".into()),
                    confidence: None,
                },
                AsrToken {
                    text: "oui".into(),
                    start_s: at(0.5),
                    end_s: at(1.0),
                    speaker: Some("ASR_0".into()),
                    confidence: None,
                },
                AsrToken {
                    text: ".".into(),
                    start_s: None,
                    end_s: None,
                    speaker: Some("ASR_0".into()),
                    confidence: None,
                },
            ],
            lang: LanguageCode3::fra(),
            model: None,
            source_monologues: None,
        })
        .expect("resolved ASR response");
    ctx.speaker_segments = Some(vec![
        SpeakerSegmentV2 {
            interval: batchalign_types::interval::AdmittedInterval::admit_millis(0, 500)
                .expect("an ordered fixture"),
            speaker: "HUMAN_A".into(),
        },
        SpeakerSegmentV2 {
            interval: batchalign_types::interval::AdmittedInterval::admit_millis(500, 1_000)
                .expect("an ordered fixture"),
            speaker: "HUMAN_B".into(),
        },
    ]);

    let ctx = stage_asr_postprocess(ctx).await.expect("postprocess");
    let SpeakerAssignmentOutcome::Dedicated(evidence) = &ctx.state.speaker_assignments else {
        panic!("dedicated projection must carry its producer evidence");
    };
    assert_eq!(evidence.observations().summary().directly_supported, 2);
    assert_eq!(evidence.observations().summary().inferred, 1);
    let ctx = stage_build_chat(ctx).await.expect("build chat");

    let SpeakerAssignmentOutcome::Dedicated(evidence) = ctx.state.speaker_assignments() else {
        panic!("CHAT assembly must retain the assignment evidence");
    };
    assert!(evidence.needs_review());

    let chat = ctx.state.as_str().to_owned();
    assert!(chat.contains("*PAR0:\tbonjour ."), "{chat}");
    assert!(chat.contains("*PAR1:\toui ."), "{chat}");
    assert_eq!(chat.lines().filter(|line| line.starts_with('*')).count(), 2);
    assert!(
        chat.contains("Speaker projection requires review"),
        "{chat}"
    );
}

/// Requested evidence is an output obligation, not a best-effort log. A
/// deterministic blocked directory refuses the transition before CHAT build.
#[tokio::test]
async fn transcribe_speaker_projection_refuses_unwritable_requested_evidence() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let blocked = tempdir.path().join("blocked");
    std::fs::write(&blocked, b"not a directory").expect("blocked directory fixture");
    let audio = tempdir.path().join("sample.wav");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let pool = WorkerPool::new(PoolConfig::default());
    let opts = test_transcribe_options(Some(SpeakerBackendV2::Pyannote));
    let mut ctx = TranscribePipelineContext::new_with_rev_inference(
        &audio,
        PipelineServices::new(&pool, &cache),
        &opts,
        DebugDumper::new(Some(&blocked)),
        UtsegEvidenceSink::Disabled,
        &PRODUCTION_REV_INFERENCE,
    )
    .admit_response(AsrResponse {
        tokens: vec![AsrToken {
            text: "hello".into(),
            start_s: at(0.0),
            end_s: at(0.5),
            speaker: Some("ASR_0".into()),
            confidence: None,
        }],
        lang: LanguageCode3::eng(),
        model: None,
        source_monologues: None,
    })
    .expect("recognized response");
    ctx.speaker_segments = Some(vec![SpeakerSegmentV2 {
        interval: batchalign_types::interval::AdmittedInterval::admit_millis(0, 500)
            .expect("segment"),
        speaker: "A".into(),
    }]);
    let result = stage_asr_postprocess(ctx).await;
    assert!(matches!(result, Err(ServerError::Persistence(message))
        if message.contains("speaker projection evidence")));
}

/// Both preparation entry points give one answer.
///
/// The untraced path IS the transform's function now, so this holds the
/// traced path to it. Number expansion is where the two could drift: the
/// traced path skips `expand_number` for tokens carrying no ASCII digit, so
/// a token that expanded without one would diverge silently.
#[test]
fn snapshot_and_plain_preparation_agree() {
    use crate::api::AudioPositionSeconds;
    use batchalign_transform::asr_postprocess::{
        AsrElement, AsrElementKind, AsrMonologue, AsrOutput, AsrRawText, SpeakerIndex,
    };

    let element = |value: &str, ts: f64, end_ts: f64| AsrElement {
        value: AsrRawText::new(value),
        ts: Some(AudioPositionSeconds::try_from(ts).expect("fixture position")),
        end_ts: Some(AudioPositionSeconds::try_from(end_ts).expect("fixture position")),
        kind: AsrElementKind::Text,
    };
    let output = AsrOutput {
        monologues: vec![AsrMonologue {
            speaker: SpeakerIndex(0),
            elements: vec![
                element("I", 0.0, 0.2),
                element("counted", 0.2, 0.6),
                element("100", 0.6, 1.0),
                element("sheep", 1.0, 1.4),
                element("in", 1.4, 1.6),
                element("2001", 1.6, 2.2),
                element(".", 2.2, 2.3),
            ],
        }],
    };

    let plain = prepare_asr_chunks(&output, "eng")
        .expect("test: ASR post-processing must not refuse this input");
    let mut snapshot = AsrPipelineSnapshot::default();
    let traced = prepare_asr_chunks_with_snapshot(
        &output,
        asr_postprocess::AsrTextLanguage::One("eng"),
        Some(&mut snapshot),
    )
    .expect("test: ASR post-processing must not refuse this input");

    assert!(
        plain.iter().any(|chunk| chunk
            .words
            .iter()
            .any(|word| word.text.as_str() == "hundred")),
        "the fixture must actually expand a number, or it proves nothing: {plain:?}"
    );
    assert_eq!(plain, traced, "one preparation step, two entry points");

    // The same holds for code-switched text, where both entry points must
    // keep the numerals rather than write them out.
    let switched = asr_postprocess::AsrTextLanguage::CodeSwitched { primary: "eng" };
    let plain = prepare_asr_chunks(&output, switched)
        .expect("test: ASR post-processing must not refuse this input");
    let mut snapshot = AsrPipelineSnapshot::default();
    let traced = prepare_asr_chunks_with_snapshot(&output, switched, Some(&mut snapshot))
        .expect("test: ASR post-processing must not refuse this input");
    assert!(
        plain
            .iter()
            .any(|chunk| chunk.words.iter().any(|word| word.text.as_str() == "100")),
        "{plain:?}"
    );
    assert_eq!(plain, traced, "one preparation step, two entry points");
}

/// When the request is detection, stage_build_chat must resolve to the
/// ASR-detected language for CHAT headers (regression test for job
/// 696870c7-02b where `@Languages: auto` leaked into output).
#[tokio::test]
async fn build_chat_stage_resolves_auto_to_detected_language() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let pool = WorkerPool::new(PoolConfig::default());
    let services = PipelineServices::new(&pool, &cache);
    let audio_path = tempdir.path().join("sample.wav");

    // Opts with lang="auto": simulates --lang auto from CLI
    let mut opts = test_transcribe_options_in(None, crate::api::LanguageSpec::Auto);
    opts.diarize = false;

    let ctx = TranscribePipelineContext::new_with_rev_inference(
        &audio_path,
        services,
        &opts,
        DebugDumper::disabled(),
        UtsegEvidenceSink::Disabled,
        &PRODUCTION_REV_INFERENCE,
    );

    // ASR response with detected language "spa"
    let ctx = ctx
        .admit_response(AsrResponse {
            tokens: vec![AsrToken {
                text: "hola".into(),
                start_s: at(0.0),
                end_s: at(0.5),
                speaker: None,
                confidence: None,
            }],
            lang: LanguageCode3::spa(),
            model: None,
            source_monologues: None,
        })
        .expect("resolved ASR response");

    // Run post-processing to generate utterances
    let ctx = stage_asr_postprocess(ctx).await.expect("postprocess");

    // Run build_chat; this should resolve "auto" → "spa"
    let ctx = stage_build_chat(ctx).await.expect("build_chat");

    let chat_text = ctx.state.as_str();

    // The @Languages header must contain the detected language, NOT "auto"
    let languages_line = chat_text
        .lines()
        .find(|l| l.starts_with("@Languages:"))
        .expect("@Languages header missing");
    assert!(
        languages_line.contains("spa"),
        "@Languages should contain detected 'spa', got: {languages_line}"
    );
    assert!(
        !languages_line.contains("auto"),
        "@Languages must NOT contain sentinel 'auto', got: {languages_line}"
    );
}

#[tokio::test]
async fn postprocess_stage_resolves_auto_before_chat_build() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let pool = WorkerPool::new(PoolConfig::default());
    let services = PipelineServices::new(&pool, &cache);
    let audio_path = tempdir.path().join("sample.wav");

    let mut opts = test_transcribe_options_in(None, crate::api::LanguageSpec::Auto);
    opts.diarize = false;

    let ctx = TranscribePipelineContext::new_with_rev_inference(
        &audio_path,
        services,
        &opts,
        DebugDumper::disabled(),
        UtsegEvidenceSink::Disabled,
        &PRODUCTION_REV_INFERENCE,
    );
    let ctx = ctx
        .admit_response(AsrResponse {
            tokens: vec![AsrToken {
                text: "hola".into(),
                start_s: at(0.0),
                end_s: at(0.5),
                speaker: None,
                confidence: None,
            }],
            lang: LanguageCode3::spa(),
            model: None,
            source_monologues: None,
        })
        .expect("resolved ASR response");

    let ctx = stage_asr_postprocess(ctx).await.expect("postprocess");
    assert_eq!(
        ctx.state.asr.language,
        TranscriptLanguage::One(LanguageCode3::spa())
    );
}

/// RED FIRST (2026-09-16): an ASR response with no tokens refuses the job,
/// naming the stage that produced nothing.
///
/// This used to return `Ok(())`, and `stage_build_chat` then wrote a
/// headers-only CHAT file, so a run that recognized not one word finished
/// as a completed job carrying an empty transcript.
///
/// It replaces `build_chat_stage_resolves_auto_for_empty_response`, which
/// asserted that the empty file carried the resolved language rather than
/// `auto`. There is no empty file to make that assertion about any more.
/// The property that test guarded, that `auto` is never stamped into
/// `@Languages`, is held on the path that still writes a file by
/// `build_chat_stage_resolves_auto_to_detected_language`.
#[tokio::test]
async fn an_asr_response_with_no_tokens_is_refused_naming_the_asr_stage() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let pool = WorkerPool::new(PoolConfig::default());
    let services = PipelineServices::new(&pool, &cache);
    let audio_path = tempdir.path().join("sample.wav");

    let opts = test_transcribe_options_in(None, crate::api::LanguageSpec::Auto);

    let ctx = TranscribePipelineContext::new_with_rev_inference(
        &audio_path,
        services,
        &opts,
        DebugDumper::disabled(),
        UtsegEvidenceSink::Disabled,
        &PRODUCTION_REV_INFERENCE,
    );
    let ctx = ctx
        .admit_response(AsrResponse {
            tokens: vec![],
            lang: LanguageCode3::fra(),
            model: None,
            source_monologues: None,
        })
        .expect("resolved ASR response");

    let refusal = stage_asr_postprocess(ctx)
        .await
        .err()
        .expect("an ASR response with no tokens must refuse the job");
    assert!(
        matches!(
            refusal,
            ServerError::EmptyTranscription(EmptyTranscription::Asr)
        ),
        "the refusal must name the ASR stage, got: {refusal:?}"
    );
}
