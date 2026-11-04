use super::*;

#[test]
fn dedicated_speaker_diarization_runs_when_backend_is_available_even_if_asr_has_labels() {
    let response = AsrResponse {
        seam_ambiguities: Vec::new(),
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

    assert_eq!(
        dedicated_diarization_to_run(
            crate::transcribe::SpeakerPlan::for_test(2, Some(SpeakerBackendV2::Pyannote)),
            &response
        ),
        Some(SpeakerBackendV2::Pyannote),
        "explicit diarization should still run even when ASR already carries first-pass speaker labels"
    );
}

#[test]
fn dedicated_speaker_diarization_skips_when_response_is_empty() {
    let response = AsrResponse {
        seam_ambiguities: Vec::new(),
        tokens: vec![],
        lang: LanguageCode3::eng(),
        model: None,
        source_monologues: None,
    };

    assert_eq!(
        dedicated_diarization_to_run(
            crate::transcribe::SpeakerPlan::for_test(2, Some(SpeakerBackendV2::Pyannote)),
            &response
        ),
        None,
        "empty ASR responses should not trigger dedicated speaker diarization"
    );
}

#[tokio::test]
async fn speaker_diarization_stage_skips_when_backend_is_unavailable() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let pool = WorkerPool::new(PoolConfig::for_tests());
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
            seam_ambiguities: Vec::new(),
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
    let pool = WorkerPool::new(PoolConfig::for_tests());
    let services = PipelineServices::new(&pool, &cache);
    let audio_path = tempdir.path().join("sample.wav");
    let mut opts = test_transcribe_options(None);
    opts.speakers = crate::transcribe::SpeakerPlan::for_test(2, None);
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
            seam_ambiguities: Vec::new(),
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
    let pool = WorkerPool::new(PoolConfig::for_tests());
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
            seam_ambiguities: Vec::new(),
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
    let pool = WorkerPool::new(PoolConfig::for_tests());
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
        seam_ambiguities: Vec::new(),
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
    let pool = WorkerPool::new(PoolConfig::for_tests());
    let services = PipelineServices::new(&pool, &cache);
    let audio_path = tempdir.path().join("sample.wav");

    // Opts with lang="auto": simulates --lang auto from CLI
    let mut opts = test_transcribe_options_in(None, crate::api::LanguageSpec::Auto);
    opts.speakers = crate::transcribe::SpeakerPlan::for_test(2, None);

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
            seam_ambiguities: Vec::new(),
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
    let pool = WorkerPool::new(PoolConfig::for_tests());
    let services = PipelineServices::new(&pool, &cache);
    let audio_path = tempdir.path().join("sample.wav");

    let mut opts = test_transcribe_options_in(None, crate::api::LanguageSpec::Auto);
    opts.speakers = crate::transcribe::SpeakerPlan::for_test(2, None);

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
            seam_ambiguities: Vec::new(),
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
    let pool = WorkerPool::new(PoolConfig::for_tests());
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
            seam_ambiguities: Vec::new(),
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

/// A repetition at a Whisper window seam that the reconciler kept once
/// (`worker::whisper_windows`) travels on the response through the lowering
/// (including the monotone fit of the overlapping copies), is placed on the
/// utterance that keeps the copy, is written there as a `%com`, and is the one
/// shortfall assembly records, located in the written document, so the file
/// finishes diagnosed (exit 7) rather than clean, and rather than refused.
#[tokio::test]
async fn a_window_seam_kept_once_is_written_as_a_comment_and_reported_as_a_shortfall() {
    use crate::worker::asr_result_v2::test_fixtures::{test_requested, window_response};

    let tempdir = tempfile::tempdir().expect("tempdir");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let pool = WorkerPool::new(PoolConfig::for_tests());
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
    // Window 0 heard "go" twice in the audio it shares with window 1 (owned
    // up to 22.0 s); window 1 (from 19.0 s, owning from 22.0 s) heard "go"
    // once there, at 20.0-22.0 s. Which of window 0's copies it repeats
    // cannot be told, so window 1's copy is dropped and the seam recorded.
    let response = crate::worker::asr_result_v2::parse_asr_response_v2(
        &window_response(
            serde_json::json!([
                {"text":"we", "start_s":19.0, "end_s":19.4},
                {"text":"go", "start_s":20.0, "end_s":22.0},
                {"text":"go", "start_s":20.5, "end_s":22.5},
                {"text":"now.", "start_s":24.0, "end_s":24.5}]),
            serde_json::json!([
                {"text":"go", "start_s":1.0, "end_s":3.0},
                {"text":"then", "start_s":11.0, "end_s":11.4},
                {"text":"home.", "start_s":11.5, "end_s":12.0}]),
        ),
        Some(&LanguageCode3::eng()),
        &test_requested(),
        crate::types::worker_v2::AsrBackendV2::LocalWhisper,
    )
    .expect("the recording is transcribed, with the seam recorded");
    assert_eq!(response.seam_ambiguities.len(), 1);
    let ctx = ctx.admit_response(response).expect("resolved ASR response");

    let ctx = stage_asr_postprocess(ctx).await.expect("postprocess");
    let placed: Vec<usize> = ctx
        .state
        .utterances
        .iter()
        .map(|utterance| utterance.seam_ambiguities.len())
        .collect();
    assert_eq!(
        placed.first(),
        Some(&1),
        "the record sits on the utterance that keeps the copy: {placed:?}"
    );
    assert!(
        placed.iter().skip(1).all(|&records| records == 0),
        "and on no other: {placed:?}"
    );

    let ctx = stage_build_chat(ctx).await.expect("build chat");
    let Built::Admitted(ChatReady {
        document,
        shortfalls,
        ..
    }) = ctx.state
    else {
        panic!("the transcript is legal CHAT; the seam is a shortfall, not a finding");
    };
    let chat = document.as_str();
    // Retrace detection reads the two kept copies as a repetition, which is
    // what they look like once the seam decision is made; the comment says
    // why there are two. The pause before "now" cuts the turn there, so the
    // utterance that keeps the copy ends with a synthesized period. The
    // comment is the line after it, and the only comment in the file.
    assert!(
        chat.contains(
            "*PAR0:\tWe go [/] go . \u{15}19000_22500\u{15}\n%com:\tbatchalign window seam at \
             22000 ms: repeated \"go\" kept once, review here\n"
        ),
        "{chat}"
    );
    assert_eq!(chat.matches("%com:").count(), 1, "{chat}");
    let [
        Shortfall::WindowSeamAmbiguous {
            seams: 1,
            first_seams,
        },
    ] = shortfalls.as_slice()
    else {
        panic!("one seam shortfall, got {shortfalls:?}");
    };
    let [record] = first_seams.as_slice() else {
        panic!("one seam record, got {first_seams:?}");
    };
    assert!(
        record.location.to_string().contains("We go [/] go ."),
        "the record locates the written utterance: {}",
        record.location
    );
    assert_eq!(record.seam_ms, 22_000);
    assert_eq!(record.text, "go");
    assert_eq!(record.dropped_from_window, 1);
    // The shortfall is what makes the written file diagnosed rather than
    // clean: the route the CLI reports as exit code 7.
    assert!(matches!(
        OutputReport::of(&ProducedOutput::Admitted(document), &shortfalls),
        OutputReport::Diagnosed(_)
    ));
}
