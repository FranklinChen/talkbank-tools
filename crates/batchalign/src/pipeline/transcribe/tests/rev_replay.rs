use super::*;

/// A durable Rev cache hit must replay the same evidence through the full
/// Rust post-processing and CHAT construction path. The causal receipt is
/// intentionally different (`inferred_not_found` versus `replayed`), but
/// its typed semantic projection and every downstream output are stable.
#[tokio::test]
async fn rev_cold_and_replayed_transcribe_are_semantically_identical() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let audio_path = tempdir.path().join("sample.wav");
    tokio::fs::write(&audio_path, b"provider media")
        .await
        .expect("write provider media");
    let cache_dir = tempdir.path().join("cache");
    let cold_debug = tempdir.path().join("cold-debug");
    let replay_debug = tempdir.path().join("replay-debug");
    let pool = WorkerPool::new(PoolConfig::default());
    let inference = CountingRevInference {
        calls: AtomicUsize::new(0),
    };
    let mut opts = test_transcribe_options(None);
    opts.diarize = false;

    let cache = UtteranceCache::sqlite(Some(cache_dir.clone()))
        .await
        .expect("cold cache");
    let cold = run_transcribe_pipeline_with_rev_inference(
        &audio_path,
        PipelineServices::new(&pool, &cache),
        &opts,
        None,
        Some(&cold_debug),
        &inference,
    )
    .await
    .expect("cold transcribe");
    drop(cache);

    let reopened = UtteranceCache::sqlite(Some(cache_dir))
        .await
        .expect("reopened cache");
    let replayed = run_transcribe_pipeline_with_rev_inference(
        &audio_path,
        PipelineServices::new(&pool, &reopened),
        &opts,
        None,
        Some(&replay_debug),
        &inference,
    )
    .await
    .expect("replayed transcribe");

    assert_eq!(inference.calls.load(Ordering::SeqCst), 1);
    assert_same_transcribe_semantics(
        cold.output.document.as_str(),
        replayed.output.document.as_str(),
    );
    assert_eq!(
        std::fs::read(only_debug_artifact(&cold_debug, "_asr_response.json"))
            .expect("cold ASR artifact"),
        std::fs::read(only_debug_artifact(&replay_debug, "_asr_response.json"))
            .expect("replayed ASR artifact")
    );

    let cold_trace = cold.rev_evidence.expect("cold Rev trace");
    let replay_trace = replayed.rev_evidence.expect("replayed Rev trace");
    assert_eq!(
        cold_trace.cache_outcome(),
        RevAsrEvidenceCacheOutcome::InferredNotFound
    );
    assert_eq!(
        replay_trace.cache_outcome(),
        RevAsrEvidenceCacheOutcome::Replayed
    );
    assert_eq!(
        cold_trace.semantic_projection(),
        replay_trace.semantic_projection()
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(
            &std::fs::read(only_debug_artifact(&cold_debug, "_rev_evidence.json"))
                .expect("cold Rev trace artifact")
        )
        .expect("cold Rev trace JSON"),
        serde_json::to_value(&cold_trace).expect("cold typed trace JSON")
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(
            &std::fs::read(only_debug_artifact(&replay_debug, "_rev_evidence.json"))
                .expect("replayed Rev trace artifact")
        )
        .expect("replayed Rev trace JSON"),
        serde_json::to_value(&replay_trace).expect("replayed typed trace JSON")
    );
}

/// Rev.AI evidence for a code-switched English/Spanish request.
/// The bilingual replay fixture: one Spanish word, one English word, an
/// unresolved bare numeral and a terminator.
fn bilingual_replay_tokens() -> Vec<AsrToken> {
    vec![
        AsrToken {
            text: "quiero".into(),
            start_s: at(0.1),
            end_s: at(0.4),
            speaker: Some("ASR_0".into()),
            confidence: Some(0.9),
        },
        AsrToken {
            text: "water".into(),
            start_s: at(0.5),
            end_s: at(0.9),
            speaker: Some("ASR_0".into()),
            confidence: Some(0.8),
        },
        AsrToken {
            text: "25".into(),
            start_s: at(0.9),
            end_s: at(1.2),
            speaker: Some("ASR_0".into()),
            confidence: Some(0.8),
        },
        AsrToken {
            text: ".".into(),
            start_s: None,
            end_s: None,
            speaker: Some("ASR_0".into()),
            confidence: None,
        },
    ]
}

/// Run the offline legacy replay through the whole transcribe pipeline.
async fn replay(tokens: Vec<AsrToken>) -> Result<String, ServerError> {
    replay_output(tokens)
        .await
        .map(|output| output.document.into_text())
}

/// [`replay`], keeping the proof so a test can read its standing.
async fn replay_output(tokens: Vec<AsrToken>) -> Result<TranscribeOutput, ServerError> {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let media = tempdir.path().join("sample.wav");
    let asr = tempdir.path().join("sample_asr_response.json");
    let manifest = tempdir.path().join("sample.replay.json");
    std::fs::write(&media, b"fingerprinted media").expect("media");
    std::fs::write(
        &asr,
        serde_json::to_vec_pretty(&AsrResponse {
            tokens,
            lang: LanguageCode3::eng(),
            model: None,
            source_monologues: None,
        })
        .expect("ASR JSON"),
    )
    .expect("ASR artifact");
    write_legacy_replay_manifest(
        LegacyReplayManifestRequest {
            recording_id: "sample",
            media_path: &media,
            asr_response_path: &asr,
            speaker_turns_path: None,
            producer: LegacyProjectedAsrProducer::RevAi,
        },
        &manifest,
    )
    .expect("manifest");
    let replay = admit_legacy_replay_manifest(&manifest).expect("admitted replay");

    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let pool = WorkerPool::new(PoolConfig::default());
    let live = test_transcribe_options(None);
    let pair = crate::api::LanguageSpec::try_from("eng,spa").expect("a pair");
    let opts = TranscribeOptions {
        plan: crate::transcribe::types::AdmittedTranscribePlan::admit(
            ReplayAsrPlan::for_legacy_replay(2, &pair).unwrap(),
            false,
            UtsegFallbackPolicy::Refuse,
        )
        .unwrap(),
        diarize: false,
        speaker_backend: None,
        with_morphosyntax: live.with_morphosyntax,
        cache_policies: live.cache_policies,
        write_wor: live.write_wor,
        media_name: live.media_name,
        engine_extras: live.engine_extras,
    };
    run_transcribe_pipeline_with_legacy_replay(
        replay,
        TranscribeUtsegExecution::production(opts.plan.with_utseg()),
        PipelineServices::new(&pool, &cache),
        &opts,
        None,
        None,
    )
    .await
}

/// The whole transcribe pipeline runs on a worker stack a quarter of
/// production's. On 2026-10-06 a production tokio worker (2 MiB stack)
/// overflowed during transcription: stage futures were passed by value
/// through generic observers, so the pipeline's ~140 KB state machine was
/// copied onto the stack at five poll layers (measured 280-557 KB each).
/// This polls the real pipeline, not a pointer size; if a stage is ever
/// passed inline again, this thread overflows and the test fails.
#[test]
fn the_transcribe_pipeline_runs_within_a_quarter_of_a_worker_stack() {
    let mut lexical_control = bilingual_replay_tokens();
    lexical_control.retain(|token| token.text.as_str() != "25");
    let outcome = std::thread::Builder::new()
        .name("small-stack-transcribe".into())
        .stack_size(512 * 1024)
        .spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime")
                .block_on(replay(lexical_control))
        })
        .expect("thread")
        .join()
        .expect("the pipeline completed on a 512 KiB stack");
    assert!(
        outcome
            .expect("valid bilingual replay")
            .contains("@Languages:\teng, spa")
    );
}

/// A code-switched pair declares both of its languages: `@Languages` in
/// the pair's order, and one `lang=` provenance field naming the pair.
/// Live transcription withholds a pair (`LanguagePairSupport::Withheld`),
/// so the one route that still produces a pair transcript is replay of
/// evidence recorded under a pair, which is what this exercises: the
/// plan's pair is the transcript language whatever single code the
/// recorded response carries.
#[tokio::test]
async fn a_language_pair_is_declared_in_languages_and_provenance() {
    let tokens = bilingual_replay_tokens();
    let mut lexical_control = tokens.clone();
    lexical_control.retain(|token| token.text.as_str() != "25");
    let chat = replay(lexical_control)
        .await
        .expect("valid bilingual replay");
    assert!(
        chat.lines().any(|line| line == "@Languages:\teng, spa"),
        "{chat}"
    );
    let stamp = chat
        .lines()
        .find(|line| line.contains("fc-ba3 transcribe"))
        .expect("provenance stamp");
    assert!(stamp.contains("lang=eng,spa"), "{stamp}");

    // The unresolved bare numeral is kept verbatim for human review, so the
    // generated transcript does not pass admission. It is still written,
    // diagnosed with the finding, instead of vanishing behind a file error.
    let diagnosed = replay_output(tokens)
        .await
        .expect("a generated transcript with a refused token is written, not refused");
    let OutputReport::Diagnosed(report) =
        OutputReport::of(&diagnosed.document, &diagnosed.shortfalls)
    else {
        panic!("an unresolved bare numeral must diagnose the transcript");
    };
    assert!(
        report
            .findings()
            .any(|finding| finding.code.as_deref() == Some("E220")),
        "{report:?}"
    );
    let diagnosed = diagnosed.document;
    assert!(diagnosed.as_str().contains("25"), "{}", diagnosed.as_str());
    assert!(
        diagnosed.as_str().contains("@Languages:\teng, spa"),
        "the rest of the transcript is intact: {}",
        diagnosed.as_str()
    );
}

/// A projected-evidence replay enters the current word-level speaker
/// projection and CHAT builder without possessing a live provider
/// capability. This is the end-to-end guard for the research replay path,
/// not merely a manifest-parser test.
#[tokio::test]
async fn legacy_replay_runs_current_speaker_projection_without_live_inference() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let media = tempdir.path().join("sample.wav");
    let asr = tempdir.path().join("sample_asr_response.json");
    let turns = tempdir.path().join("sample.turns.json");
    let manifest = tempdir.path().join("sample.replay.json");
    std::fs::write(&media, b"fingerprinted media").expect("media");
    std::fs::write(
        &asr,
        serde_json::to_vec_pretty(&AsrResponse {
            tokens: vec![
                AsrToken {
                    text: "bonjour".into(),
                    start_s: at(0.0),
                    end_s: at(0.5),
                    speaker: Some("ASR_0".into()),
                    confidence: Some(0.9),
                },
                AsrToken {
                    text: "oui".into(),
                    start_s: at(0.5),
                    end_s: at(1.0),
                    speaker: Some("ASR_0".into()),
                    confidence: Some(0.8),
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
        .expect("ASR JSON"),
    )
    .expect("ASR artifact");
    std::fs::write(
        &turns,
        br#"{"source":"batchalign3:pyannote_ai:precision-2","turns":[{"track":"PAR0","start_ms":0,"end_ms":500},{"track":"PAR1","start_ms":500,"end_ms":1000}]}"#,
    )
    .expect("turns");
    write_legacy_replay_manifest(
        LegacyReplayManifestRequest {
            recording_id: "sample",
            media_path: &media,
            asr_response_path: &asr,
            speaker_turns_path: Some(&turns),
            producer: LegacyProjectedAsrProducer::RevAi,
        },
        &manifest,
    )
    .expect("manifest");
    let replay = admit_legacy_replay_manifest(&manifest).expect("admitted replay");

    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let pool = WorkerPool::new(PoolConfig::default());
    let opts = test_transcribe_options_in(
        Some(SpeakerBackendV2::PyannoteAi),
        LanguageCode3::fra().into(),
    );
    let opts = TranscribeOptions {
        plan: crate::transcribe::types::AdmittedTranscribePlan::admit(
            ReplayAsrPlan::for_legacy_replay(2, &LanguageCode3::fra().into()).unwrap(),
            false,
            UtsegFallbackPolicy::Refuse,
        )
        .unwrap(),
        diarize: opts.diarize,
        speaker_backend: opts.speaker_backend,
        with_morphosyntax: opts.with_morphosyntax,
        cache_policies: opts.cache_policies,
        write_wor: opts.write_wor,
        media_name: opts.media_name,
        engine_extras: opts.engine_extras,
    };
    let chat = run_transcribe_pipeline_with_legacy_replay(
        replay,
        TranscribeUtsegExecution::production(opts.plan.with_utseg()),
        PipelineServices::new(&pool, &cache),
        &opts,
        None,
        Some(tempdir.path()),
    )
    .await
    .expect("offline replay")
    .document
    .into_text();

    assert!(chat.contains("*PAR0:\tbonjour ."), "{chat}");
    assert!(chat.contains("*PAR1:\toui ."), "{chat}");
    assert!(chat.contains("asr=rev"), "{chat}");
    assert!(
        chat.contains("Speaker projection requires review"),
        "{chat}"
    );
    let report: serde_json::Value = serde_json::from_slice(
        &std::fs::read(only_debug_artifact(
            tempdir.path(),
            "_speaker_projection.json",
        ))
        .expect("projection artifact"),
    )
    .expect("projection wire");
    assert_eq!(report["summary"]["directly_supported"], 2);
    assert_eq!(report["summary"]["inferred"], 1);
    assert_eq!(
        report["evidence"]["assignments"][2]["basis"]["kind"],
        "previous_word"
    );
}
