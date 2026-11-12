#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::*;
use crate::api::LanguageCode3;
use crate::cache::UtteranceCache;
use crate::chat_ops::fa::coordinates::{Ms, Recording};
use crate::chat_ops::fa::{
    AudioIdentity, EndOverlapPolicy, ExistingWorBoundaryPolicy, MainBulletAuthority,
    MainBulletPolicy, ReviewLevel, WhisperTextMode, WordGapHealing,
};
use crate::params::{AudioContext, CachePolicy, FaParams};
use crate::pipeline::PipelineServices;
use crate::types::engines::FaEngineName;
use crate::types::params::WorTierPolicy;
use crate::worker::pool::{GpuModelPlan, PoolConfig, WorkerPool};
use talkbank_parser::TreeSitterParser;
use talkbank_parser_tests::test_error::ParseProductTestExt;

#[tokio::test]
async fn cached_primary_retries_only_projected_gaps_and_retains_evidence() {
    use crate::api::NonNegativeSeconds;
    use crate::cache::tasks::FORCED_ALIGNMENT;
    use crate::chat_ops::fa::{FaGroup, FaProjection, FaProjectionPolicy, TimeSpan, WordEndPolicy};
    use crate::fa::AdmittedCachedFaTimings;
    use crate::fa::raw_evidence::{ExpectedFaWords, FaEvidenceRoute, FaRawEvidence};
    use crate::fa::units::{DispatchPlan, FaDispatchInputs, resolve_group_timings};
    use crate::types::worker_v2::{
        ExecuteResponseV2, IndexedWordTimingResultV2, TaskResultV2, WorkerRequestIdV2,
    };

    #[derive(Clone, Copy)]
    enum Control {
        MissingGap,
        CompleteWithHealing,
    }
    for control in [Control::MissingGap, Control::CompleteWithHealing] {
        let mut chat = TreeSitterParser::new().unwrap().parse_chat_file(
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tSPK Speaker\n@ID:\teng|test|SPK|||||Speaker|||\n@Media:\tsample, audio\n*SPK:\thello there . \u{15}100_200\u{15}\n@End\n"
    ).expect_built();
        let authority = MainBulletAuthority::bind(MainBulletPolicy::KeepGiven, &chat).unwrap();
        let sources = authority
            .supplied_retry_sources(&chat, &Recording::of_duration(Ms(400)).unwrap())
            .unwrap();
        let group = FaGroup::test_fixture(
            TimeSpan::new(0, 400),
            sources.sources()[0].words().to_vec(),
            vec![talkbank_model::UtteranceIdx::new(0)],
        );
        let groups = vec![group];
        let directory = tempfile::tempdir().unwrap();
        let audio_path = directory.path().join("absent.wav");
        let identity = AudioIdentity::from_metadata("absent.wav", 0, 0);
        let audio = AudioContext {
            audio_path: &audio_path,
            audio_identity: &identity,
            audio_duration: None,
        };
        let pool = WorkerPool::new(PoolConfig::for_tests());
        let cache = UtteranceCache::sqlite(Some(directory.path().join("cache")))
            .await
            .unwrap();
        let namespace = crate::engine_reports::FaCacheNamespace::for_test("test-fa");
        let gpu = GpuModelPlan::alignment(FaEngineName::Wave2Vec, None);
        let services = super::super::FaServices {
            pipeline: PipelineServices::new(&pool, &cache),
            cache_namespace: &namespace,
            gpu: &gpu,
            attention_capture_dir: None,
        };
        let params = FaParams {
            gap_healing: match control {
                Control::MissingGap => WordGapHealing::PreserveMeasured,
                Control::CompleteWithHealing => WordGapHealing::Heal,
            },
            whisper_text: WhisperTextMode::SpaceJoined,
            existing_wor_boundaries: ExistingWorBoundaryPolicy::Preserve,
            end_overlap_policy: EndOverlapPolicy::ClampAllAdjacent,
            main_bullets: MainBulletPolicy::KeepGiven,
            engine: FaEngineName::Wave2Vec,
            cache_policy: match control {
                Control::MissingGap => CachePolicy::RequireCache,
                Control::CompleteWithHealing => CachePolicy::UseCache,
            },
            wor_tier: WorTierPolicy::Include,
            bullet_repair: false,
            review_level: ReviewLevel::None,
        };
        let language = LanguageCode3::eng();
        let context = FaPreparationContext {
            audio: &audio,
            worker_lang: &language,
            services,
            params: &params,
        };
        let plan = DispatchPlan::build(&groups, &identity, params.whisper_text, params.engine);
        let primary = plan.units().next().unwrap();
        let retry = DispatchUnit::supplied(&sources.sources()[0], context);
        let entries = match control {
            Control::MissingGap => vec![
                (
                    primary,
                    vec![
                        crate::chat_ops::fa::WordTiming::fixture(110, 150),
                        crate::chat_ops::fa::WordTiming::fixture(258, 318),
                    ],
                ),
                (
                    &retry,
                    vec![
                        crate::chat_ops::fa::WordTiming::fixture(140, 160),
                        crate::chat_ops::fa::WordTiming::fixture(170, 190),
                    ],
                ),
            ],
            Control::CompleteWithHealing => vec![(
                primary,
                vec![
                    crate::chat_ops::fa::WordTiming::fixture(110, 150),
                    crate::chat_ops::fa::WordTiming::fixture(170, 190),
                ],
            )],
        };
        for (unit, timings) in entries {
            let response = ExecuteResponseV2::success(
                WorkerRequestIdV2::from("retry-fixture"),
                TaskResultV2::IndexedWordTimingResult(IndexedWordTimingResultV2 {
                    indexed_timings: vec![None; timings.len()],
                }),
                NonNegativeSeconds::try_from(0.01).unwrap(),
            );
            let raw = FaRawEvidence::admit_requested(
                &response,
                FaEngineName::Wave2Vec,
                &namespace,
                ExpectedFaWords::new(timings.len()),
                unit.cache_key(),
                FaEvidenceRoute::Direct,
            )
            .unwrap()
            .into_replayable()
            .unwrap();
            let value = AdmittedCachedFaTimings::encode_from_raw(timings, &raw).unwrap();
            cache
                .put_batch(
                    &[(unit.cache_key().as_str().to_owned(), value)],
                    FORCED_ALIGNMENT,
                    &namespace,
                )
                .await
                .unwrap();
        }
        drop(plan);
        let reusable = Default::default();
        let anchors = crate::chat_ops::fa::AnchorIndex::not_observed();
        let population = super::super::preparation::prepare_population(
            groups, sources, &chat, &reusable, &anchors, context, None,
        )
        .await
        .unwrap();
        let resolved = resolve_group_timings(FaDispatchInputs {
            population,
            chat_file: &chat,
            reusable_utterances: &reusable,
            recovery: &[],
            progress: None,
            context: "retry test",
        })
        .await
        .unwrap();
        let injected = resolved
            .inject(
                &mut chat,
                FaProjection::new(
                    FaProjectionPolicy::new(
                        WordEndPolicy::measured(params.gap_healing),
                        ExistingWorBoundaryPolicy::Preserve,
                        EndOverlapPolicy::ClampAllAdjacent,
                    ),
                    authority,
                ),
                true,
            )
            .unwrap();
        let (_, _, evidence) = injected
            .retry
            .finalize(
                injected.applied,
                &mut chat,
                BulletRepairPolicy::Disabled,
                None,
            )
            .await
            .unwrap();
        let trace = serde_json::to_value(&evidence).unwrap();
        let output = batchalign_transform::serialize::to_chat_string(&chat);
        match control {
            Control::MissingGap => {
                assert_eq!(trace[0]["outcome"]["kind"], "timings");
                assert_eq!(trace[0]["outcome"]["timings"][0]["start_ms"], 140);
                assert_eq!(trace[0]["outcome"]["timings"][1]["start_ms"], 170);
                assert!(output.contains("\u{15}110_150\u{15}"));
                assert!(output.contains("\u{15}170_190\u{15}"));
            }
            Control::CompleteWithHealing => {
                assert!(
                    evidence.is_empty(),
                    "complete projected output grants no retry: {trace}"
                );
                assert!(output.contains("%wor:"));
            }
        }
        let mut result = crate::types::results::FaResult::without_groups(
            chat,
            params.gap_healing,
            params.engine.as_wire_name(),
            &namespace,
            crate::types::results::RecordingRead::Read,
        );
        result.supplied_retries = Some(evidence);
        let timeline = result.into_timeline_trace();
        let mut wire = serde_json::to_value(timeline).unwrap();
        assert_eq!(wire["evidence_schema_version"], 13);
        let recovered: crate::types::traces::FaTimelineTrace =
            serde_json::from_value(wire.clone()).unwrap();
        assert!(recovered.supplied_retries.is_some());
        wire.as_object_mut().unwrap().remove("supplied_retries");
        let legacy: crate::types::traces::FaTimelineTrace = serde_json::from_value(wire).unwrap();
        assert!(legacy.supplied_retries.is_none());
        pool.shutdown().await;
    }
}

#[test]
fn complete_compound_filler_survives_projection_without_retry() {
    use crate::chat_ops::fa::{
        FaGroup, FaProjection, FaProjectionPolicy, TimeSpan, WordEndPolicy,
        apply_fa_results_with_projection_policy,
    };

    let mut chat = TreeSitterParser::new().unwrap().parse_chat_file(
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tSPK Speaker\n@ID:\teng|test|SPK|||||Speaker|||\n@Media:\tsample, audio\n*SPK:\t&-you_know okay . \u{15}100_300\u{15}\n@End\n"
    ).expect_built();
    let authority = MainBulletAuthority::bind(MainBulletPolicy::KeepGiven, &chat).unwrap();
    let sources = authority
        .supplied_retry_sources(&chat, &Recording::of_duration(Ms(400)).unwrap())
        .unwrap();
    let source = &sources.sources()[0];
    let timings = vec![
        crate::chat_ops::fa::WordTiming::fixture(110, 130),
        crate::chat_ops::fa::WordTiming::fixture(140, 160),
        crate::chat_ops::fa::WordTiming::fixture(170, 190),
    ];
    let group = FaGroup::test_fixture(
        TimeSpan::new(0, 400),
        source.words().to_vec(),
        vec![source.index()],
    );
    let applied = apply_fa_results_with_projection_policy(
        &mut chat,
        &[group],
        &[timings],
        FaProjection::new(
            FaProjectionPolicy::new(
                WordEndPolicy::measured(WordGapHealing::PreserveMeasured),
                ExistingWorBoundaryPolicy::Preserve,
                EndOverlapPolicy::ClampAllAdjacent,
            ),
            authority,
        ),
        true,
    )
    .unwrap();
    let imposed = applied.then_impose(&mut chat).unwrap();
    let SuppliedRetryStep::Complete(completed) = SuppliedRetrySession::new(
        imposed,
        &mut chat,
        sources
            .into_sources()
            .into_iter()
            .map(|source| (source, ())),
    )
    .unwrap()
    .next() else {
        panic!("complete compound projection grants no retry")
    };
    let finalized = completed
        .then_finalize(BulletRepairPolicy::Disabled)
        .unwrap();
    let _ = finalized.monotonicity();
    let output = batchalign_transform::serialize::to_chat_string(&chat);
    assert!(output.contains("\u{15}110_160\u{15}"));
    assert!(output.contains("\u{15}170_190\u{15}"));
}

#[tokio::test]
async fn optional_required_cache_miss_cannot_prepare_decode_or_infer() {
    let directory = tempfile::tempdir().unwrap();
    // Deliberately absent: any decoder attempt would fail this boundary.
    let path = directory.path().join("absent.wav");
    let identity = AudioIdentity::from_metadata("absent.wav", 0, 0);
    let audio = AudioContext {
        audio_path: &path,
        audio_identity: &identity,
        audio_duration: None,
    };
    let chat = TreeSitterParser::new().unwrap().parse_chat_file(
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tSPK Speaker\n@ID:\teng|test|SPK|||||Speaker|||\n@Media:\tabsent, audio\n*SPK:\thello . \u{15}100_200\u{15}\n@End\n"
    ).expect_built();
    let sources = MainBulletAuthority::bind(MainBulletPolicy::KeepGiven, &chat)
        .unwrap()
        .supplied_retry_sources(&chat, &Recording::of_duration(Ms(400)).unwrap())
        .unwrap();
    let pool = WorkerPool::new(PoolConfig {
        test_echo: true,
        ..PoolConfig::for_tests()
    });
    let cache = UtteranceCache::sqlite(Some(directory.path().join("cache")))
        .await
        .unwrap();
    let namespace = crate::engine_reports::FaCacheNamespace::for_test("test-echo");
    let gpu = GpuModelPlan::alignment(FaEngineName::Wave2Vec, None);
    let services = super::super::FaServices {
        pipeline: PipelineServices::new(&pool, &cache),
        cache_namespace: &namespace,
        gpu: &gpu,
        attention_capture_dir: None,
    };
    let language = LanguageCode3::eng();
    let params = FaParams {
        gap_healing: WordGapHealing::PreserveMeasured,
        whisper_text: WhisperTextMode::SpaceJoined,
        existing_wor_boundaries: ExistingWorBoundaryPolicy::Preserve,
        end_overlap_policy: EndOverlapPolicy::ClampAllAdjacent,
        main_bullets: MainBulletPolicy::KeepGiven,
        engine: FaEngineName::Wave2Vec,
        cache_policy: CachePolicy::RequireCache,
        wor_tier: WorTierPolicy::Include,
        bullet_repair: false,
        review_level: ReviewLevel::None,
    };
    let context = FaPreparationContext {
        audio: &audio,
        worker_lang: &language,
        services,
        params: &params,
    };
    let population = super::super::preparation::prepare_population(
        Vec::new(),
        sources,
        &chat,
        &Default::default(),
        &crate::chat_ops::fa::AnchorIndex::not_observed(),
        context,
        None,
    )
    .await
    .unwrap();
    let mut cached = population.cached;
    let (source, claim) = population.retry.into_scoped().next().unwrap();
    let mut fallbacks = Vec::new();
    let result =
        infer_missing_input(&source, claim, context, &mut cached, &mut fallbacks, None).await;
    pool.shutdown().await;
    assert!(
        matches!(result.unwrap(), SuppliedRetryAnswer::Unresolved { reason } if reason.contains("required-cache miss") && reason.contains("inference forbidden"))
    );
    assert!(fallbacks.is_empty());
}
