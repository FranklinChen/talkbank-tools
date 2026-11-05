//! Actual artifact preparation and routed lease ownership with a model-free
//! worker. This does not qualify acoustic execution or model allocations.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::*;
use crate::cache::UtteranceCache;
use crate::chat_ops::fa::coordinates::{Ms, Recording};
use crate::chat_ops::fa::{
    AnchorIndex, AudioIdentity, EndOverlapPolicy, ExistingWorBoundaryPolicy, MainBulletPolicy,
    ReviewLevel, WhisperTextMode, WordGapHealing, group_utterances,
};
use crate::engine_reports::FaCacheNamespace;
use crate::pipeline::PipelineServices;
use crate::types::params::WorTierPolicy;
use crate::worker::pool::{GpuModelPlan, PoolConfig, WorkerPool};
use talkbank_parser::TreeSitterParser;
use talkbank_parser_tests::test_error::ParseProductTestExt;

fn params() -> FaParams {
    FaParams {
        gap_healing: WordGapHealing::Heal,
        whisper_text: WhisperTextMode::SpaceJoined,
        existing_wor_boundaries: ExistingWorBoundaryPolicy::Preserve,
        end_overlap_policy: EndOverlapPolicy::ClampAllAdjacent,
        main_bullets: MainBulletPolicy::DeriveFromWords,
        engine: FaEngineName::Wave2Vec,
        cache_policy: CachePolicy::SkipCache,
        wor_tier: WorTierPolicy::Include,
        bullet_repair: false,
        review_level: ReviewLevel::None,
    }
}

#[tokio::test]
async fn retained_preparation_survives_removal_of_the_original_media_until_dispatch() {
    let directory = tempfile::tempdir().unwrap();
    let audio_path = directory.path().join("sample.wav");
    crate::media::probe::fixtures::write_pcm_wav(&audio_path, 1);
    let chat = TreeSitterParser::new().unwrap().parse_chat_file(
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tSPK Speaker\n@ID:\teng|test|SPK|||||Speaker|||\n@Media:\tsample, audio\n*SPK:\thello world . \u{15}0_100\u{15}\n@End\n"
    ).expect_built();
    let identity = AudioIdentity::from_metadata("sample.wav", 0, 0);
    let audio = AudioContext {
        audio_path: &audio_path,
        audio_identity: &identity,
        audio_duration: None,
    };
    let anchors = AnchorIndex::not_observed();
    let params = params();
    let groups = group_utterances(
        &chat,
        params.aggregation_target(),
        &Recording::of_duration(Ms(100)).unwrap(),
        &anchors,
    )
    .groups;
    let pool = WorkerPool::new(PoolConfig {
        test_echo: true,
        python_path: concat!(env!("CARGO_MANIFEST_DIR"), "/../../.venv/bin/python").into(),
        ..PoolConfig::for_tests()
    });
    let cache = UtteranceCache::sqlite(Some(directory.path().join("cache")))
        .await
        .unwrap();
    let namespace = FaCacheNamespace::for_test("test-echo");
    let gpu = GpuModelPlan::alignment(FaEngineName::Wave2Vec, None);
    let services = FaServices {
        pipeline: PipelineServices::new(&pool, &cache),
        cache_namespace: &namespace,
        gpu: &gpu,
        attention_capture_dir: None,
    };
    let language = LanguageCode3::eng();
    let outcome = async {
        let population = prepare_population(groups, crate::chat_ops::fa::MainBulletAuthority::DeriveFromWords.supplied_retry_sources(&chat, &Recording::of_duration(Ms(100)).unwrap()).unwrap(), &chat, &HashSet::new(), &anchors,
            FaPreparationContext { audio: &audio, worker_lang: &language, services, params: &params }, None).await?;
        let plan = super::super::units::DispatchPlan::build(&population.groups, &identity, params.whisper_text, params.engine);
        let unit = plan.units().next().expect("one prepared request");
        let super::super::transport::FaInferencePlan::Authorized(authorization) =
            super::super::transport::plan_fa_inference(CachePolicy::SkipCache, vec![unit])?
        else { panic!("one missed request") };
        let mut dispatches = population.requests.bind(authorization)?;
        let prepared = dispatches.pop().unwrap().prepared;
        let prepared_root = prepared.inputs.artifacts.root().to_path_buf();
        assert!(prepared_root.is_dir(), "prepared inputs are still owned");
        assert!(prepared_root.join("audio").read_dir().unwrap().next().is_some());
        std::fs::remove_file(&audio_path).unwrap();
        let response = prepared.execute(services, WorkerLanguage::from(language.clone())).await?;
        assert!(prepared_root.exists(), "response admission still owns fallback inputs");
        assert!(matches!(&response, PreparedFaExecution::Direct { response, .. } if matches!(response.read(), ExecuteOutcomeRef::Success(TaskResultV2::IndexedWordTimingResult(_)))));
        drop(response);
        assert!(!prepared_root.exists(), "terminal admission releases its owned artifacts");
        Ok::<_, ServerError>(())
    }.await;
    pool.shutdown().await;
    outcome.expect("no media decode after preparation");
}

#[tokio::test]
async fn admitted_cached_geometry_requires_neither_a_decoder_nor_a_model_preparer() {
    let directory = tempfile::tempdir().unwrap();
    let audio_path = directory.path().join("already-resolved.wav");
    let chat = TreeSitterParser::new().unwrap().parse_chat_file(
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tSPK Speaker\n@ID:\teng|test|SPK|||||Speaker|||\n@Media:\talready-resolved, audio\n*SPK:\thello world . \u{15}0_100\u{15}\n@End\n"
    ).expect_built();
    let identity = AudioIdentity::from_metadata("already-resolved.wav", 0, 0);
    let audio = AudioContext {
        audio_path: &audio_path,
        audio_identity: &identity,
        audio_duration: None,
    };
    let anchors = AnchorIndex::not_observed();
    let mut params = params();
    params.cache_policy = CachePolicy::UseCache;
    let groups = group_utterances(
        &chat,
        params.aggregation_target(),
        &Recording::of_duration(Ms(100)).unwrap(),
        &anchors,
    )
    .groups;
    let pool = WorkerPool::new(PoolConfig {
        python_path: directory
            .path()
            .join("inference-is-forbidden")
            .to_string_lossy()
            .into_owned(),
        ..PoolConfig::for_tests()
    });
    let cache = UtteranceCache::sqlite(Some(directory.path().join("cache")))
        .await
        .unwrap();
    let namespace = FaCacheNamespace::for_test("declared-cached-wave-v1");
    let gpu = GpuModelPlan::alignment(FaEngineName::Wave2Vec, None);
    let services = FaServices {
        pipeline: PipelineServices::new(&pool, &cache),
        cache_namespace: &namespace,
        gpu: &gpu,
        attention_capture_dir: None,
    };
    let language = LanguageCode3::eng();
    let key = {
        let plan = super::super::units::DispatchPlan::build(
            &groups,
            &identity,
            params.whisper_text,
            params.engine,
        );
        plan.units().next().unwrap().cache_key().clone()
    };
    let response = ExecuteResponseV2::success(
        "declared-cache".into(),
        TaskResultV2::IndexedWordTimingResult(crate::types::worker_v2::IndexedWordTimingResultV2 {
            indexed_timings: vec![None, None],
        }),
        crate::api::NonNegativeSeconds::try_from(0.01).unwrap(),
    );
    let raw = super::super::raw_evidence::FaRawEvidence::admit_requested(
        &response,
        params.engine,
        &namespace,
        super::super::raw_evidence::ExpectedFaWords::new(2),
        &key,
        super::super::raw_evidence::FaEvidenceRoute::Direct,
    )
    .unwrap()
    .into_replayable()
    .unwrap();
    let timings = vec![
        crate::chat_ops::fa::WordTiming::fixture(10, 30),
        crate::chat_ops::fa::WordTiming::fixture(40, 60),
    ];
    let value = super::super::AdmittedCachedFaTimings::encode_from_raw(timings, &raw).unwrap();
    cache
        .put_batch(
            &[(key.as_str().to_owned(), value)],
            crate::cache::tasks::FORCED_ALIGNMENT,
            &namespace,
        )
        .await
        .unwrap();
    let context = FaPreparationContext {
        audio: &audio,
        worker_lang: &language,
        services,
        params: &params,
    };
    let mut probe = LiveResourceProbe {
        context,
        registry: PreparationRegistry::new(),
        cached: super::super::units::collect_candidate_cache(&[], context).await,
    };
    let GroupSpan::Single(single) = groups[0].span() else {
        panic!("source candidate")
    };
    let reference = match probe.prepare_window(single.request()).await.unwrap() {
        ResourceProbeOutcome::Prepared(reference) => reference,
        ResourceProbeOutcome::Limited(_) => {
            panic!("unexamined child geometry has an admitted cache answer")
        }
    };
    let mut selected = SelectedPreparations::new();
    selected.select(reference);
    let frozen = probe.registry.freeze(selected).unwrap();
    assert!(
        matches!(
            frozen.into_values().next(),
            Some(PreparedFaInput::Cached(_))
        ),
        "a child/context cache hit grants a cached answer, never a model-execution owner"
    );
    let outcome = prepare_population(
        groups,
        crate::chat_ops::fa::MainBulletAuthority::DeriveFromWords
            .supplied_retry_sources(&chat, &Recording::of_duration(Ms(100)).unwrap())
            .unwrap(),
        &chat,
        &HashSet::new(),
        &anchors,
        FaPreparationContext {
            audio: &audio,
            worker_lang: &language,
            services,
            params: &params,
        },
        None,
    )
    .await;
    pool.shutdown().await;
    assert!(
        outcome.is_ok(),
        "exact admitted cache evidence cannot require unavailable media or a model: {}",
        outcome.err().unwrap()
    );
}

#[tokio::test]
async fn an_empty_decode_keeps_its_measurement_without_loading_a_model_or_failing_the_file() {
    let directory = tempfile::tempdir().unwrap();
    let audio_path = directory.path().join("sample.wav");
    crate::media::probe::fixtures::write_pcm_wav(&audio_path, 1);
    // Declared container extent and actual decoder output deliberately disagree.
    // The decoder fact is separate from recording-window admission.
    let recording = Recording::of_duration(Ms(2_000)).unwrap();
    let window = FaWindow::within(
        &recording,
        crate::chat_ops::fa::coordinates::FileMs::new(1_000),
        crate::chat_ops::fa::coordinates::FileMs::new(2_000),
    )
    .unwrap();
    let identity = AudioIdentity::from_metadata("sample.wav", 0, 0);
    let audio = AudioContext {
        audio_path: &audio_path,
        audio_identity: &identity,
        audio_duration: None,
    };
    let params = params();
    let pool = WorkerPool::new(PoolConfig {
        python_path: directory
            .path()
            .join("inference-is-forbidden")
            .to_string_lossy()
            .into_owned(),
        ..PoolConfig::for_tests()
    });
    let cache = UtteranceCache::sqlite(Some(directory.path().join("cache")))
        .await
        .unwrap();
    let namespace = FaCacheNamespace::for_test("declared-wave-v1");
    let gpu = GpuModelPlan::alignment(FaEngineName::Wave2Vec, None);
    let services = FaServices {
        pipeline: PipelineServices::new(&pool, &cache),
        cache_namespace: &namespace,
        gpu: &gpu,
        attention_capture_dir: None,
    };
    let language = LanguageCode3::eng();
    let mut probe = LiveResourceProbe {
        context: FaPreparationContext {
            audio: &audio,
            worker_lang: &language,
            services,
            params: &params,
        },
        registry: PreparationRegistry::new(),
        cached: super::super::units::collect_candidate_cache(
            &[],
            FaPreparationContext {
                audio: &audio,
                worker_lang: &language,
                services,
                params: &params,
            },
        )
        .await,
    };
    let words = vec![FaWord {
        utterance_index: talkbank_model::UtteranceIdx::new(0),
        utterance_word_index: talkbank_model::WordIdx::new(0),
        text: "hello".into(),
    }];
    let group = FaGroup::test_fixture(
        crate::chat_ops::fa::TimeSpan::new(window.audio_start().get(), window.end().get()),
        words,
        vec![talkbank_model::UtteranceIdx::new(0)],
    );
    let GroupSpan::Single(single) = group.span() else {
        panic!("declared single input")
    };
    let outcome = probe.prepare_window(single.request()).await;
    pool.shutdown().await;
    assert!(
        matches!(
            outcome,
            Ok(ResourceProbeOutcome::Limited(
                FaResourceConstraintV2::EmptyDecodedAudio {
                    shortfall: crate::types::worker_v2::DecodedAudioShortfallV2::NoBytes
                }
            ))
        ),
        "empty input remains an explicit unavailable population"
    );
}

#[tokio::test]
async fn shared_contexts_reuse_native_preparation_and_execute_one_owner_for_two_answer_ranges() {
    let directory = tempfile::tempdir().unwrap();
    let audio_path = directory.path().join("shared.wav");
    crate::media::probe::fixtures::write_pcm_wav(&audio_path, 1);
    let identity = AudioIdentity::from_metadata("shared.wav", 0, 0);
    let audio = AudioContext {
        audio_path: &audio_path,
        audio_identity: &identity,
        audio_duration: None,
    };
    let params = params();
    let pool = WorkerPool::new(PoolConfig {
        test_echo: true,
        python_path: concat!(env!("CARGO_MANIFEST_DIR"), "/../../.venv/bin/python").into(),
        ..PoolConfig::for_tests()
    });
    let cache = UtteranceCache::sqlite(Some(directory.path().join("cache")))
        .await
        .unwrap();
    let namespace = FaCacheNamespace::for_test("test-echo");
    let gpu = GpuModelPlan::alignment(FaEngineName::Wave2Vec, None);
    let services = FaServices {
        pipeline: PipelineServices::new(&pool, &cache),
        cache_namespace: &namespace,
        gpu: &gpu,
        attention_capture_dir: None,
    };
    let language = LanguageCode3::eng();
    let group = FaGroup::test_fixture(
        crate::chat_ops::fa::TimeSpan::new(0, 100),
        ["hello", "world"]
            .into_iter()
            .enumerate()
            .map(|(index, text)| FaWord {
                utterance_index: talkbank_model::UtteranceIdx::new(0),
                utterance_word_index: talkbank_model::WordIdx::new(index),
                text: text.into(),
            })
            .collect(),
        vec![talkbank_model::UtteranceIdx::new(0)],
    );
    let outcome = async {
        let context = FaPreparationContext {
            audio: &audio,
            worker_lang: &language,
            services,
            params: &params,
        };
        let mut probe = LiveResourceProbe {
            context,
            registry: PreparationRegistry::new(),
            cached: super::super::units::collect_candidate_cache(
                &[],
                FaPreparationContext {
                    audio: &audio,
                    worker_lang: &language,
                    services,
                    params: &params,
                },
            )
            .await,
        };
        let GroupSpan::Single(single) = group.span() else {
            panic!("one source request")
        };
        let ResourceProbeOutcome::Prepared(first) = probe.prepare_window(single.request()).await?
        else {
            panic!("model-free preparation")
        };
        // Any second materialization now fails. Context expansion must reuse
        // the settled report before even constructing the decoder future.
        std::fs::remove_file(&audio_path).unwrap();
        let ResourceProbeOutcome::Prepared(repeated) =
            probe.prepare_window(single.request()).await?
        else {
            panic!("retained preparation")
        };
        assert_eq!(first, repeated);
        let split = resource_partition::test_support::shared_context_partition(
            single.request(),
            &mut probe,
        )
        .await
        .map_err(plan_error)?;
        let mut selected = SelectedPreparations::new();
        let split = split.finish(|_, reference| selected.select(reference));
        let entries = probe
            .registry
            .freeze(selected)
            .unwrap()
            .into_iter()
            .filter_map(|(scope, input)| match input {
                PreparedFaInput::Cached(_) => None,
                PreparedFaInput::Inference(prepared) => Some((scope, *prepared)),
            })
            .collect();
        let requests = PreparedFaRequests { entries };
        assert_eq!(requests.entries.len(), 1, "one physical input owner");
        let groups = vec![FaGroup::from_resource_split(split)];
        let plan = super::super::units::DispatchPlan::build(
            &groups,
            &identity,
            params.whisper_text,
            params.engine,
        );
        let units: Vec<_> = plan.units().collect();
        assert_eq!(units.len(), 2);
        assert_eq!(units[0].words(), units[1].words());
        let super::super::transport::FaInferencePlan::Authorized(authorization) =
            super::super::transport::plan_fa_inference(CachePolicy::SkipCache, units.clone())?
        else {
            panic!("two missed projections")
        };
        let mut dispatches = requests.bind(authorization)?;
        assert_eq!(
            dispatches.len(),
            1,
            "only one consuming execution capability"
        );
        let dispatch = dispatches.pop().unwrap();
        assert_eq!(dispatch.aliases.len(), 1);
        assert_eq!(dispatch.aliases[0].ordinal(), units[1].ordinal());
        let root = dispatch.prepared.inputs.artifacts.root().to_path_buf();
        let execution = dispatch
            .prepared
            .execute(services, WorkerLanguage::from(language.clone()))
            .await?;
        let PreparedFaExecution::Direct { response, fallback } = execution else {
            panic!("model-free direct path")
        };
        assert!(matches!(
            response.read(),
            ExecuteOutcomeRef::Success(TaskResultV2::IndexedWordTimingResult(_))
        ));
        assert!(root.is_dir(), "response owns its original prepared inputs");
        // Exercise the fallback transition without re-opening the source. This
        // model-free control qualifies ownership/transport, not acoustic output.
        let fallback = fallback
            .execute(
                services,
                WorkerLanguage::from(language.clone()),
                FaEngineName::Whisper,
                units[0].cache_key().as_str(),
            )
            .await?;
        assert!(matches!(fallback, FaFallbackExecution::Response(_)));
        assert!(!root.exists(), "fallback consumes the final artifact owner");
        Ok::<_, ServerError>(())
    }
    .await;
    pool.shutdown().await;
    outcome.unwrap();
}
