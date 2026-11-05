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
    let result = resolve_attempt(&source, claim, context, &mut cached, &mut fallbacks, None).await;
    pool.shutdown().await;
    assert!(
        matches!(result.unwrap(), SuppliedRetryAnswer::Unresolved { reason } if reason.contains("required-cache miss") && reason.contains("inference forbidden"))
    );
    assert!(fallbacks.is_empty());
}
