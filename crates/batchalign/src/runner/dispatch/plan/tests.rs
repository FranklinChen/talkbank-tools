use std::collections::BTreeMap;

use serde_json::json;
use tokio_util::sync::CancellationToken;

use super::*;
use crate::api::{JobId, LanguageCode3, NumSpeakers, ReleasedCommand};
use crate::config::ServerConfig;
use crate::options::{
    AlignOptions, AsrEngineName, BenchmarkOptions, CommandOptions, CommonOptions, DiarizeOptions,
    MorphotagOptions, OpensmileOptions, SpeakerEngineName, TranscribeOptions as TranscribeCommand,
};
use crate::store::{
    RunnerDispatchConfig, RunnerFilesystemConfig, RunnerJobIdentity, RunnerJobSnapshot,
};
use crate::transcribe::AsrWorkerMode;

#[test]
fn auto_speakers_http_contract_reaches_provider_and_speaker_boundary() {
    let submission: crate::api::JobSubmission = serde_json::from_value(json!({
        "command": "transcribe", "lang": "eng", "num_speakers": 2,
        "options": {"command": "transcribe", "asr_engine": "rev", "diarize": true,
            "wor": true, "auto_speakers": true, "engine_overrides": {"speaker": "pyannote_ai"},
            "override_media_cache": true, "debug_dir": "/data/evidence"},
        "paths_mode": true, "source_paths": ["/data/input.wav"],
        "output_paths": ["/data/output.cha"], "display_names": ["input.wav"]
    }))
    .expect("HTTP submission contract");
    submission
        .validate()
        .expect("supported automatic speaker request");
    let mut job = make_snapshot(
        ReleasedCommand::Transcribe,
        submission.options,
        BTreeMap::new(),
    );
    job.dispatch.num_speakers = NumSpeakers(2);
    let plan =
        TranscribeDispatchPlan::from_job(&job, &ServerConfig::default()).expect("dispatch plan");
    assert_eq!(
        plan.base_options.plan.asr().backend(),
        AsrBackend::RustRevAi
    );
    assert_eq!(plan.base_options.expected_speakers(), None);
    // A persisted job bypasses HTTP validation, but cannot bypass plan admission.
    let mut invalid = job.clone();
    invalid.dispatch.options = serde_json::from_value(serde_json::json!({
        "command": "transcribe", "auto_speakers": true,
        "engine_overrides": {"asr": "whisper"}
    }))
    .unwrap();
    assert!(matches!(
        TranscribeDispatchPlan::from_job(&invalid, &ServerConfig::default()),
        Err(DispatchPlanRefusal::AsrPolicy(
            crate::transcribe::TranscribeAsrPlanError::UnsupportedAutomaticCount
        ))
    ));
    assert!(plan.base_options.speaker_backend.is_some());
    assert!(plan.base_options.write_wor);
    assert_eq!(
        plan.base_options.cache_policies.rev_asr,
        crate::params::CachePolicy::SkipCache
    );
    assert_eq!(
        plan.base_options.cache_policies.speaker,
        crate::params::CachePolicy::SkipCache
    );
}

fn make_snapshot(
    command: ReleasedCommand,
    options: CommandOptions,
    runtime_state: BTreeMap<String, serde_json::Value>,
) -> RunnerJobSnapshot {
    RunnerJobSnapshot {
        run_generation: crate::store::RunGeneration::FIRST,
        identity: RunnerJobIdentity {
            job_id: JobId::from("job-plan"),
            correlation_id: "test-correlation".into(),
        },
        dispatch: RunnerDispatchConfig {
            command,
            lang: crate::api::LanguageSpec::Resolved(LanguageCode3::eng()),
            num_speakers: NumSpeakers(3),
            options,
            runtime_state,
            debug_traces: false,
        },
        filesystem: RunnerFilesystemConfig {
            paths_mode: false,
            source_paths: Vec::new(),
            output_paths: Vec::new(),
            before_paths: Vec::new(),
            staging_dir: Default::default(),
            media_mapping: Default::default(),
            media_subdir: Default::default(),
            source_dir: Default::default(),
        },
        cancel_token: CancellationToken::new(),
        pending_files: Vec::new(),
    }
}

#[test]
fn batched_plan_uses_morphotag_translation() {
    let mut common = CommonOptions {
        override_media_cache: true,
        ..Default::default()
    };
    common
        .mwt
        .insert("gonna".into(), vec!["going".into(), "to".into()]);
    let snapshot = make_snapshot(
        ReleasedCommand::Morphotag,
        CommandOptions::Morphotag(MorphotagOptions {
            common,
            retokenize: true,
            skipmultilang: true,
            merge_abbrev: true.into(),

            ..Default::default()
        }),
        BTreeMap::new(),
    );

    let plan = BatchedInferDispatchPlan::from_job(&snapshot);

    assert_eq!(plan.tokenization_mode, TokenizationMode::StanzaRetokenize);
    assert_eq!(plan.multilingual_policy, MultilingualPolicy::SkipNonPrimary);
    assert!(plan.should_merge_abbrev);
    assert_eq!(
        plan.mwt.get("gonna"),
        Some(&vec!["going".to_string(), "to".to_string()])
    );
}

#[test]
fn transcribe_plan_reads_runtime_flags_and_speaker_override() {
    let common = CommonOptions {
        override_media_cache: true,
        ..Default::default()
    };
    let mut runtime_state = BTreeMap::new();
    runtime_state.insert("utseg".into(), json!(false));
    runtime_state.insert("morphosyntax".into(), json!(true));
    let snapshot = make_snapshot(
        ReleasedCommand::Transcribe,
        CommandOptions::Transcribe(TranscribeCommand {
            auto_speakers: false,
            common,
            asr_engine: AsrEngineName::HkAliyun,
            diarize: true,
            wor: false.into(),
            merge_abbrev: true.into(),
            batch_size: 32,
            utseg_fallback: false.into(),
        }),
        runtime_state,
    );

    let plan = TranscribeDispatchPlan::from_job(&snapshot, &ServerConfig::default())
        .expect("transcribe plan");

    assert!(matches!(
        plan.base_options.plan.asr().backend(),
        AsrBackend::Worker(AsrWorkerMode::HkAliyunV2)
    ));
    assert!(plan.base_options.diarize);
    assert_eq!(
        plan.base_options.speaker_backend,
        Some(SpeakerBackendV2::PyannoteAi)
    );
    assert_eq!(
        plan.base_options.language(),
        crate::api::AsrLanguageRequest::One(LanguageCode3::eng())
    );
    assert_eq!(plan.base_options.expected_speakers(), Some(NumSpeakers(3)));
    assert!(!plan.base_options.plan.with_utseg());
    assert!(plan.base_options.with_morphosyntax);
    assert_eq!(
        plan.base_options.cache_policies,
        TranscribeCachePolicies::uniform(crate::params::CachePolicy::SkipCache)
    );
    assert!(plan.should_merge_abbrev);
}

#[test]
fn transcribe_s_plan_defaults_to_pyannote_ai_precision_2() {
    let snapshot = make_snapshot(
        ReleasedCommand::TranscribeS,
        CommandOptions::TranscribeS(TranscribeCommand {
            auto_speakers: false,
            common: CommonOptions::default(),
            asr_engine: AsrEngineName::RevAi,
            diarize: true,
            wor: false.into(),
            merge_abbrev: false.into(),
            batch_size: 8,
            utseg_fallback: false.into(),
        }),
        BTreeMap::new(),
    );

    let plan = TranscribeDispatchPlan::from_job(&snapshot, &ServerConfig::default())
        .expect("transcribe_s plan");

    assert!(matches!(
        plan.base_options.plan.asr().backend(),
        AsrBackend::RustRevAi
    ));
    assert!(plan.base_options.diarize);
    assert_eq!(
        plan.base_options.speaker_backend,
        Some(SpeakerBackendV2::PyannoteAi)
    );
    assert_eq!(
        plan.base_options.language(),
        crate::api::AsrLanguageRequest::One(LanguageCode3::eng())
    );
    assert_eq!(plan.base_options.expected_speakers(), Some(NumSpeakers(3)));
    assert!(plan.base_options.plan.with_utseg());
    assert!(!plan.base_options.with_morphosyntax);
    assert_eq!(
        plan.base_options.cache_policies,
        TranscribeCachePolicies::uniform(crate::params::CachePolicy::UseCache)
    );
    assert!(!plan.should_merge_abbrev);
}

/// RED FIRST (review item 7): a job naming an engine this build cannot
/// run must PROPAGATE the refusal. `.ok()?` used to turn it into `None`,
/// which routing answered with a `warn!` and a bare return: the job was
/// dropped with no file failed and no error recorded.
#[test]
fn transcribe_plan_propagates_an_unimplemented_engine_refusal() {
    let snapshot = make_snapshot(
        ReleasedCommand::Transcribe,
        CommandOptions::Transcribe(TranscribeCommand {
            auto_speakers: false,
            common: CommonOptions::default(),
            asr_engine: AsrEngineName::WhisperX,
            diarize: false,
            wor: false.into(),
            merge_abbrev: false.into(),
            batch_size: 8,
            utseg_fallback: false.into(),
        }),
        BTreeMap::new(),
    );

    let refusal = TranscribeDispatchPlan::from_job(&snapshot, &ServerConfig::default())
        .err()
        .expect("an unimplemented engine must refuse the plan, not drop it");
    assert!(
        matches!(refusal, DispatchPlanRefusal::Engine(_)),
        "the refusal must say it was the ENGINE, not unreadable options: {refusal:?}"
    );
    assert!(
        refusal.to_string().contains("whisperx"),
        "the refusal must name the engine, got: {refusal}"
    );
}

/// RED FIRST (review item 3): an align job whose persisted options are not
/// align options used to become `None`, which routing answered with a
/// `warn!` and a bare `return`: no file failed and the job simply stopped.
#[test]
fn fa_plan_refuses_options_it_cannot_read_instead_of_dropping_the_job() {
    let snapshot = make_snapshot(
        ReleasedCommand::Align,
        CommandOptions::Transcribe(TranscribeCommand {
            auto_speakers: false,
            common: CommonOptions::default(),
            asr_engine: AsrEngineName::RevAi,
            diarize: false,
            wor: false.into(),
            merge_abbrev: false.into(),
            batch_size: 8,
            utseg_fallback: false.into(),
        }),
        BTreeMap::new(),
    );

    // `let ... else` rather than `expect_err`, which would demand `Debug`
    // on the plan itself just to report a case that cannot happen here.
    let Err(refusal) = FaDispatchPlan::from_job(&snapshot, &ServerConfig::default()) else {
        panic!("unreadable align options must refuse the plan, not drop it");
    };
    assert!(
        matches!(refusal, DispatchPlanRefusal::Options),
        "unreadable options must say so: {refusal:?}"
    );
}

#[test]
fn media_root_plan_admits_absolute_and_refuses_historical_relative_metadata() {
    for value in ["", ".", "media", "../media"] {
        let options: CommandOptions = serde_json::from_value(serde_json::json!({
            "command": "align", "media_dir": value
        }))
        .expect("historical options remain readable");
        let snapshot = make_snapshot(ReleasedCommand::Align, options, BTreeMap::new());
        assert!(matches!(
            FaDispatchPlan::from_job(&snapshot, &ServerConfig::default()),
            Err(DispatchPlanRefusal::MediaRoot(_))
        ));
    }
    let temp = tempfile::tempdir().unwrap();
    let options = CommandOptions::Align(crate::options::AlignOptions {
        media_dir: Some(
            crate::options::AbsoluteMediaRoot::admit(temp.path())
                .unwrap()
                .into(),
        ),
        ..Default::default()
    });
    let snapshot = make_snapshot(ReleasedCommand::Align, options, BTreeMap::new());
    let plan = FaDispatchPlan::from_job(&snapshot, &ServerConfig::default()).unwrap();
    assert_eq!(plan.media_dir.unwrap().as_path(), temp.path());
}

/// RED FIRST (review item 3): the same silent drop existed on the
/// media-analysis arm, where a diarize job with a corrupt options row
/// returned `None`.
#[test]
fn media_analysis_plan_refuses_options_it_cannot_read() {
    let snapshot = make_snapshot(
        ReleasedCommand::Diarize,
        CommandOptions::Transcribe(TranscribeCommand {
            auto_speakers: false,
            common: CommonOptions::default(),
            asr_engine: AsrEngineName::RevAi,
            diarize: false,
            wor: false.into(),
            merge_abbrev: false.into(),
            batch_size: 8,
            utseg_fallback: false.into(),
        }),
        BTreeMap::new(),
    );

    let Err(refusal) = MediaAnalysisDispatchPlan::from_job(&snapshot, &ServerConfig::default())
    else {
        panic!("a corrupt diarize options row must refuse the plan, not drop it");
    };
    assert!(
        matches!(refusal, DispatchPlanRefusal::Options),
        "unreadable options must say so: {refusal:?}"
    );
}

#[test]
fn transcribe_plan_preserves_required_cache_policy() {
    let snapshot = make_snapshot(
        ReleasedCommand::Transcribe,
        CommandOptions::Transcribe(TranscribeCommand {
            auto_speakers: false,
            common: CommonOptions {
                require_media_cache: true,
                ..Default::default()
            },
            asr_engine: AsrEngineName::RevAi,
            diarize: true,
            wor: false.into(),
            merge_abbrev: false.into(),
            batch_size: 8,
            utseg_fallback: false.into(),
        }),
        BTreeMap::new(),
    );

    let plan = TranscribeDispatchPlan::from_job(&snapshot, &ServerConfig::default())
        .expect("transcribe plan");

    assert_eq!(
        plan.base_options.cache_policies,
        TranscribeCachePolicies::uniform(crate::params::CachePolicy::RequireCache)
    );
}

#[test]
fn transcribe_plan_can_refresh_rev_without_refreshing_speaker_evidence() {
    let snapshot = make_snapshot(
        ReleasedCommand::TranscribeS,
        CommandOptions::TranscribeS(TranscribeCommand {
            auto_speakers: false,
            common: CommonOptions {
                override_media_cache_tasks: vec!["rev_asr_evidence".to_owned()],
                ..Default::default()
            },
            asr_engine: AsrEngineName::RevAi,
            diarize: true,
            wor: false.into(),
            merge_abbrev: false.into(),
            batch_size: 8,
            utseg_fallback: false.into(),
        }),
        BTreeMap::new(),
    );

    let plan = TranscribeDispatchPlan::from_job(&snapshot, &ServerConfig::default())
        .expect("transcribe plan");

    assert_eq!(
        plan.base_options.cache_policies.rev_asr,
        crate::params::CachePolicy::SkipCache
    );
    assert_eq!(
        plan.base_options.cache_policies.speaker,
        crate::params::CachePolicy::UseCache
    );
}

#[test]
fn transcribe_plan_can_refresh_speaker_without_refreshing_rev_evidence() {
    let snapshot = make_snapshot(
        ReleasedCommand::TranscribeS,
        CommandOptions::TranscribeS(TranscribeCommand {
            auto_speakers: false,
            common: CommonOptions {
                override_media_cache_tasks: vec!["speaker_diarization_raw_evidence".to_owned()],
                ..Default::default()
            },
            asr_engine: AsrEngineName::RevAi,
            diarize: true,
            wor: false.into(),
            merge_abbrev: false.into(),
            batch_size: 8,
            utseg_fallback: false.into(),
        }),
        BTreeMap::new(),
    );

    let plan = TranscribeDispatchPlan::from_job(&snapshot, &ServerConfig::default())
        .expect("transcribe plan");

    assert_eq!(
        plan.base_options.cache_policies.rev_asr,
        crate::params::CachePolicy::UseCache
    );
    assert_eq!(
        plan.base_options.cache_policies.speaker,
        crate::params::CachePolicy::SkipCache
    );
}

#[test]
fn align_plan_keeps_fa_and_utr_cache_policies_distinct() {
    let snapshot = make_snapshot(
        ReleasedCommand::Align,
        CommandOptions::Align(AlignOptions {
            common: CommonOptions {
                override_media_cache_tasks: vec!["utr_asr".to_owned()],
                ..Default::default()
            },
            ..AlignOptions::default()
        }),
        BTreeMap::new(),
    );

    let plan = FaDispatchPlan::from_job(&snapshot, &ServerConfig::default()).expect("align plan");

    assert_eq!(
        plan.options.fa_params.cache_policy,
        crate::params::CachePolicy::UseCache
    );
    assert_eq!(plan.utr_cache_policy, crate::params::CachePolicy::SkipCache);
}

#[test]
fn transcribe_s_plan_honors_explicit_local_pyannote_override() {
    let mut common = CommonOptions::default();
    common.engine_overrides.speaker = Some(crate::options::SpeakerEngineName::Pyannote);
    let snapshot = make_snapshot(
        ReleasedCommand::TranscribeS,
        CommandOptions::TranscribeS(TranscribeCommand {
            auto_speakers: false,
            common,
            asr_engine: AsrEngineName::RevAi,
            diarize: true,
            wor: false.into(),
            merge_abbrev: false.into(),
            batch_size: 8,
            utseg_fallback: false.into(),
        }),
        BTreeMap::new(),
    );

    let plan = TranscribeDispatchPlan::from_job(&snapshot, &ServerConfig::default())
        .expect("transcribe_s plan");

    assert_eq!(
        plan.base_options.speaker_backend,
        Some(SpeakerBackendV2::Pyannote)
    );
}

#[test]
fn benchmark_plan_builds_rust_owned_transcribe_options() {
    let snapshot = make_snapshot(
        ReleasedCommand::Benchmark,
        CommandOptions::Benchmark(BenchmarkOptions {
            common: CommonOptions {
                override_media_cache: true,
                ..Default::default()
            },
            asr_engine: AsrEngineName::RevAi,
            wor: true.into(),
            merge_abbrev: true.into(),
        }),
        BTreeMap::new(),
    );

    let plan = BenchmarkDispatchPlan::from_job(&snapshot, &ServerConfig::default())
        .expect("benchmark plan");

    assert!(matches!(
        plan.base_options.plan.asr().backend(),
        AsrBackend::RustRevAi
    ));
    assert_eq!(plan.base_options.expected_speakers(), Some(NumSpeakers(3)));
    assert!(!plan.base_options.plan.with_utseg());
    assert!(!plan.base_options.with_morphosyntax);
    assert!(plan.base_options.write_wor);
    assert!(plan.should_merge_abbrev);
    assert!(plan.mwt.is_empty());
}

/// Benchmark scores in one language, so a job carrying anything else is
/// refused once, at plan time, for every file, rather than failing inside
/// each file's retry loop.
#[test]
fn benchmark_plan_refuses_anything_but_one_language() {
    for lang in ["auto", "eng,spa"] {
        let mut snapshot = make_snapshot(
            ReleasedCommand::Benchmark,
            CommandOptions::Benchmark(BenchmarkOptions {
                common: CommonOptions::default(),
                asr_engine: AsrEngineName::RevAi,
                wor: false.into(),
                merge_abbrev: false.into(),
            }),
            BTreeMap::new(),
        );
        snapshot.dispatch.lang = crate::api::LanguageSpec::try_from(lang).expect("a spec");
        assert!(matches!(
            BenchmarkDispatchPlan::from_job(&snapshot, &ServerConfig::default()),
            Err(DispatchPlanRefusal::BenchmarkNeedsOneLanguage(_))
        ));
    }
}

#[test]
fn media_analysis_plan_reads_opensmile_feature_set() {
    let snapshot = make_snapshot(
        ReleasedCommand::Opensmile,
        CommandOptions::Opensmile(OpensmileOptions {
            common: CommonOptions::default(),
            feature_set: "ComParE_2016".into(),
        }),
        BTreeMap::new(),
    );

    // Pin memory_tier so resolved_memory_tier() does not call
    // MemoryTier::detect() (which reads live host RAM and is not
    // mockable). The plan's `worker_bootstrap` is derived from
    // the resolved tier; without pinning, this assertion shifts
    // between dev machines (Large/Fleet → Profile) and small CI
    // runners (Small → Task). The expected kernel plan derives from this
    // same explicit configuration.
    let cfg = ServerConfig {
        memory_tier: Some(crate::types::runtime::MemoryTierKind::Large),
        ..Default::default()
    };
    let plan = MediaAnalysisDispatchPlan::from_job(&snapshot, &cfg).expect("media analysis plan");

    assert_eq!(
        plan,
        MediaAnalysisDispatchPlan::Opensmile {
            kernel_plan: CommandKernelPlan::for_command_with_policy(
                ReleasedCommand::Opensmile,
                1,
                &crate::host_policy::HostExecutionPolicy::from_server_config(&cfg),
            ),
            feature_set: "ComParE_2016".into(),
        }
    );
}

#[test]
fn standalone_diarize_plan_carries_backend_and_required_cache_policy() {
    let snapshot = make_snapshot(
        ReleasedCommand::Diarize,
        CommandOptions::Diarize(DiarizeOptions {
            common: CommonOptions {
                require_media_cache: true,
                ..Default::default()
            },
            speaker_engine: SpeakerEngineName::PyannoteAi,
            expected_speakers: None,
            output_mode: crate::options::DiarizeOutputMode::TurnsJson,
        }),
        BTreeMap::new(),
    );

    let plan = MediaAnalysisDispatchPlan::from_job(&snapshot, &ServerConfig::default())
        .expect("standalone diarize plan");
    let MediaAnalysisDispatchPlan::Diarize {
        backend,
        expected_speakers,
        cache_policy,
        ..
    } = plan
    else {
        panic!("expected standalone diarize plan");
    };

    assert_eq!(backend, SpeakerBackendV2::PyannoteAi);
    assert_eq!(expected_speakers, None);
    assert_eq!(cache_policy, crate::params::CachePolicy::RequireCache);
}
