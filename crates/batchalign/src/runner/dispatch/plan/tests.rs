use std::collections::BTreeMap;

use serde_json::json;

use super::*;

/// The host policy plans are built against in these tests: declared, so a
/// plan is the same on every machine that runs them.
fn test_policy() -> crate::host_policy::HostExecutionPolicy {
    crate::host_facts::HostSnapshot::for_tests(&crate::config::ServerConfig::default()).policy()
}
use crate::api::{LanguageCode3, NumSpeakers, ReleasedCommand};
use crate::options::{
    AlignOptions, AsrEngineName, BenchmarkOptions, CommandOptions, CommonOptions, DiarizeOptions,
    OpensmileOptions, SpeakerEngineName, TranscribeOptions as TranscribeCommand,
};
use crate::store::RunnerDispatchConfig;
use crate::transcribe::AsrWorkerMode;
/// Guards: Submitted automatic-speaker JSON reaches recognizer and speaker plans with separate cache policies.
/// Kind: contract
/// Not a type: The family proof cannot establish how the actual JSON fields, Rev recognizer, dedicated diarizer and automatic count are translated together.
/// State: Pure admitted option values and a declared host policy need no database, filesystem or worker execution.
/// Concurrency: Each case owns its request and policy values. There is no shared cache, environment mutation or asynchronous worker task.
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
    let mut job = make_request(submission.options, BTreeMap::new());
    job.dispatch.num_speakers = NumSpeakers(2);
    let plan = TranscribeDispatchPlan::from_request(
        job.request(),
        &test_policy(),
        &crate::model_manifest::TencentEngineSetting::LARGE_MODEL_2,
    )
    .expect("dispatch plan");
    assert_eq!(
        plan.base_options.plan.asr().backend(),
        AsrBackend::RustRevAi
    );
    assert_eq!(plan.base_options.expected_speakers(), None);
    // Plan admission checks recognizer policy before worker execution.
    // An automatic count with a recognizer that labels no speakers
    // is admissible (nothing needs the count); one with a recognizer whose
    // integrated diarization must be given the count, and that is the run's
    // one speaker engine (no `diarize`), is refused by name.
    let whisper_options = serde_json::from_value(serde_json::json!({
        "command": "transcribe", "auto_speakers": true,
        "engine_overrides": {"asr": "whisper"}
    }))
    .unwrap();
    let whisper = make_request(whisper_options, BTreeMap::new());
    let whisper_plan = TranscribeDispatchPlan::from_request(
        whisper.request(),
        &test_policy(),
        &crate::model_manifest::TencentEngineSetting::LARGE_MODEL_2,
    )
    .expect("an automatic count is not the recognizer's to refuse");
    assert_eq!(whisper_plan.base_options.expected_speakers(), None);
    let tencent_options = serde_json::from_value(serde_json::json!({
        "command": "transcribe", "auto_speakers": true,
        "engine_overrides": {"asr": "tencent"}
    }))
    .unwrap();
    let tencent = make_request(tencent_options, BTreeMap::new());
    assert!(matches!(
        TranscribeDispatchPlan::from_request(
            tencent.request(),
            &test_policy(),
            &crate::model_manifest::TencentEngineSetting::LARGE_MODEL_2
        ),
        Err(DispatchPlanRefusal::SpeakerPolicy(
            crate::transcribe::SpeakerPlanError::AutoCountUnsupported { .. }
        ))
    ));
    assert!(plan.base_options.speakers.dedicated().is_some());
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

struct PlanFixture {
    dispatch: RunnerDispatchConfig,
}

impl PlanFixture {
    fn request<Options: super::request::PlanOptions>(&self) -> PlanRequest<'_, Options> {
        PlanRequest::from_dispatch(&self.dispatch, 1).expect("fixture option family")
    }

    fn media_request(&self) -> MediaAnalysisRequest<'_> {
        MediaAnalysisRequest::from_dispatch(&self.dispatch, 1).expect("fixture media family")
    }
}

fn make_request(
    options: CommandOptions,
    runtime_state: BTreeMap<String, serde_json::Value>,
) -> PlanFixture {
    PlanFixture {
        dispatch: RunnerDispatchConfig {
            command: options.command(),
            lang: crate::api::LanguageSpec::Resolved(LanguageCode3::eng()),
            num_speakers: NumSpeakers(3),
            options,
            runtime_state,
            debug_traces: false,
        },
    }
}
/// Guards: Runtime toggles and explicit Aliyun and speaker overrides reach transcription planning.
/// Kind: policy
/// Not a type: The family proof cannot establish how runtime JSON booleans, recognizer precedence, dedicated speaker routing and merge choices combine in the admitted plan.
/// State: Pure admitted option values and a declared host policy need no database, filesystem or worker execution.
/// Concurrency: Each case owns its request and policy values. There is no shared cache, environment mutation or asynchronous worker task.
#[test]
fn transcribe_plan_reads_runtime_flags_and_speaker_override() {
    let common = CommonOptions {
        override_media_cache: true,
        ..Default::default()
    };
    let mut runtime_state = BTreeMap::new();
    runtime_state.insert("utseg".into(), json!(false));
    runtime_state.insert("morphosyntax".into(), json!(true));
    let snapshot = make_request(
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

    let plan = TranscribeDispatchPlan::from_request(
        snapshot.request(),
        &test_policy(),
        &crate::model_manifest::TencentEngineSetting::LARGE_MODEL_2,
    )
    .expect("transcribe plan");

    assert_eq!(plan.kernel_plan.command, ReleasedCommand::Transcribe);
    assert!(matches!(
        plan.base_options.plan.asr().backend(),
        AsrBackend::Worker(AsrWorkerMode::HkAliyunV2)
    ));
    assert!(plan.base_options.speakers.dedicated().is_some());
    assert_eq!(
        plan.base_options.speakers.dedicated(),
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
    assert_eq!(
        plan.base_options.merge_abbrev,
        crate::params::MergeAbbrevPolicy::Merge
    );
}
/// Guards: The `transcribe_s` request retains Rev and the default dedicated Precision-2 speaker backend.
/// Kind: policy
/// Not a type: A selected transcription family cannot establish the default speaker backend, segmentation choice and cache policies assigned to the `transcribe_s` command.
/// State: Pure admitted option values and a declared host policy need no database, filesystem or worker execution.
/// Concurrency: Each case owns its request and policy values. There is no shared cache, environment mutation or asynchronous worker task.
#[test]
fn transcribe_s_plan_defaults_to_pyannote_ai_precision_2() {
    let snapshot = make_request(
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

    let plan = TranscribeDispatchPlan::from_request(
        snapshot.request(),
        &test_policy(),
        &crate::model_manifest::TencentEngineSetting::LARGE_MODEL_2,
    )
    .expect("transcribe_s plan");

    assert_eq!(plan.kernel_plan.command, ReleasedCommand::TranscribeS);
    assert!(matches!(
        plan.base_options.plan.asr().backend(),
        AsrBackend::RustRevAi
    ));
    assert!(plan.base_options.speakers.dedicated().is_some());
    assert_eq!(
        plan.base_options.speakers.dedicated(),
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
    assert_eq!(
        plan.base_options.merge_abbrev,
        crate::params::MergeAbbrevPolicy::Keep
    );
}

/// RED FIRST (review item 7): a job naming an engine this build cannot
/// run must PROPAGATE the refusal. `.ok()?` used to turn it into `None`,
/// which routing answered with a `warn!` and a bare return: the job was
/// dropped with no file failed and no error recorded.
/// Guards: A `WhisperX` request exposes its named unimplemented-engine refusal before execution.
/// Kind: policy
/// Not a type: The family proof admits a transcription option shape and cannot establish which recognized engine names have an implementation in this build.
/// State: Pure admitted option values and a declared host policy need no database, filesystem or worker execution.
/// Concurrency: Each case owns its request and policy values. There is no shared cache, environment mutation or asynchronous worker task.
#[test]
fn transcribe_plan_propagates_an_unimplemented_engine_refusal() {
    let snapshot = make_request(
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

    let refusal = TranscribeDispatchPlan::from_request(
        snapshot.request(),
        &test_policy(),
        &crate::model_manifest::TencentEngineSetting::LARGE_MODEL_2,
    )
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
/// Guards: Planning admits an absolute media root and refuses empty or relative declarations.
/// Kind: policy
/// Not a type: The request family proof cannot establish the meaning of raw stored path strings before absolute-root admission.
/// State: Pure admitted option values and a declared host policy need no database, filesystem or worker execution.
/// Concurrency: Each case owns its request and policy values. There is no shared cache, environment mutation or asynchronous worker task.
#[test]
fn media_root_plan_admits_absolute_and_refuses_relative_metadata() {
    for value in ["", ".", "media", "../media"] {
        let options: CommandOptions = serde_json::from_value(serde_json::json!({
            "command": "align", "media_dir": value
        }))
        .expect("historical options remain readable");
        let snapshot = make_request(options, BTreeMap::new());
        assert!(matches!(
            FaDispatchPlan::from_request(snapshot.request(), &test_policy()),
            Err(DispatchPlanRefusal::MediaRoot(_))
        ));
    }
    let root = std::path::Path::new("/media");
    let options = CommandOptions::Align(crate::options::AlignOptions {
        media_dir: Some(
            crate::options::AbsoluteMediaRoot::admit(root)
                .unwrap()
                .into(),
        ),
        ..Default::default()
    });
    let snapshot = make_request(options, BTreeMap::new());
    let plan = FaDispatchPlan::from_request(snapshot.request(), &test_policy()).unwrap();
    assert_eq!(plan.media_dir.unwrap().as_path(), root);
}
/// Guards: Required evidence-cache policy reaches both transcription evidence consumers.
/// Kind: policy
/// Not a type: The selected family cannot establish the propagation of required-cache policy to both recognizer and speaker evidence consumers.
/// State: Pure admitted option values and a declared host policy need no database, filesystem or worker execution.
/// Concurrency: Each case owns its request and policy values. There is no shared cache, environment mutation or asynchronous worker task.
#[test]
fn transcribe_plan_preserves_required_cache_policy() {
    let snapshot = make_request(
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

    let plan = TranscribeDispatchPlan::from_request(
        snapshot.request(),
        &test_policy(),
        &crate::model_manifest::TencentEngineSetting::LARGE_MODEL_2,
    )
    .expect("transcribe plan");

    assert_eq!(
        plan.base_options.cache_policies,
        TranscribeCachePolicies::uniform(crate::params::CachePolicy::RequireCache)
    );
}
/// Guards: A Rev-only refresh preserves speaker evidence reuse.
/// Kind: policy
/// Not a type: Both cache choices are valid and the family proof cannot establish the requested task-specific separation of recognizer refresh and speaker reuse.
/// State: Pure admitted option values and a declared host policy need no database, filesystem or worker execution.
/// Concurrency: Each case owns its request and policy values. There is no shared cache, environment mutation or asynchronous worker task.
#[test]
fn transcribe_plan_can_refresh_rev_without_refreshing_speaker_evidence() {
    let snapshot = make_request(
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

    let plan = TranscribeDispatchPlan::from_request(
        snapshot.request(),
        &test_policy(),
        &crate::model_manifest::TencentEngineSetting::LARGE_MODEL_2,
    )
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
/// Guards: A speaker-only refresh preserves Rev evidence reuse.
/// Kind: policy
/// Not a type: Both cache choices are valid and the family proof cannot establish the requested task-specific separation of speaker refresh and recognizer reuse.
/// State: Pure admitted option values and a declared host policy need no database, filesystem or worker execution.
/// Concurrency: Each case owns its request and policy values. There is no shared cache, environment mutation or asynchronous worker task.
#[test]
fn transcribe_plan_can_refresh_speaker_without_refreshing_rev_evidence() {
    let snapshot = make_request(
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

    let plan = TranscribeDispatchPlan::from_request(
        snapshot.request(),
        &test_policy(),
        &crate::model_manifest::TencentEngineSetting::LARGE_MODEL_2,
    )
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
/// Guards: A UTR-only refresh preserves forced-alignment evidence reuse.
/// Kind: policy
/// Not a type: A selected alignment family cannot establish the independent UTR and FA cache choices requested by actual task names in the common options.
/// State: Pure admitted option values and a declared host policy need no database, filesystem or worker execution.
/// Concurrency: Each case owns its request and policy values. There is no shared cache, environment mutation or asynchronous worker task.
#[test]
fn align_plan_keeps_fa_and_utr_cache_policies_distinct() {
    let snapshot = make_request(
        CommandOptions::Align(AlignOptions {
            common: CommonOptions {
                override_media_cache_tasks: vec!["utr_asr".to_owned()],
                ..Default::default()
            },
            ..AlignOptions::default()
        }),
        BTreeMap::new(),
    );

    let plan =
        FaDispatchPlan::from_request(snapshot.request(), &test_policy()).expect("align plan");

    assert_eq!(
        plan.options.fa_params.cache_policy,
        crate::params::CachePolicy::UseCache
    );
    assert_eq!(
        plan.utr_cache_policy.response(),
        crate::params::CachePolicy::SkipCache
    );
}
/// Guards: Required UTR evidence permits fresh FA and conflicting UTR cache requests are refused.
/// Kind: policy
/// Not a type: The family proof cannot establish the semantic combination of raw cache-task names, required UTR evidence, FA refresh and a contradictory UTR refresh.
/// State: Pure admitted option values and a declared host policy need no database, filesystem or worker execution.
/// Concurrency: Each case owns its request and policy values. There is no shared cache, environment mutation or asynchronous worker task.
#[test]
fn required_asr_http_policy_reaches_utr_without_forbidding_fresh_fa() {
    let snapshot = make_request(
        CommandOptions::Align(AlignOptions {
            common: serde_json::from_value(json!({
                "require_media_cache_tasks": ["utr_asr"],
                "override_media_cache_tasks": ["forced_alignment"]
            }))
            .unwrap(),
            ..Default::default()
        }),
        BTreeMap::new(),
    );
    let plan = FaDispatchPlan::from_request(snapshot.request(), &test_policy()).unwrap();
    assert_eq!(plan.utr_cache_policy.response(), CachePolicy::RequireCache);
    assert_eq!(
        plan.utr_cache_policy.rev_evidence(),
        CachePolicy::RequireCache
    );
    assert_eq!(plan.options.fa_params.cache_policy, CachePolicy::SkipCache);
    let conflicting = make_request(
        CommandOptions::Align(AlignOptions {
            common: serde_json::from_value(json!({
                "require_media_cache_tasks": ["utr_asr"],
                "override_media_cache_tasks": ["utr_asr"]
            }))
            .unwrap(),
            ..Default::default()
        }),
        BTreeMap::new(),
    );
    assert!(matches!(
        FaDispatchPlan::from_request(conflicting.request(), &test_policy()),
        Err(DispatchPlanRefusal::CachePolicy(_))
    ));
}
/// Guards: Explicit local pyannote supersedes the default dedicated speaker service.
/// Kind: policy
/// Not a type: Both diarizers are valid choices and the family proof cannot establish the precedence of the explicit local speaker override.
/// State: Pure admitted option values and a declared host policy need no database, filesystem or worker execution.
/// Concurrency: Each case owns its request and policy values. There is no shared cache, environment mutation or asynchronous worker task.
#[test]
fn transcribe_s_plan_honors_explicit_local_pyannote_override() {
    let mut common = CommonOptions::default();
    common.engine_overrides.speaker = Some(crate::options::SpeakerEngineName::Pyannote);
    let snapshot = make_request(
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

    let plan = TranscribeDispatchPlan::from_request(
        snapshot.request(),
        &test_policy(),
        &crate::model_manifest::TencentEngineSetting::LARGE_MODEL_2,
    )
    .expect("transcribe_s plan");

    assert_eq!(
        plan.base_options.speakers.dedicated(),
        Some(SpeakerBackendV2::Pyannote)
    );
}
/// Guards: Benchmark planning admits recognizer-only execution and preserves word-output choices.
/// Kind: policy
/// Not a type: The family proof cannot establish the benchmark policy that omits segmentation, morphology and abbreviation merging while retaining the requested recognizer output.
/// State: Pure admitted option values and a declared host policy need no database, filesystem or worker execution.
/// Concurrency: Each case owns its request and policy values. There is no shared cache, environment mutation or asynchronous worker task.
#[test]
fn benchmark_plan_builds_rust_owned_transcribe_options() {
    let snapshot = make_request(
        CommandOptions::Benchmark(BenchmarkOptions {
            common: CommonOptions {
                override_media_cache: true,
                ..Default::default()
            },
            asr_engine: AsrEngineName::RevAi,
            wor: true.into(),
        }),
        BTreeMap::new(),
    );

    let plan = BenchmarkDispatchPlan::from_request(
        snapshot.request(),
        &test_policy(),
        &crate::model_manifest::TencentEngineSetting::LARGE_MODEL_2,
    )
    .expect("benchmark plan");

    assert!(matches!(
        plan.base_options.plan.asr().backend(),
        AsrBackend::RustRevAi
    ));
    assert_eq!(plan.base_options.expected_speakers(), Some(NumSpeakers(3)));
    assert!(!plan.base_options.plan.with_utseg());
    assert!(!plan.base_options.with_morphosyntax);
    assert!(plan.base_options.write_wor);
    assert_eq!(
        plan.base_options.merge_abbrev,
        crate::params::MergeAbbrevPolicy::Keep
    );
    assert!(plan.mwt.is_empty());
}

/// Benchmark scores in one language, so a job carrying anything else is
/// refused once, at plan time, for every file, rather than failing inside
/// each file's retry loop.
/// Guards: Benchmark refuses automatic and multilingual language specifications before execution.
/// Kind: policy
/// Not a type: The benchmark family admits an option shape and cannot establish whether the independently supplied language declaration names one resolved comparison language.
/// State: Pure admitted option values and a declared host policy need no database, filesystem or worker execution.
/// Concurrency: Each case owns its request and policy values. There is no shared cache, environment mutation or asynchronous worker task.
#[test]
fn benchmark_plan_refuses_anything_but_one_language() {
    for lang in ["auto", "eng,spa"] {
        let mut snapshot = make_request(
            CommandOptions::Benchmark(BenchmarkOptions {
                common: CommonOptions::default(),
                asr_engine: AsrEngineName::RevAi,
                wor: false.into(),
            }),
            BTreeMap::new(),
        );
        snapshot.dispatch.lang = crate::api::LanguageSpec::try_from(lang).expect("a spec");
        assert!(matches!(
            BenchmarkDispatchPlan::from_request(
                snapshot.request(),
                &test_policy(),
                &crate::model_manifest::TencentEngineSetting::LARGE_MODEL_2
            ),
            Err(DispatchPlanRefusal::BenchmarkNeedsOneLanguage(_))
        ));
    }
}
/// Guards: The supplied `ComParE` feature set reaches openSMILE with the retained host policy.
/// Kind: policy
/// Not a type: Selected media options cannot establish preservation of the `ComParE` string or the resource plan chosen under an explicit host policy.
/// State: Pure admitted option values and a declared host policy need no database, filesystem or worker execution.
/// Concurrency: Each case owns its request and policy values. There is no shared cache, environment mutation or asynchronous worker task.
#[test]
fn media_analysis_plan_reads_opensmile_feature_set() {
    let snapshot = make_request(
        CommandOptions::Opensmile(OpensmileOptions {
            common: CommonOptions::default(),
            feature_set: "ComParE_2016".into(),
        }),
        BTreeMap::new(),
    );

    // The plan's `worker_bootstrap` is derived from the host policy, which
    // the plan takes as an argument rather than reading from the machine, so
    // the assertion is the same on every host. The expected kernel plan
    // derives from this same policy.
    let policy = test_policy();
    let plan = MediaAnalysisDispatchPlan::from_request(snapshot.media_request(), &policy)
        .expect("media analysis plan");

    assert_eq!(
        plan,
        MediaAnalysisDispatchPlan::Opensmile {
            kernel_plan: CommandKernelPlan::for_command_with_policy(
                ReleasedCommand::Opensmile,
                1,
                &policy,
            ),
            feature_set: "ComParE_2016".into(),
        }
    );
}
/// Guards: Standalone diarization retains its backend, automatic count and required evidence cache.
/// Kind: policy
/// Not a type: The media family proof cannot establish the combined backend, absence of a fixed count and required-cache policy chosen from the supplied options.
/// State: Pure admitted option values and a declared host policy need no database, filesystem or worker execution.
/// Concurrency: Each case owns its request and policy values. There is no shared cache, environment mutation or asynchronous worker task.
#[test]
fn standalone_diarize_plan_carries_backend_and_required_cache_policy() {
    let snapshot = make_request(
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

    let plan = MediaAnalysisDispatchPlan::from_request(snapshot.media_request(), &test_policy())
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

/// The measured defect, from the job's options down: one align job with
/// Whisper UTR plans ONE GPU worker. Its family plan's GPU models give the UTR
/// and FA routes the same worker, under the profile mode large hosts run.
/// Guards: Whisper UTR and forced alignment resolve to one shared GPU worker route.
/// Kind: contract
/// Not a type: The family proof cannot establish how the real worker router keys the retained FA and UTR engine identities under profile bootstrap.
/// State: A dormant local worker pool exercises its actual route-key calculation. No worker process, model or GPU is started.
/// Concurrency: Each case owns its request and policy values. There is no shared cache, environment mutation or asynchronous worker task.
#[tokio::test]
async fn an_align_job_with_whisper_utr_plans_one_gpu_worker() {
    let mut options = AlignOptions::default();
    options.utr.engine = Some(crate::options::UtrEngine::Whisper);
    let snapshot = make_request(CommandOptions::Align(options), BTreeMap::new());
    let plan =
        FaDispatchPlan::from_request(snapshot.request(), &test_policy()).expect("align plan");
    let gpu = plan.gpu_models();

    let mut config = crate::worker::pool::PoolConfig::for_tests();
    config.runtime.bootstrap_mode = crate::worker::WorkerBootstrapMode::Profile;
    let pool = crate::worker::pool::WorkerPool::new(config);
    let english = crate::api::WorkerLanguage::from(LanguageCode3::eng());
    let utr = pool
        .gpu_route(&gpu, crate::worker::pool::GpuTask::Asr, english.clone())
        .expect("Whisper UTR is a worker-hosted ASR engine");
    let fa = pool
        .gpu_route(&gpu, crate::worker::pool::GpuTask::Fa, english)
        .expect("align declares its FA engine");
    assert!(utr.shares_worker_with(&fa));
}

/// Guards: Inconsistent raw command/options tags cannot mint an alignment or media planning request.
/// Kind: boundary
/// Not a type: Incoming JSON can name an align command with transcription options or disagree within the media family before request admission establishes their relationship.
/// State: Real JSON decoding and family admission run in memory. No job registry, database, filesystem or worker is created.
/// Concurrency: Each case owns its decoded submission and configuration. There are no shared effects or background tasks.
#[test]
fn planning_request_admission_rejects_inconsistent_command_tags() {
    enum RawCase {
        AlignmentWithTranscription,
        DiarizationWithTranscription,
        OpensmileWithAvqi,
    }
    for case in [
        RawCase::AlignmentWithTranscription,
        RawCase::DiarizationWithTranscription,
        RawCase::OpensmileWithAvqi,
    ] {
        let raw = match case {
            RawCase::AlignmentWithTranscription => json!({
                "command": "align", "lang": "eng", "options": {"command": "transcribe"}
            }),
            RawCase::DiarizationWithTranscription => json!({
                "command": "diarize", "lang": "eng", "options": {"command": "transcribe"}
            }),
            RawCase::OpensmileWithAvqi => json!({
                "command": "opensmile", "lang": "eng", "options": {"command": "avqi"}
            }),
        };
        let submission: crate::api::JobSubmission =
            serde_json::from_value(raw).expect("raw request shape");
        let dispatch = RunnerDispatchConfig {
            command: submission.command,
            options: submission.options,
            lang: submission.lang,
            num_speakers: submission.num_speakers,
            runtime_state: BTreeMap::new(),
            debug_traces: false,
        };
        let refusal = match case {
            RawCase::AlignmentWithTranscription => {
                PlanRequest::<AlignOptions>::from_dispatch(&dispatch, 1).err()
            }
            RawCase::DiarizationWithTranscription | RawCase::OpensmileWithAvqi => {
                MediaAnalysisRequest::from_dispatch(&dispatch, 1).err()
            }
        };
        assert!(matches!(refusal, Some(DispatchPlanRefusal::Options)));
    }
}
