//! Consuming dispatch plans derived from selected option families.
//!
//! Request admission binds a concrete option family to borrowed language,
//! speaker, runtime and workload facts. Plan construction consumes that request
//! before orchestration begins. Callers retain the resulting plan through
//! worker selection and resource admission.

use crate::chat_ops::morphosyntax_ops::MwtDict;

use crate::chat_ops::CacheTaskName;
use crate::host_policy::HostExecutionPolicy;
use crate::options::{AlignOptions, BenchmarkOptions, TranscribeOptions as TranscribeCommand};
use crate::params::{CacheOverrides, CachePolicy};
use crate::runner::dispatch::kernel_plan::CommandKernelPlan;

mod request;
use crate::transcribe::{AsrBackend, TranscribeCachePolicies, TranscribeOptions};
use crate::types::worker_v2::SpeakerBackendV2;
use request::MediaOptions;
pub(crate) use request::{MediaAnalysisRequest, PlanRequest};

use super::options::{
    BenchmarkDispatchParams, FaDispatchParams, OpensmileDispatchParams, TranscribeDispatchParams,
    extract_benchmark_dispatch_params, extract_fa_dispatch_params,
    extract_opensmile_dispatch_params, extract_transcribe_dispatch_params,
};

/// Typed plan for forced alignment dispatch.
pub(crate) struct FaDispatchPlan {
    /// Resource-aware execution profile for the command's remaining workload.
    pub kernel_plan: CommandKernelPlan,
    /// Fully extracted FA option bundle.
    pub options: FaDispatchParams,
    /// Cache policy for the UTR ASR pre-pass and fallback paths.
    ///
    /// UTR and forced alignment are independently addressable cache tasks.
    /// Keeping this policy out of `FaParams` prevents an FA refresh request
    /// from changing UTR behavior, or vice versa.
    pub utr_cache_policy: crate::params::UtrCachePolicies,
    /// Producer-admitted root; historical request declarations cannot enter I/O.
    pub media_dir: Option<crate::options::AbsoluteMediaRoot>,
}

impl FaDispatchPlan {
    /// Consume a selected alignment request into its executable plan.
    ///
    /// Returns the typed refusal rather than `None` for the reason
    /// [`DispatchPlanRefusal`] gives: an `Option` here reached a `warn!` and a
    /// bare `return` in `runner::routing`, so an align job whose persisted
    /// options could not be read vanished with no file failed and nothing for
    /// the operator to act on.
    pub(crate) fn from_request(
        request: PlanRequest<'_, AlignOptions>,
        host_policy: &HostExecutionPolicy,
    ) -> Result<Self, DispatchPlanRefusal> {
        let options = request.options();
        let context = request.context();
        let overrides = CacheOverrides::from_common(&options.common)?;
        let fa_cache_policy = overrides.policy_for(CacheTaskName::ForcedAlignment);
        let utr_cache_policy = crate::params::UtrCachePolicies::from_overrides(&overrides);
        let params = extract_fa_dispatch_params(options, fa_cache_policy);
        let media_dir = options
            .media_dir
            .as_ref()
            .map(crate::options::MediaRootDeclaration::admit_absolute)
            .transpose()?;
        Ok(Self {
            kernel_plan: context.kernel_plan(host_policy),
            options: params,
            utr_cache_policy,
            media_dir,
        })
    }
}

impl FaDispatchPlan {
    /// The GPU-profile engines this align job's requests name: its FA engine,
    /// and its UTR engine when a worker hosts it.
    pub(crate) fn gpu_models(&self) -> crate::worker::pool::GpuModelPlan {
        crate::worker::pool::GpuModelPlan::alignment(
            self.options.fa_params.engine,
            self.options.utr_engine.as_ref().map(AsrBackend::from),
        )
    }
}

/// Why a dispatch plan could not be built.
///
/// `from_job` used to return `Option<Self>`, and the engine-selection refusal
/// reached it through `.ok()?`. A job naming an engine this build cannot run
/// therefore became `None`, indistinguishable from unreadable options, and
/// `runner::routing` answered `None` with a `warn!` and a bare `return`: the
/// job was dropped with no file failed, no error recorded and nothing for the
/// operator to see but one log line. The refusal now travels.
#[derive(Debug, thiserror::Error)]
pub(crate) enum DispatchPlanRefusal {
    /// Request/storage metadata did not establish an absolute media root.
    #[error(transparent)]
    MediaRoot(#[from] crate::options::MediaRootRefusal),
    #[error(transparent)]
    AsrPolicy(#[from] crate::transcribe::TranscribeAsrPlanError),
    /// The job's speaker count or separation is not one the engines the run
    /// involves can take.
    #[error(transparent)]
    SpeakerPolicy(#[from] crate::transcribe::SpeakerPlanError),
    /// The command and options did not establish the requested plan family.
    #[error("command plan could not be built from job options")]
    Options,
    #[error(transparent)]
    CachePolicy(#[from] crate::params::InvalidCachePolicy),
    /// Benchmark scores a recording against a gold transcript in one language.
    #[error(
        "benchmark requires one resolved `--lang <iso3>` to score against its gold transcript; got '{0}'"
    )]
    BenchmarkNeedsOneLanguage(crate::api::LanguageSpec),
    /// The job names an engine that is accepted as a name but not implemented
    /// in this build.
    #[error("{0}")]
    Engine(#[from] crate::types::engines::EngineNotImplemented),
    /// The job asks for utterance segmentation in a language that has no
    /// segmenter under its fallback policy.
    ///
    /// Raised at PLAN time, which is the point of it: this used to be
    /// discovered by the Python worker at the end of the pipeline, so the job
    /// ran a full ASR pass first and then failed with a message naming the
    /// wire format (`invalid utseg V2 result`) rather than the missing model.
    #[error(transparent)]
    UtsegUnavailable(#[from] crate::utseg_route::UtsegUnavailable),
}

/// Typed plan for transcribe dispatch.
///
/// The transcribe pipeline consumes a concrete `TranscribeOptions` bundle,
/// the abbreviation-merge choice included. Runtime-only toggles (`utseg`,
/// `morphosyntax`) are resolved here so the dispatch module stops re-reading
/// the store-owned `runtime_state` bag.
#[derive(Clone)]
pub(crate) struct TranscribeDispatchPlan {
    /// Resource-aware execution profile for the command's remaining workload.
    pub kernel_plan: CommandKernelPlan,
    /// Base transcribe options cloned per file before media-specific values are
    /// filled in.
    pub base_options: TranscribeOptions,
}

impl TranscribeDispatchPlan {
    /// Consume a selected transcription request into its executable plan.
    pub(crate) fn from_request(
        request: PlanRequest<'_, TranscribeCommand>,
        host_policy: &HostExecutionPolicy,
        tencent: &crate::model_manifest::TencentEngineSetting,
    ) -> Result<Self, DispatchPlanRefusal> {
        let options = request.options();
        let context = request.context();
        let overrides = CacheOverrides::from_common(&options.common)?;
        let cache_policies = TranscribeCachePolicies {
            rev_asr: overrides.policy_for(CacheTaskName::RevAsrEvidence),
            speaker: overrides.policy_for(CacheTaskName::SpeakerDiarizationRawEvidence),
        };
        let TranscribeDispatchParams {
            auto_speakers,
            asr_engine,
            speaker_engine,
            diarize,
            merge_abbrev,
            cache_policies,
            wor_tier,
            utseg_fallback,
            batch_size: _,
            engine_extras,
        } = extract_transcribe_dispatch_params(options, cache_policies);
        let with_utseg = runtime_flag(context.runtime_state, "utseg", true);
        let with_morphosyntax = runtime_flag(context.runtime_state, "morphosyntax", false);

        // Refuse a language with no segmenter HERE, before a single ASR
        // request is dispatched. Whether utterance segmentation can run at all
        // is a property of the job's language and its fallback policy, both of
        // which are known now; it used to be rediscovered by the Python worker
        // after the whole ASR stage had been run and paid for.
        //
        // A refusal here means a persisted job names an engine or language
        // this build cannot run. Submission validation refuses those up front,
        // so reaching this arm means the job predates the check. The refusal is
        // PROPAGATED: `.ok()?` used to turn it into `None`, which routing
        // dropped silently.
        let asr = crate::transcribe::TranscribeAsrPlan::from_request(
            AsrBackend::try_from_engine(&asr_engine)?,
            &engine_extras,
            context.language,
        )?;
        // The speaker plan is admitted against the recognizer, so a count an
        // engine the run involves cannot take is refused here, by name, and
        // the recognizer path never sees the count at all.
        let speakers = crate::transcribe::SpeakerPlan::admit(
            crate::transcribe::SpeakerCount::admit(auto_speakers, context.num_speakers)?,
            crate::transcribe::SpeakerSeparation::from_request(diarize, speaker_engine),
            // Planned with the account's Tencent engine, which decides
            // whether a Tencent recognizer separates speakers.
            Some(crate::transcribe::Recognizer::planned(
                asr.backend(),
                tencent,
            )),
        )?;

        let plan = crate::transcribe::types::AdmittedTranscribePlan::admit(
            asr,
            with_utseg,
            utseg_fallback,
        )?;

        Ok(Self {
            kernel_plan: context.kernel_plan(host_policy),
            base_options: TranscribeOptions {
                plan,
                speakers,
                with_morphosyntax,
                cache_policies,
                write_wor: wor_tier.should_write(),
                merge_abbrev,
                media_name: None,
                engine_extras,
            },
        })
    }
}

impl TranscribeDispatchPlan {
    /// The GPU-profile engines this transcribe job's requests name.
    pub(crate) fn gpu_models(&self) -> crate::worker::pool::GpuModelPlan {
        self.base_options.gpu_models()
    }
}

/// Typed plan for benchmark dispatch.
#[derive(Clone)]
pub(crate) struct BenchmarkDispatchPlan {
    /// Resource-aware execution profile for the command's remaining workload.
    pub kernel_plan: CommandKernelPlan,
    /// Base transcribe options reused by the benchmark pipeline's ASR phase.
    pub base_options: TranscribeOptions,
    /// MWT dictionary handed to the compare phase.
    pub mwt: MwtDict,
}

impl BenchmarkDispatchPlan {
    /// Consume a selected benchmark request into its executable plan.
    pub(crate) fn from_request(
        request: PlanRequest<'_, BenchmarkOptions>,
        host_policy: &HostExecutionPolicy,
        tencent: &crate::model_manifest::TencentEngineSetting,
    ) -> Result<Self, DispatchPlanRefusal> {
        let options = request.options();
        let context = request.context();
        // Refused here, once, for the whole job. This used to be a check inside
        // each file's retry loop that failed the file and continued, so every
        // retry attempt failed the same way and the file ended with no terminal
        // state recorded. The code itself is read by the ASR plan below; the
        // comparison takes none (each transcript's `@Languages:` decides its
        // morphotag), so nothing here keeps it.
        let crate::api::LanguageSpec::Resolved(_) = context.language else {
            return Err(DispatchPlanRefusal::BenchmarkNeedsOneLanguage(
                context.language.clone(),
            ));
        };
        let cache_policy =
            CacheOverrides::from_common(&options.common)?.policy_for(CacheTaskName::RevAsrEvidence);
        let BenchmarkDispatchParams {
            asr_engine,
            wor_tier,
            cache_policy,
            engine_extras,
        } = extract_benchmark_dispatch_params(options, cache_policy);

        // A refusal here means a persisted job names an engine this build
        // cannot run. Submission validation refuses those up front, so
        // reaching this arm means the job predates the check. The refusal is
        // PROPAGATED: `.ok()?` used to turn it into `None`, which routing
        // dropped silently.
        let asr = crate::transcribe::TranscribeAsrPlan::from_request(
            AsrBackend::try_from_engine(&asr_engine)?,
            &engine_extras,
            context.language,
        )?;
        // Benchmark scores the recognizer alone: its count reaches the
        // recognizer's own diarization and nothing else.
        let speakers = crate::transcribe::SpeakerPlan::admit(
            crate::transcribe::SpeakerCount::admit(false, context.num_speakers)?,
            crate::transcribe::SpeakerSeparation::RecognizerOnly,
            // Planned with the account's Tencent engine, which decides
            // whether a Tencent recognizer separates speakers.
            Some(crate::transcribe::Recognizer::planned(
                asr.backend(),
                tencent,
            )),
        )?;

        Ok(Self {
            kernel_plan: context.kernel_plan(host_policy),
            base_options: TranscribeOptions {
                plan: crate::transcribe::types::AdmittedTranscribePlan::admit(
                    asr,
                    false,
                    crate::params::UtsegFallbackPolicy::Refuse,
                )?,
                speakers,
                with_morphosyntax: false,
                cache_policies: TranscribeCachePolicies::uniform(cache_policy),
                write_wor: wor_tier.should_write(),
                // Benchmark scores what the recognizer produced; the merge
                // is a transcribe option only.
                merge_abbrev: crate::params::MergeAbbrevPolicy::Keep,
                media_name: None,
                engine_extras,
            },
            mwt: MwtDict::default(),
        })
    }
}

impl BenchmarkDispatchPlan {
    /// The GPU-profile engines this benchmark job's requests name.
    pub(crate) fn gpu_models(&self) -> crate::worker::pool::GpuModelPlan {
        self.base_options.gpu_models()
    }
}

/// Typed plan for media-analysis dispatch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum MediaAnalysisDispatchPlan {
    /// OpenSMILE needs the selected feature-set string.
    Opensmile {
        /// Resource-aware execution profile for the command's remaining workload.
        kernel_plan: CommandKernelPlan,
        /// Feature set to request from the worker.
        feature_set: String,
    },
    /// AVQI currently has no command-specific options.
    Avqi {
        /// Resource-aware execution profile for the command's remaining workload.
        kernel_plan: CommandKernelPlan,
    },
    /// Standalone diarization with an admitted source/output mode.
    Diarize {
        /// Resource-aware execution profile for the command's remaining workload.
        kernel_plan: CommandKernelPlan,
        /// Fully resolved speaker backend; dispatch cannot invent a default.
        backend: SpeakerBackendV2,
        /// Expected speaker count; `None` lets the diarizer auto-detect.
        expected_speakers: Option<crate::api::NumSpeakers>,
        /// Evidence-cache policy resolved from the job's global cache flags.
        cache_policy: CachePolicy,
        /// Explicit media evidence or declared-participant CHAT rewrite.
        output_mode: crate::options::DiarizeOutputMode,
    },
}

impl MediaAnalysisDispatchPlan {
    /// The GPU-profile engines these jobs' requests name: diarization's
    /// speaker backend; none for openSMILE and AVQI, which run on IO workers.
    pub(crate) fn gpu_models(&self) -> crate::worker::pool::GpuModelPlan {
        match self {
            Self::Diarize { backend, .. } => {
                crate::worker::pool::GpuModelPlan::diarization(*backend)
            }
            Self::Opensmile { .. } | Self::Avqi { .. } => {
                crate::worker::pool::GpuModelPlan::without_engines()
            }
        }
    }

    /// Consume one selected media request under the host's retained policy.
    pub(crate) fn from_request(
        request: MediaAnalysisRequest<'_>,
        host_policy: &HostExecutionPolicy,
    ) -> Result<Self, DispatchPlanRefusal> {
        let options = request.options();
        let context = request.context();
        let kernel_plan = context.kernel_plan(host_policy);
        Ok(match options {
            MediaOptions::Opensmile(options) => {
                let OpensmileDispatchParams { feature_set } =
                    extract_opensmile_dispatch_params(options);
                Self::Opensmile {
                    kernel_plan,
                    feature_set,
                }
            }
            MediaOptions::Avqi => Self::Avqi { kernel_plan },
            MediaOptions::Diarize(options) => Self::Diarize {
                kernel_plan,
                backend: options.speaker_engine.worker_backend(),
                expected_speakers: options
                    .expected_speakers
                    .map(|count| crate::api::NumSpeakers(count.get())),
                cache_policy: CacheOverrides::from_common(&options.common)?
                    .policy_for(CacheTaskName::SpeakerDiarizationRawEvidence),
                output_mode: options.output_mode.clone(),
            },
        })
    }
}

/// Resolve one runtime-only flag with its documented default.
fn runtime_flag(
    state: &std::collections::BTreeMap<String, serde_json::Value>,
    key: &str,
    default: bool,
) -> bool {
    state
        .get(key)
        .and_then(|value| value.as_bool())
        .unwrap_or(default)
}

#[cfg(test)]
#[path = "plan/tests.rs"]
mod tests;
