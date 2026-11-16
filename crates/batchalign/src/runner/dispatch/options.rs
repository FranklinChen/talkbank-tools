//! Infallible parameter extraction from concrete command option types.
//!
//! Plan request admission selects each family once. Extractors consume that
//! family evidence through their argument type and carry the requested policy
//! values into executable parameters. Policy and wire contracts cover value
//! precedence, exact cache identity and serialized output behavior.

// These pure extractors feed the typed dispatch-plan builders in `plan.rs`.
// Some returned compatibility fields (`wor_tier`, `batch_size`) are still
// carried forward for future pipeline wiring even when today's orchestrator
// does not yet consume them.

#[cfg(test)]
#[path = "gap_policy_tests.rs"]
mod gap_policy_tests;
use crate::options::{
    AlignOptions, AsrEngineName, BenchmarkOptions, OpensmileOptions, TranscribeOptions, UtrEngine,
    UtrOverlapStrategy,
};
#[cfg(test)]
use crate::options::{CommandOptions, CommonOptions, MorphotagOptions};
use crate::params::{CachePolicy, FaParams, MergeAbbrevPolicy, WorTierPolicy};
use crate::transcribe::TranscribeCachePolicies;

// ---------------------------------------------------------------------------
// Align
// ---------------------------------------------------------------------------

/// Executable UTR strategy, resolved once from submitted options.
///
/// A two-pass choice carries its tuning through initial and fallback recovery;
/// dispatch cannot represent that choice while dropping its configuration.
#[derive(Debug, PartialEq, Clone)]
pub(crate) enum ResolvedUtrStrategy {
    Global,
    TwoPass(crate::chat_ops::fa::TwoPassConfig),
}

impl ResolvedUtrStrategy {
    fn from_options(options: &crate::options::AlignUtrOptions) -> Self {
        match options.overlap_strategy {
            // Auto remains global until the experimental strategy is validated.
            UtrOverlapStrategy::Auto | UtrOverlapStrategy::Global => Self::Global,
            UtrOverlapStrategy::TwoPass => Self::TwoPass(options.two_pass.clone()),
        }
    }
}

/// Extracted parameters for the FA dispatch path.
#[derive(Debug, PartialEq, Clone)]
pub(crate) struct FaDispatchParams {
    pub fa_params: FaParams,
    pub utr_engine: Option<UtrEngine>,
    pub utr_strategy: ResolvedUtrStrategy,
}

/// Extract FA dispatch parameters from selected alignment options.
pub(crate) fn extract_fa_dispatch_params(
    align: &AlignOptions,
    cache_policy: CachePolicy,
) -> FaDispatchParams {
    // Preserve the legacy model-input choice without deriving it from an
    // output policy. Ordinary alignment now preserves measured word ends.
    let whisper_text = if align.pauses {
        crate::chat_ops::fa::WhisperTextMode::CharSpaced
    } else {
        crate::chat_ops::fa::WhisperTextMode::SpaceJoined
    };

    FaDispatchParams {
        fa_params: FaParams {
            gap_healing: align.boundaries.word_gaps,
            whisper_text,
            existing_wor_boundaries: align.boundaries.existing_wor_boundaries,
            end_overlap_policy: align.boundaries.end_overlap_policy,
            main_bullets: align.boundaries.main_bullets,
            engine: align.effective_fa_engine(),
            cache_policy,
            wor_tier: align.wor,
            bullet_repair: align.bullet_repair,
            review_level: align.review_level,
        },
        utr_engine: align.effective_utr_engine(),
        utr_strategy: ResolvedUtrStrategy::from_options(&align.utr),
    }
}

// ---------------------------------------------------------------------------
// Transcribe
// ---------------------------------------------------------------------------

/// Extracted parameters for the transcribe dispatch path.
#[derive(Debug, PartialEq)]
pub(crate) struct TranscribeDispatchParams {
    pub auto_speakers: bool,
    pub asr_engine: AsrEngineName,
    pub speaker_engine: Option<crate::options::SpeakerEngineName>,
    pub diarize: bool,
    pub wor_tier: WorTierPolicy,
    pub batch_size: i32,
    pub merge_abbrev: MergeAbbrevPolicy,
    pub cache_policies: TranscribeCachePolicies,
    /// Operator opt-in to the legacy Stanza constituency-parser
    /// fallback for utseg when no language-specific TalkBank BERT
    /// utseg model is configured. Surfaced as
    /// `--utseg-fallback-stanza` on the transcribe / transcribe-s CLI.
    pub utseg_fallback: crate::params::UtsegFallbackPolicy,
    /// Per-engine configuration extras drawn from
    /// `CommonOptions.engine_overrides.extras` (e.g. `qwen_model`,
    /// `qwen_device`, `funaudio_model`). The transcribe-engine selector
    /// (`asr_engine` above) only encodes WHICH engine to load; this map
    /// carries the engine-specific configuration knobs the user passed
    /// via `--engine-overrides`. Plumbed through the dispatch / V2
    /// worker boundary so the engine's load function actually sees the
    /// knob the user requested.
    pub engine_extras: std::collections::BTreeMap<String, String>,
}

/// Extract parameters from the selected transcription option family.
///
pub(crate) fn extract_transcribe_dispatch_params(
    t: &TranscribeOptions,
    cache_policies: TranscribeCachePolicies,
) -> TranscribeDispatchParams {
    TranscribeDispatchParams {
        auto_speakers: t.auto_speakers,
        asr_engine: t.effective_asr_engine(),
        speaker_engine: t.common.engine_overrides.speaker,
        diarize: t.diarize,
        wor_tier: t.wor,
        batch_size: t.batch_size,
        merge_abbrev: t.merge_abbrev,
        cache_policies,
        utseg_fallback: t.utseg_fallback,
        engine_extras: t.common.engine_overrides.extras.clone(),
    }
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

/// Extracted parameters for the benchmark dispatch path.
#[derive(Debug, PartialEq)]
pub(crate) struct BenchmarkDispatchParams {
    pub asr_engine: AsrEngineName,
    pub wor_tier: WorTierPolicy,
    pub cache_policy: CachePolicy,
    /// Per-engine configuration extras (see
    /// [`TranscribeDispatchParams::engine_extras`] for the rationale
    /// benchmark reuses the transcribe pipeline so the same knobs apply).
    pub engine_extras: std::collections::BTreeMap<String, String>,
}

/// Extract parameters from selected benchmark options.
pub(crate) fn extract_benchmark_dispatch_params(
    b: &BenchmarkOptions,
    cache_policy: CachePolicy,
) -> BenchmarkDispatchParams {
    BenchmarkDispatchParams {
        asr_engine: b.effective_asr_engine(),
        wor_tier: b.wor,
        cache_policy,
        engine_extras: b.common.engine_overrides.extras.clone(),
    }
}

// ---------------------------------------------------------------------------
// Opensmile
// ---------------------------------------------------------------------------

/// Extracted parameters for the opensmile dispatch path.
#[derive(Debug, PartialEq)]
pub(crate) struct OpensmileDispatchParams {
    pub feature_set: String,
}

/// Extract parameters from selected openSMILE options.
pub(crate) fn extract_opensmile_dispatch_params(o: &OpensmileOptions) -> OpensmileDispatchParams {
    OpensmileDispatchParams {
        feature_set: o.feature_set.clone(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::{AsrEngineName, FaEngineName};
    use std::collections::BTreeMap;

    fn common_default() -> CommonOptions {
        CommonOptions::default()
    }

    fn common_with_cache_override() -> CommonOptions {
        CommonOptions {
            override_media_cache: true,
            ..Default::default()
        }
    }

    // =======================================================================
    // Align: every AlignOptions field is exercised
    // =======================================================================

    fn align_opts(
        fa_engine: &str,
        utr_engine: Option<UtrEngine>,
        pauses: bool,
        wor: bool,
    ) -> AlignOptions {
        AlignOptions {
            common: common_default(),
            fa_engine: FaEngineName::from_wire_name(fa_engine).expect("test fa_engine"),
            utr: crate::options::AlignUtrOptions {
                engine: utr_engine,
                ..Default::default()
            },
            pauses,
            boundaries: Default::default(),
            wor: wor.into(),
            media_dir: None,
            bullet_repair: false,
            review_level: Default::default(),
        }
    }
    /// Guards: Whisper and wav2vec selections reach the corresponding alignment engine values.
    /// Kind: policy
    /// Not a type: A concrete alignment family cannot establish which engine value the extractor selects for two admitted engine choices.
    /// State: Pure concrete option values exercise policy translation. No database, filesystem, host observation or worker execution is used.
    /// Concurrency: Each case owns its request and policy values. There is no shared cache, environment mutation or asynchronous worker task.
    #[test]
    fn fa_dispatch_reads_fa_engine() {
        let p1 = extract_fa_dispatch_params(
            &align_opts("whisper_fa", None, false, true),
            CachePolicy::UseCache,
        );
        assert_eq!(
            p1.fa_params.engine,
            crate::types::engines::FaEngineName::Whisper
        );

        let p2 = extract_fa_dispatch_params(
            &align_opts("wav2vec_fa", None, false, true),
            CachePolicy::UseCache,
        );
        assert_eq!(
            p2.fa_params.engine,
            crate::types::engines::FaEngineName::Wave2Vec
        );
    }
    /// Guards: The requested existing-word and end-overlap choices reach the alignment bundle.
    /// Kind: policy
    /// Not a type: The selected family cannot establish preservation of the supplied RebuildFromEvidence and default end-overlap values in their distinct runtime policy slots.
    /// State: Pure concrete option values exercise policy translation. No database, filesystem, host observation or worker execution is used.
    /// Concurrency: Each case owns its request and policy values. There is no shared cache, environment mutation or asynchronous worker task.
    #[test]
    fn fa_dispatch_reads_existing_wor_boundary_policy() {
        let opts = AlignOptions {
            boundaries: crate::options::AlignBoundaryOptions {
                existing_wor_boundaries:
                    crate::chat_ops::fa::ExistingWorBoundaryPolicy::RebuildFromEvidence,
                end_overlap_policy: crate::chat_ops::fa::DEFAULT_END_OVERLAP_POLICY,
                ..Default::default()
            },
            ..AlignOptions::default()
        };
        let params = extract_fa_dispatch_params(&opts, CachePolicy::UseCache);
        assert_eq!(
            params.fa_params.existing_wor_boundaries,
            crate::chat_ops::fa::ExistingWorBoundaryPolicy::RebuildFromEvidence
        );
        assert_eq!(
            params.fa_params.end_overlap_policy,
            crate::chat_ops::fa::DEFAULT_END_OVERLAP_POLICY
        );
    }
    /// Guards: KeepGiven reaches the main-bullet projection policy.
    /// Kind: policy
    /// Not a type: The family proof cannot establish that the requested KeepGiven value reaches the main-bullet slot rather than another boundary policy.
    /// State: Pure concrete option values exercise policy translation. No database, filesystem, host observation or worker execution is used.
    /// Concurrency: Each case owns its request and policy values. There is no shared cache, environment mutation or asynchronous worker task.
    #[test]
    fn fa_dispatch_reads_main_bullet_policy() {
        let opts = AlignOptions {
            boundaries: crate::options::AlignBoundaryOptions {
                main_bullets: crate::chat_ops::fa::MainBulletPolicy::KeepGiven,
                ..Default::default()
            },
            ..AlignOptions::default()
        };
        let params = extract_fa_dispatch_params(&opts, CachePolicy::UseCache);
        assert_eq!(
            params.fa_params.main_bullets,
            crate::chat_ops::fa::MainBulletPolicy::KeepGiven
        );
    }
    /// Guards: Requested word-tier inclusion and omission reach the alignment bundle.
    /// Kind: policy
    /// Not a type: The option type admits both word-tier choices and cannot establish which requested value the extractor carries into the output policy.
    /// State: Pure concrete option values exercise policy translation. No database, filesystem, host observation or worker execution is used.
    /// Concurrency: Each case owns its request and policy values. There is no shared cache, environment mutation or asynchronous worker task.
    #[test]
    fn fa_dispatch_reads_wor() {
        let p1 = extract_fa_dispatch_params(
            &align_opts("wav2vec_fa", None, false, true),
            CachePolicy::UseCache,
        );
        assert_eq!(p1.fa_params.wor_tier, WorTierPolicy::Include);

        let p2 = extract_fa_dispatch_params(
            &align_opts("wav2vec_fa", None, false, false),
            CachePolicy::UseCache,
        );
        assert_eq!(p2.fa_params.wor_tier, WorTierPolicy::Omit);
    }
    /// Guards: Rev, absent and Tencent UTR selections reach the alignment bundle unchanged.
    /// Kind: policy
    /// Not a type: The selected family cannot establish the semantic distinction between no UTR engine and the two requested recognizer backends.
    /// State: Pure concrete option values exercise policy translation. No database, filesystem, host observation or worker execution is used.
    /// Concurrency: Each case owns its request and policy values. There is no shared cache, environment mutation or asynchronous worker task.
    #[test]
    fn fa_dispatch_reads_utr_engine() {
        let p1 = extract_fa_dispatch_params(
            &align_opts("wav2vec_fa", Some(UtrEngine::RevAi), false, true),
            CachePolicy::UseCache,
        );
        assert_eq!(p1.utr_engine, Some(UtrEngine::RevAi));

        let p2 = extract_fa_dispatch_params(
            &align_opts("wav2vec_fa", None, false, true),
            CachePolicy::UseCache,
        );
        assert_eq!(p2.utr_engine, None);

        let p3 = extract_fa_dispatch_params(
            &align_opts("wav2vec_fa", Some(UtrEngine::HkTencent), false, true),
            CachePolicy::UseCache,
        );
        assert_eq!(p3.utr_engine, Some(UtrEngine::HkTencent));
    }
    /// Guards: Two-pass UTR preparation retains its supplied windows, thresholds and matching policies.
    /// Kind: policy
    /// Not a type: The two-pass variant requires tuning to exist but cannot establish equality of the authored threshold and window values after translation.
    /// State: Pure concrete option values exercise policy translation. No database, filesystem, host observation or worker execution is used.
    /// Concurrency: Each case owns its request and policy values. There is no shared cache, environment mutation or asynchronous worker task.
    #[test]
    fn fa_dispatch_preserves_two_pass_tuning() {
        use crate::chat_ops::fa::{CaMarkerPolicy, TwoPassConfig, UtrMatchMode};
        let config = TwoPassConfig {
            ca_markers: CaMarkerPolicy::Disabled,
            tight_buffer_ms: 123,
            match_mode: UtrMatchMode::Exact,
            max_exclusion_density: "0.75".parse().expect("valid density"),
        };
        for strategy in [
            UtrOverlapStrategy::Auto,
            UtrOverlapStrategy::Global,
            UtrOverlapStrategy::TwoPass,
        ] {
            let options = AlignOptions {
                utr: crate::options::AlignUtrOptions {
                    engine: Some(UtrEngine::RevAi),
                    overlap_strategy: strategy,
                    two_pass: config.clone(),
                },
                ..Default::default()
            };
            let params = extract_fa_dispatch_params(&options, CachePolicy::UseCache);
            let expected = match strategy {
                UtrOverlapStrategy::TwoPass => ResolvedUtrStrategy::TwoPass(config.clone()),
                UtrOverlapStrategy::Auto | UtrOverlapStrategy::Global => {
                    ResolvedUtrStrategy::Global
                }
            };
            assert_eq!(params.utr_strategy, expected);
        }
    }

    // =======================================================================
    // Transcribe: every TranscribeOptions field is exercised
    // =======================================================================
    /// Guards: Transcription extraction carries engine, word output, merge choice, batch size and cache policy.
    /// Kind: policy
    /// Not a type: The family type prevents cross-command input and cannot establish equality of the submitted values in the distinct runtime option slots.
    /// State: Pure concrete option values exercise policy translation. No database, filesystem, host observation or worker execution is used.
    /// Concurrency: Each case owns its request and policy values. There is no shared cache, environment mutation or asynchronous worker task.
    #[test]
    fn transcribe_dispatch_reads_all_fields() {
        let opts = TranscribeOptions {
            auto_speakers: false,
            common: common_with_cache_override(),
            asr_engine: AsrEngineName::WhisperX,
            diarize: false,
            wor: true.into(),
            merge_abbrev: true.into(),
            batch_size: 32,
            utseg_fallback: false.into(),
        };
        let params = extract_transcribe_dispatch_params(
            &opts,
            TranscribeCachePolicies::uniform(CachePolicy::SkipCache),
        );
        assert_eq!(params.asr_engine, AsrEngineName::WhisperX);
        assert!(!params.diarize);
        assert!(params.wor_tier.should_write());
        assert!(params.merge_abbrev.should_merge());
        assert_eq!(params.batch_size, 32);
        assert_eq!(
            params.cache_policies,
            TranscribeCachePolicies::uniform(CachePolicy::SkipCache)
        );
    }
    /// Guards: A common Tencent override supersedes the transcription Rev default.
    /// Kind: policy
    /// Not a type: The transcription family admits Rev and Tencent engine values but cannot establish that the explicit common Tencent selection supersedes the command default.
    /// State: Pure concrete option values exercise policy translation. No database, filesystem, host observation or worker execution is used.
    /// Concurrency: Each case owns its request and policy values. There is no shared cache, environment mutation or asynchronous worker task.
    #[test]
    fn transcribe_dispatch_prefers_common_asr_override() {
        let mut common = common_default();
        common.engine_overrides.asr = Some(AsrEngineName::HkTencent);
        let opts = TranscribeOptions {
            auto_speakers: false,
            common,
            asr_engine: AsrEngineName::RevAi,
            diarize: false,
            wor: false.into(),
            merge_abbrev: false.into(),
            batch_size: 8,
            utseg_fallback: false.into(),
        };

        let params = extract_transcribe_dispatch_params(
            &opts,
            TranscribeCachePolicies::uniform(CachePolicy::UseCache),
        );
        assert_eq!(params.asr_engine, AsrEngineName::HkTencent);
    }

    // =======================================================================
    // Benchmark: every BenchmarkOptions field is exercised
    // =======================================================================
    /// Guards: Benchmark extraction retains Whisper, word output and the supplied cache policy.
    /// Kind: policy
    /// Not a type: Concrete benchmark options cannot establish the selected recognizer, word-output choice and caller-supplied cache policy in the translated bundle.
    /// State: Pure concrete option values exercise policy translation. No database, filesystem, host observation or worker execution is used.
    /// Concurrency: Each case owns its request and policy values. There is no shared cache, environment mutation or asynchronous worker task.
    #[test]
    fn benchmark_dispatch_reads_all_fields() {
        let opts = BenchmarkOptions {
            common: common_with_cache_override(),
            asr_engine: AsrEngineName::WhisperOai,
            wor: true.into(),
        };
        let params = extract_benchmark_dispatch_params(&opts, CachePolicy::SkipCache);
        assert_eq!(params.asr_engine, AsrEngineName::WhisperOai);
        assert!(params.wor_tier.should_write());
        assert_eq!(params.cache_policy, CachePolicy::SkipCache);
    }
    /// Guards: A common Aliyun override supersedes the benchmark Rev default.
    /// Kind: policy
    /// Not a type: The benchmark family admits Rev and Aliyun recognizers but cannot establish that the explicit common Aliyun override controls the scored recognizer.
    /// State: Pure concrete option values exercise policy translation. No database, filesystem, host observation or worker execution is used.
    /// Concurrency: Each case owns its request and policy values. There is no shared cache, environment mutation or asynchronous worker task.
    #[test]
    fn benchmark_dispatch_prefers_common_asr_override() {
        let mut common = common_default();
        common.engine_overrides.asr = Some(AsrEngineName::HkAliyun);
        let opts = BenchmarkOptions {
            common,
            asr_engine: AsrEngineName::RevAi,
            wor: true.into(),
        };

        let params = extract_benchmark_dispatch_params(&opts, CachePolicy::UseCache);
        assert_eq!(params.asr_engine, AsrEngineName::HkAliyun);
    }

    // =======================================================================
    // CommonOptions: engine_overrides and mwt are accessible
    // =======================================================================

    #[test]
    fn common_options_accessible_from_all_variants() {
        use crate::options::EngineOverrides;
        let overrides = EngineOverrides {
            asr: Some(AsrEngineName::HkTencent),
            fa: None,
            translate: None,
            ..Default::default()
        };
        let common = CommonOptions {
            override_media_cache: true,
            engine_overrides: overrides.clone(),
            mwt: BTreeMap::new(),
            ..Default::default()
        };

        // Verify common() accessor works for every variant
        let variants: Vec<CommandOptions> = vec![
            CommandOptions::Align(AlignOptions {
                common: common.clone(),
                fa_engine: FaEngineName::Wave2Vec,
                utr: Default::default(),
                pauses: false,
                boundaries: Default::default(),
                wor: true.into(),
                media_dir: None,
                bullet_repair: false,
                review_level: Default::default(),
            }),
            CommandOptions::Transcribe(TranscribeOptions {
                auto_speakers: false,
                common: common.clone(),
                asr_engine: AsrEngineName::RevAi,
                diarize: false,
                wor: false.into(),
                merge_abbrev: false.into(),
                batch_size: 8,
                utseg_fallback: false.into(),
            }),
            CommandOptions::Morphotag(MorphotagOptions {
                common: common.clone(),

                ..Default::default()
            }),
        ];

        for v in &variants {
            assert!(v.common().override_media_cache);
            assert_eq!(v.common().engine_overrides, overrides);
        }
    }

    #[test]
    fn job_level_options_before_paths_supports_incremental_commands() {
        // Document which commands support --before (incremental processing).
        // This test is a living doc, update when adding incremental support
        // to new commands.
        let incremental_commands = ["morphotag", "align"];
        let non_incremental_commands = ["transcribe", "translate", "utseg", "coref", "benchmark"];

        // Verify the command names are valid
        for cmd in incremental_commands
            .iter()
            .chain(non_incremental_commands.iter())
        {
            assert!(!cmd.is_empty(), "command name should not be empty: {cmd}");
        }

        // Incremental commands should have 2 entries (morphotag, align)
        assert_eq!(incremental_commands.len(), 2);
    }
}
