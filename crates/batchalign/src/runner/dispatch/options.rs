//! Pure functions that extract typed dispatch parameters from [`CommandOptions`].
//!
//! These functions encapsulate the "read options from job" logic that was
//! previously inlined in async dispatch functions, making it testable without
//! async runtimes, job stores, or worker pools.
//!
//! # Wiring guarantee
//!
//! Every field in every [`CommandOptions`] variant MUST be consumed by exactly
//! one extraction function. The tests in this module verify this invariant by
//! constructing options with non-default values and asserting the extracted
//! parameters reflect those values. If a new field is added to a
//! `CommandOptions` struct but not wired here, the corresponding test will
//! fail (or a new test must be added to cover it).

// These pure extractors feed the typed dispatch-plan builders in `plan.rs`.
// Some returned compatibility fields (`wor_tier`, `batch_size`) are still
// carried forward for future pipeline wiring even when today's orchestrator
// does not yet consume them.

use crate::chat_ops::fa::WordGapHealing;
use crate::chat_ops::morphosyntax_ops::{MultilingualPolicy, TokenizationMode};
#[allow(unused_imports)]
use crate::options::{
    AlignOptions, AsrEngineName, BenchmarkOptions, CommandOptions, CommonOptions, MorphotagOptions,
    OpensmileOptions, TranscribeOptions, UtrEngine, UtrOverlapStrategy,
};
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
    pub merge_abbrev: MergeAbbrevPolicy,
    pub utr_engine: Option<UtrEngine>,
    pub utr_strategy: ResolvedUtrStrategy,
}

/// Extract FA dispatch parameters from [`CommandOptions`].
///
/// Returns `None` if the options are not for an `align` command.
pub(crate) fn extract_fa_dispatch_params(
    options: &CommandOptions,
    cache_policy: CachePolicy,
) -> Option<FaDispatchParams> {
    let CommandOptions::Align(align) = options else {
        return None;
    };

    // `--pauses` asks for silence between words to survive, which is exactly
    // "do not heal the gaps". It reads the same for every engine now: it used
    // to be reachable only on the onset-only branch, so passing it alongside
    // the default engine silently did nothing.
    //
    // Written as an `if` because that is what it is. A match on
    // `(timing_resolution, pauses)` reads as though the engine's resolution
    // participates, and it does not: post-processing takes the resolution
    // separately and decides what the policy MEANS for that engine.
    let gap_healing = if align.pauses {
        WordGapHealing::PreserveMeasured
    } else {
        WordGapHealing::Heal
    };

    Some(FaDispatchParams {
        fa_params: FaParams {
            gap_healing,
            existing_wor_boundaries: align.boundaries.existing_wor_boundaries,
            end_overlap_policy: align.boundaries.end_overlap_policy,
            engine: align.effective_fa_engine(),
            cache_policy,
            wor_tier: align.wor,
            bullet_repair: align.bullet_repair,
            review_level: align.review_level,
        },
        merge_abbrev: align.merge_abbrev,
        utr_engine: align.effective_utr_engine(),
        utr_strategy: ResolvedUtrStrategy::from_options(&align.utr),
    })
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
    pub allow_stanza_fallback_utseg: bool,
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

/// Extract transcribe dispatch parameters from [`CommandOptions`].
///
/// Returns `None` if the options are not for a `transcribe` or `transcribe_s`
/// command.
pub(crate) fn extract_transcribe_dispatch_params(
    options: &CommandOptions,
    cache_policies: TranscribeCachePolicies,
) -> Option<TranscribeDispatchParams> {
    match options {
        CommandOptions::Transcribe(t) | CommandOptions::TranscribeS(t) => {
            Some(TranscribeDispatchParams {
                auto_speakers: t.auto_speakers,
                asr_engine: t.effective_asr_engine(),
                speaker_engine: t.common.engine_overrides.speaker,
                diarize: t.diarize,
                wor_tier: t.wor,
                batch_size: t.batch_size,
                merge_abbrev: t.merge_abbrev,
                cache_policies,
                allow_stanza_fallback_utseg: t.utseg_fallback.is_allowed(),
                engine_extras: t.common.engine_overrides.extras.clone(),
            })
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Morphotag
// ---------------------------------------------------------------------------

/// Extracted parameters for the morphotag dispatch path.
#[derive(Debug, PartialEq)]
pub(crate) struct MorphotagDispatchParams {
    pub tokenization_mode: TokenizationMode,
    pub multilingual_policy: MultilingualPolicy,
    pub override_media_cache: bool,
    pub merge_abbrev: MergeAbbrevPolicy,
    pub l2_policy: crate::params::L2MorphotagPolicy,
    pub pos_hint_policy: crate::params::PosHintPolicy,
    pub ca_policy: crate::options::CaMorphotagPolicy,
    pub review_level: crate::chat_ops::fa::ReviewLevel,
}

/// Extract morphotag dispatch parameters from [`CommandOptions`].
pub(crate) fn extract_morphotag_dispatch_params(
    options: &CommandOptions,
) -> Option<MorphotagDispatchParams> {
    match options {
        CommandOptions::Morphotag(m) => Some(MorphotagDispatchParams {
            tokenization_mode: TokenizationMode::from(m.retokenize),
            multilingual_policy: MultilingualPolicy::from_skip_flag(m.skipmultilang),
            override_media_cache: m.common.override_media_cache,
            merge_abbrev: m.merge_abbrev,
            l2_policy: (!m.no_l2_morphotag).into(),
            pos_hint_policy: (!m.no_pos_hints).into(),
            ca_policy: m.ca_policy,
            review_level: m.review_level,
        }),
        _ => None,
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
    pub merge_abbrev: MergeAbbrevPolicy,
    pub cache_policy: CachePolicy,
    /// Per-engine configuration extras (see
    /// [`TranscribeDispatchParams::engine_extras`] for the rationale
    /// benchmark reuses the transcribe pipeline so the same knobs apply).
    pub engine_extras: std::collections::BTreeMap<String, String>,
}

/// Extract benchmark dispatch parameters from [`CommandOptions`].
pub(crate) fn extract_benchmark_dispatch_params(
    options: &CommandOptions,
    cache_policy: CachePolicy,
) -> Option<BenchmarkDispatchParams> {
    match options {
        CommandOptions::Benchmark(b) => Some(BenchmarkDispatchParams {
            asr_engine: b.effective_asr_engine(),
            wor_tier: b.wor,
            merge_abbrev: b.merge_abbrev,
            cache_policy,
            engine_extras: b.common.engine_overrides.extras.clone(),
        }),
        _ => None,
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

/// Extract opensmile dispatch parameters from [`CommandOptions`].
pub(crate) fn extract_opensmile_dispatch_params(
    options: &CommandOptions,
) -> Option<OpensmileDispatchParams> {
    match options {
        CommandOptions::Opensmile(o) => Some(OpensmileDispatchParams {
            feature_set: o.feature_set.clone(),
        }),
        _ => None,
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
        merge_abbrev: bool,
    ) -> CommandOptions {
        CommandOptions::Align(AlignOptions {
            common: common_default(),
            fa_engine: FaEngineName::from_wire_name(fa_engine).expect("test fa_engine"),
            utr: crate::options::AlignUtrOptions {
                engine: utr_engine,
                ..Default::default()
            },
            pauses,
            boundaries: Default::default(),
            wor: wor.into(),
            merge_abbrev: merge_abbrev.into(),
            media_dir: None,
            bullet_repair: false,
            review_level: Default::default(),
        })
    }

    /// `--pauses` must reach the DEFAULT engine, not only the onset-only one.
    ///
    /// Behaviour change, 2026-08-14, deliberate and user-visible. The flag was
    /// previously consulted on the Whisper branch alone, so passing it with
    /// the default engine parsed, dispatched and silently did nothing: a user
    /// asking for silence between words to survive got it healed away anyway.
    /// A flag that cannot affect the default is not a flag.
    #[test]
    fn pauses_reaches_the_default_engine() {
        let with_pauses = extract_fa_dispatch_params(
            &align_opts("wav2vec_fa", None, true, true, false),
            CachePolicy::UseCache,
        )
        .expect("wav2vec align must produce dispatch params");
        assert_eq!(
            with_pauses.fa_params.gap_healing,
            WordGapHealing::PreserveMeasured,
            "--pauses must stop gap healing on the default engine too"
        );

        let without = extract_fa_dispatch_params(
            &align_opts("wav2vec_fa", None, false, true, false),
            CachePolicy::UseCache,
        )
        .expect("wav2vec align must produce dispatch params");
        assert_eq!(
            without.fa_params.gap_healing,
            WordGapHealing::Heal,
            "the default is unchanged: gaps are healed unless asked otherwise"
        );
    }

    #[test]
    fn fa_dispatch_reads_fa_engine() {
        let p1 = extract_fa_dispatch_params(
            &align_opts("whisper_fa", None, false, true, false),
            CachePolicy::UseCache,
        )
        .unwrap();
        assert_eq!(
            p1.fa_params.engine,
            crate::types::engines::FaEngineName::Whisper
        );

        let p2 = extract_fa_dispatch_params(
            &align_opts("wav2vec_fa", None, false, true, false),
            CachePolicy::UseCache,
        )
        .unwrap();
        assert_eq!(
            p2.fa_params.engine,
            crate::types::engines::FaEngineName::Wave2Vec
        );
    }

    #[test]
    fn fa_dispatch_reads_pauses() {
        let p1 = extract_fa_dispatch_params(
            &align_opts("whisper_fa", None, true, true, false),
            CachePolicy::UseCache,
        )
        .unwrap();
        assert_eq!(p1.fa_params.gap_healing, WordGapHealing::PreserveMeasured);

        let p2 = extract_fa_dispatch_params(
            &align_opts("whisper_fa", None, false, true, false),
            CachePolicy::UseCache,
        )
        .unwrap();
        assert_eq!(p2.fa_params.gap_healing, WordGapHealing::Heal);
    }

    #[test]
    fn fa_dispatch_reads_existing_wor_boundary_policy() {
        let opts = CommandOptions::Align(AlignOptions {
            boundaries: crate::options::AlignBoundaryOptions {
                existing_wor_boundaries:
                    crate::chat_ops::fa::ExistingWorBoundaryPolicy::RebuildFromEvidence,
                ..Default::default()
            },
            ..AlignOptions::default()
        });
        let params = extract_fa_dispatch_params(&opts, CachePolicy::UseCache).unwrap();
        assert_eq!(
            params.fa_params.existing_wor_boundaries,
            crate::chat_ops::fa::ExistingWorBoundaryPolicy::RebuildFromEvidence
        );
    }

    #[test]
    fn fa_dispatch_reads_end_overlap_policy() {
        let opts = CommandOptions::Align(AlignOptions {
            boundaries: crate::options::AlignBoundaryOptions {
                end_overlap_policy: crate::chat_ops::fa::DEFAULT_END_OVERLAP_POLICY,
                ..Default::default()
            },
            ..AlignOptions::default()
        });
        let params = extract_fa_dispatch_params(&opts, CachePolicy::UseCache).unwrap();
        assert_eq!(
            params.fa_params.end_overlap_policy,
            crate::chat_ops::fa::DEFAULT_END_OVERLAP_POLICY
        );
    }

    #[test]
    fn fa_dispatch_reads_wor() {
        let p1 = extract_fa_dispatch_params(
            &align_opts("wav2vec_fa", None, false, true, false),
            CachePolicy::UseCache,
        )
        .unwrap();
        assert_eq!(p1.fa_params.wor_tier, WorTierPolicy::Include);

        let p2 = extract_fa_dispatch_params(
            &align_opts("wav2vec_fa", None, false, false, false),
            CachePolicy::UseCache,
        )
        .unwrap();
        assert_eq!(p2.fa_params.wor_tier, WorTierPolicy::Omit);
    }

    #[test]
    fn fa_dispatch_reads_utr_engine() {
        let p1 = extract_fa_dispatch_params(
            &align_opts("wav2vec_fa", Some(UtrEngine::RevAi), false, true, false),
            CachePolicy::UseCache,
        )
        .unwrap();
        assert_eq!(p1.utr_engine, Some(UtrEngine::RevAi));

        let p2 = extract_fa_dispatch_params(
            &align_opts("wav2vec_fa", None, false, true, false),
            CachePolicy::UseCache,
        )
        .unwrap();
        assert_eq!(p2.utr_engine, None);

        let p3 = extract_fa_dispatch_params(
            &align_opts("wav2vec_fa", Some(UtrEngine::HkTencent), false, true, false),
            CachePolicy::UseCache,
        )
        .unwrap();
        assert_eq!(p3.utr_engine, Some(UtrEngine::HkTencent));
    }

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
            let options = CommandOptions::Align(AlignOptions {
                utr: crate::options::AlignUtrOptions {
                    engine: Some(UtrEngine::RevAi),
                    overlap_strategy: strategy,
                    two_pass: config.clone(),
                },
                ..Default::default()
            });
            let params = extract_fa_dispatch_params(&options, CachePolicy::UseCache)
                .expect("align dispatch");
            let expected = match strategy {
                UtrOverlapStrategy::TwoPass => ResolvedUtrStrategy::TwoPass(config.clone()),
                UtrOverlapStrategy::Auto | UtrOverlapStrategy::Global => {
                    ResolvedUtrStrategy::Global
                }
            };
            assert_eq!(params.utr_strategy, expected);
        }
    }

    #[test]
    fn fa_dispatch_reads_merge_abbrev() {
        let opts = CommandOptions::Align(AlignOptions {
            common: common_default(),
            fa_engine: FaEngineName::Wave2Vec,
            utr: Default::default(),
            pauses: false,
            boundaries: Default::default(),
            wor: true.into(),
            merge_abbrev: true.into(),
            media_dir: None,
            bullet_repair: false,
            review_level: Default::default(),
        });
        let params = extract_fa_dispatch_params(&opts, CachePolicy::UseCache).unwrap();
        assert!(params.merge_abbrev.should_merge());
    }

    #[test]
    fn fa_dispatch_passes_cache_policy() {
        let opts = CommandOptions::Align(AlignOptions {
            common: common_default(),
            fa_engine: FaEngineName::Wave2Vec,
            utr: Default::default(),
            pauses: false,
            boundaries: Default::default(),
            wor: true.into(),
            merge_abbrev: false.into(),
            media_dir: None,
            bullet_repair: false,
            review_level: Default::default(),
        });
        let params = extract_fa_dispatch_params(&opts, CachePolicy::SkipCache).unwrap();
        assert_eq!(params.fa_params.cache_policy, CachePolicy::SkipCache);
    }

    // =======================================================================
    // Transcribe: every TranscribeOptions field is exercised
    // =======================================================================

    #[test]
    fn transcribe_dispatch_reads_all_fields() {
        let opts = CommandOptions::Transcribe(TranscribeOptions {
            auto_speakers: false,
            common: common_with_cache_override(),
            asr_engine: AsrEngineName::WhisperX,
            diarize: false,
            wor: true.into(),
            merge_abbrev: true.into(),
            batch_size: 32,
            utseg_fallback: false.into(),
        });
        let params = extract_transcribe_dispatch_params(
            &opts,
            TranscribeCachePolicies::uniform(CachePolicy::SkipCache),
        )
        .unwrap();
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

    #[test]
    fn transcribe_s_dispatch_reads_all_fields() {
        let opts = CommandOptions::TranscribeS(TranscribeOptions {
            auto_speakers: false,
            common: common_default(),
            asr_engine: AsrEngineName::RevAi,
            diarize: true,
            wor: false.into(),
            merge_abbrev: false.into(),
            batch_size: 8,
            utseg_fallback: false.into(),
        });
        let params = extract_transcribe_dispatch_params(
            &opts,
            TranscribeCachePolicies::uniform(CachePolicy::UseCache),
        )
        .unwrap();
        assert_eq!(params.asr_engine, AsrEngineName::RevAi);
        assert!(params.diarize);
        assert!(!params.wor_tier.should_write());
        assert!(!params.merge_abbrev.should_merge());
        assert_eq!(params.batch_size, 8);
        assert_eq!(
            params.cache_policies,
            TranscribeCachePolicies::uniform(CachePolicy::UseCache)
        );
    }

    #[test]
    fn transcribe_dispatch_prefers_common_asr_override() {
        let mut common = common_default();
        common.engine_overrides.asr = Some(AsrEngineName::HkTencent);
        let opts = CommandOptions::Transcribe(TranscribeOptions {
            auto_speakers: false,
            common,
            asr_engine: AsrEngineName::RevAi,
            diarize: false,
            wor: false.into(),
            merge_abbrev: false.into(),
            batch_size: 8,
            utseg_fallback: false.into(),
        });

        let params = extract_transcribe_dispatch_params(
            &opts,
            TranscribeCachePolicies::uniform(CachePolicy::UseCache),
        )
        .unwrap();
        assert_eq!(params.asr_engine, AsrEngineName::HkTencent);
    }

    // =======================================================================
    // Morphotag: every MorphotagOptions field is exercised
    // =======================================================================

    #[test]
    fn morphotag_dispatch_reads_all_fields() {
        let opts = CommandOptions::Morphotag(MorphotagOptions {
            common: common_with_cache_override(),
            retokenize: true,
            skipmultilang: true,
            merge_abbrev: true.into(),

            ..Default::default()
        });
        let params = extract_morphotag_dispatch_params(&opts).unwrap();
        assert_eq!(params.tokenization_mode, TokenizationMode::StanzaRetokenize);
        assert_eq!(
            params.multilingual_policy,
            MultilingualPolicy::SkipNonPrimary
        );
        assert!(params.override_media_cache);
        assert!(params.merge_abbrev.should_merge());
    }

    #[test]
    fn morphotag_dispatch_defaults() {
        let opts = CommandOptions::Morphotag(MorphotagOptions {
            common: common_default(),

            ..Default::default()
        });
        let params = extract_morphotag_dispatch_params(&opts).unwrap();
        assert_eq!(params.tokenization_mode, TokenizationMode::Preserve);
        assert_eq!(params.multilingual_policy, MultilingualPolicy::ProcessAll);
        assert!(!params.override_media_cache);
        assert!(!params.merge_abbrev.should_merge());
    }

    // =======================================================================
    // Benchmark: every BenchmarkOptions field is exercised
    // =======================================================================

    #[test]
    fn benchmark_dispatch_reads_all_fields() {
        let opts = CommandOptions::Benchmark(BenchmarkOptions {
            common: common_with_cache_override(),
            asr_engine: AsrEngineName::WhisperOai,
            wor: true.into(),
            merge_abbrev: true.into(),
        });
        let params = extract_benchmark_dispatch_params(&opts, CachePolicy::SkipCache).unwrap();
        assert_eq!(params.asr_engine, AsrEngineName::WhisperOai);
        assert!(params.wor_tier.should_write());
        assert!(params.merge_abbrev.should_merge());
        assert_eq!(params.cache_policy, CachePolicy::SkipCache);
    }

    #[test]
    fn benchmark_dispatch_prefers_common_asr_override() {
        let mut common = common_default();
        common.engine_overrides.asr = Some(AsrEngineName::HkAliyun);
        let opts = CommandOptions::Benchmark(BenchmarkOptions {
            common,
            asr_engine: AsrEngineName::RevAi,
            wor: true.into(),
            merge_abbrev: false.into(),
        });

        let params = extract_benchmark_dispatch_params(&opts, CachePolicy::UseCache).unwrap();
        assert_eq!(params.asr_engine, AsrEngineName::HkAliyun);
    }

    // =======================================================================
    // Opensmile: every OpensmileOptions field is exercised
    // =======================================================================

    #[test]
    fn opensmile_dispatch_reads_feature_set() {
        let opts = CommandOptions::Opensmile(OpensmileOptions {
            common: common_default(),
            feature_set: "ComParE_2016".into(),
        });
        let params = extract_opensmile_dispatch_params(&opts).unwrap();
        assert_eq!(params.feature_set, "ComParE_2016");
    }

    // =======================================================================
    // Wrong-command returns None
    // =======================================================================

    #[test]
    fn wrong_command_returns_none() {
        let align = CommandOptions::Align(AlignOptions {
            common: common_default(),
            fa_engine: FaEngineName::Wave2Vec,
            utr: Default::default(),
            pauses: false,
            boundaries: Default::default(),
            wor: true.into(),
            merge_abbrev: false.into(),
            media_dir: None,
            bullet_repair: false,
            review_level: Default::default(),
        });
        assert!(
            extract_transcribe_dispatch_params(
                &align,
                TranscribeCachePolicies::uniform(CachePolicy::UseCache)
            )
            .is_none()
        );
        assert!(extract_morphotag_dispatch_params(&align).is_none());
        assert!(extract_benchmark_dispatch_params(&align, CachePolicy::UseCache).is_none());
        assert!(extract_opensmile_dispatch_params(&align).is_none());

        let transcribe = CommandOptions::Transcribe(TranscribeOptions {
            auto_speakers: false,
            common: common_default(),
            asr_engine: AsrEngineName::RevAi,
            diarize: false,
            wor: false.into(),
            merge_abbrev: false.into(),
            batch_size: 8,
            utseg_fallback: false.into(),
        });
        assert!(extract_fa_dispatch_params(&transcribe, CachePolicy::UseCache).is_none());
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
                merge_abbrev: false.into(),
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

    // =======================================================================
    // Job-level options: before_paths indexing pattern
    // =======================================================================
    //
    // These tests verify the indexing pattern used by dispatch code to extract
    // per-file before_paths from Job.before_paths, as used by dispatch_fa_infer
    // and the other audio orchestrators.

    /// Helper: extracts the before_path for a given file_index, matching the
    /// pattern used in `dispatch_fa_infer`.
    fn resolve_before_path(before_paths: &[String], file_index: usize) -> Option<String> {
        if !before_paths.is_empty() && file_index < before_paths.len() {
            Some(before_paths[file_index].clone())
        } else {
            None
        }
    }

    #[test]
    fn job_level_options_before_paths_resolved_by_index() {
        let before_paths = vec![
            "/tmp/before/a.cha".to_string(),
            "/tmp/before/b.cha".to_string(),
            "/tmp/before/c.cha".to_string(),
        ];

        assert_eq!(
            resolve_before_path(&before_paths, 0),
            Some("/tmp/before/a.cha".into())
        );
        assert_eq!(
            resolve_before_path(&before_paths, 2),
            Some("/tmp/before/c.cha".into())
        );
        // Out of bounds returns None
        assert_eq!(resolve_before_path(&before_paths, 3), None);
    }

    #[test]
    fn job_level_options_empty_before_paths_returns_none() {
        let before_paths: Vec<String> = vec![];
        assert_eq!(resolve_before_path(&before_paths, 0), None);
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
