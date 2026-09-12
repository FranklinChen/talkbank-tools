//! Shared option-building logic for job submission.
//!
//! [`build_typed_options()`] converts parsed CLI args into a [`CommandOptions`]
//! enum variant for type-safe job submission.

use crate::chat_ops::CacheTaskName;
use crate::chat_ops::cache_key::CacheOverrideTaskName;
use crate::chat_ops::fa::CaMarkerPolicy as AppCaMarkerPolicy;
use crate::chat_ops::speaker_identity::InvalidEnrollmentSet;
use crate::options::{
    AlignOptions, AvqiOptions, BenchmarkOptions, CommandOptions, CommonOptions, CompareOptions,
    CorefOptions, DiarizeOptions, EngineOverrides, MorphotagOptions, OpensmileOptions,
    SpeakerIdentifyOptions, TranscribeOptions, TranslateOptions,
    UtrOverlapStrategy as AppUtrOverlapStrategy, UtsegOptions,
};
use crate::params::{CacheOverrides, MergeAbbrevPolicy, WorTierPolicy};

use super::{
    CaMarkerPolicy as CliCaMarkerPolicy, Commands, CommonOpts, DiarizationMode, GlobalOpts,
    UtrOverlapStrategy as CliUtrOverlapStrategy,
};

/// Parse one `--engine-overrides` JSON payload into typed `EngineOverrides`.
///
/// Rejects unknown keys and invalid engine names at parse time.
pub(crate) fn parse_engine_overrides_json(input: &str) -> Result<EngineOverrides, String> {
    serde_json::from_str::<EngineOverrides>(input)
        .map_err(|error| format!("invalid --engine-overrides JSON: {error}"))
}

/// Parse an optional JSON string into typed `EngineOverrides`.
///
/// Returns default (empty) overrides if `input` is `None` or empty.
pub fn parse_engine_overrides(input: &Option<String>) -> EngineOverrides {
    // clap-validation invariant: `--engine-overrides` is parsed by
    // `clap::ValueParser` which calls `parse_engine_overrides_json`
    // and rejects invalid input at the CLI parse boundary. Reaching
    // this expect would mean the validator drifted out of sync with
    // the canonical parser: caught by the args integration tests.
    #[allow(clippy::expect_used)]
    match input.as_deref() {
        None | Some("") => EngineOverrides::default(),
        Some(json) => parse_engine_overrides_json(json)
            .expect("clap should reject invalid --engine-overrides JSON before option building"),
    }
}

/// Resolve a simple `--foo` / `--no-foo` pair after clap has applied defaults.
fn resolve_flag_pair(enabled: bool, disabled: bool) -> bool {
    enabled && !disabled
}

/// Resolve the merge-abbreviation option family into a typed policy.
fn resolve_merge_abbrev_policy(enabled: bool, disabled: bool) -> MergeAbbrevPolicy {
    resolve_flag_pair(enabled, disabled).into()
}

/// Resolve the `%wor` option family into a typed policy.
fn resolve_wor_tier_policy(enabled: bool, disabled: bool) -> WorTierPolicy {
    resolve_flag_pair(enabled, disabled).into()
}

/// Map the CLI `--review-level` enum onto the domain [`ReviewLevel`].
///
/// Shared by the `align` and `morphotag` arms so the two commands stay
/// byte-for-byte symmetric on review-tier verbosity.
///
/// [`ReviewLevel`]: crate::chat_ops::fa::ReviewLevel
fn resolve_review_level(
    level: super::commands::CliReviewLevel,
) -> crate::chat_ops::fa::ReviewLevel {
    use super::commands::CliReviewLevel;
    use crate::chat_ops::fa::ReviewLevel;
    match level {
        CliReviewLevel::None => ReviewLevel::None,
        CliReviewLevel::LowConfidence => ReviewLevel::LowConfidence,
        CliReviewLevel::All => ReviewLevel::All,
    }
}

/// Parse a wire name into a [`CacheTaskName`].
///
/// Only audio tasks are cached. Text-task names are accepted for
/// backward-compatible CLI scripting but resolve to `None` with a
/// warning.
fn parse_cache_task(name: &str) -> Option<CacheTaskName> {
    match CacheTaskName::classify_override_name(name) {
        CacheOverrideTaskName::Cacheable(task) => Some(task),
        CacheOverrideTaskName::TextNlpUnsupported => {
            eprintln!(
                "warning: --override-media-cache-tasks {name} ignored \
                 (batchalign3 does not cache text NLP)"
            );
            None
        }
        CacheOverrideTaskName::Unknown => {
            eprintln!("warning: unknown cache task name '{name}', ignoring");
            None
        }
    }
}

/// Resolve the cache override policy from CLI flags.
pub fn resolve_cache_overrides(global: &GlobalOpts) -> CacheOverrides {
    if global.media_cache.require_media_cache {
        CacheOverrides::RequireAll
    } else if !global.media_cache.override_media_cache_tasks.is_empty() {
        let tasks = global
            .media_cache
            .override_media_cache_tasks
            .iter()
            .filter_map(|s| parse_cache_task(s))
            .collect();
        CacheOverrides::Tasks(tasks)
    } else if global.media_cache.override_media_cache {
        CacheOverrides::All
    } else {
        CacheOverrides::None
    }
}

/// Resolve a `--debug-dir` path to absolute form for transmission to the
/// server. The server interprets `CommonOptions::debug_dir` against its own
/// working directory, which on a remote daemon is opaque to the user and on
/// a local daemon is rarely the user's `cwd`. By canonicalizing on the
/// client we eliminate the cross-process path-frame ambiguity.
///
/// Uses `std::path::absolute` (Rust 1.79+) which works on non-existent paths
/// (unlike `canonicalize`); the directory is created lazily when the server
/// first writes to it. The result stays in the typed `PathBuf` domain;
/// stringification for the wire `CommonOptions::debug_dir: Option<String>`
/// happens at the boundary in `build_typed_options`.
fn canonicalize_debug_dir(p: &std::path::Path) -> std::path::PathBuf {
    std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf())
}

/// Build typed command options from parsed CLI args.
///
/// `Ok(None)` for non-processing commands (serve, jobs, version, etc.).
///
/// # Why this returns a `Result`
///
/// One command's options cannot be built from arguments clap accepted.
/// `speaker-identify` takes several `--enroll` spans, and clap validates each
/// on its own; that two of them share a label, or claim the same audio for
/// different speakers, is a relation BETWEEN them that only
/// [`SpeakerIdentifyArgs::enrollment_set`] sees.
///
/// That used to be handled by discarding the error here with `.ok()` and
/// relying on `cli::run_command` having run the same check first. The ordering
/// was real and the comment describing it was true, and it was still the wrong
/// shape: nothing stopped a second caller reaching this function directly and
/// silently receiving `None`, which every other arm uses to mean "not a
/// processing command". A failing validation and an absent command were the
/// same value. They are different types now, so the confusion has no
/// representation and the check cannot be skipped.
pub fn build_typed_options(
    cmd: &Commands,
    global: &GlobalOpts,
) -> Result<Option<CommandOptions>, InvalidEnrollmentSet> {
    let common = CommonOptions {
        override_media_cache: global.media_cache.override_media_cache,
        require_media_cache: global.media_cache.require_media_cache,
        engine_overrides: parse_engine_overrides(&global.engine_overrides),
        debug_dir: global.debug_dir.as_deref().map(canonicalize_debug_dir),
        override_media_cache_tasks: global.media_cache.override_media_cache_tasks.clone(),
        ..Default::default()
    };

    Ok(match cmd {
        Commands::Align(a) => {
            // Resolution lives on `AlignArgs`, with the flags whose invariants
            // it depends on, and is infallible. It used to live here as two
            // inline ladders ending in `from_wire_name(..).ok()?`, which turned
            // a mistyped engine name into `None` from this whole function: no
            // engine, no message, and a run that proceeded as if nothing had
            // been asked for.
            let fa_engine = a.fa_selection();
            let utr_engine = a.utr_selection();
            let utr_overlap_strategy = match a.utr_args.tuning.utr_strategy {
                CliUtrOverlapStrategy::Auto => AppUtrOverlapStrategy::Auto,
                CliUtrOverlapStrategy::Global => AppUtrOverlapStrategy::Global,
                CliUtrOverlapStrategy::TwoPass => AppUtrOverlapStrategy::TwoPass,
            };
            let utr_ca_markers = match a.utr_args.tuning.utr_ca_markers {
                CliCaMarkerPolicy::Enabled => AppCaMarkerPolicy::Enabled,
                CliCaMarkerPolicy::Disabled => AppCaMarkerPolicy::Disabled,
            };
            Some(CommandOptions::Align(AlignOptions {
                common,
                fa_engine,
                utr: crate::options::AlignUtrOptions {
                    engine: utr_engine,
                    overlap_strategy: utr_overlap_strategy,
                    two_pass: crate::chat_ops::fa::TwoPassConfig {
                        ca_markers: utr_ca_markers,
                        max_exclusion_density: a.utr_args.tuning.utr_density_threshold,
                        tight_buffer_ms: a.utr_args.tuning.utr_tight_buffer,
                        match_mode: match a.utr_args.tuning.utr_fuzzy {
                            Some(threshold) => {
                                crate::chat_ops::fa::UtrMatchMode::Fuzzy { threshold }
                            }
                            None => crate::chat_ops::fa::TwoPassConfig::default().match_mode,
                        },
                    },
                },
                pauses: a.pauses,
                boundaries: crate::options::AlignBoundaryOptions {
                    existing_wor_boundaries: a.boundaries.existing_wor_boundaries,
                    end_overlap_policy: a.boundaries.end_overlap_policy,
                },
                wor: resolve_wor_tier_policy(a.wor, a.nowor),
                merge_abbrev: resolve_merge_abbrev_policy(a.merge_abbrev, a.no_merge_abbrev),
                bullet_repair: a.bullet_repair,
                review_level: resolve_review_level(a.review_level),
                media_dir: a.media_dir.clone(),
            }))
        }
        Commands::Transcribe(a) => {
            // One resolver, shared with `benchmark`, and infallible: clap has
            // already rejected an unknown engine name at parse time with the
            // valid list, so there is no failure left to handle here and no
            // `None` for a caller to mistake for something else.
            let selection = a.asr.selection();
            let asr_engine = selection.engine();

            // Apply what the NAME implies, without letting it beat what the
            // user typed. `--asr-engine paraformer` means funaudio carrying the
            // Paraformer checkpoint, but an explicit
            // `--engine-overrides '{"funaudio_model":"..."}'` is more specific
            // and wins. Discarding these would make the new name parse and then
            // silently run plain funaudio, which is the failure it exists to fix.
            let mut common = common;
            selection.apply_implied(&mut common.engine_overrides);
            if let Some(engine) = a.speaker_engine {
                common.engine_overrides.speaker = Some(engine);
            }
            // Resolve diarization: BA2 compat bools override the enum
            let diarize = if a.diarize {
                true
            } else if a.nodiarize {
                false
            } else {
                match a.diarization {
                    DiarizationMode::Auto | DiarizationMode::Disabled => false,
                    DiarizationMode::Enabled => true,
                }
            };
            let variant = TranscribeOptions {
                auto_speakers: a.auto_speakers,
                common,
                asr_engine,
                diarize,
                wor: resolve_wor_tier_policy(a.wor, a.nowor),
                merge_abbrev: resolve_merge_abbrev_policy(a.merge_abbrev, a.no_merge_abbrev),
                utseg_fallback: a.utseg_fallback_stanza.into(),
                batch_size: 8,
            };
            if diarize {
                Some(CommandOptions::TranscribeS(variant))
            } else {
                Some(CommandOptions::Transcribe(variant))
            }
        }
        Commands::Translate(a) => Some(CommandOptions::Translate(TranslateOptions {
            common,
            // No mapping: the flag parses straight into the domain enum, so
            // there is no match here to be exhaustive over the wrong type.
            translate_engine: a.translate_engine.clone(),
            merge_abbrev: resolve_merge_abbrev_policy(a.merge_abbrev, a.no_merge_abbrev),
        })),
        Commands::Morphotag(a) => Some(CommandOptions::Morphotag(MorphotagOptions {
            common,
            retokenize: a.retokenize && !a.keeptokens,
            skipmultilang: a.skipmultilang && !a.multilang,
            merge_abbrev: resolve_merge_abbrev_policy(a.merge_abbrev, a.no_merge_abbrev),
            // Keep the domain / JSON field name so the wire format remains
            // stable while the public CLI stays default-on with an explicit
            // opt-out flag.
            no_l2_morphotag: a.policy.no_l2_morphotag,
            no_pos_hints: a.policy.no_pos_hints,
            ca_policy: a.policy.ca_policy,
            // Off by default; `morphotag --review-level low-confidence|all`
            // opts in, symmetric with `align`.
            review_level: resolve_review_level(a.review_level),
        })),
        Commands::Coref(a) => Some(CommandOptions::Coref(CorefOptions {
            common,
            merge_abbrev: resolve_merge_abbrev_policy(a.merge_abbrev, a.no_merge_abbrev),
        })),
        Commands::Utseg(a) => Some(CommandOptions::Utseg(UtsegOptions {
            common,
            merge_abbrev: resolve_merge_abbrev_policy(a.merge_abbrev, a.no_merge_abbrev),
            utseg_fallback: a.utseg_fallback_stanza.into(),
        })),
        Commands::Benchmark(a) => {
            // The same resolver `transcribe` uses. This arm used to hold its own
            // copy of the whole precedence chain, including the `.ok()?` that
            // turned an unknown engine name into no engine and no message.
            let selection = a.asr.selection();
            let mut common = common;
            selection.apply_implied(&mut common.engine_overrides);
            Some(CommandOptions::Benchmark(BenchmarkOptions {
                common,
                asr_engine: selection.engine(),
                wor: resolve_wor_tier_policy(a.wor, a.nowor),
                merge_abbrev: resolve_merge_abbrev_policy(a.merge_abbrev, a.no_merge_abbrev),
            }))
        }
        Commands::Opensmile(a) => Some(CommandOptions::Opensmile(OpensmileOptions {
            common,
            feature_set: a.feature_set.clone(),
        })),
        Commands::Compare(a) => Some(CommandOptions::Compare(CompareOptions {
            common,
            merge_abbrev: resolve_merge_abbrev_policy(a.merge_abbrev, a.no_merge_abbrev),
        })),
        Commands::Avqi(_) => Some(CommandOptions::Avqi(AvqiOptions { common })),
        // `?`, not `.ok()`. Clap validates each `--enroll` on its own; that
        // two of them share a label, or claim the same audio, is a relation
        // BETWEEN them that only `enrollment_set` sees. This used to discard
        // that error and lean on `cli::run_command` having checked first,
        // which is an ordering held by a comment. Now the only way to reach
        // the options is through the validation, and a caller that skips it
        // has no signature to travel through.
        Commands::SpeakerIdentify(a) => {
            Some(CommandOptions::SpeakerIdentify(SpeakerIdentifyOptions {
                common,
                enrollments: a.enrollment_set()?,
                threshold: a.threshold,
                tiers: a.tiers.clone(),
                permutation: crate::chat_ops::speaker_identity::PermutationPlan {
                    seed: a.permutation_seed,
                    count: a.permutations,
                },
            }))
        }
        Commands::Diarize(a) => Some(CommandOptions::Diarize(DiarizeOptions {
            common,
            speaker_engine: a.speaker_engine,
            expected_speakers: a.num_speakers.map(crate::api::NumSpeakers),
        })),
        _ => None,
    })
}

/// Extract `CommonOpts` from a processing command, if present.
pub fn common_opts(cmd: &Commands) -> Option<&CommonOpts> {
    match cmd {
        Commands::Align(a) => Some(&a.common),
        Commands::Transcribe(a) => Some(&a.common),
        Commands::Translate(a) => Some(&a.common),
        Commands::Morphotag(a) => Some(&a.common),
        Commands::Coref(a) => Some(&a.common),
        Commands::Utseg(a) => Some(&a.common),
        Commands::Benchmark(a) => Some(&a.common),
        Commands::Compare(a) => Some(&a.common),
        Commands::Diarize(a) => Some(&a.common),
        Commands::SpeakerIdentify(a) => Some(&a.common),
        _ => None,
    }
}

/// Extract `--before` from commands that support incremental processing.
pub fn extract_before(cmd: &Commands) -> Option<&std::path::Path> {
    match cmd {
        Commands::Align(a) => a.incremental.before.as_deref(),
        Commands::Morphotag(a) => a.incremental.before.as_deref(),
        _ => None,
    }
}

/// Extract --bank from a processing command, if applicable.
pub fn extract_bank(cmd: &Commands) -> Option<&str> {
    match cmd {
        Commands::Benchmark(a) => a.bank.as_deref(),
        Commands::Opensmile(a) => a.bank.as_deref(),
        _ => None,
    }
}

/// Extract --subdir from a processing command, if applicable.
pub fn extract_subdir(cmd: &Commands) -> Option<&str> {
    match cmd {
        Commands::Benchmark(a) => a.subdir.as_deref(),
        Commands::Opensmile(a) => a.subdir.as_deref(),
        _ => None,
    }
}

/// Extract --lexicon path from morphotag, if present.
pub fn extract_lexicon(cmd: &Commands) -> Option<&str> {
    match cmd {
        Commands::Morphotag(a) => a.lexicon.as_deref(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Only the tests still name the engine enums directly; production code
    // reaches ASR through AsrSelection, and FA/UTR through the resolution
    // methods on the args types.
    use crate::options::{AsrEngineName, FaEngineName};

    #[test]
    fn parse_engine_overrides_none() {
        assert!(parse_engine_overrides(&None).is_empty());
    }

    #[test]
    fn parse_engine_overrides_empty_string() {
        assert!(parse_engine_overrides(&Some(String::new())).is_empty());
    }

    #[test]
    fn parse_engine_overrides_valid_json() {
        let input = Some(r#"{"asr": "tencent", "fa": "cantonese_fa"}"#.to_string());
        let overrides = parse_engine_overrides(&input);
        assert_eq!(overrides.asr, Some(AsrEngineName::HkTencent));
        assert_eq!(overrides.fa, Some(FaEngineName::Wav2vecCanto));
    }

    #[test]
    fn parse_engine_overrides_json_rejects_invalid_shape() {
        let error = parse_engine_overrides_json(r#"{"asr":{"name":"whisper"}}"#)
            .expect_err("nested objects should be rejected");
        assert!(error.contains("invalid"));
    }

    #[test]
    fn parse_engine_overrides_empty_object() {
        let input = Some("{}".to_string());
        assert!(parse_engine_overrides(&input).is_empty());
    }

    #[test]
    fn parse_engine_overrides_preserves_unknown_keys_as_extras() {
        // Unknown keys route to extras for the Python worker; known
        // keys (asr/fa/translate) still validate engine NAMES strictly
        //: see `engine_overrides_known_engine_validation_still_fires`.
        let overrides = parse_engine_overrides_json(r#"{"mor": "custom_mor"}"#)
            .expect("unknown KEYS now flow through as extras");
        assert_eq!(overrides.asr, None);
        assert_eq!(overrides.fa, None);
        assert_eq!(overrides.translate, None);
        assert_eq!(
            overrides.extras.get("mor").map(String::as_str),
            Some("custom_mor")
        );
    }

    /// `--debug-dir` is interpreted by the server (not the client). When the
    /// CLI submits a job, a relative `--debug-dir` value gets resolved against
    /// the *server's* working directory, which on a remote daemon is opaque
    /// to the user and on a local daemon is rarely the user's `cwd`. The
    /// client must canonicalize the path to absolute form before sending.
    ///
    /// Bug report (2026-04-19): running
    /// `batchalign3 transcribe in -o out2 --debug-dir debug2` against a local
    /// daemon resulted in artifacts landing at the *daemon's* working
    /// directory rather than the client's. Canonicalizing on the client side
    /// before submission fixes the asymmetry.
    #[test]
    fn canonicalize_debug_dir_resolves_relative_to_absolute() {
        let absolute = canonicalize_debug_dir(std::path::Path::new("debug2"));
        assert!(
            absolute.is_absolute(),
            "expected canonicalize_debug_dir to return absolute path, got: {}",
            absolute.display()
        );
    }

    #[test]
    fn canonicalize_debug_dir_preserves_already_absolute() {
        let input = std::path::Path::new("/tmp/already_absolute");
        let out = canonicalize_debug_dir(input);
        assert_eq!(out, std::path::PathBuf::from("/tmp/already_absolute"));
    }
}
