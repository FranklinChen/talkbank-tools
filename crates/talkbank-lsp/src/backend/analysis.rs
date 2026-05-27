//! CLAN analysis integration for the LSP.
//!
//! Handles `talkbank/analyze` execute-command requests by running analysis
//! commands from `talkbank_clan` and returning JSON results.
//!
//! This module is intentionally an adapter, not a command-construction hub.
//! `talkbank-clan` now owns the reusable `AnalysisCommandName`,
//! `AnalysisRequestBuilder`, and `AnalysisService` boundaries, so the LSP only
//! translates execute-command payloads into typed library inputs.
//!
//! # Related CHAT Manual Sections
//!
//! - <https://talkbank.org/0info/manuals/CHAT.html#File_Format>
//! - <https://talkbank.org/0info/manuals/CHAT.html#File_Headers>
//! - <https://talkbank.org/0info/manuals/CHAT.html#Main_Tier>
//! - <https://talkbank.org/0info/manuals/CHAT.html#Dependent_Tiers>

use serde_json::Value;
use talkbank_clan::database;
use talkbank_clan::framework::DiscoveredChatFiles;
use talkbank_clan::service::AnalysisService;
use talkbank_clan::service_types::{
    AnalysisCommandName, AnalysisOptions, AnalysisPlan, AnalysisRequestBuilder,
};
use tower_lsp::jsonrpc::Result as LspResult;
use tower_lsp::lsp_types::Url;

use super::LspBackendError;
use super::execute_commands::{AnalyzeRequest, DiscoverDatabasesRequest, ExecuteCommandRequest};

/// Feature-oriented execute-command service for CLAN analysis requests.
pub(crate) struct AnalysisCommandService;

impl AnalysisCommandService {
    /// Dispatch one analysis-family execute-command request.
    pub(crate) fn dispatch(&self, request: ExecuteCommandRequest) -> LspResult<Option<Value>> {
        // Routing invariant: `ExecuteCommandRoutingService::dispatch`
        // (in `requests/execute_command.rs`) calls `request.family()`
        // and routes to the analysis service only for variants whose
        // family is `ExecuteCommandFamily::Analysis`. The wildcard arm
        // is therefore unreachable by construction. Follow-up:
        // partition `ExecuteCommandRequest` into typed sub-enums per
        // family so the compiler proves exhaustiveness — see
        // `docs/panic-audit/talkbank-lsp.md`.
        #[allow(clippy::unreachable)]
        match request {
            ExecuteCommandRequest::Analyze(request) => {
                command_response(handle_analyze(&request), "Analysis error")
            }
            ExecuteCommandRequest::KidevalDatabases(request)
            | ExecuteCommandRequest::EvalDatabases(request) => command_response(
                handle_discover_databases(&request),
                "Database discovery error",
            ),
            _ => unreachable!("analysis service received unsupported execute-command request"),
        }
    }
}

fn command_response(
    result: Result<Value, LspBackendError>,
    prefix: &str,
) -> LspResult<Option<Value>> {
    match result {
        Ok(json) => Ok(Some(json)),
        Err(error) => Ok(Some(Value::String(format!("{prefix}: {error}")))),
    }
}

/// Handle a `talkbank/analyze` execute-command request.
///
/// Returns JSON output from the analysis command.
pub(crate) fn handle_analyze(request: &AnalyzeRequest) -> Result<Value, LspBackendError> {
    let target_uri =
        Url::parse(&request.target_uri).map_err(LspBackendError::invalid_uri_parse("file URI"))?;
    let file_path = target_uri
        .to_file_path()
        .map_err(LspBackendError::uri_not_file_path("file URI"))?;

    let options = build_analysis_options(request)?;
    let plan = AnalysisRequestBuilder::new(options).build().map_err(
        |error: talkbank_clan::service_types::AnalysisServiceError| {
            LspBackendError::ExternalServiceFailed {
                service: "Analysis plan build",
                reason: error.to_string(),
            }
        },
    )?;
    let service = AnalysisService::new();

    match plan {
        AnalysisPlan::Service(analysis_request) => {
            let discovered_files = DiscoveredChatFiles::from_path(&file_path);
            if discovered_files.is_empty() {
                return Err(LspBackendError::ExternalServiceFailed {
                    service: "Analysis",
                    reason: "No .cha files found".to_string(),
                });
            }
            let files = discovered_files.into_files();

            service
                .execute_json(analysis_request, &files)
                .map_err(|error| LspBackendError::ExternalServiceFailed {
                    service: "Analysis",
                    reason: error.to_string(),
                })
        }
        AnalysisPlan::Rely(rely_request) => service
            .execute_rely_json(rely_request, &file_path)
            .map_err(|error| LspBackendError::ExternalServiceFailed {
                service: "Analysis (rely)",
                reason: error.to_string(),
            }),
    }
}

/// Translate the LSP's typed execute-command payload into raw
/// library options.
///
/// Populates only the nested options struct matching
/// `request.command_name`; the rest stay at their defaults. The
/// builder reads only one variant per call, so populating the
/// others would just be clones thrown away.
fn build_analysis_options(request: &AnalyzeRequest) -> Result<AnalysisOptions, LspBackendError> {
    use talkbank_clan::framework::{
        CodeDepth, FrequencyThreshold, KeywordPattern, TierKind, UtteranceLimit, WordLimit,
    };
    use talkbank_clan::service_types::{
        ChainsOptions, CodesOptions, ComboOptions, CorelexOptions, DssOptions,
        EvalOptions as EvalOpts, FlucalcOptions, FreqOptions, IpsynOptions, KeymapOptions,
        KidevalOptions, KwalOptions, MaxwdOptions, MltOptions, MluOptions, MortableOptions,
        RelyOptions, ScriptOptions, SugarOptions, TrnfixOptions, UniqOptions, VocdOptions,
        WdsizeOptions,
    };

    let options = &request.options;
    let tier = || options.tier.as_deref().map(TierKind::from);
    let capitalization = || match options.capitalization.as_deref() {
        Some("initial") => talkbank_clan::framework::CapitalizationFilter::InitialUpper,
        Some("mid") => talkbank_clan::framework::CapitalizationFilter::MidUpper,
        _ => talkbank_clan::framework::CapitalizationFilter::Any,
    };
    let keywords = || -> Vec<KeywordPattern> {
        options
            .keywords
            .iter()
            .map(|s| KeywordPattern::from(s.as_str()))
            .collect()
    };
    let max_utterances = || options.max_utterances.map(UtteranceLimit::from);
    let database_path = || options.database_path.clone();
    let database_filter = || options.database_filter.clone().map(Into::into);

    let rely_second_file = || -> Result<Option<std::path::PathBuf>, LspBackendError> {
        options
            .second_file
            .as_deref()
            .map(|uri| {
                let url = Url::parse(uri)
                    .map_err(LspBackendError::invalid_uri_parse("second file URI"))?;
                url.to_file_path()
                    .map_err(LspBackendError::uri_not_file_path("second file URI"))
            })
            .transpose()
    };

    let built = match request.command_name {
        AnalysisCommandName::Freq => AnalysisOptions::Freq(FreqOptions {
            mor: options.mor,
            capitalization: capitalization(),
            reverse_concordance: false,
            word_list_only: false,
            types_tokens_only: false,
            case_sensitive: false,
            // LSP `talkbank/analyze` does not surface CLAN's
            // `+sWORD` / `-sWORD` patterns yet; pass an empty
            // per-word filter so FREQ emits all words. The mode
            // MUST be ``PerWordEmit`` because FREQ applies word
            // filtering at per-word emit time, not at the
            // utterance gate (`Default::default()` would set
            // `UtteranceContext` which is silently wrong for
            // FREQ — see the comment on
            // ``FreqOptions::word_filter`` in talkbank-clan).
            word_filter: talkbank_clan::framework::WordFilter {
                include: Vec::new(),
                exclude: Vec::new(),
                case_sensitive: false,
                mode: talkbank_clan::framework::WordFilterMode::PerWordEmit,
            },
        }),
        AnalysisCommandName::Mlu => AnalysisOptions::Mlu(MluOptions {
            words: options.words,
            solo_word_exclusions: options.solo_word_exclusions.clone(),
        }),
        AnalysisCommandName::Mlt => AnalysisOptions::Mlt(MltOptions {
            solo_word_exclusions: options.solo_word_exclusions.clone(),
        }),
        AnalysisCommandName::Wdsize => AnalysisOptions::Wdsize(WdsizeOptions {
            main_tier: options.main_tier,
            length_filter: None,
        }),
        AnalysisCommandName::Maxwd => AnalysisOptions::Maxwd(MaxwdOptions {
            limit: options.limit.map(WordLimit::from),
            exclude_lengths: options.exclude_lengths.clone(),
            ..MaxwdOptions::default()
        }),
        AnalysisCommandName::Kwal => AnalysisOptions::Kwal(KwalOptions {
            keywords: keywords(),
            strict_match: options.strict_match,
            case_sensitive: false,
            legal_chat: false,
            context_before: 0,
            context_after: 0,
        }),
        AnalysisCommandName::Combo => AnalysisOptions::Combo(ComboOptions {
            search: options.search.clone(),
            exclude_search: options.exclude_search.clone(),
            first_match_only: options.first_match_only,
            dedupe_matches: options.dedupe_matches,
            case_sensitive: false,
            context_before: 0,
            context_after: 0,
        }),
        AnalysisCommandName::Dist => {
            AnalysisOptions::Dist(talkbank_clan::service_types::DistOptions {
                once_per_turn: options.once_per_turn,
                case_sensitive: false,
            })
        }
        AnalysisCommandName::Vocd => AnalysisOptions::Vocd(VocdOptions {
            capitalization: capitalization(),
            case_sensitive: false,
        }),
        AnalysisCommandName::Codes => AnalysisOptions::Codes(CodesOptions {
            max_depth: options.max_depth.map(CodeDepth::from),
        }),
        AnalysisCommandName::Chains => AnalysisOptions::Chains(ChainsOptions { tier: tier() }),
        AnalysisCommandName::Corelex => AnalysisOptions::Corelex(CorelexOptions {
            threshold: options.threshold.map(FrequencyThreshold::from),
        }),
        AnalysisCommandName::Dss => AnalysisOptions::Dss(DssOptions {
            rules_path: None,
            max_utterances: max_utterances(),
        }),
        AnalysisCommandName::Eval => AnalysisOptions::Eval(EvalOpts {
            database_path: database_path(),
            database_filter: database_filter(),
        }),
        AnalysisCommandName::EvalDialect => AnalysisOptions::EvalDialect(EvalOpts {
            database_path: database_path(),
            database_filter: database_filter(),
        }),
        AnalysisCommandName::Flucalc => AnalysisOptions::Flucalc(FlucalcOptions {
            syllable_mode: options.syllable_mode,
        }),
        AnalysisCommandName::Ipsyn => AnalysisOptions::Ipsyn(IpsynOptions {
            rules_path: None,
            max_utterances: max_utterances(),
        }),
        AnalysisCommandName::Keymap => AnalysisOptions::Keymap(KeymapOptions {
            keywords: keywords(),
            tier: tier(),
        }),
        AnalysisCommandName::Kideval => AnalysisOptions::Kideval(KidevalOptions {
            dss_rules_path: None,
            ipsyn_rules_path: None,
            dss_max_utterances: options.dss_max_utterances.map(UtteranceLimit::from),
            ipsyn_max_utterances: options.ipsyn_max_utterances.map(UtteranceLimit::from),
            database_path: database_path(),
            database_filter: database_filter(),
        }),
        AnalysisCommandName::Mortable => AnalysisOptions::Mortable(MortableOptions {
            script_path: options.script_path.clone(),
        }),
        AnalysisCommandName::Rely => AnalysisOptions::Rely(RelyOptions {
            second_file: rely_second_file()?,
            tier: tier(),
        }),
        AnalysisCommandName::Script => AnalysisOptions::Script(ScriptOptions {
            template_path: options.template_path.clone(),
        }),
        AnalysisCommandName::Sugar => AnalysisOptions::Sugar(SugarOptions {
            min_utterances: options.min_utterances.map(UtteranceLimit::from),
        }),
        AnalysisCommandName::Trnfix => AnalysisOptions::Trnfix(TrnfixOptions {
            tier1: options.tier1.as_deref().map(TierKind::from),
            tier2: options.tier2.as_deref().map(TierKind::from),
        }),
        AnalysisCommandName::Uniq => AnalysisOptions::Uniq(UniqOptions {
            sort_by_frequency: options.sort_by_frequency,
        }),
        AnalysisCommandName::Wdlen => AnalysisOptions::Wdlen,
        AnalysisCommandName::Freqpos => {
            AnalysisOptions::Freqpos(talkbank_clan::service_types::FreqposOptions::default())
        }
        AnalysisCommandName::Timedur => AnalysisOptions::Timedur,
        AnalysisCommandName::Gemlist => AnalysisOptions::Gemlist,
        AnalysisCommandName::Cooccur => {
            AnalysisOptions::Cooccur(talkbank_clan::service_types::CooccurOptions::default())
        }
        AnalysisCommandName::Chip => AnalysisOptions::Chip,
        AnalysisCommandName::Phonfreq => AnalysisOptions::Phonfreq,
        AnalysisCommandName::Modrep => AnalysisOptions::Modrep,
        AnalysisCommandName::Complexity => AnalysisOptions::Complexity,
    };
    Ok(built)
}

/// Handle a `talkbank/kidevalDatabases` request.
///
/// Returns JSON array of available databases.
pub(crate) fn handle_discover_databases(
    request: &DiscoverDatabasesRequest,
) -> Result<Value, LspBackendError> {
    let databases = database::discover_databases(&request.library_dir).map_err(|e| {
        LspBackendError::ExternalServiceFailed {
            service: "Database discovery",
            reason: e.to_string(),
        }
    })?;

    Ok(serde_json::to_value(&databases)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use crate::backend::execute_commands::{AnalysisOptionsRequest, AnalyzeRequest};
    use talkbank_clan::service_types::AnalysisCommandName;

    #[test]
    fn build_analysis_options_converts_second_file_uri() {
        let secondary = Url::from_file_path("/tmp/secondary.cha").expect("secondary file URL");
        let request = AnalyzeRequest {
            command_name: AnalysisCommandName::Rely,
            target_uri: "file:///tmp/primary.cha".to_owned(),
            options: AnalysisOptionsRequest {
                second_file: Some(secondary.to_string()),
                ..AnalysisOptionsRequest::default()
            },
        };

        let options = build_analysis_options(&request).expect("options should build");

        match options {
            AnalysisOptions::Rely(o) => {
                assert_eq!(o.second_file, Some(PathBuf::from("/tmp/secondary.cha")))
            }
            other => panic!("expected Rely variant, got {other:?}"),
        }
    }

    #[test]
    fn build_analysis_options_rejects_non_file_second_uri() {
        let request = AnalyzeRequest {
            command_name: AnalysisCommandName::Rely,
            target_uri: "file:///tmp/primary.cha".to_owned(),
            options: AnalysisOptionsRequest {
                second_file: Some("https://example.com/secondary.cha".to_owned()),
                ..AnalysisOptionsRequest::default()
            },
        };

        let error = build_analysis_options(&request).expect_err("non-file URI should fail");
        assert!(
            matches!(
                &error,
                LspBackendError::UriNotFilePath {
                    label: "second file URI",
                },
            ),
            "expected UriNotFilePath {{ label: 'second file URI' }}, got {error:?}",
        );
    }

    /// DIST's `oncePerTurn` field (CLAN `+g`) threads from the LSP
    /// request schema to `DistOptions::once_per_turn`. Default
    /// (omitted) ⇒ `false`; explicit `true` ⇒ `true`.
    #[test]
    fn build_analysis_options_routes_dist_once_per_turn() {
        let request = AnalyzeRequest {
            command_name: AnalysisCommandName::Dist,
            target_uri: "file:///tmp/x.cha".to_owned(),
            options: AnalysisOptionsRequest {
                once_per_turn: true,
                ..AnalysisOptionsRequest::default()
            },
        };
        match build_analysis_options(&request).expect("options should build") {
            AnalysisOptions::Dist(o) => assert!(o.once_per_turn),
            other => panic!("expected Dist variant, got {other:?}"),
        }
    }

    /// FREQ's `capitalization: "initial"` maps to
    /// `CapitalizationFilter::InitialUpper`. Mirrors the CLI's
    /// `--capitalization initial`.
    #[test]
    fn build_analysis_options_routes_freq_initial_capitalization() {
        use talkbank_clan::framework::CapitalizationFilter;
        let request = AnalyzeRequest {
            command_name: AnalysisCommandName::Freq,
            target_uri: "file:///tmp/x.cha".to_owned(),
            options: AnalysisOptionsRequest {
                capitalization: Some("initial".to_owned()),
                ..AnalysisOptionsRequest::default()
            },
        };
        match build_analysis_options(&request).expect("options should build") {
            AnalysisOptions::Freq(o) => {
                assert_eq!(o.capitalization, CapitalizationFilter::InitialUpper);
            }
            other => panic!("expected Freq variant, got {other:?}"),
        }
    }

    /// VOCD `capitalization: "mid"` → `MidUpper`. Sibling of the
    /// FREQ test — same enum mapping, different command.
    #[test]
    fn build_analysis_options_routes_vocd_mid_capitalization() {
        use talkbank_clan::framework::CapitalizationFilter;
        let request = AnalyzeRequest {
            command_name: AnalysisCommandName::Vocd,
            target_uri: "file:///tmp/x.cha".to_owned(),
            options: AnalysisOptionsRequest {
                capitalization: Some("mid".to_owned()),
                ..AnalysisOptionsRequest::default()
            },
        };
        match build_analysis_options(&request).expect("options should build") {
            AnalysisOptions::Vocd(o) => {
                assert_eq!(o.capitalization, CapitalizationFilter::MidUpper);
            }
            other => panic!("expected Vocd variant, got {other:?}"),
        }
    }

    /// An unrecognized `capitalization` string degrades to `Any`
    /// rather than failing the request. The wire mapping
    /// deliberately tolerates unknown values so adding a third
    /// CapitalizationFilter variant later doesn't break clients.
    #[test]
    fn build_analysis_options_unknown_capitalization_falls_back_to_any() {
        use talkbank_clan::framework::CapitalizationFilter;
        let request = AnalyzeRequest {
            command_name: AnalysisCommandName::Freq,
            target_uri: "file:///tmp/x.cha".to_owned(),
            options: AnalysisOptionsRequest {
                capitalization: Some("bogus".to_owned()),
                ..AnalysisOptionsRequest::default()
            },
        };
        match build_analysis_options(&request).expect("options should build") {
            AnalysisOptions::Freq(o) => {
                assert_eq!(o.capitalization, CapitalizationFilter::Any);
            }
            other => panic!("expected Freq variant, got {other:?}"),
        }
    }

    /// COMBO `firstMatchOnly` and `dedupeMatches` thread from the
    /// LSP request to `ComboOptions::{first_match_only,
    /// dedupe_matches}`.
    #[test]
    fn build_analysis_options_routes_combo_first_match_and_dedupe() {
        let request = AnalyzeRequest {
            command_name: AnalysisCommandName::Combo,
            target_uri: "file:///tmp/x.cha".to_owned(),
            options: AnalysisOptionsRequest {
                first_match_only: true,
                dedupe_matches: true,
                ..AnalysisOptionsRequest::default()
            },
        };
        match build_analysis_options(&request).expect("options should build") {
            AnalysisOptions::Combo(o) => {
                assert!(o.first_match_only);
                assert!(o.dedupe_matches);
            }
            other => panic!("expected Combo variant, got {other:?}"),
        }
    }
}
