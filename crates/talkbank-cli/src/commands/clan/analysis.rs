//! CLAN analysis-command adapters for the `chatter` CLI.
//!
//! Each match arm here now performs only CLI-facing adaptation: convert parsed
//! clap arguments into a typed
//! [`talkbank_clan::service::AnalysisCommandName`] plus
//! [`talkbank_clan::service::AnalysisOptions`], then delegate defaults,
//! validation, and execution to the library-owned builder and service
//! boundaries. Keep command construction and shared output policy inside
//! `talkbank-clan`; keep CLI argument mapping and terminal UX here.

use crate::cli::{CapitalizationArg, ClanCommands, FreqposPositionArg};
use talkbank_clan::commands::codes::CodesConfig;
use talkbank_clan::commands::corelex::CorelexConfig;
use talkbank_clan::commands::dss::DssConfig;
use talkbank_clan::commands::freqpos::PositionClassification;
use talkbank_clan::commands::ipsyn::IpsynConfig;
use talkbank_clan::commands::keymap::KeymapConfig;
use talkbank_clan::commands::maxwd::MaxwdConfig;
use talkbank_clan::commands::rely::RelyConfig;
use talkbank_clan::commands::trnfix::TrnfixConfig;
use talkbank_clan::framework::CapitalizationFilter;
use talkbank_clan::service_types::{
    AnalysisOptions, ChainsOptions, CodesOptions, ComboOptions, CorelexOptions, DistOptions,
    DssOptions, EvalOptions, FlucalcOptions, FreqOptions, IpsynOptions, KeymapOptions,
    KidevalOptions, KwalOptions, MaxwdOptions, MltOptions, MluOptions, MortableOptions,
    RelyOptions, ScriptOptions, SugarOptions, TrnfixOptions, UniqOptions, VocdOptions,
    WdsizeOptions,
};

use super::helpers::{
    load_search_expr_files_or_exit, run_analysis_and_print, run_paired_analysis_and_print,
};

pub(super) fn dispatch(command: ClanCommands) -> Result<(), ClanCommands> {
    match command {
        ClanCommands::Freq {
            path,
            mor,
            capitalization,
            reverse_concordance,
            word_list_only,
            types_tokens_only,
            common,
            ..
        } => {
            let case_sensitive = common.case_sensitive;
            run_analysis_and_print(
                AnalysisOptions::Freq(FreqOptions {
                    mor,
                    capitalization: capitalization_to_filter(capitalization),
                    reverse_concordance,
                    word_list_only,
                    types_tokens_only,
                    case_sensitive,
                }),
                &path,
                &common,
            );
        }
        ClanCommands::Mlu {
            path,
            words,
            mut exclude_solo_word,
            exclude_solo_word_file,
            common,
            ..
        } => {
            exclude_solo_word.extend(load_search_expr_files_or_exit(&exclude_solo_word_file));
            run_analysis_and_print(
                AnalysisOptions::Mlu(MluOptions {
                    words,
                    solo_word_exclusions: exclude_solo_word,
                }),
                &path,
                &common,
            );
        }
        ClanCommands::Mlt {
            path,
            mut exclude_solo_word,
            exclude_solo_word_file,
            common,
            ..
        } => {
            exclude_solo_word.extend(load_search_expr_files_or_exit(&exclude_solo_word_file));
            run_analysis_and_print(
                AnalysisOptions::Mlt(MltOptions {
                    solo_word_exclusions: exclude_solo_word,
                }),
                &path,
                &common,
            )
        }
        ClanCommands::Wdlen { path, common, .. } => {
            run_analysis_and_print(AnalysisOptions::Wdlen, &path, &common);
        }
        ClanCommands::Wdsize {
            path,
            main_tier,
            length_filter,
            common,
            ..
        } => {
            run_analysis_and_print(
                AnalysisOptions::Wdsize(WdsizeOptions {
                    main_tier,
                    length_filter,
                }),
                &path,
                &common,
            );
        }
        ClanCommands::Maxwd {
            path,
            limit,
            unique_length_only,
            exclude_length,
            common,
            ..
        } => {
            let case_sensitive = common.case_sensitive;
            run_analysis_and_print(
                AnalysisOptions::Maxwd(MaxwdOptions {
                    limit: option_if_not_default(limit, MaxwdConfig::default().limit),
                    unique_length_only,
                    exclude_lengths: exclude_length,
                    case_sensitive,
                }),
                &path,
                &common,
            );
        }
        ClanCommands::Freqpos {
            path,
            position_classification,
            common,
            ..
        } => {
            let pc = match position_classification {
                FreqposPositionArg::Last => PositionClassification::FirstLastOther,
                FreqposPositionArg::Second => PositionClassification::FirstSecondOther,
            };
            let case_sensitive = common.case_sensitive;
            run_analysis_and_print(
                AnalysisOptions::Freqpos(talkbank_clan::service_types::FreqposOptions {
                    position_classification: pc,
                    case_sensitive,
                }),
                &path,
                &common,
            );
        }
        ClanCommands::Timedur { path, common, .. } => {
            run_analysis_and_print(AnalysisOptions::Timedur, &path, &common);
        }
        ClanCommands::Kwal {
            path,
            keyword,
            strict_match,
            legal_chat,
            context_before,
            context_after,
            common,
            ..
        } => {
            let case_sensitive = common.case_sensitive;
            run_analysis_and_print(
                AnalysisOptions::Kwal(KwalOptions {
                    keywords: keyword
                        .into_iter()
                        .map(talkbank_clan::framework::KeywordPattern::from)
                        .collect(),
                    strict_match,
                    case_sensitive,
                    legal_chat,
                    context_before,
                    context_after,
                }),
                &path,
                &common,
            );
        }
        ClanCommands::Gemlist { path, common, .. } => {
            run_analysis_and_print(AnalysisOptions::Gemlist, &path, &common);
        }
        ClanCommands::Combo {
            path,
            mut search,
            mut exclude_search,
            search_file,
            exclude_search_file,
            first_match_only,
            dedupe_matches,
            context_before,
            context_after,
            common,
            ..
        } => {
            search.extend(load_search_expr_files_or_exit(&search_file));
            exclude_search.extend(load_search_expr_files_or_exit(&exclude_search_file));
            let case_sensitive = common.case_sensitive;
            run_analysis_and_print(
                AnalysisOptions::Combo(ComboOptions {
                    search,
                    exclude_search,
                    first_match_only,
                    dedupe_matches,
                    case_sensitive,
                    context_before,
                    context_after,
                }),
                &path,
                &common,
            );
        }
        ClanCommands::Cooccur {
            path,
            no_frequency_counts,
            cluster_size,
            common,
            ..
        } => {
            run_analysis_and_print(
                AnalysisOptions::Cooccur(talkbank_clan::service_types::CooccurOptions {
                    no_frequency_counts,
                    cluster_size,
                }),
                &path,
                &common,
            );
        }
        ClanCommands::Dist {
            path,
            once_per_turn,
            common,
            ..
        } => {
            let case_sensitive = common.case_sensitive;
            run_analysis_and_print(
                AnalysisOptions::Dist(DistOptions {
                    once_per_turn,
                    case_sensitive,
                }),
                &path,
                &common,
            )
        }
        ClanCommands::Chip { path, common, .. } => {
            run_analysis_and_print(AnalysisOptions::Chip, &path, &common);
        }
        ClanCommands::Phonfreq { path, common, .. } => {
            run_analysis_and_print(AnalysisOptions::Phonfreq, &path, &common);
        }
        ClanCommands::Modrep { path, common, .. } => {
            run_analysis_and_print(AnalysisOptions::Modrep, &path, &common);
        }
        ClanCommands::Vocd {
            path,
            capitalization,
            common,
            ..
        } => {
            let case_sensitive = common.case_sensitive;
            run_analysis_and_print(
                AnalysisOptions::Vocd(VocdOptions {
                    capitalization: capitalization_to_filter(capitalization),
                    case_sensitive,
                }),
                &path,
                &common,
            )
        }
        ClanCommands::Uniq {
            path, sort, common, ..
        } => {
            run_analysis_and_print(
                AnalysisOptions::Uniq(UniqOptions {
                    sort_by_frequency: sort,
                }),
                &path,
                &common,
            );
        }
        ClanCommands::Codes {
            path,
            max_depth,
            common,
            ..
        } => {
            run_analysis_and_print(
                AnalysisOptions::Codes(CodesOptions {
                    max_depth: option_if_not_default(max_depth, CodesConfig::default().max_depth),
                }),
                &path,
                &common,
            );
        }
        ClanCommands::Trnfix {
            path,
            tier1,
            tier2,
            common,
            ..
        } => {
            let default = TrnfixConfig::default();
            run_analysis_and_print(
                AnalysisOptions::Trnfix(TrnfixOptions {
                    tier1: option_if_not_default(tier1, default.tier1),
                    tier2: option_if_not_default(tier2, default.tier2),
                }),
                &path,
                &common,
            );
        }
        ClanCommands::Sugar {
            path,
            min_utterances,
            common,
            ..
        } => {
            if common.speaker.is_empty() {
                super::helpers::exit_with_clan_refusal(
                    "Please specify at least one speaker tier code with \"+t\" option on command line.",
                );
            }
            run_analysis_and_print(
                AnalysisOptions::Sugar(SugarOptions { min_utterances }),
                &path,
                &common,
            );
        }
        ClanCommands::Mortable {
            path,
            script,
            common,
            ..
        } => {
            let script = script.unwrap_or_else(|| {
                super::helpers::exit_with_clan_refusal(
                    "Please specify language script file name with \"+l\" option.\n\
                     For example, \"mortable +leng\" or \"mortable +leng.cut\".",
                )
            });
            run_analysis_and_print(
                AnalysisOptions::Mortable(MortableOptions {
                    script_path: Some(script),
                }),
                &path,
                &common,
            );
        }
        ClanCommands::Chains {
            path, tier, common, ..
        } => {
            let tier = tier.unwrap_or_else(|| {
                super::helpers::exit_with_clan_refusal(
                    "Please specify a code tier with \"+t\" option.",
                )
            });
            run_analysis_and_print(
                AnalysisOptions::Chains(ChainsOptions { tier: Some(tier) }),
                &path,
                &common,
            );
        }
        ClanCommands::Complexity { path, common, .. } => {
            run_analysis_and_print(AnalysisOptions::Complexity, &path, &common);
        }
        ClanCommands::Corelex {
            path,
            threshold,
            common,
            ..
        } => {
            run_analysis_and_print(
                AnalysisOptions::Corelex(CorelexOptions {
                    threshold: option_if_not_default(
                        threshold,
                        CorelexConfig::default().min_frequency,
                    ),
                }),
                &path,
                &common,
            );
        }
        ClanCommands::Keymap {
            path,
            keyword,
            tier,
            common,
            ..
        } => {
            run_analysis_and_print(
                AnalysisOptions::Keymap(KeymapOptions {
                    keywords: keyword
                        .into_iter()
                        .map(talkbank_clan::framework::KeywordPattern::from)
                        .collect(),
                    tier: option_if_not_default(tier, KeymapConfig::default().tier),
                }),
                &path,
                &common,
            );
        }
        ClanCommands::Script {
            path,
            template,
            common,
            ..
        } => {
            run_analysis_and_print(
                AnalysisOptions::Script(ScriptOptions {
                    template_path: Some(template),
                }),
                &path,
                &common,
            );
        }
        ClanCommands::Rely {
            file1,
            file2,
            tier,
            format,
        } => {
            run_paired_analysis_and_print(
                AnalysisOptions::Rely(RelyOptions {
                    second_file: Some(file2),
                    tier: option_if_not_default(tier, RelyConfig::default().tier),
                }),
                &file1,
                format,
            );
        }
        ClanCommands::Flucalc { path, common, .. } => {
            run_analysis_and_print(
                AnalysisOptions::Flucalc(FlucalcOptions::default()),
                &path,
                &common,
            );
        }
        ClanCommands::Dss {
            path,
            rules,
            max_utterances,
            common,
            ..
        } => {
            if common.speaker.is_empty() {
                super::helpers::exit_with_clan_refusal(
                    "Please specify at least one speaker tier name with \"+t\" option.",
                );
            }
            run_analysis_and_print(
                AnalysisOptions::Dss(DssOptions {
                    rules_path: rules,
                    max_utterances: option_if_not_default(
                        max_utterances,
                        DssConfig::default().max_utterances,
                    ),
                }),
                &path,
                &common,
            );
        }
        ClanCommands::Ipsyn {
            path,
            rules,
            max_utterances,
            common,
            ..
        } => {
            if rules.is_none() {
                super::helpers::exit_with_clan_refusal(
                    "Please specify ipsyn rules file name with \"+l\" option.\n\
                     For example, \"ipsyn +leng\" or \"ipsyn +leng.cut\".",
                );
            }
            run_analysis_and_print(
                AnalysisOptions::Ipsyn(IpsynOptions {
                    rules_path: rules,
                    max_utterances: option_if_not_default(
                        max_utterances,
                        IpsynConfig::default().max_utterances,
                    ),
                }),
                &path,
                &common,
            );
        }
        ClanCommands::Eval { path, common, .. } => {
            if common.speaker.is_empty() {
                super::helpers::exit_with_clan_refusal(
                    "Please specify at least one speaker tier code with \"+t\" option on command line.",
                );
            }
            run_analysis_and_print(
                AnalysisOptions::Eval(EvalOptions::default()),
                &path,
                &common,
            );
        }
        ClanCommands::Kideval {
            path,
            dss_rules,
            ipsyn_rules,
            common,
            ..
        } => {
            // CLAN's kideval refuses without `+l<script>` (the
            // language file that bundles DSS + IPSYN + EVAL rules
            // for one language). chatter has separate --dss-rules
            // and --ipsyn-rules flags; require at least one so the
            // refusal triggers when neither is given. Match CLAN's
            // exact two-line wording.
            if dss_rules.is_none() && ipsyn_rules.is_none() {
                // CLAN's kideval refusal includes a leading blank
                // line; preserve that quirk for byte-level parity.
                super::helpers::exit_with_clan_refusal(
                    "\nPlease specify language script file name with \"+l\" option.\n\
                     For example, \"kideval +leng\" or \"kideval +leng.cut\".",
                );
            }
            run_analysis_and_print(
                AnalysisOptions::Kideval(KidevalOptions {
                    dss_rules_path: dss_rules,
                    ipsyn_rules_path: ipsyn_rules,
                    ..KidevalOptions::default()
                }),
                &path,
                &common,
            );
        }
        ClanCommands::EvalD { path, common, .. } => {
            if common.speaker.is_empty() {
                super::helpers::exit_with_clan_refusal(
                    "Please specify at least one speaker tier code with \"+t\" option on command line.",
                );
            }
            run_analysis_and_print(
                AnalysisOptions::EvalDialect(EvalOptions::default()),
                &path,
                &common,
            );
        }
        other => return Err(other),
    }
    Ok(())
}

fn option_if_not_default<T: PartialEq>(value: T, default: T) -> Option<T> {
    if value == default { None } else { Some(value) }
}

/// Convert the CLI `--capitalization` argument into the domain
/// `CapitalizationFilter` consumed by FREQ and VOCD. `None`
/// (flag absent) maps to `Any` — no filter.
fn capitalization_to_filter(arg: Option<CapitalizationArg>) -> CapitalizationFilter {
    match arg {
        None => CapitalizationFilter::Any,
        Some(CapitalizationArg::Initial) => CapitalizationFilter::InitialUpper,
        Some(CapitalizationArg::Mid) => CapitalizationFilter::MidUpper,
    }
}
