//! Shared CLI helper functions for CLAN command wrappers.
//!
//! This module owns the outer-wrapper responsibilities that should stay in the
//! CLI layer: reading and writing files, building CLI-owned filter selections,
//! and adapting typed `AnalysisCommandName` plus `AnalysisOptions` values into
//! calls on the library-owned [`talkbank_clan::service::AnalysisRequestBuilder`]
//! and [`talkbank_clan::service::AnalysisService`]. The actual CLAN analysis
//! execution boundary lives in `talkbank-clan`.

use std::path::{Path, PathBuf};

use crate::cli::{ClanOutputFormat, CommonAnalysisArgs};
use talkbank_clan::framework::{
    DiscoveredChatFiles, FilterConfig, GemFilter, GemLabel, LoadWordListError, OutputFormat,
    SpeakerFilter, TransformCommand, WordFilter, WordPattern, format_clan_banner,
    load_word_list_file, run_transform,
};
use talkbank_clan::service::AnalysisService;
use talkbank_clan::service_types::{
    AnalysisCommandName, AnalysisOptions, AnalysisPlan, AnalysisRequest, AnalysisRequestBuilder,
    ClanScopeMode,
};
use talkbank_model::SpeakerCode;

/// Legacy-CLAN version string emitted in the banner.
///
/// CLAN's `VersionNumber()` prints a build date string in
/// `(DD-Mon-YYYY)` shape — the second `(` and `)` come from CLAN's
/// `printf` template. chatter's banner template adds the parens too,
/// so the constant here is the bare version content with no
/// surrounding parens.
///
/// The value is injected by [`build.rs`](../../../../build.rs) using
/// chrono's `%e-%b-%Y` format (e.g. `21-May-2026`). CLAN itself uses
/// a hardcoded string updated by hand at release time; we substitute
/// the chatter build date, which is more honest and still matches the
/// `DD-Mon-YYYY` shape researchers parse out of the banner.
pub(super) const CLAN_BANNER_VERSION: &str = env!("CLAN_BUILD_DATE");

/// Return the CLAN-style timestamp string for the current moment, matching
/// the `ctime()` format CLAN's mainloop uses (e.g.
/// `"Thu May 21 17:47:15 2026"`).
fn clan_timestamp_now() -> String {
    use chrono::Local;
    Local::now().format("%a %b %e %H:%M:%S %Y").to_string()
}

/// Build the CLAN-style line-1 invocation echo from chatter's full argv.
///
/// CLAN's banner line 1 echoes the user's command-line argv verbatim:
///
/// ```text
/// freq +scat path/to/file.cha
/// ```
///
/// chatter is invoked as `chatter clan <command> <args...>`. The CLAN
/// analog drops the `chatter clan` prefix and starts at the CLAN
/// subcommand name (`<command>`). The slice from `clan_pos + 1` onward
/// is what we want — joined by single spaces.
///
/// **Chatter-only flag filtering.** Some chatter flags have no CLAN
/// analog and should not appear in the echo. Today we filter:
/// * `--format <X>` / `--format=<X>` / `-f <X>` / `-f=<X>` — chatter-
///   specific output-format selector. The CLAN banner only emits when
///   format is `clan`, so the flag is noise even when present.
///
/// Other chatter-only flags (`--per-file`, `--output`, `--id-filter`,
/// …) are not filtered yet — they will be addressed per-command in
/// Phase 1.7 of the CLAN parity plan
/// (`scripts/clan-parity/PLAN.md`). For the typical migration use
/// case — researchers pasting CLAN-style `+flag` arguments into
/// `chatter clan` — no filtering kicks in.
///
/// Pure function: takes argv + clan position; the caller threads in
/// `std::env::args()` and the `clan` index from the dispatcher.
pub(super) fn build_clan_invocation_echo(args: &[String], clan_pos: Option<usize>) -> String {
    let Some(clan_pos) = clan_pos else {
        return String::new();
    };
    let tail = match args.get(clan_pos + 1..) {
        Some(slice) => slice,
        None => return String::new(),
    };

    let mut out: Vec<&str> = Vec::with_capacity(tail.len());
    let mut i = 0;
    while i < tail.len() {
        let arg = tail[i].as_str();
        match arg {
            "--format" | "-f" => {
                // Skip the flag and its value if present.
                i += if i + 1 < tail.len() { 2 } else { 1 };
            }
            _ if arg.starts_with("--format=") || arg.starts_with("-f=") => {
                i += 1;
            }
            _ => {
                out.push(arg);
                i += 1;
            }
        }
    }
    out.join(" ")
}

/// Find the position of the `clan` subcommand in chatter's argv.
///
/// Returns `None` if `clan` does not appear after argv[0]. Used only
/// from the banner-emission path, so by construction the value is
/// always `Some(_)` when reached — but we return `Option` for
/// testability and defensive programming.
fn find_clan_subcommand_position(args: &[String]) -> Option<usize> {
    args.iter()
        .enumerate()
        .skip(1)
        .find_map(|(i, arg)| if arg == "clan" { Some(i) } else { None })
}

/// Capture the runtime argv and build the CLAN-style invocation echo.
///
/// Public wrapper around [`build_clan_invocation_echo`] that handles
/// the `std::env::args` lookup. Pure function lives below for tests.
pub(super) fn clan_invocation_echo() -> String {
    let args: Vec<String> = std::env::args().collect();
    let clan_pos = find_clan_subcommand_position(&args);
    build_clan_invocation_echo(&args, clan_pos)
}

pub(super) fn run_normalize_alias(path: &Path, output: Option<&Path>) {
    let content = read_file_or_exit(path);
    let options = talkbank_model::ParseValidateOptions::default();
    match talkbank_transform::normalize_chat(&content, options) {
        Ok(normalized) => write_output_or_exit(&normalized, output),
        Err(e) => exit_with_error(format!("Error: {e}")),
    }
}

pub(super) fn read_file_or_exit(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| {
        exit_with_error(format!("Error reading {}: {e}", path.display()));
    })
}

pub(super) fn parse_chat_or_exit(path: &Path) -> talkbank_model::ChatFile {
    let content = read_file_or_exit(path);
    talkbank_transform::parse_and_validate(
        &content,
        talkbank_model::ParseValidateOptions::default(),
    )
    .unwrap_or_else(|e| exit_with_error(format!("Error parsing {}: {e}", path.display())))
}

pub(super) fn write_output_or_exit(content: &str, output: Option<&Path>) {
    if let Some(path) = output {
        if let Err(e) = std::fs::write(path, content) {
            exit_with_error(format!("Error writing {}: {e}", path.display()));
        }
    } else {
        print!("{content}");
    }
}

pub(super) fn run_converter(
    result: Result<talkbank_model::ChatFile, talkbank_clan::framework::TransformError>,
    output: Option<&Path>,
) {
    match result {
        Ok(chat) => write_output_or_exit(&chat.to_string(), output),
        Err(e) => exit_with_error(format!("Error: {e}")),
    }
}

pub(super) fn run_analysis_and_print(
    options: AnalysisOptions,
    paths: &[PathBuf],
    common: &CommonAnalysisArgs,
) {
    let command_name = options.command_name();
    let plan = build_analysis_plan_or_exit(options);
    let AnalysisPlan::Service(request) = plan else {
        exit_with_error(format!(
            "Error: {command_name} requires paired-file execution"
        ));
    };

    run_request_and_print(command_name, request, paths, common);
}

pub(super) fn run_paired_analysis_and_print(
    options: AnalysisOptions,
    primary_file: &Path,
    format: ClanOutputFormat,
) {
    let command_name = options.command_name();
    let plan = build_analysis_plan_or_exit(options);
    let AnalysisPlan::Rely(request) = plan else {
        exit_with_error(format!(
            "Error: {command_name} does not support paired-file execution"
        ));
    };

    let service = AnalysisService::new();
    match service.execute_rely_rendered(request, primary_file, convert_format(format)) {
        Ok(result) => print!("{result}"),
        Err(error) => exit_with_error(format!("Error: {error}")),
    }
}

fn build_analysis_plan_or_exit(options: AnalysisOptions) -> AnalysisPlan {
    AnalysisRequestBuilder::new(options)
        .build()
        .unwrap_or_else(|error| exit_with_error(format!("Error: {error}")))
}

fn run_request_and_print(
    command_name: AnalysisCommandName,
    request: AnalysisRequest,
    paths: &[PathBuf],
    common: &CommonAnalysisArgs,
) {
    let discovered_files = DiscoveredChatFiles::from_paths(paths);
    for skipped_path in discovered_files.skipped_paths() {
        eprintln!(
            "Warning: {:?} is not a file or directory, skipping",
            skipped_path
        );
    }

    let files = discovered_files.into_files();
    if files.is_empty() {
        exit_with_error("Error: no .cha files found".to_owned());
    }

    let filter = build_filter(common).unwrap_or_else(|err| {
        exit_with_error(format!("Error: {err}"));
    });
    let service = AnalysisService::with_filter(filter);
    let format = convert_format(common.format);
    let want_clan_banner = matches!(format, OutputFormat::Clan);
    let scope = clan_scope_for(command_name, common);

    // CLAN's banner line 1 echoes the user's argv verbatim
    // (e.g. `freq +scat <path>`); we compute it once per call.
    let invocation = clan_invocation_echo();

    if common.per_file {
        match service.execute_rendered_per_file(request, &files, format) {
            Ok(results) => {
                for (path, result) in results {
                    if want_clan_banner {
                        print!(
                            "{}",
                            format_clan_banner(
                                &invocation,
                                &command_name.to_string(),
                                CLAN_BANNER_VERSION,
                                &scope,
                                &clan_source_for(&path),
                                &clan_timestamp_now(),
                            )
                        );
                    } else {
                        println!("From file: {}", path.display());
                    }
                    print!("{result}");
                    if !want_clan_banner {
                        println!();
                    }
                }
            }
            Err(e) => exit_with_error(format!("Error: {e}")),
        }
    } else {
        match service.execute_rendered(request, &files, format) {
            Ok(result) => {
                if want_clan_banner {
                    // Aggregated mode: CLAN emits the banner once. Use the
                    // first input path as the `From file …` reference.
                    let source = files
                        .first()
                        .map(|p| clan_source_for(p))
                        .unwrap_or_else(|| "From pipe input".to_owned());
                    print!(
                        "{}",
                        format_clan_banner(
                            &invocation,
                            &command_name.to_string(),
                            CLAN_BANNER_VERSION,
                            &scope,
                            &source,
                            &clan_timestamp_now(),
                        )
                    );
                }
                print!("{result}");
            }
            Err(e) => exit_with_error(format!("Error: {e}")),
        }
    }
}

/// Describe the analysis scope the way CLAN's mainloop does. The
/// three banner shapes — main-only, dep-only, and combined — match
/// CLAN's `cutt.cpp` mainloop (near line 12100) branching on `nomain`
/// and `tct`. Per-command selection lives in
/// [`AnalysisCommandName::clan_scope_mode`].
fn clan_scope_for(command_name: AnalysisCommandName, common: &CommonAnalysisArgs) -> String {
    let main_scope = build_main_scope(&common.speaker, &common.exclude_speaker, &common.role);
    match command_name.clan_scope_mode() {
        ClanScopeMode::MainOnly => main_scope,
        ClanScopeMode::DependentOnly(tier) => {
            format!("ONLY dependent tiers matching: %{};", tier.to_uppercase())
        }
        ClanScopeMode::MainAndDependent(tier) => format!(
            "{main_scope}\n\tand those speakers' ONLY dependent tiers matching: %{};",
            tier.to_uppercase()
        ),
    }
}

/// Build the banner's "main-tier scope" sentence, the way CLAN's
/// `cutt.cpp` mainloop renders it.
///
/// CLAN's wording is tightly fixed; this enumeration is exhaustive
/// over what chatter's CLI lets the user express today via
/// `--speaker` / `--exclude-speaker` / `--role`:
///
/// | Includes | Excludes | Roles | Banner sentence |
/// |---------|----------|-------|---------------------------------------------------------------------|
/// | empty   | empty    | empty | `ALL speaker tiers` |
/// | one+    | _any_    | _any_ | `ONLY speaker main tiers matching: *CHI;` (`+t…` wins over `-t…`) |
/// | empty   | one+     | _any_ | `ALL speaker main tiers EXCEPT the ones matching: *MOT;` |
/// | empty   | empty    | one+  | `ONLY speaker main tiers with role(s): TARGET_CHILD;` |
///
/// CLAN precedence: speaker codes outrank role names. When both
/// `+t*CHI` and `+t#Target_Child` are supplied, the banner uses the
/// speaker-code shape — the role only fires when no `+t*` is
/// present. Matches the order of precedence in
/// `clan_args::rewrite_tier_speaker`.
///
/// Multiple values are joined with single spaces and each entry
/// trails its own semicolon (CLAN's per-pattern delimiter). The
/// `*` prefix is the CLAN speaker-tier sigil; chatter's rewriter
/// strips the `*` when it rewrites `+t*CHI` → `--speaker CHI`, so
/// we re-prepend it here. Role names are uppercased.
///
/// The `… with IDs matching: …` shape for `+t@ID="…"` filters is
/// out of scope here — `--id-filter` lives on a separate banner
/// pass that lowercases the pattern and emits an extra `*:;`
/// continuation (Phase 1.6 follow-up).
///
/// Pure function for testability — no I/O, no env lookup, no
/// command-specific branching (the caller's `clan_scope_for`
/// wraps this with the dep-tier suffix).
pub(super) fn build_main_scope(
    includes: &[String],
    excludes: &[String],
    roles: &[String],
) -> String {
    if !includes.is_empty() {
        let body = clan_speaker_pattern_list(includes);
        return format!("ONLY speaker main tiers matching: {body}");
    }
    if !excludes.is_empty() {
        let body = clan_speaker_pattern_list(excludes);
        return format!("ALL speaker main tiers EXCEPT the ones matching: {body}");
    }
    if !roles.is_empty() {
        let body = roles
            .iter()
            .map(|r| format!("{};", r.to_uppercase()))
            .collect::<Vec<_>>()
            .join(" ");
        return format!("ONLY speaker main tiers with role(s): {body}");
    }
    "ALL speaker tiers".to_owned()
}

/// Render a list of bare speaker codes (no `*` prefix, as
/// `clan_args::rewrite_tier_speaker` stores them) into CLAN's
/// `*CHI; *MOT;` banner shape: each code gets a leading `*` and a
/// trailing `;`, joined by single spaces.
fn clan_speaker_pattern_list(codes: &[String]) -> String {
    codes
        .iter()
        .map(|c| format!("*{c};"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Build the `From file <basename>` line that CLAN emits below the
/// `****` separator. CLAN truncates to the path's basename; we follow
/// the same convention so the banner matches.
fn clan_source_for(path: &Path) -> String {
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string());
    format!("From file <{name}>")
}

pub(super) fn run_transform_or_exit<T: TransformCommand>(
    cmd: &T,
    path: &Path,
    output: Option<&Path>,
) {
    if let Err(e) = run_transform(cmd, path, output) {
        exit_with_error(format!("Error: {e}"));
    }
}

/// Build the combined list of `WordPattern`s for one side of
/// the word filter: CLI literals first, then each file's
/// patterns in argv order (lines preserved).
fn collect_word_patterns(
    cli: &[String],
    files: &[PathBuf],
) -> Result<Vec<WordPattern>, LoadWordListError> {
    let mut patterns: Vec<WordPattern> =
        cli.iter().map(|s| WordPattern::from(s.as_str())).collect();
    for path in files {
        patterns.extend(load_word_list_file(path)?);
    }
    Ok(patterns)
}

/// Sibling of [`collect_word_patterns`] for COMBO's
/// `+s@FILE` / `-s@FILE`. Each surviving line is a search-
/// expression string (parsed downstream by `SearchExpr::parse`);
/// returns lines concatenated in argv order. Exits on the first
/// I/O failure because this loader runs at dispatch time, outside
/// the `build_filter` error-bubbling path.
pub(super) fn load_search_expr_files_or_exit(files: &[PathBuf]) -> Vec<String> {
    files
        .iter()
        .flat_map(|file| {
            talkbank_clan::framework::load_search_expr_file(file)
                .unwrap_or_else(|err| exit_with_error(format!("Error: {err}")))
        })
        .collect()
}

pub(super) fn build_filter(common: &CommonAnalysisArgs) -> Result<FilterConfig, LoadWordListError> {
    let speaker_filter = SpeakerFilter {
        include: common.speaker.iter().map(SpeakerCode::new).collect(),
        exclude: common
            .exclude_speaker
            .iter()
            .map(SpeakerCode::new)
            .collect(),
    };

    let gem_filter = GemFilter {
        include: common
            .gem
            .iter()
            .map(|s| GemLabel::from(s.as_str()))
            .collect(),
        exclude: common
            .exclude_gem
            .iter()
            .map(|s| GemLabel::from(s.as_str()))
            .collect(),
    };

    // `--include-word-file` / `--exclude-word-file` load patterns
    // from disk and append to whatever `--include-word` /
    // `--exclude-word` already accumulated. Order: CLI patterns
    // first, then file patterns in `--…-file` argv order, with
    // each file's lines in source order.
    let word_filter = WordFilter {
        include: collect_word_patterns(&common.include_word, &common.include_word_file)?,
        exclude: collect_word_patterns(&common.exclude_word, &common.exclude_word_file)?,
        case_sensitive: common.case_sensitive,
    };

    let role_filter = talkbank_clan::framework::RoleFilter {
        include: common.role.clone(),
    };

    Ok(FilterConfig {
        speakers: speaker_filter,
        gems: gem_filter,
        words: word_filter,
        utterance_range: common.range,
        id_filter: common.id_filter.clone(),
        roles: role_filter,
        ..FilterConfig::default()
    })
}

pub(super) fn convert_format(format: ClanOutputFormat) -> OutputFormat {
    match format {
        ClanOutputFormat::Text => OutputFormat::Text,
        ClanOutputFormat::Json => OutputFormat::Json,
        ClanOutputFormat::Csv => OutputFormat::Csv,
        ClanOutputFormat::Clan => OutputFormat::Clan,
    }
}

pub(super) fn exit_with_error(message: String) -> ! {
    eprintln!("{message}");
    std::process::exit(1);
}

/// Emit a CLAN-style refusal message to stderr and exit non-zero,
/// matching CLAN's behavior when a required flag is missing.
///
/// CLAN's pre-banner refusals (`Please specify a code tier with
/// "+t" option.`, `Please specify ipsyn rules file name with "+l"
/// option.`, …) are deliberately byte-level reproduced — researchers'
/// scripts may grep stderr for these. Pass the exact message CLAN
/// emits; do not paraphrase.
pub(super) fn exit_with_clan_refusal(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(1);
}

#[cfg(test)]
mod tests {
    use super::{CLAN_BANNER_VERSION, build_clan_invocation_echo, find_clan_subcommand_position};

    fn s(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|p| p.to_string()).collect()
    }

    #[test]
    fn invocation_echo_strips_chatter_clan_prefix() {
        let args = s(&["chatter", "clan", "freq", "+scat", "file.cha"]);
        let pos = find_clan_subcommand_position(&args);
        assert_eq!(
            build_clan_invocation_echo(&args, pos),
            "freq +scat file.cha"
        );
    }

    #[test]
    fn invocation_echo_handles_no_flags() {
        let args = s(&["chatter", "clan", "freq", "file.cha"]);
        let pos = find_clan_subcommand_position(&args);
        assert_eq!(build_clan_invocation_echo(&args, pos), "freq file.cha");
    }

    #[test]
    fn invocation_echo_filters_dash_dash_format_with_value() {
        let args = s(&["chatter", "clan", "freq", "--format", "clan", "file.cha"]);
        let pos = find_clan_subcommand_position(&args);
        assert_eq!(build_clan_invocation_echo(&args, pos), "freq file.cha");
    }

    #[test]
    fn invocation_echo_filters_dash_dash_format_equals() {
        let args = s(&["chatter", "clan", "freq", "--format=clan", "file.cha"]);
        let pos = find_clan_subcommand_position(&args);
        assert_eq!(build_clan_invocation_echo(&args, pos), "freq file.cha");
    }

    #[test]
    fn invocation_echo_filters_short_dash_f_with_value() {
        let args = s(&["chatter", "clan", "freq", "-f", "clan", "file.cha"]);
        let pos = find_clan_subcommand_position(&args);
        assert_eq!(build_clan_invocation_echo(&args, pos), "freq file.cha");
    }

    #[test]
    fn invocation_echo_filters_short_dash_f_equals() {
        let args = s(&["chatter", "clan", "freq", "-f=clan", "file.cha"]);
        let pos = find_clan_subcommand_position(&args);
        assert_eq!(build_clan_invocation_echo(&args, pos), "freq file.cha");
    }

    #[test]
    fn invocation_echo_skips_global_flags_before_clan() {
        let args = s(&["chatter", "--verbose", "clan", "freq", "file.cha"]);
        let pos = find_clan_subcommand_position(&args);
        assert_eq!(pos, Some(2));
        assert_eq!(build_clan_invocation_echo(&args, pos), "freq file.cha");
    }

    #[test]
    fn invocation_echo_empty_when_clan_absent() {
        let args = s(&["chatter", "validate", "file.cha"]);
        let pos = find_clan_subcommand_position(&args);
        assert_eq!(pos, None);
        assert_eq!(build_clan_invocation_echo(&args, pos), "");
    }

    #[test]
    fn main_scope_no_filter() {
        assert_eq!(super::build_main_scope(&[], &[], &[]), "ALL speaker tiers");
    }

    #[test]
    fn main_scope_single_include() {
        assert_eq!(
            super::build_main_scope(&["CHI".into()], &[], &[]),
            "ONLY speaker main tiers matching: *CHI;"
        );
    }

    #[test]
    fn main_scope_multi_include() {
        assert_eq!(
            super::build_main_scope(&["CHI".into(), "MOT".into()], &[], &[]),
            "ONLY speaker main tiers matching: *CHI; *MOT;"
        );
    }

    #[test]
    fn main_scope_single_exclude() {
        assert_eq!(
            super::build_main_scope(&[], &["MOT".into()], &[]),
            "ALL speaker main tiers EXCEPT the ones matching: *MOT;"
        );
    }

    #[test]
    fn main_scope_multi_exclude() {
        assert_eq!(
            super::build_main_scope(&[], &["MOT".into(), "FAT".into()], &[]),
            "ALL speaker main tiers EXCEPT the ones matching: *MOT; *FAT;"
        );
    }

    #[test]
    fn main_scope_include_wins_over_exclude() {
        // CLAN observed behaviour: when both +t and -t are present, the
        // banner reports only the include side (exclude still filters
        // output, but the scope line stays silent about it).
        assert_eq!(
            super::build_main_scope(&["CHI".into()], &["MOT".into()], &[]),
            "ONLY speaker main tiers matching: *CHI;"
        );
    }

    #[test]
    fn role_scope_single() {
        assert_eq!(
            super::build_main_scope(&[], &[], &["Target_Child".into()]),
            "ONLY speaker main tiers with role(s): TARGET_CHILD;"
        );
    }

    #[test]
    fn role_scope_multi() {
        assert_eq!(
            super::build_main_scope(&[], &[], &["Target_Child".into(), "Mother".into()]),
            "ONLY speaker main tiers with role(s): TARGET_CHILD; MOTHER;"
        );
    }

    /// CLAN precedence: speaker codes outrank role names. When both
    /// `+t*CHI` and `+t#Target_Child` are supplied, the banner uses
    /// the speaker shape.
    #[test]
    fn role_scope_yields_to_speaker_include() {
        assert_eq!(
            super::build_main_scope(&["CHI".into()], &[], &["Target_Child".into()]),
            "ONLY speaker main tiers matching: *CHI;"
        );
    }

    #[test]
    fn role_scope_empty_falls_back_to_all() {
        assert_eq!(super::build_main_scope(&[], &[], &[]), "ALL speaker tiers");
    }

    #[test]
    fn invocation_echo_preserves_clan_style_speaker_flag() {
        // The typical migration case: researchers paste CLAN-style
        // `+t*CHI` directly; chatter's argv rewriter expands it before
        // clap, but the echo path reads ORIGINAL argv so the CLAN-style
        // flag survives verbatim into the banner.
        let args = s(&["chatter", "clan", "freq", "+t*CHI", "file.cha"]);
        let pos = find_clan_subcommand_position(&args);
        assert_eq!(
            build_clan_invocation_echo(&args, pos),
            "freq +t*CHI file.cha"
        );
    }

    /// Verify the banner version is shaped as CLAN's `(DD-Mon-YYYY)` build
    /// date — `D-Mon-YYYY` or `DD-Mon-YYYY`, with `Mon` being the
    /// abbreviated English month name (`Jan`, `Feb`, …, `Dec`).
    ///
    /// chrono's `%e-%b-%Y` format yields a *space-padded* day for
    /// single-digit days (e.g. ` 1-May-2026`); we trim leading whitespace
    /// before the constant is read, so the test allows 1 or 2 day digits
    /// with no leading whitespace.
    #[test]
    fn banner_version_matches_clan_date_format() {
        const MONTHS: &[&str] = &[
            "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ];

        let parts: Vec<&str> = CLAN_BANNER_VERSION.split('-').collect();
        assert_eq!(
            parts.len(),
            3,
            "CLAN_BANNER_VERSION = {CLAN_BANNER_VERSION:?} \
             should split into [day, mon, year] on '-'"
        );

        let day = parts[0];
        let mon = parts[1];
        let year = parts[2];

        assert!(
            (1..=2).contains(&day.len()) && day.bytes().all(|b| b.is_ascii_digit()),
            "day {day:?} must be 1 or 2 ASCII digits"
        );
        let day_value: u32 = day.parse().expect("day parses");
        assert!(
            (1..=31).contains(&day_value),
            "day {day_value} out of range"
        );

        assert!(
            MONTHS.contains(&mon),
            "month {mon:?} must be one of {MONTHS:?}"
        );

        assert_eq!(year.len(), 4, "year {year:?} must be 4 digits");
        let year_value: i32 = year.parse().expect("year parses");
        // build.rs runs at compile time; this constant is a build date,
        // so accept any plausible build year window. 2026 is when this
        // test lands; future-proof to 2100.
        assert!(
            (2025..=2100).contains(&year_value),
            "year {year_value} out of plausible window"
        );
    }
}
