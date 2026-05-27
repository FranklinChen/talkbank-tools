//! Focused contract tests for the preserved legacy `chatter clan` surface.

use std::fs;
use std::process::Output;

use talkbank_parser_tests::test_error::TestError;
use tempfile::tempdir;

#[allow(dead_code, unused_imports, unused_macros)]
#[path = "../../talkbank-clan/tests/clan_golden/harness.rs"]
mod clan_harness;
#[path = "common/mod.rs"]
mod cli_common;
#[allow(dead_code)]
#[path = "../../talkbank-clan/tests/common/mod.rs"]
mod common;

use cli_common::{CliHarness, parse_json, stderr_string, stdout_string};

fn assert_exit_code(output: &Output, expected: i32, context: &str) {
    assert_eq!(
        output.status.code(),
        Some(expected),
        "{context}: stdout=`{}` stderr=`{}`",
        stdout_string(output),
        stderr_string(output)
    );
}

fn corpus_file(name: &str) -> String {
    common::corpus_file(name).to_string_lossy().into_owned()
}

#[test]
fn legacy_freq_rewrite_matches_modern_filters() -> Result<(), TestError> {
    let harness = CliHarness::new()?;
    let file = corpus_file("tiers/mor-gra.cha");

    let legacy = harness.run_output(&[
        "clan",
        "freq",
        "+t*CHI",
        "+scookie",
        "+z1-1",
        "--format",
        "json",
        file.as_str(),
    ])?;
    let modern = harness.run_output(&[
        "clan",
        "freq",
        "--speaker",
        "CHI",
        "--include-word",
        "cookie",
        "--range",
        "1-1",
        "--format",
        "json",
        file.as_str(),
    ])?;

    assert_exit_code(&legacy, 0, "legacy freq rewrite should succeed");
    assert_exit_code(&modern, 0, "modern freq filters should succeed");
    assert_eq!(parse_json(&legacy)?, parse_json(&modern)?);
    Ok(())
}

#[test]
fn legacy_check_g2_rewrite_matches_modern_flag() -> Result<(), TestError> {
    let harness = CliHarness::new()?;
    let file = corpus_file("core/basic-conversation.cha");

    let legacy = harness.run_output(&["clan", "check", "+g2", file.as_str()])?;
    let modern = harness.run_output(&["clan", "check", "--check-target", file.as_str()])?;

    assert_eq!(legacy.status.code(), modern.status.code());
    assert_eq!(stdout_string(&legacy), stdout_string(&modern));
    assert_eq!(stderr_string(&legacy), stderr_string(&modern));
    assert!(
        stdout_string(&legacy).contains("PARTICIPANTS TIER IS MISSING \"CHI Target_Child\".(68)"),
        "expected stable CHECK target-child diagnostic, got `{}`",
        stdout_string(&legacy)
    );
    Ok(())
}

#[test]
fn gemfreq_matches_freq_with_gem_filter() -> Result<(), TestError> {
    let harness = CliHarness::new()?;
    let file = corpus_file("core/headers-episodes.cha");

    let gemfreq = harness.run_output(&[
        "clan",
        "gemfreq",
        "--gem",
        "afternoon play",
        "--format",
        "json",
        file.as_str(),
    ])?;
    let freq = harness.run_output(&[
        "clan",
        "freq",
        "--gem",
        "afternoon play",
        "--format",
        "json",
        file.as_str(),
    ])?;

    assert_exit_code(&gemfreq, 0, "gemfreq should parse and dispatch");
    assert_exit_code(&freq, 0, "freq --gem should succeed");
    let gemfreq_json = parse_json(&gemfreq)?;
    assert_eq!(gemfreq_json, parse_json(&freq)?);
    Ok(())
}

#[test]
fn fixit_matches_normalize_output() -> Result<(), TestError> {
    let harness = CliHarness::new()?;
    let file = corpus_file("core/basic-conversation.cha");

    let fixit = harness.run_output(&["clan", "fixit", file.as_str()])?;
    let normalize = harness.run_output(&["normalize", file.as_str()])?;

    assert_exit_code(&fixit, 0, "fixit should succeed");
    assert_exit_code(&normalize, 0, "normalize should succeed");
    assert_eq!(stdout_string(&fixit), stdout_string(&normalize));
    Ok(())
}

#[test]
fn check_clean_file_keeps_stable_success_message() -> Result<(), TestError> {
    let harness = CliHarness::new()?;
    let file = corpus_file("core/basic-conversation.cha");

    let output = harness.run_output(&["clan", "check", file.as_str()])?;

    assert_exit_code(&output, 0, "clean CHECK should succeed");
    assert_eq!(stdout_string(&output), "");
    assert_eq!(stderr_string(&output), "ALL FILES CHECKED OUT OK!\n");
    Ok(())
}

#[test]
fn check_invalid_file_exits_one_with_warning() -> Result<(), TestError> {
    let harness = CliHarness::new()?;
    let dir = tempdir()?;
    let file = dir.path().join("missing-end.cha");
    fs::write(
        &file,
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n@ID:\teng|corpus|CHI|||||Child|||\n*CHI:\thello .\n",
    )?;
    let file_str = file.to_string_lossy().into_owned();

    let output = harness.run_output(&["clan", "check", file_str.as_str()])?;

    assert_exit_code(&output, 1, "invalid CHECK should fail");
    assert!(
        stdout_string(&output).contains("Missing @End header at end of file(7)"),
        "expected missing @End diagnostic, got `{}`",
        stdout_string(&output)
    );
    assert!(
        stderr_string(&output)
            .contains("Please repeat CHECK until no error messages are reported!"),
        "expected stable CHECK warning, got `{}`",
        stderr_string(&output)
    );
    Ok(())
}

#[test]
fn check_list_errors_succeeds_without_path() -> Result<(), TestError> {
    let harness = CliHarness::new()?;

    let output = harness.run_output(&["clan", "check", "--list-errors"])?;

    assert_exit_code(&output, 0, "check --list-errors should not require a path");
    assert!(
        stdout_string(&output).contains("1: Expected characters are: @ or % or *."),
        "expected CHECK error listing header, got `{}`",
        stdout_string(&output)
    );
    assert!(
        stdout_string(&output).contains("7: \"@End\" is missing at the end of the file."),
        "expected stable CHECK error listing entry, got `{}`",
        stdout_string(&output)
    );
    assert_eq!(stderr_string(&output), "");
    Ok(())
}

#[test]
fn check_directory_processes_all_cha_files() -> Result<(), TestError> {
    let harness = CliHarness::new()?;
    let dir = tempdir()?;

    // Two valid files in a directory
    let valid = "\u{FEFF}@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|2;00.||||Target_Child|||\n*CHI:\thello .\n@End\n";
    fs::write(dir.path().join("a.cha"), valid)?;
    fs::write(dir.path().join("b.cha"), valid)?;

    let dir_str = dir.path().to_string_lossy().into_owned();
    let output = harness.run_output(&["clan", "check", dir_str.as_str()])?;

    assert_exit_code(&output, 0, "directory of valid files should succeed");
    assert_eq!(stdout_string(&output), "");
    assert_eq!(stderr_string(&output), "ALL FILES CHECKED OUT OK!\n");
    Ok(())
}

#[test]
fn check_directory_reports_errors_across_files() -> Result<(), TestError> {
    let harness = CliHarness::new()?;
    let dir = tempdir()?;

    let valid = "\u{FEFF}@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|2;00.||||Target_Child|||\n*CHI:\thello .\n@End\n";
    let invalid = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n@ID:\teng|corpus|CHI|||||Child|||\n*CHI:\thello .\n";
    fs::write(dir.path().join("good.cha"), valid)?;
    fs::write(dir.path().join("bad.cha"), invalid)?;

    let dir_str = dir.path().to_string_lossy().into_owned();
    let output = harness.run_output(&["clan", "check", dir_str.as_str()])?;

    assert_exit_code(&output, 1, "directory with broken file should fail");
    assert!(
        stdout_string(&output).contains("Missing @End header at end of file"),
        "expected @End error in output, got `{}`",
        stdout_string(&output)
    );
    assert!(
        stderr_string(&output).contains("Please repeat CHECK"),
        "expected repeat warning, got `{}`",
        stderr_string(&output)
    );
    Ok(())
}

#[test]
fn check_multiple_file_args() -> Result<(), TestError> {
    let harness = CliHarness::new()?;
    let file = corpus_file("core/basic-conversation.cha");
    let file2 = corpus_file("tiers/mor-gra.cha");

    let output = harness.run_output(&["clan", "check", file.as_str(), file2.as_str()])?;

    assert_exit_code(&output, 0, "multiple valid files should succeed");
    assert_eq!(stderr_string(&output), "ALL FILES CHECKED OUT OK!\n");
    Ok(())
}

#[test]
fn check_plus_u_maps_to_check_ud_in_check_context() -> Result<(), TestError> {
    let harness = CliHarness::new()?;
    let file = corpus_file("tiers/mor-gra.cha");

    let legacy = harness.run_output(&["clan", "check", "+u", file.as_str()])?;
    let modern = harness.run_output(&["clan", "check", "--check-ud", file.as_str()])?;

    assert_eq!(legacy.status.code(), modern.status.code());
    assert_eq!(stdout_string(&legacy), stdout_string(&modern));
    assert_eq!(stderr_string(&legacy), stderr_string(&modern));
    Ok(())
}

#[test]
fn mor_placeholder_stays_deliberately_unimplemented() -> Result<(), TestError> {
    let harness = CliHarness::new()?;

    let output = harness.run_output(&["clan", "mor"])?;

    assert_exit_code(&output, 1, "mor should remain an explicit placeholder");
    assert_eq!(stdout_string(&output), "");
    assert!(
        stderr_string(&output).contains("MOR is deliberately not implemented"),
        "expected explicit placeholder message, got `{}`",
        stderr_string(&output)
    );
    assert!(
        stderr_string(&output).contains("Use batchalign"),
        "expected batchalign redirect, got `{}`",
        stderr_string(&output)
    );
    Ok(())
}

/// CHAT2SRT `+v` flips the output format from SRT to WebVTT
/// (CLAN: `OSX-CLAN/src/clan/chat2srt.cpp:108` `case 'v'`). chatter
/// splits this into two distinct subcommands: `chat2srt` (SRT) and
/// `chat2vtt` (WebVTT), each with its own clap surface. The
/// rewriter needs to detect `chat2srt +v` and rewrite the
/// subcommand token to `chat2vtt`, dropping `+v` — the first
/// "subcommand alias" rewrite in this codebase (prior rewriters
/// only modified single tokens, not the subcommand selection).
#[test]
fn legacy_chat2srt_v_switches_to_chat2vtt() -> Result<(), TestError> {
    let harness = CliHarness::new()?;
    let file = corpus_file("core/basic-conversation.cha");

    let output = harness.run_output(&["clan", "chat2srt", "+v", file.as_str()])?;

    assert_exit_code(
        &output,
        0,
        "chat2srt +v should run successfully (switched to chat2vtt)",
    );
    let stdout = stdout_string(&output);
    assert!(
        stdout.starts_with("WEBVTT"),
        "expected WEBVTT output (chat2vtt format), got: {stdout}"
    );
    Ok(())
}

/// CHAT2ELAN `+eEXT` is the media-extension flag (per
/// `OSX-CLAN/src/clan/chat2elan.cpp:117`, `case 'e'`). chatter has
/// `--media-extension <EXT>` on the chat2elan clap surface (see
/// `clan_commands.rs`). Without a per-chat2elan rewriter arm,
/// `+eEXT` falls through to the generic `(b'+', b'e') =>
/// rewrite_check_error(rest)` arm which routes it to
/// `--error EXT` — a `check`-family flag chat2elan doesn't accept.
#[test]
fn legacy_chat2elan_e_routes_to_media_extension() -> Result<(), TestError> {
    let harness = CliHarness::new()?;
    let file = corpus_file("core/basic-conversation.cha");

    let output = harness.run_output(&["clan", "chat2elan", "+e.wav", file.as_str()])?;

    assert_exit_code(
        &output,
        0,
        "chat2elan +e.wav should route to --media-extension .wav",
    );
    let stdout = stdout_string(&output);
    assert!(
        stdout.contains("media.wav") || stdout.contains("MEDIA_URL=\"file:///media.wav\""),
        "expected EAF MEDIA_DESCRIPTOR referencing `.wav`, got: {stdout}"
    );
    Ok(())
}

/// FIXBULLETS `-oN` rewrites to `--offset -N` (negative milliseconds).
/// The legacy unit test on the rewriter alone (in `clan_args.rs::tests`)
/// passes because it only asserts the rewriter's `Vec<String>` output,
/// but the subprocess seam fails: clap interprets a free-standing
/// `-3` as a short-flag attempt, not as the value of `--offset`. The
/// rewriter must use the `--offset=-N` (`=` syntax) form to
/// disambiguate. This test pins the actual user-visible behavior.
#[test]
fn legacy_fixbullets_negative_offset_runs_via_subprocess() -> Result<(), TestError> {
    let harness = CliHarness::new()?;
    let file = corpus_file("core/basic-conversation.cha");

    let output = harness.run_output(&["clan", "fixbullets", "-o3", file.as_str()])?;

    assert_exit_code(
        &output,
        0,
        "fixbullets -o3 should run successfully (negative offset)",
    );
    Ok(())
}

#[test]
fn clan_output_format_matches_legacy_freq_when_available() -> Result<(), TestError> {
    if !common::require_clan_command("freq", "skipping legacy CLAN compatibility contract") {
        return Ok(());
    }

    let harness = CliHarness::new()?;
    let file = common::corpus_file("tiers/mor-gra.cha");
    let file_str = file.to_string_lossy().into_owned();

    let chatter = harness.run_output(&[
        "clan",
        "freq",
        "+t*CHI",
        "--format",
        "clan",
        file_str.as_str(),
    ])?;

    assert_exit_code(&chatter, 0, "clan-format freq should succeed");

    let legacy = clan_harness::run_clan("freq", file.as_path(), &["+t*CHI"])
        .expect("legacy CLAN freq should run when CLAN_BIN_DIR is configured");

    assert_eq!(stdout_string(&chatter).trim(), legacy.trim());
    Ok(())
}

/// RED test for the FREQ `+sWORD` per-word filter parity gap.
///
/// CLAN's `+s "the"` for FREQ is a per-word filter: count only matching
/// words. On `corpus/reference/core/basic-conversation.cha`, the CHI
/// block must show ONLY "the" (1 type, 1 token); the MOT block must
/// appear with 0 types/0 tokens (MOT used no "the").
///
/// Chatter currently applies `--include-word` at the utterance level
/// via `FilterConfig.words` (see
/// `crates/talkbank-clan/src/framework/filter.rs:320-336`), which both
/// over-counts (CHI utterance contains `the` so the WHOLE utterance is
/// counted: "chocolate", "ones", "the") and drops the MOT block
/// entirely (MOT utterance has no "the" so it's excluded by the
/// utterance gate).
///
/// Verified empirically and source-grounded 2026-05-27:
/// `docs/investigations/2026-05-27-freq-include-word-architectural-finding.md`.
///
/// Fix tracked under Tier-1 backlog item 1 (WordFilterMode
/// `UtteranceContext` / `PerWordEmit` split):
/// `docs/investigations/2026-05-27-freq-full-implementation-audit.md`.
///
/// This test is the highest-level seam at which the per-word filter
/// semantic surfaces. Per the depth-first-one-CLAN-command-at-a-time
/// rule + top-down TDD discipline, this RED test must be the starting
/// point for the WordFilterMode fix.
#[test]
fn legacy_freq_s_word_filters_per_word_not_per_utterance() -> Result<(), TestError> {
    let harness = CliHarness::new()?;
    let file = corpus_file("core/basic-conversation.cha");

    let output =
        harness.run_output(&["clan", "freq", "+sthe", "--format", "clan", file.as_str()])?;
    assert_exit_code(&output, 0, "freq +sthe should succeed");

    let stdout = stdout_string(&output);

    // CHI's utterance `the chocolate ones !` contains "the".
    // CLAN-faithful semantic: count only "the" (1 type, 1 token).
    // Buggy chatter behavior: counts whole utterance (3 types, 3 tokens).
    assert!(
        stdout.contains("Speaker: *CHI:"),
        "expected CHI speaker block; got:\n{stdout}"
    );
    assert!(
        stdout.contains("  1 the"),
        "expected CHI block to count 'the'; got:\n{stdout}"
    );
    assert!(
        !stdout.contains("  1 chocolate"),
        "+sthe should filter out 'chocolate' (not a search word); got:\n{stdout}"
    );
    assert!(
        !stdout.contains("  1 ones"),
        "+sthe should filter out 'ones' (not a search word); got:\n{stdout}"
    );

    // MOT's utterance `we can make some together .` contains no "the".
    // CLAN-faithful semantic: MOT block appears with 0 types / 0 tokens.
    // Buggy chatter behavior: MOT utterance fails the utterance gate
    // (no matching word) and the MOT block is omitted entirely.
    assert!(
        stdout.contains("Speaker: *MOT:"),
        "expected MOT speaker block even with 0 matches; got:\n{stdout}"
    );

    Ok(())
}
