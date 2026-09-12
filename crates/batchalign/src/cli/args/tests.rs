use super::*;
use crate::api::ReleasedCommand;
use crate::cli::args::InputKind;
use crate::options::{
    AsrEngineName, CommandOptions, FaEngineName, TranslateEngineName, UtrEngine as AppUtrEngine,
};
use crate::types::engines::{EngineBackend, SelectableEngine};
use clap::{CommandFactory, Parser};
use rstest::rstest;
use std::path::{Path, PathBuf};

fn typed_options_for(args: &[&str]) -> CommandOptions {
    let cli = Cli::parse_from(args);
    build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .unwrap()
}

#[test]
fn auto_speakers_roundtrips_and_conflicts_with_explicit_count() {
    let options = typed_options_for(&["batchalign3", "transcribe", "input.wav", "--auto-speakers"]);
    let encoded = serde_json::to_value(&options).expect("wire options");
    let decoded: CommandOptions = serde_json::from_value(encoded).expect("wire roundtrip");
    assert_eq!(decoded, options);
    let CommandOptions::Transcribe(options) = options else { panic!("transcribe options") };
    assert!(options.auto_speakers);
    assert!(Cli::try_parse_from(["batchalign3", "transcribe", "input.wav", "--auto-speakers", "--num-speakers", "3"]).is_err());
    assert!(Cli::try_parse_from(["batchalign3", "transcribe", "input.wav", "--num-speakers", "0"]).is_err());
    let default = typed_options_for(&["batchalign3", "transcribe", "input.wav"]);
    let mut old_wire = serde_json::to_value(default).expect("wire options");
    old_wire.as_object_mut().expect("object").remove("auto_speakers");
    let CommandOptions::Transcribe(old) = serde_json::from_value(old_wire).expect("old requests remain valid") else { panic!("transcribe options") };
    assert!(!old.auto_speakers);
}

fn assert_typed_options_equivalent(lhs: &[&str], rhs: &[&str], note: &str) {
    let lhs_opts = typed_options_for(lhs);
    let rhs_opts = typed_options_for(rhs);
    assert_eq!(
        lhs_opts, rhs_opts,
        "{note}\nleft args: {:?}\nright args: {:?}",
        lhs, rhs
    );
}

fn assert_parse_error_contains(args: &[&str], fragments: &[&str]) {
    let error = Cli::try_parse_from(args)
        .expect_err("CLI parse should fail for conflicting arguments")
        .to_string();
    for fragment in fragments {
        assert!(
            error.contains(fragment),
            "expected parse error to contain {:?}, got `{error}`",
            fragment
        );
    }
}

fn render_subcommand_help(name: &str) -> String {
    let mut cmd = Cli::command();
    let sub = cmd
        .find_subcommand_mut(name)
        .unwrap_or_else(|| panic!("missing subcommand {name}"));
    let mut buf = Vec::new();
    sub.write_long_help(&mut buf).expect("write help");
    String::from_utf8(buf).expect("utf8 help")
}

fn merge_abbrev_value(options: &CommandOptions) -> Option<bool> {
    match options {
        CommandOptions::Align(opts) => Some(opts.merge_abbrev.should_merge()),
        CommandOptions::Transcribe(opts) | CommandOptions::TranscribeS(opts) => {
            Some(opts.merge_abbrev.should_merge())
        }
        CommandOptions::Translate(opts) => Some(opts.merge_abbrev.should_merge()),
        CommandOptions::Morphotag(opts) => Some(opts.merge_abbrev.should_merge()),
        CommandOptions::Coref(opts) => Some(opts.merge_abbrev.should_merge()),
        CommandOptions::Compare(opts) => Some(opts.merge_abbrev.should_merge()),
        CommandOptions::Utseg(opts) => Some(opts.merge_abbrev.should_merge()),
        CommandOptions::Benchmark(opts) => Some(opts.merge_abbrev.should_merge()),
        _ => None,
    }
}

fn wor_value(options: &CommandOptions) -> Option<bool> {
    match options {
        CommandOptions::Align(opts) => Some(opts.wor.should_write()),
        CommandOptions::Transcribe(opts) | CommandOptions::TranscribeS(opts) => {
            Some(opts.wor.should_write())
        }
        CommandOptions::Benchmark(opts) => Some(opts.wor.should_write()),
        _ => None,
    }
}

#[test]
fn parse_morphotag() {
    let cli = Cli::parse_from(["batchalign3", "morphotag", "corpus/"]);
    assert!(matches!(cli.command, Commands::Morphotag(_)));
}

#[test]
fn parse_transcribe_replay_pre_chat_policy_scope() {
    let cli = Cli::parse_from([
        "batchalign3",
        "eval",
        "transcribe-replay",
        "run",
        "recording.replay.json",
        "--output",
        "output",
        "--utseg-passes",
        "policy-on-pre-chat-only",
    ]);
    let Commands::Eval(eval) = cli.command else {
        panic!("expected eval command")
    };
    let EvalAction::TranscribeReplay(replay) = eval.action else {
        panic!("expected transcribe replay")
    };
    let TranscribeReplayAction::Run(run) = replay.action else {
        panic!("expected replay run")
    };
    assert_eq!(
        run.utseg_passes,
        TranscribeReplayUtsegPasses::PolicyOnPreChatOnly
    );
}

#[test]
fn parse_utr_alignment_replay() {
    let cli = Cli::parse_from([
        "batchalign3",
        "eval",
        "utr-alignment",
        "--chat",
        "input.cha",
        "--tokens",
        "tokens.json",
        "--output",
        "report.json",
        "--fuzzy-threshold",
        "0.85",
        "--participation",
        "exclude-marked-overlap",
    ]);
    let Commands::Eval(eval) = cli.command else {
        panic!("expected eval command")
    };
    let EvalAction::UtrAlignment(args) = eval.action else {
        panic!("expected UTR alignment replay")
    };
    assert_eq!(args.chat, PathBuf::from("input.cha"));
    assert_eq!(args.tokens, PathBuf::from("tokens.json"));
    assert_eq!(args.output, PathBuf::from("report.json"));
    assert_eq!(
        args.fuzzy_threshold.map(|threshold| threshold.value()),
        Some(0.85)
    );
    assert_eq!(
        args.participation,
        UtrAlignmentParticipation::ExcludeMarkedOverlap
    );
}

#[test]
fn reject_out_of_domain_utr_fuzzy_threshold_at_cli_boundary() {
    let error = Cli::try_parse_from([
        "batchalign3",
        "eval",
        "utr-alignment",
        "--chat",
        "input.cha",
        "--tokens",
        "tokens.json",
        "--output",
        "report.json",
        "--fuzzy-threshold",
        "1.01",
    ])
    .expect_err("an invalid threshold must never enter the command state");

    assert!(error.to_string().contains("between 0 and 1"));
}

#[test]
fn reject_out_of_domain_utr_overlap_density_at_cli_boundary() {
    let error = Cli::try_parse_from([
        "batchalign3",
        "align",
        "corpus/",
        "--utr-density-threshold",
        "1.01",
    ])
    .expect_err("an invalid overlap density must never enter command state");

    assert!(error.to_string().contains("between 0 and 1"));
}

#[test]
fn morphotag_defaults_l2_on() {
    let options = typed_options_for(&["batchalign3", "morphotag", "corpus/"]);
    match options {
        CommandOptions::Morphotag(opts) => {
            assert!(
                !opts.no_l2_morphotag,
                "default morphotag invocation should keep L2 dispatch on"
            );
        }
        other => panic!("expected Morphotag options, got {other:?}"),
    }
}

#[test]
fn morphotag_defaults_to_honoring_ca_header() {
    let options = typed_options_for(&["batchalign3", "morphotag", "corpus/"]);
    match options {
        CommandOptions::Morphotag(opts) => {
            assert_eq!(opts.ca_policy, crate::options::CaMorphotagPolicy::Honor)
        }
        other => panic!("expected Morphotag options, got {other:?}"),
    }
}

#[test]
fn morphotag_accepts_explicit_ca_analyze_policy() {
    let options = typed_options_for(&[
        "batchalign3",
        "morphotag",
        "corpus/",
        "--ca-policy",
        "analyze",
    ]);
    match options {
        CommandOptions::Morphotag(opts) => {
            assert_eq!(opts.ca_policy, crate::options::CaMorphotagPolicy::Analyze)
        }
        other => panic!("expected Morphotag options, got {other:?}"),
    }
}

#[test]
fn morphotag_no_l2_flag_opts_out() {
    let options = typed_options_for(&["batchalign3", "morphotag", "corpus/", "--no-l2-morphotag"]);
    match options {
        CommandOptions::Morphotag(opts) => {
            assert!(
                opts.no_l2_morphotag,
                "--no-l2-morphotag should disable L2 dispatch"
            );
        }
        other => panic!("expected Morphotag options, got {other:?}"),
    }
}

#[test]
fn morphotag_rejects_removed_positive_l2_flag() {
    assert_parse_error_contains(
        &["batchalign3", "morphotag", "corpus/", "--l2-morphotag"],
        &["unexpected argument '--l2-morphotag'"],
    );
}

#[test]
fn morphotag_default_review_level_is_none() {
    let options = typed_options_for(&["batchalign3", "morphotag", "corpus/"]);
    match options {
        CommandOptions::Morphotag(opts) => assert_eq!(
            opts.review_level,
            crate::chat_ops::fa::ReviewLevel::None,
            "legacy review-level wire default must remain stable"
        ),
        other => panic!("expected Morphotag options, got {other:?}"),
    }
}

#[test]
fn morphotag_review_level_flag_opts_in() {
    // Compatibility parsing remains symmetric with `align --review-level`.
    // The resulting value never authorizes CHAT-tier generation.
    let options = typed_options_for(&[
        "batchalign3",
        "morphotag",
        "corpus/",
        "--review-level",
        "all",
    ]);
    match options {
        CommandOptions::Morphotag(opts) => assert_eq!(
            opts.review_level,
            crate::chat_ops::fa::ReviewLevel::All,
            "--review-level all should retain its legacy wire value"
        ),
        other => panic!("expected Morphotag options, got {other:?}"),
    }
}

#[test]
fn parse_align_with_options() {
    let cli = Cli::parse_from([
        "batchalign3",
        "--verbose",
        "align",
        "input/",
        "-o",
        "output/",
        "--whisper-fa",
        "--pauses",
    ]);
    assert_eq!(cli.global.verbose, 1);
    if let Commands::Align(a) = &cli.command {
        assert!(a.whisper_fa);
        assert!(a.pauses);
        assert_eq!(a.common.output.as_deref(), Some(Path::new("output/")));
    } else {
        panic!("expected Align");
    }
}

#[test]
fn align_rebuild_boundary_flag_reaches_typed_options() {
    let cli = Cli::parse_from([
        "batchalign3",
        "align",
        "input/",
        "--existing-wor-boundaries",
        "rebuild-from-evidence",
    ]);
    let options = build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .expect("typed align options");
    let CommandOptions::Align(options) = options else {
        panic!("expected Align");
    };
    assert_eq!(
        options.boundaries.existing_wor_boundaries,
        crate::chat_ops::fa::ExistingWorBoundaryPolicy::RebuildFromEvidence
    );
}

#[test]
fn align_default_end_overlap_policy_is_the_one_named_constant() {
    // No `--end-overlap-policy` flag: the CLI's `default_value_t` and
    // `AlignBoundaryOptions`'s own `Default` must both read
    // `DEFAULT_END_OVERLAP_POLICY`, never an independent `EndOverlapPolicy`
    // derive, so the two cannot disagree (2026-09-01 review, item 8).
    let cli = Cli::parse_from(["batchalign3", "align", "input/"]);
    let options = build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .expect("typed align options");
    let CommandOptions::Align(options) = options else {
        panic!("expected Align");
    };
    assert_eq!(
        options.boundaries.end_overlap_policy,
        crate::chat_ops::fa::DEFAULT_END_OVERLAP_POLICY
    );
}

#[test]
fn align_cross_speaker_overlap_flag_reaches_typed_options() {
    let cli = Cli::parse_from([
        "batchalign3",
        "align",
        "input/",
        "--end-overlap-policy",
        "preserve-cross-speaker",
    ]);
    let options = build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .expect("typed align options");
    let CommandOptions::Align(options) = options else {
        panic!("expected Align");
    };
    assert_eq!(
        options.boundaries.end_overlap_policy,
        crate::chat_ops::fa::DEFAULT_END_OVERLAP_POLICY
    );
}

#[test]
fn parse_align_with_engine_overrides() {
    let cli = Cli::parse_from([
        "batchalign3",
        "align",
        "input/",
        "--fa-engine-custom",
        "wav2vec_fa_canto",
        "--utr-engine-custom",
        "tencent_utr",
    ]);
    if let Commands::Align(a) = &cli.command {
        // Asserting the resolved SELECTION, not the string the flag held: the
        // flags now parse into the engine enums, so there is no raw name left
        // to compare and no later stage that could fail to resolve one.
        assert_eq!(a.fa_selection(), FaEngineName::Wav2vecCanto);
        assert_eq!(
            a.utr_args.selection.utr_engine_custom,
            Some(AppUtrEngine::HkTencent)
        );
    } else {
        panic!("expected Align");
    }
}

#[test]
fn parse_transcribe_with_lang() {
    let cli = Cli::parse_from([
        "batchalign3",
        "transcribe",
        "audio/",
        "--lang",
        "spa",
        "--num-speakers",
        "3",
        "--whisperx",
    ]);
    if let Commands::Transcribe(a) = &cli.command {
        assert_eq!(a.lang, "spa");
        assert_eq!(a.num_speakers, 3);
        assert!(a.asr.whisperx);
    } else {
        panic!("expected Transcribe");
    }
}

#[test]
fn parse_transcribe_lang_auto() {
    let cli = Cli::parse_from(["batchalign3", "transcribe", "audio/", "--lang", "auto"]);
    if let Commands::Transcribe(a) = &cli.command {
        assert_eq!(a.lang, "auto", "--lang auto must pass through as-is");
    } else {
        panic!("expected Transcribe");
    }
}

#[test]
fn parse_transcribe_asr_engine_override() {
    let cli = Cli::parse_from([
        "batchalign3",
        "transcribe",
        "audio/",
        "--asr-engine-custom",
        "tencent",
    ]);
    if let Commands::Transcribe(a) = &cli.command {
        assert_eq!(
            a.asr.selection().engine(),
            AsrEngineName::HkTencent,
            "the hidden legacy flag still resolves"
        );
    } else {
        panic!("expected Transcribe");
    }
}

// -----------------------------------------------------------------------------
// --utseg-fallback-stanza: operator opt-in to the legacy Stanza
// constituency-parser fallback for utseg when no language-specific
// TalkBank BERT model is configured. Replaces the
// `BA3_UTSEG_FALLBACK_STANZA` env var with a typed CLI surface
// discoverable from `--help`. Exposed on every utseg-invoking
// subcommand: transcribe, transcribe-s, and utseg.
// -----------------------------------------------------------------------------

#[test]
fn parse_transcribe_with_utseg_fallback_stanza() {
    let cli = Cli::parse_from([
        "batchalign3",
        "transcribe",
        "audio/",
        "--lang",
        "spa",
        "--utseg-fallback-stanza",
    ]);
    if let Commands::Transcribe(a) = &cli.command {
        assert!(
            a.utseg_fallback_stanza,
            "--utseg-fallback-stanza must set the flag on TranscribeArgs"
        );
    } else {
        panic!("expected Transcribe");
    }
}

#[test]
fn transcribe_s_typed_options_carry_utseg_fallback_stanza() {
    // The `transcribe_s` shape is produced when `transcribe --diarize`
    // is invoked. The flag must propagate through to the typed
    // `CommandOptions::TranscribeS` variant.
    let opts = typed_options_for(&[
        "batchalign3",
        "transcribe",
        "audio/",
        "--diarize",
        "--utseg-fallback-stanza",
    ]);
    match opts {
        CommandOptions::TranscribeS(o) => assert!(
            o.utseg_fallback.is_allowed(),
            "--utseg-fallback-stanza must reach TranscribeS typed options under --diarize"
        ),
        other => panic!("expected TranscribeS options, got {other:?}"),
    }
}

#[test]
fn parse_utseg_with_utseg_fallback_stanza() {
    let cli = Cli::parse_from([
        "batchalign3",
        "utseg",
        "corpus/",
        "--lang",
        "spa",
        "--utseg-fallback-stanza",
    ]);
    if let Commands::Utseg(a) = &cli.command {
        assert!(
            a.utseg_fallback_stanza,
            "--utseg-fallback-stanza must set the flag on UtsegArgs"
        );
    } else {
        panic!("expected Utseg");
    }
}

#[test]
fn utseg_fallback_default_is_refuse() {
    // Without the flag, every utseg-invoking subcommand must default
    // to refusing the Stanza substitution (BUG-032 invariant).
    let opts = typed_options_for(&["batchalign3", "transcribe", "audio/"]);
    match opts {
        CommandOptions::Transcribe(o) => assert!(
            !o.utseg_fallback.is_allowed(),
            "default Transcribe must refuse Stanza utseg fallback"
        ),
        other => panic!("expected Transcribe options, got {other:?}"),
    }

    let opts = typed_options_for(&["batchalign3", "utseg", "corpus/"]);
    match opts {
        CommandOptions::Utseg(o) => assert!(
            !o.utseg_fallback.is_allowed(),
            "default Utseg must refuse Stanza utseg fallback"
        ),
        other => panic!("expected Utseg options, got {other:?}"),
    }
}

#[test]
fn utseg_fallback_allowed_when_flag_passed() {
    let opts = typed_options_for(&[
        "batchalign3",
        "transcribe",
        "audio/",
        "--utseg-fallback-stanza",
    ]);
    match opts {
        CommandOptions::Transcribe(o) => assert!(
            o.utseg_fallback.is_allowed(),
            "--utseg-fallback-stanza must lift TranscribeOptions.utseg_fallback to AllowStanza"
        ),
        other => panic!("expected Transcribe options, got {other:?}"),
    }
}

#[test]
fn parse_translate_with_file_list_and_output() {
    let cli = Cli::parse_from([
        "batchalign3",
        "translate",
        "--file-list",
        "inputs.txt",
        "-o",
        "output/",
    ]);
    if let Commands::Translate(a) = &cli.command {
        assert!(a.common.paths.is_empty());
        assert_eq!(a.common.file_list.as_deref(), Some(Path::new("inputs.txt")));
        assert_eq!(a.common.output.as_deref(), Some(Path::new("output/")));
    } else {
        panic!("expected Translate");
    }
}

#[test]
fn parse_coref_in_place() {
    let cli = Cli::parse_from(["batchalign3", "coref", "--in-place", "corpus/"]);
    if let Commands::Coref(a) = &cli.command {
        assert!(a.common.in_place);
        assert_eq!(a.common.paths, vec![PathBuf::from("corpus/")]);
    } else {
        panic!("expected Coref");
    }
}

#[test]
fn parse_align_with_before_and_output() {
    let cli = Cli::parse_from([
        "batchalign3",
        "align",
        "corpus/",
        "--before",
        "baseline/",
        "-o",
        "out/",
    ]);
    if let Commands::Align(a) = &cli.command {
        assert_eq!(a.common.paths, vec![PathBuf::from("corpus/")]);
        assert_eq!(
            a.incremental.before.as_deref(),
            Some(Path::new("baseline/"))
        );
        assert_eq!(a.common.output.as_deref(), Some(Path::new("out/")));
    } else {
        panic!("expected Align");
    }
}

#[test]
fn parse_morphotag_with_before() {
    let cli = Cli::parse_from([
        "batchalign3",
        "morphotag",
        "corpus/",
        "--before",
        "baseline/",
    ]);
    if let Commands::Morphotag(a) = &cli.command {
        assert_eq!(
            a.incremental.before.as_deref(),
            Some(Path::new("baseline/"))
        );
    } else {
        panic!("expected Morphotag");
    }
}

#[rstest]
#[case(&["batchalign3", "transcribe", "audio/", "--before", "old/"])]
#[case(&["batchalign3", "translate", "corpus/", "--before", "old/"])]
#[case(&["batchalign3", "coref", "corpus/", "--before", "old/"])]
#[case(&["batchalign3", "compare", "corpus/", "--before", "old/"])]
#[case(&["batchalign3", "utseg", "corpus/", "--before", "old/"])]
#[case(&["batchalign3", "benchmark", "audio/", "--before", "old/"])]
fn unsupported_commands_reject_before_flag(#[case] args: &[&str]) {
    assert_parse_error_contains(args, &["unexpected argument '--before'"]);
}

#[test]
fn help_shows_before_only_on_supported_commands() {
    let align_help = render_subcommand_help("align");
    let morphotag_help = render_subcommand_help("morphotag");
    let transcribe_help = render_subcommand_help("transcribe");
    let translate_help = render_subcommand_help("translate");

    assert!(align_help.contains("--before <PATH>"));
    assert!(morphotag_help.contains("--before <PATH>"));
    assert!(!transcribe_help.contains("--before <PATH>"));
    assert!(!translate_help.contains("--before <PATH>"));
}

#[test]
fn parse_utseg_with_file_list_lang_and_speakers() {
    let cli = Cli::parse_from([
        "batchalign3",
        "utseg",
        "--lang",
        "spa",
        "--num-speakers",
        "3",
        "--file-list",
        "inputs.txt",
    ]);
    if let Commands::Utseg(a) = &cli.command {
        assert_eq!(a.lang, "spa");
        assert_eq!(a.num_speakers, 3);
        assert_eq!(a.common.file_list.as_deref(), Some(Path::new("inputs.txt")));
        assert!(a.common.paths.is_empty());
    } else {
        panic!("expected Utseg");
    }
}

#[test]
fn parse_opensmile_with_bank_and_subdir() {
    let cli = Cli::parse_from([
        "batchalign3",
        "opensmile",
        "input/",
        "output/",
        "--feature-set",
        "ComParE_2016",
        "--lang",
        "spa",
        "--bank",
        "phon-data",
        "--subdir",
        "Eng-NA",
    ]);
    if let Commands::Opensmile(args) = &cli.command {
        assert_eq!(args.input_dir, PathBuf::from("input/"));
        assert_eq!(args.output_dir, PathBuf::from("output/"));
        assert_eq!(args.feature_set, "ComParE_2016");
        assert_eq!(args.lang, "spa");
        assert_eq!(args.bank.as_deref(), Some("phon-data"));
        assert_eq!(args.subdir.as_deref(), Some("Eng-NA"));
    } else {
        panic!("expected Opensmile");
    }
}

#[test]
fn parse_ba2_compat_flags() {
    let cli = Cli::parse_from([
        "batchalign3",
        "align",
        "--rev",
        "--wav2vec",
        "--no-merge-abbrev",
        "input/",
    ]);
    if let Commands::Align(a) = &cli.command {
        assert!(a.utr_args.selection.rev);
        assert!(a.wav2vec);
        assert!(a.no_merge_abbrev);
    } else {
        panic!("expected Align");
    }
}

#[test]
fn parse_serve_start_foreground() {
    let cli = Cli::parse_from([
        "batchalign3",
        "serve",
        "start",
        "--foreground",
        "--test-echo",
        "--port",
        "9000",
    ]);
    if let Commands::Serve(s) = &cli.command {
        if let ServeAction::Start(args) = &s.action {
            assert!(args.foreground);
            assert!(args.test_echo);
            assert_eq!(args.port, Some(9000));
        } else {
            panic!("expected Start");
        }
    } else {
        panic!("expected Serve");
    }
}

#[test]
fn parse_serve_start_python() {
    let cli = Cli::parse_from([
        "batchalign3",
        "serve",
        "start",
        "--python",
        "/custom/python3",
    ]);
    if let Commands::Serve(s) = &cli.command {
        if let ServeAction::Start(args) = &s.action {
            assert_eq!(args.python.as_deref(), Some("/custom/python3"));
            assert!(args.port.is_none());
            assert!(args.host.is_none());
            assert!(!args.test_echo);
        } else {
            panic!("expected Start");
        }
    } else {
        panic!("expected Serve");
    }
}

#[test]
fn parse_jobs_with_id() {
    let cli = Cli::parse_from([
        "batchalign3",
        "jobs",
        "abc-123",
        "--server",
        "http://localhost:8000",
    ]);
    if let Commands::Jobs(j) = &cli.command {
        assert_eq!(j.job_id.as_deref(), Some("abc-123"));
        assert_eq!(j.server.as_deref(), Some("http://localhost:8000"));
    } else {
        panic!("expected Jobs");
    }
}

#[test]
fn parse_server_env() {
    // Simulate BATCHALIGN_SERVER env var via --server
    let cli = Cli::parse_from([
        "batchalign3",
        "--server",
        "http://myhost:8000",
        "morphotag",
        "corpus/",
    ]);
    assert_eq!(cli.global.server.as_deref(), Some("http://myhost:8000"));
}

#[test]
fn parse_open_dashboard() {
    let cli = Cli::parse_from(["batchalign3", "--open-dashboard", "align", "corpus/"]);
    assert!(cli.global.open_dashboard);
    assert!(!cli.global.no_open_dashboard);
}

#[test]
fn parse_no_open_dashboard() {
    let cli = Cli::parse_from(["batchalign3", "--no-open-dashboard", "align", "corpus/"]);
    assert!(cli.global.open_dashboard);
    assert!(cli.global.no_open_dashboard);
}

#[rstest]
#[case(&["batchalign3", "align", "--wor", "--nowor", "corpus/"], &["--wor", "--nowor"])]
#[case(&["batchalign3", "align", "--merge-abbrev", "--no-merge-abbrev", "corpus/"], &["--merge-abbrev", "--no-merge-abbrev"])]
#[case(&["batchalign3", "transcribe", "--wor", "--nowor", "audio/"], &["--wor", "--nowor"])]
#[case(&["batchalign3", "morphotag", "--retokenize", "--keeptokens", "corpus/"], &["--retokenize", "--keeptokens"])]
#[case(&["batchalign3", "morphotag", "--skipmultilang", "--multilang", "corpus/"], &["--skipmultilang", "--multilang"])]
fn conflicting_flags_are_rejected(#[case] args: &[&str], #[case] expected_fragments: &[&str]) {
    assert_parse_error_contains(args, expected_fragments);
}

#[test]
fn parse_hidden_commands() {
    let cli = Cli::parse_from(["batchalign3", "bench", "align", "in/", "out/"]);
    assert!(matches!(cli.command, Commands::Bench(_)));
}

#[test]
fn parse_models_prep() {
    let cli = Cli::parse_from([
        "batchalign3",
        "models",
        "prep",
        "my_run",
        "input/",
        "output/",
        "--min-length",
        "5",
    ]);
    if let Commands::Models(args) = &cli.command {
        if let ModelsAction::Prep(prep) = &args.action {
            assert_eq!(prep.run_name, "my_run");
            assert_eq!(prep.input_dir, PathBuf::from("input/"));
            assert_eq!(prep.output_dir, PathBuf::from("output/"));
            assert_eq!(prep.min_length, 5);
        } else {
            panic!("expected Prep");
        }
    } else {
        panic!("expected Models");
    }
}

#[test]
fn parse_models_train_passthrough() {
    let cli = Cli::parse_from([
        "batchalign3",
        "models",
        "train",
        "utterance",
        "train",
        "--epochs",
        "5",
    ]);
    if let Commands::Models(args) = &cli.command {
        if let ModelsAction::Train(train) = &args.action {
            assert_eq!(
                train.args,
                vec!["utterance", "train", "--epochs", "5"]
                    .into_iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
            );
        } else {
            panic!("expected Train");
        }
    } else {
        panic!("expected Models");
    }
}

#[test]
fn parse_avqi() {
    let cli = Cli::parse_from(["batchalign3", "avqi", "input/", "output/", "--lang", "eng"]);
    if let Commands::Avqi(args) = &cli.command {
        assert_eq!(args.input_dir, PathBuf::from("input/"));
        assert_eq!(args.output_dir, PathBuf::from("output/"));
        assert_eq!(args.lang, "eng");
    } else {
        panic!("expected Avqi");
    }
}

#[test]
fn parse_bench_options() {
    let cli = Cli::parse_from([
        "batchalign3",
        "bench",
        "transcribe_s",
        "input/",
        "output/",
        "--runs",
        "3",
        "--workers",
        "4",
        "--use-cache",
    ]);
    if let Commands::Bench(args) = &cli.command {
        assert_eq!(args.command, BenchTarget::TranscribeS);
        assert_eq!(args.in_dir, PathBuf::from("input/"));
        assert_eq!(args.out_dir, PathBuf::from("output/"));
        assert_eq!(args.runs, 3);
        assert_eq!(args.workers, Some(4));
        assert!(args.use_cache);
    } else {
        panic!("expected Bench");
    }
}

#[test]
fn parse_setup_non_interactive_rev() {
    let cli = Cli::parse_from([
        "batchalign3",
        "setup",
        "--engine",
        "rev",
        "--rev-key",
        "secret",
        "--non-interactive",
    ]);
    if let Commands::Setup(args) = &cli.command {
        assert_eq!(args.engine, Some(SetupEngine::Rev));
        assert_eq!(args.rev_key.as_deref(), Some("secret"));
        assert!(args.non_interactive);
    } else {
        panic!("expected Setup");
    }
}

#[test]
fn parse_logs_options() {
    let cli = Cli::parse_from(["batchalign3", "logs", "--last", "--raw", "--count", "5"]);
    if let Commands::Logs(args) = &cli.command {
        assert!(args.last);
        assert!(args.raw);
        assert_eq!(args.count, 5);
    } else {
        panic!("expected Logs");
    }
}

#[test]
fn parse_cache_stats() {
    let cli = Cli::parse_from(["batchalign3", "cache", "stats"]);
    if let Commands::Cache(args) = &cli.command {
        assert!(matches!(args.action, Some(CacheAction::Stats)));
    } else {
        panic!("expected Cache");
    }
}

#[test]
fn parse_cache_clear_all_yes() {
    let cli = Cli::parse_from(["batchalign3", "cache", "clear", "--all", "-y"]);
    if let Commands::Cache(args) = &cli.command {
        if let Some(CacheAction::Clear(clear)) = &args.action {
            assert!(clear.all);
            assert!(clear.yes);
        } else {
            panic!("expected Clear");
        }
    } else {
        panic!("expected Cache");
    }
}

#[test]
fn parse_cache_legacy_stats() {
    let cli = Cli::parse_from(["batchalign3", "cache", "--stats"]);
    if let Commands::Cache(args) = &cli.command {
        assert!(args.action.is_none());
        assert!(args.stats);
        assert!(!args.clear);
    } else {
        panic!("expected Cache");
    }
}

#[test]
fn parse_cache_legacy_clear_all_yes() {
    let cli = Cli::parse_from(["batchalign3", "cache", "--clear", "--all", "-y"]);
    if let Commands::Cache(args) = &cli.command {
        assert!(args.action.is_none());
        assert!(args.clear);
        assert!(args.all);
        assert!(args.yes);
    } else {
        panic!("expected Cache");
    }
}

#[test]
fn build_options_morphotag() {
    let cli = Cli::parse_from([
        "batchalign3",
        "--override-media-cache",
        "morphotag",
        "--retokenize",
        "corpus/",
    ]);
    let opts = build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .unwrap();
    assert!(opts.common().override_media_cache);
    match opts {
        CommandOptions::Morphotag(m) => assert!(m.retokenize),
        _ => panic!("expected Morphotag"),
    }
}

// -----------------------------------------------------------------------
// build_typed_options: comprehensive coverage
// -----------------------------------------------------------------------

#[test]
fn build_options_align_defaults() {
    let cli = Cli::parse_from(["batchalign3", "align", "corpus/"]);
    let opts = build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .unwrap();
    match opts {
        CommandOptions::Align(a) => {
            // Derived, not restated: this test is about the default being
            // plumbed through, not about which engine it is.
            assert_eq!(a.fa_engine, FaEngineName::DEFAULT);
            assert_eq!(a.utr.engine, Some(AppUtrEngine::RevAi));
            assert!(!a.pauses);
            assert!(a.wor.should_write());
            assert!(!a.merge_abbrev.should_merge());
            assert!(!a.common.override_media_cache);
        }
        _ => panic!("expected Align"),
    }
}

#[test]
fn build_options_align_single_flags() {
    // Each flag's effect on the Align options struct.
    type AlignCase = (
        &'static [&'static str],
        Box<dyn Fn(&crate::options::AlignOptions)>,
    );
    let cases: Vec<AlignCase> = vec![
        (
            &["batchalign3", "align", "--whisper-fa", "corpus/"] as &[&str],
            Box::new(|a| assert_eq!(a.fa_engine, FaEngineName::Whisper)),
        ),
        (
            &[
                "batchalign3",
                "align",
                "--fa-engine-custom",
                "wav2vec_fa_canto",
                "corpus/",
            ],
            Box::new(|a| assert_eq!(a.fa_engine, FaEngineName::Wav2vecCanto)),
        ),
        (
            &[
                "batchalign3",
                "align",
                "--utr-engine-custom",
                "tencent_utr",
                "corpus/",
            ],
            Box::new(|a| assert_eq!(a.utr.engine, Some(AppUtrEngine::HkTencent))),
        ),
        (
            &["batchalign3", "align", "--no-utr", "corpus/"],
            Box::new(|a| assert!(a.utr.engine.is_none())),
        ),
        (
            &["batchalign3", "align", "--nowor", "corpus/"],
            Box::new(|a| assert!(!a.wor.should_write())),
        ),
        (
            &["batchalign3", "align", "--pauses", "corpus/"],
            Box::new(|a| assert!(a.pauses)),
        ),
        (
            &["batchalign3", "align", "--merge-abbrev", "corpus/"],
            Box::new(|a| assert!(a.merge_abbrev.should_merge())),
        ),
    ];
    for (args, check) in cases {
        let opts = typed_options_for(args);
        match opts {
            CommandOptions::Align(a) => check(&a),
            _ => panic!("expected Align for {args:?}"),
        }
    }
}

#[test]
fn legacy_align_aliases_match_canonical_options() {
    assert_typed_options_equivalent(
        &["batchalign3", "align", "--whisper", "corpus/"],
        &["batchalign3", "align", "--utr-engine", "whisper", "corpus/"],
        "hidden --whisper should match canonical --utr-engine whisper",
    );
    assert_typed_options_equivalent(
        &["batchalign3", "align", "--rev", "corpus/"],
        &["batchalign3", "align", "--utr-engine", "rev", "corpus/"],
        "hidden --rev should match canonical --utr-engine rev",
    );
    assert_typed_options_equivalent(
        &["batchalign3", "align", "--whisper-fa", "corpus/"],
        &["batchalign3", "align", "--fa-engine", "whisper", "corpus/"],
        "hidden --whisper-fa should match canonical --fa-engine whisper",
    );
    assert_typed_options_equivalent(
        &["batchalign3", "align", "--wav2vec", "corpus/"],
        &["batchalign3", "align", "--fa-engine", "wav2vec", "corpus/"],
        "hidden --wav2vec should match canonical --fa-engine wav2vec",
    );
}

#[test]
fn legacy_align_alias_precedence_matches_documented_order() {
    assert_typed_options_equivalent(
        &[
            "batchalign3",
            "align",
            "--utr-engine-custom",
            "tencent_utr",
            "--whisper",
            "corpus/",
        ],
        &[
            "batchalign3",
            "align",
            "--utr-engine-custom",
            "tencent_utr",
            "corpus/",
        ],
        "custom UTR engine should override hidden BA2 align aliases",
    );
    assert_typed_options_equivalent(
        &[
            "batchalign3",
            "align",
            "--fa-engine-custom",
            "wav2vec_fa_canto",
            "--whisper-fa",
            "corpus/",
        ],
        &[
            "batchalign3",
            "align",
            "--fa-engine-custom",
            "wav2vec_fa_canto",
            "corpus/",
        ],
        "custom FA engine should override hidden BA2 align aliases",
    );
    assert_typed_options_equivalent(
        &[
            "batchalign3",
            "align",
            "--utr-engine",
            "rev",
            "--whisper",
            "corpus/",
        ],
        &["batchalign3", "align", "--utr-engine", "whisper", "corpus/"],
        "hidden --whisper should override canonical --utr-engine rev when --rev is absent",
    );
    assert_typed_options_equivalent(
        &["batchalign3", "align", "--whisper", "--rev", "corpus/"],
        &["batchalign3", "align", "--utr-engine", "rev", "corpus/"],
        "hidden --rev should cancel the hidden --whisper override path",
    );
}

#[test]
fn build_options_transcribe_defaults() {
    let cli = Cli::parse_from(["batchalign3", "transcribe", "audio/"]);
    let opts = build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .unwrap();
    match opts {
        CommandOptions::Transcribe(t) => {
            assert_eq!(t.asr_engine, AsrEngineName::RevAi);
            assert!(!t.diarize);
            assert!(!t.wor.should_write());
            assert!(!t.merge_abbrev.should_merge());
            assert_eq!(t.batch_size, 8); // default, no longer CLI-configurable
        }
        _ => panic!("expected Transcribe"),
    }
}

#[rstest]
#[case(&["batchalign3", "transcribe", "--whisperx", "audio/"], AsrEngineName::WhisperX)]
#[case(&["batchalign3", "transcribe", "--whisper-oai", "audio/"], AsrEngineName::WhisperOai)]
#[case(&["batchalign3", "transcribe", "--whisper", "audio/"], AsrEngineName::Whisper)]
#[case(&["batchalign3", "transcribe", "--asr-engine-custom", "tencent", "audio/"], AsrEngineName::HkTencent)]
fn build_options_transcribe_asr_engine(#[case] args: &[&str], #[case] expected: AsrEngineName) {
    let opts = typed_options_for(args);
    match opts {
        CommandOptions::Transcribe(t) => assert_eq!(t.asr_engine, expected, "{args:?}"),
        _ => panic!("expected Transcribe for {args:?}"),
    }
}

#[test]
fn build_options_transcribe_diarize() {
    let cli = Cli::parse_from(["batchalign3", "transcribe", "--diarize", "audio/"]);
    let opts = build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .unwrap();
    match opts {
        CommandOptions::TranscribeS(t) => assert!(t.diarize),
        _ => panic!("expected TranscribeS"),
    }
}

#[test]
fn build_options_transcribe_accepts_explicit_speaker_engine() {
    let cli = Cli::parse_from([
        "batchalign3",
        "transcribe",
        "--diarize",
        "--speaker-engine",
        "pyannote",
        "audio/",
    ]);
    let opts = build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .unwrap();
    match opts {
        CommandOptions::TranscribeS(t) => assert_eq!(
            t.common.engine_overrides.speaker,
            Some(crate::options::SpeakerEngineName::Pyannote)
        ),
        _ => panic!("expected TranscribeS"),
    }
}

#[test]
fn build_options_transcribe_diarize_matches_batchalign2_baseline_defaults() {
    let cli = Cli::parse_from(["batchalign3", "transcribe", "--diarize", "audio/"]);
    let opts = build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .unwrap();
    let profile = CommonOpts::command_profile(&cli.command);

    match opts {
        CommandOptions::TranscribeS(t) => {
            assert_eq!(t.asr_engine, AsrEngineName::RevAi);
            assert!(t.diarize);
            assert!(!t.wor.should_write());
            assert!(!t.merge_abbrev.should_merge());
            assert_eq!(t.batch_size, 8); // default, no longer CLI-configurable
        }
        other => panic!("expected TranscribeS, got {other:?}"),
    }

    assert_eq!(profile.command, ReleasedCommand::TranscribeS);
    assert_eq!(profile.lang, "eng");
    assert_eq!(profile.num_speakers, 2);
    // The KIND, not a list. Asserting the list pinned `["mp3","mp4","wav"]`
    // and so locked in a set narrower than what the pipeline can process:
    // `m4a`, `flac`, `ogg`, `aac`, `wma` and `webm` were silently skipped.
    assert_eq!(profile.input_kind, InputKind::Media);
    assert!(
        profile.input_kind.accepts("m4a"),
        "a format the pipeline reads"
    );
    assert!(
        !profile.input_kind.accepts("cha"),
        "a transcript is not media"
    );
}

#[test]
fn legacy_transcribe_aliases_match_canonical_options() {
    assert_typed_options_equivalent(
        &["batchalign3", "transcribe", "--whisper", "audio/"],
        &[
            "batchalign3",
            "transcribe",
            "--asr-engine",
            "whisper",
            "audio/",
        ],
        "hidden --whisper should match canonical --asr-engine whisper",
    );
    assert_typed_options_equivalent(
        &["batchalign3", "transcribe", "--whisperx", "audio/"],
        &[
            "batchalign3",
            "transcribe",
            "--asr-engine",
            "whisperx",
            "audio/",
        ],
        "hidden --whisperx should match canonical --asr-engine whisperx",
    );
    assert_typed_options_equivalent(
        &["batchalign3", "transcribe", "--whisper-oai", "audio/"],
        &[
            "batchalign3",
            "transcribe",
            "--asr-engine",
            "whisper-oai",
            "audio/",
        ],
        "hidden --whisper-oai should match canonical --asr-engine whisper-oai",
    );
    assert_typed_options_equivalent(
        &["batchalign3", "transcribe", "--rev", "audio/"],
        &["batchalign3", "transcribe", "--asr-engine", "rev", "audio/"],
        "hidden --rev should match canonical --asr-engine rev",
    );
    assert_typed_options_equivalent(
        &["batchalign3", "transcribe", "--diarize", "audio/"],
        &[
            "batchalign3",
            "transcribe",
            "--diarization",
            "enabled",
            "audio/",
        ],
        "hidden --diarize should match canonical --diarization enabled",
    );
    assert_typed_options_equivalent(
        &["batchalign3", "transcribe", "--nodiarize", "audio/"],
        &[
            "batchalign3",
            "transcribe",
            "--diarization",
            "disabled",
            "audio/",
        ],
        "hidden --nodiarize should match canonical --diarization disabled",
    );
}

#[test]
fn legacy_transcribe_alias_precedence_matches_documented_order() {
    assert_typed_options_equivalent(
        &[
            "batchalign3",
            "transcribe",
            "--asr-engine-custom",
            "tencent",
            "--whisper",
            "audio/",
        ],
        &[
            "batchalign3",
            "transcribe",
            "--asr-engine-custom",
            "tencent",
            "audio/",
        ],
        "custom ASR engine should override hidden BA2 transcribe aliases",
    );
    assert_typed_options_equivalent(
        &[
            "batchalign3",
            "transcribe",
            "--asr-engine",
            "rev",
            "--whisper",
            "audio/",
        ],
        &[
            "batchalign3",
            "transcribe",
            "--asr-engine",
            "whisper",
            "audio/",
        ],
        "hidden --whisper should override canonical --asr-engine rev",
    );
    assert_typed_options_equivalent(
        &[
            "batchalign3",
            "transcribe",
            "--diarization",
            "disabled",
            "--diarize",
            "audio/",
        ],
        &[
            "batchalign3",
            "transcribe",
            "--diarization",
            "enabled",
            "audio/",
        ],
        "hidden --diarize should override canonical disabled diarization",
    );
    assert_typed_options_equivalent(
        &[
            "batchalign3",
            "transcribe",
            "--diarization",
            "enabled",
            "--nodiarize",
            "audio/",
        ],
        &[
            "batchalign3",
            "transcribe",
            "--diarization",
            "disabled",
            "audio/",
        ],
        "hidden --nodiarize should override canonical enabled diarization",
    );
    assert_typed_options_equivalent(
        &[
            "batchalign3",
            "transcribe",
            "--diarize",
            "--nodiarize",
            "audio/",
        ],
        &[
            "batchalign3",
            "transcribe",
            "--diarization",
            "enabled",
            "audio/",
        ],
        "hidden --diarize should win when both hidden diarization flags are present",
    );
}

#[test]
fn build_options_translate() {
    let cli = Cli::parse_from(["batchalign3", "translate", "corpus/"]);
    let opts = build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .unwrap();
    match opts {
        CommandOptions::Translate(t) => {
            assert!(!t.merge_abbrev.should_merge());
        }
        _ => panic!("expected Translate"),
    }
}

#[test]
fn build_options_merge_abbrev_matrix_for_processing_commands() {
    let cases = [
        (
            vec!["batchalign3", "align", "corpus/"],
            false,
            "align defaults to no merge-abbrev",
        ),
        (
            vec!["batchalign3", "align", "--merge-abbrev", "corpus/"],
            true,
            "align enables merge-abbrev",
        ),
        (
            vec!["batchalign3", "align", "--no-merge-abbrev", "corpus/"],
            false,
            "align honors no-merge-abbrev",
        ),
        (
            vec!["batchalign3", "transcribe", "audio/"],
            false,
            "transcribe defaults to no merge-abbrev",
        ),
        (
            vec!["batchalign3", "transcribe", "--merge-abbrev", "audio/"],
            true,
            "transcribe enables merge-abbrev",
        ),
        (
            vec!["batchalign3", "transcribe", "--no-merge-abbrev", "audio/"],
            false,
            "transcribe honors no-merge-abbrev",
        ),
        (
            vec!["batchalign3", "translate", "corpus/"],
            false,
            "translate defaults to no merge-abbrev",
        ),
        (
            vec!["batchalign3", "translate", "--merge-abbrev", "corpus/"],
            true,
            "translate enables merge-abbrev",
        ),
        (
            vec!["batchalign3", "translate", "--no-merge-abbrev", "corpus/"],
            false,
            "translate honors no-merge-abbrev",
        ),
        (
            vec!["batchalign3", "morphotag", "corpus/"],
            false,
            "morphotag defaults to no merge-abbrev",
        ),
        (
            vec!["batchalign3", "morphotag", "--merge-abbrev", "corpus/"],
            true,
            "morphotag enables merge-abbrev",
        ),
        (
            vec!["batchalign3", "morphotag", "--no-merge-abbrev", "corpus/"],
            false,
            "morphotag honors no-merge-abbrev",
        ),
        (
            vec!["batchalign3", "coref", "corpus/"],
            false,
            "coref defaults to no merge-abbrev",
        ),
        (
            vec!["batchalign3", "coref", "--merge-abbrev", "corpus/"],
            true,
            "coref enables merge-abbrev",
        ),
        (
            vec!["batchalign3", "coref", "--no-merge-abbrev", "corpus/"],
            false,
            "coref honors no-merge-abbrev",
        ),
        (
            vec!["batchalign3", "compare", "corpus/"],
            false,
            "compare defaults to no merge-abbrev",
        ),
        (
            vec!["batchalign3", "compare", "--merge-abbrev", "corpus/"],
            true,
            "compare enables merge-abbrev",
        ),
        (
            vec!["batchalign3", "compare", "--no-merge-abbrev", "corpus/"],
            false,
            "compare honors no-merge-abbrev",
        ),
        (
            vec!["batchalign3", "utseg", "corpus/"],
            false,
            "utseg defaults to no merge-abbrev",
        ),
        (
            vec!["batchalign3", "utseg", "--merge-abbrev", "corpus/"],
            true,
            "utseg enables merge-abbrev",
        ),
        (
            vec!["batchalign3", "utseg", "--no-merge-abbrev", "corpus/"],
            false,
            "utseg honors no-merge-abbrev",
        ),
        (
            vec!["batchalign3", "benchmark", "audio/"],
            false,
            "benchmark defaults to no merge-abbrev",
        ),
        (
            vec!["batchalign3", "benchmark", "--merge-abbrev", "audio/"],
            true,
            "benchmark enables merge-abbrev",
        ),
        (
            vec!["batchalign3", "benchmark", "--no-merge-abbrev", "audio/"],
            false,
            "benchmark honors no-merge-abbrev",
        ),
    ];

    for (args, expected, note) in cases {
        let opts = typed_options_for(&args);
        assert_eq!(merge_abbrev_value(&opts), Some(expected), "{note}");
    }
}

#[test]
fn build_options_wor_matrix_for_processing_commands() {
    let cases = [
        (
            vec!["batchalign3", "align", "corpus/"],
            true,
            "align defaults to writing %wor",
        ),
        (
            vec!["batchalign3", "align", "--nowor", "corpus/"],
            false,
            "align honors --nowor",
        ),
        (
            vec!["batchalign3", "transcribe", "audio/"],
            false,
            "transcribe defaults to omitting %wor",
        ),
        (
            vec!["batchalign3", "transcribe", "--wor", "audio/"],
            true,
            "transcribe honors --wor",
        ),
        (
            vec!["batchalign3", "transcribe", "--nowor", "audio/"],
            false,
            "transcribe honors --nowor",
        ),
        (
            vec!["batchalign3", "benchmark", "audio/"],
            true,
            "benchmark defaults to writing %wor (mirrors align, \
             forced alignment is always run as the comparison anchor)",
        ),
        (
            vec!["batchalign3", "benchmark", "--wor", "audio/"],
            true,
            "benchmark honors --wor",
        ),
        (
            vec!["batchalign3", "benchmark", "--nowor", "audio/"],
            false,
            "benchmark honors --nowor",
        ),
    ];

    for (args, expected, note) in cases {
        let opts = typed_options_for(&args);
        assert_eq!(wor_value(&opts), Some(expected), "{note}");
    }
}

#[test]
fn build_options_coref_merge_abbrev() {
    let cli = Cli::parse_from(["batchalign3", "coref", "--merge-abbrev", "corpus/"]);
    let opts = build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .unwrap();
    match opts {
        CommandOptions::Coref(c) => assert!(c.merge_abbrev.should_merge()),
        _ => panic!("expected Coref"),
    }
}

#[test]
fn build_options_utseg() {
    let cli = Cli::parse_from(["batchalign3", "utseg", "corpus/"]);
    let opts = build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .unwrap();
    match opts {
        CommandOptions::Utseg(u) => {
            assert!(!u.merge_abbrev.should_merge());
        }
        _ => panic!("expected Utseg"),
    }
}

#[test]
fn build_options_benchmark_defaults() {
    let cli = Cli::parse_from(["batchalign3", "benchmark", "audio/"]);
    let opts = build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .unwrap();
    match opts {
        CommandOptions::Benchmark(b) => {
            assert_eq!(b.asr_engine, AsrEngineName::RevAi);
            // Mirrors `align`: see BenchmarkOptions::wor rustdoc.
            assert!(b.wor.should_write());
            assert!(!b.merge_abbrev.should_merge());
        }
        _ => panic!("expected Benchmark"),
    }
}

#[rstest]
#[case(&["batchalign3", "benchmark", "--whisper", "audio/"], AsrEngineName::Whisper)]
#[case(&["batchalign3", "benchmark", "--whisper-oai", "audio/"], AsrEngineName::WhisperOai)]
#[case(&["batchalign3", "benchmark", "--asr-engine-custom", "funaudio", "audio/"], AsrEngineName::HkFunaudio)]
fn build_options_benchmark_asr_engine(#[case] args: &[&str], #[case] expected: AsrEngineName) {
    let opts = typed_options_for(args);
    match opts {
        CommandOptions::Benchmark(b) => assert_eq!(b.asr_engine, expected, "{args:?}"),
        _ => panic!("expected Benchmark for {args:?}"),
    }
}

#[test]
fn legacy_benchmark_aliases_match_canonical_options() {
    assert_typed_options_equivalent(
        &["batchalign3", "benchmark", "--whisper", "audio/"],
        &[
            "batchalign3",
            "benchmark",
            "--asr-engine",
            "whisper",
            "audio/",
        ],
        "hidden benchmark --whisper should match canonical --asr-engine whisper",
    );
    assert_typed_options_equivalent(
        &["batchalign3", "benchmark", "--whisper-oai", "audio/"],
        &[
            "batchalign3",
            "benchmark",
            "--asr-engine",
            "whisper-oai",
            "audio/",
        ],
        "hidden benchmark --whisper-oai should match canonical --asr-engine whisper-oai",
    );
    assert_typed_options_equivalent(
        &["batchalign3", "benchmark", "--rev", "audio/"],
        &["batchalign3", "benchmark", "--asr-engine", "rev", "audio/"],
        "hidden benchmark --rev should match canonical --asr-engine rev",
    );
}

#[test]
fn legacy_benchmark_alias_precedence_matches_documented_order() {
    assert_typed_options_equivalent(
        &[
            "batchalign3",
            "benchmark",
            "--asr-engine-custom",
            "funaudio",
            "--whisper",
            "audio/",
        ],
        &[
            "batchalign3",
            "benchmark",
            "--asr-engine-custom",
            "funaudio",
            "audio/",
        ],
        "custom benchmark ASR engine should override hidden BA2 aliases",
    );
    assert_typed_options_equivalent(
        &[
            "batchalign3",
            "benchmark",
            "--asr-engine",
            "rev",
            "--whisper",
            "audio/",
        ],
        &[
            "batchalign3",
            "benchmark",
            "--asr-engine",
            "whisper",
            "audio/",
        ],
        "hidden benchmark --whisper should override canonical --asr-engine rev",
    );
}

#[test]
fn build_options_opensmile_defaults() {
    let cli = Cli::parse_from(["batchalign3", "opensmile", "in/", "out/"]);
    let opts = build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .unwrap();
    match opts {
        CommandOptions::Opensmile(o) => assert_eq!(o.feature_set, "eGeMAPSv02"),
        _ => panic!("expected Opensmile"),
    }
}

#[test]
fn build_options_opensmile_compare() {
    let cli = Cli::parse_from([
        "batchalign3",
        "opensmile",
        "--feature-set",
        "ComParE_2016",
        "in/",
        "out/",
    ]);
    let opts = build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .unwrap();
    match opts {
        CommandOptions::Opensmile(o) => assert_eq!(o.feature_set, "ComParE_2016"),
        _ => panic!("expected Opensmile"),
    }
}

#[test]
fn build_options_avqi_defaults() {
    let opts = typed_options_for(&["batchalign3", "avqi", "input/", "output/"]);
    match opts {
        CommandOptions::Avqi(a) => {
            assert!(!a.common.override_media_cache);
        }
        _ => panic!("expected Avqi"),
    }
}

#[test]
fn build_options_override_media_cache_global() {
    let cli = Cli::parse_from(["batchalign3", "--override-media-cache", "align", "corpus/"]);
    let opts = build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .unwrap();
    assert!(opts.common().override_media_cache);
}

#[test]
fn build_options_require_media_cache_global() {
    let cli = Cli::parse_from(["batchalign3", "--require-media-cache", "align", "corpus/"]);
    let opts = build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .unwrap();
    assert!(opts.common().require_media_cache);
}

#[test]
fn standalone_diarize_defaults_to_local_pyannote_without_paid_inference() {
    let cli = Cli::parse_from(["batchalign3", "diarize", "audio/"]);
    let opts = build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .unwrap();
    let CommandOptions::Diarize(options) = opts else {
        panic!("expected standalone diarize options");
    };
    assert_eq!(
        options.speaker_engine,
        crate::options::SpeakerEngineName::Pyannote
    );
}

#[test]
fn standalone_diarize_accepts_explicit_pyannote_ai() {
    let cli = Cli::parse_from([
        "batchalign3",
        "diarize",
        "--speaker-engine",
        "pyannote-ai",
        "audio/",
    ]);
    let opts = build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .unwrap();
    let CommandOptions::Diarize(options) = opts else {
        panic!("expected standalone diarize options");
    };
    assert_eq!(
        options.speaker_engine,
        crate::options::SpeakerEngineName::PyannoteAi
    );
}

#[test]
fn require_media_cache_conflicts_with_every_refresh_form() {
    assert_parse_error_contains(
        &[
            "batchalign3",
            "--require-media-cache",
            "--override-media-cache",
            "align",
            "corpus/",
        ],
        &["--require-media-cache", "--override-media-cache"],
    );
    assert_parse_error_contains(
        &[
            "batchalign3",
            "--require-media-cache",
            "--override-media-cache-tasks",
            "forced_alignment",
            "align",
            "corpus/",
        ],
        &["--require-media-cache", "--override-media-cache-tasks"],
    );
}

// -----------------------------------------------------------------------
// command_profile (parametrized)
// -----------------------------------------------------------------------

#[rstest]
#[case(&["batchalign3", "align", "corpus/"], ReleasedCommand::Align, "eng", 1, InputKind::Chat)]
#[case(&["batchalign3", "transcribe", "audio/"], ReleasedCommand::Transcribe, "eng", 2, InputKind::Media)]
#[case(&["batchalign3", "transcribe", "--diarize", "audio/"], ReleasedCommand::TranscribeS, "eng", 2, InputKind::Media)]
// per-file commands: translate/morphotag/coref. The `lang` field on
// CommandProfile carries the wire string `"per-file"`, which parses to
// `LanguageSpec::PerFile` at submission. No English placeholder ever
// appears for these commands in job records, dashboards, or worker
// pre-warming.
#[case(&["batchalign3", "translate", "corpus/"], ReleasedCommand::Translate, "per-file", 1, InputKind::Chat)]
#[case(&["batchalign3", "morphotag", "corpus/"], ReleasedCommand::Morphotag, "per-file", 1, InputKind::Chat)]
#[case(&["batchalign3", "coref", "corpus/"], ReleasedCommand::Coref, "per-file", 1, InputKind::Chat)]
#[case(&["batchalign3", "compare", "corpus/"], ReleasedCommand::Compare, "eng", 2, InputKind::Chat)]
#[case(&["batchalign3", "compare", "--lang", "spa", "--num-speakers", "3", "corpus/"], ReleasedCommand::Compare, "spa", 3, InputKind::Chat)]
#[case(&["batchalign3", "utseg", "--lang", "spa", "--num-speakers", "3", "corpus/"], ReleasedCommand::Utseg, "spa", 3, InputKind::Chat)]
#[case(&["batchalign3", "benchmark", "audio/"], ReleasedCommand::Benchmark, "eng", 2, InputKind::Media)]
#[case(&["batchalign3", "opensmile", "in/", "out/"], ReleasedCommand::Opensmile, "eng", 1, InputKind::Media)]
#[case(&["batchalign3", "avqi", "in/", "out/", "--lang", "yue"], ReleasedCommand::Avqi, "yue", 1, InputKind::Media)]
fn command_profile_matches_expected(
    #[case] args: &[&str],
    #[case] expected_cmd: ReleasedCommand,
    #[case] expected_lang: &str,
    #[case] expected_speakers: u32,
    #[case] expected_kind: InputKind,
) {
    let cli = Cli::parse_from(args);
    let profile = CommonOpts::command_profile(&cli.command);
    assert_eq!(
        profile.command, expected_cmd,
        "command mismatch for {args:?}"
    );
    assert_eq!(profile.lang, expected_lang, "lang mismatch for {args:?}");
    assert_eq!(
        profile.num_speakers, expected_speakers,
        "num_speakers mismatch for {args:?}"
    );
    assert_eq!(
        profile.input_kind, expected_kind,
        "input kind mismatch for {args:?}"
    );
}

// -----------------------------------------------------------------------
// common_opts (parametrized)
// -----------------------------------------------------------------------

/// BA2 parity: `coref` does NOT accept `--lang`. The command is English-only
/// and per-file routing comes from the file's `@Languages:` header. This
/// regression test catches any reintroduction of `--lang` on coref. See the
/// 2026-05-03 incident (morphotag had the same shape; sentinel `--lang`
/// silently rewrote non-English files).
#[test]
fn coref_rejects_lang_flag_for_ba2_parity() {
    let result = Cli::try_parse_from(["batchalign3", "coref", "corpus/", "--lang", "eng"]);
    assert!(
        result.is_err(),
        "coref must NOT accept --lang (BA2 parity); CLI parse should fail"
    );
}

/// BA2 parity: `translate` does NOT accept `--lang`. Source language for each
/// file comes from that file's `@Languages:` header (BA2's
/// `pipelines/translate/seamless.py:40` reads `doc.langs[0]` per file); the
/// translation TARGET is hardcoded to English (BA2 `seamless.py:41`,
/// `tgt_lang="eng"`). Re-introducing `--lang` here would recreate the
/// 2026-05-03 morphotag failure mode where a job-level sentinel silently
/// overrode per-file routing.
#[test]
fn translate_rejects_lang_flag_for_ba2_parity() {
    let result = Cli::try_parse_from(["batchalign3", "translate", "corpus/", "--lang", "spa"]);
    assert!(
        result.is_err(),
        "translate must NOT accept --lang (BA2 parity); CLI parse should fail"
    );
}

#[rstest]
#[case(&["batchalign3", "align", "x/"])]
#[case(&["batchalign3", "transcribe", "x/"])]
#[case(&["batchalign3", "translate", "x/"])]
#[case(&["batchalign3", "morphotag", "x/"])]
#[case(&["batchalign3", "coref", "x/"])]
#[case(&["batchalign3", "compare", "x/"])]
#[case(&["batchalign3", "utseg", "x/"])]
#[case(&["batchalign3", "benchmark", "x/"])]
fn common_opts_processing_commands_return_some(#[case] args: &[&str]) {
    let cli = Cli::parse_from(args);
    assert!(
        common_opts(&cli.command).is_some(),
        "common_opts should be Some for {args:?}"
    );
}

#[rstest]
#[case(&["batchalign3", "version"])]
#[case(&["batchalign3", "opensmile", "in/", "out/"])]
#[case(&["batchalign3", "avqi", "in/", "out/"])]
fn common_opts_non_processing_commands_return_none(#[case] args: &[&str]) {
    let cli = Cli::parse_from(args);
    assert!(
        common_opts(&cli.command).is_none(),
        "common_opts should be None for {args:?}"
    );
}

// -----------------------------------------------------------------------
// extract_bank / extract_subdir / extract_lexicon (parametrized)
// -----------------------------------------------------------------------

#[rstest]
#[case(&["batchalign3", "benchmark", "--bank", "childes-data", "audio/"], Some("childes-data"))]
#[case(&["batchalign3", "opensmile", "--bank", "mybank", "in/", "out/"], Some("mybank"))]
#[case(&["batchalign3", "morphotag", "corpus/"], None)]
fn extract_bank_matches(#[case] args: &[&str], #[case] expected: Option<&str>) {
    let cli = Cli::parse_from(args);
    assert_eq!(extract_bank(&cli.command), expected, "{args:?}");
}

#[rstest]
#[case(&["batchalign3", "benchmark", "--subdir", "Eng-NA", "audio/"], Some("Eng-NA"))]
#[case(&["batchalign3", "opensmile", "in/", "out/", "--subdir", "Eng-NA"], Some("Eng-NA"))]
#[case(&["batchalign3", "align", "corpus/"], None)]
fn extract_subdir_matches(#[case] args: &[&str], #[case] expected: Option<&str>) {
    let cli = Cli::parse_from(args);
    assert_eq!(extract_subdir(&cli.command), expected, "{args:?}");
}

#[rstest]
#[case(&["batchalign3", "morphotag", "--lexicon", "lex.csv", "corpus/"], Some("lex.csv"))]
#[case(&["batchalign3", "align", "corpus/"], None)]
fn extract_lexicon_matches(#[case] args: &[&str], #[case] expected: Option<&str>) {
    let cli = Cli::parse_from(args);
    assert_eq!(extract_lexicon(&cli.command), expected, "{args:?}");
}

// -----------------------------------------------------------------------
// --engine-overrides global flag
// -----------------------------------------------------------------------

#[test]
fn parse_engine_overrides_global_flag() {
    let cli = Cli::parse_from([
        "batchalign3",
        "--engine-overrides",
        r#"{"asr": "tencent", "fa": "cantonese_fa"}"#,
        "morphotag",
        "corpus/",
    ]);
    assert_eq!(
        cli.global.engine_overrides.as_deref(),
        Some(r#"{"asr": "tencent", "fa": "cantonese_fa"}"#)
    );
}

#[test]
fn parse_engine_overrides_absent() {
    let cli = Cli::parse_from(["batchalign3", "morphotag", "corpus/"]);
    assert!(cli.global.engine_overrides.is_none());
}

#[test]
fn build_options_engine_overrides_populates_common() {
    let cli = Cli::parse_from([
        "batchalign3",
        "--engine-overrides",
        r#"{"asr": "tencent"}"#,
        "morphotag",
        "corpus/",
    ]);
    let opts = build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .unwrap();
    assert_eq!(
        opts.common().engine_overrides.asr,
        Some(AsrEngineName::HkTencent)
    );
}

#[test]
fn build_options_engine_overrides_empty_by_default() {
    let cli = Cli::parse_from(["batchalign3", "align", "corpus/"]);
    let opts = build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .unwrap();
    assert!(opts.common().engine_overrides.is_empty());
}

#[rstest]
#[case("not valid json")]
#[case(r#"{"asr":{"name":"whisper"}}"#)]
fn build_options_engine_overrides_invalid_values_are_rejected(#[case] overrides: &str) {
    assert_parse_error_contains(
        &[
            "batchalign3",
            "--engine-overrides",
            overrides,
            "morphotag",
            "corpus/",
        ],
        &["invalid --engine-overrides JSON"],
    );
}

#[test]
fn build_options_engine_overrides_multiple_commands() {
    // Verify engine overrides flow through to different command variants
    for (args, variant_name) in [
        (
            vec![
                "batchalign3",
                "--engine-overrides",
                r#"{"fa": "whisper_fa"}"#,
                "align",
                "corpus/",
            ],
            "align",
        ),
        (
            vec![
                "batchalign3",
                "--engine-overrides",
                r#"{"asr": "tencent"}"#,
                "transcribe",
                "audio/",
            ],
            "transcribe",
        ),
        (
            vec![
                "batchalign3",
                "--engine-overrides",
                r#"{"asr": "whisper"}"#,
                "translate",
                "corpus/",
            ],
            "translate",
        ),
    ] {
        let cli = Cli::parse_from(args);
        let opts = build_typed_options(&cli.command, &cli.global)
            .expect("enrollments validate")
            .unwrap();
        assert!(
            !opts.common().engine_overrides.is_empty(),
            "engine_overrides should be non-empty for {variant_name}"
        );
    }
}

// CLI parsing retains per-engine extras (for example, `qwen_model` and
// `qwen_device`) in the typed command options. The pool serializes that typed
// value only at the Python worker boundary.

#[test]
fn engine_overrides_accept_qwen_model_and_device_extras() {
    let cli = Cli::parse_from([
        "batchalign3",
        "--engine-overrides",
        r#"{"asr":"qwen","qwen_model":"Qwen/Qwen3-ASR-0.6B","qwen_device":"cuda"}"#,
        "transcribe",
        "audio/",
    ]);
    let opts = build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .unwrap();
    let overrides = &opts.common().engine_overrides;

    assert_eq!(overrides.asr, Some(AsrEngineName::HkQwen));

    assert_eq!(
        overrides.extras.get("qwen_model").map(String::as_str),
        Some("Qwen/Qwen3-ASR-0.6B")
    );
    assert_eq!(
        overrides.extras.get("qwen_device").map(String::as_str),
        Some("cuda")
    );
}

// Regression: 2026-06-11 fleet incident (four consecutive failed align
// jobs on one host). A user-supplied FA engine override (the documented
// `--engine-overrides '{"fa": "wav2vec_canto"}'` form) was re-serialized
// for the worker boundary via the PERSISTENCE wire names ("wav2vec_fa" /
// "whisper_fa" / "cantonese_fa"), which the Python worker's
// `resolve_fa_engine()` rejects. Every worker spawned for capability
// discovery died before its ready signal and the job failed with
// "Failed to bootstrap live worker capabilities". CLI parsing must normalize
// every accepted spelling into the one typed FA selection; the pool owns the
// sole worker-boundary serialization.
#[test]
fn user_fa_override_normalizes_to_typed_selection() {
    for (user_spelling, expected_engine) in [
        ("wav2vec_canto", FaEngineName::Wav2vecCanto),
        ("cantonese_fa", FaEngineName::Wav2vecCanto),
        ("wav2vec_fa_canto", FaEngineName::Wav2vecCanto),
        ("wave2vec", FaEngineName::Wave2Vec),
        ("wav2vec_fa", FaEngineName::Wave2Vec),
        ("whisper_fa", FaEngineName::Whisper),
    ] {
        let overrides_arg = format!(r#"{{"fa": "{user_spelling}"}}"#);
        let cli = Cli::parse_from([
            "batchalign3",
            "--engine-overrides",
            &overrides_arg,
            "align",
            "corpus/",
        ]);
        let opts = build_typed_options(&cli.command, &cli.global)
            .expect("enrollments validate")
            .unwrap();
        assert_eq!(
            opts.common().engine_overrides.fa,
            Some(expected_engine),
            "typed FA selection for user spelling {user_spelling:?}"
        );
    }
}

#[test]
fn engine_overrides_extras_survive_without_asr_field() {
    // Extras alone count as a non-empty override (operator tuning
    // the default-resolved engine without re-selecting it).
    let cli = Cli::parse_from([
        "batchalign3",
        "--engine-overrides",
        r#"{"qwen_model":"Qwen/Qwen3-ASR-0.6B"}"#,
        "transcribe",
        "audio/",
    ]);
    let opts = build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .unwrap();
    let overrides = &opts.common().engine_overrides;

    assert_eq!(overrides.asr, None);
    assert!(!overrides.is_empty(), "extras alone count as non-empty");
    assert!(
        overrides.extras.contains_key("qwen_model"),
        "qwen_model extra was dropped at the typed CLI boundary"
    );
}

// ---- Translate-engine CLI flag ----

#[test]
fn translate_engine_flag_defaults_to_google_when_absent() {
    let cli = Cli::parse_from(["batchalign3", "translate", "corpus/"]);
    let opts = build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .unwrap();
    match opts {
        CommandOptions::Translate(t) => {
            assert_eq!(t.translate_engine, TranslateEngineName::Google);
            assert_eq!(t.effective_translate_engine(), TranslateEngineName::Google);
        }
        other => panic!("expected Translate variant, got: {other:?}"),
    }
}

#[test]
fn translate_engine_flag_seamless_is_parsed() {
    let cli = Cli::parse_from([
        "batchalign3",
        "translate",
        "--translate-engine",
        "seamless",
        "corpus/",
    ]);
    let opts = build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .unwrap();
    match opts {
        CommandOptions::Translate(t) => {
            assert_eq!(t.translate_engine, TranslateEngineName::Seamless);
            assert_eq!(
                t.effective_translate_engine(),
                TranslateEngineName::Seamless,
            );
        }
        other => panic!("expected Translate variant, got: {other:?}"),
    }
}

#[test]
fn translate_engine_flag_nllb_is_parsed() {
    let cli = Cli::parse_from([
        "batchalign3",
        "translate",
        "--translate-engine",
        "nllb",
        "corpus/",
    ]);
    let opts = build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .unwrap();
    match opts {
        CommandOptions::Translate(t) => {
            assert_eq!(t.translate_engine, TranslateEngineName::Nllb);
            assert_eq!(t.effective_translate_engine(), TranslateEngineName::Nllb,);
        }
        other => panic!("expected Translate variant, got: {other:?}"),
    }
}

#[test]
fn translate_engine_flag_tencent_is_parsed() {
    let cli = Cli::parse_from([
        "batchalign3",
        "translate",
        "--translate-engine",
        "tencent",
        "corpus/",
    ]);
    let opts = build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .unwrap();
    match opts {
        CommandOptions::Translate(t) => {
            assert_eq!(t.translate_engine, TranslateEngineName::Tencent);
            assert_eq!(t.effective_translate_engine(), TranslateEngineName::Tencent,);
        }
        other => panic!("expected Translate variant, got: {other:?}"),
    }
}

#[test]
fn translate_engine_flag_aliyun_is_parsed() {
    // Aliyun MT is the cloud translate option that supports Cantonese
    // as a source language. The CLI flag parses to
    // ``TranslateEngineName::Aliyun`` and the typed-options
    // ``effective_translate_engine()`` returns the same, no surprises
    // around precedence in the no-global-override case.
    let cli = Cli::parse_from([
        "batchalign3",
        "translate",
        "--translate-engine",
        "aliyun",
        "corpus/",
    ]);
    let opts = build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .unwrap();
    match opts {
        CommandOptions::Translate(t) => {
            assert_eq!(t.translate_engine, TranslateEngineName::Aliyun);
            assert_eq!(t.effective_translate_engine(), TranslateEngineName::Aliyun,);
        }
        other => panic!("expected Translate variant, got: {other:?}"),
    }
}

#[test]
fn translate_engine_global_override_beats_explicit_flag() {
    // --engine-overrides is the shared cross-command override mechanism;
    // it must take precedence over the per-command --translate-engine
    // flag (mirrors how --engine-overrides beats --asr-engine in
    // transcribe).
    let cli = Cli::parse_from([
        "batchalign3",
        "--engine-overrides",
        r#"{"translate":"seamless"}"#,
        "translate",
        "--translate-engine",
        "google",
        "corpus/",
    ]);
    let opts = build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .unwrap();
    match opts {
        CommandOptions::Translate(t) => {
            assert_eq!(t.translate_engine, TranslateEngineName::Google);
            assert_eq!(
                t.effective_translate_engine(),
                TranslateEngineName::Seamless,
                "shared --engine-overrides must beat the per-command flag",
            );
        }
        other => panic!("expected Translate variant, got: {other:?}"),
    }
}

// -----------------------------------------------------------------------------
// ASR engine selection: one flag, derived possible-values, `paraformer` as a
// first-class name.
//
// These exist because the previous surface was undiscoverable in four separate
// ways, all of them in the CLI itself: `--asr-engine` advertised five of the
// ten engines that exist; the other five were reachable only through a
// differently-named flag whose help gave an "e.g." and never a list;
// `paraformer`, a name people search for, appeared nowhere in the CLI at all;
// and an unrecognised value was swallowed in silence rather than reported.
// -----------------------------------------------------------------------------

/// `paraformer` is selectable by name, and resolves to the funaudio engine
/// carrying the Paraformer checkpoint.
///
/// It is not its own backend: Paraformer is the FunAudio engine loading the
/// `paraformer-zh` checkpoint. The name is what people search for, so the name
/// is what the CLI accepts.
#[test]
fn parse_transcribe_paraformer_resolves_to_funaudio_with_checkpoint() {
    let cli = Cli::parse_from([
        "batchalign3",
        "transcribe",
        "audio/",
        "--lang",
        "zho",
        "--asr-engine",
        "paraformer",
    ]);
    let Commands::Transcribe(a) = &cli.command else {
        panic!("expected Transcribe");
    };
    let selection = a.asr.selection();
    assert_eq!(selection.engine(), AsrEngineName::HkFunaudio);
    assert_eq!(
        selection.implied_overrides(),
        &[("funaudio_model", "paraformer-zh")],
        "the selection must carry the checkpoint that makes it Paraformer"
    );
}

/// Every engine that exists is reachable through the one flag.
///
/// Derived from `AsrEngineName`, so the help cannot drift back to a subset:
/// adding a variant without a name here fails to compile.
#[test]
fn every_asr_engine_is_selectable_by_its_wire_name() {
    for engine in AsrEngineName::ALL {
        let name = engine.wire_name();
        let cli = Cli::parse_from(["batchalign3", "transcribe", "audio/", "--asr-engine", name]);
        let Commands::Transcribe(a) = &cli.command else {
            panic!("expected Transcribe");
        };
        let selection = a.asr.selection();
        assert_eq!(
            &selection.engine(),
            engine,
            "{name} resolved to the wrong engine"
        );
    }
}

/// An unrecognised engine name is REPORTED, never silently swallowed.
///
/// The old path did `AsrEngineName::from_wire_name(engine).ok()?` inside a
/// function returning `Option`, so a typo produced no error and no engine: the
/// user saw the run proceed as if nothing had been asked for.
#[test]
fn an_unknown_asr_engine_name_is_rejected_at_parse_time() {
    let error = Cli::try_parse_from([
        "batchalign3",
        "transcribe",
        "audio/",
        "--asr-engine",
        "paraformr",
    ])
    .expect_err("a misspelled engine must not parse");
    let message = error.to_string();
    assert!(
        message.contains("paraformr"),
        "the error must name the value the user typed: {message}"
    );
}

/// The old flag keeps working, so existing scripts and book examples do not
/// break; it is hidden from help rather than removed.
#[test]
fn the_legacy_custom_engine_flag_still_resolves() {
    let cli = Cli::parse_from([
        "batchalign3",
        "transcribe",
        "audio/",
        "--asr-engine-custom",
        "funaudio",
    ]);
    let Commands::Transcribe(a) = &cli.command else {
        panic!("expected Transcribe");
    };
    let selection = a.asr.selection();
    assert_eq!(selection.engine(), AsrEngineName::HkFunaudio);
}

/// The historical `whisper-oai` spelling keeps working.
///
/// Deriving the flag's values from `AsrEngineName` exposed that this engine had
/// two public names: the clap enum said `whisper-oai`, the wire name is
/// `whisper_oai`. Both resolve, and only the wire name is advertised.
#[test]
fn the_legacy_whisper_oai_spelling_still_resolves() {
    for spelling in ["whisper-oai", "whisper_oai"] {
        let cli = Cli::parse_from([
            "batchalign3",
            "transcribe",
            "audio/",
            "--asr-engine",
            spelling,
        ]);
        let Commands::Transcribe(a) = &cli.command else {
            panic!("expected Transcribe");
        };
        assert_eq!(
            a.asr.selection().engine(),
            AsrEngineName::WhisperOai,
            "{spelling} must resolve"
        );
    }
}

/// An explicit `--engine-overrides` beats what the engine NAME implies.
///
/// `paraformer` implies `funaudio_model=paraformer-zh`, but a user naming a
/// specific checkpoint is being more precise, not less, so their value wins.
#[test]
fn explicit_engine_overrides_beat_the_implied_checkpoint() {
    let cli = Cli::parse_from([
        "batchalign3",
        "--engine-overrides",
        r#"{"funaudio_model":"paraformer-zh-streaming"}"#,
        "transcribe",
        "audio/",
        "--lang",
        "zho",
        "--asr-engine",
        "paraformer",
    ]);
    let options = crate::cli::args::options::build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .expect("options must resolve");
    let crate::options::CommandOptions::Transcribe(t) = options else {
        panic!("expected transcribe options");
    };
    assert_eq!(
        t.common.engine_overrides.extras.get("funaudio_model"),
        Some(&"paraformer-zh-streaming".to_string()),
        "the user's explicit checkpoint must win over the name's default"
    );
}

/// `benchmark` gets the same engine surface as `transcribe`, not a subset.
///
/// It previously advertised three of the ten engines through its own clap enum
/// and carried its own copy of the precedence chain, so the same fix had to be
/// made twice or the second copy kept the defect. Sharing one flattened struct
/// is what makes that impossible rather than merely fixed.
#[test]
fn benchmark_and_transcribe_share_one_engine_surface() {
    for engine in AsrEngineName::ALL {
        let name = engine.wire_name();
        let bench = Cli::parse_from(["batchalign3", "benchmark", "audio/", "--asr-engine", name]);
        let Commands::Benchmark(b) = &bench.command else {
            panic!("expected Benchmark");
        };
        assert_eq!(
            &b.asr.selection().engine(),
            engine,
            "benchmark must accept {name} exactly as transcribe does"
        );
    }

    // Including the name that prompted all of this.
    let bench = Cli::parse_from([
        "batchalign3",
        "benchmark",
        "audio/",
        "--asr-engine",
        "paraformer",
    ]);
    let Commands::Benchmark(b) = &bench.command else {
        panic!("expected Benchmark");
    };
    assert_eq!(b.asr.selection().engine(), AsrEngineName::HkFunaudio);
    assert_eq!(
        b.asr.selection().implied_overrides(),
        &[("funaudio_model", "paraformer-zh")]
    );
}

/// A misspelled engine is rejected on `benchmark` too, not swallowed.
#[test]
fn benchmark_rejects_an_unknown_engine_name() {
    let error = Cli::try_parse_from([
        "batchalign3",
        "benchmark",
        "audio/",
        "--asr-engine-custom",
        "paraformr",
    ])
    .expect_err("a misspelled engine must not parse");
    assert!(
        error.to_string().contains("paraformr"),
        "the error must name the value typed: {error}"
    );
}

// ---------------------------------------------------------------------------
// UTR and FA engine selection.
//
// `--asr-engine` was given one owner and a parse-time rejection. The other two
// engine categories were left as they were, so the same defect survives twice:
// `--utr-engine` and `--fa-engine` each advertise a subset, each have a
// separately named `--*-engine-custom` taking an unvalidated string, and each
// resolve it with `from_wire_name(..).ok()?`, which turns a typo into no engine
// and no message.
//
// A user reported the consequence from the field: prompted by a UTR
// language-support error naming `--utr-engine-custom`, they had to guess which
// engine to pass, and remembered the mechanism backwards afterwards.
// ---------------------------------------------------------------------------

/// Every UTR engine that exists is reachable by name through the visible flag.
///
/// The ASR defect in the large: the flag advertised five of ten engines and the
/// rest lived behind a differently-named flag whose help gave an "e.g.". The
/// Cantonese/Hakka UTR engine is in exactly that position today, and it is the
/// one a Cantonese user needs.
#[test]
fn every_utr_engine_is_reachable_through_the_visible_flag() {
    for engine in AppUtrEngine::ALL {
        let name = engine.selection_name();
        let cli = Cli::try_parse_from([
            "batchalign3",
            "align",
            "input/",
            "--utr",
            "--utr-engine",
            name,
        ])
        .unwrap_or_else(|error| panic!("--utr-engine {name} must parse: {error}"));
        let Commands::Align(a) = &cli.command else {
            panic!("expected Align");
        };
        assert_eq!(
            a.utr_selection(),
            Some(engine.clone()),
            "--utr-engine {name} must select {engine:?}"
        );
    }
}

/// Every FA engine that exists is reachable by name through the visible flag.
#[test]
fn every_fa_engine_is_reachable_through_the_visible_flag() {
    for engine in FaEngineName::ALL {
        let name = engine.selection_name();
        let cli = Cli::try_parse_from(["batchalign3", "align", "input/", "--fa-engine", name])
            .unwrap_or_else(|error| panic!("--fa-engine {name} must parse: {error}"));
        let Commands::Align(a) = &cli.command else {
            panic!("expected Align");
        };
        assert_eq!(
            a.fa_selection(),
            engine.clone(),
            "--fa-engine {name} must select {engine:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// The remaining two engine seams: translate, and the UTR override key.
//
// `--asr-engine`, `--utr-engine` and `--fa-engine` now derive their values from
// the enums. Translate was the fourth category and kept the shape the other
// three shed: a CLI-private `TranslateEngine` mirrored onto the domain
// `TranslateEngineName` by a hand-written match that is exhaustive over the CLI
// copy, so a domain variant with no CLI twin compiles and is unreachable.
//
// Separately, `EngineOverrides` types `asr` / `fa` / `translate` and has no
// `utr`, so `--engine-overrides '{"utr": ...}'` was accepted, forwarded to the
// Python worker as an opaque extra, and ignored by everything.
// ---------------------------------------------------------------------------

/// Every translate engine is reachable by name through the visible flag.
#[test]
fn every_translate_engine_is_reachable_through_the_visible_flag() {
    for engine in TranslateEngineName::ALL {
        let name = engine.selection_name();
        let cli = Cli::try_parse_from([
            "batchalign3",
            "translate",
            "input/",
            "--translate-engine",
            name,
        ])
        .unwrap_or_else(|error| panic!("--translate-engine {name} must parse: {error}"));
        let Commands::Translate(a) = &cli.command else {
            panic!("expected Translate");
        };
        assert_eq!(
            a.translate_engine,
            engine.clone(),
            "--translate-engine {name} must select {engine:?}"
        );
    }
}

/// `--engine-overrides '{"utr": ...}'` selects the UTR engine.
///
/// It used to be swallowed: `utr` is not one of the typed keys, so it landed in
/// `extras` and was shipped to a Python worker that has no say in UTR engine
/// selection at all. The user saw their chosen engine ignored, silently.
#[test]
fn an_engine_overrides_utr_key_selects_the_utr_engine() {
    let cli = Cli::parse_from([
        "batchalign3",
        "--engine-overrides",
        r#"{"utr": "tencent_utr"}"#,
        "align",
        "input/",
        "--utr",
    ]);
    let options = build_typed_options(&cli.command, &cli.global)
        .expect("enrollments validate")
        .expect("align options");
    let CommandOptions::Align(align) = options else {
        panic!("expected Align");
    };
    assert_eq!(
        align.common.engine_overrides.utr,
        Some(AppUtrEngine::HkTencent),
        "the utr key must be typed, not swallowed into extras"
    );
    assert_eq!(
        align.effective_utr_engine(),
        Some(AppUtrEngine::HkTencent),
        "an explicit override must beat the flag default, as it does for asr/fa"
    );
}

/// A misspelled engine name is rejected while parsing, in every category and
/// through both the visible flag and its hidden legacy alias.
///
/// One table rather than six near-identical bodies. Six is what the file had:
/// each category grew its own copy as it was converted, which is the same
/// per-category duplication the production code just shed.
#[rstest]
#[case(&["batchalign3", "align", "input/", "--utr", "--utr-engine", "tencnt_utr"], "tencnt_utr")]
#[case(&["batchalign3", "align", "input/", "--utr", "--utr-engine-custom", "tencnt_utr"], "tencnt_utr")]
#[case(&["batchalign3", "align", "input/", "--fa-engine", "cantnese"], "cantnese")]
#[case(&["batchalign3", "align", "input/", "--fa-engine-custom", "wav2vec_fa_cant"], "wav2vec_fa_cant")]
#[case(&["batchalign3", "translate", "input/", "--translate-engine", "aliyn"], "aliyn")]
#[case(&["batchalign3", "transcribe", "audio/", "--asr-engine", "paraformr"], "paraformr")]
#[case(&["batchalign3", "benchmark", "audio/", "--asr-engine-custom", "paraformr"], "paraformr")]
fn an_unknown_engine_name_is_rejected_at_parse_time(#[case] argv: &[&str], #[case] typed: &str) {
    let error = Cli::try_parse_from(argv).expect_err("a misspelled engine must not parse");
    assert!(
        error.to_string().contains(typed),
        "the error must name the value the user typed: {error}"
    );
}

// ---------------------------------------------------------------------------
// speaker-identify: the argument surface, and the ordering `build_typed_options`
// relies on.
// ---------------------------------------------------------------------------

/// The enrollment form reaches the typed options as validated values, in
/// recording order, with the threshold the caller stated.
#[test]
fn speaker_identify_enrollments_and_threshold_reach_the_typed_options() {
    let options = typed_options_for(&[
        "batchalign3",
        "speaker-identify",
        "corpus/",
        "--enroll",
        "9000-14000:CHI",
        "--enroll",
        "1500-9000:INV",
        "--threshold",
        "0.62",
        "--tiers",
        "PAR0,CHI",
    ]);
    let CommandOptions::SpeakerIdentify(options) = options else {
        panic!("expected speaker-identify options, got {options:?}");
    };
    let labels: Vec<&str> = options
        .enrollments
        .as_slice()
        .iter()
        .map(|enrollment| enrollment.label().as_str())
        .collect();
    assert_eq!(labels, ["INV", "CHI"], "the set is held in recording order");
    assert_eq!(options.threshold.get(), 0.62);
    assert_eq!(options.tiers, ["PAR0", "CHI"]);
}

/// A threshold is REQUIRED. There is no defensible default, so omitting it is
/// an argument error rather than a silent policy nobody chose.
#[test]
fn speaker_identify_refuses_to_run_without_a_threshold() {
    let parsed = Cli::try_parse_from([
        "batchalign3",
        "speaker-identify",
        "corpus/",
        "--enroll",
        "1500-9000:INV",
    ]);
    assert!(parsed.is_err(), "--threshold must be required");
}

/// At least one enrollment is REQUIRED: with none, there is nothing to
/// identify anyone against.
#[test]
fn speaker_identify_refuses_to_run_without_an_enrollment() {
    let parsed = Cli::try_parse_from([
        "batchalign3",
        "speaker-identify",
        "corpus/",
        "--threshold",
        "0.5",
    ]);
    assert!(parsed.is_err(), "--enroll must be required");
}

/// A malformed span is refused by argument parsing, with the text quoted, so
/// nothing downstream ever holds an unvalidated enrollment.
#[test]
fn speaker_identify_refuses_a_malformed_enrollment_span() {
    for bad in ["1500-9000", "9000-1500:INV", "1500-9000:", "x-9000:INV"] {
        let parsed = Cli::try_parse_from([
            "batchalign3",
            "speaker-identify",
            "corpus/",
            "--enroll",
            bad,
            "--threshold",
            "0.5",
        ]);
        assert!(parsed.is_err(), "{bad:?} must be refused by --enroll");
    }
}

/// A threshold no cosine similarity can reach is refused by the value parser.
#[test]
fn speaker_identify_refuses_an_unreachable_threshold() {
    for bad in ["1.5", "-2", "banana"] {
        let parsed = Cli::try_parse_from([
            "batchalign3",
            "speaker-identify",
            "corpus/",
            "--enroll",
            "1500-9000:INV",
            "--threshold",
            bad,
        ]);
        assert!(parsed.is_err(), "{bad:?} must be refused by --threshold");
    }
}

/// A cross-argument problem reaches the caller as a typed error.
///
/// Clap accepts each `--enroll` on its own, so two labels the same, or two
/// spans claiming the same audio, survive parsing. `build_typed_options` runs
/// the set construction itself and propagates its refusal, so there is no
/// ordering between two callers for anyone to get wrong: the only route to
/// the options runs through the validation.
#[rstest]
#[case::duplicate_label(&["0-5000:INV", "6000-9000:INV"])]
#[case::overlapping_spans(&["0-5000:INV", "4000-9000:CHI"])]
fn speaker_identify_cross_argument_problems_are_refused_before_options_are_built(
    #[case] enrollments: &[&str],
) {
    let mut argv = vec![
        "batchalign3".to_owned(),
        "speaker-identify".to_owned(),
        "corpus/".to_owned(),
        "--threshold".to_owned(),
        "0.5".to_owned(),
    ];
    for enrollment in enrollments {
        argv.push("--enroll".to_owned());
        argv.push((*enrollment).to_owned());
    }
    let cli = Cli::parse_from(&argv);
    let Commands::SpeakerIdentify(args) = &cli.command else {
        panic!("expected the speaker-identify subcommand");
    };

    assert!(
        args.enrollment_set().is_err(),
        "the set constructor must refuse this invocation"
    );
    // The typed error PROPAGATES. This used to assert `is_none()`, which was
    // the defect: a failed validation and a non-processing command were the
    // same value, so a caller could not tell them apart and the check could be
    // skipped by anyone who did not know to run it first.
    assert!(
        build_typed_options(&cli.command, &cli.global).is_err(),
        "build_typed_options must surface the refusal, not return an absent value"
    );
}
