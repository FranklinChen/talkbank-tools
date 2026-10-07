//! Offline replay of global UTR word-to-token evidence.

use std::path::Path;

use batchalign_transform::parse_source_with_parser;
use serde::Serialize;
use talkbank_model::{
    NullErrorSink,
    model::{FileStem, TranscriptName},
};
use talkbank_parser::TreeSitterParser;

use crate::chat_ops::fa::utr::{
    AsrTimingToken, GlobalUtrParticipation, UtrAlignmentPlan, UtrMatchMode,
    observe_global_utr_alignment,
};
use crate::cli::args::{UtrAlignmentEvalArgs, UtrAlignmentParticipation};
use crate::cli::error::CliError;
use crate::cli::eval_cmd::InputIdentity;

#[derive(Debug, Serialize)]
struct UtrAlignmentReport<'a> {
    schema_version: u8,
    build: &'static str,
    chat: InputIdentity,
    source_name: ReportedSourceName<'a>,
    tokens: InputIdentity,
    match_mode: UtrMatchMode,
    participation: GlobalUtrParticipation,
    plan: UtrAlignmentPlan,
    interleaving: Vec<crate::chat_ops::fa::utr::interleaving::InterleavedUtrPair>,
}

/// Admission identity is distinct from the path and hash of the replay bytes.
/// A caller declaration records a claim, not independently proven provenance.
#[derive(Debug, Serialize)]
struct ReportedSourceName<'a> {
    stem: &'a str,
    basis: SourceNameBasis,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum SourceNameBasis {
    InputPath,
    CallerDeclared,
}

/// A complete UTR report ready for atomic publication.
///
/// Keeping serialization in a state that exists before any destination is
/// opened prevents JSON failures from creating an artifact that looks
/// authoritative but is incomplete.
struct SerializedUtrAlignmentReport(Vec<u8>);

impl SerializedUtrAlignmentReport {
    fn encode(report: &UtrAlignmentReport<'_>) -> Result<Self, CliError> {
        let mut bytes = serde_json::to_vec_pretty(report)?;
        bytes.push(b'\n');
        Ok(Self(bytes))
    }

    /// Atomically publish without replacing an existing evidence artifact.
    fn persist_noclobber(self, output: &Path) -> Result<(), CliError> {
        crate::atomic_file::write_atomically(
            output,
            &self.0,
            crate::atomic_file::Existing::Keep,
            crate::atomic_file::Audience::UmaskDefault,
        )?;
        Ok(())
    }
}

/// Replay one retained CHAT/token pair without inference or CHAT mutation.
pub fn run(args: &UtrAlignmentEvalArgs) -> Result<(), CliError> {
    let source_path = args.source_name.as_deref().unwrap_or(&args.chat);
    let stem = FileStem::from_path(source_path).ok_or_else(|| {
        CliError::InvalidArgument("UTR replay requires a usable UTF-8 transcript filename".into())
    })?;
    let chat_bytes = std::fs::read(&args.chat)?;
    let chat_text = std::str::from_utf8(&chat_bytes)
        .map_err(|error| CliError::InvalidArgument(format!("CHAT input is not UTF-8: {error}")))?;
    let parser = TreeSitterParser::new()?;
    // A replay retains every tier. The named admission producer owns its proof;
    // parseability alone cannot authorize matching or publishing evidence.
    let chat = parse_source_with_parser(&parser, chat_text)
        .admit(TranscriptName::Named(stem), &NullErrorSink)?
        .into_valid_file();
    let token_bytes = std::fs::read(&args.tokens)?;
    let tokens: Vec<AsrTimingToken> = serde_json::from_slice(&token_bytes)?;
    let match_mode = match args.fuzzy_threshold {
        None => UtrMatchMode::Exact,
        Some(threshold) => UtrMatchMode::Fuzzy { threshold },
    };
    let participation = match args.participation {
        UtrAlignmentParticipation::AllUtterances => GlobalUtrParticipation::AllUtterances,
        UtrAlignmentParticipation::ExcludeMarkedOverlap => {
            GlobalUtrParticipation::ExcludeMarkedOverlap
        }
    };
    let report = UtrAlignmentReport {
        // 10: a `matched` utterance's endpoint may be bounded by every
        // token it can match rather than proved by one; its proposal then
        // covers those tokens; a region lists `withdrawn_claims`, tokens
        // contested with a neighbouring region. 9: the plan's `strategy` is the pass's order
        // model; `regions` records each anchored region's span and
        // algorithm; a budget refusal names its region and budget;
        // `retained_unsearched` marks a timed utterance in a refused region.
        schema_version: 10,
        build: crate::build_hash(),
        chat: InputIdentity::of(&args.chat, &chat_bytes),
        source_name: ReportedSourceName {
            stem: stem.as_str(),
            basis: if args.source_name.is_some() {
                SourceNameBasis::CallerDeclared
            } else {
                SourceNameBasis::InputPath
            },
        },
        tokens: InputIdentity::of(&args.tokens, &token_bytes),
        match_mode,
        participation,
        plan: observe_global_utr_alignment(chat.document(), &tokens, match_mode, participation),
        interleaving: crate::chat_ops::fa::utr::interleaving::observe(
            chat.document(),
            &tokens,
            match_mode,
            participation,
        ),
    };
    SerializedUtrAlignmentReport::encode(&report)?.persist_noclobber(&args.output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offline_pair_report_keeps_relaxed_evidence_separate_from_the_production_plan() {
        let dir = tempfile::tempdir().expect("temporary directory");
        let chat = dir.path().join("input.cha");
        let tokens = dir.path().join("tokens.json");
        let output = dir.path().join("report.json");
        let source = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant, INV Investigator\n@ID:\teng|test|PAR|||||Participant|||\n@ID:\teng|test|INV|||||Investigator|||\n*PAR:\tone two three .\n*INV:\tyes .\n@End\n";
        std::fs::write(&chat, source).expect("source");
        std::fs::write(
            &tokens,
            r#"[{"text":"one yes two three","start_ms":100,"end_ms":900}]"#,
        )
        .expect("tokens");
        run(&UtrAlignmentEvalArgs {
            chat: chat.clone(),
            source_name: None,
            tokens,
            output: output.clone(),
            fuzzy_threshold: None,
            participation: UtrAlignmentParticipation::AllUtterances,
        })
        .expect("offline report");
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(output).expect("report")).expect("wire JSON");
        let pair = &report["interleaving"][0];
        assert_eq!(pair["observation"]["status"], "complete");
        assert_eq!(pair["observation"]["matched_words"], 4);
        assert_eq!(
            pair["observation"]["words"][3]["common"]["token"],
            serde_json::json!({"token_index":0,"word_index":1})
        );
        assert_eq!(report["plan"]["strategy"], "interleaved");
        assert_eq!(
            report["plan"]["regions"][0]["strategy"],
            "local_interleaving"
        );
        assert_eq!(report["plan"]["utterances"][1]["status"], "matched");
        assert_eq!(
            report["plan"]["utterances"][1]["admitted_matches"]["first"]["token"],
            serde_json::json!({"token_index":0,"word_index":1}),
            "production authority comes from the separate whole-source composition"
        );
        assert_eq!(
            std::fs::read_to_string(chat).expect("source retained"),
            source
        );
    }

    #[test]
    fn refuses_an_existing_output_instead_of_clobbering_it() {
        let dir = tempfile::tempdir().expect("temporary directory");
        let chat = dir.path().join("input.cha");
        let tokens = dir.path().join("tokens.json");
        let output = dir.path().join("report.json");
        std::fs::write(
            &chat,
            "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n*PAR:\thello .\n@End\n",
        )
        .expect("write CHAT");
        std::fs::write(&tokens, r#"[{"text":"hello","start_ms":100,"end_ms":300}]"#)
            .expect("write tokens");
        std::fs::write(&output, "keep").expect("write sentinel");

        let error = run(&UtrAlignmentEvalArgs {
            chat,
            source_name: None,
            tokens,
            output: output.clone(),
            fuzzy_threshold: None,
            participation: UtrAlignmentParticipation::AllUtterances,
        })
        .expect_err("existing output must be refused");

        assert!(matches!(error, CliError::Io(_)));
        assert_eq!(
            std::fs::read_to_string(output).expect("read sentinel"),
            "keep"
        );
        assert_eq!(
            std::fs::read_dir(dir.path())
                .expect("read temporary directory")
                .count(),
            3,
            "failed publication must not leave a staging artifact"
        );
    }

    #[test]
    fn writes_fingerprinted_match_evidence_without_changing_chat() {
        let dir = tempfile::tempdir().expect("temporary directory");
        let chat = dir.path().join("input.cha");
        let tokens = dir.path().join("tokens.json");
        let output = dir.path().join("report.json");
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n*PAR:\thello .\n@End\n";
        std::fs::write(&chat, chat_text).expect("write CHAT");
        std::fs::write(&tokens, r#"[{"text":"hello","start_ms":100,"end_ms":300}]"#)
            .expect("write tokens");

        run(&UtrAlignmentEvalArgs {
            chat: chat.clone(),
            source_name: None,
            tokens,
            output: output.clone(),
            fuzzy_threshold: Some(0.85.try_into().expect("valid fuzzy threshold")),
            participation: UtrAlignmentParticipation::AllUtterances,
        })
        .expect("write report");

        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&output).expect("read report"))
                .expect("parse report");
        assert_eq!(report["schema_version"], 10);
        assert_eq!(report["interleaving"], serde_json::json!([]));
        assert_eq!(
            report["source_name"],
            serde_json::json!({
                "stem": "input", "basis": "input_path"
            })
        );
        assert_eq!(report["participation"], "all_utterances");
        assert_eq!(
            report["plan"],
            serde_json::json!({
                "strategy": "monotonic",
                "regions": [{
                    "span": {
                        "first_utterance": 0,
                        "last_utterance": 0,
                        "onset_floor": {"kind": "stream_edge"},
                        "onset_ceiling": {"kind": "stream_edge"}
                    },
                    "strategy": "global_dp"
                }],
                "search_envelopes": [null],
                "utterances": [{
                    "status": "matched",
                    "utterance_index": 0,
                    "alignable_words": 1,
                    "matches": {
                        "first": {
                            "word": {"utterance_index": 0, "word_index": 0},
                            "token": {"token_index": 0, "word_index": 0},
                            "chat_text": "hello",
                            "asr_text": "hello",
                            "relation": {"kind": "exact"}
                        },
                        "rest": []
                    },
                    "admitted_matches": {
                        "first": {
                            "word": {"utterance_index": 0, "word_index": 0},
                            "token": {"token_index": 0, "word_index": 0},
                            "chat_text": "hello",
                            "asr_text": "hello",
                            "relation": {"kind": "exact"}
                        },
                        "rest": []
                    },
                    "proposal": {
                        "status": "positive",
                        "start_ms": 100,
                        "end_ms": 300
                    }
                }]
            }),
            "schema 9 preserves single-speaker selection, admission and search evidence"
        );
        assert_eq!(std::fs::read_to_string(chat).expect("read CHAT"), chat_text);
        assert_eq!(
            std::fs::read(&output).expect("read report").last(),
            Some(&b'\n'),
            "published reports use a stable newline-terminated encoding"
        );
    }
}
