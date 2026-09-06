//! Offline replay of global UTR word-to-token evidence.

use std::io::Write;
use std::path::Path;

use serde::Serialize;
use talkbank_parser::{ParseProduct, TreeSitterParser};

use crate::chat_ops::fa::utr::{
    AsrTimingToken, GlobalUtrParticipation, UtrAlignmentPlan, UtrMatchMode,
    observe_global_utr_alignment,
};
use crate::cli::args::{UtrAlignmentEvalArgs, UtrAlignmentParticipation};
use crate::cli::error::CliError;

#[derive(Debug, Serialize)]
struct InputIdentity {
    path: String,
    bytes: usize,
    blake3: String,
}

impl InputIdentity {
    fn of(path: &Path, bytes: &[u8]) -> Self {
        Self {
            path: path.display().to_string(),
            bytes: bytes.len(),
            blake3: blake3::hash(bytes).to_hex().to_string(),
        }
    }
}

#[derive(Debug, Serialize)]
struct UtrAlignmentReport {
    schema_version: u8,
    build: &'static str,
    chat: InputIdentity,
    tokens: InputIdentity,
    match_mode: UtrMatchMode,
    participation: GlobalUtrParticipation,
    plan: UtrAlignmentPlan,
}

/// A complete UTR report ready for atomic publication.
///
/// Keeping serialization in a state that exists before any destination is
/// opened prevents JSON failures from creating an artifact that looks
/// authoritative but is incomplete.
struct SerializedUtrAlignmentReport(Vec<u8>);

impl SerializedUtrAlignmentReport {
    fn encode(report: &UtrAlignmentReport) -> Result<Self, CliError> {
        let mut bytes = serde_json::to_vec_pretty(report)?;
        bytes.push(b'\n');
        Ok(Self(bytes))
    }

    /// Atomically publish without replacing an existing evidence artifact.
    fn persist_noclobber(self, output: &Path) -> Result<(), CliError> {
        let parent = output
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let mut staged = tempfile::NamedTempFile::new_in(parent)?;
        staged.write_all(&self.0)?;
        staged.as_file().sync_all()?;
        let published = staged
            .persist_noclobber(output)
            .map_err(|error| CliError::Io(error.error))?;
        published.sync_all()?;
        #[cfg(unix)]
        std::fs::File::open(parent)?.sync_all()?;
        Ok(())
    }
}

/// Replay one retained CHAT/token pair without inference or CHAT mutation.
pub fn run(args: &UtrAlignmentEvalArgs) -> Result<(), CliError> {
    let chat_bytes = std::fs::read(&args.chat)?;
    let token_bytes = std::fs::read(&args.tokens)?;
    let chat_text = std::str::from_utf8(&chat_bytes)
        .map_err(|error| CliError::InvalidArgument(format!("CHAT input is not UTF-8: {error}")))?;
    let parser = TreeSitterParser::new()
        .map_err(|error| CliError::InvalidArgument(format!("parser init: {error}")))?;
    let chat = match parser.parse_chat_file(chat_text) {
        ParseProduct::Built { file, diagnostics } if diagnostics.is_empty() => file,
        ParseProduct::Built { diagnostics, .. } | ParseProduct::Unbuildable { diagnostics } => {
            return Err(CliError::InvalidArgument(format!(
                "CHAT input did not parse cleanly: {}",
                diagnostics
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("; ")
            )));
        }
    };
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
        schema_version: 2,
        build: crate::cli::build_hash(),
        chat: InputIdentity::of(&args.chat, &chat_bytes),
        tokens: InputIdentity::of(&args.tokens, &token_bytes),
        match_mode,
        participation,
        plan: observe_global_utr_alignment(&chat, &tokens, match_mode, participation),
    };
    SerializedUtrAlignmentReport::encode(&report)?.persist_noclobber(&args.output)
}

#[cfg(test)]
mod tests {
    use super::*;

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
            tokens,
            output: output.clone(),
            fuzzy_threshold: Some(0.85.try_into().expect("valid fuzzy threshold")),
            participation: UtrAlignmentParticipation::AllUtterances,
        })
        .expect("write report");

        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&output).expect("read report"))
                .expect("parse report");
        assert_eq!(report["schema_version"], 2);
        assert_eq!(report["participation"], "all_utterances");
        assert_eq!(
            report["plan"],
            serde_json::json!({
                "strategy": "global_dp",
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
                    "proposal": {
                        "status": "positive",
                        "start_ms": 100,
                        "end_ms": 300
                    }
                }]
            }),
            "typed evidence refactors must preserve the admitted report schema"
        );
        assert_eq!(std::fs::read_to_string(chat).expect("read CHAT"), chat_text);
        assert_eq!(
            std::fs::read(&output).expect("read report").last(),
            Some(&b'\n'),
            "published reports use a stable newline-terminated encoding"
        );
    }
}
