//! Binary-boundary tests for offline UTR alignment replay.
//!
//! These invoke the executable an operator uses. No server, model, media, or
//! provider credential is involved.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::todo,
    clippy::unimplemented
)]

use crate::cli_common;

use cli_common::CliHarness;
use predicates::prelude::*;

const CHAT: &str = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n*PAR:\thello .\n@End\n";
const TOKENS: &str = r#"[{"text":"hello","start_ms":100,"end_ms":300}]"#;

#[test]
fn offline_utr_correspondence_separates_selected_path_from_timing_authority() {
    let harness = CliHarness::new();
    let chat = harness.home_dir().join("input.cha");
    let tokens = harness.home_dir().join("tokens.json");
    let output = harness.home_dir().join("report.json");
    std::fs::write(&chat, CHAT.replace("hello .", "hello now .")).expect("CHAT");
    std::fs::write(
        &tokens,
        r#"[
        {"text":"hello","start_ms":100,"end_ms":200},
        {"text":"hello","start_ms":300,"end_ms":400},
        {"text":"now","start_ms":500,"end_ms":600}
    ]"#,
    )
    .expect("tokens");
    harness
        .cmd()
        .args(["eval", "utr-alignment"])
        .arg("--chat")
        .arg(&chat)
        .arg("--tokens")
        .arg(&tokens)
        .arg("--output")
        .arg(&output)
        .assert()
        .success();
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output).expect("report")).expect("JSON");
    assert_eq!(report["schema_version"], 10);
    let utterance = &report["plan"]["utterances"][0];
    assert_eq!(utterance["matches"]["first"]["chat_text"], "hello");
    assert_eq!(utterance["admitted_matches"]["first"]["chat_text"], "now");
    assert_eq!(utterance["admitted_matches"]["rest"], serde_json::json!([]));
    assert!(utterance.get("proposal").is_none());
    insta::assert_json_snapshot!(serde_json::json!({
        "status": utterance["status"], "missing_endpoints": utterance["missing_endpoints"]
    }), @r###"
    {
      "missing_endpoints": "first",
      "status": "interior_only"
    }
    "###);
}

#[test]
fn offline_utr_replay_reports_all_matched_extrema_and_an_invalid_interior_token() {
    let harness = CliHarness::new();
    let chat = harness.home_dir().join("input.cha");
    let text = CHAT.replace("*PAR:\thello .", "*PAR:\thello world again .");
    std::fs::write(&chat, &text).expect("write CHAT");
    let cases = [
        (
            "nested",
            [(100, 900), (200, 300), (400, 500)],
            serde_json::json!({"status":"positive","start_ms":100,"end_ms":900}),
        ),
        (
            "interior",
            [(100, 200), (150, 1200), (300, 400)],
            serde_json::json!({"status":"positive","start_ms":100,"end_ms":1200}),
        ),
        (
            "invalid",
            [(100, 200), (300, 300), (400, 500)],
            serde_json::json!({"status":"non_positive","start_ms":300,"end_ms":300}),
        ),
    ];
    let mut observed = Vec::new();
    for (name, timings, expected) in cases {
        let tokens = harness.home_dir().join(format!("tokens-{name}.json"));
        let output = harness.home_dir().join(format!("report-{name}.json"));
        let provider = ["hello", "world", "again"]
            .into_iter()
            .zip(timings)
            .map(|(text, (start_ms, end_ms))| {
                serde_json::json!({
                    "text":text,"start_ms":start_ms,"end_ms":end_ms,
                })
            })
            .collect::<Vec<_>>();
        std::fs::write(&tokens, serde_json::to_vec(&provider).expect("token JSON"))
            .expect("write provider evidence");
        harness
            .cmd()
            .args(["eval", "utr-alignment"])
            .arg("--chat")
            .arg(&chat)
            .arg("--tokens")
            .arg(&tokens)
            .arg("--output")
            .arg(&output)
            .assert()
            .success();
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(output).expect("read report"))
                .expect("report JSON");
        let proposal = &report["plan"]["utterances"][0]["proposal"];
        assert_eq!(proposal, &expected, "{name}");
        observed.push(serde_json::json!({"case":name,"proposal":proposal}));
    }
    assert_eq!(std::fs::read_to_string(chat).expect("read CHAT"), text);
    insta::assert_json_snapshot!(observed, @r###"
    [
      {
        "case": "nested",
        "proposal": {
          "end_ms": 900,
          "start_ms": 100,
          "status": "positive"
        }
      },
      {
        "case": "interior",
        "proposal": {
          "end_ms": 1200,
          "start_ms": 100,
          "status": "positive"
        }
      },
      {
        "case": "invalid",
        "proposal": {
          "end_ms": 300,
          "start_ms": 300,
          "status": "non_positive"
        }
      }
    ]
    "###);
}

#[test]
fn offline_utr_replay_checks_declared_original_name_without_debug_suffix_guessing() {
    let harness = CliHarness::new();
    let chat = harness.home_dir().join("recording_utr_input.cha");
    let tokens = harness.home_dir().join("tokens.json");
    let text = CHAT
        .replace("*PAR:", "@Media:\trecording, audio\n*PAR:")
        .replace("hello .", "hello . \u{15}100_300\u{15}");
    std::fs::write(&chat, &text).expect("write renamed debug snapshot");
    std::fs::write(&tokens, TOKENS).expect("write tokens");

    for name in [None, Some("wrong.cha"), Some("/")] {
        let output = harness
            .home_dir()
            .join(format!("refused-{}.json", name.is_some()));
        let mut command = harness.cmd();
        command
            .args(["eval", "utr-alignment"])
            .arg("--chat")
            .arg(&chat)
            .arg("--tokens")
            .arg(harness.home_dir().join("absent-tokens.json"))
            .arg("--output")
            .arg(&output);
        if let Some(name) = name {
            command.args(["--source-name", name]);
        }
        command.assert().code(2);
        assert!(!output.exists(), "a name refusal cannot publish evidence");
    }

    let output = harness.home_dir().join("admitted.json");
    harness
        .cmd()
        .args(["eval", "utr-alignment"])
        .arg("--chat")
        .arg(&chat)
        .arg("--tokens")
        .arg(&tokens)
        .arg("--output")
        .arg(&output)
        .args(["--source-name", "recording.cha"])
        .assert()
        .success();
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output).expect("read admitted report"))
            .expect("report JSON");
    assert_eq!(report["schema_version"], 10);
    assert_eq!(
        report["source_name"],
        serde_json::json!({
            "stem": "recording", "basis": "caller_declared"
        })
    );
    assert_eq!(report["plan"]["utterances"][0]["status"], "matched");
    assert_eq!(std::fs::read_to_string(chat).unwrap(), text);
    assert_eq!(std::fs::read_to_string(tokens).unwrap(), TOKENS);
}

#[cfg(unix)]
#[test]
fn offline_utr_replay_refuses_non_utf8_name_instead_of_anonymous_admission() {
    use std::os::unix::ffi::OsStringExt;

    let harness = CliHarness::new();
    let chat = harness.home_dir().join("input.cha");
    let output = harness.home_dir().join("report.json");
    std::fs::write(&chat, CHAT).expect("write CHAT");
    harness
        .cmd()
        .args(["eval", "utr-alignment"])
        .arg("--chat")
        .arg(&chat)
        .arg("--tokens")
        .arg(harness.home_dir().join("absent-tokens.json"))
        .arg("--output")
        .arg(&output)
        .arg("--source-name")
        .arg(std::ffi::OsString::from_vec(b"invalid\xff.cha".to_vec()))
        .assert()
        .code(2);
    assert!(!output.exists());
    assert_eq!(std::fs::read_to_string(chat).unwrap(), CHAT);
}

#[test]
fn offline_utr_replay_requires_complete_named_retained_source_before_token_read() {
    let cases = [
        CHAT.replace("@Languages:\teng\n", ""),
        CHAT.replace("@End\n", "%mor:\tn|hello n|extra .\n@End\n"),
        CHAT.replace("@End\n", "%gra:\t1|999|ROOT\n@End\n"),
        CHAT.replace("@End\n", "%wor:\thello \u{15}900_100\u{15} .\n@End\n"),
        CHAT.replace("*PAR:", "*UNK:"),
        CHAT.replace("*PAR:", "@Media:\tother, audio\n*PAR:")
            .replace("hello .", "hello . \u{15}100_300\u{15}"),
    ];
    for text in cases {
        let harness = CliHarness::new();
        let chat = harness.home_dir().join("input.cha");
        let tokens = harness.home_dir().join("absent-tokens.json");
        let output = harness.home_dir().join("report.json");
        std::fs::write(&chat, &text).expect("write rejection-only source");

        harness
            .cmd()
            .args(["eval", "utr-alignment"])
            .arg("--chat")
            .arg(&chat)
            .arg("--tokens")
            .arg(&tokens)
            .arg("--output")
            .arg(&output)
            .assert()
            .code(2);
        assert!(!output.exists(), "a source refusal cannot publish evidence");
        assert_eq!(std::fs::read_to_string(chat).unwrap(), text);
    }
}

#[test]
fn offline_utr_replay_writes_typed_evidence_without_changing_chat() {
    let harness = CliHarness::new();
    let chat = harness.home_dir().join("input.cha");
    let tokens = harness.home_dir().join("tokens.json");
    let output = harness.home_dir().join("report.json");
    std::fs::write(&chat, CHAT).expect("write CHAT");
    std::fs::write(&tokens, TOKENS).expect("write tokens");

    harness
        .cmd()
        .args(["eval", "utr-alignment"])
        .args(["--chat", chat.to_str().expect("CHAT path")])
        .args(["--tokens", tokens.to_str().expect("token path")])
        .args(["--output", output.to_str().expect("output path")])
        .args(["--fuzzy-threshold", "0.85"])
        .assert()
        .success();

    let report: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&output).expect("the subprocess must publish a report"),
    )
    .expect("report JSON");
    assert_eq!(
        report["source_name"],
        serde_json::json!({
            "stem": "input", "basis": "input_path"
        })
    );
    assert_eq!(report["plan"]["utterances"][0]["status"], "matched");
    assert_eq!(
        std::fs::read_to_string(&chat).expect("read unchanged CHAT"),
        CHAT
    );
}

#[test]
fn offline_utr_projects_segment_words_and_retains_provider_timing() {
    let harness = CliHarness::new();
    let chat = harness.home_dir().join("input.cha");
    let tokens = harness.home_dir().join("tokens.json");
    let text = CHAT.replace("*PAR:\thello .", "*PAR:\thello world .\n*PAR:\tagain .");
    let provider = r#"[{"text":"  ","start_ms":0,"end_ms":50},{"text":" hello\u2003world again ","start_ms":100,"end_ms":900}]"#;
    std::fs::write(&chat, &text).expect("write CHAT");
    std::fs::write(&tokens, provider).expect("write retained provider segments");

    for fuzzy in [false, true] {
        let output = harness.home_dir().join(format!("report-{fuzzy}.json"));
        let mut command = harness.cmd();
        command
            .args(["eval", "utr-alignment"])
            .arg("--chat")
            .arg(&chat)
            .arg("--tokens")
            .arg(&tokens)
            .arg("--output")
            .arg(&output);
        if fuzzy {
            command.args(["--fuzzy-threshold", "0.85"]);
        }
        command.assert().success();
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(output).expect("read report"))
                .expect("report JSON");
        assert_eq!(report["schema_version"], 10);
        let utterances = report["plan"]["utterances"].as_array().expect("utterances");
        assert_eq!(utterances.len(), 2);
        for utterance in utterances {
            assert_eq!(utterance["status"], "matched");
            assert_eq!(utterance["proposal"]["start_ms"], 100);
            assert_eq!(
                utterance["proposal"]["end_ms"], 900,
                "lexical projection must not invent sub-segment times"
            );
        }
        let first = &utterances[0]["matches"];
        assert_eq!(first["first"]["asr_text"], "hello");
        assert_eq!(
            first["first"]["token"],
            serde_json::json!({"token_index":1,"word_index":0})
        );
        assert_eq!(first["rest"][0]["asr_text"], "world");
        assert_eq!(
            first["rest"][0]["token"],
            serde_json::json!({"token_index":1,"word_index":1})
        );
        assert_eq!(
            utterances[1]["matches"]["first"]["token"],
            serde_json::json!({"token_index":1,"word_index":2})
        );
    }
    assert_eq!(std::fs::read_to_string(chat).expect("read CHAT"), text);
    assert_eq!(
        std::fs::read_to_string(tokens).expect("read retained segments"),
        provider
    );
}

#[test]
fn offline_utr_replay_rejects_invalid_policy_before_writing() {
    let harness = CliHarness::new();
    let chat = harness.home_dir().join("input.cha");
    let tokens = harness.home_dir().join("tokens.json");
    let output = harness.home_dir().join("report.json");
    std::fs::write(&chat, CHAT).expect("write CHAT");
    std::fs::write(&tokens, TOKENS).expect("write tokens");

    harness
        .cmd()
        .args(["eval", "utr-alignment"])
        .args(["--chat", chat.to_str().expect("CHAT path")])
        .args(["--tokens", tokens.to_str().expect("token path")])
        .args(["--output", output.to_str().expect("output path")])
        .args(["--fuzzy-threshold", "1.01"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("between 0 and 1"));

    assert!(
        !output.exists(),
        "invalid CLI state must not create a report"
    );
}

#[test]
fn offline_utr_replay_refuses_to_replace_a_published_report() {
    let harness = CliHarness::new();
    let chat = harness.home_dir().join("input.cha");
    let tokens = harness.home_dir().join("tokens.json");
    let output = harness.home_dir().join("report.json");
    std::fs::write(&chat, CHAT).expect("write CHAT");
    std::fs::write(&tokens, TOKENS).expect("write tokens");
    std::fs::write(&output, "retain this evidence\n").expect("write sentinel");

    harness
        .cmd()
        .args(["eval", "utr-alignment"])
        .args(["--chat", chat.to_str().expect("CHAT path")])
        .args(["--tokens", tokens.to_str().expect("token path")])
        .args(["--output", output.to_str().expect("output path")])
        .assert()
        .failure();

    assert_eq!(
        std::fs::read_to_string(output).expect("read sentinel"),
        "retain this evidence\n"
    );
}
