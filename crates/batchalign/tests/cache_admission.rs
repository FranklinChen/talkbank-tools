//! Real CLI and HTTP admission boundaries, using only synthetic retained
//! evidence and isolated databases. No inference request is authorized here.

use crate::cli_common::{CliHarness, MINIMAL_CHAT};
use serde_json::json;
use sqlx::{ConnectOptions, Connection};

#[tokio::test]
async fn retained_cache_cli_replays_legacy_evidence_without_mutating_database() {
    let harness = CliHarness::new();
    let root = tempfile::tempdir().unwrap();
    let digest = "1".repeat(64);
    // Independent wire fixture: request-key field order is part of revision 2.
    let canonical = format!(
        r#"{{"domain":"rev_asr_evidence","schema_version":2,"provider_media_blake3":"{digest}","preparation_recipe":"source_bytes_legacy_audio_mpeg_v1","upload_file_name":"provider-media.wav","upload_mime":"audio/mpeg","upload_metadata":"batchalign3_1111111111111111","requested_language":"eng","expected_speakers":1,"request_policy_revision":"langid-skip-itn-v1","model_revision":"revai:asynchronous-transcript-v1"}}"#
    );
    let key = blake3::hash(canonical.as_bytes()).to_hex().to_string();
    let mut trace: serde_json::Value = serde_json::from_str(&canonical).unwrap();
    trace["trace_schema_version"] = json!(2);
    trace["source_media_blake3"] = json!(digest);
    trace["raw_evidence_key"] = json!(key);
    let trace_path = root.path().join("trace.json");
    std::fs::write(&trace_path, serde_json::to_vec(&trace).unwrap()).unwrap();
    let db_path = root.path().join("cache.db");
    let mut db = sqlx::sqlite::SqliteConnectOptions::new()
        .filename(&db_path)
        .create_if_missing(true)
        .connect()
        .await
        .unwrap();
    sqlx::query("CREATE TABLE cache_entries (key TEXT, task TEXT, data TEXT, engine_version TEXT)")
        .execute(&mut db)
        .await
        .unwrap();
    let evidence = json!({"schema_version": 2, "request_fingerprint": key,
        "evidence": {"transcript": {"monologues": []}, "resolved_language": "eng"}});
    sqlx::query("INSERT INTO cache_entries VALUES (?, 'rev_asr_evidence', ?, 'revai:asynchronous-transcript-v1')")
        .bind(&key).bind(evidence.to_string()).execute(&mut db).await.unwrap();
    db.close().await.unwrap();
    let before = std::fs::read(&db_path).unwrap();
    let output = harness
        .cmd()
        .args(["cache", "inspect-rev", "--trace"])
        .arg(&trace_path)
        .arg("--db")
        .arg(&db_path)
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(report["raw"]["fidelity"], "legacy_typed_projection");
    assert_eq!(report["recording_bytes_revalidated"], false);
    assert_eq!(std::fs::read(&db_path).unwrap(), before);

    let mut db = sqlx::sqlite::SqliteConnectOptions::new()
        .filename(&db_path)
        .connect()
        .await
        .unwrap();
    sqlx::query("DELETE FROM cache_entries")
        .execute(&mut db)
        .await
        .unwrap();
    db.close().await.unwrap();
    let before = std::fs::read(&db_path).unwrap();
    harness
        .cmd()
        .args(["cache", "inspect-rev", "--trace"])
        .arg(&trace_path)
        .arg("--db")
        .arg(&db_path)
        .assert()
        .code(1);
    assert_eq!(std::fs::read(&db_path).unwrap(), before);
    let absent = root.path().join("absent.db");
    harness
        .cmd()
        .args(["cache", "inspect-rev", "--trace"])
        .arg(&trace_path)
        .arg("--db")
        .arg(&absent)
        .assert()
        .failure();
    assert!(!absent.exists());
    let audio = root.path().join("metadata-only.wav");
    std::fs::write(&audio, b"not decoded or read as recording bytes").unwrap();
    let before = std::fs::read(&db_path).unwrap();
    harness
        .cmd()
        .args(["cache", "inspect-utr", "--audio-path"])
        .arg(&audio)
        .args(["--lang", "eng", "--engine", "rev", "--db"])
        .arg(&db_path)
        .assert()
        .code(1);
    assert_eq!(std::fs::read(&db_path).unwrap(), before);
    for (path, language) in [
        (&audio, "not-a-language"),
        (&root.path().to_path_buf(), "eng"),
    ] {
        harness
            .cmd()
            .args(["cache", "inspect-utr", "--audio-path"])
            .arg(path)
            .args(["--lang", language, "--engine", "rev", "--db"])
            .arg(&db_path)
            .assert()
            .code(2);
        assert_eq!(std::fs::read(&db_path).unwrap(), before);
    }
}

#[test]
fn conflicting_cache_cli_policy_fails_before_server_contact() {
    let harness = CliHarness::new();
    let input = harness.home_dir().join("sample.cha");
    std::fs::write(&input, MINIMAL_CHAT).unwrap();
    let output = harness
        .cmd()
        .args([
            "--require-media-cache-tasks",
            "utr_asr",
            "--override-media-cache-tasks",
            "utr_asr",
            "align",
        ])
        .arg(&input)
        .assert()
        .code(2)
        .get_output()
        .stderr
        .clone();
    assert!(
        String::from_utf8(output)
            .unwrap()
            .contains("invalid media-cache policy")
    );
}

#[tokio::test]
async fn conflicting_cache_http_policy_is_rejected_without_creating_job() {
    let session = crate::common::test_server_fixture::acquire_test_server_session().await;
    let response = session
        .client()
        .post(format!("{}/jobs", session.base_url()))
        .json(&json!({
            "command": "align", "lang": "eng", "num_speakers": 1,
            "files": [{"filename": "sample.cha", "content": MINIMAL_CHAT}],
            "options": {"command": "align", "require_media_cache_tasks": ["utr_asr"],
                "override_media_cache_tasks": ["utr_asr"]}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    assert!(
        response
            .text()
            .await
            .unwrap()
            .contains("cannot be both required and refreshed")
    );
    let jobs: serde_json::Value = session
        .client()
        .get(format!("{}/jobs", session.base_url()))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(jobs, json!([]));
    session.close().await;
}
