//! Data-migration tests for the SQLite evidence cache.

use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use tempfile::TempDir;

use super::*;

/// Regression for the v0.3.0 FA cache-namespace defect.
///
/// A pool-wide first-worker capability once labelled Whisper raw evidence
/// with the `Wave2Vec` model version. Schema 1 did not embed the producer
/// version, so relabelling it would invent provenance and it must be
/// quarantined outside live lookup. Schema 2 does embed the exact
/// selected-worker version, so a mismatched database label can be repaired
/// from the payload. Correctly labelled and unrelated rows must survive.
/// Guards: Opening legacy FA data repairs supported namespaces and quarantines contradictory evidence with its reason.
/// Kind: journey
/// Not a type: Old SQLite rows are external bytes that can claim conflicting producer and envelope versions.
/// State: Private database seeded in the historical shape witnesses migration, quarantine and resulting queries.
/// Concurrency: Private database with opens and queries ordered by awaits.
#[tokio::test]
async fn opening_cache_repairs_or_quarantines_fa_raw_evidence_namespaces() {
    let dir = TempDir::new().expect("temp cache directory");
    let db_path = dir.path().join("cache.db");
    let options = SqliteConnectOptions::new()
        .filename(&db_path)
        .create_if_missing(true);
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect_with(options)
        .await
        .expect("seed legacy cache");
    sqlx::query(
        "CREATE TABLE cache_entries (\
            key TEXT PRIMARY KEY, \
            task TEXT NOT NULL, \
            engine_version TEXT NOT NULL, \
            batchalign_version TEXT NOT NULL, \
            created_at TEXT NOT NULL, \
            data BLOB NOT NULL\
        )",
    )
    .execute(&pool)
    .await
    .expect("seed legacy schema");

    let schema_one_whisper = serde_json::json!({
        "schema_version": 1,
        "requested_engine": "whisper_fa"
    });
    let schema_two_mislabelled = serde_json::json!({
        "schema_version": 2,
        "request_engine_identity": {
            "version": "whisper-fa-large-v2",
            "origin": "selected_worker_capability"
        },
        "requested_engine": "whisper_fa"
    });
    let schema_one_wave = serde_json::json!({
        "schema_version": 1,
        "requested_engine": "wav2vec_fa"
    });
    let schema_one_cantonese = serde_json::json!({
        "schema_version": 1,
        "requested_engine": "cantonese_fa"
    });
    let unrelated = serde_json::json!({"schema_version": 1});

    for (key, task, engine_version, data) in [
        (
            "schema-one-mislabelled",
            "forced_alignment_raw_evidence",
            "wave2vec-fa-mms-2.11.0",
            &schema_one_whisper,
        ),
        (
            "schema-two-mislabelled",
            "forced_alignment_raw_evidence",
            "wave2vec-fa-mms-2.11.0",
            &schema_two_mislabelled,
        ),
        (
            "schema-one-whisper-correct",
            "forced_alignment_raw_evidence",
            "whisper-fa-large-v2",
            &schema_one_whisper,
        ),
        (
            "schema-one-wave-correct",
            "forced_alignment_raw_evidence",
            "wave2vec-fa-mms-2.11.0",
            &schema_one_wave,
        ),
        (
            "schema-one-cantonese-correct",
            "forced_alignment_raw_evidence",
            "wav2vec-canto-mms-300m",
            &schema_one_cantonese,
        ),
        (
            "schema-one-wave-mislabelled",
            "forced_alignment_raw_evidence",
            "whisper-fa-large-v2",
            &schema_one_wave,
        ),
        (
            "schema-one-cantonese-mislabelled",
            "forced_alignment_raw_evidence",
            "wave2vec-fa-mms-2.11.0",
            &schema_one_cantonese,
        ),
        ("unrelated", "utr_asr", "wave2vec-fa-mms-2.11.0", &unrelated),
    ] {
        sqlx::query(
            "INSERT INTO cache_entries \
             (key, task, engine_version, batchalign_version, created_at, data) \
             VALUES (?, ?, ?, 'test', '2026-08-31T00:00:00Z', ?)",
        )
        .bind(key)
        .bind(task)
        .bind(engine_version)
        .bind(serde_json::to_string(data).expect("fixture JSON"))
        .execute(&pool)
        .await
        .expect("seed legacy row");
    }
    pool.close().await;

    let backend = SqliteBackend::open(Some(dir.path().to_path_buf()))
        .await
        .expect("migrate cache");

    let quarantined: (i64, String, String, String) = sqlx::query_as(
        "SELECT COUNT(*), MIN(reason), MIN(engine_version), MIN(CAST(data AS TEXT)) \
         FROM cache_quarantine WHERE key = 'schema-one-mislabelled'",
    )
    .fetch_one(&backend.pool)
    .await
    .expect("the unprovable row remains available as quarantine evidence");
    assert_eq!(quarantined.0, 1);
    assert_eq!(
        quarantined.1,
        "legacy_fa_raw_evidence_engine_namespace_unprovable"
    );
    assert_eq!(quarantined.2, "wave2vec-fa-mms-2.11.0");
    assert_eq!(
        quarantined.3,
        serde_json::to_string(&schema_one_whisper).expect("fixture JSON"),
        "quarantine must retain the exact stored payload"
    );

    let quarantined_keys: Vec<String> =
        sqlx::query_scalar("SELECT key FROM cache_quarantine ORDER BY key")
            .fetch_all(&backend.pool)
            .await
            .expect("read all quarantined engine-family contradictions");
    assert_eq!(
        quarantined_keys,
        [
            "schema-one-cantonese-mislabelled",
            "schema-one-mislabelled",
            "schema-one-wave-mislabelled",
        ]
    );

    assert_eq!(
        backend
            .get(
                "schema-one-mislabelled",
                "forced_alignment_raw_evidence",
                "wave2vec-fa-mms-2.11.0",
            )
            .await
            .expect("read deleted schema-one row"),
        None,
        "schema one cannot be truthfully relabelled without a producer version"
    );
    assert_eq!(
        backend
            .get(
                "schema-two-mislabelled",
                "forced_alignment_raw_evidence",
                "whisper-fa-large-v2",
            )
            .await
            .expect("read repaired schema-two row"),
        Some(schema_two_mislabelled),
        "schema two owns the exact version that repairs its database label"
    );
    assert_eq!(
        backend
            .get(
                "schema-one-wave-correct",
                "forced_alignment_raw_evidence",
                "wave2vec-fa-mms-2.11.0",
            )
            .await
            .expect("read correctly labelled row"),
        Some(schema_one_wave)
    );
    assert_eq!(
        backend
            .get(
                "schema-one-whisper-correct",
                "forced_alignment_raw_evidence",
                "whisper-fa-large-v2",
            )
            .await
            .expect("read correctly labelled Whisper row"),
        Some(schema_one_whisper)
    );
    assert_eq!(
        backend
            .get(
                "schema-one-cantonese-correct",
                "forced_alignment_raw_evidence",
                "wav2vec-canto-mms-300m",
            )
            .await
            .expect("read correctly labelled Cantonese row"),
        Some(schema_one_cantonese)
    );
    assert_eq!(
        backend
            .get("unrelated", "utr_asr", "wave2vec-fa-mms-2.11.0")
            .await
            .expect("read unrelated row"),
        Some(unrelated)
    );
}
