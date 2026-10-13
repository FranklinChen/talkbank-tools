//! Optional data-plane diagnostics for Whisper FA. Never timing authority.

use crate::error::ServerError;
use crate::types::worker_v2::{
    ExecuteResponseV2, TaskResultV2, WhisperAttentionCaptureReceiptV2,
    WhisperAttentionCaptureRequestV2, WorkerArtifactPathV2,
};
use crate::worker::execute_result_v2::require_success_result;
use sqlx::{Connection, Row};
use std::path::Path;

/// Allocate only a name; the worker must exclusively create the file.
pub(super) fn request(
    dir: &Path,
    semantic_key: &str,
) -> Result<WhisperAttentionCaptureRequestV2, ServerError> {
    std::fs::create_dir_all(dir)
        .map_err(|e| ServerError::Validation(format!("attention capture directory: {e}")))?;
    let dir = dir
        .canonicalize()
        .map_err(|e| ServerError::Validation(format!("attention capture directory: {e}")))?;
    let path = dir.join(format!(
        "whisper-attention-{}-{}.sqlite",
        semantic_key,
        uuid::Uuid::new_v4()
    ));
    WhisperAttentionCaptureRequestV2::new(
        WorkerArtifactPathV2::from(path.to_string_lossy().into_owned()),
        semantic_key.to_owned(),
    )
    .map_err(ServerError::Validation)
}

async fn verify(receipt: &WhisperAttentionCaptureReceiptV2) -> Result<(), String> {
    let path = Path::new(receipt.capture.destination().as_ref());
    let bytes = tokio::fs::read(path).await.map_err(|e| e.to_string())?;
    if blake3::hash(&bytes).to_hex().as_str() != receipt.blake3 {
        return Err("attention capture digest mismatch".into());
    }
    let options = sqlx::sqlite::SqliteConnectOptions::new()
        .filename(path)
        .read_only(true);
    let mut connection = sqlx::SqliteConnection::connect_with(&options)
        .await
        .map_err(|e| e.to_string())?;
    let rows = sqlx::query("SELECT schema_version,request_id,semantic_key FROM capture")
        .fetch_all(&mut connection)
        .await
        .map_err(|e| e.to_string())?;
    if rows.len() != 1 {
        return Err("attention capture must contain exactly one request".into());
    }
    let row = &rows[0];
    if row
        .try_get::<i64, _>("schema_version")
        .map_err(|e| e.to_string())?
        != 1
        || row
            .try_get::<String, _>("request_id")
            .map_err(|e| e.to_string())?
            != receipt.request_id.as_ref()
        || row
            .try_get::<String, _>("semantic_key")
            .map_err(|e| e.to_string())?
            != receipt.capture.semantic_key()
    {
        return Err("attention capture request identity mismatch".into());
    }
    let complete: i64 = sqlx::query_scalar("SELECT
        frame_count > 0 AND
        (SELECT count(*) FROM heads) > 0 AND
        (SELECT count(*) FROM labels WHERE row_entry_frame IS NOT NULL) =
          (SELECT count(*) FROM labels) - prefix_length AND
        NOT EXISTS(SELECT 1 FROM labels WHERE row_entry_frame < 0 OR row_entry_frame >= frame_count) AND
        (SELECT count(*) FROM costs) = frame_count * ((SELECT count(*) FROM labels) - prefix_length) AND
        (SELECT count(*) FROM raw_attention) = (SELECT count(*) FROM heads) * (SELECT count(*) FROM costs)
        FROM capture")
        .fetch_one(&mut connection).await.map_err(|e| e.to_string())?;
    connection.close().await.map_err(|e| e.to_string())?;
    if complete != 1 {
        return Err("attention capture numeric population is incomplete".into());
    }
    Ok(())
}

/// Existing numeric captures may be reused, but a token-only cache hit cannot
/// satisfy a diagnostic request. Ordinary cache admission still runs afterwards.
pub(super) async fn cached_available(value: Option<&serde_json::Value>, key: &str) -> bool {
    let Some(value) = value.and_then(|v| v.pointer("/response/result/attention_capture")) else {
        return false;
    };
    let Ok(receipt) = serde_json::from_value::<WhisperAttentionCaptureReceiptV2>(value.clone())
    else {
        return false;
    };
    receipt.capture.semantic_key() == key && verify(&receipt).await.is_ok()
}

/// A requested capture is mandatory evidence, not a best-effort debug write.
pub(super) async fn admit_response(
    requested: &WhisperAttentionCaptureRequestV2,
    request_id: &crate::types::worker_v2::WorkerRequestIdV2,
    response: &ExecuteResponseV2,
) -> Result<(), ServerError> {
    let TaskResultV2::WhisperTokenTimingResult(result) =
        require_success_result(response, "Whisper attention capture")
            .map_err(|e| ServerError::Validation(e.to_string()))?
    else {
        return Err(ServerError::Validation(
            "attention capture returned the wrong result kind".into(),
        ));
    };
    let receipt = result
        .attention_capture
        .as_ref()
        .ok_or_else(|| ServerError::Validation("requested attention capture is missing".into()))?;
    if &receipt.capture != requested || &receipt.request_id != request_id {
        return Err(ServerError::Validation(
            "attention capture response belongs to another request".into(),
        ));
    }
    verify(receipt).await.map_err(ServerError::Validation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::NonNegativeSeconds;
    use crate::types::worker_v2::{WhisperTokenTimingResultV2, WorkerRequestIdV2};

    #[tokio::test]
    async fn receipt_admission_binds_request_and_reuses_only_intact_numeric_artifact() {
        let directory = tempfile::tempdir().unwrap();
        let capture = request(directory.path(), &"a".repeat(64)).unwrap();
        let path = Path::new(capture.destination().as_ref());
        let options = sqlx::sqlite::SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true);
        let mut connection = sqlx::SqliteConnection::connect_with(&options)
            .await
            .unwrap();
        sqlx::raw_sql("CREATE TABLE capture(schema_version INTEGER, request_id TEXT, semantic_key TEXT, frame_count INTEGER, prefix_length INTEGER);
            CREATE TABLE labels(row_entry_frame INTEGER); INSERT INTO labels VALUES(NULL),(0);
            CREATE TABLE heads(ordinal INTEGER); INSERT INTO heads VALUES(0);
            CREATE TABLE raw_attention(value REAL); INSERT INTO raw_attention VALUES(0),(0);
            CREATE TABLE costs(value REAL); INSERT INTO costs VALUES(0),(0);")
            .execute(&mut connection).await.unwrap();
        sqlx::query("INSERT INTO capture VALUES(1,'request',?,2,1)")
            .bind(capture.semantic_key())
            .execute(&mut connection)
            .await
            .unwrap();
        connection.close().await.unwrap();
        let receipt = WhisperAttentionCaptureReceiptV2 {
            capture: capture.clone(),
            request_id: WorkerRequestIdV2::from("request"),
            blake3: blake3::hash(&std::fs::read(path).unwrap())
                .to_hex()
                .to_string(),
        };
        let response = ExecuteResponseV2::success(
            receipt.request_id.clone(),
            TaskResultV2::WhisperTokenTimingResult(WhisperTokenTimingResultV2 {
                tokens: vec![],
                attention_capture: Some(receipt.clone()),
            }),
            NonNegativeSeconds::try_from(0.1).unwrap(),
        );
        admit_response(&capture, &receipt.request_id, &response)
            .await
            .unwrap();
        assert!(
            admit_response(&capture, &WorkerRequestIdV2::from("other"), &response)
                .await
                .is_err()
        );
        let cached = serde_json::json!({"response": {"result": {"attention_capture": receipt}}});
        assert!(cached_available(Some(&cached), capture.semantic_key()).await);
        assert!(!cached_available(None, capture.semantic_key()).await);
        assert!(!cached_available(Some(&cached), &"b".repeat(64)).await);
        std::fs::write(path, b"changed evidence").unwrap();
        assert!(!cached_available(Some(&cached), capture.semantic_key()).await);
    }

    #[tokio::test]
    async fn token_only_result_cannot_satisfy_requested_capture() {
        let directory = tempfile::tempdir().unwrap();
        let capture = request(directory.path(), &"a".repeat(64)).unwrap();
        let id = WorkerRequestIdV2::from("request");
        let response = ExecuteResponseV2::success(
            id.clone(),
            TaskResultV2::WhisperTokenTimingResult(WhisperTokenTimingResultV2 {
                tokens: vec![],
                attention_capture: None,
            }),
            NonNegativeSeconds::try_from(0.1).unwrap(),
        );
        assert!(admit_response(&capture, &id, &response).await.is_err());
    }
}
