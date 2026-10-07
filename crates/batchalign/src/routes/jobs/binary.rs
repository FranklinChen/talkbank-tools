//! Binary result delivery from a recorded successful artifact, never an
//! arbitrary requested filesystem path. Reads use one held file and bounded
//! chunks; JSON result endpoints carry its independent identity descriptor.

use crate::AppState;
use crate::api::{BinaryResultDescriptor, JobId};
use crate::error::ServerError;
use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{HeaderValue, header};
use axum::response::Response;
use std::num::NonZeroU64;
use std::sync::Arc;
use tokio::io::AsyncReadExt;

/// Stream a recorded successful binary result. This endpoint inherits the
/// same router/middleware boundary as job metadata and text results.
#[utoipa::path(
    get,
    path = "/jobs/{job_id}/artifacts/{filename}",
    tag = "jobs",
    params(
        ("job_id" = String, Path, description = "Job identifier"),
        ("filename" = String, Path, description = "Exact recorded artifact identity")
    ),
    responses(
        (status = 200, description = "Binary recording bytes", content_type = "application/octet-stream"),
        (status = 404, description = "No successful binary result", body = crate::openapi::ErrorResponse)
    )
)]
pub(crate) async fn get_binary_result(
    State(state): State<Arc<AppState>>,
    Path((job_id, filename)): Path<(String, String)>,
) -> Result<Response, ServerError> {
    let job_id = JobId::from(job_id);
    let detail = state
        .control
        .backend
        .get_job_detail(&job_id)
        .await
        .ok_or_else(|| ServerError::JobNotFound(job_id.clone()))?;
    binary_response(&detail, &filename).await
}

pub(super) async fn binary_response(
    detail: &crate::store::JobDetail,
    filename: &str,
) -> Result<Response, ServerError> {
    open_artifact(detail, filename).await?.into_response()
}

/// One recorded binary result paired with its held regular file. Metadata and
/// streaming cannot bypass the same source/directory/type admission.
pub(super) struct OpenedBinaryArtifact {
    file: tokio::fs::File,
    media_type: &'static str,
    byte_len: NonZeroU64,
}

pub(super) async fn open_artifact(
    detail: &crate::store::JobDetail,
    filename: &str,
) -> Result<OpenedBinaryArtifact, ServerError> {
    let entry = detail
        .results
        .iter()
        .find(|entry| entry.filename.as_ref() == filename && entry.error.is_none())
        .ok_or_else(|| ServerError::FileNotFound("no successful recorded artifact".into()))?;
    let media_type = entry
        .content_type
        .binary_media_type()
        .ok_or_else(|| ServerError::FileNotFound("recorded result is not binary".into()))?;
    let root = tokio::fs::canonicalize(detail.staging_dir.join("output").as_path()).await?;
    let requested = root.join(entry.filename.as_ref());
    // Refuse a planted leaf symlink, including one pointing inside the job.
    let metadata = tokio::fs::symlink_metadata(&requested).await?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(ServerError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "binary artifact is not a regular file",
        )));
    }
    let canonical = tokio::fs::canonicalize(&requested).await?;
    if !canonical.starts_with(&root) {
        return Err(ServerError::FileNotFound(
            "artifact escapes its job output directory".into(),
        ));
    }
    let file = tokio::fs::File::open(canonical).await?;
    let byte_len = NonZeroU64::new(file.metadata().await?.len()).ok_or_else(|| {
        ServerError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "empty binary artifact",
        ))
    })?;
    Ok(OpenedBinaryArtifact {
        file,
        media_type,
        byte_len,
    })
}

impl OpenedBinaryArtifact {
    pub(super) async fn descriptor(&mut self) -> Result<BinaryResultDescriptor, ServerError> {
        Ok(BinaryResultDescriptor::inspect(&mut self.file).await?)
    }

    fn into_response(self) -> Result<Response, ServerError> {
        let stream = futures::stream::try_unfold(self.file, |mut file| async move {
            let mut buffer = vec![0u8; 64 * 1024];
            let read = file.read(&mut buffer).await?;
            if read == 0 {
                return Ok::<_, std::io::Error>(None);
            }
            buffer.truncate(read);
            Ok(Some((buffer, file)))
        });
        let mut response = Response::new(Body::from_stream(stream));
        response.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static(self.media_type),
        );
        let length = HeaderValue::from_str(&self.byte_len.to_string())
            .map_err(|error| ServerError::Io(std::io::Error::other(error)))?;
        response
            .headers_mut()
            .insert(header::CONTENT_LENGTH, length);
        Ok(response)
    }
}

#[cfg(test)]
mod tests;
