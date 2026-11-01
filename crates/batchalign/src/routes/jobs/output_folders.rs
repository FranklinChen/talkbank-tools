//! Loopback-only read projection of persisted job output mappings. No path is
//! accepted from the caller and no filesystem or job mutation happens here.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::Json;
use axum::extract::{ConnectInfo, FromRequestParts, Path, State};
use axum::http::request::Parts;

use crate::AppState;
use crate::api::{JobId, JobOutputFolders};
use crate::error::ServerError;

/// Only a loopback peer can request local output-location authority.
pub(crate) struct LocalOutputRequest(JobId);

impl LocalOutputRequest {
    fn admit(peer: SocketAddr, job_id: String) -> Result<Self, ServerError> {
        if !peer.ip().is_loopback() {
            return Err(ServerError::Validation(
                "Output folders can be inspected only on the processing computer.".into(),
            ));
        }
        Ok(Self(JobId::from(job_id)))
    }
}

impl<S: Send + Sync> FromRequestParts<S> for LocalOutputRequest {
    type Rejection = ServerError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let ConnectInfo(peer) = ConnectInfo::<SocketAddr>::from_request_parts(parts, state)
            .await
            .map_err(|error| ServerError::Validation(error.to_string()))?;
        let Path(job_id) = Path::<String>::from_request_parts(parts, state)
            .await
            .map_err(|error| ServerError::Validation(error.to_string()))?;
        Self::admit(peer, job_id)
    }
}

/// Inspect recorded destinations without guessing an output root or reading
/// file contents. The native action separately admits directory existence.
#[utoipa::path(
    get, path = "/jobs/{job_id}/output-folders", tag = "jobs",
    params(("job_id" = String, Path, description = "Job identifier")),
    responses(
        (status = 200, description = "Recorded local output folders", body = JobOutputFolders),
        (status = 400, description = "Not a local request", body = crate::openapi::ErrorResponse),
        (status = 404, description = "Job not found", body = crate::openapi::ErrorResponse)
    )
)]
pub(crate) async fn get_output_folders(
    State(state): State<Arc<AppState>>,
    LocalOutputRequest(job_id): LocalOutputRequest,
) -> Result<Json<JobOutputFolders>, ServerError> {
    let detail = state
        .control
        .backend
        .get_job_detail(&job_id)
        .await
        .ok_or_else(|| ServerError::JobNotFound(job_id.clone()))?;
    Ok(Json(detail.output_folders))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router,
        body::{Body, to_bytes},
        http::{Request, StatusCode},
        routing::get,
    };
    use tower::ServiceExt;

    #[tokio::test]
    async fn output_folder_http_boundary_admits_only_loopback_peers() {
        for (peer, status) in [
            ("127.0.0.1:7000", StatusCode::OK),
            ("[::1]:7000", StatusCode::OK),
            ("192.0.2.1:7000", StatusCode::BAD_REQUEST),
        ] {
            let app = Router::new().route(
                "/jobs/{job_id}/output-folders",
                get(|LocalOutputRequest(id): LocalOutputRequest| async move { id.to_string() }),
            );
            let mut request = Request::builder()
                .uri("/jobs/recorded-job/output-folders")
                .body(Body::empty())
                .unwrap();
            request
                .extensions_mut()
                .insert(ConnectInfo(peer.parse::<SocketAddr>().unwrap()));
            let response = app.oneshot(request).await.unwrap();
            assert_eq!(response.status(), status);
            if status == StatusCode::OK {
                assert_eq!(
                    to_bytes(response.into_body(), 1024).await.unwrap(),
                    "recorded-job"
                );
            }
        }
    }
}
