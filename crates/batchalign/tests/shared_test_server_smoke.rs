// Integration test target: Cargo compiles this as a separate crate,
// so the lib's `cfg_attr(test, ...)` allow does not apply. Test code
// uses `unwrap`/`expect` by convention.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable
)]

//! Smoke tests for the shared test-server fixture.
//!
//! Pin three behaviors of the fixture before any production test binary
//! depends on it:
//!
//! - Two acquired sessions return distinct `base_url`s and distinct
//!   `state_dir`s: control-plane state cannot bleed between tests.
//! - Both sessions share a single warmed `WorkerPool`: across both
//!   acquires combined, `prepare_workers` runs at most once.
//! - Each session's `/health` endpoint responds 200, proving the axum
//!   server is actually listening on the reported port.

use crate::common;

use common::test_server_fixture::{acquire_test_server_session, times_prepared};

#[tokio::test]
async fn two_sessions_have_distinct_state_dirs_and_base_urls() {
    crate::prerequisite::PythonWorker::require();

    let session_a = acquire_test_server_session().await;
    let session_b = acquire_test_server_session().await;

    assert_ne!(
        session_a.base_url(),
        session_b.base_url(),
        "two acquired sessions must report distinct base URLs"
    );
    assert_ne!(
        session_a.state_dir(),
        session_b.state_dir(),
        "two acquired sessions must own distinct state directories"
    );
}

#[tokio::test]
async fn two_sessions_share_one_warm_pool() {
    crate::prerequisite::PythonWorker::require();

    // First acquire warms the pool. Second acquire must reuse it.
    let _session_a = acquire_test_server_session().await;
    let _session_b = acquire_test_server_session().await;

    assert_eq!(
        times_prepared(),
        1,
        "shared fixture must invoke prepare_workers exactly once across multiple sessions"
    );
}

#[tokio::test]
async fn session_health_endpoint_is_reachable() {
    crate::prerequisite::PythonWorker::require();

    let session = acquire_test_server_session().await;

    let resp = session
        .client()
        .get(format!("{}/health", session.base_url()))
        .send()
        .await
        .expect("GET /health should succeed");
    assert_eq!(resp.status(), 200, "health endpoint should respond 200");
}
