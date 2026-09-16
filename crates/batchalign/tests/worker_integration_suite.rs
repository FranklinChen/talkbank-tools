//! Worker protocol, routing, lifecycle, and pool integration tests.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::todo,
    clippy::unimplemented
)]

mod common;
mod live_deadline;

#[path = "pool_per_key_worker_throughput.rs"]
mod pool_per_key_worker_throughput;
#[path = "shared_worker_fixture_smoke.rs"]
mod shared_worker_fixture_smoke;
#[path = "worker_failure_paths.rs"]
mod worker_failure_paths;
#[path = "worker_integration.rs"]
mod worker_integration;
#[path = "worker_pool_thundering_herd.rs"]
mod worker_pool_thundering_herd;
#[path = "worker_protocol_matrix.rs"]
mod worker_protocol_matrix;
#[path = "worker_routing_and_lifecycle.rs"]
mod worker_routing_and_lifecycle;
#[path = "worker_v2_fa_roundtrip.rs"]
mod worker_v2_fa_roundtrip;
