//! Shared concurrent GPU worker wrappers.
//!
//! ## Module layout
//!
//! | File | Responsibility |
//! |------|----------------|
//! | `mod.rs` | Envelope deserialization helpers, re-exports |
//! | `channel.rs` | `SharedGpuChannel`, the one transport: dispatch, sequential ops, liveness |
//! | `routes.rs` | Pending dispatches and the control slot, both closing when the stream ends |
//! | `reader.rs` | The reader task that routes each line to its owner |
//! | `stdio.rs` | `SharedGpuWorker`: the channel over a spawned child's stdio, plus process ownership |
//! | `tcp.rs` | `SharedGpuTcpWorker`: the channel over a daemon's TCP connection |

mod channel;
mod reader;
mod routes;
mod stdio;
mod tcp;

pub(crate) use routes::Retirement;
pub(crate) use stdio::SharedGpuWorker;
pub(crate) use tcp::SharedGpuTcpWorker;

use tokio::sync::Semaphore;

use crate::types::worker_v2::ExecuteResponseV2;
pub(crate) use crate::worker::EnsureTaskResponse;

/// Convert a `gpu_thread_pool_size` value into a permit count for the
/// dispatch semaphore. Floor at 1 (zero permits would deadlock every
/// caller) and clamp to `Semaphore::MAX_PERMITS`. Shared between the
/// stdio and TCP shared-GPU-worker constructors so both transports
/// derive the same in-flight ceiling from the same input.
pub(super) fn dispatch_permits_from(gpu_thread_pool_size: u32) -> usize {
    gpu_thread_pool_size
        .max(1)
        .min(u32::try_from(Semaphore::MAX_PERMITS).unwrap_or(u32::MAX)) as usize
}

/// Deserialization envelope types used by the reader loop to parse
/// JSON-lines responses from the worker process.
pub(super) mod envelopes {
    /// Helper envelope for deserializing `{"op": "execute_v2", "response": {...}}`.
    #[derive(serde::Deserialize)]
    pub(crate) struct ExecuteResponseV2Envelope {
        pub(crate) response: super::ExecuteResponseV2,
    }

    /// Helper envelope for deserializing `{"op": "capabilities", "response": {...}}`.
    #[derive(serde::Deserialize)]
    pub(crate) struct CapabilitiesResponseEnvelope {
        pub(crate) response: crate::worker::WorkerCapabilities,
    }

    /// Helper envelope for deserializing `{"op": "ensure_task", "response": {...}}`.
    #[derive(serde::Deserialize)]
    pub(crate) struct EnsureTaskResponseEnvelope {
        pub(crate) response: super::EnsureTaskResponse,
    }
}
