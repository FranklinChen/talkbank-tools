//! How one worker process serves requests, decided once, in Rust.
//!
//! # Why this is a type with one owner
//!
//! Until 2026-09-30 two parties answered "does this worker serve requests
//! concurrently", and they disagreed. The Rust pool treated EVERY GPU-profile
//! worker as a shared concurrent process: one process per key, many requests
//! multiplexed into it, `max_workers_per_key` never consulted. The Python
//! worker decided for itself, probing CUDA: on a CPU-only host (or under
//! `--force-cpu`) it served requests one after another. On such a host all the
//! requests for a key queued on one process while the configured capacity sat
//! unused: concurrent Whisper UTR requests from two jobs ran strictly one at
//! a time while `max_workers_per_key` allowed several processes.
//!
//! Now the pool decides, from the same runtime inputs it launches the worker
//! with, and passes the decision to Python as `--serving`, which Python obeys
//! and no longer derives. The pool routes by the same value, so the dispatch
//! shape and the serving loop cannot disagree.
//!
//! # The decision, as a cross-product
//!
//! | profile | runtime                                   | serving              |
//! |---------|-------------------------------------------|----------------------|
//! | GPU     | `force_cpu`                               | one request/process  |
//! | GPU     | not `force_cpu`, `gpu_thread_pool_size` 1 | one request/process  |
//! | GPU     | not `force_cpu`, `gpu_thread_pool_size` n>1 | shared, n in flight |
//! | Stanza  | any                                       | one request/process  |
//! | IO      | any                                       | one request/process  |
//!
//! Forced CPU is one request per process because each PyTorch request on CPU
//! already uses every core through OpenMP; threads sharing a process only
//! oversubscribe them. A pool size of one is one request per process because a
//! shared process admitting one request at a time is exactly a single
//! exclusive worker, and the only concurrency left is more processes.
//!
//! Stanza is one request per process in every runtime. Every call on a loaded
//! Stanza pipeline goes through that pipeline's own guard (Stanza does not
//! promise a pipeline is safe to call from two threads), so requests
//! multiplexed into one free-threaded process would only queue on the
//! pipeline of their language. Concurrency for a key therefore comes from
//! more processes, up to `max_workers_per_key`, in a free-threaded runtime as
//! in a GIL one.

use std::num::NonZeroU32;

use super::WorkerProfile;
use super::handle::WorkerRuntimeConfig;

/// How one worker process serves requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerServing {
    /// One process serves up to `in_flight` requests at once on threads that
    /// share its loaded models. The pool keeps ONE process per key and
    /// multiplexes requests into it.
    SharedConcurrent {
        /// Requests the process serves at once.
        in_flight: NonZeroU32,
    },
    /// One process serves one request at a time. The pool scales a key out to
    /// `max_workers_per_key` processes and checks each out exclusively.
    OneRequestPerProcess,
}

impl WorkerServing {
    /// The serving mode for workers of `profile` launched with `runtime`.
    pub fn decide(profile: WorkerProfile, runtime: &WorkerRuntimeConfig) -> Self {
        let pool_size = NonZeroU32::new(runtime.gpu_thread_pool_size);
        match profile {
            WorkerProfile::Gpu => match (runtime.force_cpu, pool_size) {
                (true, _) => Self::OneRequestPerProcess,
                (false, Some(in_flight)) if in_flight.get() > 1 => {
                    Self::SharedConcurrent { in_flight }
                }
                // A pool of one (or zero, which admits nothing) is one
                // request per process.
                (false, Some(_) | None) => Self::OneRequestPerProcess,
            },
            WorkerProfile::Stanza | WorkerProfile::Io => Self::OneRequestPerProcess,
        }
    }

    /// Whether the pool dispatches through one shared process per key.
    pub fn is_shared(self) -> bool {
        match self {
            Self::SharedConcurrent { .. } => true,
            Self::OneRequestPerProcess => false,
        }
    }

    /// The worker command-line arguments that carry this decision: every
    /// launcher of a Python worker appends exactly these, so no launcher can
    /// state the mode without the thread count or the reverse.
    ///
    /// `--serving` is the loop the worker enters; `--gpu-thread-pool-size` is
    /// how many requests that loop admits at once (the flag keeps its old
    /// name for the TCP registry and existing daemons).
    pub(crate) fn worker_args(self) -> [String; 4] {
        let (mode, threads) = match self {
            Self::SharedConcurrent { in_flight } => ("concurrent", in_flight.get()),
            Self::OneRequestPerProcess => ("sequential", 1),
        };
        [
            "--serving".to_owned(),
            mode.to_owned(),
            "--gpu-thread-pool-size".to_owned(),
            threads.to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime(force_cpu: bool, gpu_thread_pool_size: u32) -> WorkerRuntimeConfig {
        WorkerRuntimeConfig {
            force_cpu,
            gpu_thread_pool_size,
            ..WorkerRuntimeConfig::for_tests()
        }
    }

    /// Policy table (see the module docs): each row of the cross-product.
    #[test]
    fn serving_follows_the_documented_cross_product() {
        let shared = |n: u32| WorkerServing::SharedConcurrent {
            in_flight: NonZeroU32::new(n).expect("nonzero"),
        };
        let cases = [
            (
                WorkerProfile::Gpu,
                runtime(true, 4),
                WorkerServing::OneRequestPerProcess,
            ),
            (
                WorkerProfile::Gpu,
                runtime(false, 1),
                WorkerServing::OneRequestPerProcess,
            ),
            (
                WorkerProfile::Gpu,
                runtime(false, 0),
                WorkerServing::OneRequestPerProcess,
            ),
            (WorkerProfile::Gpu, runtime(false, 4), shared(4)),
            // Stanza is one request per process whatever the runtime: the
            // decision no longer reads whether Python is free-threaded, so no
            // runtime can multiplex requests onto one guarded pipeline.
            (
                WorkerProfile::Stanza,
                runtime(false, 3),
                WorkerServing::OneRequestPerProcess,
            ),
            (
                WorkerProfile::Io,
                runtime(false, 4),
                WorkerServing::OneRequestPerProcess,
            ),
        ];
        for (profile, runtime, expected) in cases {
            assert_eq!(
                WorkerServing::decide(profile, &runtime),
                expected,
                "{profile:?} force_cpu={} pool={}",
                runtime.force_cpu,
                runtime.gpu_thread_pool_size
            );
        }
    }
}
