//! Per-job file parallelism auto-tuning.

use crate::api::{NumWorkers, ReleasedCommand};
use crate::config::ServerConfig;
use crate::host_facts::EffectiveConfig;
use crate::runtime;

/// Compute the number of parallel file workers for a job.
///
/// Reads from the resolved [`EffectiveConfig`] (operator override merged
/// with host-facts recommendation) for the per-command worker cap, the
/// host-facts-derived GPU thread-pool cap, and the GPU-heavy-vs-CPU
/// category split. The legacy [`ServerConfig`] is consulted only for
/// `resolved_memory_tier()`, which honors the operator's
/// `memory_tier` override (used by the test affordance for
/// constrained-memory simulation on large hosts), this knob has no
/// `EffectiveConfig` analog yet.
///
/// File-count clamping (`min(num_files)`) and CPU clamping
/// (`min(available_parallelism)`) stay at the dispatch site because
/// they're per-job quantities; host-quantities live in `EffectiveConfig`.
///
/// This function intentionally does **not** do host-memory math anymore.
/// It only applies file-count, operator-configured, CPU, and
/// per-category caps. Host-wide memory clamping now happens in the
/// coordinator-backed admission step so worker startup and job
/// execution share one memory model.
pub(in crate::runner) fn compute_job_workers(
    command: ReleasedCommand,
    num_files: usize,
    effective: &EffectiveConfig,
    config: &ServerConfig,
) -> NumWorkers {
    if num_files <= 1 {
        return NumWorkers(1);
    }

    // A command is "GPU-heavy" for concurrency purposes only when the
    // host is actually serving it on a GPU. Under `force_cpu` (Apple
    // Silicon with MPS excluded, hosts without CUDA, operator-set CPU
    // mode), all commands run through CPU workers; there is no shared
    // GPU model process to contend on, so per-job parallelism is bounded
    // by CPU/thread caps, not by `gpu_thread_pool_size`.
    //
    // Without this guard, `--workers N` is silently shadowed: the
    // operator requests N, `gpu_thread_pool_size = 1` (recommended for
    // non-functional GPUs) collapses `category_cap` to 1, and the
    // final clamp ignores the explicit knob. The integration test
    // `tests/serve_start_workers_persisted.rs` pins the user-visible
    // symptom; the unit test
    // `compute_workers_force_cpu_treats_gpu_commands_as_cpu_bound` in
    // `runner::util` pins the planner-level invariant.
    let is_gpu_heavy = batchalign_types::command_spec::command_spec_for(command).profile
        == Some(batchalign_types::worker_profile::WorkerProfile::Gpu)
        && !effective.force_cpu;

    let by_cpu = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);
    let tier_cap = config.resolved_memory_tier().max_suggested_workers;

    // Apply per-category cap. GPU commands share one model process and
    // should not dispatch more in-process requests than the configured
    // GPU thread pool intends to serve. Auto-tuned jobs must also
    // respect the host tier's suggested worker ceiling so
    // medium-memory machines do not request more file parallelism than
    // startup reservations can support.
    let gpu_thread_pool_size = effective.gpu_thread_pool_size as usize;
    let category_cap = if is_gpu_heavy {
        runtime::max_gpu_workers().min(gpu_thread_pool_size)
    } else {
        runtime::max_thread_workers()
    }
    .min(tier_cap);

    // The per-command resolved cap from `EffectiveConfig` already
    // incorporates the operator's uniform override (`Some(n)` from
    // `ConfigOverrides::max_workers_per_job`) when set, falling
    // through to the per-command host-facts recommendation otherwise.
    let resolved_cap = effective.max_workers_per_job(&command) as usize;

    NumWorkers(
        num_files
            .min(by_cpu)
            .min(resolved_cap)
            .clamp(1, category_cap),
    )
}
