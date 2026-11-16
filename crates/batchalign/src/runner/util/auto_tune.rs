//! Per-job file parallelism auto-tuning.

use crate::api::{NumWorkers, ReleasedCommand};
use crate::host_facts::HostSnapshot;
use crate::runtime;

/// Compute parallelism from one retained host snapshot and the workload.
/// The host memory coordinator subsequently admits the requested workers.
pub(in crate::runner) fn compute_job_workers(
    command: ReleasedCommand,
    num_files: usize,
    host: &HostSnapshot,
) -> NumWorkers {
    let effective = host.effective();
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
    // CPU-only execution follows the CPU category even for a command whose
    // catalog profile usually uses a GPU. The declared-host policy matrix
    // retains this choice alongside the file, CPU, operator and tier limits.
    let is_gpu_heavy = batchalign_types::command_spec::command_spec_for(command).profile
        == Some(batchalign_types::worker_profile::WorkerProfile::Gpu)
        && !effective.force_cpu;

    let by_cpu = host.facts().cpu_logical_count as usize;
    let tier_cap = host.tier().max_suggested_workers;

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
