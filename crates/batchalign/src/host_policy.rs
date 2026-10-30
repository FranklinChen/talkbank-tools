//! Host-level execution policy resolved once from runtime configuration.
//!
//! Command metadata says what a command *can* do. The host policy says what this
//! machine should actually allow right now: how many jobs to admit, and whether
//! local workers should bootstrap an entire shared profile or only one task to
//! minimize resident memory.

use crate::command_model::ConstrainedHostPolicy;
use crate::runtime::{MemoryTier, MemoryTierKind};
use crate::worker::WorkerBootstrapMode;

/// Hard upper bound on auto-tuned concurrent job slots.
pub const AUTO_CONCURRENT_MAX_SLOTS: usize = 8;

/// Fallback slot count when the CPU count cannot be detected.
pub const AUTO_CONCURRENT_FALLBACK_SLOTS: usize = 4;

/// Resolved host-level execution policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostExecutionPolicy {
    /// Memory tier resolved for this host.
    pub memory_tier: MemoryTier,
    /// How local worker processes should bootstrap.
    pub bootstrap_mode: WorkerBootstrapMode,
}

impl HostExecutionPolicy {
    /// Resolve host policy from one concrete memory tier.
    ///
    /// - Small (<24 GB): `Task` mode, separate single-task workers.
    /// - Medium (24-48 GB): `LazyProfile` mode, one worker per profile but
    ///   models loaded on demand via `ensure_task`. Prevents the speculative
    ///   10-15 GB eager loading that causes memory guard deadlocks.
    /// - Large/Fleet (>48 GB): `Profile` mode, eager loading.
    pub fn from_memory_tier(memory_tier: MemoryTier) -> Self {
        match memory_tier.kind {
            MemoryTierKind::Small => Self {
                memory_tier,
                bootstrap_mode: WorkerBootstrapMode::Task,
            },
            MemoryTierKind::Medium => Self {
                memory_tier,
                bootstrap_mode: WorkerBootstrapMode::LazyProfile,
            },
            MemoryTierKind::Large | MemoryTierKind::Fleet => Self {
                memory_tier,
                bootstrap_mode: WorkerBootstrapMode::Profile,
            },
        }
    }

    /// Whether this host should take the constrained-memory path.
    ///
    /// Both `Task` (small) and `LazyProfile` (medium) are constrained, they
    /// need careful memory management. Only `Profile` (large/fleet) is
    /// unconstrained.
    pub fn is_constrained_host(self) -> bool {
        !matches!(self.bootstrap_mode, WorkerBootstrapMode::Profile)
    }

    /// Resolve auto job concurrency using the current host policy.
    pub fn auto_max_concurrent_jobs(self) -> usize {
        let by_cpu = std::thread::available_parallelism()
            .map(|parallelism| parallelism.get())
            .unwrap_or(AUTO_CONCURRENT_FALLBACK_SLOTS);
        auto_max_concurrent_from(by_cpu, self.memory_tier.max_suggested_workers)
    }

    /// Resolve the per-command file-parallelism hint under current host limits.
    pub(crate) fn resolved_file_parallelism(
        self,
        constrained_host: ConstrainedHostPolicy,
        suggested_parallelism: usize,
    ) -> usize {
        if self.is_constrained_host()
            && matches!(
                constrained_host,
                ConstrainedHostPolicy::SequentialFallback
                    | ConstrainedHostPolicy::DelegatedToSubcommands
            )
        {
            1
        } else {
            suggested_parallelism.max(1)
        }
    }
}

/// Resolve auto job concurrency from CPU and memory caps.
pub fn auto_max_concurrent_from(by_cpu: usize, by_memory: usize) -> usize {
    by_cpu
        .clamp(1, AUTO_CONCURRENT_MAX_SLOTS)
        .min(by_memory.max(1))
}

#[cfg(test)]
mod tests {
    use super::HostExecutionPolicy;
    use crate::command_model::ConstrainedHostPolicy;
    use crate::runtime::MemoryTier;
    use crate::worker::WorkerBootstrapMode;

    #[test]
    fn small_hosts_choose_task_bootstrap() {
        let policy = HostExecutionPolicy::from_memory_tier(MemoryTier::from_total_mb(16_000));
        assert_eq!(policy.bootstrap_mode, WorkerBootstrapMode::Task);
        assert!(policy.is_constrained_host());
    }

    #[test]
    fn medium_hosts_choose_lazy_profile() {
        let policy = HostExecutionPolicy::from_memory_tier(MemoryTier::from_total_mb(32_000));
        assert_eq!(policy.bootstrap_mode, WorkerBootstrapMode::LazyProfile);
        assert!(policy.is_constrained_host());
    }

    #[test]
    fn large_hosts_keep_profile_bootstrap() {
        let policy = HostExecutionPolicy::from_memory_tier(MemoryTier::from_total_mb(64_000));
        assert_eq!(policy.bootstrap_mode, WorkerBootstrapMode::Profile);
        assert!(!policy.is_constrained_host());
    }

    #[test]
    fn constrained_hosts_clamp_sequential_fallback_commands() {
        let policy = HostExecutionPolicy::from_memory_tier(MemoryTier::from_total_mb(16_000));
        assert_eq!(
            policy.resolved_file_parallelism(ConstrainedHostPolicy::SequentialFallback, 4),
            1
        );
        assert_eq!(
            policy.resolved_file_parallelism(ConstrainedHostPolicy::DelegatedToSubcommands, 4),
            1
        );
    }

    #[test]
    fn auto_job_slots_use_memory_cap() {
        let policy = HostExecutionPolicy::from_memory_tier(MemoryTier::from_total_mb(16_000));
        assert_eq!(policy.auto_max_concurrent_jobs(), 1);
    }
}
