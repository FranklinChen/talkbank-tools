//! `ServerConfig` methods: validation and memory-tier resolution.
//!
//! These are `impl ServerConfig` blocks that depend on runtime types
//! (`MemoryTier`) and the default-value helpers from [`super::server`].

use super::server::*;

impl ServerConfig {
    /// Resolve the effective memory tier, applying config overrides.
    ///
    /// Priority: explicit `memory_tier` field in config → auto-detect from RAM.
    /// Individual startup reservation overrides (`gpu_startup_mb`, etc.) are
    /// applied on top of the resolved tier.
    pub fn resolved_memory_tier(&self) -> crate::types::runtime::MemoryTier {
        use crate::types::runtime::{MemoryTier, MemoryTierKind};

        let mut tier = match self.memory_tier {
            Some(MemoryTierKind::Small) => MemoryTier::from_total_mb(16_000),
            Some(MemoryTierKind::Medium) => MemoryTier::from_total_mb(32_000),
            Some(MemoryTierKind::Large) => MemoryTier::from_total_mb(64_000),
            Some(MemoryTierKind::Fleet) => MemoryTier::from_total_mb(256_000),
            None => MemoryTier::detect(),
        };
        if let Some(value) = self.gpu_startup_mb {
            tier.gpu_startup_mb = value;
        }
        if let Some(value) = self.stanza_startup_mb {
            tier.stanza_startup_mb = value;
        }
        if let Some(value) = self.io_startup_mb {
            tier.io_startup_mb = value;
        }
        tier
    }

    /// Resolve host-memory headroom (MB).
    ///
    /// `Some(n)` is an explicit operator override (used today by
    /// `--sequential` mode to set `MemoryMb(1)`); `None` falls
    /// through to the hardcoded
    /// [`batchalign_types::memory::MIN_FREE_MEMORY_MB`]
    /// floor, the same number the worker-pool admission gate
    /// enforces. The previous tier-derived fallback
    /// (`resolved_memory_tier().headroom_mb`, 2/4/8 GB by host RAM)
    /// has been retired: it tried to encode workload sizing into a
    /// floor that should only express OS-protection headroom.
    pub fn resolved_memory_gate_mb(&self) -> crate::api::MemoryMb {
        match self.memory_gate_mb {
            Some(value) => value,
            None => batchalign_types::memory::MIN_FREE_MEMORY_MB,
        }
    }

    /// Report admission corrections without mutation or filesystem access.
    pub fn validate(&self) -> Vec<String> {
        [
            self.job_ttl_days.warning(),
            self.memory_gate_poll_s.warning(),
            self.max_concurrent_worker_startups.warning(),
        ].into_iter().flatten().collect()
    }
}
