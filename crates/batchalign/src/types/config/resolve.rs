//! `ServerConfig` methods: validation and memory-tier resolution.
//!
//! These are `impl ServerConfig` blocks that depend on runtime types
//! (`MemoryTier`) and the default-value helpers from [`super::server`].

use super::server::*;

impl ServerConfig {
    /// This config with the CLI's device switches applied: `--force-cpu` and
    /// `--allow-mps` are presence-only, so present means `Some(true)` and absent
    /// leaves the file's value (default `None`) for the host recommendation.
    /// The one place that boundary conversion is spelled.
    pub fn with_device_flags(mut self, force_cpu: bool, allow_mps: bool) -> Self {
        if force_cpu {
            self.force_cpu = Some(true);
        }
        if allow_mps {
            self.allow_mps = Some(true);
        }
        self
    }

    /// The memory tier on a host with `ram_total_mb` of RAM: the operator's
    /// declared `memory_tier` if any, otherwise the tier that RAM sizes to,
    /// with the per-profile startup overrides applied. Pure over the facts it
    /// is given, which come from one reading of the host
    /// ([`crate::host_facts::HostSnapshot`]).
    pub fn memory_tier_for_total(&self, ram_total_mb: u64) -> crate::types::runtime::MemoryTier {
        let tier = self
            .declared_memory_tier()
            .unwrap_or_else(|| crate::types::runtime::MemoryTier::from_total_mb(ram_total_mb));
        self.with_startup_overrides(tier)
    }

    /// The tier the operator declared, if any.
    fn declared_memory_tier(&self) -> Option<crate::types::runtime::MemoryTier> {
        use crate::types::runtime::{MemoryTier, MemoryTierKind};
        self.memory_tier.map(|kind| match kind {
            MemoryTierKind::Small => MemoryTier::from_total_mb(16_000),
            MemoryTierKind::Medium => MemoryTier::from_total_mb(32_000),
            MemoryTierKind::Large => MemoryTier::from_total_mb(64_000),
            MemoryTierKind::Fleet => MemoryTier::from_total_mb(256_000),
        })
    }

    /// `tier` with the operator's per-profile startup reservations applied.
    fn with_startup_overrides(
        &self,
        mut tier: crate::types::runtime::MemoryTier,
    ) -> crate::types::runtime::MemoryTier {
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
    /// (the memory tier's `headroom_mb`, 2/4/8 GB by host RAM)
    /// has been retired: it tried to encode workload sizing into a
    /// floor that should only express OS-protection headroom.
    pub fn resolved_memory_gate_mb(&self) -> crate::api::MemoryMb {
        match self.memory_gate_mb {
            Some(value) => value,
            None => batchalign_types::memory::MIN_FREE_MEMORY_MB,
        }
    }
}
