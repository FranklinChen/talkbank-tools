//! One reading of the host, taken where a process is composed.
//!
//! Total memory, the CPU count and the GPU are fixed for a process's
//! lifetime, and [`HostFacts`] says so: it is a startup snapshot. Everything
//! derived from those facts (the operator's overrides resolved against them,
//! the memory tier, the execution policy) is derived here, once, from the one
//! reading, and handed down as this value.
//!
//! Before this type, a server read the host in five places at startup and once
//! more for every job (the dispatch context re-ran detection, including the
//! `nvidia-smi` probe on non-Apple hosts), each a separate place an unreadable
//! host had to be handled and a separate chance for two readings to disagree.
//! Now the composing caller ([`HostSnapshot::read`]) is the only one that can
//! meet [`HostUnreadable`]; everything downstream takes the snapshot and cannot
//! fail on the host.
//!
//! Live readings (available memory for admission, the process table, the load
//! average) are deliberately not here: they change while the process runs, and
//! each consumer has its own policy for an unreadable one.

use super::{ConfigOverrides, EffectiveConfig, HostFacts, HostFactsSource, RealHostFactsSource};
use crate::config::ServerConfig;
use crate::host_memory::HostUnreadable;
use crate::host_policy::HostExecutionPolicy;
use crate::types::runtime::MemoryTier;

/// The host's fixed facts and everything derived from them for one config.
#[derive(Debug, Clone)]
pub struct HostSnapshot {
    facts: HostFacts,
    effective: EffectiveConfig,
    tier: MemoryTier,
    max_concurrent_jobs: usize,
}

impl HostSnapshot {
    /// Read this host once and resolve `config` against it.
    pub fn read(config: &ServerConfig) -> Result<Self, HostUnreadable> {
        Ok(Self::from_facts(config, RealHostFactsSource.detect()?))
    }

    /// Resolve `config` against facts already in hand (a mock source, or a
    /// reading taken for another purpose).
    pub fn from_facts(config: &ServerConfig, facts: HostFacts) -> Self {
        let effective = EffectiveConfig::resolve(&ConfigOverrides::from(config), &facts);
        let tier = config.memory_tier_for_total(facts.ram_total_mb);
        // `Some(n)` is the operator's explicit override; `None` falls through
        // to the tier's auto-tune.
        let max_concurrent_jobs = match config.max_concurrent_jobs {
            Some(n) => n as usize,
            None => HostExecutionPolicy::from_memory_tier(tier).auto_max_concurrent_jobs(),
        };
        Self {
            facts,
            effective,
            tier,
            max_concurrent_jobs,
        }
    }

    /// A snapshot of a declared 64 GB Apple Silicon host, for tests: no test's
    /// outcome may depend on the machine running it.
    pub fn for_tests(config: &ServerConfig) -> Self {
        Self::from_facts(config, super::test_helpers::apple_silicon_64gb())
    }

    /// The facts as read.
    pub fn facts(&self) -> &HostFacts {
        &self.facts
    }

    /// The operator's overrides resolved against the facts.
    pub fn effective(&self) -> &EffectiveConfig {
        &self.effective
    }

    /// The memory tier: the operator's declared tier, or the facts' RAM.
    pub fn tier(&self) -> MemoryTier {
        self.tier
    }

    /// How many jobs may run at once: the operator's `max_concurrent_jobs`,
    /// or the tier's auto-tune.
    pub fn max_concurrent_jobs(&self) -> usize {
        self.max_concurrent_jobs
    }

    /// The execution policy the tier implies.
    pub fn policy(&self) -> HostExecutionPolicy {
        HostExecutionPolicy::from_memory_tier(self.tier)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::runtime::MemoryTierKind;

    #[test]
    fn a_declared_tier_wins_over_the_facts_and_undeclared_follows_them() {
        let undeclared = HostSnapshot::for_tests(&ServerConfig::default());
        assert_eq!(
            undeclared.tier(),
            MemoryTier::from_total_mb(undeclared.facts().ram_total_mb)
        );
        let declared = HostSnapshot::for_tests(&ServerConfig {
            memory_tier: Some(MemoryTierKind::Small),
            ..Default::default()
        });
        assert_eq!(declared.tier().kind, MemoryTierKind::Small);
    }

    #[test]
    fn reading_this_host_resolves_from_the_same_facts_it_reports() {
        let snapshot =
            HostSnapshot::read(&ServerConfig::default()).expect("the test host reports its memory");
        assert!(snapshot.facts().ram_total_mb > 0);
        assert_eq!(
            snapshot.tier(),
            MemoryTier::from_total_mb(snapshot.facts().ram_total_mb)
        );
    }
}
