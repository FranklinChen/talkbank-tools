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
//! average) are deliberately not stored here: they change while the process
//! runs, and each consumer has its own policy for an unreadable one. What is
//! here is WHERE they come from ([`HostReadings`]): the live machine for a
//! snapshot that [`HostSnapshot::read`] took, the declared facts for one that
//! [`HostSnapshot::for_tests`] made. Every consumer that reads memory or load
//! takes it from the snapshot (directly, or through the [`RuntimeHost`] a
//! worker launch carries), so a declared host is declared all the way down:
//! the tier, the admission gates, the spawn guard, the job planner and the
//! ledger describe one machine.

use super::{ConfigOverrides, EffectiveConfig, HostFacts, HostFactsSource, RealHostFactsSource};
use crate::config::ServerConfig;
use crate::host_memory::{
    DeclaredHost, HostMemoryCoordinator, HostMemoryRuntimeConfig, HostReadings, HostUnreadable,
    PrivateLedger, private_test_ledger_path,
};
use crate::host_policy::HostExecutionPolicy;
use crate::types::runtime::MemoryTier;

/// The host's fixed facts and everything derived from them for one config.
#[derive(Debug, Clone)]
pub struct HostSnapshot {
    facts: HostFacts,
    effective: EffectiveConfig,
    tier: MemoryTier,
    max_concurrent_jobs: usize,
    readings: HostReadings,
    ledger: HostMemoryRuntimeConfig,
}

impl HostSnapshot {
    /// Read this host once and resolve `config` against it.
    pub fn read(config: &ServerConfig) -> Result<Self, HostUnreadable> {
        Ok(Self::resolve(
            config,
            RealHostFactsSource.detect()?,
            HostReadings::Live,
            HostMemoryRuntimeConfig::from_server_config(config),
        ))
    }

    /// Resolve `config` against `facts`, whose live figures come from
    /// `readings` and whose leases go in `ledger`. Private: a snapshot is
    /// either read (live, the machine's ledger) or declared (declared
    /// figures, a ledger of its own), and the two constructors are what pair
    /// them.
    fn resolve(
        config: &ServerConfig,
        facts: HostFacts,
        readings: HostReadings,
        ledger: HostMemoryRuntimeConfig,
    ) -> Self {
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
            readings,
            ledger,
        }
    }

    /// A snapshot of a declared host, for tests: no test's outcome may depend
    /// on the machine running it.
    ///
    /// The host is [`laptop_16gb`](super::test_helpers::laptop_16gb), shaped
    /// after the GitHub Actions runner: the smallest machine BA3 supports, in
    /// the Small tier. It is the only test host. Test servers used to run on
    /// a declared 64 GB host while their worker pools read the live machine,
    /// so one test described two machines and a third decided.
    ///
    /// Its ledger is this process's own
    /// ([`PrivateLedger::DeclaredHost`]): leases taken against declared
    /// memory never meet the machine's real servers, or another test
    /// binary's.
    pub fn for_tests(config: &ServerConfig) -> Self {
        let facts = super::test_helpers::laptop_16gb();
        let readings = HostReadings::Declared(DeclaredHost::from_facts(&facts));
        let ledger = HostMemoryRuntimeConfig::from_sources(
            private_test_ledger_path(PrivateLedger::DeclaredHost),
            config.resolved_memory_gate_mb(),
            config.max_concurrent_worker_startups,
        );
        Self::resolve(config, facts, readings, ledger)
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

    /// Where this host's live figures come from.
    pub fn readings(&self) -> HostReadings {
        self.readings
    }

    /// The host-memory ledger coordinator for this host: the operator's
    /// ledger settings, deciding on this host's readings.
    pub fn memory_coordinator(&self) -> HostMemoryCoordinator {
        HostMemoryCoordinator::new(self.ledger.clone(), self.readings)
    }

    /// The part of this host a worker launch carries.
    pub fn runtime_host(&self) -> RuntimeHost {
        RuntimeHost {
            tier: self.tier,
            readings: self.readings,
            ledger: self.ledger.clone(),
        }
    }
}

/// The host a worker launch is sized for and decides on: its memory tier,
/// where its live figures come from, and its memory ledger.
///
/// Made only by [`HostSnapshot::runtime_host`], so the three describe one
/// machine. They were separate public fields of the launch config, and a
/// struct update could pair a declared test tier with live readings, which
/// every test pool did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeHost {
    tier: MemoryTier,
    readings: HostReadings,
    ledger: HostMemoryRuntimeConfig,
}

impl RuntimeHost {
    /// The memory tier the launch is sized for.
    pub fn tier(&self) -> MemoryTier {
        self.tier
    }

    /// Where the launch's live figures come from.
    pub fn readings(&self) -> &HostReadings {
        &self.readings
    }

    /// A ledger coordinator deciding on this host's readings.
    pub fn memory_coordinator(&self) -> HostMemoryCoordinator {
        HostMemoryCoordinator::new(self.ledger.clone(), self.readings)
    }

    /// The same host with another ledger (a test's own path or reserve).
    /// The ledger carries no tier or readings, so this cannot pair the
    /// host with another machine's figures.
    pub fn with_ledger(self, ledger: HostMemoryRuntimeConfig) -> Self {
        Self { ledger, ..self }
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
            memory_tier: Some(MemoryTierKind::Large),
            ..Default::default()
        });
        assert_eq!(declared.tier().kind, MemoryTierKind::Large);
    }

    /// A declared snapshot's readings are its facts; a read one is live.
    #[test]
    fn a_declared_snapshot_reads_its_own_facts() {
        let declared = HostSnapshot::for_tests(&ServerConfig::default());
        let memory = declared.readings().memory().expect("declared");
        assert_eq!(memory.total_mb.0, declared.facts().ram_total_mb);
        assert_eq!(memory.available_mb.0, declared.facts().ram_available_mb);
        assert_eq!(
            declared.readings().logical_cpus(),
            declared.facts().cpu_logical_count
        );
        assert_eq!(declared.runtime_host().tier(), declared.tier());
        let read = HostSnapshot::read(&ServerConfig::default()).expect("readable");
        assert_eq!(read.readings(), HostReadings::Live);
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
