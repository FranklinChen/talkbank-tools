//! `EffectiveConfig`: the result of merging operator overrides with
//! `recommend()`'s output, against the detected `HostFacts`.
//!
//! Every production code site that today reads from `ServerConfig`,
//! `PoolConfig`, or `WorkerRuntimeConfig` for one of the host-facts-
//! migrated knobs eventually moves to read from `EffectiveConfig`
//! (Phase C2 of the migration). This module owns the merge logic;
//! callers do not implement override-vs-recommendation resolution
//! themselves.
//!
//! Two types in this file:
//!
//! - **`ConfigOverrides`**: what the operator (or CLI) explicitly
//!   set. Each field is `Option<T>`; `Some(v)` means "use `v`",
//!   `None` means "fall through to the recommendation."
//!   Constructed independently of `ServerConfig`, so the merge logic
//!   does not depend on the legacy `0 = auto` sentinel idiom.
//!   Phase C2 adds `From<&ServerConfig>` impls knob-by-knob to
//!   bridge the legacy fields here.
//!
//! - **`EffectiveConfig`**: the resolved values. Constructed only
//!   via `EffectiveConfig::resolve(overrides, facts)`. Holds the
//!   detected `HostFacts` so per-command queries
//!   (`max_workers_per_job(command)`) can recompute the
//!   recommendation lazily without callers re-passing facts.
//!
//! Layer 3 owns the merge-precedence rules; see
//! `book/src/batchalign/developer/host-facts.md`.

use crate::api::{MemoryMb, ReleasedCommand};
use crate::config::ServerConfig;

use super::HostFacts;
use super::recommendations::{
    PerProfile, RecommendedKnobs, recommend, recommend_max_workers_per_job,
};

/// Operator-supplied overrides, one per migrated knob. `None` means
/// "no override; use the recommendation." `Some(v)` means "use `v`."
///
/// A small dedicated type (rather than passing `&ServerConfig` directly
/// into `resolve`) keeps the merge logic independent of the current
/// `ServerConfig` shape. Phase C2's per-knob migrations populate this
/// struct via `From<&ServerConfig>` impls, the resolve function
/// itself does not change as those migrations land.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConfigOverrides {
    /// Override for `gpu_thread_pool_size`. `None` recommends.
    pub gpu_thread_pool_size: Option<u32>,
    /// Override for `force_cpu`. `Some(true)` forces CPU even on
    /// CUDA-capable hosts (legitimate testing override); `Some(false)`
    /// requests GPU even on hosts where it's not functional (validator
    /// will warn). `None` recommends.
    pub force_cpu: Option<bool>,
    /// Override for `allow_mps`. There is no host recommendation:
    /// the default is always `false` (safe CPU) unless the operator
    /// opts in.
    pub allow_mps: Option<bool>,
    /// Override for `max_total_workers`. `None` recommends.
    pub max_total_workers: Option<u32>,
    /// Override for `max_concurrent_jobs`. `None` recommends.
    pub max_concurrent_jobs: Option<u32>,
    /// Override for the per-command `max_workers_per_job`. Today
    /// applied uniformly across commands (legacy `ServerConfig` has
    /// one knob, not per-command); a future refinement can split this
    /// per command without breaking the merge contract.
    pub max_workers_per_job: Option<u32>,
    /// Per-profile overrides for `max_workers_per_key`. Each profile
    /// can be overridden independently of the others.
    pub max_workers_per_key_by_profile: PerProfileOverrides,
    /// Override for `memory_gate_mb`. `None` recommends.
    pub memory_gate_mb: Option<MemoryMb>,
}

/// Lift a [`ServerConfig`] into a [`ConfigOverrides`].
///
/// `ServerConfig` is the on-disk YAML shape; `ConfigOverrides` is the
/// runtime input to [`EffectiveConfig::resolve`]. Each migrated knob
/// maps one-to-one; the single `max_workers_per_key` knob fans out
/// uniformly to all three per-profile slots (gpu/stanza/io), matching
/// the legacy "one `PoolConfig.max_workers_per_key` applies to every
/// profile in the pool" semantics. Per-profile differentiation would
/// require a `ServerConfig` shape change.
///
/// Knobs that have not yet migrated to `Option<T>` (`force_cpu`)
/// stay `None` here; [`EffectiveConfig::resolve`] then derives them
/// from the host recommendation. This impl is the bridge that lets
/// every call site read from the resolved view without knowing
/// which knobs have crossed the migration boundary.
impl From<&ServerConfig> for ConfigOverrides {
    fn from(cfg: &ServerConfig) -> Self {
        Self {
            gpu_thread_pool_size: cfg.gpu_thread_pool_size,
            force_cpu: cfg.force_cpu,
            allow_mps: cfg.allow_mps,
            max_total_workers: cfg.max_total_workers,
            max_workers_per_job: cfg.max_workers_per_job,
            max_concurrent_jobs: cfg.max_concurrent_jobs,
            memory_gate_mb: cfg.memory_gate_mb,
            max_workers_per_key_by_profile: PerProfileOverrides {
                gpu: cfg.max_workers_per_key,
                stanza: cfg.max_workers_per_key,
                io: cfg.max_workers_per_key,
            },
        }
    }
}

/// Per-profile overrides for `max_workers_per_key`.
///
/// Mirrors the shape of `PerProfile<u32>` from `recommendations.rs`
/// but with `Option<u32>` per field, independent override per profile.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PerProfileOverrides {
    /// Override for the GPU profile worker count per key.
    pub gpu: Option<u32>,
    /// Override for the Stanza profile worker count per key.
    pub stanza: Option<u32>,
    /// Override for the IO profile worker count per key.
    pub io: Option<u32>,
}

/// Resolved per-host configuration values, post-merge.
///
/// Construction goes through [`EffectiveConfig::resolve`]. Holds the
/// `HostFacts` snapshot internally so the per-command
/// `max_workers_per_job` query can lazily compute the per-command
/// recommendation when no override is set, without forcing callers
/// to pass facts at every call site.
#[derive(Debug, Clone)]
pub struct EffectiveConfig {
    /// Resolved `gpu_thread_pool_size`.
    pub gpu_thread_pool_size: u32,
    /// Resolved `force_cpu`.
    pub force_cpu: bool,
    /// Resolved `allow_mps` (no recommendation; defaults false).
    pub allow_mps: bool,
    /// Resolved `max_total_workers`.
    pub max_total_workers: u32,
    /// Resolved `max_concurrent_jobs`.
    pub max_concurrent_jobs: u32,
    /// Resolved per-profile `max_workers_per_key`.
    pub max_workers_per_key_by_profile: PerProfile<u32>,
    /// Resolved `memory_gate_mb`.
    pub memory_gate_mb: MemoryMb,

    /// Operator override for `max_workers_per_job`, retained so
    /// `max_workers_per_job(command)` can apply it uniformly across
    /// commands. `None` means "fall through to the per-command
    /// recommendation computed from `host_facts`."
    max_workers_per_job_override: Option<u32>,

    /// Snapshot of the facts used to construct this config, retained
    /// for the per-command recommendation computation. Cloned once at
    /// resolve time; the struct is small.
    host_facts: HostFacts,
}

impl EffectiveConfig {
    /// Resolve effective values by merging operator overrides with
    /// the recommendation derived from `HostFacts`.
    ///
    /// Merge rule per knob: `override.unwrap_or(recommendation)`.
    /// Per-profile and per-command merges follow the same rule, applied
    /// per field / per command.
    ///
    /// Pure function: same `(overrides, facts)` always produces the
    /// same `EffectiveConfig`. No I/O, no time-dependence.
    #[must_use]
    pub fn resolve(overrides: &ConfigOverrides, facts: &HostFacts) -> Self {
        let r: RecommendedKnobs = recommend(facts);

        let max_workers_per_key_by_profile = PerProfile {
            gpu: overrides
                .max_workers_per_key_by_profile
                .gpu
                .unwrap_or(r.max_workers_per_key_by_profile.gpu),
            stanza: overrides
                .max_workers_per_key_by_profile
                .stanza
                .unwrap_or(r.max_workers_per_key_by_profile.stanza),
            io: overrides
                .max_workers_per_key_by_profile
                .io
                .unwrap_or(r.max_workers_per_key_by_profile.io),
        };

        Self {
            gpu_thread_pool_size: overrides
                .gpu_thread_pool_size
                .unwrap_or(r.gpu_thread_pool_size),
            force_cpu: overrides.force_cpu.unwrap_or(r.force_cpu),
            allow_mps: overrides.allow_mps.unwrap_or(false),
            max_total_workers: overrides.max_total_workers.unwrap_or(r.max_total_workers),
            max_concurrent_jobs: overrides
                .max_concurrent_jobs
                .unwrap_or(r.max_concurrent_jobs),
            max_workers_per_key_by_profile,
            memory_gate_mb: overrides
                .memory_gate_mb
                .unwrap_or(batchalign_types::memory::MIN_FREE_MEMORY_MB),
            max_workers_per_job_override: overrides.max_workers_per_job,
            host_facts: facts.clone(),
        }
    }

    /// Resolved `max_workers_per_job` for one command.
    ///
    /// When the operator set `max_workers_per_job` it wins for every
    /// command (matches the legacy single-knob `ServerConfig` shape).
    /// Otherwise the per-command recommendation is computed from the
    /// stored facts.
    #[must_use]
    pub fn max_workers_per_job(&self, command: &ReleasedCommand) -> u32 {
        self.max_workers_per_job_override
            .unwrap_or_else(|| recommend_max_workers_per_job(&self.host_facts, command))
    }
}

#[cfg(test)]
mod tests {
    use super::super::{GpuPresence, test_helpers::apple_silicon_64gb};
    use super::*;

    fn cmd(name: &str) -> ReleasedCommand {
        ReleasedCommand::try_from(name).expect("test command literal must be valid")
    }

    // -------------------------------------------------------------------
    // Per-knob merge precedence: override Some → effective = override;
    // override None → effective = recommendation.
    // -------------------------------------------------------------------

    /// Guards: With no operator overrides, every resolved knob equals its host recommendation and the memory gate falls back to the minimum free-memory constant.
    /// Kind: contract
    /// Not a type: `ConfigOverrides` uses Option per knob, but a type cannot carry the recommendation table's chosen value for each knob; resolution over a declared host is the remaining pure mapping.
    /// State: None. One declared 64 GB Apple Silicon snapshot supplies every recommendation; no host is read.
    /// Concurrency: None. The snapshot, overrides and resolved config are local to this test.
    #[test]
    fn no_overrides_yields_recommendation() {
        let facts = apple_silicon_64gb();
        let effective = EffectiveConfig::resolve(&ConfigOverrides::default(), &facts);
        let r = recommend(&facts);
        assert_eq!(effective.gpu_thread_pool_size, r.gpu_thread_pool_size);
        assert_eq!(effective.force_cpu, r.force_cpu);
        assert_eq!(effective.max_total_workers, r.max_total_workers);
        assert_eq!(effective.max_concurrent_jobs, r.max_concurrent_jobs);
        assert_eq!(
            effective.max_workers_per_key_by_profile,
            r.max_workers_per_key_by_profile
        );
        assert_eq!(
            effective.memory_gate_mb,
            batchalign_types::memory::MIN_FREE_MEMORY_MB
        );
    }

    /// Guards: Every scalar operator override wins over its host recommendation in one resolution.
    /// Kind: policy
    /// Not a type: `ConfigOverrides` carries each override as Option, but the choice that Some beats the recommendation is a merge rule, not a state a type can forbid; one resolution keeps all four field oracles.
    /// State: None. One declared 64 GB Apple Silicon snapshot and one override set cover all four scalar knobs.
    /// Concurrency: None. The snapshot, overrides and resolved config are local to this test.
    #[test]
    fn scalar_overrides_win_over_recommendation() {
        let facts = apple_silicon_64gb();
        let overrides = ConfigOverrides {
            gpu_thread_pool_size: Some(7),
            max_total_workers: Some(2),
            max_concurrent_jobs: Some(3),
            memory_gate_mb: Some(MemoryMb(16_000)),
            ..Default::default()
        };
        let effective = EffectiveConfig::resolve(&overrides, &facts);
        assert_eq!(effective.gpu_thread_pool_size, 7);
        assert_eq!(effective.max_total_workers, 2);
        assert_eq!(effective.max_concurrent_jobs, 3);
        assert_eq!(effective.memory_gate_mb, MemoryMb(16_000));
    }

    /// Guards: A `force_cpu` override survives both an Apple Silicon true recommendation and a CUDA false recommendation.
    /// Kind: policy
    /// Not a type: An Option<bool> override cannot encode that the operator's explicit choice outranks a contradictory recommendation; the two declared hosts remain the residual boundary.
    /// State: None. Two in-memory fact shapes (Apple Silicon and CUDA) supply the opposing recommendations; no GPU is probed.
    /// Concurrency: None. Each resolution owns its facts and result.
    #[test]
    fn force_cpu_override_survives_either_recommendation() {
        let apple = apple_silicon_64gb();
        assert!(recommend(&apple).force_cpu, "Apple Silicon recommends true");
        let false_override = ConfigOverrides {
            force_cpu: Some(false),
            ..Default::default()
        };
        assert!(!EffectiveConfig::resolve(&false_override, &apple).force_cpu);

        let mut cuda = apple;
        cuda.gpu = GpuPresence::NvidiaCuda {
            device_count: 1,
            total_vram_mb: 24_000,
            driver_version: "555.42".into(),
        };
        assert!(!recommend(&cuda).force_cpu, "CUDA recommends false");
        let true_override = ConfigOverrides {
            force_cpu: Some(true),
            ..Default::default()
        };
        assert!(EffectiveConfig::resolve(&true_override, &cuda).force_cpu);
    }

    /// Guards: Overriding one profile's per-key worker count leaves the other two profiles at their recommendations.
    /// Kind: policy
    /// Not a type: `PerProfileOverrides` has one Option per profile, but the independence of those three fields through merge is a rule of `EffectiveConfig`, not a state a type can forbid.
    /// State: None. One declared host supplies the three recommendations; all values are in memory.
    /// Concurrency: None. The snapshot and resolved config are local to this test.
    #[test]
    fn max_workers_per_key_per_profile_override_is_independent() {
        let facts = apple_silicon_64gb();
        let r = recommend(&facts).max_workers_per_key_by_profile;
        let overrides = ConfigOverrides {
            max_workers_per_key_by_profile: PerProfileOverrides {
                gpu: Some(2),
                stanza: None,
                io: None,
            },
            ..Default::default()
        };
        let effective = EffectiveConfig::resolve(&overrides, &facts);
        assert_eq!(effective.max_workers_per_key_by_profile.gpu, 2);
        assert_eq!(effective.max_workers_per_key_by_profile.stanza, r.stanza);
        assert_eq!(effective.max_workers_per_key_by_profile.io, r.io);
    }

    // -------------------------------------------------------------------
    // Per-command max_workers_per_job: precomputed across
    // ReleasedCommand::ALL.
    // -------------------------------------------------------------------

    /// Guards: With no override, the resolved per-job count equals the recommendation for every released command.
    /// Kind: contract
    /// Not a type: `EffectiveConfig` stores the declared facts to recompute per-command, but that the no-override branch delegates to `recommend_max_workers_per_job` for each of the 14 commands is a mapping no type pins.
    /// State: None. One declared 64 GB host and the released-command set are all in memory.
    /// Concurrency: None. The resolved config is read-only across the loop.
    #[test]
    fn max_workers_per_job_no_override_matches_recommendation_per_command() {
        let facts = apple_silicon_64gb();
        let effective = EffectiveConfig::resolve(&ConfigOverrides::default(), &facts);
        for command in ReleasedCommand::ALL {
            assert_eq!(
                effective.max_workers_per_job(&command),
                recommend_max_workers_per_job(&facts, &command),
                "no-override mismatch for {command:?}"
            );
        }
    }

    /// Guards: One per-job override applies uniformly to all 14 released commands, decoupling it from the total-worker cap.
    /// Kind: policy
    /// Not a type: The override is a single Option<u32> on `EffectiveConfig`; that it fans out to every command rather than interacting with `max_total_workers` is a legacy-compatibility rule a type cannot express.
    /// State: None. One override and one declared host cover the whole released-command set in memory.
    /// Concurrency: None. The resolved config is read-only across the loop.
    #[test]
    fn max_workers_per_job_override_applies_to_every_command() {
        let facts = apple_silicon_64gb();
        let overrides = ConfigOverrides {
            max_workers_per_job: Some(5),
            ..Default::default()
        };
        let effective = EffectiveConfig::resolve(&overrides, &facts);
        for command in ReleasedCommand::ALL {
            assert_eq!(
                effective.max_workers_per_job(&command),
                5,
                "override should apply to {command:?}"
            );
        }
    }

    /// Guards: Per-job lookup works on a command built from its name, not only on the released-command array constant.
    /// Kind: wire
    /// Not a type: `ReleasedCommand::try_from` is the untrusted-name admission edge; a type cannot prove the resolved lookup accepts that admitted value as the array member does.
    /// State: None. The command is parsed from a string literal; no process or host is read.
    /// Concurrency: None. All values are owned by this call.
    #[test]
    fn max_workers_per_job_lookup_by_arbitrary_instance() {
        let facts = apple_silicon_64gb();
        let effective = EffectiveConfig::resolve(&ConfigOverrides::default(), &facts);
        let value = effective.max_workers_per_job(&cmd("transcribe"));
        assert!(value >= 1);
    }

    /// Guards: Resolving the same overrides and facts twice produces identical effective values and per-job results.
    /// Kind: contract
    /// Not a type: `EffectiveConfig` is a pure merge with no interior state, but the compiler cannot prove that equal inputs yield equal resolved knobs; this pins determinism for downstream caching.
    /// State: None. Two in-memory resolutions over one declared host.
    /// Concurrency: None. Both resolutions are serial and own their values.
    #[test]
    fn resolve_is_pure() {
        let facts = apple_silicon_64gb();
        let overrides = ConfigOverrides {
            gpu_thread_pool_size: Some(2),
            ..Default::default()
        };
        let a = EffectiveConfig::resolve(&overrides, &facts);
        let b = EffectiveConfig::resolve(&overrides, &facts);
        assert_eq!(a.gpu_thread_pool_size, b.gpu_thread_pool_size);
        assert_eq!(a.force_cpu, b.force_cpu);
        assert_eq!(a.max_total_workers, b.max_total_workers);
        assert_eq!(
            a.max_workers_per_job(&cmd("transcribe")),
            b.max_workers_per_job(&cmd("transcribe"))
        );
    }
}
