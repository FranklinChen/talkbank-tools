//! Config-vs-host-facts validation.
//!
//! At server startup, after [`super::HostSnapshot`] has merged operator
//! overrides with the host-facts recommendation,
//! [`validate`] inspects the *original* operator overrides against the
//! detected facts and reports two kinds of finding:
//!
//! - [`ConfigWarning`]: the override is suboptimal but the server can
//!   still run. Surfaced as `tracing::warn!` lines at startup so the
//!   operator notices, but never blocks. Conservative-vs-recommendation
//!   mismatches (operator over-conservative: fewer workers than the
//!   host could support) are intentionally **silent**: the operator
//!   knows their host better than `recommend()` does, and silence is
//!   the right ergonomics for "intentionally cautious."
//! - [`ConfigError`]: the override would deterministically crash or
//!   produce wrong output (e.g., per-job worker × concurrent jobs ×
//!   peak RAM exceeds physical RAM). Reserved for cases where running
//!   would clearly fail; the server refuses to start with a message
//!   that includes the recommendation.
//!
//! Layer 4 warns rather than rejects for most findings; see
//! `book/src/batchalign/developer/host-facts.md`.
//!
//! ## Why not use `EffectiveConfig` directly
//!
//! Validation needs to know which knobs the operator *explicitly set*
//! versus which fell through to the recommendation, because the
//! "operator contradicts facts" warnings only fire on explicit
//! overrides. `EffectiveConfig` has already merged the two, so the
//! distinction is gone there. `validate` reads from `ServerConfig`
//! (where `Some(_)` means explicit, `None` means deferred to
//! recommendation) plus `HostFacts` directly.

use std::fmt;

use super::recommendations::{
    recommend_force_cpu, recommend_max_concurrent_jobs, recommend_max_total_workers,
    worst_case_per_job_peak_ram_mb,
};
use super::{GpuPresence, HostFacts};
use crate::config::ServerConfig;

/// One non-fatal finding from [`validate`].
///
/// Each variant carries the configured value, the relevant detected
/// fact, and the recommended alternative, the variant alone is enough
/// to render a self-explaining operator message via the [`Display`]
/// impl. New variants must follow the same shape so tracing
/// integration stays uniform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigWarning {
    /// Operator set `gpu_thread_pool_size` greater than 1 on a host
    /// whose GPU is not functional for batchalign (Apple Silicon MPS,
    /// no GPU at all, CUDA host with `device_count == 0`). Above-1 has
    /// no benefit on CPU-only execution and adds in-process contention
    /// without any parallelism gain.
    GpuThreadPoolSizeAboveOneOnCpu {
        /// The operator's configured value.
        configured: u32,
        /// The recommended value for this host (always 1 when the GPU
        /// is non-functional, so the message is self-explaining).
        recommended: u32,
    },
    /// Operator set `max_concurrent_jobs` higher than the host-facts
    /// recommendation derived from `ram_total_mb` and CPU
    /// availability. The configured value will admit more parallel
    /// jobs than the host can comfortably support, increasing the
    /// risk of memory-pressure stalls and worker OOMs.
    ///
    /// Conservative-vs-recommendation (configured `<=` recommended)
    /// is intentionally silent; only over-eager configurations
    /// warn.
    MaxConcurrentJobsAboveRamBudget {
        /// The operator's configured value.
        configured: u32,
        /// The recommended value for this host (derived from
        /// `recommend_max_concurrent_jobs`).
        recommended: u32,
    },
    /// Operator set `max_total_workers` higher than the host-facts
    /// recommendation derived from `ram_total_mb`. The configured
    /// ceiling permits more concurrent worker processes than physical
    /// RAM can support, increasing the risk of OOM under sustained
    /// load. Conservative-vs-recommendation is silent.
    MaxTotalWorkersAboveRamBudget {
        /// The operator's configured value.
        configured: u32,
        /// The recommended value for this host (derived from
        /// `recommend_max_total_workers`).
        recommended: u32,
    },
    /// Operator set `force_cpu = false` on a host whose GPU is not
    /// functional for batchalign. The configured value asserts
    /// "use the GPU" but the GPU pipeline cannot proceed, the
    /// worker will still fall back to CPU at runtime, but the
    /// asserted intent is wrong. Common cause: an old `server.yaml`
    /// from a CUDA host copied to an Apple Silicon host.
    ForceCpuFalseOnNonFunctionalGpu {
        /// The detected GPU presence (so the operator sees what
        /// "non-functional" means in their case). Carried as the
        /// typed enum rather than a Debug-formatted string so JSON
        /// consumers see structured data and the Display impl can
        /// format on demand.
        gpu: GpuPresence,
    },
}

impl fmt::Display for ConfigWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GpuThreadPoolSizeAboveOneOnCpu {
                configured,
                recommended,
            } => write!(
                f,
                "gpu_thread_pool_size={configured} on a host with no functional GPU; \
                 recommended value is {recommended}. The configured threads will \
                 contend for one CPU-bound model process without parallelism gain. \
                 Set `gpu_thread_pool_size: {recommended}` in server.yaml or omit \
                 the field to use the host-aware recommendation.",
            ),
            Self::MaxConcurrentJobsAboveRamBudget {
                configured,
                recommended,
            } => write!(
                f,
                "max_concurrent_jobs={configured} exceeds the host-facts \
                 recommendation of {recommended}. The configured value admits \
                 more parallel jobs than this host can comfortably support; \
                 expect memory-pressure stalls and worker OOMs under load. \
                 Set `max_concurrent_jobs: {recommended}` in server.yaml or \
                 omit the field to use the host-aware recommendation.",
            ),
            Self::MaxTotalWorkersAboveRamBudget {
                configured,
                recommended,
            } => write!(
                f,
                "max_total_workers={configured} exceeds the host-facts \
                 recommendation of {recommended} (derived from ram_total_mb / \
                 6 GB per worker, clamped to [2, 32]). The configured ceiling \
                 permits more concurrent worker processes than physical RAM \
                 can support; expect OOMs under sustained load. Set \
                 `max_total_workers: {recommended}` in server.yaml or omit \
                 the field to use the host-aware recommendation.",
            ),
            Self::ForceCpuFalseOnNonFunctionalGpu { gpu } => write!(
                f,
                "force_cpu=false is set, but this host's GPU is not \
                 functional for batchalign ({gpu:?}). The asserted \
                 intent is wrong: the worker will fall back to CPU at \
                 runtime regardless. Remove `force_cpu` from server.yaml \
                 (the host-facts recommendation will set it correctly) or \
                 set `force_cpu: true` to make the intent explicit.",
            ),
        }
    }
}

/// A fatal finding from [`validate`]. Server refuses to start.
///
/// Errors are reserved for cases where the configuration would
/// deterministically crash or produce wrong output. The plan calls
/// for further variants (e.g., per-command peak-RAM checks) as the
/// host-facts model gains richer fact shapes; today's single variant
/// catches the most common deploy-time mistake, over-eager
/// `max_concurrent_jobs` on a low-RAM host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    /// Operator's `max_concurrent_jobs` value, when multiplied by
    /// the worst-case per-job peak RAM (the heaviest worker
    /// profile, today GPU at 16 GB), exceeds the host's physical
    /// RAM. There is no scheduling outcome that fits, every
    /// jobset that uses the heaviest profile would OOM. The
    /// server refuses to start; the operator must drop
    /// `max_concurrent_jobs` (or omit it to use the recommendation,
    /// which is by construction safe) before the daemon can boot.
    MaxConcurrentJobsWouldDeterministicallyOom {
        /// The operator's configured value.
        configured: u32,
        /// Detected total RAM in MB.
        ram_total_mb: u64,
        /// Worst-case per-job peak RAM (in MB) used in the check.
        per_job_peak_mb: u64,
    },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MaxConcurrentJobsWouldDeterministicallyOom {
                configured,
                ram_total_mb,
                per_job_peak_mb,
            } => write!(
                f,
                "max_concurrent_jobs={configured} would deterministically \
                 OOM this host: worst-case per-job peak RAM is \
                 {per_job_peak_mb} MB and {configured} × {per_job_peak_mb} \
                 = {} MB exceeds the detected ram_total_mb of \
                 {ram_total_mb}. Drop `max_concurrent_jobs` from \
                 server.yaml (the host-facts recommendation is by \
                 construction safe) or set a value such that \
                 max_concurrent_jobs × {per_job_peak_mb} <= {ram_total_mb}. \
                 To run a single job locally without the daemon's memory \
                 gate at all, invoke `batchalign3 <command> --sequential` \
                 - `--sequential` runs the command in-process with no \
                 daemon, no concurrent-job admission, and no memory gate.",
                u64::from(*configured) * per_job_peak_mb,
            ),
        }
    }
}

/// The findings of one [`validate`] call: advisory warnings, which either
/// outcome may carry, and the startup verdict.
///
/// There is deliberately no `Default`, because a default verdict would
/// silently mean "admitted".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigValidation {
    /// Non-fatal findings; surface but do not block startup. May be empty.
    pub warnings: Vec<ConfigWarning>,
    /// Whether startup may proceed.
    pub verdict: StartupVerdict,
}

/// Startup is either admitted or refused, and a refusal carries the finding
/// that caused it.
///
/// A closed pair rather than an error list plus a `has_errors` query: a
/// caller deciding whether to boot must match on it, so "forgot to check for
/// errors and started anyway" does not compile, and an empty refusal cannot
/// be built.
///
/// `Refused` carries exactly one [`ConfigError`] because [`validate`] has
/// exactly one fatal rule. A second fatal rule changes this payload, and the
/// compiler then walks every consumer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StartupVerdict {
    /// Startup may proceed.
    Admitted,
    /// Startup must not proceed, for this reason.
    Refused(ConfigError),
}

/// Inspect operator overrides in `cfg` against detected `facts` and
/// report any contradictions.
///
/// The function is pure: same `(cfg, facts)` always produces the same
/// `ConfigValidation`. No I/O, no time-dependence, no logging. Callers
/// are responsible for converting findings to `tracing` lines or
/// startup errors.
#[must_use]
pub fn validate(cfg: &ServerConfig, facts: &HostFacts) -> ConfigValidation {
    let mut warnings = Vec::new();
    // Admitted unless the one fatal rule below refuses.
    let mut verdict = StartupVerdict::Admitted;

    if let Some(configured) = cfg.gpu_thread_pool_size
        && configured > 1
        && !facts.gpu.is_functional_for_batchalign()
    {
        warnings.push(ConfigWarning::GpuThreadPoolSizeAboveOneOnCpu {
            configured,
            recommended: 1,
        });
    }

    if let Some(configured) = cfg.max_concurrent_jobs {
        let recommended = recommend_max_concurrent_jobs(facts);
        if configured > recommended {
            warnings.push(ConfigWarning::MaxConcurrentJobsAboveRamBudget {
                configured,
                recommended,
            });
        }
        // Worst-case per-job × N > ram is the deterministic-OOM
        // case: even if every job uses the heaviest worker profile,
        // physical RAM cannot accommodate. This is an error
        // (refuses startup) rather than a warning because the
        // operator has no scheduling option that fits, the
        // configured cap is incorrect by construction.
        let tier = crate::types::runtime::MemoryTier::from_total_mb(facts.ram_total_mb);
        let per_job_peak_mb = worst_case_per_job_peak_ram_mb(&tier);
        if u64::from(configured) * per_job_peak_mb > facts.ram_total_mb {
            verdict =
                StartupVerdict::Refused(ConfigError::MaxConcurrentJobsWouldDeterministicallyOom {
                    configured,
                    ram_total_mb: facts.ram_total_mb,
                    per_job_peak_mb,
                });
        }
    }

    if let Some(configured) = cfg.max_total_workers {
        let recommended = recommend_max_total_workers(facts);
        if configured > recommended {
            warnings.push(ConfigWarning::MaxTotalWorkersAboveRamBudget {
                configured,
                recommended,
            });
        }
    }

    // `force_cpu = Some(false)` on a host whose recommendation is
    // `true` (non-functional GPU) is the contradiction to flag.
    // `Some(true)` on a CUDA host is the symmetric case but it's
    // operationally fine: the worker just doesn't use the GPU,
    // which is a legitimate testing affordance, so it's silent.
    // `None` defers to the recommendation by construction; never
    // contradicts.
    if cfg.force_cpu == Some(false) && recommend_force_cpu(facts) {
        warnings.push(ConfigWarning::ForceCpuFalseOnNonFunctionalGpu {
            gpu: facts.gpu.clone(),
        });
    }

    ConfigValidation { warnings, verdict }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host_facts::test_helpers::{apple_silicon_64gb, linux_cuda_24gb};
    // Test concurrency pure-validate: None. `validate` and the `Display` impls are pure functions of in-memory values, so these tests share no resource and can run in any order or in parallel.

    /// Guards: Operator overrides above one on a non-functional GPU warn; matching, functional-GPU and deferred values stay silent.
    /// Kind: policy
    /// Not a type: Each override value is individually valid in `ServerConfig`; only its combination with external host facts is contradictory, so the admitted `Option<u32>` and detected GPU class remain a policy matrix.
    /// State: None. Two declared in-memory host shapes and four explicit `Option<u32>` values; no daemon starts.
    /// Concurrency: [pure-validate]
    #[test]
    fn gpu_thread_pool_size_warning_policy_matrix() {
        let cfg = |gpu_thread_pool_size| ServerConfig {
            gpu_thread_pool_size,
            ..Default::default()
        };
        let cases: Vec<(&str, ServerConfig, HostFacts, Vec<ConfigWarning>)> = vec![
            (
                "above one on apple",
                cfg(Some(4)),
                apple_silicon_64gb(),
                vec![ConfigWarning::GpuThreadPoolSizeAboveOneOnCpu {
                    configured: 4,
                    recommended: 1,
                }],
            ),
            (
                "matches recommendation on cpu",
                cfg(Some(1)),
                apple_silicon_64gb(),
                vec![],
            ),
            ("above one on cuda", cfg(Some(4)), linux_cuda_24gb(), vec![]),
            ("deferred on cpu", cfg(None), apple_silicon_64gb(), vec![]),
        ];
        for (label, cfg, facts, expected) in cases {
            let result = validate(&cfg, &facts);
            assert_eq!(result.warnings, expected, "{label} warnings");
            assert!(
                result.verdict == StartupVerdict::Admitted,
                "{label} must not refuse startup"
            );
        }
    }

    // -----------------------------------------------------------------
    // MaxConcurrentJobsAboveRamBudget
    // -----------------------------------------------------------------

    /// Guards: Only an explicit `max_concurrent_jobs` above the host-derived recommendation warns; equal, below and deferred values stay silent.
    /// Kind: policy
    /// Not a type: `Option<u32>` already distinguishes deferred from explicit overrides, but "explicit and over budget" is a semantic comparison with external RAM and CPU facts, not a state a type can forbid.
    /// State: None. One declared in-memory 64 GB host and four `max_concurrent_jobs` override states; no job is admitted.
    /// Concurrency: [pure-validate]
    #[test]
    fn max_concurrent_jobs_warning_policy_matrix() {
        let facts = apple_silicon_64gb();
        let recommended = super::recommend_max_concurrent_jobs(&facts);
        let cfg = |max_concurrent_jobs| ServerConfig {
            max_concurrent_jobs,
            ..Default::default()
        };
        let cases: Vec<(&str, ServerConfig, Vec<ConfigWarning>)> = vec![
            (
                "above recommendation",
                cfg(Some(99)),
                vec![ConfigWarning::MaxConcurrentJobsAboveRamBudget {
                    configured: 99,
                    recommended,
                }],
            ),
            ("equals recommendation", cfg(Some(recommended)), vec![]),
            ("below recommendation", cfg(Some(1)), vec![]),
            ("deferred", cfg(None), vec![]),
        ];
        for (label, cfg, expected) in cases {
            let result = validate(&cfg, &facts);
            assert_eq!(result.warnings, expected, "{label} warnings");
        }
    }

    // -----------------------------------------------------------------
    // MaxTotalWorkersAboveRamBudget
    // -----------------------------------------------------------------

    /// Guards: Only an explicit `max_total_workers` above the RAM-derived recommendation warns; equal, below and deferred values stay silent.
    /// Kind: policy
    /// Not a type: `Option<u32>` distinguishes deferral, but "explicit and over budget" remains a semantic comparison with external RAM facts, so the four admitted override states stay a policy matrix.
    /// State: None. One declared in-memory 64 GB host and four `max_total_workers` override states; no worker process starts.
    /// Concurrency: [pure-validate]
    #[test]
    fn max_total_workers_warning_policy_matrix() {
        let facts = apple_silicon_64gb();
        let recommended = super::recommend_max_total_workers(&facts);
        let cfg = |max_total_workers| ServerConfig {
            max_total_workers,
            ..Default::default()
        };
        let cases: Vec<(&str, ServerConfig, Vec<ConfigWarning>)> = vec![
            (
                "above recommendation",
                cfg(Some(64)),
                vec![ConfigWarning::MaxTotalWorkersAboveRamBudget {
                    configured: 64,
                    recommended,
                }],
            ),
            ("equals recommendation", cfg(Some(recommended)), vec![]),
            ("below recommendation", cfg(Some(2)), vec![]),
            ("deferred", cfg(None), vec![]),
        ];
        for (label, cfg, expected) in cases {
            let result = validate(&cfg, &facts);
            assert_eq!(result.warnings, expected, "{label} warnings");
        }
    }

    // -----------------------------------------------------------------
    // ForceCpuFalseOnNonFunctionalGpu
    // -----------------------------------------------------------------

    /// Guards: Only explicit `force_cpu = false` on a host whose inferred value is `true` warns; matching and deferred values stay silent.
    /// Kind: policy
    /// Not a type: `Option<bool>` admits all three operator states; the contradiction depends on an external GPU fact and the deliberate choice that `true` on CUDA is a legitimate testing affordance.
    /// State: None. Two declared in-memory host shapes and three explicit `Option<bool>` values; no worker is dispatched.
    /// Concurrency: [pure-validate]
    #[test]
    fn force_cpu_warning_policy_matrix() {
        let cfg = |force_cpu| ServerConfig {
            force_cpu,
            ..Default::default()
        };
        let cases: Vec<(&str, ServerConfig, HostFacts, Vec<ConfigWarning>)> = vec![
            (
                "false on apple",
                cfg(Some(false)),
                apple_silicon_64gb(),
                vec![ConfigWarning::ForceCpuFalseOnNonFunctionalGpu {
                    gpu: apple_silicon_64gb().gpu,
                }],
            ),
            ("false on cuda", cfg(Some(false)), linux_cuda_24gb(), vec![]),
            (
                "true on apple",
                cfg(Some(true)),
                apple_silicon_64gb(),
                vec![],
            ),
            (
                "true on cuda is a testing affordance",
                cfg(Some(true)),
                linux_cuda_24gb(),
                vec![],
            ),
            ("deferred", cfg(None), apple_silicon_64gb(), vec![]),
        ];
        for (label, cfg, facts, expected) in cases {
            let result = validate(&cfg, &facts);
            assert_eq!(result.warnings, expected, "{label} warnings");
        }
    }

    // -----------------------------------------------------------------
    // Multi-variant composition
    // -----------------------------------------------------------------

    // -----------------------------------------------------------------
    // MaxConcurrentJobsWouldDeterministicallyOom (the first ConfigError)
    // -----------------------------------------------------------------

    // ---------------------------------------------------------------
    // 16 GB laptop / GHA-runner UX contract.
    //
    // The 2026-05-08 rearch ("memory-gate-landed", "idle-eviction-pre-
    // pass-landed", "cpu-loadavg-gate-landed", "cap-at-4-retired") was
    // supposed to make memory gating dynamic, observe actual RSS,
    // gate at admission, evict under pressure. The runtime gates land
    // that contract, but the boot-time host-facts validator was left
    // command-agnostic: it compares `max_concurrent_jobs × 16000 MB`
    // (a hardcoded GPU-worker peak) against `ram_total_mb`, regardless
    // of which commands the operator will actually run.
    //
    // On a 16 GB laptop running a default `uv tool install batchalign3`
    // for morphotag, that gate refuses startup: `1 × 16000 > 15989`.
    // The user sees no signal that `--sequential` would let them run
    // the same job locally. Tests below pin both halves of the fix:
    //
    // 1. The default 16 GB laptop must accept startup (no operator
    //    `max_concurrent_jobs` override); the recommendation by
    //    construction must be safe.
    // 2. Whenever the gate DOES refuse startup (legitimate
    //    over-subscription on any tier), the error must point the
    //    operator at the `--sequential` workaround.
    // ---------------------------------------------------------------

    use super::super::test_helpers::laptop_16gb;

    /// Guards: A 16 GB laptop accepts the default and an explicitly conservative job count but still refuses genuine oversubscription.
    /// Kind: boundary
    /// Not a type: Laptop RAM and operator job count are independently admitted values; only their product against the external tier peak is unsafe, so neither `Option<u32>` nor the facts type can forbid the bad combination.
    /// State: None. One declared in-memory laptop fixture and three override states; no daemon starts.
    /// Concurrency: [pure-validate]
    #[test]
    fn laptop_16gb_startup_policy_matrix() {
        let cfg = |max_concurrent_jobs| ServerConfig {
            max_concurrent_jobs,
            ..Default::default()
        };
        let facts = laptop_16gb();
        let cases: Vec<(&str, ServerConfig, bool)> = vec![
            ("default", ServerConfig::default(), false),
            ("explicit one", cfg(Some(1)), false),
            ("oversubscribed eight", cfg(Some(8)), true),
        ];
        for (label, cfg, expect_error) in cases {
            let result = validate(&cfg, &facts);
            assert_eq!(
                matches!(result.verdict, StartupVerdict::Refused(_)),
                expect_error,
                "{label} startup verdict"
            );
        }
    }

    // -----------------------------------------------------------------
    // Composition, the fatal boundary, and the operator-facing words
    // -----------------------------------------------------------------

    /// The refusal inside a verdict, for tables that judge only the fatal
    /// rule and leave the warning rules to their own matrices.
    fn refusal(validation: &ConfigValidation) -> Option<&ConfigError> {
        match &validation.verdict {
            StartupVerdict::Admitted => None,
            StartupVerdict::Refused(error) => Some(error),
        }
    }

    /// Guards: Every rule runs independently and in rule order: one config tripping three warning rules and the fatal rule yields exactly those three warnings plus the refusal, and the default config is an empty admission on every fixture host.
    /// Kind: policy
    /// Not a type: The rules read disjoint `ServerConfig` fields, but whether one finding suppresses or reorders another is a behavioural choice inside `validate`, observable only by running a combination.
    /// State: None. Three declared in-memory host fixtures; no daemon starts.
    /// Concurrency: [pure-validate]
    #[test]
    fn rules_compose_independently_and_default_is_clean() {
        let facts = apple_silicon_64gb();
        let cfg = ServerConfig {
            gpu_thread_pool_size: Some(8),
            max_concurrent_jobs: Some(99),
            max_total_workers: Some(64),
            ..Default::default()
        };
        // 99 jobs is above the warning budget and, at a 16,000 MB worst-case
        // peak, also far beyond the 65,536 MB of RAM: one value trips a
        // warning rule and the fatal rule at once.
        assert_eq!(
            validate(&cfg, &facts),
            ConfigValidation {
                verdict: StartupVerdict::Refused(
                    ConfigError::MaxConcurrentJobsWouldDeterministicallyOom {
                        configured: 99,
                        ram_total_mb: 65_536,
                        per_job_peak_mb: 16_000,
                    }
                ),
                warnings: vec![
                    ConfigWarning::GpuThreadPoolSizeAboveOneOnCpu {
                        configured: 8,
                        recommended: 1,
                    },
                    ConfigWarning::MaxConcurrentJobsAboveRamBudget {
                        configured: 99,
                        recommended: super::recommend_max_concurrent_jobs(&facts),
                    },
                    ConfigWarning::MaxTotalWorkersAboveRamBudget {
                        configured: 64,
                        recommended: super::recommend_max_total_workers(&facts),
                    },
                ],
            }
        );

        for (label, facts) in [
            ("apple silicon", apple_silicon_64gb()),
            ("cuda", linux_cuda_24gb()),
            ("16 gb laptop", laptop_16gb()),
        ] {
            assert_eq!(
                validate(&ServerConfig::default(), &facts),
                ConfigValidation {
                    warnings: vec![],
                    verdict: StartupVerdict::Admitted,
                },
                "default config on {label}"
            );
        }
    }

    /// Guards: The fatal rule refuses exactly when configured jobs times the worst-case peak strictly exceeds detected RAM (an exact fit is admitted, one megabyte short is refused), and the refusal carries the configured value, the RAM and the peak it used; deferral and the recommended value are admitted.
    /// Kind: measurement
    /// Not a type: `Option<u32>` admits the override and the closed error variant carries its values, but the strict u64 product comparison against external RAM and the tier peak is arithmetic no type pins.
    /// State: None. Three declared in-memory host shapes (two derived from the 64 GB fixture by RAM alone) and five override states; no daemon starts.
    /// Concurrency: [pure-validate]
    #[test]
    fn max_concurrent_jobs_oom_boundary_matrix() {
        let cfg = |max_concurrent_jobs| ServerConfig {
            max_concurrent_jobs,
            ..Default::default()
        };
        let with_ram = |ram_total_mb| HostFacts {
            ram_total_mb,
            ..apple_silicon_64gb()
        };
        let oom =
            |configured, ram_total_mb| ConfigError::MaxConcurrentJobsWouldDeterministicallyOom {
                configured,
                ram_total_mb,
                per_job_peak_mb: 16_000,
            };
        let cuda = linux_cuda_24gb();
        let recommended = super::recommend_max_concurrent_jobs(&cuda);
        let cases: Vec<(&str, ServerConfig, HostFacts, Option<ConfigError>)> = vec![
            (
                "four jobs exactly fill 64,000 MB",
                cfg(Some(4)),
                with_ram(64_000),
                None,
            ),
            (
                "four jobs on 63,999 MB are one short",
                cfg(Some(4)),
                with_ram(63_999),
                Some(oom(4, 63_999)),
            ),
            (
                "five jobs on 65,536 MB",
                cfg(Some(5)),
                apple_silicon_64gb(),
                Some(oom(5, 65_536)),
            ),
            ("deferred", cfg(None), apple_silicon_64gb(), None),
            ("recommended value", cfg(Some(recommended)), cuda, None),
        ];
        for (label, cfg, facts, expected) in cases {
            let verdict = validate(&cfg, &facts);
            assert_eq!(refusal(&verdict), expected.as_ref(), "{label}");
        }
    }

    /// Guards: Every warning and error variant renders a sentence naming the configured value, the detected fact or recommendation it contradicts, and the remedy, including the `--sequential` route out of a refused startup.
    /// Kind: transcript
    /// Not a type: Each variant carries its values as typed fields, but whether the rendered sentence actually names them and the remedy is free text for a human operator; only the rendered words can be checked.
    /// State: None. One value of each variant built in memory.
    /// Concurrency: [pure-validate]
    #[test]
    fn every_finding_renders_its_values_and_remedy() {
        let cases: Vec<(&str, String, &[&str])> = vec![
            (
                "gpu thread pool",
                ConfigWarning::GpuThreadPoolSizeAboveOneOnCpu {
                    configured: 4,
                    recommended: 1,
                }
                .to_string(),
                &[
                    "gpu_thread_pool_size=4",
                    "recommended value is 1",
                    "`gpu_thread_pool_size: 1`",
                ],
            ),
            (
                "concurrent jobs",
                ConfigWarning::MaxConcurrentJobsAboveRamBudget {
                    configured: 12,
                    recommended: 4,
                }
                .to_string(),
                &[
                    "max_concurrent_jobs=12",
                    "recommendation of 4",
                    "`max_concurrent_jobs: 4`",
                ],
            ),
            (
                "total workers",
                ConfigWarning::MaxTotalWorkersAboveRamBudget {
                    configured: 64,
                    recommended: 10,
                }
                .to_string(),
                &[
                    "max_total_workers=64",
                    "recommendation of 10",
                    "`max_total_workers: 10`",
                ],
            ),
            (
                "force cpu",
                ConfigWarning::ForceCpuFalseOnNonFunctionalGpu {
                    gpu: GpuPresence::AppleMps {
                        functional_for_batchalign: false,
                        reason_excluded: None,
                    },
                }
                .to_string(),
                &["force_cpu=false", "AppleMps", "`force_cpu: true`"],
            ),
            (
                "deterministic oom",
                ConfigError::MaxConcurrentJobsWouldDeterministicallyOom {
                    configured: 8,
                    ram_total_mb: 32_000,
                    per_job_peak_mb: 16_000,
                }
                .to_string(),
                &[
                    "max_concurrent_jobs=8",
                    "32000",
                    "16000",
                    "= 128000 MB",
                    "Drop `max_concurrent_jobs`",
                    "--sequential",
                ],
            ),
        ];
        for (label, rendered, required) in cases {
            for needle in required {
                assert!(
                    rendered.contains(needle),
                    "{label} must mention {needle:?}: {rendered}"
                );
            }
        }
    }
}
