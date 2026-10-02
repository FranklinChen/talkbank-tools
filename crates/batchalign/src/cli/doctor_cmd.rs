//! `batchalign3 doctor`: pre-flight diagnostic for the worker pipeline.
//!
//! Spawns a test worker, sends known inputs through the morphosyntax
//! pipeline, and validates the output structure. Catches machine-specific
//! issues (stale models, missing processors, MWT quirks) before they
//! become production failures.

use crate::cli::args::DoctorArgs;
use crate::cli::error::CliError;
use crate::cli::python::resolve_python_executable;

use crate::config::{self, RuntimeLayout};
use crate::host_facts::{
    self, EffectiveConfig, HostFacts, HostFactsSource, RealHostFactsSource, RecommendedKnobs,
    recommend,
};

use crate::media::tools::MediaTool;
use std::io::{BufRead, Write};
use std::ops::Not;
use std::process::{Command, Stdio};
use std::time::Instant;

/// Result of a single diagnostic check.
#[derive(Debug, serde::Serialize)]
struct CheckResult {
    name: String,
    status: CheckStatus,
    detail: String,
    /// How long the check took; `None` for a check that is a reading rather
    /// than a timed probe (memory, the config load), shown as untimed and
    /// written as `null`, never as a measured `0`.
    duration_ms: Option<u64>,
}

/// Outcome of a diagnostic check.
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum CheckStatus {
    Pass,
    Fail,
    /// Reserved for checks that are not applicable to the current environment.
    #[allow(dead_code)]
    Skip,
}

impl std::fmt::Display for CheckStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pass => write!(f, "PASS"),
            Self::Fail => write!(f, "FAIL"),
            Self::Skip => write!(f, "SKIP"),
        }
    }
}

/// Aggregated host-facts view for doctor output.
///
/// Bundles the raw [`HostFacts`] snapshot, the resolved
/// [`EffectiveConfig`] knobs that will govern this host's runtime,
/// and any validation findings. Serialized verbatim under the
/// `host_facts` key in JSON mode; rendered as a section in human
/// mode. The schema is the operator's contract for `--format json`,
/// so adding fields is backwards-compatible but renaming or
/// removing them is not.
#[derive(Debug, serde::Serialize)]
struct HostFactsReport {
    detected: HostFacts,
    effective: EffectiveConfigSummary,
    validation: ValidationReport,
}

/// JSON-friendly summary of the resolved config knobs. The flat
/// scalar fields (`gpu_thread_pool_size`, `force_cpu`, …) project
/// [`EffectiveConfig`]'s resolved values; the `audit` sibling adds
/// per-knob `recommendation` + `operator_override` + `redundant`
/// detail so operators can identify `server.yaml` overrides that are
/// no-ops relative to the host-facts recommendation
/// (`redundant: true` entries are deletion candidates). Constructing
/// the audit requires the original [`ServerConfig`] and [`HostFacts`]
/// in addition to the resolved [`EffectiveConfig`].
#[derive(Debug, serde::Serialize)]
struct EffectiveConfigSummary {
    gpu_thread_pool_size: u32,
    force_cpu: bool,
    max_total_workers: u32,
    max_concurrent_jobs: u32,
    memory_gate_mb: u64,
    max_workers_per_key_gpu: u32,
    max_workers_per_key_stanza: u32,
    max_workers_per_key_io: u32,
    /// Per-knob recommendation/override/redundancy detail. Used by
    /// operators auditing whether each `server.yaml` override is
    /// still load-bearing relative to the live host-facts
    /// recommendation.
    audit: ConfigAudit,
}

impl EffectiveConfigSummary {
    /// Build the summary from the resolved [`EffectiveConfig`] plus
    /// the original [`ServerConfig`] and [`HostFacts`] needed to
    /// reconstruct the per-knob audit (recommended / override /
    /// redundant flags).
    fn from_effective(
        e: &EffectiveConfig,
        cfg: &crate::config::ServerConfig,
        facts: &HostFacts,
    ) -> Self {
        Self {
            gpu_thread_pool_size: e.gpu_thread_pool_size,
            force_cpu: e.force_cpu,
            max_total_workers: e.max_total_workers,
            max_concurrent_jobs: e.max_concurrent_jobs,
            memory_gate_mb: e.memory_gate_mb.0,
            max_workers_per_key_gpu: e.max_workers_per_key_by_profile.gpu,
            max_workers_per_key_stanza: e.max_workers_per_key_by_profile.stanza,
            max_workers_per_key_io: e.max_workers_per_key_by_profile.io,
            audit: ConfigAudit::build(cfg, facts),
        }
    }
}

/// Per-knob trio of (effective, recommended, operator_override) plus
/// a `redundant` flag set when the override is present and equal to
/// the recommendation. The flag is the operator's deletion signal:
/// `redundant: true` means the override can be removed from
/// `server.yaml` (or `fleet-inventory.yml`) without changing
/// behavior.
#[derive(Debug, serde::Serialize)]
struct KnobAudit<T: Clone + PartialEq + serde::Serialize> {
    effective: T,
    recommended: T,
    /// `None` when the operator did not override this knob.
    #[serde(rename = "override")]
    operator_override: Option<T>,
    /// True iff `operator_override == Some(recommended)`, i.e. the
    /// override is exactly what the recommender would produce, so
    /// removing it is a no-op.
    redundant: bool,
}

impl<T: Clone + PartialEq + serde::Serialize> KnobAudit<T> {
    fn new(recommended: T, operator_override: Option<T>) -> Self {
        let effective = operator_override
            .clone()
            .unwrap_or_else(|| recommended.clone());
        let redundant = matches!(&operator_override, Some(v) if *v == recommended);
        Self {
            effective,
            recommended,
            operator_override,
            redundant,
        }
    }
}

/// Per-knob audit bundle. Field names mirror the flat
/// `EffectiveConfigSummary` knobs so a JSON consumer can navigate
/// directly from `effective.<knob>` to `effective.audit.<knob>` for
/// the override-vs-recommendation detail.
#[derive(Debug, serde::Serialize)]
struct ConfigAudit {
    gpu_thread_pool_size: KnobAudit<u32>,
    force_cpu: KnobAudit<bool>,
    max_total_workers: KnobAudit<u32>,
    max_concurrent_jobs: KnobAudit<u32>,
    memory_gate_mb: KnobAudit<u64>,
    /// `max_workers_per_key` is the one knob where `ServerConfig`
    /// carries a single uniform value but the recommender produces a
    /// per-profile (gpu/stanza/io) shape. The audit reports the
    /// uniform-override case using the gpu profile's recommended
    /// value as the canonical comparison point; if the operator's
    /// uniform override does not equal *all three* recommended
    /// profile values simultaneously, `redundant` is false.
    max_workers_per_key: KnobAudit<u32>,
}

impl ConfigAudit {
    fn build(cfg: &crate::config::ServerConfig, facts: &HostFacts) -> Self {
        let r: RecommendedKnobs = recommend(facts);
        // `max_workers_per_key` is the uniform-override-vs-per-profile
        // exception: the operator's single value is redundant only
        // when all three profile recommendations coincide AND match it.
        let mwpk_override = cfg.max_workers_per_key;
        let mwpk_rec = &r.max_workers_per_key_by_profile;
        let mwpk_uniform_redundant = mwpk_override
            .is_some_and(|n| n == mwpk_rec.gpu && n == mwpk_rec.stanza && n == mwpk_rec.io);
        let mwpk_audit = KnobAudit {
            effective: mwpk_override.unwrap_or(mwpk_rec.gpu),
            recommended: mwpk_rec.gpu,
            operator_override: mwpk_override,
            redundant: mwpk_uniform_redundant,
        };
        Self {
            gpu_thread_pool_size: KnobAudit::new(r.gpu_thread_pool_size, cfg.gpu_thread_pool_size),
            force_cpu: KnobAudit::new(r.force_cpu, cfg.force_cpu),
            max_total_workers: KnobAudit::new(r.max_total_workers, cfg.max_total_workers),
            max_concurrent_jobs: KnobAudit::new(r.max_concurrent_jobs, cfg.max_concurrent_jobs),
            memory_gate_mb: KnobAudit::new(
                cfg.resolved_memory_tier().headroom_mb.0,
                cfg.memory_gate_mb.map(|m| m.0),
            ),
            max_workers_per_key: mwpk_audit,
        }
    }

    /// Names of knobs whose operator override is redundant, i.e.
    /// equals the recommendation and could be deleted from
    /// `server.yaml` without behavior change.
    fn redundant_knob_names(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.gpu_thread_pool_size.redundant {
            out.push("gpu_thread_pool_size");
        }
        if self.force_cpu.redundant {
            out.push("force_cpu");
        }
        if self.max_total_workers.redundant {
            out.push("max_total_workers");
        }
        if self.max_concurrent_jobs.redundant {
            out.push("max_concurrent_jobs");
        }
        if self.memory_gate_mb.redundant {
            out.push("memory_gate_mb");
        }
        if self.max_workers_per_key.redundant {
            out.push("max_workers_per_key");
        }
        out
    }
}

/// JSON-friendly projection of [`host_facts::ConfigValidation`].
/// Each finding renders as its rich `Display` string so the JSON
/// consumer doesn't have to know the variant tag layout.
#[derive(Debug, serde::Serialize)]
struct ValidationReport {
    warnings: Vec<String>,
    errors: Vec<String>,
}

impl ValidationReport {
    fn from_validation(v: &host_facts::ConfigValidation) -> Self {
        Self {
            warnings: v.warnings.iter().map(|w| w.to_string()).collect(),
            errors: v.errors.iter().map(|e| e.to_string()).collect(),
        }
    }
}

/// Run the doctor command.
pub async fn run(args: &DoctorArgs) -> Result<(), CliError> {
    let mut results: Vec<CheckResult> = Vec::new();

    // `--explain <knob>` traces why one resolved value is what it is
    // (operator override vs. recommendation rule). Implies `--check`
    //: never spawns a worker, never runs the model pipeline.
    if let Some(knob) = &args.explain {
        return run_explain(knob, &args.format);
    }

    // `--check` skips the worker-pipeline phase entirely (no Python
    // spawn, no model load); only host-facts inspection runs. Useful
    // for fast config-sanity verification.
    if args.check {
        return run_host_facts_only(&args.format, args.warnings_as_errors);
    }

    let python = args
        .python
        .clone()
        .unwrap_or_else(resolve_python_executable);

    // --- Check 1: Python availability ---
    let start = Instant::now();
    let python_check = Command::new(&python)
        .args(["-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"])
        .output();

    record(
        &mut results,
        "python",
        start,
        match python_check {
            Ok(output) if output.status.success() => {
                let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
                Ok(format!("{python} -> Python {version}"))
            }
            Ok(output) => Err(format!(
                "Python exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            )),
            Err(e) => Err(format!("Cannot spawn {python}: {e}")),
        },
    );

    // --- Check 2: Worker module importable ---
    let start = Instant::now();
    // Isolated as a real worker is, so this checks the import a worker does.
    let import_check = crate::worker::python::isolated_python(&python)
        .args(["-c", "from batchalign.worker import main; print('ok')"])
        .output();

    record(
        &mut results,
        "worker_import",
        start,
        match import_check {
            Ok(output) if output.status.success() => Ok("batchalign.worker importable".to_owned()),
            Ok(output) => Err(format!(
                "Import failed: {}",
                String::from_utf8_lossy(&output.stderr)
                    .trim()
                    .chars()
                    .take(200)
                    .collect::<String>()
            )),
            Err(e) => Err(format!("Cannot spawn: {e}")),
        },
    );

    let start = Instant::now();
    record(&mut results, "ffmpeg", start, check_ffmpeg());

    let start = Instant::now();
    record(
        &mut results,
        "media_decode",
        start,
        check_media_decode(&python)
            .map(|capability| {
                format!(
                    "decoded {} frames at {} Hz",
                    capability.frames, capability.sample_rate
                )
            })
            .map_err(|fault| fault.explain()),
    );

    let start = Instant::now();
    let echo_check = Worker::spawn(
        &python,
        &[
            "--test-echo",
            "--task",
            "morphosyntax",
            "--lang",
            &args.lang,
        ],
    )
    .and_then(|worker| worker.await_ready(std::time::Duration::from_secs(60)))
    .and_then(|worker| {
        let pid = worker.pid();
        worker
            .shutdown()
            .map(|()| format!("Ready signal received (pid {pid})"))
    });
    record(&mut results, "worker_ready_echo", start, echo_check);

    // --- Check 4: Real morphosyntax worker (loads Stanza model) ---
    let start = Instant::now();
    let morpho_check = talkbank_model::LanguageCode::new(&args.lang)
        .map_err(|e| format!("--lang {:?} is not a language code: {e}", args.lang))
        .and_then(|lang| smoke_batch_items(&lang))
        .and_then(|items| {
            Worker::spawn(&python, &["--task", "morphosyntax", "--lang", &args.lang])
                .and_then(|worker| worker.await_ready(std::time::Duration::from_secs(120)))
                .and_then(|worker| {
                    worker.morphosyntax_batch(
                        &args.lang,
                        &items,
                        std::time::Duration::from_secs(120),
                    )
                })
        });
    record(&mut results, "morphosyntax_smoke", start, morpho_check);

    // --- Check 5: Memory ---
    let mem_info = sysinfo::System::new_all();
    let total_mb = mem_info.total_memory() / (1024 * 1024);
    let available_mb = mem_info.available_memory() / (1024 * 1024);
    results.push(CheckResult {
        name: "memory".into(),
        status: if available_mb >= 4096 {
            CheckStatus::Pass
        } else {
            CheckStatus::Fail
        },
        detail: format!("{available_mb} MB available / {total_mb} MB total"),
        duration_ms: None,
    });

    // --- Server config ---
    // A malformed server.yaml is a failed check, not a silent default: the
    // host-facts report below then uses the defaults, and says so here.
    let config = match load_doctor_server_config() {
        Ok(config) => {
            results.push(CheckResult {
                name: "server_config".into(),
                status: CheckStatus::Pass,
                detail: "server.yaml reads (or is absent: defaults)".into(),
                duration_ms: None,
            });
            config
        }
        Err(error) => {
            results.push(CheckResult {
                name: "server_config".into(),
                status: CheckStatus::Fail,
                detail: format!("{error}; the host-facts report below uses the defaults"),
                duration_ms: None,
            });
            crate::config::ServerConfig::default()
        }
    };

    // --- Host-facts report ---
    let host_facts_report = build_host_facts_report(&config);

    // --- Output ---
    let any_fail = results
        .iter()
        .any(|r| matches!(r.status, CheckStatus::Fail));
    let any_validation_error = host_facts_report.validation.errors.is_empty().not();

    match args.format {
        crate::cli::args::DoctorFormat::Human => {
            for r in &results {
                let icon = match r.status {
                    CheckStatus::Pass => "\u{2713}",
                    CheckStatus::Fail => "\u{2717}",
                    CheckStatus::Skip => "-",
                };
                let timing = match r.duration_ms {
                    Some(ms) => format!(" ({ms} ms)"),
                    None => String::new(),
                };
                eprintln!(
                    "  {icon} [{:>4}] {:25} {}{timing}",
                    r.status, r.name, r.detail
                );
            }
            print_host_facts_human(&host_facts_report);
            if any_fail || any_validation_error {
                eprintln!(
                    "\nSome checks FAILED. Fix the issues above before using this machine for production."
                );
            } else {
                eprintln!("\nAll checks passed.");
            }
        }
        crate::cli::args::DoctorFormat::Json => {
            // Compose a single object so JSON consumers parse one
            // payload instead of two arrays from stdout.
            let payload = serde_json::json!({
                "checks": results,
                "host_facts": host_facts_report,
            });
            let json = serde_json::to_string_pretty(&payload).map_err(|e| {
                CliError::InvalidArgument(format!("JSON serialization failed: {e}"))
            })?;
            println!("{json}");
        }
    }

    if any_fail || any_validation_error {
        Err(CliError::InvalidArgument("doctor checks failed".into()))
    } else {
        Ok(())
    }
}

/// Run host-facts inspection only (the `--check` path). Skips the
/// Python worker-pipeline phase entirely.
///
/// Exit policy:
/// - Errors always fail (`Err`).
/// - Warnings fail iff `warnings_as_errors == true`, the CI-gate
///   posture for zero-warning deployments. Without the flag, warnings
///   are surfaced but the process exits 0; the operator may have
///   intentionally overridden a recommendation.
fn run_host_facts_only(
    format: &crate::cli::args::DoctorFormat,
    warnings_as_errors: bool,
) -> Result<(), CliError> {
    let report = build_host_facts_report(&load_doctor_server_config()?);
    let any_validation_error = !report.validation.errors.is_empty();
    let any_warning = !report.validation.warnings.is_empty();
    match format {
        crate::cli::args::DoctorFormat::Human => {
            print_host_facts_human(&report);
            if any_validation_error {
                eprintln!(
                    "\nValidation found errors. Resolve the configurations above before deploying."
                );
            } else if any_warning && warnings_as_errors {
                eprintln!(
                    "\nValidation passed with warnings; --warnings-as-errors is set, so they are fatal."
                );
            } else if any_warning {
                eprintln!(
                    "\nValidation passed with warnings. The deployment is safe but not optimal."
                );
            } else {
                eprintln!("\nValidation passed cleanly.");
            }
        }
        crate::cli::args::DoctorFormat::Json => {
            let json = serde_json::to_string_pretty(&report).map_err(|e| {
                CliError::InvalidArgument(format!("JSON serialization failed: {e}"))
            })?;
            println!("{json}");
        }
    }
    check_exit_outcome(any_validation_error, any_warning, warnings_as_errors)
}

/// Pure decision policy for `doctor --check`'s exit code.
///
/// Errors always fail. Warnings fail iff `warnings_as_errors == true`.
/// Pulled out as a free function so the policy can be unit-tested
/// without exercising the full `build_host_facts_report` pipeline.
fn check_exit_outcome(
    any_error: bool,
    any_warning: bool,
    warnings_as_errors: bool,
) -> Result<(), CliError> {
    if any_error {
        return Err(CliError::InvalidArgument(
            "host-facts validation found errors".into(),
        ));
    }
    if any_warning && warnings_as_errors {
        return Err(CliError::InvalidArgument(
            "host-facts validation found warnings (--warnings-as-errors)".into(),
        ));
    }
    Ok(())
}

/// Trace why one knob's resolved value is what it is.
///
/// Builds a per-knob [`KnobExplanation`] from the deployed
/// [`ServerConfig`] + detected [`HostFacts`], then renders in the
/// requested format. Unknown knob names produce a usage error
/// listing the valid choices.
fn run_explain(knob: &str, format: &crate::cli::args::DoctorFormat) -> Result<(), CliError> {
    let config = load_doctor_server_config()?;
    let facts = RealHostFactsSource.detect();
    let explanation = explain_knob(knob, &config, &facts).ok_or_else(|| {
        CliError::InvalidArgument(format!(
            "unknown knob `{knob}`; valid choices: \
             gpu_thread_pool_size, force_cpu, max_total_workers, \
             max_concurrent_jobs, max_workers_per_key, memory_gate_mb"
        ))
    })?;
    match format {
        crate::cli::args::DoctorFormat::Human => print_explanation_human(&explanation),
        crate::cli::args::DoctorFormat::Json => {
            let json = serde_json::to_string_pretty(&explanation).map_err(|e| {
                CliError::InvalidArgument(format!("JSON serialization failed: {e}"))
            })?;
            println!("{json}");
        }
    }
    Ok(())
}

/// Where a resolved value came from.
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum ValueSource {
    /// Operator set an explicit value in `server.yaml`. The
    /// recommendation is shown alongside so the operator can see
    /// what they overrode.
    OperatorOverride,
    /// No operator override; the host-facts recommendation prevails.
    Recommendation,
}

/// Per-knob explanation rendered by `--explain`. The string fields
/// (`rule`, `facts_used`) are pre-rendered narrative; JSON consumers
/// receive the same human-readable text under stable keys.
#[derive(Debug, serde::Serialize)]
struct KnobExplanation {
    knob: String,
    resolved_value: String,
    source: ValueSource,
    /// Always populated. When `source = OperatorOverride`, this is
    /// what the recommendation *would have* returned, useful for
    /// "is my override still needed?" reasoning.
    recommendation: String,
    /// Narrative description of the recommendation rule (e.g.,
    /// "1 when GPU is non-functional, 4 otherwise").
    rule: String,
    /// Narrative description of the facts that drove the
    /// recommendation (e.g., "gpu = AppleMps { ... }, ram_total_mb
    /// = 65536").
    facts_used: String,
}

/// Build an explanation for one knob; `None` if the knob name is
/// unknown.
fn explain_knob(
    knob: &str,
    cfg: &crate::config::ServerConfig,
    facts: &HostFacts,
) -> Option<KnobExplanation> {
    use host_facts::{
        recommend_force_cpu, recommend_gpu_thread_pool_size, recommend_max_concurrent_jobs,
        recommend_max_total_workers,
    };
    match knob {
        "gpu_thread_pool_size" => {
            let recommended = recommend_gpu_thread_pool_size(facts);
            let (resolved, source) = match cfg.gpu_thread_pool_size {
                Some(n) => (n, ValueSource::OperatorOverride),
                None => (recommended, ValueSource::Recommendation),
            };
            Some(KnobExplanation {
                knob: knob.to_owned(),
                resolved_value: resolved.to_string(),
                source,
                recommendation: recommended.to_string(),
                rule: "1 when GPU is non-functional for batchalign \
                       (Apple Silicon MPS, no GPU, CUDA with device_count=0); \
                       4 otherwise"
                    .to_owned(),
                facts_used: format!("gpu = {:?}", facts.gpu),
            })
        }
        "force_cpu" => {
            let recommended = recommend_force_cpu(facts);
            let (resolved, source) = match cfg.force_cpu {
                Some(b) => (b, ValueSource::OperatorOverride),
                None => (recommended, ValueSource::Recommendation),
            };
            Some(KnobExplanation {
                knob: knob.to_owned(),
                resolved_value: resolved.to_string(),
                source,
                recommendation: recommended.to_string(),
                rule: "true when GPU is non-functional for batchalign; false otherwise. \
                       The CLI `--force-cpu` switch sets `Some(true)` at the builder \
                       boundary; server.yaml can express either Some(true) or Some(false)."
                    .to_owned(),
                facts_used: format!("gpu = {:?}", facts.gpu),
            })
        }
        "max_total_workers" => {
            let recommended = recommend_max_total_workers(facts);
            let (resolved, source) = match cfg.max_total_workers {
                Some(n) => (n, ValueSource::OperatorOverride),
                None => (recommended, ValueSource::Recommendation),
            };
            Some(KnobExplanation {
                knob: knob.to_owned(),
                resolved_value: resolved.to_string(),
                source,
                recommendation: recommended.to_string(),
                rule: "clamp(ram_total_mb / 6 GB, 2, 32); fallback 4 when ram_total_mb = 0"
                    .to_owned(),
                facts_used: format!("ram_total_mb = {}", facts.ram_total_mb),
            })
        }
        "max_concurrent_jobs" => {
            let recommended = recommend_max_concurrent_jobs(facts);
            let (resolved, source) = match cfg.max_concurrent_jobs {
                Some(n) => (n, ValueSource::OperatorOverride),
                None => (recommended, ValueSource::Recommendation),
            };
            Some(KnobExplanation {
                knob: knob.to_owned(),
                resolved_value: resolved.to_string(),
                source,
                recommendation: recommended.to_string(),
                rule: "tier-and-CPU-bounded job concurrency (see \
                       recommend_max_concurrent_jobs in host_facts/recommendations.rs)"
                    .to_owned(),
                facts_used: format!(
                    "ram_total_mb = {}, cpu_logical_count = {}",
                    facts.ram_total_mb, facts.cpu_logical_count
                ),
            })
        }
        "max_workers_per_key" => {
            // Per-profile shape; report all three. `ServerConfig`
            // carries one operator value that fans out uniformly.
            let recommended = host_facts::recommend_max_workers_per_key(facts);
            let resolved_value = match cfg.max_workers_per_key {
                Some(n) => format!("gpu={n} stanza={n} io={n} (uniform override across profiles)"),
                None => format!(
                    "gpu={} stanza={} io={}",
                    recommended.gpu, recommended.stanza, recommended.io
                ),
            };
            let source = if cfg.max_workers_per_key.is_some() {
                ValueSource::OperatorOverride
            } else {
                ValueSource::Recommendation
            };
            Some(KnobExplanation {
                knob: knob.to_owned(),
                resolved_value,
                source,
                recommendation: format!(
                    "gpu={} stanza={} io={}",
                    recommended.gpu, recommended.stanza, recommended.io
                ),
                rule: "per-profile RAM-derived: gpu = ram/16GB clamped \
                       to [1,8]; stanza = ram/12GB clamped to [1,8]; \
                       io = 1 (flat). The legacy single ServerConfig \
                       knob applies uniformly across profiles."
                    .to_owned(),
                facts_used: format!("ram_total_mb = {}", facts.ram_total_mb),
            })
        }
        "memory_gate_mb" => {
            // `memory_gate_mb` falls through to the tier-derived
            // headroom rather than a `recommend_*` function (the
            // tier resolution honors the operator's `memory_tier`
            // override).
            let resolved = cfg.resolved_memory_gate_mb();
            let recommendation_value = cfg.resolved_memory_tier().headroom_mb;
            let source = if cfg.memory_gate_mb.is_some() {
                ValueSource::OperatorOverride
            } else {
                ValueSource::Recommendation
            };
            Some(KnobExplanation {
                knob: knob.to_owned(),
                resolved_value: format!("{} MB", resolved.0),
                source,
                recommendation: format!("{} MB (from MemoryTier headroom)", recommendation_value.0),
                rule: "tier-derived host-memory headroom; falls through \
                       to resolved_memory_tier().headroom_mb when no \
                       operator override is set"
                    .to_owned(),
                facts_used: format!(
                    "memory_tier = {:?}, ram_total_mb = {}",
                    cfg.memory_tier, facts.ram_total_mb
                ),
            })
        }
        _ => None,
    }
}

/// Render an explanation for human consumption.
fn print_explanation_human(e: &KnobExplanation) {
    eprintln!("\n{}: {}", e.knob, e.resolved_value);
    let source = match e.source {
        ValueSource::OperatorOverride => "operator override (server.yaml)",
        ValueSource::Recommendation => "host-facts recommendation",
    };
    eprintln!("  source:         {source}");
    if matches!(e.source, ValueSource::OperatorOverride) {
        eprintln!(
            "  recommended:    {} (would apply if no override)",
            e.recommendation
        );
    }
    eprintln!("  rule:           {}", e.rule);
    eprintln!("  facts:          {}", e.facts_used);
}

/// Production entry point: loads the deployed `server.yaml` and
/// detects facts via [`RealHostFactsSource`], then delegates to the
/// pure [`build_host_facts_report_from`] for the actual assembly.
fn build_host_facts_report(config: &crate::config::ServerConfig) -> HostFactsReport {
    build_host_facts_report_from(config, &RealHostFactsSource)
}

/// Pure assembly: bundles a [`HostFacts`] snapshot, the resolved
/// [`EffectiveConfig`], and the validation report into one
/// [`HostFactsReport`]. Both inputs are injected so tests can drive
/// the renderer with a [`host_facts::MockHostFactsSource`] and a
/// synthesized [`ServerConfig`], producing deterministic output.
fn build_host_facts_report_from(
    config: &crate::config::ServerConfig,
    source: &dyn HostFactsSource,
) -> HostFactsReport {
    let detected = source.detect();
    let overrides = host_facts::ConfigOverrides::from(config);
    let effective = EffectiveConfig::resolve(&overrides, &detected);
    let validation = host_facts::validate(config, &detected);
    let summary = EffectiveConfigSummary::from_effective(&effective, config, &detected);
    HostFactsReport {
        detected,
        effective: summary,
        validation: ValidationReport::from_validation(&validation),
    }
}

/// Best-effort load of the deployed `server.yaml` for validation. A
/// missing config is the default config (a fresh checkout); an unreadable
/// or malformed one is returned as the error it is, for the caller to
/// report.
fn load_doctor_server_config() -> Result<crate::config::ServerConfig, crate::config::ConfigError> {
    let layout = RuntimeLayout::from_env();
    config::load_config_from_layout(&layout, None)
}

/// Render the host-facts section in human-readable form.
///
/// Format intentionally avoids fancy boxes/colors so the output
/// composes cleanly with the existing check rows above it.
fn print_host_facts_human(report: &HostFactsReport) {
    let f = &report.detected;
    eprintln!("\nHost facts (snapshot at startup):");
    eprintln!(
        "  os/arch:           {:?}/{:?} ({} logical cores, {} physical)",
        f.os, f.arch, f.cpu_logical_count, f.cpu_physical_count
    );
    eprintln!(
        "  ram:               {} MB total, {} MB available",
        f.ram_total_mb, f.ram_available_mb
    );
    eprintln!("  gpu:               {:?}", f.gpu);
    if !f.detection_warnings.is_empty() {
        eprintln!("  detect warnings:   {:?}", f.detection_warnings);
    }

    let e = &report.effective;
    eprintln!("\nEffective config (after operator-override + recommendation merge):");
    eprintln!("  gpu_thread_pool_size: {}", e.gpu_thread_pool_size);
    eprintln!("  force_cpu:           {}", e.force_cpu);
    eprintln!("  max_total_workers:   {}", e.max_total_workers);
    eprintln!("  max_concurrent_jobs: {}", e.max_concurrent_jobs);
    eprintln!("  memory_gate_mb:      {}", e.memory_gate_mb);
    eprintln!(
        "  max_workers_per_key: gpu={} stanza={} io={}",
        e.max_workers_per_key_gpu, e.max_workers_per_key_stanza, e.max_workers_per_key_io,
    );

    let redundant = e.audit.redundant_knob_names();
    if redundant.is_empty() {
        eprintln!(
            "\nOverride audit:       no redundant overrides (every operator override differs from the host-facts recommendation)"
        );
    } else {
        eprintln!(
            "\nOverride audit:       {} redundant override(s) (override == recommendation; safe to delete from server.yaml):",
            redundant.len()
        );
        for name in &redundant {
            eprintln!("  - {name}");
        }
    }

    let v = &report.validation;
    if v.warnings.is_empty() && v.errors.is_empty() {
        eprintln!("\nValidation:           OK (no override contradicts detected facts)");
    } else {
        eprintln!("\nValidation findings:");
        for w in &v.warnings {
            eprintln!("  WARN  {w}");
        }
        for e in &v.errors {
            eprintln!("  ERROR {e}");
        }
    }
}

/// Spawn a worker and check it emits a valid ready signal.
/// Proof that this machine can actually DECODE audio.
///
/// Constructible only by decoding a real waveform through the same call the
/// pipeline uses, so holding one means decoding worked here, not that a module
/// looked importable.
#[derive(Debug, serde::Deserialize)]
struct MediaDecodeCapability {
    frames: u64,
    sample_rate: u32,
}

/// Why decoding is unavailable.
///
/// # What this type actually buys, stated honestly
///
/// NOTHING BRANCHES ON THE VARIANTS. The only consumer is `explain`, and its
/// only caller renders that string into a `CheckResult`. An earlier version of
/// this comment claimed each variant "is a DIFFERENT operator action, which is
/// why this is a sum type and not a string", which was an argument for a
/// distinction no code drew. Two of the variants it named no longer exist.
///
/// What it does buy is worth keeping, and is smaller than that claim: every
/// detection site must CLASSIFY its failure rather than reach for a free-form
/// string, and `explain`'s exhaustive match means a new fault cannot be added
/// without deciding what to tell the operator. That is a discipline on the
/// author, not a capability for the caller.
///
/// If a caller ever needs to branch, `doctor --format json` is where it would
/// surface, and the variants are already the right shape for it. Until then
/// this comment should not imply that day has arrived.
#[derive(Debug)]
enum MediaDecodeFault {
    /// The decode call itself failed, with whatever it said.
    ///
    /// Carries the diagnostic rather than an ffmpeg version: an earlier shape
    /// spawned `ffmpeg -version` a SECOND time to interpolate a version into a
    /// sentence that reads "this is NOT an FFmpeg version problem", while the
    /// `ffmpeg` check two lines above had already printed that banner.
    DecoderUnusable { detail: String },
    /// A module the decode path needs is missing from the interpreter.
    ModuleAbsent(String),
    /// The resolved interpreter could not be run.
    InterpreterUnusable(String),
    /// It ran, but returned something this check cannot interpret.
    Unintelligible(String),
}

impl MediaDecodeFault {
    /// One sentence a user can act on, never a dynamic-loader traceback.
    fn explain(&self) -> String {
        match self {
            Self::DecoderUnusable { detail } => format!(
                "audio decoding failed, so transcription and alignment will \
                 fail at job time. Batchalign decodes through soundfile and \
                 converts media with the `ffmpeg` BINARY, whose version does \
                 not matter, so this points at a broken soundfile install \
                 rather than at FFmpeg. Reinstall batchalign3. It said: \
                 {detail}"
            ),
            Self::ModuleAbsent(name) => format!(
                "the resolved interpreter is missing `{name}`, which the audio \
                 path needs. Reinstall batchalign3."
            ),
            Self::InterpreterUnusable(why) => {
                format!("could not run the resolved interpreter: {why}")
            }
            Self::Unintelligible(what) => {
                format!("the decode probe returned something unreadable: {what}")
            }
        }
    }
}

/// Decode a generated waveform through `torchaudio.load`.
///
/// # Why it decodes rather than importing
///
/// A previous version of this check imported `torchcodec.decoders` and a still
/// earlier one was DELETED on the reasoning that nothing used torchcodec. Both
/// were wrong in the same way: they asked whether a name resolved. When
/// Homebrew moved to FFmpeg 9, every torchcodec library (built for majors 4 to
/// 8) stopped loading and transcription began failing at job time, while
/// `import torchaudio` kept succeeding and `torchaudio.load` still existed as
/// an attribute. Only CALLING it reveals the fault, which is why this probe
/// decodes a waveform instead of asking whether a module imports.
///
/// The waveform is generated here rather than shipped as a fixture so the
/// check has no data dependency and cannot pass by reading a stale file.
fn check_media_decode(python: &str) -> Result<MediaDecodeCapability, MediaDecodeFault> {
    let output = Command::new(python)
        .args(["-m", "batchalign.inference.audio"])
        .stdin(Stdio::null())
        .output()
        .map_err(|e| MediaDecodeFault::InterpreterUnusable(e.to_string()))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let last = stdout.lines().last().unwrap_or_default().trim();

    if let Ok(capability) = serde_json::from_str::<MediaDecodeCapability>(last) {
        return Ok(capability);
    }
    if stderr.contains("ModuleNotFoundError") {
        // The NAME, not the whole diagnostic line: `explain` renders this as
        // "missing `{name}`", and stuffing the full "ModuleNotFoundError: No
        // module named 'soundfile'" in produced "missing `ModuleNotFoundError:
        // No module named 'soundfile'`".
        let missing = stderr
            .lines()
            .find_map(|line| {
                line.split_once("No module named")
                    .map(|(_, name)| name.trim().trim_matches(['\'', '"']).to_owned())
            })
            .unwrap_or_else(|| "a module the audio path needs".to_owned());
        return Err(MediaDecodeFault::ModuleAbsent(missing));
    }
    if output.status.success() {
        return Err(MediaDecodeFault::Unintelligible(last.to_owned()));
    }
    Err(MediaDecodeFault::DecoderUnusable {
        detail: stderr
            .lines()
            .last()
            .unwrap_or("no diagnostic")
            .trim()
            .to_owned(),
    })
}

/// Record one check's verdict, timing it from `start`.
///
/// # Why this exists
///
/// The six checks each open with the same fourteen lines: a `match` on a
/// `Result<String, String>`, two `CheckResult` literals differing only in
/// `status`, and `duration_ms` computed identically in both arms. Four were
/// byte-identical apart from the name.
///
/// That shape does not just cost lines, it INVITES the next copy, and a copy
/// is where the name and the status drift apart. A check is now one call, and
/// the Pass/Fail mapping has one definition: `Ok` is a pass, `Err` is a
/// failure, and neither arm can be given the wrong `status` because neither
/// arm is written at the call site any more.
fn record(
    results: &mut Vec<CheckResult>,
    name: &str,
    start: Instant,
    outcome: Result<String, String>,
) {
    let duration_ms = start.elapsed().as_millis() as u64;
    let (status, detail) = match outcome {
        Ok(detail) => (CheckStatus::Pass, detail),
        Err(detail) => (CheckStatus::Fail, detail),
    };
    results.push(CheckResult {
        name: name.into(),
        status,
        detail,
        duration_ms: Some(duration_ms),
    });
}

/// Whether the `ffmpeg` binary this machine needs is actually present.
///
/// # The verdict is production's, the detail is extra
///
/// Batchalign shells out to `ffmpeg` at five production call sites: it
/// transcodes arbitrary media to wav, prepares artifact segments, and probes
/// duration with `ffprobe`. Without it, any job whose input is not already a
/// wav fails. Nothing checked for it before.
///
/// PASS/FAIL is whether the binary answers `-version`. That is deliberately a
/// DIFFERENT question from the one production asks, which is whether the spawn
/// it needed returned `ErrorKind::NotFound`, and the two can disagree: on a
/// host where ffmpeg exists but `-version` exits non-zero, doctor says FAIL
/// and production reports `TranscodeError::Failed` rather than
/// `FfmpegMissing`. Doctor
/// asks the version question because it has a banner to print.
///
/// An earlier version demanded a non-empty banner AND a working `ffprobe` as
/// part of the verdict, which could fail a host where every conversion works.
///
/// The banner and `ffprobe` are still reported, as DETAIL. A missing `ffprobe`
/// is worth an operator's attention (duration probing uses it) without being
/// the same claim as "ffmpeg is unusable".
fn check_ffmpeg() -> Result<String, String> {
    // Asking for the banner asks production's question and keeps the answer it
    // already captured, so the verdict and the detail cost one spawn between
    // them. See `MediaTool::banner`.
    let Some(version) = MediaTool::Ffmpeg.banner() else {
        return Err(
            "ffmpeg is not usable, so any job whose media is not already a wav \
             will fail: transcoding, artifact segment preparation and duration \
             probing all shell out to it. Install it (macOS: `brew install \
             ffmpeg`; Debian/Ubuntu: `apt install ffmpeg`)."
                .to_owned(),
        );
    };

    // ffprobe is reported as DETAIL, never as part of the verdict: a doctor
    // that fails a host where every conversion works is reporting on something
    // other than production.
    //
    // Asked through the same predicate as ffmpeg's. This was an open-coded
    // spawn, which is how the two probes came to differ in whether they closed
    // stdin: a difference nobody chose, because nothing stated it once.
    if MediaTool::Ffprobe.banner().is_some() {
        Ok(version)
    } else {
        Ok(format!(
            "{version} (NOTE: ffprobe is not usable, so duration probing will \
             fail; both ship together, so this usually means a partial install)"
        ))
    }
}

// ---------------------------------------------------------------------------
// The Python worker seam, modelled as a type state
// ---------------------------------------------------------------------------
//
// # Why a type state rather than two functions with a `ready` flag
//
// This file previously carried two near-identical helpers, each spawning a
// worker, looping for a `{"ready": true}` line, and tracking success in a
// local `bool`. Nothing in either signature said a request may only be written
// AFTER that signal, so the rule lived in whichever function you happened to
// read.
//
// Now `spawn` yields `Worker<Spawned>`, which has no method that writes a
// request, and `await_ready` CONSUMES it to yield `Worker<Ready>`, which does.
// Sending to a worker that has not signalled ready is not a discouraged
// ordering; it is a program that does not compile. The `bool` is gone because
// the state it tracked is now the type.

/// A spawned worker that has not yet signalled readiness.
struct Spawned;

/// A worker that has signalled readiness and will accept one request.
struct Ready;

/// The process handles, separated so `Worker` itself owns no `Drop` glue and a
/// state transition can move the handles out of the old state.
struct WorkerIo {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    lines: std::io::Lines<std::io::BufReader<std::process::ChildStdout>>,
    /// The pid the OPERATING SYSTEM gave us when we spawned it.
    ///
    /// Kept distinct from the pid the worker reports about itself. They are
    /// two different facts and one used to overwrite the other in place, with
    /// `unwrap_or` papering over the case where the worker said nothing: after
    /// that assignment nothing could tell which source a given value came
    /// from, which is the shape where a value proxies for a richer fact.
    spawned_pid: u32,
    /// The pid the worker announced in its ready line, if it announced one.
    ///
    /// `None` means it signalled ready without naming itself, which is not the
    /// same as "the pid is the one we spawned".
    reported_pid: Option<u32>,
}

impl Drop for WorkerIo {
    /// Never leave a worker behind. Every early return in this module drops
    /// the `Worker`, and a diagnostic command that leaks Python processes on
    /// the failure paths is worse than the fault it was run to find.
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A Python worker subprocess, parameterised by what it is ready to do.
struct Worker<State> {
    io: WorkerIo,
    _state: std::marker::PhantomData<State>,
}

impl Worker<Spawned> {
    /// Start `python -m batchalign.worker` with `args`.
    fn spawn(python: &str, args: &[&str]) -> Result<Self, String> {
        let mut cmd = Command::new(python);
        cmd.args(["-m", "batchalign.worker"]);
        cmd.args(args);
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| format!("Spawn failed: {e}"))?;
        let stdout = child.stdout.take().ok_or("No stdout")?;
        let stdin = child.stdin.take().ok_or("No stdin")?;
        let spawned_pid = child.id();
        Ok(Self {
            io: WorkerIo {
                child,
                stdin,
                lines: std::io::BufReader::new(stdout).lines(),
                spawned_pid,
                reported_pid: None,
            },
            _state: std::marker::PhantomData,
        })
    }

    /// Consume the spawned worker and yield one that is ready to be asked.
    ///
    /// Consuming is the point: after this returns there is no value left that
    /// represents "spawned but not ready", so no later code can act on one.
    fn await_ready(mut self, timeout: std::time::Duration) -> Result<Worker<Ready>, String> {
        let deadline = Instant::now() + timeout;
        while let Some(line) = self.io.lines.next() {
            if Instant::now() > deadline {
                return Err(format!(
                    "Timeout ({}s) waiting for ready signal",
                    timeout.as_secs()
                ));
            }
            let line = line.map_err(|e| format!("Read error: {e}"))?;
            let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) else {
                continue; // worker logging, not protocol
            };
            if value.get("ready") == Some(&serde_json::Value::Bool(true)) {
                let mut io = self.io;
                io.reported_pid = value
                    .get("pid")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|reported| u32::try_from(reported).ok());
                return Ok(Worker {
                    io,
                    _state: std::marker::PhantomData,
                });
            }
        }
        Err("Worker exited without ready signal".into())
    }
}

impl Worker<Ready> {
    /// The pid to show an operator: what the worker called itself, falling
    /// back to what we spawned. The FALLBACK IS AT THE DISPLAY EDGE, where a
    /// human reads it, rather than baked into storage where the two facts
    /// become indistinguishable.
    fn pid(&self) -> u32 {
        self.io.reported_pid.unwrap_or(self.io.spawned_pid)
    }

    /// Ask the worker to exit, and wait for it.
    fn shutdown(mut self) -> Result<(), String> {
        writeln!(self.io.stdin, r#"{{"op":"shutdown"}}"#)
            .map_err(|e| format!("Write failed: {e}"))?;
        self.io
            .child
            .wait()
            .map_err(|e| format!("Wait failed: {e}"))?;
        Ok(())
    }

    /// Send one morphosyntax batch and return the worker's raw response.
    ///
    /// # Why this takes the real payload type
    ///
    /// It used to take `&[Vec<&str>]` and build each item with a `json!`
    /// literal. That literal was written when `terminator` did not exist. When
    /// the schema gained it as a REQUIRED field with no default (deliberately:
    /// a default period would be a sentinel that is also a legal value),
    /// production was forced to follow by the compiler and this call site was
    /// not, because a `json!` object accepts any shape. Every doctor run since
    /// reported three "Invalid batch item" errors, and the doctor was the one
    /// tool nobody suspected.
    ///
    /// Taking `&[MorphosyntaxBatchItem]` puts this seam back under the
    /// compiler. The next required field breaks the build here.
    fn morphosyntax_batch(
        mut self,
        lang: &str,
        items: &[crate::chat_ops::morphosyntax_ops::MorphosyntaxBatchItem],
        timeout: std::time::Duration,
    ) -> Result<String, String> {
        let encoded = items
            .iter()
            .map(serde_json::to_value)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Could not encode batch items: {e}"))?;

        let request = serde_json::json!({
            "op": "batch_infer",
            "request": { "task": "morphosyntax", "lang": lang, "items": encoded }
        });
        let body = serde_json::to_string(&request)
            .map_err(|e| format!("Could not encode request: {e}"))?;
        writeln!(self.io.stdin, "{body}").map_err(|e| format!("Write failed: {e}"))?;
        self.io
            .stdin
            .flush()
            .map_err(|e| format!("Flush failed: {e}"))?;

        let deadline = Instant::now() + timeout;
        while let Some(line) = self.io.lines.next() {
            if Instant::now() > deadline {
                return Err(format!(
                    "Timeout ({}s) waiting for batch_infer response",
                    timeout.as_secs()
                ));
            }
            let line = line.map_err(|e| format!("Read error: {e}"))?;
            let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) else {
                continue; // worker logging, not protocol
            };
            match value.get("op").and_then(serde_json::Value::as_str) {
                Some("error") => {
                    let err = value
                        .get("error")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("unknown");
                    return Err(format!("Worker error: {err}"));
                }
                Some("batch_infer") => {
                    let summary = summarize_morphosyntax_response(&value, items.len());
                    let _ = self.shutdown();
                    return summary;
                }
                _ => continue,
            }
        }
        Err("Worker exited without batch_infer response".into())
    }
}

/// The doctor's self-test utterances, built the way production builds them.
///
/// # Why it parses a transcript instead of listing words
///
/// `collect_payloads` is the function the real morphotag path calls, so going
/// through it means the check exercises chatter's parser, the payload builder
/// and the worker wire in one pass, rather than an imitation of them that can
/// drift. It also means the terminator, the special-form table and the
/// resolved language are DERIVED from a parsed transcript rather than asserted
/// here, so none of them can be silently wrong or silently absent.
fn smoke_batch_items(
    lang: &talkbank_model::LanguageCode,
) -> Result<Vec<crate::chat_ops::morphosyntax_ops::MorphosyntaxBatchItem>, String> {
    let code = lang.as_str();
    let document = format!(
        "@UTF8\n@Begin\n@Languages:\t{code}\n@Participants:\tCHI Child\n\
         @ID:\t{code}|doctor|CHI|||||Child|||\n\
         *CHI:\tthe dog runs .\n\
         *CHI:\tI dont know ?\n\
         *CHI:\ta .\n\
         @End\n"
    );

    let parser = talkbank_parser::TreeSitterParser::new()
        .map_err(|e| format!("chatter's parser failed to load: {e}"))?;
    let errors = talkbank_model::ErrorCollector::new();
    let chat_file = parser.parse_chat_file_streaming(&document, &errors);

    let reported = errors.to_vec();
    if let Some(first) = reported.first() {
        return Err(format!(
            "the doctor's own probe transcript no longer parses ({} error(s), \
             first: {first:?}). That is a chatter defect or a CHAT rule change, \
             not a worker fault.",
            reported.len()
        ));
    }

    let collection = crate::chat_ops::morphosyntax_ops::collect_payloads(
        &chat_file,
        lang,
        std::slice::from_ref(lang),
        crate::chat_ops::morphosyntax_ops::MultilingualPolicy::ProcessAll,
    );
    if collection.batch_items.is_empty() {
        return Err(
            "the probe transcript produced no batch items; the payload builder \
             considers every one of its utterances unanalysable"
                .to_owned(),
        );
    }
    Ok(collection
        .batch_items
        .into_iter()
        .map(crate::chat_ops::morphosyntax_ops::CollectedUtterance::into_item)
        .collect())
}

/// Check that every word the worker returned carries the fields morphotag needs.
fn summarize_morphosyntax_response(
    value: &serde_json::Value,
    expected: usize,
) -> Result<String, String> {
    let results = value
        .pointer("/response/results")
        .and_then(serde_json::Value::as_array)
        .ok_or("No results array in response")?;

    if results.len() != expected {
        return Err(format!(
            "Expected {expected} results, got {}",
            results.len()
        ));
    }

    let mut total_words = 0usize;
    let mut problems: Vec<String> = Vec::new();

    for (ri, result) in results.iter().enumerate() {
        let Some(sentences) = result
            .pointer("/result/raw_sentences")
            .and_then(serde_json::Value::as_array)
        else {
            if let Some(err) = result.get("error").and_then(serde_json::Value::as_str) {
                problems.push(format!("result {ri}: worker error: {err}"));
            }
            continue;
        };
        for (si, sentence) in sentences.iter().enumerate() {
            let Some(words) = sentence.as_array() else {
                continue;
            };
            for (wi, word) in words.iter().enumerate() {
                total_words += 1;
                for field in ["text", "lemma", "upos", "deprel"] {
                    let present = word
                        .get(field)
                        .is_some_and(|v| *v != serde_json::Value::Null);
                    if present {
                        continue;
                    }
                    // An MWT range token legitimately lacks the analysis
                    // fields; it carries only the surface text.
                    let is_range = word
                        .get("id")
                        .and_then(serde_json::Value::as_array)
                        .is_some_and(|ids| ids.len() > 1);
                    if !is_range || field == "text" {
                        let text = word
                            .get("text")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or("?");
                        problems.push(format!(
                            "result {ri} sent {si} word {wi} ('{text}'): missing {field}"
                        ));
                    }
                }
            }
        }
    }

    if problems.is_empty() {
        Ok(format!(
            "{expected} sentences, {total_words} words, all fields present"
        ))
    } else {
        Err(format!(
            "{} field issues: {}",
            problems.len(),
            problems.join("; ")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ServerConfig;
    use crate::host_facts::ConfigOverrides;
    use crate::host_facts::test_helpers::apple_silicon_64gb;

    /// `EffectiveConfigSummary::from_effective` projects every public
    /// knob from `EffectiveConfig` exactly once. The schema is the
    /// operator's contract for `--format json`; this test pins the
    /// shape so a future field addition reminds the author to think
    /// about backwards compatibility.
    #[test]
    fn effective_config_summary_includes_every_resolved_knob() {
        let facts = apple_silicon_64gb();
        let cfg = ServerConfig::default();
        let overrides = ConfigOverrides::default();
        let effective = EffectiveConfig::resolve(&overrides, &facts);
        let summary = EffectiveConfigSummary::from_effective(&effective, &cfg, &facts);
        // gpu_thread_pool_size: Apple Silicon recommends 1
        assert_eq!(summary.gpu_thread_pool_size, 1);
        // force_cpu: Apple Silicon recommends true
        assert!(summary.force_cpu);
        // memory_gate_mb: drawn from the recommended tier headroom
        assert!(summary.memory_gate_mb > 0);
        // Per-profile knobs all populated (none zero on a 64 GB host)
        assert!(summary.max_workers_per_key_gpu >= 1);
        assert!(summary.max_workers_per_key_stanza >= 1);
        assert!(summary.max_workers_per_key_io >= 1);
    }

    #[test]
    fn audit_with_no_overrides_has_empty_redundancy_list() {
        let facts = apple_silicon_64gb();
        let cfg = ServerConfig::default();
        let overrides = ConfigOverrides::default();
        let effective = EffectiveConfig::resolve(&overrides, &facts);
        let summary = EffectiveConfigSummary::from_effective(&effective, &cfg, &facts);
        assert!(summary.audit.redundant_knob_names().is_empty());
        assert!(
            summary
                .audit
                .gpu_thread_pool_size
                .operator_override
                .is_none()
        );
        assert!(!summary.audit.gpu_thread_pool_size.redundant);
        assert!(summary.audit.max_total_workers.operator_override.is_none());
        assert!(!summary.audit.max_total_workers.redundant);
    }

    #[test]
    fn audit_flags_redundant_override_matching_recommendation() {
        use crate::host_facts::recommend_max_total_workers;
        let facts = apple_silicon_64gb();
        let recommended_total = recommend_max_total_workers(&facts);
        let cfg = ServerConfig {
            max_total_workers: Some(recommended_total),
            ..Default::default()
        };
        let overrides = ConfigOverrides::from(&cfg);
        let effective = EffectiveConfig::resolve(&overrides, &facts);
        let summary = EffectiveConfigSummary::from_effective(&effective, &cfg, &facts);
        assert!(summary.audit.max_total_workers.redundant);
        assert_eq!(
            summary.audit.max_total_workers.operator_override,
            Some(recommended_total)
        );
        assert!(
            summary
                .audit
                .redundant_knob_names()
                .contains(&"max_total_workers")
        );
    }

    #[test]
    fn audit_does_not_flag_overrides_that_differ_from_recommendation() {
        use crate::host_facts::recommend_max_total_workers;
        let facts = apple_silicon_64gb();
        let recommended_total = recommend_max_total_workers(&facts);
        let custom = recommended_total
            .checked_add(1)
            .expect("recommended max never saturates u32");
        let cfg = ServerConfig {
            max_total_workers: Some(custom),
            ..Default::default()
        };
        let overrides = ConfigOverrides::from(&cfg);
        let effective = EffectiveConfig::resolve(&overrides, &facts);
        let summary = EffectiveConfigSummary::from_effective(&effective, &cfg, &facts);
        assert!(!summary.audit.max_total_workers.redundant);
        assert_eq!(
            summary.audit.max_total_workers.recommended,
            recommended_total
        );
        assert_eq!(summary.audit.max_total_workers.effective, custom);
        assert!(
            !summary
                .audit
                .redundant_knob_names()
                .contains(&"max_total_workers")
        );
    }

    /// On an unconfigured host (default `ServerConfig`), the
    /// validation report's warnings and errors are both empty.
    /// Pins the "default config -> clean validation" contract that
    /// production startup relies on.
    #[test]
    fn validation_report_is_empty_for_default_config() {
        let cfg = ServerConfig::default();
        let facts = apple_silicon_64gb();
        let v = host_facts::validate(&cfg, &facts);
        let report = ValidationReport::from_validation(&v);
        assert!(report.warnings.is_empty());
        assert!(report.errors.is_empty());
    }

    /// An operator setting `gpu_thread_pool_size: 4` on Apple Silicon
    /// triggers the configured-vs-recommended warning, which surfaces
    /// in the `ValidationReport` as a rendered string. The `--format
    /// json` consumer reads strings, not enum tags, so the
    /// `Display` -> string conversion is the API contract.
    #[test]
    fn validation_report_renders_warning_strings() {
        let cfg = ServerConfig {
            gpu_thread_pool_size: Some(4),
            ..Default::default()
        };
        let facts = apple_silicon_64gb();
        let v = host_facts::validate(&cfg, &facts);
        let report = ValidationReport::from_validation(&v);
        assert_eq!(report.warnings.len(), 1);
        assert!(
            report.warnings[0].contains("gpu_thread_pool_size=4"),
            "warning string should include the configured value: {:?}",
            report.warnings[0]
        );
    }

    // -----------------------------------------------------------------
    // --explain <knob>
    // -----------------------------------------------------------------

    /// Unknown knob names produce `None`; the caller renders that as
    /// a usage error listing the valid choices. Pinning here so a
    /// future knob addition that forgets the dispatcher arm is
    /// caught.
    #[test]
    fn explain_returns_none_for_unknown_knob() {
        let cfg = ServerConfig::default();
        let facts = apple_silicon_64gb();
        assert!(explain_knob("not_a_real_knob", &cfg, &facts).is_none());
    }

    /// Default config -> source = Recommendation; resolved value
    /// equals the recommendation; the rule and facts strings are
    /// non-empty.
    #[test]
    fn explain_gpu_thread_pool_size_with_no_override_uses_recommendation() {
        let cfg = ServerConfig::default();
        let facts = apple_silicon_64gb();
        let exp = explain_knob("gpu_thread_pool_size", &cfg, &facts).expect("known knob");
        assert!(matches!(exp.source, ValueSource::Recommendation));
        assert_eq!(exp.resolved_value, "1"); // Apple Silicon recommendation
        assert_eq!(exp.recommendation, "1");
        assert!(!exp.rule.is_empty());
        assert!(
            exp.facts_used.contains("AppleMps"),
            "facts: {}",
            exp.facts_used
        );
    }

    /// `Some(4)` operator override: source = OperatorOverride;
    /// resolved value = 4; recommendation field still shows what
    /// the recommendation would have been (1 on Apple Silicon).
    #[test]
    fn explain_gpu_thread_pool_size_with_override_reports_both_values() {
        let cfg = ServerConfig {
            gpu_thread_pool_size: Some(4),
            ..Default::default()
        };
        let facts = apple_silicon_64gb();
        let exp = explain_knob("gpu_thread_pool_size", &cfg, &facts).expect("known knob");
        assert!(matches!(exp.source, ValueSource::OperatorOverride));
        assert_eq!(exp.resolved_value, "4");
        assert_eq!(exp.recommendation, "1");
    }

    // -----------------------------------------------------------------
    // Snapshot tests for `--format json`
    //
    // These pin the operator-facing JSON wire format. Determinism
    // comes from `MockHostFactsSource` (synthesized facts) +
    // hardcoded `ServerConfig` values: neither the host's live
    // RAM/CPU nor the YAML-on-disk influences the output.
    // -----------------------------------------------------------------

    /// Default config on Apple Silicon: clean validation, recommendations
    /// flow through unchanged.
    #[test]
    fn snapshot_host_facts_report_default_apple_silicon_json() {
        // Pin memory_tier so resolved_memory_tier() does not call
        // MemoryTier::detect() (which reads live host RAM via sysinfo
        // and is not driven by MockHostFactsSource). Without this,
        // the audit's `memory_gate_mb.recommended` shifts with the
        // host's RAM tier and the snap diverges between Apple Silicon
        // dev machines (Fleet, 8000 MB) and small CI runners (Small,
        // 2000 MB).
        let cfg = ServerConfig {
            memory_tier: Some(crate::types::runtime::MemoryTierKind::Fleet),
            ..Default::default()
        };
        let source = host_facts::MockHostFactsSource::new(apple_silicon_64gb());
        let report = build_host_facts_report_from(&cfg, &source);
        insta::assert_json_snapshot!("host_facts_report_default_apple_silicon", report);
    }

    /// Operator override that triggers a warning: gpu_thread_pool_size=4
    /// on a non-functional GPU. Snapshot pins both the rendered warning
    /// string and the resolved override propagation.
    #[test]
    fn snapshot_host_facts_report_with_warning_apple_silicon_json() {
        let cfg = ServerConfig {
            gpu_thread_pool_size: Some(4),
            // See the default-snapshot test above for why memory_tier
            // is pinned here.
            memory_tier: Some(crate::types::runtime::MemoryTierKind::Fleet),
            ..Default::default()
        };
        let source = host_facts::MockHostFactsSource::new(apple_silicon_64gb());
        let report = build_host_facts_report_from(&cfg, &source);
        insta::assert_json_snapshot!("host_facts_report_with_warning_apple_silicon", report);
    }

    /// `--explain` payload for one knob with no operator override:
    /// pins the recommendation-narrative shape and field names.
    #[test]
    fn snapshot_explain_gpu_thread_pool_size_default_apple_silicon_json() {
        let cfg = ServerConfig::default();
        let facts = apple_silicon_64gb();
        let explanation = explain_knob("gpu_thread_pool_size", &cfg, &facts).expect("known knob");
        insta::assert_json_snapshot!(
            "explain_gpu_thread_pool_size_default_apple_silicon",
            explanation
        );
    }

    /// `--explain` payload with an operator override: the snapshot
    /// captures both the resolved value (4) and the recommendation
    /// (1) so consumers can see the contradiction.
    #[test]
    fn snapshot_explain_gpu_thread_pool_size_overridden_apple_silicon_json() {
        let cfg = ServerConfig {
            gpu_thread_pool_size: Some(4),
            ..Default::default()
        };
        let facts = apple_silicon_64gb();
        let explanation = explain_knob("gpu_thread_pool_size", &cfg, &facts).expect("known knob");
        insta::assert_json_snapshot!(
            "explain_gpu_thread_pool_size_overridden_apple_silicon",
            explanation
        );
    }

    /// Each migrated knob has a dispatcher arm; iterate through the
    /// canonical list to catch additions that forget the wire-up.
    #[test]
    fn explain_handles_every_documented_knob() {
        let cfg = ServerConfig::default();
        let facts = apple_silicon_64gb();
        for knob in [
            "gpu_thread_pool_size",
            "force_cpu",
            "max_total_workers",
            "max_concurrent_jobs",
            "max_workers_per_key",
            "memory_gate_mb",
        ] {
            assert!(
                explain_knob(knob, &cfg, &facts).is_some(),
                "missing dispatcher arm for `{knob}`"
            );
        }
    }

    // -----------------------------------------------------------------
    // --check exit policy (--warnings-as-errors)
    // -----------------------------------------------------------------

    /// No findings, default policy: exit 0.
    #[test]
    fn check_exit_clean_validation_succeeds() {
        assert!(check_exit_outcome(false, false, false).is_ok());
        assert!(check_exit_outcome(false, false, true).is_ok());
    }

    /// Warnings without `--warnings-as-errors`: exit 0 (operator
    /// may have intentionally overridden a recommendation).
    #[test]
    fn check_exit_warnings_alone_succeed_by_default() {
        assert!(check_exit_outcome(false, true, false).is_ok());
    }

    /// Warnings WITH `--warnings-as-errors`: exit non-zero. The
    /// CI-gate posture for zero-warning deployments.
    #[test]
    fn check_exit_warnings_with_warnings_as_errors_fail() {
        assert!(check_exit_outcome(false, true, true).is_err());
    }

    /// Errors always fail, regardless of `--warnings-as-errors`.
    #[test]
    fn check_exit_errors_always_fail() {
        assert!(check_exit_outcome(true, false, false).is_err());
        assert!(check_exit_outcome(true, false, true).is_err());
        assert!(check_exit_outcome(true, true, false).is_err());
        assert!(check_exit_outcome(true, true, true).is_err());
    }

    /// `HostFactsReport` round-trips through serde_json without
    /// errors. JSON consumers treat the schema as stable, so any
    /// future field change must consciously break this test.
    #[test]
    fn host_facts_report_serializes_to_json() {
        let cfg = ServerConfig::default();
        let facts = apple_silicon_64gb();
        let overrides = ConfigOverrides::from(&cfg);
        let effective = EffectiveConfig::resolve(&overrides, &facts);
        let validation = host_facts::validate(&cfg, &facts);
        let facts_for_summary = facts.clone();
        let report = HostFactsReport {
            detected: facts,
            effective: EffectiveConfigSummary::from_effective(&effective, &cfg, &facts_for_summary),
            validation: ValidationReport::from_validation(&validation),
        };
        let json = serde_json::to_value(&report).expect("serialize");
        // Top-level keys are the operator's contract.
        assert!(json.get("detected").is_some());
        assert!(json.get("effective").is_some());
        assert!(json.get("validation").is_some());
        // EffectiveConfigSummary keys: pin a representative subset.
        let eff = json.get("effective").unwrap();
        for key in [
            "gpu_thread_pool_size",
            "force_cpu",
            "max_total_workers",
            "max_concurrent_jobs",
            "memory_gate_mb",
            "max_workers_per_key_gpu",
            "max_workers_per_key_stanza",
            "max_workers_per_key_io",
        ] {
            assert!(
                eff.get(key).is_some(),
                "missing `{key}` in effective summary: {eff:#?}"
            );
        }
    }
}
