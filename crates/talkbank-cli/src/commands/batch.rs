//! `chatter batch` — loop `chatter pipeline` over a directory of
//! matched donor/reference pairs.
//!
//! Implemented as a subprocess driver: for each donor file, spawn
//! `<self> pipeline DONOR REF -o OUT/basename.cha ...` and aggregate
//! exit codes into per-session outcomes. Subprocess overhead per
//! session is in the millisecond range; per-session work (parse +
//! Jaccard + merge) is in the second range; the subprocess
//! invocation pattern is fine until corpus sizes push into the
//! thousands of sessions per batch.

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use tracing::{Level, info, span, warn};

use crate::exit_codes::{EXIT_INPUT_ERROR, EXIT_LOW_CONFIDENCE, EXIT_PRECONDITION, EXIT_SUCCESS};
use talkbank_transform::sanity_scan::SanityScanThreshold;

use super::pipeline::PipelineArgs;

/// All inputs for one `chatter batch` invocation.
///
/// Mirrors the per-session [`PipelineArgs`] shape but binds the
/// directory-level inputs (`donor_dir` / `reference_dir` / `output_dir`)
/// the driver iterates over, plus the batch-only `skip_existing` knob.
/// The driver constructs a fresh [`PipelineArgs`] per session inside
/// the loop.
pub struct BatchArgs<'a> {
    /// Directory of donor CHAT files to process.
    pub donor_dir: &'a Path,
    /// Directory of reference CHAT files matched by basename.
    pub reference_dir: &'a Path,
    /// Anchor speaker code (passed through to every per-session run).
    pub anchor: &'a str,
    /// Inserted-role spec (passed through to every per-session run).
    pub inserted_role: &'a str,
    /// Retain set (passed through to every per-session run).
    pub retain: &'a [String],
    /// Confidence threshold (passed through to every per-session run).
    pub confidence_threshold: f64,
    /// Optional pending-entries TOML (passed through; accumulates
    /// across sessions).
    pub write_pending_path: Option<&'a Path>,
    /// Optional override-file TOML (passed through; sessions with a
    /// matching entry replay rather than re-run reference mode).
    pub override_file_path: Option<&'a Path>,
    /// Optional audit-trail override-file destination (passed
    /// through; reference-mode clean-winners append entries here).
    /// Distinct from `override_file_path`, though operators commonly
    /// point both at the same file.
    pub write_override_path: Option<&'a Path>,
    /// `Some(threshold)` enables the post-loop sanity scan over
    /// `output_dir` + `write_override_path`, appending flagged
    /// sessions to `write_pending_path`. `None` disables the scan.
    /// clap's `requires_all` on `--sanity-scan` guarantees both
    /// write paths are present whenever this is `Some`.
    pub sanity_scan: Option<SanityScanThreshold>,
    /// If true, donors whose output already exists in `output_dir`
    /// are skipped (the batch becomes idempotent under operator
    /// re-runs).
    pub skip_existing: bool,
    /// Destination directory for per-session merged CHAT files.
    pub output_dir: &'a Path,
}

/// Top-level entry for `chatter batch`. Exit codes:
/// - 0: batch driver completed (every matched session produced an
///   outcome, success or low-confidence-refusal).
/// - 1: I/O error reading donor / reference directories.
/// - 2: at least one session failed with a non-low-confidence error
///   (precondition violation, parse error). Operator should inspect
///   the per-session output for the failing file.
pub fn run_batch(args: BatchArgs<'_>) {
    let BatchArgs {
        donor_dir,
        reference_dir,
        anchor,
        inserted_role,
        retain,
        confidence_threshold,
        write_pending_path,
        override_file_path,
        write_override_path,
        sanity_scan,
        skip_existing,
        output_dir,
    } = args;

    let _span = span!(
        Level::INFO,
        "chatter_batch",
        donor_dir = %donor_dir.display(),
        reference_dir = %reference_dir.display(),
    )
    .entered();

    if let Err(e) = fs::create_dir_all(output_dir) {
        warn!(
            "failed to create output dir {}: {}",
            output_dir.display(),
            e
        );
        eprintln!("Error creating output dir {}: {}", output_dir.display(), e);
        std::process::exit(EXIT_INPUT_ERROR);
    }

    let donor_files = match list_cha_files(donor_dir) {
        Ok(v) => v,
        Err(e) => {
            warn!("failed to list donor dir {}: {}", donor_dir.display(), e);
            eprintln!("Error reading donor dir {}: {}", donor_dir.display(), e);
            std::process::exit(EXIT_INPUT_ERROR);
        }
    };
    if donor_files.is_empty() {
        eprintln!("No .cha files found in donor dir {}", donor_dir.display());
        std::process::exit(EXIT_INPUT_ERROR);
    }

    let self_exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            warn!("failed to resolve current_exe: {}", e);
            eprintln!("Error resolving chatter executable: {}", e);
            std::process::exit(EXIT_INPUT_ERROR);
        }
    };

    let mut successes = 0usize;
    let mut refusals = 0usize;
    let mut errors = 0usize;
    let mut unmatched = 0usize;
    let mut skipped = 0usize;

    for donor in donor_files {
        let basename = match donor.file_name() {
            Some(n) => n.to_os_string(),
            None => {
                warn!("donor path has no file name: {}", donor.display());
                errors += 1;
                continue;
            }
        };
        let reference = reference_dir.join(&basename);
        if !reference.exists() {
            warn!(
                "no matching reference for donor {}; expected {}",
                donor.display(),
                reference.display()
            );
            eprintln!(
                "Warning: skipping {} — no matching reference at {}",
                donor.display(),
                reference.display()
            );
            unmatched += 1;
            continue;
        }
        let out_path = output_dir.join(&basename);
        if skip_existing && out_path.exists() {
            info!("⏭ skipped (output exists): {}", donor.display());
            skipped += 1;
            continue;
        }
        let pipeline_args = PipelineArgs {
            donor: &donor,
            reference: &reference,
            anchor,
            inserted_role,
            retain,
            confidence_threshold,
            write_pending_path,
            override_file_path,
            write_override_path,
            output: &out_path,
        };
        let outcome = run_pipeline_subprocess(&self_exe, &pipeline_args);
        match outcome {
            SessionOutcome::Success => {
                info!("✓ {}", donor.display());
                successes += 1;
            }
            SessionOutcome::LowConfidence => {
                info!("⚠ low-confidence: {}", donor.display());
                refusals += 1;
            }
            SessionOutcome::Error(code) => {
                warn!("✗ {} (exit {})", donor.display(), code);
                eprintln!("Error: {} failed (exit {})", donor.display(), code);
                errors += 1;
            }
        }
    }

    let total = successes + refusals + errors + unmatched + skipped;
    eprintln!(
        "batch summary: {total} matched donor(s); {successes} merged, {refusals} pending adjudication, {errors} errored, {unmatched} unmatched (no reference), {skipped} skipped (output existed)"
    );
    if errors > 0 {
        std::process::exit(EXIT_PRECONDITION);
    }
    if let (Some(threshold), Some(override_path), Some(pending_path)) =
        (sanity_scan, write_override_path, write_pending_path)
    {
        run_sanity_scan_subprocess(
            &self_exe,
            output_dir,
            override_path,
            anchor,
            threshold,
            pending_path,
        );
    }
}

/// Spawn `chatter sanity-scan` as a final post-loop step. Inherits
/// stdout/stderr so the operator sees the scan's per-session
/// `flagged` / `ok` lines + summary. Propagates the scan's exit
/// code: 0 if no flags, 4 if any flags, 1 on read/parse error.
fn run_sanity_scan_subprocess(
    self_exe: &Path,
    output_dir: &Path,
    override_path: &Path,
    anchor: &str,
    threshold: SanityScanThreshold,
    write_pending: &Path,
) {
    let status = Command::new(self_exe)
        .arg("sanity-scan")
        .arg(output_dir)
        .arg("--override-file")
        .arg(override_path)
        .arg("--anchor")
        .arg(anchor)
        .arg("--threshold")
        .arg(threshold.0.to_string())
        .arg("--write-pending")
        .arg(write_pending)
        .status();
    match status {
        Ok(s) => match s.code() {
            Some(code) => std::process::exit(code),
            // Signal-killed: propagate as input error (no useful
            // exit code to forward).
            None => std::process::exit(EXIT_INPUT_ERROR),
        },
        Err(e) => {
            warn!("sanity-scan subprocess spawn failed: {}", e);
            eprintln!("Error: sanity-scan subprocess spawn failed: {e}");
            std::process::exit(EXIT_INPUT_ERROR);
        }
    }
}

/// Per-session outcome from the subprocess driver. Discriminates the
/// three exit-code paths from `chatter pipeline` that the batch
/// summary cares about.
enum SessionOutcome {
    /// Pipeline exited 0 — merged file produced.
    Success,
    /// Pipeline exited 4 — speaker-id refused for low confidence.
    /// The pending entry was written (if `--write-pending` was
    /// supplied to the batch). The operator runs `chatter
    /// adjudicate` next.
    LowConfidence,
    /// Pipeline exited with another code — parse error, precondition
    /// violation, or subprocess failure. Carries the exit code for
    /// the summary.
    Error(i32),
}

/// Spawn one `chatter pipeline` subprocess for a session. The
/// per-session arg list mirrors the `Pipeline` subcommand's clap
/// surface.
fn run_pipeline_subprocess(self_exe: &Path, args: &PipelineArgs<'_>) -> SessionOutcome {
    let mut cmd = Command::new(self_exe);
    cmd.arg("pipeline")
        .arg(args.donor)
        .arg(args.reference)
        .arg("--anchor")
        .arg(args.anchor)
        .arg("--inserted-role")
        .arg(args.inserted_role)
        .arg("--confidence-threshold")
        .arg(args.confidence_threshold.to_string())
        .arg("-o")
        .arg(args.output);
    for r in args.retain {
        cmd.arg("--retain").arg(r);
    }
    if let Some(pending) = args.write_pending_path {
        cmd.arg("--write-pending").arg(pending);
    }
    if let Some(overrides) = args.override_file_path {
        cmd.arg("--override-file").arg(overrides);
    }
    if let Some(write_override) = args.write_override_path {
        cmd.arg("--write-override").arg(write_override);
    }
    match cmd.status() {
        Ok(status) => match status.code() {
            Some(code) if code == EXIT_SUCCESS => SessionOutcome::Success,
            Some(code) if code == EXIT_LOW_CONFIDENCE => SessionOutcome::LowConfidence,
            Some(code) => SessionOutcome::Error(code),
            // `None` means the subprocess was terminated by a signal
            // (no exit code). Encode as a sentinel that won't collide
            // with any valid CLI exit code.
            None => SessionOutcome::Error(-1),
        },
        Err(e) => {
            warn!("subprocess spawn failed: {}", e);
            SessionOutcome::Error(-2)
        }
    }
}

/// List every `*.cha` file directly under `dir`, sorted by path so
/// the batch's per-session order is deterministic.
fn list_cha_files(dir: &Path) -> Result<Vec<PathBuf>, std::io::Error> {
    let mut out: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension() == Some(OsStr::new("cha")) && path.is_file() {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}
