//! `chatter pipeline` — per-session end-to-end: relabel an
//! anonymous donor via reference-mode speaker-id, then merge the
//! result with the reference. The single-call shortcut for the
//! common case of "I have one donor and one reference and want the
//! final merged file."
//!
//! Thin orchestrator: invokes `run_reference_mode` (from the
//! `speaker_id` shim) to relabel the donor, then calls the
//! library-level `merge_chats` directly. LowConfidence / pending /
//! parse-error / precondition exit codes all bubble through the
//! existing `speaker_id` and `transcript_merge` exit machinery.

use std::fs;
use std::path::Path;

use tracing::{Level, info, span, warn};

use crate::exit_codes::{EXIT_INPUT_ERROR, EXIT_PRECONDITION};
use talkbank_model::{ParseValidateOptions, SpeakerCode};
use talkbank_transform::speaker_id::{ConfidenceThreshold, OverrideFile};
use talkbank_transform::transcript_merge::{default_strip_tiers, merge_chats};

use super::speaker_id::{
    ReferenceModeArgs, apply_override_entry, derive_session_id, run_reference_mode,
    write_override_entry,
};

/// All inputs for one `chatter pipeline` invocation.
///
/// Pulled into a struct so the call surface stays readable as
/// pipeline grows new operator-facing knobs, and so the batch
/// subprocess driver in [`super::batch`] can construct one per
/// session without a 10-arg helper signature.
///
/// Borrows are tied to the caller's allocations — the struct never
/// outlives a single invocation of [`run_pipeline`].
pub struct PipelineArgs<'a> {
    /// Donor CHAT file to relabel + merge into the reference.
    pub donor: &'a Path,
    /// Reference CHAT file providing the canonical speaker set.
    pub reference: &'a Path,
    /// Reference speaker code that survives intact (typically `CHI`).
    pub anchor: &'a str,
    /// Role spec for inserted donor speakers, formatted `CODE:Role`
    /// (e.g. `INV:Investigator`).
    pub inserted_role: &'a str,
    /// Donor speaker codes whose lines must survive the merge.
    pub retain: &'a [String],
    /// Minimum Jaccard-margin confidence accepted by speaker-id;
    /// lower margins refuse to a pending entry instead of merging.
    pub confidence_threshold: f64,
    /// If set, low-confidence sessions append a pending entry here
    /// rather than failing the operator pipeline silently.
    pub write_pending_path: Option<&'a Path>,
    /// If set, sessions with a matching entry replay the recorded
    /// mapping instead of re-running reference mode.
    pub override_file_path: Option<&'a Path>,
    /// If set and reference mode produces a clean-winner merge, the
    /// auto-decision is appended to this file with `mode = "auto"`.
    /// Distinct from `override_file_path`: that one is read for
    /// replay; this one is written for audit.
    pub write_override_path: Option<&'a Path>,
    /// Destination for the final merged CHAT file.
    pub output: &'a Path,
}

/// Top-level entry for `chatter pipeline`.
///
/// Exit-code contract:
/// - 0: relabeled + merged output written.
/// - 1: I/O or parse error on donor / reference.
/// - 2: precondition violation (`merge` retain set missing, language
///   mismatch, ambiguous speaker; or `speaker-id` reference missing
///   anchor / too few donor speakers).
/// - 4: speaker-id low confidence (with pending entry written if
///   `--write-pending` supplied). No merged output produced.
pub fn run_pipeline(args: PipelineArgs<'_>) {
    let PipelineArgs {
        donor,
        reference,
        anchor,
        inserted_role,
        retain,
        confidence_threshold,
        write_pending_path,
        override_file_path,
        write_override_path,
        output,
    } = args;
    let _span = span!(
        Level::INFO,
        "chatter_pipeline",
        donor = %donor.display(),
        reference = %reference.display(),
    )
    .entered();

    let donor_content = match fs::read_to_string(donor) {
        Ok(s) => s,
        Err(e) => {
            warn!("failed to read donor {}: {}", donor.display(), e);
            eprintln!("Error reading {}: {}", donor.display(), e);
            std::process::exit(EXIT_INPUT_ERROR);
        }
    };
    let reference_content = match fs::read_to_string(reference) {
        Ok(s) => s,
        Err(e) => {
            warn!("failed to read reference {}: {}", reference.display(), e);
            eprintln!("Error reading {}: {}", reference.display(), e);
            std::process::exit(EXIT_INPUT_ERROR);
        }
    };

    let options = ParseValidateOptions::default();
    let session_id = derive_session_id(donor);
    // Pre-parse the override file when one is supplied so the replay
    // path doesn't re-read it. Read errors here (schema-version
    // mismatch, malformed TOML) fall through to reference mode, which
    // surfaces a real error via its own machinery if the run fails.
    let override_file_loaded =
        override_file_path.and_then(|p| OverrideFile::read_or_default(p).ok());
    let override_entry = override_file_loaded
        .as_ref()
        .and_then(|f| f.get(&session_id));
    let relabeled = match override_entry {
        Some(entry) => apply_override_entry(&donor_content, entry, options.clone()),
        None => {
            let outcome = run_reference_mode(ReferenceModeArgs {
                donor_content: &donor_content,
                reference_path: reference,
                anchor,
                inserted_role_spec: inserted_role,
                threshold: ConfidenceThreshold(confidence_threshold),
                write_pending_path,
                input_path: donor,
                options: options.clone(),
            });
            if let Some(path) = write_override_path {
                write_override_entry(path, donor, &outcome);
            }
            outcome.relabeled
        }
    };

    let retain_codes: Vec<SpeakerCode> = retain.iter().map(SpeakerCode::new).collect();
    let strip = default_strip_tiers();
    let merged = match merge_chats(
        &reference_content,
        &relabeled,
        &retain_codes,
        &strip,
        options,
    ) {
        Ok(s) => s,
        Err(e) => {
            warn!("merge step failed: {}", e);
            eprintln!("Error: {}", e);
            // Mirror the exit-code contract of `chatter merge`:
            // precondition violations → 2, parse errors → 1.
            let code = match e {
                talkbank_transform::transcript_merge::MergeError::Parse(_) => EXIT_INPUT_ERROR,
                _ => EXIT_PRECONDITION,
            };
            std::process::exit(code);
        }
    };

    if let Err(e) = fs::write(output, merged) {
        warn!("failed to write {}: {}", output.display(), e);
        eprintln!("Error writing {}: {}", output.display(), e);
        std::process::exit(EXIT_INPUT_ERROR);
    }
    info!("wrote pipeline output: {}", output.display());
}
