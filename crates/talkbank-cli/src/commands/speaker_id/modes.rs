//! The three operation modes of `chatter speaker-id`:
//! explicit-mapping (manual `--mapping`), reference-mode
//! (text-similarity), and override-file replay. Each returns the
//! relabeled CHAT text or `std::process::exit`s with the contract
//! exit code.

use std::fs;
use std::path::Path;
use tracing::warn;

use crate::exit_codes::EXIT_INPUT_ERROR;
use talkbank_model::{ParseValidateOptions, ParticipantRole, SpeakerCode};
use talkbank_transform::parse_and_validate;
use talkbank_transform::speaker_id::{
    ConfidenceThreshold, MappingSpec, MergeOverride, OverrideFile, SpeakerAssignment,
    SpeakerIdError, apply_mapping, apply_mapping_chat, identify_mapping, parse_mapping_spec,
};

use super::support::{
    exit_with_override_file_error, exit_with_speaker_id_error, parse_inserted_role,
};
use super::writes::write_pending_entry;

/// Carries the relabeled CHAT plus everything an override-file entry
/// needs to record about the decision. Exposed `pub(crate)` so the
/// per-session `chatter pipeline` shim can reuse the
/// reference-mode helpers without duplicating their LowConfidence /
/// `--write-pending` handling.
pub(crate) struct ReferenceModeOutcome {
    pub(crate) relabeled: String,
    pub(crate) report: talkbank_transform::speaker_id::DonorMatchReport,
    pub(crate) mapping: MappingSpec,
    pub(crate) inserted_code: SpeakerCode,
    pub(crate) inserted_role_tag: ParticipantRole,
}

/// Explicit-mapping mode: parse the `--mapping` spec, apply it,
/// return the relabeled CHAT text.
pub(super) fn run_explicit_mode(
    content: &str,
    spec: &str,
    options: ParseValidateOptions,
) -> String {
    let mapping = match parse_mapping_spec(spec) {
        Ok(m) => m,
        Err(e) => {
            warn!("mapping parse failed: {}", e);
            eprintln!("Error: {}", e);
            std::process::exit(crate::exit_codes::EXIT_PRECONDITION);
        }
    };
    match apply_mapping(content, &mapping, options) {
        Ok(s) => s,
        Err(e) => exit_with_speaker_id_error(e),
    }
}

/// Reference mode: identify the donor speaker matching the reference
/// anchor, build a mapping (winner → drop; non-winner → inserted
/// role), apply. The returned outcome carries both the relabeled
/// text and the data needed to write an override-file entry.
///
/// If `write_pending_path` is `Some` and `identify_mapping` returns
/// `SpeakerIdError::LowConfidence`, a pending-adjudication entry is
/// appended to the named file before exiting with code 4. The pending
/// entry's `suggested` field carries the algorithm's would-have-been
/// decision so the operator can accept-as-is in `chatter adjudicate`.
/// All inputs to one reference-mode invocation. Constructed by the
/// CLI orchestrators (`chatter speaker-id` and `chatter pipeline`)
/// from their respective clap surfaces.
pub(crate) struct ReferenceModeArgs<'a> {
    /// Already-loaded donor CHAT text (the caller's `fs::read_to_string`
    /// result).
    pub donor_content: &'a str,
    /// Reference CHAT file to load + parse.
    pub reference_path: &'a Path,
    /// Reference anchor speaker code (typically `CHI`).
    pub anchor: &'a str,
    /// Inserted-role spec for non-anchor donor speakers.
    pub inserted_role_spec: &'a str,
    /// Jaccard winner→runner-up margin threshold.
    pub threshold: ConfidenceThreshold,
    /// If set, low-confidence refusals append a pending entry here
    /// before exit 4.
    pub write_pending_path: Option<&'a Path>,
    /// Donor input path — needed for the pending entry's session ID
    /// derivation.
    pub input_path: &'a Path,
    /// Parser options threaded through to `parse_and_validate`.
    pub options: ParseValidateOptions,
}

pub(crate) fn run_reference_mode(args: ReferenceModeArgs<'_>) -> ReferenceModeOutcome {
    let ReferenceModeArgs {
        donor_content,
        reference_path,
        anchor,
        inserted_role_spec,
        threshold,
        write_pending_path,
        input_path,
        options,
    } = args;
    let reference_content = match fs::read_to_string(reference_path) {
        Ok(s) => s,
        Err(e) => {
            warn!("failed to read {}: {}", reference_path.display(), e);
            eprintln!("Error reading {}: {}", reference_path.display(), e);
            std::process::exit(EXIT_INPUT_ERROR);
        }
    };

    let donor_chat = match parse_and_validate(donor_content, options.clone()) {
        Ok(c) => c,
        Err(e) => {
            warn!("donor parse failed: {}", e);
            eprintln!("Error parsing donor: {}", e);
            std::process::exit(EXIT_INPUT_ERROR);
        }
    };
    let reference_chat = match parse_and_validate(&reference_content, options.clone()) {
        Ok(c) => c,
        Err(e) => {
            warn!("reference parse failed: {}", e);
            eprintln!("Error parsing reference: {}", e);
            std::process::exit(EXIT_INPUT_ERROR);
        }
    };

    // Parse `--inserted-role` upfront so it's available on both the
    // happy path AND the low-confidence path that writes the pending
    // entry's `suggested.inserted_role`.
    let (inserted_code, inserted_role) = match parse_inserted_role(inserted_role_spec) {
        Ok(pair) => pair,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(crate::exit_codes::EXIT_PRECONDITION);
        }
    };

    let anchor_code = SpeakerCode::new(anchor);
    let report = match identify_mapping(&reference_chat, &anchor_code, &donor_chat, threshold) {
        Ok(r) => r,
        Err(SpeakerIdError::LowConfidence { report, threshold }) => {
            if let Some(pending_path) = write_pending_path {
                write_pending_entry(
                    pending_path,
                    input_path,
                    &report,
                    threshold,
                    &donor_chat,
                    &inserted_code,
                    &inserted_role,
                );
            }
            exit_with_speaker_id_error(SpeakerIdError::LowConfidence { report, threshold })
        }
        Err(e) => exit_with_speaker_id_error(e),
    };

    let mut mapping = MappingSpec::new();
    mapping.insert(report.winner.clone(), SpeakerAssignment::Drop);
    for spk in donor_chat.unique_utterance_speakers() {
        if spk != report.winner {
            mapping.insert(
                spk,
                SpeakerAssignment::Rename {
                    code: inserted_code.clone(),
                    role: inserted_role.clone(),
                },
            );
        }
    }

    // Reuse the already-parsed donor AST — `apply_mapping_chat`
    // skips the redundant second `parse_and_validate` that the
    // string-entry `apply_mapping` would otherwise do.
    let relabeled = apply_mapping_chat(&donor_chat, &mapping);

    ReferenceModeOutcome {
        relabeled,
        report,
        mapping,
        inserted_code,
        inserted_role_tag: inserted_role,
    }
}

/// Override-file replay mode: load the file, look up the recorded
/// entry by session ID, apply it. Standalone-CLI entry point.
pub(crate) fn run_override_file_mode(
    input_content: &str,
    override_path: &Path,
    session_id: &str,
    options: ParseValidateOptions,
) -> String {
    let file = match OverrideFile::read_or_default(override_path) {
        Ok(f) => f,
        Err(e) => exit_with_override_file_error(override_path, e),
    };
    let entry = match file.get(session_id) {
        Some(e) => e,
        None => {
            let available: Vec<String> = file.session_ids().map(str::to_string).collect();
            exit_with_speaker_id_error(SpeakerIdError::SessionIdNotFound {
                session_id: session_id.to_string(),
                available,
            })
        }
    };
    apply_override_entry(input_content, entry, options)
}

/// Apply an already-loaded override entry to donor content. Pipeline
/// callers that have already parsed the override file once skip the
/// re-read by going through this function directly.
pub(crate) fn apply_override_entry(
    input_content: &str,
    entry: &MergeOverride,
    options: ParseValidateOptions,
) -> String {
    let mapping = entry.to_mapping_spec();
    match apply_mapping(input_content, &mapping, options) {
        Ok(s) => s,
        Err(e) => exit_with_speaker_id_error(e),
    }
}
