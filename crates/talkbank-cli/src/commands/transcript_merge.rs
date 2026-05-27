//! `chatter merge` — structural merge of two CHAT transcripts.
//!
//! Phase A cycle 1: thin CLI shim over
//! `talkbank_transform::transcript_merge::merge_chats`. Parses the
//! two input files, calls merge, writes output (or stdout).
//!
//! Refinement scheduled per cycle plan in
//! `book/src/architecture/merge-test-plan.md`.

use std::fs;
use std::path::{Path, PathBuf};
use tracing::{Level, info, span, warn};

use talkbank_transform::transcript_merge::{default_strip_tiers, merge_chats, MergeError};

/// Top-level entry for `chatter merge file1 file2 --retain <SPK[,SPK...]>`.
///
/// Cycle 1 is a happy-path-only smoke driver: it bails with exit
/// code 1 on any I/O or parse error and exit code 0 on success.
/// Later cycles introduce precondition-specific exit codes (2 for
/// retain-missing / language-mismatch / etc.) per the user-guide
/// contract.
pub fn run_merge(file1: &Path, file2: &Path, retain: &[String], output: Option<&PathBuf>) {
    let _span = span!(
        Level::INFO,
        "chatter_merge",
        file1 = %file1.display(),
        file2 = %file2.display(),
    )
    .entered();

    let content1 = match fs::read_to_string(file1) {
        Ok(s) => s,
        Err(e) => {
            warn!("failed to read {}: {}", file1.display(), e);
            eprintln!("Error reading {}: {}", file1.display(), e);
            std::process::exit(1);
        }
    };
    let content2 = match fs::read_to_string(file2) {
        Ok(s) => s,
        Err(e) => {
            warn!("failed to read {}: {}", file2.display(), e);
            eprintln!("Error reading {}: {}", file2.display(), e);
            std::process::exit(1);
        }
    };

    let options = talkbank_model::ParseValidateOptions::default();
    // Cycle 4: strip_tiers is configurable; CLI uses the default
    // set until a later cycle adds the `--strip-tiers` flag.
    let strip = default_strip_tiers();
    let merged = match merge_chats(&content1, &content2, retain, &strip, options) {
        Ok(s) => s,
        Err(e) => {
            warn!("merge failed: {}", e);
            eprintln!("Error: {}", e);
            // Exit-code mapping per the user-guide contract:
            // - precondition violations → 2
            // - invalid input (parse errors) → 1
            // Future MergeError variants from later precondition
            // cycles get explicit arms here.
            let code = match e {
                MergeError::RetainSpeakersMissing { .. } => 2,
                MergeError::NoTimelineInFile1 => 2,
                MergeError::Parse(_) => 1,
            };
            std::process::exit(code);
        }
    };

    match output {
        Some(path) => {
            if let Err(e) = fs::write(path, merged) {
                warn!("failed to write {}: {}", path.display(), e);
                eprintln!("Error writing {}: {}", path.display(), e);
                std::process::exit(1);
            }
            info!("wrote merged file: {}", path.display());
        }
        None => {
            print!("{merged}");
        }
    }
}
