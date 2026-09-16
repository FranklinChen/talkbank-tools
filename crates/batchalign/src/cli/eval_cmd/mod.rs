//! `batchalign3 eval ...`: evaluation subcommands.
//!
//! Offline evaluators. They consume retained artifacts, never submit ordinary
//! processing jobs, and never modify the artifacts they read. `eval
//! l2-morphotag` ports the Python analyzer that used to live in `scripts/`;
//! `eval transcribe-replay`, `eval utr-alignment` and `eval utseg-replay`
//! replay retained evidence through the current local pipeline. Future
//! evaluation tools (WER by corpus, morphotag for non-L2) land under this same
//! namespace.

pub mod l2_morphotag;
pub mod transcribe_replay;
pub mod utr_alignment;
pub mod utseg_replay;

use std::path::Path;

use serde::Serialize;

use crate::cli::args::{EvalAction, EvalArgs};
use crate::cli::error::CliError;

/// One artifact an evaluation report names, with the content digest it was
/// read at.
///
/// Shared by every evaluator, so one report cannot identify its inputs
/// differently from another, and the digest is always BA3's own content hash
/// rather than whatever the caller had to hand. Recording it is what makes a
/// report replayable: a later reader can tell whether the artifact it is
/// looking at is the one the report was computed from.
#[derive(Debug, Serialize)]
pub(crate) struct InputIdentity {
    path: String,
    bytes: usize,
    blake3: String,
}

impl InputIdentity {
    /// Identify `bytes`, as read from `path`.
    pub(crate) fn of(path: &Path, bytes: &[u8]) -> Self {
        Self {
            path: path.display().to_string(),
            bytes: bytes.len(),
            blake3: blake3::hash(bytes).to_hex().to_string(),
        }
    }
}

/// Dispatch an `eval ...` invocation to the right sub-handler.
pub async fn run(args: &EvalArgs) -> Result<(), CliError> {
    match &args.action {
        EvalAction::L2Morphotag(a) => l2_morphotag::run(a),
        EvalAction::TranscribeReplay(a) => transcribe_replay::run(a).await,
        EvalAction::UtrAlignment(a) => utr_alignment::run(a),
        EvalAction::UtsegReplay(a) => utseg_replay::run(a),
    }
}
