//! Env-gated real-file integration tests for segment-aware timing drift.
//!
//! These tests run the full `batchalign3 align` pipeline on the real files that
//! motivated the segment-aware UTR investigation. Audio is large
//! (multi-hundred MB); we do not commit it. Instead a contributor stages the
//! corpus locally and points `BATCHALIGN3_DRIFT_CORPUS_DIR` at the staging
//! directory.
//!
//! Staging layout (each subdir contains `<stem>.cha` plus an adjacent media
//! file: symlinked to NFS is fine):
//!
//! ```text
//! $BATCHALIGN3_DRIFT_CORPUS_DIR/
//!   micase/   { adv700ju047.cha + .mp3, ... 8 files }
//!   samtale/  { seiden2.cha, seiden2.mp4 }
//!   biling/   { 28.cha, 28.mp3 }         (MLE-MPF/28, fra/eng bilingual,
//!                                         62% +< density, chosen because
//!                                         Koge has no NFS media)
//!   rhd/      { minga044.cha, minga044.<ext> }
//! ```
//!
//! The four tests are `#[ignore]`d: the corpus is private, so a default run
//! reports them as ignored rather than as passes. An explicit `--ignored` run
//! fails when the env var is unset, when a subdirectory is missing or empty,
//! or when a staged file lacks its media (I-2: no vacuous pass).
//!
//! These tests are expected to fail on the current GlobalUtr pipeline (MICASE
//! crashes with `InvalidAudioWindow`; samtale strips many silent timings;
//! biling/rhd similar drift). Phase 3 of the segment-aware UTR rewrite is
//! what flips them green.
//!
//! ## Module layout
//!
//! - [`crate::common::drift_staging`]: filesystem plumbing
//!   (`require_drift_corpus_dir`, `StagedDriftFile`, `enumerate_staged_files`,
//!   `locate_adjacent_media`, `CorpusFileName`).
//! - [`crate::ml_golden::align::drift_runner`]: align-pipeline driver and
//!   outcome aggregator (`FileOutcome`, `run_one_file`,
//!   `assert_all_files_pass`).
//! - This file: test bodies only.

use batchalign::worker::InferTask;

use crate::common::drift_staging::{enumerate_staged_files, require_drift_corpus_dir};
use crate::common::{LiveDirectJobClient, require_live_direct};
use crate::ml_golden::align::drift_runner::{FileOutcome, assert_all_files_pass, run_one_file};

#[tokio::test]
#[ignore = "private drift corpus: set BATCHALIGN3_DRIFT_CORPUS_DIR and run with --ignored"]
async fn drift_micase_all_eight_failures() {
    let root = require_drift_corpus_dir();
    let micase_dir = root.join("micase");
    if !micase_dir.is_dir() {
        panic!(
            "PREREQUISITE MISSING: drift_micase_all_eight_failures: {} does not exist",
            micase_dir.display()
        )
    }
    let session = require_live_direct(
        InferTask::Fa,
        "drift_micase_all_eight_failures: direct session does not support FA infer",
    )
    .await;
    let jobs = LiveDirectJobClient::new(&session);

    let expected = [
        "adv700ju047",
        "col999mg053",
        "les405jg078",
        "mtg270sg049",
        "ofc270mg048",
        "sgr200ju125",
        "sgr999mx115",
        "tou999mx062",
    ];
    let staged = enumerate_staged_files(&micase_dir, &expected);
    if staged.is_empty() {
        panic!(
            "PREREQUISITE MISSING: drift_micase_all_eight_failures: no .cha files found in {}",
            micase_dir.display()
        )
    }

    let mut outcomes: Vec<FileOutcome> = Vec::new();
    for f in &staged {
        outcomes.push(run_one_file(&jobs, f, "eng").await);
    }
    assert_all_files_pass("drift_micase_all_eight_failures", &outcomes);
}

#[tokio::test]
#[ignore = "private drift corpus: set BATCHALIGN3_DRIFT_CORPUS_DIR and run with --ignored"]
async fn drift_samtale_seiden2() {
    let root = require_drift_corpus_dir();
    let dir = root.join("samtale");
    if !dir.is_dir() {
        panic!(
            "PREREQUISITE MISSING: drift_samtale_seiden2: {} does not exist",
            dir.display()
        )
    }
    let session = require_live_direct(
        InferTask::Fa,
        "drift_samtale_seiden2: direct session does not support FA infer",
    )
    .await;
    let jobs = LiveDirectJobClient::new(&session);

    let staged = enumerate_staged_files(&dir, &["seiden2"]);
    if staged.is_empty() {
        panic!(
            "PREREQUISITE MISSING: drift_samtale_seiden2: no .cha files found in {}",
            dir.display()
        )
    }
    let mut outcomes: Vec<FileOutcome> = Vec::new();
    for f in &staged {
        outcomes.push(run_one_file(&jobs, f, "dan").await);
    }
    assert_all_files_pass("drift_samtale_seiden2", &outcomes);
}

#[tokio::test]
#[ignore = "private drift corpus: set BATCHALIGN3_DRIFT_CORPUS_DIR and run with --ignored"]
async fn drift_biling_mle_mpf_28() {
    // MLE-MPF/28.cha-855 `+<` markers across 1369 utterances (62% density).
    // Chosen over biling-data/Koge/danish-group/306a-gs.cha (also 62% density)
    // because Koge has no matching media on NFS, while MLE-MPF/28.mp3 does.
    // Languages declared in the CHAT header: `fra, eng`, we pass `fra` to
    // the Rev.AI engine, which supports French.
    let root = require_drift_corpus_dir();
    let dir = root.join("biling");
    if !dir.is_dir() {
        panic!(
            "PREREQUISITE MISSING: drift_biling_mle_mpf_28: {} does not exist",
            dir.display()
        )
    }
    let session = require_live_direct(
        InferTask::Fa,
        "drift_biling_mle_mpf_28: direct session does not support FA infer",
    )
    .await;
    let jobs = LiveDirectJobClient::new(&session);

    let staged = enumerate_staged_files(&dir, &["28"]);
    if staged.is_empty() {
        panic!(
            "PREREQUISITE MISSING: drift_biling_mle_mpf_28: no .cha files found in {}",
            dir.display()
        )
    }
    let mut outcomes: Vec<FileOutcome> = Vec::new();
    for f in &staged {
        outcomes.push(run_one_file(&jobs, f, "fra").await);
    }
    assert_all_files_pass("drift_biling_mle_mpf_28", &outcomes);
}

#[tokio::test]
#[ignore = "private drift corpus: set BATCHALIGN3_DRIFT_CORPUS_DIR and run with --ignored"]
async fn drift_rhd_minga044() {
    let root = require_drift_corpus_dir();
    let dir = root.join("rhd");
    if !dir.is_dir() {
        panic!(
            "PREREQUISITE MISSING: drift_rhd_minga044: {} does not exist",
            dir.display()
        )
    }
    let session = require_live_direct(
        InferTask::Fa,
        "drift_rhd_minga044: direct session does not support FA infer",
    )
    .await;
    let jobs = LiveDirectJobClient::new(&session);

    let staged = enumerate_staged_files(&dir, &["minga044"]);
    if staged.is_empty() {
        panic!(
            "PREREQUISITE MISSING: drift_rhd_minga044: no .cha files found in {}",
            dir.display()
        )
    }
    let mut outcomes: Vec<FileOutcome> = Vec::new();
    for f in &staged {
        outcomes.push(run_one_file(&jobs, f, "eng").await);
    }
    assert_all_files_pass("drift_rhd_minga044", &outcomes);
}
