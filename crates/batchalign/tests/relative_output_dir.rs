//! A relative `-o` must not nest the output directory inside itself.
//!
//! Field report (external user, 2026-08-11): `batchalign3 <cmd> A -o B`
//! produced a second `B` inside `B`. The processed transcripts landed in
//! `B/B/` while the copied non-matching files landed in `B/`, so every run
//! left two output trees and the operator deleted one by hand.
//!
//! The cause was a double join: the discovery pass already roots every
//! output at the requested output directory (`out_dir.join(rel)`), and the
//! writer then rooted the result a second time whenever the path it was
//! handed was relative. An absolute `-o` took the other branch and was
//! always correct, which is why every existing test (all of them using an
//! absolute `tempfile::tempdir()`) passed.
//!
//! This test owns its own process, because cargo builds each integration
//! test file as a separate binary, so it may set the current directory. A
//! relative `-o` has no meaning without a cwd, and the cwd is exactly the
//! axis the defect lived on.
// Integration tests are exempt from the crate's deny-level panic lints,
// matching the src/lib.rs `#![cfg_attr(test, allow(...))]` pattern
// (see docs/panic-audit/).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::todo,
    clippy::unimplemented
)]

use std::path::{Path, PathBuf};

use batchalign::api::{ContentType, FileResult};
use batchalign::cli::args::InputKind;
use batchalign::cli::discover::{build_server_names, discover_server_inputs};
use batchalign::cli::output::write_result;

/// Exercises the real client-side path composition: discovery builds the
/// output paths and the name map exactly as the dispatcher does, then the
/// writer places a returned result. Only the daemon round trip is elided.
///
/// Both cases live in one test function on purpose. They each depend on the
/// process-wide current directory, so running them as two `#[test]`s would
/// race in cargo's default thread-per-test harness.
#[test]
fn relative_output_dir_is_not_nested_inside_itself() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    std::env::set_current_dir(tmp.path()).expect("chdir into temp dir");

    // The operator's layout: input `A`, output `B`, both relative.
    std::fs::create_dir_all("A").expect("create input dir");
    std::fs::write("A/session.cha", "@UTF8\n@Begin\n@End\n").expect("write input file");

    // Case 1: the output directory already exists, which is what the real
    // dispatcher guarantees (it creates every output parent before
    // submitting). This is the case the field report hit.
    std::fs::create_dir_all("B").expect("create output dir");
    write_one_result(Path::new("B"));
    assert!(
        Path::new("B/session.cha").is_file(),
        "the output belongs directly in the requested output directory"
    );
    assert!(
        !Path::new("B/B").exists(),
        "the output directory must not be created inside itself"
    );

    // Case 2: the output directory does not exist yet. The writer creates
    // parents itself, so this must succeed rather than fail the
    // containment check against a path it never canonicalized.
    write_one_result(Path::new("C"));
    assert!(
        Path::new("C/session.cha").is_file(),
        "a not-yet-existing output directory is created, not rejected"
    );
    assert!(
        !Path::new("C/C").exists(),
        "the output directory must not be created inside itself"
    );
}

/// Run discovery plus a single result write against `out_dir`, exactly as
/// the dispatcher does for one completed file.
fn write_one_result(out_dir: &Path) {
    let inputs = vec![PathBuf::from("A")];

    let (files, outputs) =
        discover_server_inputs(&inputs, Some(out_dir), InputKind::Chat).expect("discover inputs");
    assert_eq!(files.len(), 1, "one input file was planted");

    let (server_names, result_map) =
        build_server_names(&files, &outputs, &inputs).expect("build server names");

    let result = FileResult {
        filename: server_names[0].as_str().into(),
        content: "@UTF8\n@Begin\n*CHI:\thello .\n@End\n".to_string(),
        content_type: ContentType::Chat,
        error: None,
        provenance: batchalign::api::FileProvenance::NotRead,
    };

    let written = write_result(&result, &result_map, out_dir).expect("write result");
    assert!(written, "a result with no error must be written");
}
