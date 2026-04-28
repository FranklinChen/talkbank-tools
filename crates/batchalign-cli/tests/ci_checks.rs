//! Thin proxy — delegates to `cargo run -q -p xtask -- lint-ci-hygiene`.
//!
//! The actual hygiene checks (version sync, legacy terms, retired packages)
//! live in `xtask/src/ci_hygiene.rs` to avoid compiling a full integration
//! test binary just for structural lints.

use std::process::Command;

#[test]
fn ci_hygiene_passes() {
    let status = Command::new("cargo")
        .args(["run", "-q", "-p", "xtask", "--", "lint-ci-hygiene"])
        .status()
        .expect("failed to run cargo run -q -p xtask -- lint-ci-hygiene");
    assert!(
        status.success(),
        "cargo run -q -p xtask -- lint-ci-hygiene failed"
    );
}
