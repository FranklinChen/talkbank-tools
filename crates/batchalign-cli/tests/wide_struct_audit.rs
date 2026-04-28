//! Thin proxy — delegates to `cargo xtask lint-wide-structs`.
//!
//! The actual audit logic lives in `xtask/src/wide_struct_audit.rs` to avoid
//! compiling a full integration test binary just for a structural lint.

use std::process::Command;

#[test]
fn wide_struct_audit_passes() {
    let status = Command::new("cargo")
        .args(["xtask", "lint-wide-structs"])
        .status()
        .expect("failed to run cargo xtask lint-wide-structs");
    assert!(status.success(), "cargo xtask lint-wide-structs failed");
}
