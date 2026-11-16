//! Shared native worker-fixture startup bytes.

/// A minimal, well-formed `ready` line, the same shape
/// `worker::handle::lifecycle`'s own tests use, PID hardcoded since
/// nothing here asserts on it.
pub(crate) fn ready_line() -> String {
    format!(
        concat!(
            "{{\"ready\":true,\"pid\":424242,\"transport\":\"stdio\",\"runtime\":{{",
            "\"schema_version\":1,\"python_version\":\"3.13.12\",",
            "\"python_executable_sha256\":\"{}\",",
            "\"batchalign_package_tree_sha256\":\"{}\",",
            "\"batchalign_core_extension_sha256\":\"{}\",",
            "\"distribution_inventory_sha256\":\"{}\"}}}}\n"
        ),
        "a".repeat(64),
        "b".repeat(64),
        "c".repeat(64),
        "d".repeat(64),
    )
}
