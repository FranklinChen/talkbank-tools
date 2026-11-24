//! Content-addressed identity reported by a Python worker at startup.
//!
//! The wire type deliberately contains no filesystem paths. A successfully
//! parsed value proves the schema revision and all four full SHA-256 shapes;
//! callers cannot construct the identity from arbitrary strings.

use serde::{Deserialize, Serialize};
use std::sync::Mutex;

use batchalign_types::evidence::Sha256Digest;

/// Runtime identity proven by the current ready-handshake schema.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct WorkerRuntimeIdentity {
    schema_version: u8,
    python_version: String,
    python_executable_sha256: Sha256Digest,
    batchalign_package_tree_sha256: Sha256Digest,
    batchalign_core_extension_sha256: Sha256Digest,
    distribution_inventory_sha256: Sha256Digest,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawWorkerRuntimeIdentity {
    schema_version: u8,
    python_version: String,
    python_executable_sha256: Sha256Digest,
    batchalign_package_tree_sha256: Sha256Digest,
    batchalign_core_extension_sha256: Sha256Digest,
    distribution_inventory_sha256: Sha256Digest,
}

impl<'de> Deserialize<'de> for WorkerRuntimeIdentity {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = RawWorkerRuntimeIdentity::deserialize(deserializer)?;
        if raw.schema_version != 1 {
            return Err(serde::de::Error::custom(
                "worker runtime identity schema_version must be 1",
            ));
        }
        if raw.python_version.trim().is_empty() {
            return Err(serde::de::Error::custom(
                "worker runtime python_version must not be empty",
            ));
        }
        Ok(Self {
            schema_version: raw.schema_version,
            python_version: raw.python_version,
            python_executable_sha256: raw.python_executable_sha256,
            batchalign_package_tree_sha256: raw.batchalign_package_tree_sha256,
            batchalign_core_extension_sha256: raw.batchalign_core_extension_sha256,
            distribution_inventory_sha256: raw.distribution_inventory_sha256,
        })
    }
}

impl WorkerRuntimeIdentity {
    /// Python semantic version reported by the executing interpreter.
    #[must_use]
    pub fn python_version(&self) -> &str {
        &self.python_version
    }

    /// Content digest of the resolved Python executable.
    #[must_use]
    pub const fn python_executable_sha256(&self) -> &Sha256Digest {
        &self.python_executable_sha256
    }
}

/// Process-lifetime runtime identity pinned by the first admitted worker.
///
/// A server that ran two Python code identities could not attribute a result
/// to either one from its health receipt. The pool therefore admits the first
/// identity and refuses every later mismatch; mixed-runtime execution is
/// unrepresentable after admission succeeds.
#[derive(Default)]
pub(crate) struct ObservedWorkerRuntimes {
    identity: Mutex<Option<WorkerRuntimeIdentity>>,
}

/// A worker cannot join a pool already pinned to different executable code.
///
/// Keeps both identities, and its message names the components that differ
/// with both values, so a refusal says what changed. It used to be converted
/// to a variant that carried neither, and the 2026-11-05 refusal (a worker
/// whose distribution inventory changed because loading `FunASR` extended
/// `sys.path` before it observed its identity) had to be found by experiment.
#[derive(Debug)]
pub struct WorkerRuntimeMismatch {
    expected: Box<WorkerRuntimeIdentity>,
    observed: Box<WorkerRuntimeIdentity>,
}

impl WorkerRuntimeMismatch {
    /// Each component that differs: its wire name, the pinned value and the
    /// worker's.
    fn differences(&self) -> Vec<(&'static str, String, String)> {
        // Destructured exhaustively, so a field added to the identity cannot
        // be left out of the message: `admit` would refuse on it while the
        // refusal named nothing.
        let WorkerRuntimeIdentity {
            schema_version: pinned_schema,
            python_version: pinned_python,
            python_executable_sha256: pinned_executable,
            batchalign_package_tree_sha256: pinned_package,
            batchalign_core_extension_sha256: pinned_extension,
            distribution_inventory_sha256: pinned_inventory,
        } = &*self.expected;
        let WorkerRuntimeIdentity {
            schema_version: worker_schema,
            python_version: worker_python,
            python_executable_sha256: worker_executable,
            batchalign_package_tree_sha256: worker_package,
            batchalign_core_extension_sha256: worker_extension,
            distribution_inventory_sha256: worker_inventory,
        } = &*self.observed;
        [
            (
                "schema_version",
                pinned_schema.to_string(),
                worker_schema.to_string(),
            ),
            (
                "python_version",
                pinned_python.clone(),
                worker_python.clone(),
            ),
            (
                "python_executable_sha256",
                pinned_executable.as_str().to_owned(),
                worker_executable.as_str().to_owned(),
            ),
            (
                "batchalign_package_tree_sha256",
                pinned_package.as_str().to_owned(),
                worker_package.as_str().to_owned(),
            ),
            (
                "batchalign_core_extension_sha256",
                pinned_extension.as_str().to_owned(),
                worker_extension.as_str().to_owned(),
            ),
            (
                "distribution_inventory_sha256",
                pinned_inventory.as_str().to_owned(),
                worker_inventory.as_str().to_owned(),
            ),
        ]
        .into_iter()
        .filter(|(_, pinned, worker)| pinned != worker)
        .collect()
    }
}

impl std::fmt::Display for WorkerRuntimeMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("worker runtime identity differs from the runtime pinned by this server")?;
        for (index, (name, pinned, worker)) in self.differences().into_iter().enumerate() {
            let separator = if index == 0 { ": " } else { "; " };
            write!(f, "{separator}{name} pinned {pinned}, worker {worker}")?;
        }
        Ok(())
    }
}

impl std::error::Error for WorkerRuntimeMismatch {}

impl ObservedWorkerRuntimes {
    /// Pin the first runtime and refuse a different runtime before job use.
    pub(crate) fn admit(
        &self,
        identity: &WorkerRuntimeIdentity,
    ) -> Result<(), WorkerRuntimeMismatch> {
        let mut admitted = self
            .identity
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match admitted.as_ref() {
            Some(expected) if expected != identity => Err(WorkerRuntimeMismatch {
                expected: Box::new(expected.clone()),
                observed: Box::new(identity.clone()),
            }),
            Some(_) => Ok(()),
            None => {
                *admitted = Some(identity.clone());
                Ok(())
            }
        }
    }

    /// Return the path-free pinned receipt for the health endpoint.
    pub(crate) fn receipts(&self) -> Vec<WorkerRuntimeIdentity> {
        self.identity
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .cloned()
            .into_iter()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker_report(executable_digest: &str) -> serde_json::Value {
        serde_json::json!({
            "schema_version": 1,
            "python_version": "3.14.0",
            "python_executable_sha256": executable_digest,
            "batchalign_package_tree_sha256": "b".repeat(64),
            "batchalign_core_extension_sha256": "c".repeat(64),
            "distribution_inventory_sha256": "d".repeat(64),
        })
    }

    /// Guards: Runtime admission requires the current schema and every digest.
    /// A successful identity serializes exactly the declared evidence fields.
    /// Kind: wire
    /// Not a type: Valid fields are private after admission. Incoming JSON can
    /// still omit, rename or corrupt evidence, and serialization can leak fields.
    /// State: One valid report feeds both the round trip and single-field mutations.
    /// Concurrency: All reports are local values. No live process or path is read.
    #[test]
    fn runtime_wire_admits_exactly_the_reported_evidence() {
        enum WireFault {
            FutureSchema,
            InvalidExecutableDigest,
            ForeignPath,
            MissingExtensionDigest,
        }

        let report = worker_report(&"a".repeat(64));
        let admitted: WorkerRuntimeIdentity = serde_json::from_value(report.clone()).unwrap();
        assert_eq!(serde_json::to_value(admitted).unwrap(), report);

        for fault in [
            WireFault::FutureSchema,
            WireFault::InvalidExecutableDigest,
            WireFault::ForeignPath,
            WireFault::MissingExtensionDigest,
        ] {
            let mut value = report.clone();
            let fields = value.as_object_mut().unwrap();
            match fault {
                WireFault::FutureSchema => {
                    fields.insert("schema_version".into(), serde_json::json!(2));
                }
                WireFault::InvalidExecutableDigest => {
                    fields.insert(
                        "python_executable_sha256".into(),
                        serde_json::json!("not-a-digest"),
                    );
                }
                WireFault::ForeignPath => {
                    fields.insert("path".into(), serde_json::json!("/tmp/python"));
                }
                WireFault::MissingExtensionDigest => {
                    fields.remove("batchalign_core_extension_sha256");
                }
            }
            assert!(serde_json::from_value::<WorkerRuntimeIdentity>(value).is_err());
        }
    }

    /// Guards: Repeated runtime reports reuse the pinned identity. A differing
    /// identity preserves the original receipt and names the differing component.
    /// Kind: policy
    /// Not a type: Two actual admitted values can differ despite having the same
    /// type. The registry must compare those values and render the real difference.
    /// State: The test owns one registry and two independently admitted reports.
    /// Concurrency: A private registry exercises the normal admission method in
    /// deterministic order with no competing writers or live worker.
    #[test]
    fn runtime_pin_retains_the_admitted_report_and_names_a_refusal() {
        let registry = ObservedWorkerRuntimes::default();
        let original_digest = "a".repeat(64);
        let different_digest = "d".repeat(64);
        let first = serde_json::from_value(worker_report(&original_digest)).unwrap();
        let second = serde_json::from_value(worker_report(&different_digest)).unwrap();

        registry.admit(&first).unwrap();
        registry.admit(&first).unwrap();
        let mismatch = registry.admit(&second).unwrap_err();

        assert_eq!(*mismatch.expected, first);
        assert_eq!(*mismatch.observed, second);
        assert_eq!(registry.receipts(), vec![first]);
        assert_eq!(
            crate::worker::error::WorkerError::from(mismatch).to_string(),
            format!(
                "worker runtime identity differs from the runtime pinned by this server: \
                 python_executable_sha256 pinned {original_digest}, worker {different_digest}"
            )
        );
    }
}
