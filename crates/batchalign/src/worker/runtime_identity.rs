//! Content-addressed identity reported by a Python worker at startup.
//!
//! The wire type deliberately contains no filesystem paths. A successfully
//! parsed value proves the schema revision and all four full SHA-256 shapes;
//! callers cannot construct the identity from arbitrary strings.

use serde::{Deserialize, Serialize};
use std::sync::Mutex;

/// A validated lowercase SHA-256 digest.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(transparent)]
pub struct Sha256Digest(String);

impl Sha256Digest {
    /// Canonical lowercase hexadecimal representation.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for Sha256Digest {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            Ok(Self(value))
        } else {
            Err("runtime digest must be 64 lowercase hexadecimal characters".into())
        }
    }
}

impl<'de> Deserialize<'de> for Sha256Digest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        String::deserialize(deserializer)?
            .try_into()
            .map_err(serde::de::Error::custom)
    }
}

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
    pub fn python_version(&self) -> &str {
        &self.python_version
    }

    /// Content digest of the resolved Python executable.
    pub fn python_executable_sha256(&self) -> &Sha256Digest {
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
/// whose distribution inventory changed because loading FunASR extended
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

    fn identity_json(schema: u8, executable_digest: &str) -> String {
        format!(
            r#"{{"schema_version":{schema},"python_version":"3.13.12","python_executable_sha256":"{executable_digest}","batchalign_package_tree_sha256":"{}","batchalign_core_extension_sha256":"{}","distribution_inventory_sha256":"{}"}}"#,
            "b".repeat(64),
            "c".repeat(64),
            "d".repeat(64),
        )
    }

    #[test]
    fn invalid_schema_or_digest_cannot_construct_runtime_identity() {
        for json in [
            identity_json(2, &"a".repeat(64)),
            identity_json(1, "not-a-digest"),
        ] {
            assert!(serde_json::from_str::<WorkerRuntimeIdentity>(&json).is_err());
        }
    }

    #[test]
    fn schema_one_refuses_unknown_fields_instead_of_silently_reinterpreting_them() {
        let json = identity_json(1, &"a".repeat(64)).replace("}", ",\"path\":\"/tmp/python\"}");

        assert!(serde_json::from_str::<WorkerRuntimeIdentity>(&json).is_err());
    }

    #[test]
    fn schema_one_requires_the_loaded_native_extension_digest() {
        let mut value: serde_json::Value =
            serde_json::from_str(&identity_json(1, &"a".repeat(64))).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("batchalign_core_extension_sha256");

        assert!(serde_json::from_value::<WorkerRuntimeIdentity>(value).is_err());
    }

    #[test]
    fn serialized_runtime_identity_contains_evidence_but_no_paths() {
        let identity: WorkerRuntimeIdentity =
            serde_json::from_str(&identity_json(1, &"a".repeat(64))).unwrap();

        let serialized = serde_json::to_string(&identity).unwrap();

        assert!(serialized.contains("python_executable_sha256"));
        assert!(serialized.contains("batchalign_core_extension_sha256"));
        assert!(!serialized.contains("path"));
    }

    #[test]
    fn observed_runtime_registry_refuses_a_second_runtime_identity() {
        let registry = ObservedWorkerRuntimes::default();
        let first: WorkerRuntimeIdentity =
            serde_json::from_str(&identity_json(1, &"a".repeat(64))).unwrap();
        let second: WorkerRuntimeIdentity =
            serde_json::from_str(&identity_json(1, &"d".repeat(64))).unwrap();

        registry.admit(&first).unwrap();
        registry.admit(&first).unwrap();
        let mismatch = registry.admit(&second).unwrap_err();

        assert_eq!(*mismatch.expected, first);
        assert_eq!(*mismatch.observed, second);
        assert_eq!(registry.receipts(), vec![first]);
    }

    /// The refusal names exactly the component that differs, with both
    /// values, and nothing that matched: what an operator needs to find the
    /// cause without reproducing it.
    #[test]
    fn a_refusal_names_the_component_that_differs() {
        let registry = ObservedWorkerRuntimes::default();
        let first: WorkerRuntimeIdentity =
            serde_json::from_str(&identity_json(1, &"a".repeat(64))).unwrap();
        let second: WorkerRuntimeIdentity =
            serde_json::from_str(&identity_json(1, &"d".repeat(64))).unwrap();
        registry.admit(&first).unwrap();

        let message = crate::worker::error::WorkerError::from(registry.admit(&second).unwrap_err())
            .to_string();

        assert_eq!(
            message,
            format!(
                "worker runtime identity differs from the runtime pinned by this server: \
                 python_executable_sha256 pinned {}, worker {}",
                "a".repeat(64),
                "d".repeat(64)
            )
        );
    }
}
