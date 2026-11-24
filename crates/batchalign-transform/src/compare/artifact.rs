//! Immutable inputs for cross-implementation comparisons.
//!
//! This module deliberately stops at the artifact contract.  It does not run
//! a producer and it does not decide which side is correct.  That makes the
//! same validated input shape usable whether both sides were produced by
//! software or one of them was produced by a human reviewer.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Version of the on-disk comparison manifest schema.
pub const RUN_MANIFEST_SCHEMA_VERSION: u32 = 1;

/// A machine implementation's identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MachineIdentity {
    /// Stable implementation label, for example `batchalign3`.
    ///
    /// Any label is accepted: the point is that a comparison can say which
    /// software produced each side, not that the set of implementations is
    /// known here.
    implementation: String,
    /// Command family that produced the artifacts.
    command: String,
    /// Reproducible build or source identity.
    build: String,
}

impl MachineIdentity {
    /// Construct a validated machine identity.
    pub fn new(
        implementation: String,
        command: String,
        build: String,
    ) -> Result<Self, ArtifactContractError> {
        require_identifier("implementation", &implementation)?;
        require_identifier("command", &command)?;
        require_identifier("build", &build)?;
        Ok(Self {
            implementation,
            command,
            build,
        })
    }

    /// Stable implementation label.
    #[must_use]
    pub fn implementation(&self) -> &str {
        &self.implementation
    }
    /// Producer command family.
    #[must_use]
    pub fn command(&self) -> &str {
        &self.command
    }
    /// Reproducible build identity.
    #[must_use]
    pub fn build(&self) -> &str {
        &self.build
    }
}

/// A human reviewer's identity.  A human is a producer, not a missing build id.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HumanIdentity {
    /// Version of the review protocol used to produce the artifact.
    protocol: String,
    /// Reviewer or cohort label, without requiring a personal name.
    cohort: String,
}

impl HumanIdentity {
    /// Construct a validated human-producer identity.
    pub fn new(protocol: String, cohort: String) -> Result<Self, ArtifactContractError> {
        require_identifier("protocol", &protocol)?;
        require_identifier("cohort", &cohort)?;
        Ok(Self { protocol, cohort })
    }

    /// Review protocol version.
    #[must_use]
    pub fn protocol(&self) -> &str {
        &self.protocol
    }
    /// Reviewer cohort label.
    #[must_use]
    pub fn cohort(&self) -> &str {
        &self.cohort
    }
}

/// Provenance for one produced artifact set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RunIdentity {
    /// Artifacts produced by an executable implementation.
    Machine(MachineIdentity),
    /// Artifacts produced by a manual review protocol.
    Human(HumanIdentity),
}

/// A content digest for one file relative to a run's artifact root.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactDigest {
    /// POSIX-style path relative to the artifact root.
    relative_path: String,
    /// Lowercase BLAKE3 digest of the exact file bytes.
    blake3: String,
}

impl ArtifactDigest {
    /// Safe root-relative path of the inventoried file.
    #[must_use]
    pub fn relative_path(&self) -> &str {
        &self.relative_path
    }
    /// Typed lowercase BLAKE3 text.
    #[must_use]
    pub fn blake3(&self) -> &str {
        &self.blake3
    }
}

/// Immutable provenance and artifact inventory for one produced run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunManifest {
    /// Manifest schema version.
    schema_version: u32,
    /// Stable identity for this particular run.
    run_id: String,
    /// Who or what produced the artifacts.
    identity: RunIdentity,
    /// Identity of the source media or CHAT input set.
    source_id: String,
    /// Environment-independent normalized producer arguments.
    #[serde(default)]
    arguments: BTreeMap<String, String>,
    /// Hashes of every regular file in the artifact root.
    artifacts: Vec<ArtifactDigest>,
}

/// The pairing relation used by a comparison plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PairingPolicy {
    /// Both runs claim to have processed the same source media.
    SameSourceMedia,
    /// Both runs claim to have processed the same source CHAT.
    SameSourceChat,
    /// The caller supplies explicit pairs (reserved for the next slice).
    ExplicitPairs,
}

/// The comparison domain selected by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonSubject {
    /// Compare transcript words.
    Transcription,
    /// Compare token timings.
    Alignment,
    /// Compare `%mor` and `%gra` structures.
    Morphotag,
}

/// A validated run plus its local artifact root.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunArtifactRoot(PathBuf);

impl RunArtifactRoot {
    /// Construct an artifact-root path wrapper.
    #[must_use]
    pub const fn new(path: PathBuf) -> Self {
        Self(path)
    }
    /// Borrow the root path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl AsRef<Path> for RunArtifactRoot {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

/// A normalized, root-relative artifact path.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RelativeArtifactPath(String);

impl RelativeArtifactPath {
    /// Construct a safe path that cannot escape an artifact root.
    pub fn new(path: impl Into<String>) -> Result<Self, ArtifactContractError> {
        let path = path.into();
        let candidate = Path::new(&path);
        if path.is_empty()
            || candidate.is_absolute()
            || path.contains('\\')
            || candidate.components().any(|component| {
                matches!(
                    component,
                    std::path::Component::ParentDir | std::path::Component::CurDir
                )
            })
        {
            return Err(ArtifactContractError::InvalidArtifactPath(path));
        }
        Ok(Self(path))
    }

    /// Return the normalized relative path text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A caller-declared left/right artifact pairing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactPair {
    /// Artifact path under the left run root.
    left: RelativeArtifactPath,
    /// Artifact path under the right run root.
    right: RelativeArtifactPath,
}

impl ArtifactPair {
    /// Construct an explicit pair of safe relative paths.
    #[must_use]
    pub const fn new(left: RelativeArtifactPath, right: RelativeArtifactPath) -> Self {
        Self { left, right }
    }
    /// Left artifact path.
    #[must_use]
    pub const fn left(&self) -> &RelativeArtifactPath {
        &self.left
    }
    /// Right artifact path.
    #[must_use]
    pub const fn right(&self) -> &RelativeArtifactPath {
        &self.right
    }
}

/// A pairing whose two paths were found in the verified manifests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedArtifactPair {
    /// Verified left artifact path.
    left: RelativeArtifactPath,
    /// Verified right artifact path.
    right: RelativeArtifactPath,
    aggregate: AggregatePolicy,
    speaker_map: Option<BTreeMap<String, String>>,
}

impl ValidatedArtifactPair {
    /// Verified left artifact path.
    #[must_use]
    pub const fn left(&self) -> &RelativeArtifactPath {
        &self.left
    }

    /// Verified right artifact path.
    #[must_use]
    pub const fn right(&self) -> &RelativeArtifactPath {
        &self.right
    }

    /// Whether this pair contributes to included aggregates.
    #[must_use]
    pub const fn aggregate(&self) -> &AggregatePolicy {
        &self.aggregate
    }

    /// Pair-effective explicit speaker assignments, if supplied.
    #[must_use]
    pub const fn speaker_map(&self) -> Option<&BTreeMap<String, String>> {
        self.speaker_map.as_ref()
    }
}

/// Aggregate treatment for an artifact pair.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum AggregatePolicy {
    /// Include the pair in aggregate values.
    Included,
    /// Retain pair evidence but omit it from aggregates.
    HeldOut {
        /// Human-readable reason for excluding the pair from aggregates.
        reason: String,
    },
}

/// A validated run plus its local artifact root.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProducedRun {
    /// Run provenance and content inventory.
    pub manifest: RunManifest,
    /// Local directory containing the inventoried artifacts.
    pub artifacts: RunArtifactRoot,
}

/// A run whose manifest and artifact bytes have been verified.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedProducedRun {
    /// Verified run provenance and content inventory.
    manifest: RunManifest,
    /// Verified local artifact directory.
    artifacts: RunArtifactRoot,
}

impl ValidatedProducedRun {
    /// Verified run manifest.
    #[must_use]
    pub const fn manifest(&self) -> &RunManifest {
        &self.manifest
    }

    /// Verified artifact root.
    #[must_use]
    pub const fn artifacts(&self) -> &RunArtifactRoot {
        &self.artifacts
    }
}

/// The first-version comparison plan: exactly two produced runs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComparisonPlan {
    /// Comparison domain.
    pub subject: ComparisonSubject,
    /// Left and right runs; neither is implicitly gold.
    pub runs: [ProducedRun; 2],
    /// How source identity is established.
    pub pairing: PairingPolicy,
    /// Explicit artifact pairs to compare.
    pub artifact_pairs: Vec<ArtifactPair>,
    /// Directory where a later runner will write reports.
    pub output: PathBuf,
}

/// Shared private state of a validated subject-specific plan.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ValidatedPlanCore {
    runs: [ValidatedProducedRun; 2],
    pairing: PairingPolicy,
    artifact_pairs: Vec<ValidatedArtifactPair>,
    output: PathBuf,
    exclusion_tokens: BTreeSet<String>,
}

/// A validated plan whose artifacts may be compared as transcripts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedTranscriptionPlan(ValidatedPlanCore);

/// A validated plan whose artifacts may be compared for token timing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedAlignmentPlan(ValidatedPlanCore);

/// A validated plan whose artifacts may be compared for `%mor` and `%gra`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedMorphotagPlan(ValidatedPlanCore);

/// Subject-specific result of validating an untrusted comparison plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidatedComparisonPlan {
    /// Transcript-comparison plan.
    Transcription(ValidatedTranscriptionPlan),
    /// Token-timing comparison plan.
    Alignment(ValidatedAlignmentPlan),
    /// Morphosyntax comparison plan.
    Morphotag(ValidatedMorphotagPlan),
}

impl ValidatedComparisonPlan {
    pub(super) fn apply_plan_options(
        mut self,
        pair_options: Vec<(AggregatePolicy, Option<BTreeMap<String, String>>)>,
        exclusion_tokens: BTreeSet<String>,
    ) -> Result<Self, ArtifactContractError> {
        let core = match &mut self {
            Self::Transcription(plan) => &mut plan.0,
            Self::Alignment(plan) => &mut plan.0,
            Self::Morphotag(plan) => &mut plan.0,
        };
        if core.artifact_pairs.len() != pair_options.len() {
            return Err(ArtifactContractError::NotPairable(
                "pair option count does not match artifact pairs".to_string(),
            ));
        }
        for (pair, (aggregate, speaker_map)) in core.artifact_pairs.iter_mut().zip(pair_options) {
            pair.aggregate = aggregate;
            pair.speaker_map = speaker_map;
        }
        core.exclusion_tokens = exclusion_tokens;
        Ok(self)
    }
}

macro_rules! validated_plan_accessors {
    ($plan:ty) => {
        impl $plan {
            /// Verified left and right runs; neither is implicitly gold.
            pub const fn runs(&self) -> &[ValidatedProducedRun; 2] {
                &self.0.runs
            }

            /// Verified source-pairing policy.
            pub const fn pairing(&self) -> PairingPolicy {
                self.0.pairing
            }

            /// Artifact pairs proven to exist in both verified inventories.
            pub fn artifact_pairs(&self) -> &[ValidatedArtifactPair] {
                &self.0.artifact_pairs
            }

            /// Report destination.
            pub fn output(&self) -> &Path {
                &self.0.output
            }

            /// Tokens excluded from transcription agreement scoring.
            pub const fn exclusion_tokens(&self) -> &BTreeSet<String> {
                &self.0.exclusion_tokens
            }
        }
    };
}

validated_plan_accessors!(ValidatedTranscriptionPlan);
validated_plan_accessors!(ValidatedAlignmentPlan);
validated_plan_accessors!(ValidatedMorphotagPlan);

/// Errors raised before any comparison metric is computed.
#[derive(Debug, thiserror::Error)]
pub enum ArtifactContractError {
    /// A required directory or file could not be read.
    #[error("cannot inspect comparison artifact {path}: {source}")]
    Io {
        /// Path that could not be inspected.
        path: PathBuf,
        /// Underlying filesystem error.
        source: std::io::Error,
    },
    /// The manifest is malformed or stale.
    #[error("invalid run manifest: {0}")]
    InvalidManifest(String),
    /// A relative artifact path is unsafe or malformed.
    #[error("invalid relative artifact path: {0:?}")]
    InvalidArtifactPath(String),
    /// The manifest's inventory does not match the artifact root.
    #[error("artifact inventory mismatch for {run_id}: {detail}")]
    InventoryMismatch {
        /// Run whose inventory is stale.
        run_id: String,
        /// Human-readable mismatch detail.
        detail: String,
    },
    /// The two runs cannot be paired under the selected policy.
    #[error("comparison runs are not pairable: {0}")]
    NotPairable(String),
    /// A requested artifact is absent from the corresponding manifest.
    #[error("artifact pair path {path:?} is absent from {side} run {run_id}")]
    ArtifactNotInManifest {
        /// Missing relative path.
        path: String,
        /// Run side containing the missing path.
        side: &'static str,
        /// Run whose manifest was searched.
        run_id: String,
    },
    /// The same artifact was paired more than once.
    #[error("artifact pair reuses a {side} path: {path:?}")]
    DuplicateArtifactPath {
        /// Reused side.
        side: &'static str,
        /// Reused relative path.
        path: String,
    },
}

impl RunManifest {
    /// Manifest schema version.
    #[must_use]
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }
    /// Stable run identifier.
    #[must_use]
    pub fn run_id(&self) -> &str {
        &self.run_id
    }
    /// Producer identity.
    #[must_use]
    pub const fn identity(&self) -> &RunIdentity {
        &self.identity
    }
    /// Immutable source-set identifier.
    #[must_use]
    pub fn source_id(&self) -> &str {
        &self.source_id
    }
    /// Normalized producer arguments.
    #[must_use]
    pub const fn arguments(&self) -> &BTreeMap<String, String> {
        &self.arguments
    }
    /// Non-empty artifact inventory.
    #[must_use]
    pub fn artifacts(&self) -> &[ArtifactDigest] {
        &self.artifacts
    }

    /// Build a manifest by hashing every regular file under `root`.
    pub fn from_artifact_root(
        root: &Path,
        run_id: String,
        identity: RunIdentity,
        source_id: String,
        arguments: BTreeMap<String, String>,
    ) -> Result<Self, ArtifactContractError> {
        let artifacts = inventory(root)?;
        Ok(Self {
            schema_version: RUN_MANIFEST_SCHEMA_VERSION,
            run_id,
            identity,
            source_id,
            arguments,
            artifacts,
        })
    }

    /// Verify that the manifest still describes the exact bytes on disk.
    pub fn verify(&self, root: &Path) -> Result<(), ArtifactContractError> {
        if self.schema_version != RUN_MANIFEST_SCHEMA_VERSION {
            return Err(ArtifactContractError::InvalidManifest(format!(
                "schema version {} is not supported (expected {})",
                self.schema_version, RUN_MANIFEST_SCHEMA_VERSION
            )));
        }
        require_identifier("run_id", &self.run_id)?;
        require_identifier("source_id", &self.source_id)?;
        validate_identity(&self.identity)?;
        let mut paths = BTreeSet::new();
        for artifact in &self.artifacts {
            RelativeArtifactPath::new(artifact.relative_path.clone())?;
            if !paths.insert(artifact.relative_path.as_str()) {
                return Err(ArtifactContractError::InvalidManifest(format!(
                    "duplicate artifact path {:?}",
                    artifact.relative_path
                )));
            }
            if artifact.blake3.len() != 64
                || !artifact
                    .blake3
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
            {
                return Err(ArtifactContractError::InvalidManifest(format!(
                    "invalid BLAKE3 digest for {:?}",
                    artifact.relative_path
                )));
            }
        }
        let actual = inventory(root)?;
        if actual.is_empty() {
            return Err(ArtifactContractError::InvalidManifest(
                "artifact root must contain at least one regular file".to_string(),
            ));
        }
        if actual != self.artifacts {
            return Err(ArtifactContractError::InventoryMismatch {
                run_id: self.run_id.clone(),
                detail: "manifest hashes or file set differ from the artifact root".to_string(),
            });
        }
        Ok(())
    }
}

impl ComparisonPlan {
    /// Validate manifests, artifact bytes, and the declared pairing policy.
    ///
    /// Success consumes the unvalidated plan and returns a plan in the
    /// validated state.  There is no public constructor for that state.
    pub fn validate(self) -> Result<ValidatedComparisonPlan, ArtifactContractError> {
        if matches!(
            self.pairing,
            PairingPolicy::SameSourceMedia | PairingPolicy::SameSourceChat
        ) && self.runs[0].manifest.source_id != self.runs[1].manifest.source_id
        {
            return Err(ArtifactContractError::NotPairable(
                "same-source pairing requires equal source_id values".to_string(),
            ));
        }
        for run in &self.runs {
            run.manifest.verify(run.artifacts.as_ref())?;
        }
        if self.runs[0].manifest.run_id == self.runs[1].manifest.run_id {
            return Err(ArtifactContractError::NotPairable(
                "the two runs must have distinct run_id values".to_string(),
            ));
        }
        if self.artifact_pairs.is_empty() {
            return Err(ArtifactContractError::NotPairable(
                "comparison requires at least one explicit artifact pair".to_string(),
            ));
        }
        let mut left_paths = BTreeSet::new();
        let mut right_paths = BTreeSet::new();
        let left_inventory: BTreeSet<&str> = self.runs[0]
            .manifest
            .artifacts
            .iter()
            .map(|artifact| artifact.relative_path.as_str())
            .collect();
        let right_inventory: BTreeSet<&str> = self.runs[1]
            .manifest
            .artifacts
            .iter()
            .map(|artifact| artifact.relative_path.as_str())
            .collect();
        let mut validated_pairs = Vec::with_capacity(self.artifact_pairs.len());
        for pair in &self.artifact_pairs {
            if !left_paths.insert(pair.left.as_str()) {
                return Err(ArtifactContractError::DuplicateArtifactPath {
                    side: "left",
                    path: pair.left.as_str().to_string(),
                });
            }
            if !right_paths.insert(pair.right.as_str()) {
                return Err(ArtifactContractError::DuplicateArtifactPath {
                    side: "right",
                    path: pair.right.as_str().to_string(),
                });
            }
            if !left_inventory.contains(pair.left.as_str()) {
                return Err(ArtifactContractError::ArtifactNotInManifest {
                    path: pair.left.as_str().to_string(),
                    side: "left",
                    run_id: self.runs[0].manifest.run_id.clone(),
                });
            }
            if !right_inventory.contains(pair.right.as_str()) {
                return Err(ArtifactContractError::ArtifactNotInManifest {
                    path: pair.right.as_str().to_string(),
                    side: "right",
                    run_id: self.runs[1].manifest.run_id.clone(),
                });
            }
            validated_pairs.push(ValidatedArtifactPair {
                left: pair.left.clone(),
                right: pair.right.clone(),
                aggregate: AggregatePolicy::Included,
                speaker_map: None,
            });
        }
        let [left, right] = self.runs;
        let core = ValidatedPlanCore {
            runs: [
                ValidatedProducedRun {
                    manifest: left.manifest,
                    artifacts: left.artifacts,
                },
                ValidatedProducedRun {
                    manifest: right.manifest,
                    artifacts: right.artifacts,
                },
            ],
            pairing: self.pairing,
            artifact_pairs: validated_pairs,
            output: self.output,
            exclusion_tokens: BTreeSet::new(),
        };
        Ok(match self.subject {
            ComparisonSubject::Transcription => {
                ValidatedComparisonPlan::Transcription(ValidatedTranscriptionPlan(core))
            }
            ComparisonSubject::Alignment => {
                ValidatedComparisonPlan::Alignment(ValidatedAlignmentPlan(core))
            }
            ComparisonSubject::Morphotag => {
                ValidatedComparisonPlan::Morphotag(ValidatedMorphotagPlan(core))
            }
        })
    }
}

fn inventory(root: &Path) -> Result<Vec<ArtifactDigest>, ArtifactContractError> {
    let metadata = fs::symlink_metadata(root).map_err(|source| ArtifactContractError::Io {
        path: root.to_path_buf(),
        source,
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(ArtifactContractError::InvalidManifest(format!(
            "artifact root {} must be a real directory",
            root.display()
        )));
    }
    let mut files = Vec::new();
    collect_files(root, root, &mut files)?;
    files.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    Ok(files)
}

fn collect_files(
    root: &Path,
    current: &Path,
    files: &mut Vec<ArtifactDigest>,
) -> Result<(), ArtifactContractError> {
    let entries = fs::read_dir(current).map_err(|source| ArtifactContractError::Io {
        path: current.to_path_buf(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| ArtifactContractError::Io {
            path: current.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        let metadata = entry
            .metadata()
            .map_err(|source| ArtifactContractError::Io {
                path: path.clone(),
                source,
            })?;
        let file_type = entry
            .file_type()
            .map_err(|source| ArtifactContractError::Io {
                path: path.clone(),
                source,
            })?;
        if file_type.is_symlink() {
            return Err(ArtifactContractError::InvalidManifest(format!(
                "symlink is not allowed in artifact root: {}",
                path.display()
            )));
        }
        if metadata.is_dir() {
            collect_files(root, &path, files)?;
        } else if metadata.is_file() {
            let bytes = fs::read(&path).map_err(|source| ArtifactContractError::Io {
                path: path.clone(),
                source,
            })?;
            let relative_path = path
                .strip_prefix(root)
                .map_err(|_| {
                    ArtifactContractError::InvalidManifest("artifact escaped root".to_string())
                })?
                .to_string_lossy()
                .replace(std::path::MAIN_SEPARATOR, "/");
            files.push(ArtifactDigest {
                relative_path,
                blake3: blake3::hash(&bytes).to_hex().to_string(),
            });
        }
    }
    Ok(())
}

fn require_identifier(name: &str, value: &str) -> Result<(), ArtifactContractError> {
    if value.trim().is_empty() || value != value.trim() || value.chars().any(char::is_control) {
        return Err(ArtifactContractError::InvalidManifest(format!(
            "{name} must be a non-empty identifier without surrounding whitespace or control characters"
        )));
    }
    Ok(())
}

fn validate_identity(identity: &RunIdentity) -> Result<(), ArtifactContractError> {
    match identity {
        RunIdentity::Machine(value) => {
            require_identifier("implementation", &value.implementation)?;
            require_identifier("command", &value.command)?;
            require_identifier("build", &value.build)
        }
        RunIdentity::Human(value) => {
            require_identifier("protocol", &value.protocol)?;
            require_identifier("cohort", &value.cohort)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn machine(label: &str) -> RunIdentity {
        RunIdentity::Machine(MachineIdentity {
            implementation: label.to_string(),
            command: "transcribe".to_string(),
            build: "test-build".to_string(),
        })
    }

    #[test]
    fn manifest_hashes_nested_files_and_verifies_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("nested");
        std::fs::create_dir(&nested).unwrap();
        let mut file = std::fs::File::create(nested.join("result.cha")).unwrap();
        file.write_all(b"@Begin\n").unwrap();

        let manifest = RunManifest::from_artifact_root(
            dir.path(),
            "run-a".to_string(),
            machine("ours"),
            "session-1".to_string(),
            BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(manifest.artifacts.len(), 1);
        assert_eq!(manifest.artifacts[0].relative_path, "nested/result.cha");
        manifest.verify(dir.path()).unwrap();

        std::fs::write(nested.join("result.cha"), b"changed\n").unwrap();
        assert!(matches!(
            manifest.verify(dir.path()),
            Err(ArtifactContractError::InventoryMismatch { .. })
        ));
    }

    #[test]
    fn plan_requires_same_source_for_implicit_pairing() {
        let left_dir = tempfile::tempdir().unwrap();
        let right_dir = tempfile::tempdir().unwrap();
        let left = RunManifest::from_artifact_root(
            left_dir.path(),
            "left".to_string(),
            machine("ours"),
            "source-a".to_string(),
            BTreeMap::new(),
        )
        .unwrap();
        let right = RunManifest::from_artifact_root(
            right_dir.path(),
            "right".to_string(),
            machine("other-impl"),
            "source-b".to_string(),
            BTreeMap::new(),
        )
        .unwrap();
        let plan = ComparisonPlan {
            subject: ComparisonSubject::Transcription,
            runs: [
                ProducedRun {
                    manifest: left,
                    artifacts: RunArtifactRoot(left_dir.path().to_path_buf()),
                },
                ProducedRun {
                    manifest: right,
                    artifacts: RunArtifactRoot(right_dir.path().to_path_buf()),
                },
            ],
            pairing: PairingPolicy::SameSourceChat,
            artifact_pairs: Vec::new(),
            output: tempfile::tempdir().unwrap().path().to_path_buf(),
        };
        assert!(matches!(
            plan.validate(),
            Err(ArtifactContractError::NotPairable(_))
        ));
    }

    #[test]
    fn validation_returns_a_distinct_verified_plan_state() {
        let left_dir = tempfile::tempdir().unwrap();
        let right_dir = tempfile::tempdir().unwrap();
        std::fs::write(left_dir.path().join("result.cha"), b"left").unwrap();
        std::fs::write(right_dir.path().join("result.cha"), b"right").unwrap();
        let left = RunManifest::from_artifact_root(
            left_dir.path(),
            "left".to_string(),
            machine("ours"),
            "source-a".to_string(),
            BTreeMap::new(),
        )
        .unwrap();
        let right = RunManifest::from_artifact_root(
            right_dir.path(),
            "right".to_string(),
            machine("other-impl"),
            "source-a".to_string(),
            BTreeMap::new(),
        )
        .unwrap();
        let plan = ComparisonPlan {
            subject: ComparisonSubject::Transcription,
            runs: [
                ProducedRun {
                    manifest: left,
                    artifacts: RunArtifactRoot(left_dir.path().to_path_buf()),
                },
                ProducedRun {
                    manifest: right,
                    artifacts: RunArtifactRoot(right_dir.path().to_path_buf()),
                },
            ],
            pairing: PairingPolicy::SameSourceChat,
            artifact_pairs: vec![ArtifactPair {
                left: RelativeArtifactPath::new("result.cha").unwrap(),
                right: RelativeArtifactPath::new("result.cha").unwrap(),
            }],
            output: tempfile::tempdir().unwrap().path().to_path_buf(),
        };
        let validated = plan.validate().unwrap();
        let ValidatedComparisonPlan::Transcription(validated) = validated else {
            panic!("transcription input must produce transcription typestate");
        };
        assert_eq!(validated.runs()[0].manifest().run_id, "left");
        assert_eq!(validated.runs()[1].manifest().run_id, "right");
    }
}
