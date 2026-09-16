//! Durable utterance-segmentation evidence for controlled experiments.

use std::collections::BTreeSet;
use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::api::{InvalidStampSafeText, StampJoiner, StampSafeText};
use crate::provenance::EngineNames;
use crate::types::worker_v2::{HubCommitV2, UtsegBoundaryModelEvidenceV2};
use crate::utseg::{
    AdmittedUtsegPrediction, LocalUtsegDecisionReceipt, UtsegPredictionOrigin, admit_prediction,
};
use batchalign_transform::utseg::UtsegBatchItem;

/// The evidence schema this build writes, and the only one it reads back.
///
/// One owner for the writer and the reader: an artifact whose version is not
/// this one is refused by name rather than read as though its shape were
/// current.
///
/// 4 since a boundary model's revision became a required part of its identity.
/// A schema-3 artifact is refused rather than read, and that is a semantic
/// decision rather than a convenience: its revision, where it has one at all,
/// records whatever a FLOATING load happened to resolve to on the day it ran,
/// scraped from `config._commit_hash`. Reading it back into a type whose
/// meaning is "the revision the plan pinned and the worker verified" would
/// silently reinterpret an accident as a pin. The refusal names the artifact
/// and tells an operator to regenerate it, which is cheap: these sidecars are
/// `--debug-dir` research artifacts, not a result cache, and no placeholder is
/// invented for the ones that never recorded a revision.
const SCHEMA_VERSION: u8 = 4;

/// Location in transcribe at which utterance segmentation ran.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum UtsegEvidencePhase {
    /// Segmentation over timed ASR chunks before CHAT construction.
    PreChat,
    /// Segmentation over main-tier words after CHAT construction.
    PostChat,
}

impl UtsegEvidencePhase {
    fn filename_component(self) -> &'static str {
        match self {
            Self::PreChat => "pre_chat",
            Self::PostChat => "post_chat",
        }
    }
}

impl fmt::Display for UtsegEvidencePhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.filename_component())
    }
}

/// Complete, versioned evidence from one utterance-segmentation batch.
///
/// Both directions of the artifact: written by [`Self::from_predictions`] and
/// read back by [`AdmittedUtsegEvidence::admit`], so the shape on disk has one
/// definition and the replay cannot drift from the writer.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct UtsegEvidenceTrace {
    /// 4 since a boundary model's revision became required; 3 removed the
    /// top-level `engine_version`, which inside transcribe carried the ASR
    /// engine's version and never the segmenter's. Each item's prediction
    /// already names its own source and model.
    schema_version: u8,
    phase: UtsegEvidencePhase,
    language: String,
    items: Vec<UtsegEvidenceItem>,
}

/// One request and the admitted prediction that is safe to apply to it.
#[derive(Debug, Serialize, Deserialize)]
struct UtsegEvidenceItem {
    item_ordinal: usize,
    words: Vec<String>,
    text: String,
    prediction: UtsegEvidencePrediction,
}

/// Closed set of inference sources that can produce utseg assignments.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
enum UtsegEvidencePrediction {
    /// TalkBank boundary model with raw and applied per-word evidence.
    BoundaryModel {
        assignments: Vec<usize>,
        evidence: UtsegBoundaryModelEvidenceV2,
        #[serde(skip_serializing_if = "Option::is_none")]
        local_decision: Option<LocalUtsegDecisionReceipt>,
    },
    /// Compatibility path whose worker did not expose evidence.
    UnobservedAssignments { assignments: Vec<usize> },
    /// Stanza constituency-tree projection.
    Constituency { assignments: Vec<usize> },
}

/// A partial batch cannot be represented as a complete experiment trace.
#[derive(Debug, thiserror::Error)]
#[error(
    "cannot construct utseg evidence trace from {request_count} requests and {prediction_count} predictions"
)]
pub(crate) struct UtsegEvidenceShapeError {
    request_count: usize,
    prediction_count: usize,
}

impl UtsegEvidenceTrace {
    /// Construct a complete trace only when every request has one admitted
    /// prediction. Admission has already established all per-word invariants.
    pub(crate) fn from_predictions(
        phase: UtsegEvidencePhase,
        language: &str,
        requests: &[(usize, UtsegBatchItem)],
        predictions: &[AdmittedUtsegPrediction],
    ) -> Result<Self, UtsegEvidenceShapeError> {
        if requests.len() != predictions.len() {
            return Err(UtsegEvidenceShapeError {
                request_count: requests.len(),
                prediction_count: predictions.len(),
            });
        }

        let items = requests
            .iter()
            .zip(predictions.iter())
            .map(|((item_ordinal, request), prediction)| UtsegEvidenceItem {
                item_ordinal: *item_ordinal,
                words: request.words.clone(),
                text: request.text.clone(),
                prediction: match prediction {
                    AdmittedUtsegPrediction::BoundaryModelWorkerDeclared { response, evidence } => {
                        UtsegEvidencePrediction::BoundaryModel {
                            assignments: response.assignments.clone(),
                            evidence: evidence.clone(),
                            local_decision: None,
                        }
                    }
                    AdmittedUtsegPrediction::BoundaryModelLocallyReapplied {
                        response,
                        evidence,
                        receipt,
                    } => UtsegEvidencePrediction::BoundaryModel {
                        assignments: response.assignments.clone(),
                        evidence: evidence.clone(),
                        local_decision: Some(receipt.clone()),
                    },
                    AdmittedUtsegPrediction::UnobservedAssignments { response } => {
                        UtsegEvidencePrediction::UnobservedAssignments {
                            assignments: response.assignments.clone(),
                        }
                    }
                    AdmittedUtsegPrediction::Constituency { response } => {
                        UtsegEvidencePrediction::Constituency {
                            assignments: response.assignments.clone(),
                        }
                    }
                },
            })
            .collect();

        Ok(Self {
            schema_version: SCHEMA_VERSION,
            phase,
            language: language.to_owned(),
            items,
        })
    }
}

/// One retained evidence artifact, admitted.
///
/// Existence proves three things about the bytes it was read from: they are
/// this build's schema, they record the phase the replay asked for, and every
/// item's prediction passed the same admission a live worker result passes
/// ([`admit_prediction`]). A replay can therefore reapply what it holds without
/// re-checking anything, and cannot reapply an artifact that was never checked,
/// because there is no other way to obtain this type.
pub(crate) struct AdmittedUtsegEvidence {
    language: String,
    items: Vec<AdmittedUtsegEvidenceItem>,
}

/// One admitted item: the request that was dispatched and the prediction the
/// run applied to it.
pub(crate) struct AdmittedUtsegEvidenceItem {
    /// The transcript position the producer recorded this request under.
    pub(crate) item_ordinal: usize,
    /// The words and text exactly as they were dispatched.
    pub(crate) request: UtsegBatchItem,
    /// The prediction that was applied, re-admitted against that request.
    pub(crate) prediction: AdmittedUtsegPrediction,
}

/// Why a retained evidence artifact cannot be admitted.
///
/// Every variant names what failed, so a refusal tells an operator which
/// artifact to regenerate rather than only that the replay stopped.
#[derive(Debug, thiserror::Error)]
pub(crate) enum UtsegEvidenceAdmissionError {
    /// The bytes are not the JSON this artifact is written as.
    #[error("retained utseg evidence is not readable as this artifact: {0}")]
    Malformed(#[from] serde_json::Error),
    /// The artifact was written by a build whose schema this one cannot read.
    #[error(
        "retained utseg evidence is schema {found}, and this build reads schema {expected}; \
         regenerate the evidence with this build"
    )]
    UnsupportedSchema {
        /// The version the artifact declares.
        found: u8,
        /// The version this build reads.
        expected: u8,
    },
    /// The artifact records the other segmentation pass.
    #[error(
        "retained utseg evidence records the {found} pass, and this replay reproduces the \
         {expected} pass"
    )]
    WrongPhase {
        /// The pass the artifact records.
        found: UtsegEvidencePhase,
        /// The pass this replay needs.
        expected: UtsegEvidencePhase,
    },
    /// One item's prediction does not fit the request retained with it.
    #[error(
        "retained utseg evidence item {index} (transcript position {item_ordinal}) is not \
         applicable to the request retained with it: {reason}"
    )]
    Item {
        /// Position in the artifact's item list.
        index: usize,
        /// The transcript position that item records.
        item_ordinal: usize,
        /// What admission refused.
        reason: String,
    },
}

impl AdmittedUtsegEvidence {
    /// Admit one retained artifact for the pass a replay reproduces.
    ///
    /// `expected_phase` is a value the caller's mode determines, not a flag a
    /// user sets, so a post-CHAT replay cannot be handed a pre-CHAT artifact.
    pub(crate) fn admit(
        bytes: &[u8],
        expected_phase: UtsegEvidencePhase,
    ) -> Result<Self, UtsegEvidenceAdmissionError> {
        let trace: UtsegEvidenceTrace = serde_json::from_slice(bytes)?;
        if trace.schema_version != SCHEMA_VERSION {
            return Err(UtsegEvidenceAdmissionError::UnsupportedSchema {
                found: trace.schema_version,
                expected: SCHEMA_VERSION,
            });
        }
        if trace.phase != expected_phase {
            return Err(UtsegEvidenceAdmissionError::WrongPhase {
                found: trace.phase,
                expected: expected_phase,
            });
        }

        let mut items = Vec::with_capacity(trace.items.len());
        for (index, item) in trace.items.into_iter().enumerate() {
            let item_ordinal = item.item_ordinal;
            let request = UtsegBatchItem {
                words: item.words,
                text: item.text,
            };
            let refuse = |reason: String| UtsegEvidenceAdmissionError::Item {
                index,
                item_ordinal,
                reason,
            };
            let prediction = match item.prediction {
                UtsegEvidencePrediction::BoundaryModel {
                    assignments,
                    evidence,
                    local_decision,
                } => {
                    let admitted = admit_prediction(
                        &request,
                        assignments,
                        UtsegPredictionOrigin::BoundaryModel(&evidence),
                    )
                    .map_err(refuse)?;
                    match local_decision {
                        Some(receipt) => {
                            admitted.with_local_decision(receipt).map_err(refuse)?
                        }
                        None => admitted,
                    }
                }
                UtsegEvidencePrediction::UnobservedAssignments { assignments } => admit_prediction(
                    &request,
                    assignments,
                    UtsegPredictionOrigin::UnnamedWorker,
                )
                .map_err(refuse)?,
                UtsegEvidencePrediction::Constituency { assignments } => {
                    admit_prediction(&request, assignments, UtsegPredictionOrigin::Constituency)
                        .map_err(refuse)?
                }
            };
            items.push(AdmittedUtsegEvidenceItem {
                item_ordinal,
                request,
                prediction,
            });
        }

        Ok(Self {
            language: trace.language,
            items,
        })
    }

    /// The language the run recorded, and the admitted items, together: the
    /// two halves of the artifact a replay needs, handed over at once so
    /// neither is read from a different artifact than the other.
    pub(crate) fn into_parts(self) -> (String, Vec<AdmittedUtsegEvidenceItem>) {
        (self.language, self.items)
    }
}

/// The inference source behind one admitted utterance-boundary prediction,
/// borrowed from the prediction and rendered once into provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum UtsegEngineIdentity<'a> {
    /// A TalkBank boundary model, by model id and the exact revision it was
    /// loaded at. Both halves always exist: the model is loaded from a pinned
    /// snapshot, so there is no boundary model without a revision to name.
    BoundaryModel {
        model_id: &'a str,
        model_revision: &'a HubCommitV2,
    },
    /// Stanza constituency-tree projection.
    StanzaConstituency,
    /// A worker that returned assignments without exposing their source.
    UnobservedWorker,
}

impl<'a> UtsegEngineIdentity<'a> {
    /// Name the source of one admitted prediction.
    fn of(prediction: &'a AdmittedUtsegPrediction) -> Self {
        match prediction {
            AdmittedUtsegPrediction::BoundaryModelWorkerDeclared { evidence, .. }
            | AdmittedUtsegPrediction::BoundaryModelLocallyReapplied { evidence, .. } => {
                Self::BoundaryModel {
                    model_id: &evidence.model_id,
                    model_revision: &evidence.model_revision,
                }
            }
            AdmittedUtsegPrediction::UnobservedAssignments { .. } => Self::UnobservedWorker,
            AdmittedUtsegPrediction::Constituency { .. } => Self::StanzaConstituency,
        }
    }
}

impl UtsegEngineIdentity<'_> {
    /// The name provenance records for this source, or `None` when this source
    /// has no name to record.
    ///
    /// A boundary model is always `<model id>@<revision>`: a stamp states what
    /// the worker said and nothing else, so there is no `unrecorded-revision`
    /// placeholder and no `unobserved-worker` stand-in. The id-only form is
    /// gone rather than merely unused: the revision is a required part of the
    /// model's identity, so a boundary model with no revision to name has no
    /// representation here and no branch to render it. A worker that returned
    /// assignments without naming their source contributes no name at all, and
    /// a file whose sources are all unnamed gets no stamp.
    ///
    /// Fallible only for a boundary model: its id comes from the worker's
    /// prediction evidence as plain text, so it is admitted here as stamp-safe
    /// text before it is joined. The revision needs no such admission, because
    /// a [`HubCommitV2`] is 40 hexadecimal characters and therefore already
    /// stamp-safe.
    fn stamp_name(&self) -> Result<Option<StampSafeText>, InvalidStampSafeText> {
        match self {
            Self::BoundaryModel {
                model_id,
                model_revision,
            } => {
                let model_id = StampSafeText::try_from(*model_id)?;
                Ok(Some(StampSafeText::join(
                    &model_id,
                    [&StampSafeText::try_from(model_revision.as_str())?],
                    StampJoiner::At,
                )))
            }
            Self::StanzaConstituency => Ok(Some(
                const { StampSafeText::from_static("stanza-constituency") },
            )),
            Self::UnobservedWorker => Ok(None),
        }
    }
}

/// Every distinct inference source behind one run's applied utterance
/// boundaries, never empty.
///
/// Built only from admitted predictions, so what provenance names is what
/// actually segmented the file rather than the pipeline's engine version
/// (which inside transcribe belongs to the ASR engine).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UtsegEngineIdentities<'a>(BTreeSet<UtsegEngineIdentity<'a>>);

impl<'a> UtsegEngineIdentities<'a> {
    /// `None` when there were no predictions, because nothing was segmented.
    pub(crate) fn from_predictions(predictions: &'a [AdmittedUtsegPrediction]) -> Option<Self> {
        let identities: BTreeSet<_> = predictions.iter().map(UtsegEngineIdentity::of).collect();
        (!identities.is_empty()).then_some(Self(identities))
    }
}

impl UtsegEngineIdentities<'_> {
    /// The `engine=` names, one per distinct source that has one, or the first
    /// source whose name is not stamp-safe text.
    ///
    /// Empty when no source named itself, which is not a stamp: see
    /// [`UtsegEngineIdentity::stamp_name`].
    pub(crate) fn engine_names(&self) -> Result<EngineNames, InvalidStampSafeText> {
        self.0
            .iter()
            .filter_map(|identity| identity.stamp_name().transpose())
            .collect()
    }
}

/// Typestate-like evidence destination: absence and enabled persistence are
/// explicit variants instead of an optional path threaded through writes.
pub(crate) enum UtsegEvidenceSink {
    /// The run did not request durable debug evidence.
    Disabled,
    /// Every requested trace must be durably written or fail the run.
    Enabled(EnabledUtsegEvidenceSink),
}

/// Validated destination for an enabled evidence run.
pub(crate) struct EnabledUtsegEvidenceSink {
    dir: PathBuf,
}

/// Observable outcome of a write request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum UtsegEvidenceWriteOutcome {
    /// Evidence collection was not requested.
    Disabled,
    /// Complete evidence was durably written to this path.
    Written(PathBuf),
}

/// Failures that prevent an enabled evidence request from being durable.
#[derive(Debug, thiserror::Error)]
pub(crate) enum UtsegEvidenceWriteError {
    /// Destination directory could not be created.
    #[error("failed to create utseg evidence directory {}: {source}", path.display())]
    CreateDirectory {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// Complete trace could not be serialized before publication.
    #[error("failed to serialize utseg evidence: {0}")]
    Serialize(#[from] serde_json::Error),
    /// Atomic publication failed.
    #[error("failed to write utseg evidence {}: {source}", path.display())]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

impl UtsegEvidenceSink {
    /// Resolve optional CLI configuration into one explicit sink state.
    pub(crate) fn new(dir: Option<&Path>) -> Self {
        match dir {
            Some(dir) => Self::Enabled(EnabledUtsegEvidenceSink {
                dir: dir.to_owned(),
            }),
            None => Self::Disabled,
        }
    }

    /// Durably publish a trace when evidence collection is enabled.
    pub(crate) fn write(
        &self,
        filename: &str,
        trace: &UtsegEvidenceTrace,
    ) -> Result<UtsegEvidenceWriteOutcome, UtsegEvidenceWriteError> {
        match self {
            Self::Disabled => Ok(UtsegEvidenceWriteOutcome::Disabled),
            Self::Enabled(enabled) => enabled.write(filename, trace),
        }
    }
}

impl EnabledUtsegEvidenceSink {
    fn write(
        &self,
        filename: &str,
        trace: &UtsegEvidenceTrace,
    ) -> Result<UtsegEvidenceWriteOutcome, UtsegEvidenceWriteError> {
        std::fs::create_dir_all(&self.dir).map_err(|source| {
            UtsegEvidenceWriteError::CreateDirectory {
                path: self.dir.clone(),
                source,
            }
        })?;
        let path = self.dir.join(format!(
            "{}_{}_utseg_evidence.json",
            evidence_stem(filename),
            trace.phase.filename_component()
        ));
        let bytes = serde_json::to_vec_pretty(trace)?;
        let mut temp = tempfile::NamedTempFile::new_in(&self.dir).map_err(|source| {
            UtsegEvidenceWriteError::Write {
                path: path.clone(),
                source,
            }
        })?;
        temp.write_all(&bytes)
            .and_then(|()| temp.as_file().sync_all())
            .map_err(|source| UtsegEvidenceWriteError::Write {
                path: path.clone(),
                source,
            })?;
        let persisted = temp
            .persist(&path)
            .map_err(|error| UtsegEvidenceWriteError::Write {
                path: path.clone(),
                source: error.error,
            })?;
        persisted
            .sync_all()
            .map_err(|source| UtsegEvidenceWriteError::Write {
                path: path.clone(),
                source,
            })?;
        #[cfg(unix)]
        std::fs::File::open(&self.dir)
            .and_then(|dir| dir.sync_all())
            .map_err(|source| UtsegEvidenceWriteError::Write {
                path: path.clone(),
                source,
            })?;
        Ok(UtsegEvidenceWriteOutcome::Written(path))
    }
}

fn evidence_stem(filename: &str) -> String {
    let path = Path::new(filename);
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("unknown");
    if path.components().count() <= 1 {
        return stem.to_owned();
    }
    let digest = blake3::hash(filename.as_bytes()).to_hex();
    format!("{stem}-{}", &digest[..12])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::worker_v2::{
        UtsegAdjacencyPolicyRevisionV2, UtsegBoundaryModelEvidenceV2, UtsegNormalizationRevisionV2,
        UtsegWordBoundaryEvidenceV2,
    };
    use crate::utseg::AdmittedUtsegPrediction;
    use batchalign_transform::utseg::{UtsegBatchItem, UtsegResponse};

    /// A commit-shaped revision for fixtures. The evidence type admits nothing
    /// else, so a fixture can no longer carry a placeholder like `revision-1`.
    const TEST_COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

    fn test_commit() -> HubCommitV2 {
        HubCommitV2::try_from(TEST_COMMIT).expect("valid fixture commit")
    }

    fn request() -> UtsegBatchItem {
        UtsegBatchItem {
            words: vec!["hello".to_owned(), "there".to_owned()],
            text: "hello there".to_owned(),
        }
    }

    fn prediction() -> AdmittedUtsegPrediction {
        AdmittedUtsegPrediction::BoundaryModelWorkerDeclared {
            response: UtsegResponse {
                assignments: vec![0, 1],
            },
            evidence: UtsegBoundaryModelEvidenceV2 {
                model_id: "talkbank/utterance-boundary".to_owned(),
                model_revision: test_commit(),
                normalization_revision: UtsegNormalizationRevisionV2::LowerStripAsciiPunctuationV1,
                adjacency_policy_revision:
                    UtsegAdjacencyPolicyRevisionV2::SuppressEarlierAdjacentNonordinaryV1,
                word_evidence: vec![
                    UtsegWordBoundaryEvidenceV2::NormalizationOmission,
                    UtsegWordBoundaryEvidenceV2::ModelShortCircuit,
                ],
            },
        }
    }

    #[test]
    fn trace_keeps_request_assignments_source_and_model_provenance_together() {
        let trace = UtsegEvidenceTrace::from_predictions(
            UtsegEvidencePhase::PreChat,
            "eng",
            &[(0, request())],
            &[prediction()],
        )
        .expect("parallel admitted predictions should form a trace");

        let value = serde_json::to_value(trace).expect("serialize evidence trace");
        assert_eq!(value["schema_version"], 4);
        assert_eq!(value["phase"], "pre_chat");
        assert_eq!(value["language"], "eng");
        assert!(value.get("engine_version").is_none());
        assert_eq!(value["items"][0]["words"][1], "there");
        assert_eq!(value["items"][0]["text"], "hello there");
        assert_eq!(value["items"][0]["prediction"]["source"], "boundary_model");
        assert_eq!(value["items"][0]["prediction"]["assignments"][1], 1);
        assert_eq!(
            value["items"][0]["prediction"]["evidence"]["model_revision"],
            TEST_COMMIT
        );
    }

    #[test]
    fn trace_refuses_to_zip_different_request_and_prediction_counts() {
        let error = UtsegEvidenceTrace::from_predictions(
            UtsegEvidencePhase::PostChat,
            "eng",
            &[(0, request())],
            &[],
        )
        .expect_err("a partial trace must not be constructible");

        assert!(error.to_string().contains("1 requests"));
        assert!(error.to_string().contains("0 predictions"));
    }

    #[test]
    fn enabled_sink_writes_a_versioned_artifact_atomically() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sink = UtsegEvidenceSink::new(Some(dir.path()));
        let trace = UtsegEvidenceTrace::from_predictions(
            UtsegEvidencePhase::PreChat,
            "eng",
            &[(0, request())],
            &[prediction()],
        )
        .expect("trace");
        let expected = serde_json::to_value(&trace).expect("serialize expected trace");

        let outcome = sink
            .write("sample.wav", &trace)
            .expect("enabled evidence request should be durable");
        let UtsegEvidenceWriteOutcome::Written(path) = outcome else {
            panic!("enabled sink should write");
        };
        assert_eq!(path, dir.path().join("sample_pre_chat_utseg_evidence.json"));
        let value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).expect("read persisted trace"))
                .expect("parse persisted trace");
        assert_eq!(value, expected);
    }

    /// The `engine=` value the sources behind `predictions` join into.
    fn joined(predictions: &[AdmittedUtsegPrediction]) -> Option<String> {
        UtsegEngineIdentities::from_predictions(predictions)
            .expect("predictions were applied")
            .engine_names()
            .expect("stamp-safe sources")
            .joined()
            .map(|joined| joined.as_str().to_owned())
    }

    #[test]
    fn engine_identities_name_the_boundary_model_and_its_revision() {
        assert_eq!(
            joined(&[prediction(), prediction()]),
            Some(format!("talkbank/utterance-boundary@{TEST_COMMIT}"))
        );
    }

    /// RED FIRST (review item 2): no placeholder names. A worker that returned
    /// assignments without naming their source contributes nothing, so a file
    /// segmented only by such a worker has no `engine=` to write and gets no
    /// stamp. That used to be written as an invented `unobserved-worker`.
    ///
    /// The companion case, a boundary model recorded by its id alone, is no
    /// longer testable and that is the point: the revision is part of the
    /// model's identity, so `<id>` with no revision has no representation to
    /// construct. The compiler refuses the fixture this test used to build.
    #[test]
    fn a_source_that_names_nothing_contributes_no_engine_name() {
        let unobserved = AdmittedUtsegPrediction::UnobservedAssignments {
            response: UtsegResponse {
                assignments: vec![0, 0],
            },
        };
        assert_eq!(joined(&[unobserved]), None);
    }

    #[test]
    fn engine_identities_list_each_distinct_source_and_are_absent_for_no_predictions() {
        let constituency = AdmittedUtsegPrediction::Constituency {
            response: UtsegResponse {
                assignments: vec![0, 0],
            },
        };
        assert_eq!(
            joined(&[prediction(), constituency]),
            Some(format!(
                "stanza-constituency+talkbank/utterance-boundary@{TEST_COMMIT}"
            ))
        );
        assert_eq!(UtsegEngineIdentities::from_predictions(&[]), None);
    }
}
