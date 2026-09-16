//! Typed utterance-boundary evidence carried by worker protocol V2.
//!
//! This module owns the closed action vocabulary, fixed-point score, and
//! per-word classification states. Keeping them separate from the response
//! envelope makes the evidence model discoverable and keeps `responses.rs`
//! below the workspace's hard file-size limit.

use serde::{Deserialize, Serialize};

use super::asr_model::HubCommitV2;

/// Closed label vocabulary emitted by the TalkBank utterance-boundary model.
///
/// Names describe model semantics instead of exposing classifier label indices,
/// whose numeric ordering is an implementation detail of the Python model.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum UtsegBoundaryActionV2 {
    /// Ordinary lexical word with no punctuation or onset label.
    Ordinary,
    /// Capitalized word classified as an utterance onset.
    CapitalizedOnset,
    /// Period-like utterance boundary.
    PeriodBoundary,
    /// Question-like utterance boundary.
    QuestionBoundary,
    /// Exclamation-like utterance boundary.
    ExclamationBoundary,
    /// Comma-like non-boundary punctuation.
    Comma,
}

impl UtsegBoundaryActionV2 {
    fn is_boundary(self) -> bool {
        matches!(
            self,
            Self::PeriodBoundary | Self::QuestionBoundary | Self::ExclamationBoundary
        )
    }

    fn is_nonordinary(self) -> bool {
        !matches!(self, Self::Ordinary)
    }
}

/// Closed lexical-normalization semantics applied before model inference.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum UtsegNormalizationRevisionV2 {
    /// Lowercase, trim, and remove ASCII period/question/exclamation/comma.
    LowerStripAsciiPunctuationV1,
}

/// Closed raw-to-applied boundary-action policy revisions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum UtsegAdjacencyPolicyRevisionV2 {
    /// Preserve current behavior: suppress the earlier adjacent non-ordinary action.
    SuppressEarlierAdjacentNonordinaryV1,
    /// Experimental replay: suppress only the earlier of adjacent true boundaries.
    SuppressEarlierAdjacentBoundariesV1,
}

impl UtsegAdjacencyPolicyRevisionV2 {
    fn suppresses(self, left: UtsegBoundaryActionV2, right: UtsegBoundaryActionV2) -> bool {
        match self {
            Self::SuppressEarlierAdjacentNonordinaryV1 => {
                left.is_nonordinary() && right.is_nonordinary()
            }
            Self::SuppressEarlierAdjacentBoundariesV1 => left.is_boundary() && right.is_boundary(),
        }
    }
}

/// Fixed-point probability in the inclusive unit interval at micro precision.
///
/// The private field and fallible deserializer prevent invalid probabilities
/// from crossing the worker boundary as ordinary integers.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(try_from = "u32", into = "u32")]
pub struct BoundaryProbabilityMicrosV2(#[schemars(range(min = 0, max = 1_000_000))] u32);

impl TryFrom<u32> for BoundaryProbabilityMicrosV2 {
    type Error = &'static str;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        if value <= 1_000_000 {
            Ok(Self(value))
        } else {
            Err("boundary probability micros must be at most 1000000")
        }
    }
}

impl From<BoundaryProbabilityMicrosV2> for u32 {
    fn from(value: BoundaryProbabilityMicrosV2) -> Self {
        value.0
    }
}

/// Evidence state for one input word, parallel to the request word list.
///
/// A normalization omission is a first-class variant, so consumers never
/// confuse "the model classified this as ordinary" with "the model never saw
/// this word". List position owns word identity; there is no redundant index
/// that could disagree with it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UtsegWordBoundaryEvidenceV2 {
    /// The normalized word reached the classifier.
    Classified {
        /// Raw classifier decision before adjacency policy.
        raw_action: UtsegBoundaryActionV2,
        /// Decision after adjacency policy, used to build assignments.
        applied_action: UtsegBoundaryActionV2,
        /// Sum of the model probabilities for the three true boundary labels.
        boundary_probability_micros: BoundaryProbabilityMicrosV2,
    },
    /// Normalization removed the word before classifier inference.
    NormalizationOmission,
    /// The normalized input was too short for classifier inference.
    ModelShortCircuit,
}

/// Provenance-bearing evidence emitted by one boundary-model invocation.
///
/// The wrapper prevents a per-word evidence vector from traveling without the
/// model identity needed to interpret or reproduce it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct UtsegBoundaryModelEvidenceV2 {
    /// HuggingFace model identifier the control plane pinned.
    pub model_id: String,
    /// The exact revision the worker loaded, and never anything else.
    ///
    /// Required, and a [`HubCommitV2`] rather than a string: the boundary model
    /// is resolved from a pinned snapshot whose commit is read off the
    /// directory on disk, so a worker that cannot say which revision it loaded
    /// refuses instead of reporting one. It was `Option<String>` while the
    /// model loaded by NAME and the revision was scraped from
    /// `config._commit_hash`, which could simply be absent; nothing downstream
    /// could distinguish "no revision" from "a revision nobody recorded".
    pub model_revision: HubCommitV2,
    /// Exact lexical-normalization semantics applied before inference.
    pub normalization_revision: UtsegNormalizationRevisionV2,
    /// Exact raw-to-applied adjacency-policy semantics.
    pub adjacency_policy_revision: UtsegAdjacencyPolicyRevisionV2,
    /// Evidence states parallel to the request word list.
    pub word_evidence: Vec<UtsegWordBoundaryEvidenceV2>,
}

/// A worker's claimed applied actions or assignments contradict its own raw
/// evidence and declared policy.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum UtsegBoundaryEvidenceConsistencyError {
    /// The applied action cannot be derived from the raw actions and policy.
    #[error(
        "applied action disagrees with declared adjacency policy at word {word_index}: expected {expected:?}, got {actual:?}"
    )]
    AppliedActionMismatch {
        /// Original request-word index.
        word_index: usize,
        /// Action obtained by replaying the declared policy.
        expected: UtsegBoundaryActionV2,
        /// Action claimed by the worker.
        actual: UtsegBoundaryActionV2,
    },
    /// Group assignments cannot be derived from the admitted applied actions.
    #[error("assignments disagree with applied boundary evidence")]
    AssignmentsMismatch,
}

/// Boundary evidence and assignments re-derived together under one requested
/// policy.
///
/// The fields are private so a caller cannot change the assignments without
/// also changing the evidence that proves them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReappliedUtsegBoundaryEvidenceV2 {
    evidence: UtsegBoundaryModelEvidenceV2,
    assignments: Vec<usize>,
}

/// Locally guarded assignments plus the boundary-model evidence they refine.
///
/// The evidence retains the adjacency-only decision. The private suppressed
/// indices are the complete explanation for why the applicable assignments
/// contain fewer splits than those raw local decisions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuardedUtsegBoundaryEvidenceV2 {
    evidence: UtsegBoundaryModelEvidenceV2,
    assignments: Vec<usize>,
    suppressed_split_before_word_indices: Vec<usize>,
}

/// A baseline assignment vector cannot prove which candidate splits are new
/// unless it is parallel to the retained evidence.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("worker assignments contain {actual} words but boundary evidence contains {expected}")]
pub struct UtsegProtectedSplitShapeError {
    expected: usize,
    actual: usize,
}

impl ReappliedUtsegBoundaryEvidenceV2 {
    /// Consume the proof and return its mutually consistent wire projections.
    pub fn into_parts(self) -> (UtsegBoundaryModelEvidenceV2, Vec<usize>) {
        (self.evidence, self.assignments)
    }

    /// Suppress only candidate boundaries whose resulting split begins at one
    /// of the proven protected word indices.
    pub fn protect_splits_before(
        self,
        protected_word_indices: &[usize],
        worker_assignments: &[usize],
    ) -> Result<GuardedUtsegBoundaryEvidenceV2, UtsegProtectedSplitShapeError> {
        if worker_assignments.len() != self.evidence.word_evidence.len() {
            return Err(UtsegProtectedSplitShapeError {
                expected: self.evidence.word_evidence.len(),
                actual: worker_assignments.len(),
            });
        }
        let mut suppress_after_word = vec![false; self.evidence.word_evidence.len()];
        let mut suppressed = Vec::new();
        for &split_before in protected_word_indices {
            if split_before == 0 || split_before >= self.assignments.len() {
                continue;
            }
            let candidate_splits =
                self.assignments[split_before - 1] != self.assignments[split_before];
            let worker_splits =
                worker_assignments[split_before - 1] != worker_assignments[split_before];
            if candidate_splits && !worker_splits {
                suppress_after_word[split_before - 1] = true;
                suppressed.push(split_before);
            }
        }
        suppressed.sort_unstable();
        suppressed.dedup();

        let mut current_group = 0;
        let assignments = self
            .evidence
            .word_evidence
            .iter()
            .enumerate()
            .map(|(word_index, evidence)| {
                let assignment = current_group;
                if !suppress_after_word[word_index]
                    && matches!(
                        evidence,
                        UtsegWordBoundaryEvidenceV2::Classified {
                            applied_action,
                            ..
                        } if applied_action.is_boundary()
                    )
                {
                    current_group += 1;
                }
                assignment
            })
            .collect();

        Ok(GuardedUtsegBoundaryEvidenceV2 {
            evidence: self.evidence,
            assignments,
            suppressed_split_before_word_indices: suppressed,
        })
    }
}

impl GuardedUtsegBoundaryEvidenceV2 {
    /// Consume the proof and return its evidence, applicable assignments, and
    /// complete local suppression receipt.
    pub fn into_parts(self) -> (UtsegBoundaryModelEvidenceV2, Vec<usize>, Vec<usize>) {
        (
            self.evidence,
            self.assignments,
            self.suppressed_split_before_word_indices,
        )
    }
}

impl UtsegBoundaryModelEvidenceV2 {
    /// Reapply one closed adjacency policy to the retained raw actions.
    ///
    /// Model inference is not repeated. Normalization omissions retain their
    /// positions, while adjacency is evaluated over the classified-word
    /// sequence exactly as it was in the worker.
    pub fn reapply_adjacency_policy(
        &self,
        policy: UtsegAdjacencyPolicyRevisionV2,
    ) -> ReappliedUtsegBoundaryEvidenceV2 {
        let classified: Vec<_> = self
            .word_evidence
            .iter()
            .enumerate()
            .filter_map(|(word_index, evidence)| match evidence {
                UtsegWordBoundaryEvidenceV2::Classified { raw_action, .. } => {
                    Some((word_index, *raw_action))
                }
                UtsegWordBoundaryEvidenceV2::NormalizationOmission
                | UtsegWordBoundaryEvidenceV2::ModelShortCircuit => None,
            })
            .collect();

        let mut word_evidence = self.word_evidence.clone();
        for (classified_index, (word_index, raw_action)) in classified.iter().copied().enumerate() {
            let applied_action = match classified.get(classified_index + 1) {
                Some((_, next_raw_action)) if policy.suppresses(raw_action, *next_raw_action) => {
                    UtsegBoundaryActionV2::Ordinary
                }
                Some(_) | None => raw_action,
            };
            if let UtsegWordBoundaryEvidenceV2::Classified {
                applied_action: target,
                ..
            } = &mut word_evidence[word_index]
            {
                *target = applied_action;
            }
        }

        let mut current_group = 0;
        let assignments = word_evidence
            .iter()
            .map(|evidence| {
                let assignment = current_group;
                if matches!(
                    evidence,
                    UtsegWordBoundaryEvidenceV2::Classified {
                        applied_action,
                        ..
                    } if applied_action.is_boundary()
                ) {
                    current_group += 1;
                }
                assignment
            })
            .collect();
        ReappliedUtsegBoundaryEvidenceV2 {
            evidence: Self {
                model_id: self.model_id.clone(),
                model_revision: self.model_revision.clone(),
                normalization_revision: self.normalization_revision,
                adjacency_policy_revision: policy,
                word_evidence,
            },
            assignments,
        }
    }

    /// Prove that declared policy, applied actions, and assignments describe
    /// one decision before the evidence can enter the application pipeline.
    pub fn validate_assignments(
        &self,
        assignments: &[usize],
    ) -> Result<(), UtsegBoundaryEvidenceConsistencyError> {
        let classified: Vec<_> = self
            .word_evidence
            .iter()
            .enumerate()
            .filter_map(|(word_index, evidence)| match evidence {
                UtsegWordBoundaryEvidenceV2::Classified {
                    raw_action,
                    applied_action,
                    boundary_probability_micros: _,
                } => Some((word_index, *raw_action, *applied_action)),
                UtsegWordBoundaryEvidenceV2::NormalizationOmission
                | UtsegWordBoundaryEvidenceV2::ModelShortCircuit => None,
            })
            .collect();

        let mut admitted_actions = vec![UtsegBoundaryActionV2::Ordinary; self.word_evidence.len()];
        for (classified_index, (word_index, raw_action, claimed_action)) in
            classified.iter().copied().enumerate()
        {
            let expected_action = match classified.get(classified_index + 1) {
                Some((_, next_raw_action, _))
                    if self
                        .adjacency_policy_revision
                        .suppresses(raw_action, *next_raw_action) =>
                {
                    UtsegBoundaryActionV2::Ordinary
                }
                Some(_) | None => raw_action,
            };
            if claimed_action != expected_action {
                return Err(
                    UtsegBoundaryEvidenceConsistencyError::AppliedActionMismatch {
                        word_index,
                        expected: expected_action,
                        actual: claimed_action,
                    },
                );
            }
            admitted_actions[word_index] = claimed_action;
        }

        let mut current_group = 0;
        let expected_assignments: Vec<_> = admitted_actions
            .into_iter()
            .map(|action| {
                let assignment = current_group;
                if action.is_boundary() {
                    current_group += 1;
                }
                assignment
            })
            .collect();
        if expected_assignments != assignments {
            return Err(UtsegBoundaryEvidenceConsistencyError::AssignmentsMismatch);
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)] // fixture construction; the Result shape is not the behavior under test
mod tests {
    use super::*;

    /// A commit-shaped revision for fixtures. The type admits nothing else, so
    /// a fixture cannot carry a placeholder like `"revision-1"` any more.
    fn commit() -> HubCommitV2 {
        HubCommitV2::try_from("0123456789abcdef0123456789abcdef01234567").expect("valid commit")
    }

    #[test]
    fn probability_refuses_values_above_the_unit_interval() {
        let result = serde_json::from_str::<BoundaryProbabilityMicrosV2>("1000001");
        assert!(
            matches!(result, Err(error) if error.to_string().contains("at most 1000000")),
            "out-of-range probability must be refused"
        );
    }

    #[test]
    fn boundary_only_replay_restores_boundary_before_capitalized_onset() {
        let probability = BoundaryProbabilityMicrosV2::try_from(900_000).expect("probability");
        let evidence = UtsegBoundaryModelEvidenceV2 {
            model_id: "model".into(),
            model_revision: commit(),
            normalization_revision: UtsegNormalizationRevisionV2::LowerStripAsciiPunctuationV1,
            adjacency_policy_revision:
                UtsegAdjacencyPolicyRevisionV2::SuppressEarlierAdjacentNonordinaryV1,
            word_evidence: vec![
                UtsegWordBoundaryEvidenceV2::Classified {
                    raw_action: UtsegBoundaryActionV2::PeriodBoundary,
                    applied_action: UtsegBoundaryActionV2::Ordinary,
                    boundary_probability_micros: probability,
                },
                UtsegWordBoundaryEvidenceV2::Classified {
                    raw_action: UtsegBoundaryActionV2::CapitalizedOnset,
                    applied_action: UtsegBoundaryActionV2::CapitalizedOnset,
                    boundary_probability_micros: probability,
                },
            ],
        };

        let (candidate, assignments) = evidence
            .reapply_adjacency_policy(
                UtsegAdjacencyPolicyRevisionV2::SuppressEarlierAdjacentBoundariesV1,
            )
            .into_parts();

        assert_eq!(assignments, vec![0, 1]);
        assert_eq!(
            candidate.adjacency_policy_revision,
            UtsegAdjacencyPolicyRevisionV2::SuppressEarlierAdjacentBoundariesV1
        );
        candidate
            .validate_assignments(&assignments)
            .expect("reapplied evidence remains internally consistent");
    }

    #[test]
    fn protected_split_rebuilds_assignments_without_mutating_model_evidence() {
        let probability = BoundaryProbabilityMicrosV2::try_from(900_000).expect("probability");
        let evidence = UtsegBoundaryModelEvidenceV2 {
            model_id: "model".into(),
            model_revision: commit(),
            normalization_revision: UtsegNormalizationRevisionV2::LowerStripAsciiPunctuationV1,
            adjacency_policy_revision:
                UtsegAdjacencyPolicyRevisionV2::SuppressEarlierAdjacentNonordinaryV1,
            word_evidence: vec![
                UtsegWordBoundaryEvidenceV2::Classified {
                    raw_action: UtsegBoundaryActionV2::PeriodBoundary,
                    applied_action: UtsegBoundaryActionV2::Ordinary,
                    boundary_probability_micros: probability,
                },
                UtsegWordBoundaryEvidenceV2::Classified {
                    raw_action: UtsegBoundaryActionV2::CapitalizedOnset,
                    applied_action: UtsegBoundaryActionV2::CapitalizedOnset,
                    boundary_probability_micros: probability,
                },
            ],
        };

        let (candidate, assignments, suppressed) = evidence
            .reapply_adjacency_policy(
                UtsegAdjacencyPolicyRevisionV2::SuppressEarlierAdjacentBoundariesV1,
            )
            .protect_splits_before(&[1], &[0, 0])
            .expect("parallel worker assignments")
            .into_parts();

        assert_eq!(assignments, vec![0, 0]);
        assert_eq!(suppressed, vec![1]);
        assert!(matches!(
            candidate.word_evidence[0],
            UtsegWordBoundaryEvidenceV2::Classified {
                applied_action: UtsegBoundaryActionV2::PeriodBoundary,
                ..
            }
        ));
    }

    #[test]
    fn protected_worker_declared_split_is_never_merged() {
        let probability = BoundaryProbabilityMicrosV2::try_from(900_000).expect("probability");
        let evidence = UtsegBoundaryModelEvidenceV2 {
            model_id: "model".into(),
            model_revision: commit(),
            normalization_revision: UtsegNormalizationRevisionV2::LowerStripAsciiPunctuationV1,
            adjacency_policy_revision:
                UtsegAdjacencyPolicyRevisionV2::SuppressEarlierAdjacentNonordinaryV1,
            word_evidence: vec![
                UtsegWordBoundaryEvidenceV2::Classified {
                    raw_action: UtsegBoundaryActionV2::PeriodBoundary,
                    applied_action: UtsegBoundaryActionV2::PeriodBoundary,
                    boundary_probability_micros: probability,
                },
                UtsegWordBoundaryEvidenceV2::Classified {
                    raw_action: UtsegBoundaryActionV2::Ordinary,
                    applied_action: UtsegBoundaryActionV2::Ordinary,
                    boundary_probability_micros: probability,
                },
            ],
        };

        let (_, assignments, suppressed) = evidence
            .reapply_adjacency_policy(
                UtsegAdjacencyPolicyRevisionV2::SuppressEarlierAdjacentBoundariesV1,
            )
            .protect_splits_before(&[1], &[0, 1])
            .expect("parallel worker assignments")
            .into_parts();

        assert_eq!(assignments, vec![0, 1]);
        assert!(suppressed.is_empty());
    }
}
