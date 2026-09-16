"""Pure policy tests for replaying raw utterance-boundary evidence."""

import pytest

from batchalign.models.utterance.evidence import (
    BoundaryAction,
    BoundaryAdjacencyPolicy,
    BoundaryProbability,
    ClassifiedBoundaryEvidence,
    NormalizedBoundaryEvidence,
    UtteranceBoundaryPrediction,
    apply_boundary_adjacency_policy,
)


def evidence(action: BoundaryAction, probability: float) -> NormalizedBoundaryEvidence:
    return NormalizedBoundaryEvidence(
        raw_action=action,
        boundary_probability=BoundaryProbability.from_float(probability),
    )


def test_current_policy_suppresses_boundary_before_nonboundary_onset() -> None:
    raw = [
        evidence(BoundaryAction.PERIOD_BOUNDARY, 0.9),
        evidence(BoundaryAction.CAPITALIZED_ONSET, 0.01),
    ]

    applied = apply_boundary_adjacency_policy(
        raw, BoundaryAdjacencyPolicy.SUPPRESS_EARLIER_ADJACENT_NONORDINARY_V1
    )

    assert applied == (
        BoundaryAction.ORDINARY,
        BoundaryAction.CAPITALIZED_ONSET,
    )


def test_boundary_only_policy_preserves_boundary_before_nonboundary_onset() -> None:
    raw = [
        evidence(BoundaryAction.PERIOD_BOUNDARY, 0.9),
        evidence(BoundaryAction.CAPITALIZED_ONSET, 0.01),
    ]

    applied = apply_boundary_adjacency_policy(
        raw, BoundaryAdjacencyPolicy.SUPPRESS_EARLIER_ADJACENT_BOUNDARIES_V1
    )

    assert applied == (
        BoundaryAction.PERIOD_BOUNDARY,
        BoundaryAction.CAPITALIZED_ONSET,
    )


def test_boundary_only_policy_still_suppresses_first_of_two_true_boundaries() -> None:
    raw = [
        evidence(BoundaryAction.QUESTION_BOUNDARY, 0.7),
        evidence(BoundaryAction.PERIOD_BOUNDARY, 0.8),
    ]

    applied = apply_boundary_adjacency_policy(
        raw, BoundaryAdjacencyPolicy.SUPPRESS_EARLIER_ADJACENT_BOUNDARIES_V1
    )

    assert applied == (
        BoundaryAction.ORDINARY,
        BoundaryAction.PERIOD_BOUNDARY,
    )


def test_prediction_refuses_applied_actions_not_derived_from_declared_policy() -> None:
    word_evidence = (
        ClassifiedBoundaryEvidence(
            raw_action=BoundaryAction.PERIOD_BOUNDARY,
            applied_action=BoundaryAction.PERIOD_BOUNDARY,
            boundary_probability=BoundaryProbability.from_float(0.9),
        ),
        ClassifiedBoundaryEvidence(
            raw_action=BoundaryAction.CAPITALIZED_ONSET,
            applied_action=BoundaryAction.CAPITALIZED_ONSET,
            boundary_probability=BoundaryProbability.from_float(0.01),
        ),
    )

    with pytest.raises(ValueError, match="applied action"):
        UtteranceBoundaryPrediction(
            model_id="test/model",
            model_revision="0123456789abcdef0123456789abcdef01234567",
            word_evidence=word_evidence,
        )
