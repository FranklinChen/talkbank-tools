"""Typed evidence emitted by the TalkBank utterance-boundary classifier.

The model owns the relationship between raw labels, applied policy labels,
normalization omissions, and resulting group assignments. Consumers receive
one immutable prediction instead of parallel primitive lists they could pair
incorrectly.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import assert_never

_PROBABILITY_SCALE = 1_000_000

_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
"""A hub commit, the only shape a boundary-model revision can take.

The model is loaded from a pinned snapshot, so its revision is always the exact
commit the resolver verified on disk. Checking the shape here means a fixture
cannot smuggle a placeholder like ``"test-revision"`` into evidence that Rust
would then have to refuse at the wire.
"""


class UtteranceNormalizationRevision(str, Enum):
    """Closed lexical-normalization semantics for boundary inference."""

    LOWER_STRIP_ASCII_PUNCTUATION_V1 = "lower-strip-ascii-punctuation-v1"


class BoundaryAdjacencyPolicy(str, Enum):
    """Closed raw-to-applied action policies available for exact replay."""

    SUPPRESS_EARLIER_ADJACENT_NONORDINARY_V1 = (
        "suppress-earlier-adjacent-nonordinary-v1"
    )
    SUPPRESS_EARLIER_ADJACENT_BOUNDARIES_V1 = "suppress-earlier-adjacent-boundaries-v1"


UTTERANCE_NORMALIZATION_REVISION = (
    UtteranceNormalizationRevision.LOWER_STRIP_ASCII_PUNCTUATION_V1
)
UTTERANCE_ADJACENCY_POLICY_REVISION = (
    BoundaryAdjacencyPolicy.SUPPRESS_EARLIER_ADJACENT_NONORDINARY_V1
)


class BoundaryAction(str, Enum):
    """Closed semantic vocabulary for the classifier's six label indices."""

    ORDINARY = "ordinary"
    CAPITALIZED_ONSET = "capitalized_onset"
    PERIOD_BOUNDARY = "period_boundary"
    QUESTION_BOUNDARY = "question_boundary"
    EXCLAMATION_BOUNDARY = "exclamation_boundary"
    COMMA = "comma"

    @property
    def label_index(self) -> int:
        """Return the stable HuggingFace classifier label index."""

        return _ACTION_TO_LABEL_INDEX[self]

    @property
    def is_boundary(self) -> bool:
        """Whether this action advances the output utterance group."""

        return self in _BOUNDARY_ACTIONS

    @classmethod
    def from_label_index(cls, label_index: int) -> BoundaryAction:
        """Admit one classifier index into the closed action vocabulary."""

        try:
            return _LABEL_INDEX_TO_ACTION[label_index]
        except KeyError as error:
            raise ValueError(
                f"utterance classifier returned unknown label index {label_index}"
            ) from error


_ACTION_TO_LABEL_INDEX: dict[BoundaryAction, int] = {
    BoundaryAction.ORDINARY: 0,
    BoundaryAction.CAPITALIZED_ONSET: 1,
    BoundaryAction.PERIOD_BOUNDARY: 2,
    BoundaryAction.QUESTION_BOUNDARY: 3,
    BoundaryAction.EXCLAMATION_BOUNDARY: 4,
    BoundaryAction.COMMA: 5,
}
_LABEL_INDEX_TO_ACTION = {
    label_index: action for action, label_index in _ACTION_TO_LABEL_INDEX.items()
}
_BOUNDARY_ACTIONS = frozenset(
    {
        BoundaryAction.PERIOD_BOUNDARY,
        BoundaryAction.QUESTION_BOUNDARY,
        BoundaryAction.EXCLAMATION_BOUNDARY,
    }
)


@dataclass(frozen=True, slots=True, init=False)
class BoundaryProbability:
    """Finite fixed-point probability in the inclusive unit interval."""

    micros: int

    @classmethod
    def from_float(cls, probability: float) -> BoundaryProbability:
        """Quantize a finite unit-interval probability at micro precision."""

        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("boundary probability must be finite and within [0, 1]")
        instance = object.__new__(cls)
        object.__setattr__(instance, "micros", round(probability * _PROBABILITY_SCALE))
        return instance


@dataclass(frozen=True, slots=True)
class ClassifiedBoundaryEvidence:
    """Raw and applied evidence for one word seen by the classifier."""

    raw_action: BoundaryAction
    applied_action: BoundaryAction
    boundary_probability: BoundaryProbability


@dataclass(frozen=True, slots=True)
class NormalizationOmission:
    """The corresponding input word normalized to no classifier token."""


@dataclass(frozen=True, slots=True)
class ModelShortCircuit:
    """The word was not classified because the normalized input was too short."""


WordBoundaryEvidence = (
    ClassifiedBoundaryEvidence | NormalizationOmission | ModelShortCircuit
)
"""One exhaustive model-evidence state parallel to an input word."""


@dataclass(frozen=True, slots=True)
class NormalizedBoundaryEvidence:
    """Raw classifier evidence for one normalized word before policy."""

    raw_action: BoundaryAction
    boundary_probability: BoundaryProbability


def apply_boundary_adjacency_policy(
    evidence: list[NormalizedBoundaryEvidence], policy: BoundaryAdjacencyPolicy
) -> tuple[BoundaryAction, ...]:
    """Replay one closed adjacency policy over immutable raw evidence."""

    applied = [item.raw_action for item in evidence]
    for word_index, item in enumerate(evidence[:-1]):
        next_item = evidence[word_index + 1]
        match policy:
            case BoundaryAdjacencyPolicy.SUPPRESS_EARLIER_ADJACENT_NONORDINARY_V1:
                suppress = item.raw_action is not BoundaryAction.ORDINARY and (
                    next_item.raw_action is not BoundaryAction.ORDINARY
                )
            case BoundaryAdjacencyPolicy.SUPPRESS_EARLIER_ADJACENT_BOUNDARIES_V1:
                suppress = (
                    item.raw_action.is_boundary and next_item.raw_action.is_boundary
                )
            case _ as unreachable:
                assert_never(unreachable)
        if suppress:
            applied[word_index] = BoundaryAction.ORDINARY
    return tuple(applied)


@dataclass(frozen=True, slots=True)
class UtteranceBoundaryPrediction:
    """Provenance-bearing immutable account parallel to original input words."""

    model_id: str
    model_revision: str
    word_evidence: tuple[WordBoundaryEvidence, ...]
    normalization_revision: UtteranceNormalizationRevision = (
        UTTERANCE_NORMALIZATION_REVISION
    )
    adjacency_policy_revision: BoundaryAdjacencyPolicy = (
        UTTERANCE_ADJACENCY_POLICY_REVISION
    )

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("utterance boundary model id must not be empty")
        if not _COMMIT_PATTERN.match(self.model_revision):
            raise ValueError(
                "utterance boundary model revision must be a 40-character hub "
                f"commit, got {self.model_revision!r}"
            )
        classified = [
            (word_index, item)
            for word_index, item in enumerate(self.word_evidence)
            if isinstance(item, ClassifiedBoundaryEvidence)
        ]
        expected_actions = apply_boundary_adjacency_policy(
            [
                NormalizedBoundaryEvidence(
                    raw_action=item.raw_action,
                    boundary_probability=item.boundary_probability,
                )
                for _, item in classified
            ],
            self.adjacency_policy_revision,
        )
        for (word_index, item), expected_action in zip(
            classified, expected_actions, strict=True
        ):
            if item.applied_action is not expected_action:
                raise ValueError(
                    "utterance boundary applied action disagrees with declared "
                    f"adjacency policy at word {word_index}"
                )

    @property
    def applied_actions(self) -> tuple[BoundaryAction, ...]:
        """Derive applied actions without storing a second parallel truth."""

        return tuple(
            item.applied_action
            if isinstance(item, ClassifiedBoundaryEvidence)
            else BoundaryAction.ORDINARY
            for item in self.word_evidence
        )

    @property
    def assignments(self) -> tuple[int, ...]:
        """Derive group assignments from the applied action sequence."""

        assignments: list[int] = []
        current_group = 0
        for action in self.applied_actions:
            assignments.append(current_group)
            if action.is_boundary:
                current_group += 1
        return tuple(assignments)
