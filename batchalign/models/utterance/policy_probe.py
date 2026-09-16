"""Reproducible policy replay over retained pre-CHAT ASR evidence.

This module deliberately separates the expensive model capture from the cheap
adjacency-policy decision.  One model prediction retains raw word evidence;
closed policies can then be compared without another provider request or model
forward pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from batchalign.models.utterance.evidence import (
    UTTERANCE_ADJACENCY_POLICY_REVISION,
    BoundaryAction,
    BoundaryAdjacencyPolicy,
    ClassifiedBoundaryEvidence,
    NormalizedBoundaryEvidence,
    UtteranceBoundaryPrediction,
    apply_boundary_adjacency_policy,
)

_PROVIDER_TAG_RE = re.compile(r"^<[^<>]+>$")
_CANDIDATE_POLICY = BoundaryAdjacencyPolicy.SUPPRESS_EARLIER_ADJACENT_BOUNDARIES_V1


class PolicyProbeError(ValueError):
    """Retained evidence could not be admitted into the policy experiment."""


@dataclass(frozen=True, slots=True)
class ProbeLanguage:
    """One explicit lowercase ISO 639-3 experiment language."""

    code: str

    @classmethod
    def admit(cls, code: str) -> ProbeLanguage:
        if (
            len(code) != 3
            or not code.isascii()
            or not code.isalpha()
            or not code.islower()
        ):
            raise PolicyProbeError(
                "utterance policy probe language must be a lowercase ISO 639-3 code"
            )
        return cls(code=code)


class RetainedElementKind(str, Enum):
    """Closed element vocabulary in a retained ``AsrResponse`` artifact."""

    TEXT = "text"
    PUNCTUATION = "punctuation"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetainedAsrToken(_StrictModel):
    """One token in the debug artifact's flattened compatibility view."""

    text: str
    start_s: float | None = Field(default=None, ge=0.0)
    end_s: float | None = Field(default=None, ge=0.0)
    speaker: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_range(self) -> RetainedAsrToken:
        if (
            self.start_s is not None
            and self.end_s is not None
            and self.end_s < self.start_s
        ):
            raise ValueError("retained token end_s must be >= start_s")
        return self


class RetainedAsrElement(_StrictModel):
    """One provider-shaped source element retained by BA3 debug output."""

    value: str
    ts: float | None = Field(default=None, ge=0.0)
    end_ts: float | None = Field(default=None, ge=0.0)
    kind: RetainedElementKind

    @model_validator(mode="after")
    def _validate_range(self) -> RetainedAsrElement:
        if self.ts is not None and self.end_ts is not None and self.end_ts < self.ts:
            raise ValueError("retained element end_ts must be >= ts")
        return self


class RetainedAsrMonologue(_StrictModel):
    """One provider monologue before BA3 segmentation."""

    speaker: int | str
    elements: tuple[RetainedAsrElement, ...]


class RetainedAsrResponse(_StrictModel):
    """Exact admitted shape of one ``*_asr_response.json`` artifact."""

    tokens: tuple[RetainedAsrToken, ...]
    lang: str
    source_monologues: tuple[RetainedAsrMonologue, ...]


class BoundaryEvidenceModel(Protocol):
    """The single expensive operation needed by the policy probe."""

    def predict_boundary_evidence(
        self, words: tuple[str, ...]
    ) -> UtteranceBoundaryPrediction: ...


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Stable identity shared by every admitted prediction in one report."""

    model_id: str
    revision: str


@dataclass(frozen=True, slots=True)
class ProbePopulation:
    """Exact population admitted from retained artifacts."""

    file_count: int
    monologue_count: int
    word_count: int
    excluded_provider_tag_count: int


@dataclass(frozen=True, slots=True)
class InputPolicySummary:
    """Counts and provenance for one retained input file."""

    name: str
    sha256: str
    monologue_count: int
    word_count: int
    excluded_provider_tag_count: int
    action_difference_count: int
    restored_boundary_count: int


@dataclass(frozen=True, slots=True)
class ProbeProgress:
    """One immutable progress event after a whole input is complete."""

    completed_files: int
    total_files: int
    latest_input: InputPolicySummary


type ProbeProgressObserver = Callable[[ProbeProgress], None]
"""Receives progress without becoming part of experiment semantics."""


@dataclass(frozen=True, slots=True)
class ActionTransitionCount:
    """How often one baseline action becomes one candidate action."""

    baseline: BoundaryAction
    candidate: BoundaryAction
    count: int


@dataclass(frozen=True, slots=True)
class KnownInterwordTiming:
    """Signed temporal distance from the left word end to right word start."""

    delta_micros: int


@dataclass(frozen=True, slots=True)
class MissingInterwordTiming:
    """At least one endpoint needed for an interword delta was unavailable."""


type InterwordTiming = KnownInterwordTiming | MissingInterwordTiming


@dataclass(frozen=True, slots=True)
class RestoredBoundary:
    """A candidate boundary erased by the current adjacency policy."""

    input_name: str
    monologue_ordinal: int
    word_index: int
    left_word: str
    right_word: str
    left_context: tuple[str, ...]
    right_context: tuple[str, ...]
    raw_action: BoundaryAction
    following_raw_action: BoundaryAction
    boundary_probability_micros: int
    interword_timing: InterwordTiming


@dataclass(frozen=True, slots=True)
class PolicyComparisonReport:
    """Complete immutable result of one controlled policy replay."""

    schema_version: int
    language: ProbeLanguage
    baseline_policy: BoundaryAdjacencyPolicy
    candidate_policy: BoundaryAdjacencyPolicy
    model: ModelIdentity
    population: ProbePopulation
    per_input: tuple[InputPolicySummary, ...]
    action_difference_count: int
    assignment_changing_difference_count: int
    transitions: tuple[ActionTransitionCount, ...]
    restored_boundaries: tuple[RestoredBoundary, ...]

    def to_json_value(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation."""

        return {
            "schema_version": self.schema_version,
            "language": self.language.code,
            "baseline_policy": self.baseline_policy.value,
            "candidate_policy": self.candidate_policy.value,
            "model": asdict(self.model),
            "population": asdict(self.population),
            "inputs": [asdict(item) for item in self.per_input],
            "action_difference_count": self.action_difference_count,
            "assignment_changing_difference_count": (
                self.assignment_changing_difference_count
            ),
            "transitions": [
                {
                    "baseline": item.baseline.value,
                    "candidate": item.candidate.value,
                    "count": item.count,
                }
                for item in self.transitions
            ],
            "restored_boundaries": [
                {
                    **asdict(item),
                    "raw_action": item.raw_action.value,
                    "following_raw_action": item.following_raw_action.value,
                    "interword_timing": _interword_timing_json(item.interword_timing),
                }
                for item in self.restored_boundaries
            ],
        }


def write_policy_report(report: PolicyComparisonReport, output_path: Path) -> None:
    """Atomically publish one stable, complete JSON report."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        report.to_json_value(), indent=2, sort_keys=True, ensure_ascii=False
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(output_path)
        directory_fd = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class _AdmittedPrediction:
    """A model prediction proven parallel to the exact source words."""

    words: tuple[str, ...]
    prediction: UtteranceBoundaryPrediction

    @classmethod
    def admit(
        cls,
        words: tuple[str, ...],
        prediction: UtteranceBoundaryPrediction,
    ) -> _AdmittedPrediction:
        if len(prediction.word_evidence) != len(words):
            raise PolicyProbeError(
                "utterance prediction evidence length does not match source words"
            )
        if prediction.adjacency_policy_revision is not (
            UTTERANCE_ADJACENCY_POLICY_REVISION
        ):
            raise PolicyProbeError(
                "utterance prediction was not produced by the baseline policy"
            )
        return cls(words=words, prediction=prediction)

    def replay_candidate(self) -> tuple[BoundaryAction, ...]:
        """Replay the candidate policy and expand it to source-word indices."""

        classified = [
            (word_index, item)
            for word_index, item in enumerate(self.prediction.word_evidence)
            if isinstance(item, ClassifiedBoundaryEvidence)
        ]
        normalized = [
            NormalizedBoundaryEvidence(
                raw_action=item.raw_action,
                boundary_probability=item.boundary_probability,
            )
            for _, item in classified
        ]
        replayed = apply_boundary_adjacency_policy(normalized, _CANDIDATE_POLICY)
        expanded = [BoundaryAction.ORDINARY] * len(self.words)
        for (word_index, _), action in zip(classified, replayed, strict=True):
            expanded[word_index] = action
        return tuple(expanded)


@dataclass(frozen=True, slots=True)
class _SourceWord:
    """One admitted lexical item with its optional provider timing."""

    text: str
    start_micros: int | None
    end_micros: int | None


@dataclass(frozen=True, slots=True)
class _SourceMonologueWords:
    """Lexical population and exclusion count produced together."""

    words: tuple[_SourceWord, ...]
    excluded_provider_tag_count: int

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(word.text for word in self.words)


def _optional_seconds_to_micros(seconds: float | None) -> int | None:
    return None if seconds is None else round(seconds * 1_000_000)


def _source_words(monologue: RetainedAsrMonologue) -> _SourceMonologueWords:
    words: list[_SourceWord] = []
    excluded_provider_tags = 0
    for element in monologue.elements:
        if element.kind is not RetainedElementKind.TEXT:
            continue
        if _PROVIDER_TAG_RE.fullmatch(element.value):
            excluded_provider_tags += 1
            continue
        words.append(
            _SourceWord(
                text=element.value,
                start_micros=_optional_seconds_to_micros(element.ts),
                end_micros=_optional_seconds_to_micros(element.end_ts),
            )
        )
    return _SourceMonologueWords(
        words=tuple(words), excluded_provider_tag_count=excluded_provider_tags
    )


def _interword_timing(left: _SourceWord, right: _SourceWord) -> InterwordTiming:
    if left.end_micros is None or right.start_micros is None:
        return MissingInterwordTiming()
    return KnownInterwordTiming(delta_micros=right.start_micros - left.end_micros)


def _interword_timing_json(timing: InterwordTiming) -> dict[str, int | str]:
    match timing:
        case KnownInterwordTiming(delta_micros=delta_micros):
            return {"kind": "known", "delta_micros": delta_micros}
        case MissingInterwordTiming():
            return {"kind": "missing"}


def _prediction_identity(prediction: UtteranceBoundaryPrediction) -> ModelIdentity:
    return ModelIdentity(
        model_id=prediction.model_id,
        revision=prediction.model_revision,
    )


def compare_retained_asr_files(
    input_paths: tuple[Path, ...],
    model: BoundaryEvidenceModel,
    *,
    expected_language: ProbeLanguage,
    progress_observer: ProbeProgressObserver | None = None,
) -> PolicyComparisonReport:
    """Capture raw evidence once and compare two closed adjacency policies."""

    if not input_paths:
        raise PolicyProbeError("at least one retained ASR input is required")

    identity: ModelIdentity | None = None
    total_monologues = 0
    total_words = 0
    total_excluded_tags = 0
    total_action_differences = 0
    transitions: Counter[tuple[BoundaryAction, BoundaryAction]] = Counter()
    restored: list[RestoredBoundary] = []
    input_summaries: list[InputPolicySummary] = []

    sorted_input_paths = sorted(input_paths, key=lambda path: path.name)
    for completed_index, input_path in enumerate(sorted_input_paths, start=1):
        raw = input_path.read_bytes()
        response = RetainedAsrResponse.model_validate_json(raw)
        if response.lang != expected_language.code:
            raise PolicyProbeError(
                f"retained input {input_path.name} has language {response.lang!r}, "
                f"expected {expected_language.code!r}"
            )
        input_monologues = 0
        input_words = 0
        input_excluded_tags = 0
        input_action_differences = 0
        input_restored_before = len(restored)

        for monologue_ordinal, monologue in enumerate(response.source_monologues):
            source = _source_words(monologue)
            words = source.texts
            input_excluded_tags += source.excluded_provider_tag_count
            if not words:
                continue

            admitted = _AdmittedPrediction.admit(
                words, model.predict_boundary_evidence(words)
            )
            prediction_identity = _prediction_identity(admitted.prediction)
            if identity is None:
                identity = prediction_identity
            elif identity != prediction_identity:
                raise PolicyProbeError("model identity drifted within one probe run")

            baseline = admitted.prediction.applied_actions
            candidate = admitted.replay_candidate()
            raw_evidence = admitted.prediction.word_evidence
            for word_index, (before, after) in enumerate(
                zip(baseline, candidate, strict=True)
            ):
                if before is after:
                    continue
                input_action_differences += 1
                transitions[(before, after)] += 1
                if before.is_boundary or not after.is_boundary:
                    continue
                item = raw_evidence[word_index]
                if not isinstance(item, ClassifiedBoundaryEvidence):
                    raise PolicyProbeError(
                        "candidate changed an action without classified evidence"
                    )
                following = next(
                    (
                        following_item
                        for following_item in raw_evidence[word_index + 1 :]
                        if isinstance(following_item, ClassifiedBoundaryEvidence)
                    ),
                    None,
                )
                if following is None:
                    raise PolicyProbeError(
                        "restored boundary has no following classified word"
                    )
                right_index = word_index + 1
                while right_index < len(raw_evidence) and not isinstance(
                    raw_evidence[right_index], ClassifiedBoundaryEvidence
                ):
                    right_index += 1
                restored.append(
                    RestoredBoundary(
                        input_name=input_path.name,
                        monologue_ordinal=monologue_ordinal,
                        word_index=word_index,
                        left_word=words[word_index],
                        right_word=words[right_index],
                        left_context=words[max(0, word_index - 5) : word_index + 1],
                        right_context=words[right_index : right_index + 6],
                        raw_action=item.raw_action,
                        following_raw_action=following.raw_action,
                        boundary_probability_micros=(item.boundary_probability.micros),
                        interword_timing=_interword_timing(
                            source.words[word_index], source.words[right_index]
                        ),
                    )
                )

            input_monologues += 1
            input_words += len(words)

        input_summary = InputPolicySummary(
            name=input_path.name,
            sha256=hashlib.sha256(raw).hexdigest(),
            monologue_count=input_monologues,
            word_count=input_words,
            excluded_provider_tag_count=input_excluded_tags,
            action_difference_count=input_action_differences,
            restored_boundary_count=len(restored) - input_restored_before,
        )
        input_summaries.append(input_summary)
        if progress_observer is not None:
            progress_observer(
                ProbeProgress(
                    completed_files=completed_index,
                    total_files=len(sorted_input_paths),
                    latest_input=input_summary,
                )
            )
        total_monologues += input_monologues
        total_words += input_words
        total_excluded_tags += input_excluded_tags
        total_action_differences += input_action_differences

    if identity is None:
        raise PolicyProbeError("retained inputs contained no text monologues")

    transition_counts = tuple(
        ActionTransitionCount(baseline=before, candidate=after, count=count)
        for (before, after), count in sorted(
            transitions.items(), key=lambda item: (item[0][0].value, item[0][1].value)
        )
    )
    return PolicyComparisonReport(
        schema_version=1,
        language=expected_language,
        baseline_policy=UTTERANCE_ADJACENCY_POLICY_REVISION,
        candidate_policy=_CANDIDATE_POLICY,
        model=identity,
        population=ProbePopulation(
            file_count=len(input_paths),
            monologue_count=total_monologues,
            word_count=total_words,
            excluded_provider_tag_count=total_excluded_tags,
        ),
        per_input=tuple(input_summaries),
        action_difference_count=total_action_differences,
        assignment_changing_difference_count=len(restored),
        transitions=transition_counts,
        restored_boundaries=tuple(restored),
    )
