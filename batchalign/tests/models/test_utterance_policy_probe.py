"""End-to-end tests for retained-evidence utterance policy comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from batchalign.models.utterance.evidence import (
    BoundaryAction,
    BoundaryProbability,
    ClassifiedBoundaryEvidence,
    UtteranceBoundaryPrediction,
)
from batchalign.models.utterance.policy_probe import (
    KnownInterwordTiming,
    PolicyProbeError,
    ProbeLanguage,
    ProbeProgress,
    compare_retained_asr_files,
    write_policy_report,
)


class FakeBoundaryModel:
    """Deterministic model seam for the retained-ASR experiment boundary."""

    def predict_boundary_evidence(
        self, words: tuple[str, ...]
    ) -> UtteranceBoundaryPrediction:
        actions = {
            "finish": BoundaryAction.PERIOD_BOUNDARY,
            "Then": BoundaryAction.CAPITALIZED_ONSET,
        }
        return UtteranceBoundaryPrediction(
            model_id="test/utterance",
            model_revision="0123456789abcdef0123456789abcdef01234567",
            word_evidence=tuple(
                ClassifiedBoundaryEvidence(
                    raw_action=actions.get(word, BoundaryAction.ORDINARY),
                    applied_action=(
                        BoundaryAction.ORDINARY
                        if word == "finish"
                        else actions.get(word, BoundaryAction.ORDINARY)
                    ),
                    boundary_probability=BoundaryProbability.from_float(
                        0.9 if word == "finish" else 0.01
                    ),
                )
                for word in words
            ),
        )


def test_compare_retained_asr_files_reports_only_assignment_changing_replay(
    tmp_path: Path,
) -> None:
    retained = tmp_path / "sample_asr_response.json"
    retained.write_text(
        json.dumps(
            {
                "tokens": [
                    {
                        "text": "we",
                        "start_s": 0.0,
                        "end_s": 0.1,
                        "speaker": "0",
                        "confidence": 0.99,
                    }
                ],
                "lang": "eng",
                "source_monologues": [
                    {
                        "speaker": 0,
                        "elements": [
                            {"value": "we", "ts": 0.0, "end_ts": 0.1, "kind": "text"},
                            {
                                "value": "finish",
                                "ts": 0.1,
                                "end_ts": 0.2,
                                "kind": "text",
                            },
                            {
                                "value": "<laugh>",
                                "ts": 0.2,
                                "end_ts": 0.3,
                                "kind": "text",
                            },
                            {
                                "value": "Then",
                                "ts": 0.3,
                                "end_ts": 0.4,
                                "kind": "text",
                            },
                            {"value": ".", "kind": "punctuation"},
                            {
                                "value": "continue",
                                "ts": 0.4,
                                "end_ts": 0.5,
                                "kind": "text",
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    progress: list[ProbeProgress] = []
    report = compare_retained_asr_files(
        (retained,),
        FakeBoundaryModel(),
        expected_language=ProbeLanguage.admit("eng"),
        progress_observer=progress.append,
    )

    assert report.schema_version == 1
    assert report.language.code == "eng"
    assert report.population.file_count == 1
    assert report.population.monologue_count == 1
    assert report.population.word_count == 4
    assert report.population.excluded_provider_tag_count == 1
    assert report.model.model_id == "test/utterance"
    assert report.model.revision == "0123456789abcdef0123456789abcdef01234567"
    assert report.assignment_changing_difference_count == 1
    assert report.per_input[0].restored_boundary_count == 1
    restored = report.restored_boundaries[0]
    assert restored.left_word == "finish"
    assert restored.right_word == "Then"
    assert restored.boundary_probability_micros == 900_000
    assert restored.word_index == 1
    assert isinstance(restored.interword_timing, KnownInterwordTiming)
    assert restored.interword_timing.delta_micros == 100_000
    assert report.to_json_value()["inputs"][0]["sha256"]
    assert progress[0].completed_files == 1
    assert progress[0].total_files == 1
    assert progress[0].latest_input.name == retained.name

    output = tmp_path / "report" / "comparison.json"
    write_policy_report(report, output)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["assignment_changing_difference_count"] == 1
    assert not tuple(output.parent.glob(f".{output.name}.*.tmp"))

    retained_value = json.loads(retained.read_text(encoding="utf-8"))
    retained_value["lang"] = "spa"
    retained.write_text(json.dumps(retained_value), encoding="utf-8")
    with pytest.raises(PolicyProbeError, match="language"):
        compare_retained_asr_files(
            (retained,),
            FakeBoundaryModel(),
            expected_language=ProbeLanguage.admit("eng"),
        )
