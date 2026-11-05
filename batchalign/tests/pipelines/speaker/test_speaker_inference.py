"""Tests for the thin Python speaker inference boundary."""

from __future__ import annotations

import tempfile
from types import SimpleNamespace

import numpy as np

from batchalign.inference.pyannote_ai import CompletedDiarizationJob
from batchalign.inference.pyannote_local import LoadedPyannotePipeline
from batchalign.inference.speaker import (
    LocalPyannoteSpeakerEvidence,
    PyannoteAISpeakerEvidence,
    SpeakerSegment,
    infer_local_pyannote_prepared_audio,
    infer_pyannote_ai_prepared_audio,
    infer_pyannote_speaker_prepared_audio,
)


def _loaded(pipe: object) -> LoadedPyannotePipeline:
    """A loaded-pipeline handle around a fake pipeline, as the speaker task
    load would hold it (its private model files are an empty directory)."""
    return LoadedPyannotePipeline(pipe, tempfile.TemporaryDirectory())


def test_local_pyannote_runs_on_the_loaded_pipeline() -> None:
    """The local entry point takes the loaded handle and runs it, loading
    nothing; its segments become local evidence."""

    seen: list[int | None] = []

    class _Pipe:
        def __call__(self, payload, *, num_speakers):
            seen.append(num_speakers)
            return SimpleNamespace(
                itertracks=lambda *, yield_label: iter(
                    [(SimpleNamespace(start=0.0, end=0.1), None, "speaker_0")]
                )
            )

    response = infer_local_pyannote_prepared_audio(
        np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        22050,
        3,
        pipeline=_loaded(_Pipe()),
    )

    assert isinstance(response.evidence, LocalPyannoteSpeakerEvidence)
    assert response.evidence.segments[0].speaker == "SPEAKER_0"
    assert seen == [3]


def test_pyannote_ai_routes_cloud_diarization(monkeypatch) -> None:
    """The high-level entrypoint should route cloud diarization explicitly."""

    captured: dict[str, object] = {}

    def fake_pyannote_ai(audio, sample_rate_hz, num_speakers):
        captured["audio"] = audio.copy()
        captured["sample_rate_hz"] = sample_rate_hz
        captured["num_speakers"] = num_speakers
        return CompletedDiarizationJob(
            job_id="job-route-test",
            output={
                "exclusiveDiarization": [
                    {"start": 0.0, "end": 0.1, "speaker": "SPEAKER_00"}
                ]
            },
        )

    monkeypatch.setattr(
        "batchalign.inference.pyannote_ai.infer_pyannote_ai", fake_pyannote_ai
    )

    response = infer_pyannote_ai_prepared_audio(
        np.asarray([0.1, 0.2], dtype=np.float32), 16000, 2
    )

    assert isinstance(response.evidence, PyannoteAISpeakerEvidence)
    assert response.evidence.job_id == "job-route-test"
    assert (
        response.evidence.output["exclusiveDiarization"][0]["speaker"] == "SPEAKER_00"
    )
    assert captured["sample_rate_hz"] == 16000
    assert captured["num_speakers"] == 2


def test_infer_pyannote_speaker_prepared_audio_shapes_waveform_and_labels() -> None:
    """Prepared-audio Pyannote inference should build mono waveforms and normalize labels."""

    captured: dict[str, object] = {}

    class _FakeResult:
        @staticmethod
        def itertracks(*, yield_label: bool):
            assert yield_label is True
            return iter(
                [
                    (SimpleNamespace(start=0.1, end=0.4), None, "speaker_2"),
                    (SimpleNamespace(start=0.5, end=0.8), None, "A"),
                ]
            )

    class _FakePipe:
        def __call__(self, payload, *, num_speakers: int):
            captured["waveform_shape"] = tuple(payload["waveform"].shape)
            captured["sample_rate"] = payload["sample_rate"]
            captured["num_speakers"] = num_speakers
            return _FakeResult()

    segments = infer_pyannote_speaker_prepared_audio(
        np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
        16000,
        3,
        pipeline=_loaded(_FakePipe()),
    )

    assert captured == {
        "waveform_shape": (1, 3),
        "sample_rate": 16000,
        "num_speakers": 3,
    }
    assert segments == [
        SpeakerSegment(start_ms=100, end_ms=400, speaker="SPEAKER_2"),
        SpeakerSegment(start_ms=500, end_ms=800, speaker="SPEAKER_A"),
    ]


def test_infer_pyannote_speaker_prepared_audio_supports_diarize_output_api() -> None:
    """Prepared-audio Pyannote inference should handle the newer DiarizeOutput shape."""

    captured: dict[str, object] = {}

    class _FakeDiarizeOutput:
        speaker_diarization = [
            (SimpleNamespace(start=0.1, end=0.4), "speaker_2"),
            (SimpleNamespace(start=0.5, end=0.8), "A"),
        ]

    class _FakePipe:
        def __call__(self, payload, *, num_speakers: int):
            captured["waveform_shape"] = tuple(payload["waveform"].shape)
            captured["sample_rate"] = payload["sample_rate"]
            captured["num_speakers"] = num_speakers
            return _FakeDiarizeOutput()

    segments = infer_pyannote_speaker_prepared_audio(
        np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
        16000,
        3,
        pipeline=_loaded(_FakePipe()),
    )

    assert captured == {
        "waveform_shape": (1, 3),
        "sample_rate": 16000,
        "num_speakers": 3,
    }
    assert segments == [
        SpeakerSegment(start_ms=100, end_ms=400, speaker="SPEAKER_2"),
        SpeakerSegment(start_ms=500, end_ms=800, speaker="SPEAKER_A"),
    ]
