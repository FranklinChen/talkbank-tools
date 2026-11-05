"""Speaker diarization inference: prepared audio -> raw speaker segments.

This module is the thinnest Python seam for diarization runtimes. The ownership
split is:

- Rust chooses when speaker diarization should run and which backend to use.
- Rust prepares audio and sends it via worker protocol V2.
- Python opens the model runtime and returns raw timestamped speaker segments.
- Rust owns any higher-level document semantics or workflow policy above this
  boundary.

Production entry points, one per backend: ``infer_pyannote_ai_prepared_audio``
(hosted, loads nothing) and ``infer_local_pyannote_prepared_audio``, which
takes the pipeline the worker's speaker task load built.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias

import numpy as np
from pydantic import BaseModel, Field

from batchalign.inference._domain_types import (
    NumSpeakers,
    SpeakerId,
    TimestampMs,
)

if TYPE_CHECKING:
    from batchalign.inference.pyannote_local import LoadedPyannotePipeline

L = logging.getLogger("batchalign.worker")


def infer_pyannote_ai_prepared_audio(
    audio: np.ndarray,
    sample_rate_hz: int,
    num_speakers: NumSpeakers | None,
) -> SpeakerResponse:
    """Diarize one prepared-audio item on pyannoteAI, a hosted API.

    ``num_speakers`` of ``None`` means "not specified", which the CLI defines
    as auto-detect (ruled by the operator, 2026-08-21): pyannoteAI is sent no
    count and estimates it. A specified count forces exactly that count.
    """

    from batchalign.inference.pyannote_ai import infer_pyannote_ai

    completed = infer_pyannote_ai(audio, sample_rate_hz, num_speakers)
    if completed.warning:
        L.warning("pyannoteAI warning: %s", completed.warning)
    return SpeakerResponse(
        evidence=PyannoteAISpeakerEvidence(
            job_id=completed.job_id,
            output=completed.output,
            warning=completed.warning,
        )
    )


def infer_local_pyannote_prepared_audio(
    audio: np.ndarray,
    sample_rate_hz: int,
    num_speakers: NumSpeakers | None,
    *,
    pipeline: LoadedPyannotePipeline,
) -> SpeakerResponse:
    """Diarize one prepared-audio item on the pipeline the speaker task loaded.

    ``num_speakers`` of ``None`` lets local pyannote estimate the count.
    """

    return SpeakerResponse(
        evidence=LocalPyannoteSpeakerEvidence(
            segments=infer_pyannote_speaker_prepared_audio(
                audio, sample_rate_hz, num_speakers, pipeline=pipeline
            )
        )
    )


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SpeakerSegment(BaseModel):
    """A single speaker segment."""

    start_ms: TimestampMs
    end_ms: TimestampMs
    speaker: SpeakerId


class PyannoteAISpeakerEvidence(BaseModel):
    """Completed paid cloud job retained before local normalization."""

    kind: Literal["pyannote_ai"] = "pyannote_ai"
    job_id: str = Field(min_length=1)
    output: dict[str, Any]
    warning: str | None = None


class LocalPyannoteSpeakerEvidence(BaseModel):
    """Segments emitted by the local pyannote runtime."""

    kind: Literal["pyannote"] = "pyannote"
    segments: list[SpeakerSegment]


SpeakerEvidence: TypeAlias = Annotated[
    PyannoteAISpeakerEvidence | LocalPyannoteSpeakerEvidence,
    Field(discriminator="kind"),
]


class SpeakerResponse(BaseModel):
    """Backend-specific speaker inference evidence."""

    evidence: SpeakerEvidence


def infer_pyannote_speaker_prepared_audio(
    audio: np.ndarray,
    sample_rate_hz: int,
    num_speakers: NumSpeakers | None,
    *,
    pipeline: LoadedPyannotePipeline,
) -> list[SpeakerSegment]:
    """Run Pyannote diarization on Rust-prepared mono PCM audio.

    ``None`` is passed straight through: pyannote estimates the speaker count
    when ``num_speakers`` is absent, which is what "not specified" means.
    """
    import torch

    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
    pipe: Any = pipeline.pipeline
    result = pipe(
        {"waveform": waveform, "sample_rate": sample_rate_hz}, num_speakers=num_speakers
    )

    segments: list[SpeakerSegment] = []
    for turn, speaker in _iter_pyannote_turns_and_speakers(result):
        segments.append(
            SpeakerSegment(
                start_ms=int(turn.start * 1000),
                end_ms=int(turn.end * 1000),
                speaker=f"SPEAKER_{speaker.split('_')[-1] if '_' in speaker else speaker}",
            )
        )
    return segments


def _iter_pyannote_turns_and_speakers(result: object):
    """Yield `(turn, speaker)` pairs across supported pyannote output shapes."""

    if hasattr(result, "itertracks"):
        for turn, _, speaker in result.itertracks(yield_label=True):
            yield turn, speaker
        return

    diarization = getattr(result, "speaker_diarization", None)
    if diarization is None:
        raise AttributeError(
            "pyannote diarization result exposes neither itertracks() nor speaker_diarization"
        )

    if hasattr(diarization, "itertracks"):
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            yield turn, speaker
        return

    for turn, speaker in diarization:
        yield turn, speaker
