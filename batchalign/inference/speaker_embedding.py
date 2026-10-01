"""Span-level speaker embedding over one prepared recording.

This module is the model host for the ``speaker_embedding`` worker task. It
does exactly one thing: given the whole prepared mono PCM view of a recording
and a list of frame spans within it, return one fixed-width acoustic vector per
span. It decides nothing. Similarity, thresholds and verdicts belong to the
Rust control plane, which owns postprocessing.

## Which model, and why no new download

The embedding model is the one the released local diarization backend already
pins: the ``embedding`` node of the graph in ``local_pyannote_model.json``,
fetched by exact Hub commit through the same pinned-artifact loader
``pyannote_local`` uses. Nothing new is downloaded and nothing new is pinned,
so the two features cannot drift onto different acoustic models.

It is loaded here as a STANDALONE embedding model rather than as part of a
diarization pipeline. That distinction is load-bearing: constructing
``pyannote.audio``'s ``SpeakerDiarization`` pipeline also pulls a PLDA
calibration artifact from a gated repository, which is why standalone
``diarize`` needs Hugging Face credentials. Embedding alone touches only the
pinned, publicly readable embedding repository, so this task runs on a machine
with no Hugging Face account at all. The gated-repository error path is still
wired, because a private mirror or a future pin could reintroduce one.

## The measured hazard this module exists to contain

Below its own minimum input length the pinned ONNX graph does not raise. It
returns a correctly shaped, correctly typed float32 vector whose every
component is NaN. Nothing downstream can tell that from a real embedding by
inspecting its type, and a NaN similarity compares false against any
threshold, so a too-short utterance would silently become "no match" rather
than "not measurable". The host therefore reads the model's own
``min_num_samples`` and returns a distinct outcome for a span below it,
carrying the length that was too short.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, Field

if TYPE_CHECKING:  # pragma: no cover - typing only
    from batchalign.inference.pyannote_local import PinnedHuggingFaceArtifact

_EMBEDDING_MODEL: SpeakerEmbeddingModel | None = None


@dataclass(frozen=True, slots=True)
class SpeakerEmbeddingSpanRequest:
    """One requested span, in frames of the prepared recording.

    Frames rather than milliseconds: the prepared PCM view is the coordinate
    system the host actually holds, and converting from file milliseconds is
    the caller's single named transition. A host that accepted milliseconds
    would have to re-derive the frame index from a sample rate it was told
    about separately, which is the same number arriving twice.
    """

    span_id: str
    start_frame: int
    end_frame: int


class EmbeddedSpan(BaseModel):
    """A span the model measured, carrying its acoustic vector."""

    kind: Literal["embedded"] = "embedded"
    vector: list[float]


class SpanTooShortForEmbedding(BaseModel):
    """A span the model refused because it is below the model's own minimum.

    This is not an error and not an empty vector. It is the honest answer to
    "what does this span sound like", and it travels so that the control plane
    can report an utterance as unscorable rather than as unmatched.
    """

    kind: Literal["too_short"] = "too_short"
    frame_count: int = Field(ge=0)


SpeakerEmbeddingOutcome = EmbeddedSpan | SpanTooShortForEmbedding


class SpeakerEmbeddingSpanResult(BaseModel):
    """One requested span's outcome, echoing the id it was requested under.

    The id is echoed rather than the caller pairing by position, because two
    parallel sequences held together by index order are exactly the shape that
    silently misattributes one speaker's evidence to another.
    """

    span_id: str
    outcome: SpeakerEmbeddingOutcome = Field(discriminator="kind")


class SpeakerEmbeddingResponse(BaseModel):
    """Every requested span's outcome, plus the bounds the model imposed.

    ``dimension`` and ``minimum_frames`` are reported rather than assumed by
    the reader: they are properties of the loaded model file, and a caller that
    hardcoded them would keep agreeing with a model that had changed.
    """

    kind: Literal["speaker_embedding_result"] = "speaker_embedding_result"
    dimension: int = Field(gt=0)
    minimum_frames: int = Field(gt=0)
    spans: list[SpeakerEmbeddingSpanResult]


@dataclass(frozen=True, slots=True)
class SpeakerEmbeddingModel:
    """A loaded embedding model together with the bounds it imposes.

    The three numbers are READ from the loaded model, never written here. A
    constant in this file would be a second place the truth lives, and it would
    go on agreeing with a model that had moved.
    """

    inference: Any
    dimension: int
    minimum_frames: int
    sample_rate_hz: int
    artifact: PinnedHuggingFaceArtifact

    def revision(self) -> str:
        """The exact Hub commit this model's weights came from."""

        return self.artifact.revision


def load_speaker_embedding_model() -> SpeakerEmbeddingModel:
    """Load the pinned embedding model once per worker process.

    Lazily, and behind the same reclassification the diarization loader uses,
    so a credential or gated-repository failure arrives as the typed
    ``ModelAccessDeniedError`` the control plane already knows how to surface.
    """

    global _EMBEDDING_MODEL

    if _EMBEDDING_MODEL is None:
        from batchalign.inference.pyannote_local import (
            _download_pinned_artifact,
            _reclassified_access_error,
            load_local_pyannote_model_graph,
            resolve_huggingface_hub_token,
        )
        from batchalign.worker._progress import emit_hf_download_if_missing

        graph = load_local_pyannote_model_graph()
        artifact = graph.embedding
        emit_hf_download_if_missing(
            artifact.repo_id,
            kind="speaker embedding",
            artifacts=(artifact.filename,),
            revision=artifact.revision,
        )
        token = resolve_huggingface_hub_token()
        checkpoint = _download_pinned_artifact(artifact, token=token)
        from batchalign.inference.audio import (
            pyannote_import_without_torchcodec_notice,
        )

        try:
            import torch

            with pyannote_import_without_torchcodec_notice():
                from pyannote.audio.pipelines.speaker_verification import (
                    PretrainedSpeakerEmbedding,
                )
        except ImportError as exc:  # pragma: no cover - deploy-config error
            raise ImportError(
                "Speaker identification requires pyannote.audio, which is not "
                "installed.\nReinstall the standard batchalign3 package and "
                "confirm 'import pyannote.audio' works in the worker Python "
                "runtime."
            ) from exc

        try:
            inference = PretrainedSpeakerEmbedding(
                checkpoint, device=torch.device("cpu")
            )
        except Exception as error:
            raise _reclassified_access_error(error) from error

        _EMBEDDING_MODEL = SpeakerEmbeddingModel(
            inference=inference,
            dimension=int(inference.dimension),
            minimum_frames=int(inference.min_num_samples),
            sample_rate_hz=int(inference.sample_rate),
            artifact=artifact,
        )
    return _EMBEDDING_MODEL


def embed_prepared_audio_spans(
    audio: np.ndarray,
    sample_rate_hz: int,
    spans: Sequence[SpeakerEmbeddingSpanRequest],
    *,
    model: SpeakerEmbeddingModel | None = None,
) -> SpeakerEmbeddingResponse:
    """Return one outcome per requested span, in the order requested.

    ``audio`` is the whole prepared mono recording. Every span is a view into
    that one array, which is what makes the resulting vectors comparable: two
    embeddings computed from separately decoded files can differ for reasons
    that have nothing to do with who was speaking.
    """

    loaded = model if model is not None else load_speaker_embedding_model()

    if sample_rate_hz != loaded.sample_rate_hz:
        raise ValueError(
            "speaker embedding requires audio at the model's sample rate "
            f"{loaded.sample_rate_hz} Hz, and was handed {sample_rate_hz} Hz"
        )

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    frame_count = int(samples.shape[0])

    results: list[SpeakerEmbeddingSpanResult] = []
    for span in spans:
        # Refused, not clipped. `samples[start:end]` past the end of the array
        # returns a SHORTER slice rather than failing, so a span the caller got
        # wrong would be embedded from the wrong material and reported as an
        # ordinary result.
        if (
            span.start_frame < 0
            or span.end_frame < span.start_frame
            or span.end_frame > frame_count
        ):
            raise ValueError(
                f"speaker embedding span {span.span_id!r} "
                f"[{span.start_frame}, {span.end_frame}) is outside the prepared "
                f"audio, which holds {frame_count} frames"
            )

        span_frames = span.end_frame - span.start_frame
        if span_frames < loaded.minimum_frames:
            results.append(
                SpeakerEmbeddingSpanResult(
                    span_id=span.span_id,
                    outcome=SpanTooShortForEmbedding(frame_count=span_frames),
                )
            )
            continue

        window = samples[span.start_frame : span.end_frame]
        vector = _embed_one(loaded, window, span.span_id)
        results.append(
            SpeakerEmbeddingSpanResult(
                span_id=span.span_id, outcome=EmbeddedSpan(vector=vector)
            )
        )

    return SpeakerEmbeddingResponse(
        dimension=loaded.dimension,
        minimum_frames=loaded.minimum_frames,
        spans=results,
    )


def _embed_one(
    model: SpeakerEmbeddingModel, window: np.ndarray, span_id: str
) -> list[float]:
    """Embed one already length-checked window, refusing a non-finite result.

    The length check above is what should make this unreachable; the check here
    is what makes "should" verifiable. A NaN reaching the control plane would
    compare false against every threshold and read as a considered verdict.
    """

    import torch

    # The pinned model takes a `(batch, channel, sample)` torch tensor. Handing
    # it the numpy array it was sliced from fails inside the third-party
    # feature extractor with an attribute error, which is a worse diagnostic
    # than the conversion is a cost.
    batch = torch.from_numpy(np.ascontiguousarray(window)).reshape(1, 1, -1)
    raw = np.asarray(model.inference(batch), dtype=np.float64).reshape(-1)

    if raw.shape[0] != model.dimension:
        raise ValueError(
            f"speaker embedding for span {span_id!r} has {raw.shape[0]} "
            f"components, and the model declares {model.dimension}"
        )
    values = [float(component) for component in raw]
    if not all(math.isfinite(component) for component in values):
        raise ValueError(
            f"speaker embedding for span {span_id!r} is not a finite vector, "
            "which the pinned model returns for input it cannot measure"
        )
    return values


__all__ = [
    "EmbeddedSpan",
    "SpanTooShortForEmbedding",
    "SpeakerEmbeddingModel",
    "SpeakerEmbeddingOutcome",
    "SpeakerEmbeddingResponse",
    "SpeakerEmbeddingSpanRequest",
    "SpeakerEmbeddingSpanResult",
    "embed_prepared_audio_spans",
    "load_speaker_embedding_model",
]
