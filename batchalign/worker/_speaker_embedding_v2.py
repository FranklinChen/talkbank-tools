"""Live worker-protocol V2 speaker-embedding executor.

The Rust control plane owns the request, the prepared audio and everything
downstream of the vectors: similarity, thresholds and verdicts. This module is
the thin model host, exactly as `_speaker_v2` is for diarization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from batchalign.inference.speaker_embedding import (
    SpeakerEmbeddingModel,
    SpeakerEmbeddingResponse,
    SpeakerEmbeddingSpanRequest,
    embed_prepared_audio_spans,
)
from batchalign.worker._types_v2 import (
    ExecuteRequestV2,
    ExecuteResponseV2,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from batchalign.worker._loads import LoadedSpeakerEmbedding


_SPAN_REQUESTS: TypeAdapter[list[SpeakerEmbeddingSpanRequest]] = TypeAdapter(
    list[SpeakerEmbeddingSpanRequest]
)
"""Built once: a runner parses every request's spans through it."""


def span_runner(
    model: SpeakerEmbeddingModel,
) -> Callable[[np.ndarray, int, str], SpeakerEmbeddingResponse]:
    """The embedding runner over a loaded model, built by the speaker load."""

    def _run_spans(
        audio: np.ndarray,
        sample_rate_hz: int,
        spans_json: str,
    ) -> SpeakerEmbeddingResponse:
        """Validate the Rust span request JSON at the model-host boundary."""

        return embed_prepared_audio_spans(
            audio,
            sample_rate_hz,
            _SPAN_REQUESTS.validate_json(spans_json),
            model=model,
        )

    return _run_spans


def execute_speaker_embedding_request_v2(
    request: ExecuteRequestV2,
    host: LoadedSpeakerEmbedding,
) -> ExecuteResponseV2:
    """Execute one live V2 embedding request on the loaded embedding host."""

    import batchalign_core

    return ExecuteResponseV2.model_validate_json(
        batchalign_core.execute_speaker_embedding_request_v2(request, host)
    )


__all__ = [
    "execute_speaker_embedding_request_v2",
    "span_runner",
]
