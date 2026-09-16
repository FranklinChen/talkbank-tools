"""Translation inference: text -> translated text.

Pure inference, no CHAT, no caching, no pipeline.

Each item's result names the engine that translated it, so the Rust server
derives translate provenance from the responses it applies rather than from a
worker-wide report taken before dispatch.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from batchalign.inference._domain_types import TranslationBackend
from batchalign.providers import (
    BatchInferRequest,
    BatchInferResponse,
    InferResponse,
)
from batchalign.worker._types import reported_engine_name

L = logging.getLogger("batchalign.worker")


class TranslateBatchItem(BaseModel):
    """A single item in the batch translate payload from Rust."""

    text: str


@dataclass(frozen=True, slots=True)
class LoadedTranslation:
    """The translation engine one worker loaded, as ONE value.

    ``backend`` selects request pacing, ``engine`` is the identity every
    translated item reports, and ``translate`` runs the model. The loader
    builds all three together; they used to be three separate worker-state
    fields, so an engine name could outlive the backend it described.
    ``engine`` is admitted by ``reported_engine_name`` on construction.
    """

    backend: TranslationBackend
    engine: str
    translate: Callable[[str, str], str]

    def __post_init__(self) -> None:
        reported_engine_name(self.engine)


def batch_infer_translate(
    req: BatchInferRequest,
    translation: LoadedTranslation,
) -> BatchInferResponse:
    """Batch translation inference: text -> translation.

    Parameters
    ----------
    req : BatchInferRequest
        Batch of TranslateBatchItem payloads.
    translation : LoadedTranslation
        The loaded engine: its callable, its reported identity, and the
        backend whose rate limits this loop honours.

    Each item becomes one of:

    - ``{"kind": "translated", "raw_translation": ..., "engine": ...}``
    - ``{"kind": "blank_input"}`` for whitespace-only text, which was never
      sent to the engine and so names none
    - an item ``error`` when the payload is invalid or the engine raised
    """
    t0 = time.monotonic()
    src_lang = req.lang if req.lang else "eng"

    results: list[InferResponse] = []
    for raw_item in req.items:
        try:
            item = TranslateBatchItem.model_validate(raw_item)
        except ValidationError:
            results.append(InferResponse(error="Invalid batch item", elapsed_s=0.0))
            continue

        if not item.text.strip():
            # Its own outcome rather than an empty translation: nothing was
            # translated, so there is no engine to name.
            results.append(InferResponse(result={"kind": "blank_input"}, elapsed_s=0.0))
            continue

        try:
            # Text arrives pre-processed from Rust (Chinese space removal etc.).
            # Return raw translation output, Rust handles post-processing.
            translated = translation.translate(item.text, src_lang)

            results.append(
                InferResponse(
                    result={
                        "kind": "translated",
                        "raw_translation": translated,
                        "engine": translation.engine,
                    },
                    elapsed_s=0.0,
                )
            )
        except Exception as e:
            L.warning("Translation failed for item: %s", e, exc_info=True)
            results.append(
                InferResponse(error=f"Translation failed: {e}", elapsed_s=0.0)
            )

        if translation.backend == TranslationBackend.GOOGLE:
            time.sleep(1.5)
        elif translation.backend == TranslationBackend.TENCENT:
            # Tencent TMT's standard free-tier QPS limit is 5 req/sec
            # for ``TextTranslate``; a tight loop hits
            # ``RequestLimitExceeded``. 0.2s/req caps us at ≤5 QPS.
            time.sleep(0.2)

    elapsed = time.monotonic() - t0
    if results:
        first = results[0]
        results[0] = InferResponse(
            result=first.result, error=first.error, elapsed_s=elapsed
        )

    L.info("batch_infer translate: %d items, %.3fs", len(req.items), elapsed)
    return BatchInferResponse(results=results)
