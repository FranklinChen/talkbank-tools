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

from batchalign.inference.provider_retry import ProviderRefusal
from batchalign.providers import (
    BatchInferRequest,
    BatchInferResponse,
    InferResponse,
)
from batchalign.worker._types import reported_engine_name
from batchalign.worker._types_v2 import (
    TranslationBlankInputItemV2,
    TranslationItemResultV2,
    TranslationNoResponseItemV2,
    TranslationProviderStatusItemV2,
    TranslationTranslatedItemV2,
)

L = logging.getLogger("batchalign.worker")


class TranslateBatchItem(BaseModel):
    """A single item in the batch translate payload from Rust."""

    text: str


@dataclass(frozen=True, slots=True)
class LoadedTranslation:
    """The translation engine one worker loaded, as ONE value.

    ``engine`` is the identity every translated item reports, and
    ``translate`` runs the model; the loader builds both together, so an
    engine name cannot outlive the callable it describes. ``engine`` is
    admitted by ``reported_engine_name`` on construction. Request pacing is
    not the worker's: the Rust control plane spaces and retries requests by
    the engine the job selected.
    """

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
    - ``{"kind": "provider_status", ...}`` when the provider answered an
      HTTP status instead of a translation, with its ``Retry-After``
    - ``{"kind": "no_response", ...}`` when the request reached no provider
      answer at all
    - an item ``error`` when the payload is invalid or the engine raised

    Nothing here sleeps or retries: the Rust control plane decides what a
    provider answer is worth, per engine, where the wait is visible to the
    job's deadline and cancellation.
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
            results.append(_item(TranslationBlankInputItemV2()))
            continue

        try:
            # Text arrives pre-processed from Rust (Chinese space removal etc.).
            # Return raw translation output, Rust handles post-processing.
            translated = translation.translate(item.text, src_lang)
            results.append(
                _item(
                    TranslationTranslatedItemV2(
                        raw_translation=translated, engine=translation.engine
                    )
                )
            )
        except ProviderRefusal as refusal:
            L.warning("Translation refused for item: %s", refusal)
            results.append(_item(_refusal_item(refusal)))
        except Exception as e:
            L.warning("Translation failed for item: %s", e, exc_info=True)
            results.append(
                InferResponse(error=f"Translation failed: {e}", elapsed_s=0.0)
            )

    elapsed = time.monotonic() - t0
    if results:
        first = results[0]
        results[0] = InferResponse(
            result=first.result, error=first.error, elapsed_s=elapsed
        )

    # Debug, not info: this runs once per utterance now, and the worker's
    # stderr is retained by the control plane for failure dumps.
    L.debug("batch_infer translate: %d items, %.3fs", len(req.items), elapsed)
    return BatchInferResponse(results=results)


def _item(outcome: TranslationItemResultV2) -> InferResponse:
    """One item outcome, serialized through its own wire model."""
    return InferResponse(
        result=outcome.model_dump(mode="json", exclude_none=True), elapsed_s=0.0
    )


def _refusal_item(refusal: ProviderRefusal) -> TranslationItemResultV2:
    """The tagged outcome for a provider refusal."""
    if refusal.response is None:
        return TranslationNoResponseItemV2(error=str(refusal.cause))
    return TranslationProviderStatusItemV2(
        status=refusal.response.status,
        retry_after_s=refusal.response.retry_after_s,
        error=str(refusal.cause),
    )
