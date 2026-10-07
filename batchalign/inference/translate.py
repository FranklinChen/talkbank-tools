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
from functools import partial
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

from batchalign.inference.provider_retry import ProviderRefusal
from batchalign.providers import (
    BatchInferResponse,
    InferResponse,
    ItemFailed,
    ItemOutcome,
    ItemProduced,
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


class TranslateInferenceRequest(BaseModel):
    """Translation-only host request: both checked languages are mandatory.

    Generic NLP requests have no target; accepting one here used to discard
    the wire target and allow an implicit English source. Malformed items
    remain item-local failures, independently of route admission.
    """

    model_config = ConfigDict(frozen=True)
    source_lang: Annotated[
        str,
        StringConstraints(
            pattern=r"^[a-z]{3}$", min_length=3, max_length=3, strict=True
        ),
    ]
    target_lang: Annotated[
        str,
        StringConstraints(
            pattern=r"^[a-z]{3}$", min_length=3, max_length=3, strict=True
        ),
    ]
    items: tuple[object, ...]


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
    translate: Callable[[str, str, str], str]

    def __post_init__(self) -> None:
        reported_engine_name(self.engine)


def batch_infer_translate(
    req: TranslateInferenceRequest,
    translation: LoadedTranslation,
) -> BatchInferResponse:
    """Batch translation inference: text -> translation.

    Parameters
    ----------
    req : TranslateInferenceRequest
        Checked source and target plus TranslateBatchItem payloads.
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

    results: list[InferResponse] = []
    for raw_item in req.items:
        try:
            item = TranslateBatchItem.model_validate(raw_item)
        except ValidationError:
            results.append(
                InferResponse.unexecuted(ItemFailed(error="Invalid batch item"))
            )
            continue

        if not item.text.strip():
            # Its own outcome rather than an empty translation: nothing was
            # translated, so there is no engine to name and no time to report.
            results.append(
                InferResponse.unexecuted(_produced(TranslationBlankInputItemV2()))
            )
            continue

        # Each item reports the time of its own translation call only.
        results.append(
            InferResponse.timed(partial(_translate_one, item, req, translation))
        )

    # The batch total is a fact about the batch, so it goes to the log, never
    # onto an item.
    elapsed = time.monotonic() - t0
    # Debug, not info: this runs once per utterance now, and the worker's
    # stderr is retained by the control plane for failure dumps.
    L.debug("batch_infer translate: %d items, %.3fs", len(req.items), elapsed)
    return BatchInferResponse(results=results)


def _translate_one(
    item: TranslateBatchItem,
    req: TranslateInferenceRequest,
    translation: LoadedTranslation,
) -> ItemOutcome:
    """Translate one item, folding every engine answer into its outcome."""
    try:
        # Text arrives pre-processed from Rust (Chinese space removal etc.).
        # Return raw translation output, Rust handles post-processing.
        translated = translation.translate(item.text, req.source_lang, req.target_lang)
        return _produced(
            TranslationTranslatedItemV2(
                raw_translation=translated, engine=translation.engine
            )
        )
    except ProviderRefusal as refusal:
        L.warning("Translation refused for item: %s", refusal)
        return _produced(_refusal_item(refusal))
    except Exception as e:
        L.warning("Translation failed for item: %s", e, exc_info=True)
        return ItemFailed(error=f"Translation failed: {e}")


def _produced(outcome: TranslationItemResultV2) -> ItemProduced:
    """One item outcome, serialized through its own wire model."""
    return ItemProduced(result=outcome.model_dump(mode="json", exclude_none=True))


def _refusal_item(refusal: ProviderRefusal) -> TranslationItemResultV2:
    """The tagged outcome for a provider refusal."""
    if refusal.response is None:
        return TranslationNoResponseItemV2(error=str(refusal.cause))
    return TranslationProviderStatusItemV2(
        status=refusal.response.status,
        retry_after_s=refusal.response.retry_after_s,
        error=str(refusal.cause),
    )
