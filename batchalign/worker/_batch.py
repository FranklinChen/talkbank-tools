"""The one loop that answers a batch-inference request item by item.

Every batch handler had the same loop: read each raw item through its model
(a payload that does not parse is that item's failure, answered without
running anything), decide for each item whether there is work to run, time
the work it runs around that item alone, and log the batch total, which is a
fact about the batch and never goes onto an item. It lives here once; a
handler says only what an item's work is.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from batchalign.worker._types import (
    BatchInferResponse,
    InferResponse,
    ItemFailed,
    ItemOutcome,
    WorkerJSONValue,
)

L = logging.getLogger("batchalign.worker")

ItemT = TypeVar("ItemT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class Unexecuted:
    """An item answered without running any work for it, so with no time:
    a document with nothing to analyze, a provider that is not loaded."""

    outcome: ItemOutcome


ItemWork = Callable[[], ItemOutcome] | Unexecuted
"""What a handler does with one item: work to run and time, or an answer
given without running anything."""


def answer_batch(
    task: str,
    raw_items: list[WorkerJSONValue],
    item_model: type[ItemT],
    work_for: Callable[[int, ItemT], ItemWork],
) -> BatchInferResponse:
    """Answer each raw item: an unparseable one fails unexecuted, any other
    gets ``work_for``'s answer, timed when it is work."""
    started_at = time.monotonic()
    results: list[InferResponse] = []
    for index, raw_item in enumerate(raw_items):
        try:
            item = item_model.model_validate(raw_item)
        except ValidationError:
            results.append(
                InferResponse.unexecuted(
                    ItemFailed(error=f"Invalid {item_model.__name__}")
                )
            )
            continue
        match work_for(index, item):
            case Unexecuted(outcome=outcome):
                results.append(InferResponse.unexecuted(outcome))
            case work:
                results.append(InferResponse.timed(work))
    L.info(
        "batch_infer %s: %d items, %.3fs",
        task,
        len(raw_items),
        time.monotonic() - started_at,
    )
    return BatchInferResponse(results=results)
