"""Infer and ``batch_infer`` dispatch routers.

This module is the request-time counterpart to the worker bootstrap modules.
Its job is deliberately narrow:

- accept typed inference requests from the stdio protocol layer
- route each task to the host its load built (``_state.loads``)
- refuse a task that has not loaded, or that this route does not serve

It must not grow into a pipeline-orchestration layer. CHAT parsing, job
ownership, retries, and workflow policy remain on the Rust side.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, assert_never

from batchalign.worker._infer_hosts import (
    UTSEG_IS_V2_ONLY,
    legacy_translate,
    refuse_batch,
)
from batchalign.worker._loads import COREF_LOAD, MORPHOSYNTAX_LOAD, TRANSLATE_LOAD
from batchalign.worker._types import (
    BatchInferRequest,
    BatchInferResponse,
    InferRequest,
    InferResponse,
    InferTask,
    ItemFailed,
    ItemOutcome,
    ItemProduced,
    _state,
    not_loaded_message,
    sleep_test_delay,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from batchalign.worker._types import LoadKey, TaskLoadRegistry

Host = TypeVar("Host")


def _infer(req: InferRequest) -> InferResponse:
    """Dispatch one inference item through the batched inference router.

    The single-item path is intentionally a thin adapter over `batch_infer` so
    task routing logic stays centralized in one place.
    """
    if _state.test_echo:
        # The configured delay stands in for the item's work, so it is what
        # the echo reports as that work's time.
        def _echo() -> ItemOutcome:
            sleep_test_delay()
            return ItemProduced(result=req.payload)

        return InferResponse.timed(_echo)

    batch_req = BatchInferRequest(task=req.task, lang=req.lang, items=[req.payload])
    batch_resp = _batch_infer(batch_req)
    return (
        batch_resp.results[0]
        if batch_resp.results
        else InferResponse.unexecuted(ItemFailed(error="Empty batch response"))
    )


def _batch_infer(req: BatchInferRequest) -> BatchInferResponse:
    """Dispatch one batch request to the task-specific inference adapter."""
    if _state.test_echo:
        # One delay for the whole batch: it belongs to no single item, so no
        # item reports it.
        sleep_test_delay()
        return BatchInferResponse(
            results=[
                InferResponse.unexecuted(ItemProduced(result=item))
                for item in req.items
            ]
        )

    return batch_infer_on(req, _state.loads)


def batch_infer_on(
    req: BatchInferRequest, loads: TaskLoadRegistry
) -> BatchInferResponse:
    """Answer one legacy batch from the hosts ``loads`` holds.

    The task's load is resolved once, here: a task that has not loaded is
    refused for every item, and the host it built is what answers otherwise.
    Morphosyntax, translate and coref have a legacy route; utseg is refused
    (its request cannot name a segmenter), and every other task runs only
    through worker protocol V2.
    """

    def on(
        key: LoadKey[Host], answer: Callable[[Host], BatchInferResponse]
    ) -> BatchInferResponse:
        host = loads.get(key)
        if host is None:
            return refuse_batch(req, not_loaded_message(key.load))
        return answer(host)

    match req.task:
        case InferTask.MORPHOSYNTAX:
            # The legacy route has no progress channel.
            return on(MORPHOSYNTAX_LOAD, lambda host: host.runner(req, None))
        case InferTask.TRANSLATE:
            return on(TRANSLATE_LOAD, lambda host: legacy_translate(host, req))
        case InferTask.COREF:
            return on(COREF_LOAD, lambda host: host.runner(req))
        case InferTask.UTSEG:
            return refuse_batch(req, UTSEG_IS_V2_ONLY)
        case (
            InferTask.FA
            | InferTask.ASR
            | InferTask.OPENSMILE
            | InferTask.AVQI
            | InferTask.SPEAKER
            | InferTask.WORD_SEGMENTATION
        ):
            return refuse_batch(
                req,
                f"{req.task.value} runs only through worker protocol V2",
            )
        case _:
            assert_never(req.task)
