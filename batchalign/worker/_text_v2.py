"""Live worker-protocol V2 executors for batched text tasks.

The control plane for these four tasks lives in Rust
(`crates/batchalign-pyo3/src/worker_text_exec.rs`), exactly as it does for the
audio and media executors: request parsing, task/payload agreement, prepared
batch loading, item-count checks, result normalization and the failure
taxonomy are all owned there. This module keeps only what genuinely needs
Python:

- the injected host of loaded model runners
- one runner ADAPTER per task, which parses the Rust-frozen batch into the
  typed batch items, builds the host's ``BatchInferRequest``, manages the
  progress callback where the host reads one, and calls the model

**See also:** `../../INTERFACE_MAP.md` section "7. Text Task Result
Normalization" for the full Rust/Python responsibility split.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from batchalign.inference.coref import (
    CorefBatchItem,
    batch_infer_coref,
)
from batchalign.inference.morphosyntax import MorphosyntaxBatchItem
from batchalign.inference.translate import TranslateBatchItem, TranslateInferenceRequest
from batchalign.inference.utseg import UtsegBatchItem
from batchalign.worker._infer_hosts import (
    build_morphosyntax_batch_infer_handler,
    build_translate_inference_handler,
    build_utseg_batch_infer_handler,
)
from batchalign.worker._types import BatchInferRequest, BatchInferResponse, InferTask
from batchalign.worker._types_v2 import (
    ExecuteRequestV2,
    ExecuteResponseV2,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class MorphosyntaxPreparedBatchV2(BaseModel):
    """Prepared morphosyntax batch payload frozen by Rust."""

    items: list[MorphosyntaxBatchItem]
    mwt: dict[str, list[str]] = Field(default_factory=dict)


class UtsegPreparedBatchV2(BaseModel):
    """Prepared utterance-segmentation batch payload frozen by Rust."""

    items: list[UtsegBatchItem]


class TranslatePreparedBatchV2(BaseModel):
    """Prepared translation batch payload frozen by Rust."""

    items: list[TranslateBatchItem]


class CorefPreparedBatchV2(BaseModel):
    """Prepared coreference batch payload frozen by Rust."""

    items: list[CorefBatchItem]


@dataclass(frozen=True, slots=True)
class TextExecutionHostV2:
    """Injected text-task execution hooks for the live V2 path."""

    morphosyntax_runner: Callable[[BatchInferRequest], BatchInferResponse] | None = None
    utseg_runner: Callable[[BatchInferRequest], BatchInferResponse] | None = None
    translate_runner: (
        Callable[[TranslateInferenceRequest], BatchInferResponse] | None
    ) = None
    coref_runner: Callable[[BatchInferRequest], BatchInferResponse] | None = None


def build_default_text_execution_host_v2() -> TextExecutionHostV2:
    """Build the live text-task V2 host from already loaded worker state."""

    return TextExecutionHostV2(
        morphosyntax_runner=build_morphosyntax_batch_infer_handler(),
        utseg_runner=build_utseg_batch_infer_handler(),
        translate_runner=build_translate_inference_handler(),
        coref_runner=batch_infer_coref,
    )


def _morphosyntax_adapter(
    runner: Callable[[BatchInferRequest], BatchInferResponse],
) -> Callable[[str, str, str, bool], BatchInferResponse]:
    """Adapt one loaded morphosyntax runner to the Rust control plane's call.

    Owns the two things the Rust side cannot: parsing the frozen batch into
    the typed batch items (pydantic ``ValidationError`` subclasses
    ``ValueError``, which the Rust taxonomy reports as ``invalid_payload``,
    preserving the old ladder), and the throttled progress callback the
    morphosyntax handler reads from worker state (thread-local-safe because
    the sequential stdio loop is single-threaded).
    """

    def _run(
        request_id: str, lang: str, batch_json: str, retokenize: bool
    ) -> BatchInferResponse:
        from batchalign.worker._protocol import write_progress_event
        from batchalign.worker._types import _state

        batch = MorphosyntaxPreparedBatchV2.model_validate_json(batch_json)

        _last_progress_time = [0.0]

        def _on_progress(completed: int, total: int) -> None:
            now = time.monotonic()
            if now - _last_progress_time[0] < 1.0 and completed < total:
                return
            _last_progress_time[0] = now
            write_progress_event(request_id, completed, total)

        _state.active_progress_callback = _on_progress
        try:
            return runner(
                BatchInferRequest(
                    task=InferTask.MORPHOSYNTAX,
                    lang=lang,
                    items=[item.model_dump(mode="json") for item in batch.items],
                    mwt=batch.mwt,
                    retokenize=retokenize,
                )
            )
        finally:
            _state.active_progress_callback = None

    return _run


def _utseg_adapter(
    runner: Callable[[BatchInferRequest], BatchInferResponse],
) -> Callable[[str, str, bool], BatchInferResponse]:
    """Adapt one loaded utseg runner to the Rust control plane's call."""

    def _run(
        lang: str, batch_json: str, allow_stanza_fallback: bool
    ) -> BatchInferResponse:
        batch = UtsegPreparedBatchV2.model_validate_json(batch_json)
        return runner(
            BatchInferRequest(
                task=InferTask.UTSEG,
                lang=lang,
                items=[item.model_dump(mode="json") for item in batch.items],
                allow_stanza_fallback=allow_stanza_fallback,
            )
        )

    return _run


def _translate_adapter(
    runner: Callable[[TranslateInferenceRequest], BatchInferResponse],
) -> Callable[[str, str, str], BatchInferResponse]:
    """Adapt one loaded translation runner to the Rust control plane's call."""

    def _run(source_lang: str, target_lang: str, batch_json: str) -> BatchInferResponse:
        batch = TranslatePreparedBatchV2.model_validate_json(batch_json)
        return runner(
            TranslateInferenceRequest(
                source_lang=source_lang,
                target_lang=target_lang,
                items=tuple(item.model_dump(mode="json") for item in batch.items),
            )
        )

    return _run


def _coref_adapter(
    runner: Callable[[BatchInferRequest], BatchInferResponse],
) -> Callable[[str, str], BatchInferResponse]:
    """Adapt one loaded coreference runner to the Rust control plane's call."""

    def _run(lang: str, batch_json: str) -> BatchInferResponse:
        batch = CorefPreparedBatchV2.model_validate_json(batch_json)
        return runner(
            BatchInferRequest(
                task=InferTask.COREF,
                lang=lang,
                items=[item.model_dump(mode="json") for item in batch.items],
            )
        )

    return _run


def execute_morphosyntax_request_v2(
    request: ExecuteRequestV2,
    host: TextExecutionHostV2,
) -> ExecuteResponseV2:
    """Execute one batched morphosyntax V2 request through the Rust control plane."""

    import batchalign_core

    runner = host.morphosyntax_runner
    return ExecuteResponseV2.model_validate_json(
        batchalign_core.execute_morphosyntax_request_v2(
            request,
            _morphosyntax_adapter(runner) if runner is not None else None,
        )
    )


def execute_utseg_request_v2(
    request: ExecuteRequestV2,
    host: TextExecutionHostV2,
) -> ExecuteResponseV2:
    """Execute one batched utterance-segmentation V2 request through the Rust control plane."""

    import batchalign_core

    runner = host.utseg_runner
    return ExecuteResponseV2.model_validate_json(
        batchalign_core.execute_utseg_request_v2(
            request,
            _utseg_adapter(runner) if runner is not None else None,
        )
    )


def execute_translate_request_v2(
    request: ExecuteRequestV2,
    host: TextExecutionHostV2,
) -> ExecuteResponseV2:
    """Execute one batched translation V2 request through the Rust control plane."""

    import batchalign_core

    runner = host.translate_runner
    return ExecuteResponseV2.model_validate_json(
        batchalign_core.execute_translate_request_v2(
            request,
            _translate_adapter(runner) if runner is not None else None,
        )
    )


def execute_coref_request_v2(
    request: ExecuteRequestV2,
    host: TextExecutionHostV2,
) -> ExecuteResponseV2:
    """Execute one batched coreference V2 request through the Rust control plane."""

    import batchalign_core

    runner = host.coref_runner
    return ExecuteResponseV2.model_validate_json(
        batchalign_core.execute_coref_request_v2(
            request,
            _coref_adapter(runner) if runner is not None else None,
        )
    )


__all__ = [
    "TextExecutionHostV2",
    "build_default_text_execution_host_v2",
    "execute_coref_request_v2",
    "execute_morphosyntax_request_v2",
    "execute_translate_request_v2",
    "execute_utseg_request_v2",
]
