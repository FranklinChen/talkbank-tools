"""Live worker-protocol V2 executors for batched text tasks.

The control plane for these four tasks lives in Rust
(`crates/batchalign-pyo3/src/worker_text_exec.rs`), exactly as it does for the
audio and media executors: request parsing, task/payload agreement, prepared
batch loading, item-count checks, result normalization and the failure
taxonomy are all owned there. This module keeps only what genuinely needs
Python:

- one runner ADAPTER per task over the host that task's load built
  (``batchalign.worker._loads``), which parses the Rust-frozen batch into the
  typed batch items, builds the host's request, hands morphosyntax the
  request's own progress callback, and calls the model

**See also:** `../../INTERFACE_MAP.md` section "7. Text Task Result
Normalization" for the full Rust/Python responsibility split.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from batchalign.inference.coref import CorefBatchItem
from batchalign.inference.morphosyntax import MorphosyntaxBatchItem
from batchalign.inference.translate import TranslateBatchItem, TranslateInferenceRequest
from batchalign.inference.utseg import UtsegBatchItem, UtsegInferenceRequest
from batchalign.worker._types import BatchInferRequest, BatchInferResponse, InferTask
from batchalign.worker._types_v2 import ExecuteRequestV2, ExecuteResponseV2

if TYPE_CHECKING:
    from collections.abc import Callable

    from batchalign.worker._loads import (
        LoadedCoref,
        LoadedMorphosyntax,
        LoadedTranslate,
        LoadedUtseg,
        LoadedWordSegmentation,
    )


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


def _morphosyntax_adapter(
    host: LoadedMorphosyntax,
) -> Callable[[str, str, str, bool], BatchInferResponse]:
    """Adapt the loaded morphosyntax runner to the Rust control plane's call.

    Owns the two things the Rust side cannot: parsing the frozen batch into
    the typed batch items (a pydantic ``ValidationError``, which the Rust
    taxonomy reports as ``invalid_payload``), and the throttled progress
    callback this request's progress events come from, handed to the runner
    as an argument. It used to travel through a worker-state field the
    handler read, which was safe only while the sequential loop ran one
    request at a time.
    """

    def _run(
        request_id: str, lang: str, batch_json: str, retokenize: bool
    ) -> BatchInferResponse:
        from batchalign.worker._protocol import write_request_progress
        from batchalign.worker._types_v2 import UnitProgressV2

        batch = MorphosyntaxPreparedBatchV2.model_validate_json(batch_json)

        _last_progress_time = [0.0]

        def _on_progress(completed: int, total: int) -> None:
            now = time.monotonic()
            if now - _last_progress_time[0] < 1.0 and completed < total:
                return
            _last_progress_time[0] = now
            write_request_progress(
                request_id,
                "stanza_processing",
                UnitProgressV2(completed=completed, total=total),
            )

        return host.runner(
            BatchInferRequest(
                task=InferTask.MORPHOSYNTAX,
                lang=lang,
                items=[item.model_dump(mode="json") for item in batch.items],
                mwt=batch.mwt,
                retokenize=retokenize,
            ),
            _on_progress,
        )

    return _run


def _utseg_adapter(
    host: LoadedUtseg,
) -> Callable[[str, str, str], BatchInferResponse]:
    """Adapt the loaded utseg runner to the Rust control plane's call.

    ``segmenter`` is the wire spelling of the request's
    ``UtsegSegmenterV2``; an unknown spelling fails pydantic validation here
    (``ValidationError``, reported as ``invalid_payload``).
    """

    def _run(lang: str, batch_json: str, segmenter: str) -> BatchInferResponse:
        batch = UtsegPreparedBatchV2.model_validate_json(batch_json)
        return host.runner(
            UtsegInferenceRequest.model_validate(
                {
                    "lang": lang,
                    "segmenter": segmenter,
                    "items": tuple(
                        item.model_dump(mode="json") for item in batch.items
                    ),
                }
            )
        )

    return _run


def _translate_adapter(
    host: LoadedTranslate,
) -> Callable[[str, str, str], BatchInferResponse]:
    """Adapt the loaded translation runner to the Rust control plane's call."""

    def _run(source_lang: str, target_lang: str, batch_json: str) -> BatchInferResponse:
        batch = TranslatePreparedBatchV2.model_validate_json(batch_json)
        return host.runner(
            TranslateInferenceRequest(
                source_lang=source_lang,
                target_lang=target_lang,
                items=tuple(item.model_dump(mode="json") for item in batch.items),
            )
        )

    return _run


def _coref_adapter(
    host: LoadedCoref,
) -> Callable[[str, str], BatchInferResponse]:
    """Adapt the loaded coreference runner to the Rust control plane's call."""

    def _run(lang: str, batch_json: str) -> BatchInferResponse:
        batch = CorefPreparedBatchV2.model_validate_json(batch_json)
        return host.runner(
            BatchInferRequest(
                task=InferTask.COREF,
                lang=lang,
                items=[item.model_dump(mode="json") for item in batch.items],
            )
        )

    return _run


def execute_morphosyntax_request_v2(
    request: ExecuteRequestV2,
    host: LoadedMorphosyntax,
) -> ExecuteResponseV2:
    """Execute one batched morphosyntax V2 request through the Rust control plane."""

    import batchalign_core

    return ExecuteResponseV2.model_validate_json(
        batchalign_core.execute_morphosyntax_request_v2(
            request, _morphosyntax_adapter(host)
        )
    )


def execute_utseg_request_v2(
    request: ExecuteRequestV2,
    host: LoadedUtseg,
) -> ExecuteResponseV2:
    """Execute one batched utterance-segmentation V2 request through the Rust control plane."""

    import batchalign_core

    return ExecuteResponseV2.model_validate_json(
        batchalign_core.execute_utseg_request_v2(request, _utseg_adapter(host))
    )


def execute_translate_request_v2(
    request: ExecuteRequestV2,
    host: LoadedTranslate,
) -> ExecuteResponseV2:
    """Execute one batched translation V2 request through the Rust control plane."""

    import batchalign_core

    return ExecuteResponseV2.model_validate_json(
        batchalign_core.execute_translate_request_v2(request, _translate_adapter(host))
    )


def execute_coref_request_v2(
    request: ExecuteRequestV2,
    host: LoadedCoref,
) -> ExecuteResponseV2:
    """Execute one batched coreference V2 request through the Rust control plane."""

    import batchalign_core

    return ExecuteResponseV2.model_validate_json(
        batchalign_core.execute_coref_request_v2(request, _coref_adapter(host))
    )


def execute_word_segmentation_request_v2(
    request: ExecuteRequestV2,
    host: LoadedWordSegmentation,
) -> ExecuteResponseV2:
    """Execute one batched word-segmentation V2 request through the Rust control plane.

    The host's runner is the pure segmentation function: it holds no model,
    and applies the segmenter each request names.
    """

    import batchalign_core

    return ExecuteResponseV2.model_validate_json(
        batchalign_core.execute_word_segmentation_request_v2(request, host.runner)
    )


__all__ = [
    "execute_coref_request_v2",
    "execute_morphosyntax_request_v2",
    "execute_translate_request_v2",
    "execute_utseg_request_v2",
    "execute_word_segmentation_request_v2",
]
