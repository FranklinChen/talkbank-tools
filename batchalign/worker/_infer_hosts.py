"""The runners the text-task loads build, and the legacy batch route over them.

Each text task's load (``_model_loading.bootstrap``) builds its runner here
over what it loaded and records the host (``batchalign.worker._loads``). Both
request routes read that one host: worker protocol V2 (``_text_v2``) and the
frozen legacy ``batch_infer`` route (``_infer``), so neither re-derives engine
policy per call and neither can see a model the other cannot.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from batchalign.inference.translate import TranslateInferenceRequest
from batchalign.worker._types import (
    BatchInferRequest,
    BatchInferResponse,
    InferResponse,
    ItemFailed,
)

if TYPE_CHECKING:
    from batchalign.inference.coref import PreparedCoref
    from batchalign.inference.translate import LoadedTranslation
    from batchalign.inference.utseg import BoundaryModel, UtsegInferenceRequest
    from batchalign.worker._loads import LoadedTranslate, ProgressCallback
    from batchalign.worker._pipeline_cache import StanzaPipelineCache


def refuse_batch(req: BatchInferRequest, message: str) -> BatchInferResponse:
    """Answer every item of a batch with the same failure, unexecuted.

    No work ran for any item, so none has an elapsed time.
    """
    return BatchInferResponse(
        results=[InferResponse.unexecuted(ItemFailed(error=message)) for _ in req.items]
    )


def build_morphosyntax_runner(
    cache: StanzaPipelineCache,
) -> Callable[[BatchInferRequest, ProgressCallback | None], BatchInferResponse]:
    """Build the morphosyntax runner over the worker's Stanza cache.

    The progress callback is the caller's argument: a V2 request passes the
    one that writes its progress events, the legacy route passes ``None``.
    """
    from batchalign.inference.morphosyntax import batch_infer_morphosyntax
    from batchalign.worker._stanza_loading import load_stanza_models

    def _run(
        req: BatchInferRequest, progress: ProgressCallback | None
    ) -> BatchInferResponse:
        """Run morphosyntax batch inference, loading each language on demand."""

        # There is deliberately no preload loop here. This used to walk every
        # language mentioned anywhere in the request and load them all before
        # inference started, which THRASHES against a bounded cache: with a
        # capacity of two and a three-language request, the third load evicted
        # the first, and the first language group then had to reload it
        # immediately. Loading three pipelines to keep two, twice over.
        # `batch_infer_morphosyntax` loads per language GROUP, at the point of
        # use, through `load_pipeline` below, so nothing is loaded before it is
        # needed and nothing is evicted before it is used.
        return batch_infer_morphosyntax(
            req=req,
            pipelines=cache,
            mwt_lexicon=req.mwt,
            progress_callback=progress,
            # The cache is bounded, so a batch spanning more languages than it
            # holds will find one evicted mid-batch. This is how it comes back
            # instead of the group being reported as unanalysable.
            load_pipeline=load_stanza_models,
        )

    return _run


def build_utseg_runner(
    boundary_model: BoundaryModel,
) -> Callable[[UtsegInferenceRequest], BatchInferResponse]:
    """Build the utterance-segmentation runner over what its load built.

    The request names its segmenter; the runner supplies what each one reads:
    what the utseg load built for the boundary model (the pinned model, or the
    fact that this worker's language has none, which the boundary segmenter
    refuses by name), and the Stanza pipeline loader (which builds the
    language's constituency pipeline on first use).
    """
    from batchalign.inference.utseg import batch_infer_utseg
    from batchalign.worker._stanza_loading import load_utseg_pipeline

    def _run(req: UtsegInferenceRequest) -> BatchInferResponse:
        return batch_infer_utseg(
            req=req,
            load_stanza_pipeline=load_utseg_pipeline,
            boundary_model=boundary_model,
        )

    return _run


def build_translate_runner(
    translation: LoadedTranslation,
) -> Callable[[TranslateInferenceRequest], BatchInferResponse]:
    """Build the translation runner over the engine its load built."""
    from batchalign.inference.translate import batch_infer_translate

    def _run(req: TranslateInferenceRequest) -> BatchInferResponse:
        return batch_infer_translate(req=req, translation=translation)

    return _run


def build_coref_runner(
    coref: PreparedCoref,
) -> Callable[[BatchInferRequest], BatchInferResponse]:
    """Build the coreference runner over the model the task prepared.

    Takes the prepared handle, which holds no pipeline: each batch builds one
    and releases it (``PreparedCoref.opened``).
    """
    from batchalign.inference.coref import batch_infer_coref

    def _run(req: BatchInferRequest) -> BatchInferResponse:
        return batch_infer_coref(req, coref)

    return _run


def legacy_translate(
    host: LoadedTranslate, req: BatchInferRequest
) -> BatchInferResponse:
    """The legacy batch route's translation, which explicitly requests English.

    V2 carries its own target. The legacy request is admitted into the same
    translation-only request, so the one runner serves both routes.
    """
    from pydantic import ValidationError

    try:
        request = TranslateInferenceRequest(
            source_lang=req.lang, target_lang="eng", items=tuple(req.items)
        )
    except ValidationError as error:
        return refuse_batch(req, f"Invalid legacy translation direction: {error}")
    return host.runner(request)


UTSEG_IS_V2_ONLY = (
    "utseg runs only through worker protocol V2, whose request names its segmenter"
)
"""The legacy route's utseg refusal.

A legacy ``BatchInferRequest`` cannot name a segmenter, and choosing one here
(from whatever this worker loaded) is the second decision the V2 request
exists to remove.
"""


__all__ = [
    "UTSEG_IS_V2_ONLY",
    "build_coref_runner",
    "build_morphosyntax_runner",
    "build_translate_runner",
    "build_utseg_runner",
    "legacy_translate",
    "refuse_batch",
]
