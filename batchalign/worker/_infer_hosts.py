"""Bootstrap-owned runtime hosts for worker batch inference.

This module keeps request-time dispatch thin for text tasks that use the
``BatchInferRequest`` dispatch path. The worker bootstrap layer decides
which concrete engines are loaded and registers one batch-infer handler per
task. Request-time code in ``_infer.py`` then routes requests to those
prepared handlers instead of re-deriving engine policy on every call.
"""

from __future__ import annotations

from batchalign.worker._types import (
    BatchInferHandler,
    BatchInferRequest,
    BatchInferResponse,
    InferResponse,
    _state,
)


def unsupported_batch_infer(message: str) -> BatchInferHandler:
    """Build a handler that reports one consistent bootstrap/runtime error."""

    def _handler(req: BatchInferRequest) -> BatchInferResponse:
        """Return the same structured error for every batch item."""
        return BatchInferResponse(
            results=[InferResponse(error=message, elapsed_s=0.0) for _ in req.items]
        )

    return _handler


def build_morphosyntax_batch_infer_handler() -> BatchInferHandler:
    """Build the morphosyntax batch handler from loaded Stanza runtime state."""
    from batchalign.inference.morphosyntax import batch_infer_morphosyntax
    from batchalign.runtime import FREE_THREADED
    from batchalign.worker._stanza_loading import load_stanza_models

    def _handler(req: BatchInferRequest) -> BatchInferResponse:
        """Run morphosyntax batch inference, loading each language on demand."""
        cache = _state.stanza_pipelines
        if cache is None:
            return unsupported_batch_infer("No Stanza models loaded")(req)

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
            nlp_lock=_state.stanza_nlp_lock,
            free_threaded=FREE_THREADED,
            mwt_lexicon=req.mwt,
            progress_callback=_state.active_progress_callback,
            # The cache is bounded, so a batch spanning more languages than it
            # holds will find one evicted mid-batch. This is how it comes back
            # instead of the group being reported as unanalysable.
            load_pipeline=load_stanza_models,
        )

    return _handler


def build_utseg_batch_infer_handler() -> BatchInferHandler:
    """Build the utterance-segmentation batch handler from loaded builder state."""
    from batchalign.inference.utseg import batch_infer_utseg

    def _handler(req: BatchInferRequest) -> BatchInferResponse:
        """Run utterance segmentation using the configured Stanza config builder."""
        if _state.utseg_config_builder is None:
            return unsupported_batch_infer("No utseg config builder loaded")(req)
        return batch_infer_utseg(
            req=req,
            build_stanza_config=_state.utseg_config_builder,
            utterance_boundary_model=_state.utterance_boundary_model,
        )

    return _handler


def build_translate_batch_infer_handler() -> BatchInferHandler:
    """Build the translation batch handler from the loaded translation engine."""
    from batchalign.inference.translate import batch_infer_translate

    def _handler(req: BatchInferRequest) -> BatchInferResponse:
        """Run translation using the engine selected during worker bootstrap."""
        if _state.translate_fn is None or _state.translate_backend is None:
            return unsupported_batch_infer("No translation engine loaded")(req)
        return batch_infer_translate(
            req=req,
            translate_fn=_state.translate_fn,
            backend=_state.translate_backend,
        )

    return _handler


__all__ = [
    "BatchInferHandler",
    "build_morphosyntax_batch_infer_handler",
    "build_translate_batch_infer_handler",
    "build_utseg_batch_infer_handler",
    "unsupported_batch_infer",
]
