"""Stanza coreference inference: sentences -> coref chains.

Pure inference, no CHAT, no caching, no pipeline.

Each resolved item names the engine that resolved it, so the Rust server
derives coref provenance from the responses it applies rather than from a
worker-wide report taken before dispatch.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING

from pydantic import BaseModel

from batchalign.providers import (
    BatchInferRequest,
    BatchInferResponse,
    ItemFailed,
    ItemOutcome,
    ItemProduced,
)
from batchalign.worker._batch import ItemWork, Unexecuted, answer_batch
from batchalign.worker._types import reported_engine_name

if TYPE_CHECKING:
    from stanza.models.common.doc import Document

L = logging.getLogger("batchalign.worker")

_COREF_PACKAGE = "ontonotes-singletons_roberta-large-lora"
"""The Stanza coreference package this producer loads."""


def coref_engine() -> str | None:
    """The coreference model identity this worker runs, or ``None``.

    ``stanza-<version>/<package>``: the installed Stanza release plus the
    coreference package the pipeline is built with (``_COREF_PACKAGE``, the
    same constant ``batch_infer_coref`` passes to ``stanza.Pipeline``), so a
    stamp names the model that resolved the chains and not only the library.
    ``None`` when no Stanza version can be named. Admitted by
    ``reported_engine_name``, so a name that would break a stamp raises.
    Shared by the producer and the capability report, which spell it one way.
    """
    from batchalign.worker._types import _state

    stanza_engine = _state.stanza_engine()
    if stanza_engine is None:
        return None
    return reported_engine_name(f"{stanza_engine}/{_COREF_PACKAGE}")


_NO_COREF_ENGINE_ERROR = (
    "coref cannot name the Stanza version that would resolve this document; "
    "refusing to report a resolution without its engine identity"
)
"""Failure for a document whose resolution could not name its engine."""


class CorefBatchItem(BaseModel):
    """A single item: one complete document as list of sentences."""

    sentences: list[list[str]]


class ChainRef(BaseModel):
    """A single coreference chain reference on a word.

    Matches Rust ``ChainRef`` in ``batchalign/src/coref.rs``.
    """

    chain_id: int
    is_start: bool
    is_end: bool


class CorefRawAnnotation(BaseModel):
    """Structured per-sentence coref data with typed chain references.

    Each element in ``words`` is parallel to the sentence's word list.
    Empty list means the word has no coreference chains.
    """

    sentence_idx: int
    words: list[list[ChainRef]]


class CorefRawResponse(BaseModel):
    """Raw structured coref response: Rust builds bracket notation from this."""

    annotations: list[CorefRawAnnotation]


class _CompletedCorefAnalysis:
    """Bind native model coverage before producing sparse wire annotations.

    Sparse chains do not imply sparse analysis. Construction verifies every
    request sentence and word, including those with no chains, so truncation
    cannot become a successful empty response.
    """

    __slots__ = ("_annotations",)

    def __init__(self, request: CorefBatchItem, document: Document) -> None:
        if len(document.sentences) != len(request.sentences):
            raise ValueError(
                "coref sentence coverage mismatch: "
                f"expected {len(request.sentences)}, got {len(document.sentences)}"
            )
        annotations: list[CorefRawAnnotation] = []
        for sentence_idx, (expected, sentence) in enumerate(
            zip(request.sentences, document.sentences, strict=True)
        ):
            actual = [word.text for word in sentence.words]
            if actual != expected:
                raise ValueError(
                    f"coref word binding mismatch in sentence {sentence_idx}: "
                    f"expected {len(expected)} words, got {len(actual)} "
                    "or different word identities"
                )
            words = [
                [
                    ChainRef(
                        chain_id=chain.chain.index,
                        is_start=chain.is_start,
                        is_end=chain.is_end,
                    )
                    for chain in word.coref_chains
                ]
                for word in sentence.words
            ]
            if any(words):
                annotations.append(
                    CorefRawAnnotation(sentence_idx=sentence_idx, words=words)
                )
        self._annotations = tuple(annotations)

    def resolved(self, engine: str) -> ItemProduced:
        """Only a completed, source-bound analysis emits a resolved item."""
        return ItemProduced(
            result={
                "kind": "resolved",
                **CorefRawResponse(annotations=list(self._annotations)).model_dump(),
                "engine": engine,
            }
        )


def batch_infer_coref(req: BatchInferRequest) -> BatchInferResponse:
    """Batch Stanza coref inference: sentences -> structured chain annotations.

    Each item is one complete document (list of sentences, each a list of words).
    Pipeline is lazily initialized and reused across documents in the batch.

    Each item becomes one of:

    - ``{"kind": "resolved", "annotations": [...], "engine": "stanza-<version>/<package>"}``
    - ``{"kind": "no_sentences"}`` for a document with no sentences, which
      never reached the model and so names no engine
    - an item ``error`` when the payload is invalid, the engine cannot be
      named, or Stanza raised. A raise used to become an EMPTY annotation
      list, which is indistinguishable on the wire from a document that
      genuinely has no coreference chains, so the file was written without
      its ``%xcoref`` tiers and reported as a success.
    """

    import stanza

    # The identity every resolved item reports: the coreference model this
    # batch builds its pipeline with. Resolved once, because neither the
    # installed Stanza nor the package can change mid-batch.
    engine = coref_engine()

    pipeline: stanza.Pipeline | None = None

    def _resolve(item_idx: int, item: CorefBatchItem, engine: str) -> ItemOutcome:
        """Resolve one item's coreference, building the pipeline on first use.

        The pipeline build lands inside the first resolved item's time: that
        item's call is the one that waited for it.
        """
        nonlocal pipeline
        try:
            if pipeline is None:
                # Coref pipeline downloads its own model files (RoBERTa-large
                # plus the ontonotes coref package) on first use; surface the
                # wait through the progress channel.
                from batchalign.worker._progress import emit_download_event

                emit_download_event(
                    stage="downloading_stanza_coref_eng",
                    user_message=(
                        "Downloading Stanza English coreference model "
                        "(one-time, ~1.5 GB; future runs will use the local "
                        "cache)…"
                    ),
                )
                pipeline = stanza.Pipeline(
                    lang="en",
                    processors="tokenize, coref",
                    package={"coref": _COREF_PACKAGE},
                    tokenize_pretokenized=True,
                )

            text = "\n\n".join(" ".join(s) for s in item.sentences)
            result = pipeline(text)

            return _CompletedCorefAnalysis(item, result).resolved(engine)
        except Exception as e:
            L.warning("Coref infer failed for item %d: %s", item_idx, e)
            return ItemFailed(error=f"Coref failed: {e}")

    def work_for(item_idx: int, item: CorefBatchItem) -> ItemWork:
        if not item.sentences:
            return Unexecuted(ItemProduced(result={"kind": "no_sentences"}))
        if engine is None:
            return Unexecuted(ItemFailed(error=_NO_COREF_ENGINE_ERROR))
        return partial(_resolve, item_idx, item, engine)

    return answer_batch("coref", req.items, CorefBatchItem, work_for)
