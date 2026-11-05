"""Stanza coreference inference: sentences -> coref chains.

Pure inference, no CHAT, no caching.

Model residency is scoped to the batch that uses it. The coref task load
(``prepare_coref``, through ``ensure_task``) builds the pipeline once so its
files download inside the loading stall window, then releases it; each batch
builds it from those local files and releases it when the batch ends
(``PreparedCoref.opened``). A coref job sends one cross-file batch, so this is
one build per job, and an idle worker holds no coreference model.

Each resolved item names the engine that resolved it, so the Rust server
derives coref provenance from the responses it applies rather than from a
worker-wide report taken before dispatch.
"""

from __future__ import annotations

import contextlib
import gc
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
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


def coref_engine() -> str:
    """The coreference model identity this worker runs.

    ``stanza-<version>/<package>``: the installed Stanza release plus the
    coreference package the pipeline is built with (``_COREF_PACKAGE``, the
    same constant ``_build_pipeline`` passes to ``stanza.Pipeline``), so a
    stamp names the model that resolved the chains and not only the library.
    Admitted by ``reported_engine_name``, so a name that would break a stamp
    raises. Raises ``RuntimeError`` when no Stanza version can be named: a
    model whose results could not say what resolved them is never loaded.
    """
    from batchalign.worker._types import _state

    stanza_engine = _state.stanza_engine()
    if stanza_engine is None:
        raise RuntimeError(
            "coref cannot name the Stanza version that would resolve documents; "
            "refusing to load a coreference model whose results could not name "
            "their engine"
        )
    return reported_engine_name(f"{stanza_engine}/{_COREF_PACKAGE}")


def _build_pipeline() -> Callable[[str], Document]:
    """Build the coreference pipeline, watched by ``download_liveness``.

    RoBERTa-large comes from the Hugging Face hub and the coref package from
    Stanza's resources; both trees are watched, so a download is announced
    when bytes arrive and its growth re-arms a loading stall window.
    """
    import stanza

    from batchalign.worker._download_liveness import (
        StanzaWithTransformer,
        download_liveness,
    )
    from batchalign.worker._stanza_capabilities import (
        processors_argument,
        requirement_processors,
    )

    with download_liveness(
        StanzaWithTransformer("coref_eng"),
        what="the Stanza English coreference model (~1.5 GB)",
    ):
        pipeline: Callable[[str], Document] = stanza.Pipeline(
            lang="en",
            processors=processors_argument(requirement_processors("coreference")),
            package={"coref": _COREF_PACKAGE},
            tokenize_pretokenized=True,
        )
    return pipeline


@dataclass(frozen=True, slots=True)
class PreparedCoref:
    """Proof that the coref task load prepared this worker's coreference model.

    Built only by ``prepare_coref``: the model's files are local, and
    ``engine`` is the identity every resolved item reports, named before
    anything downloaded. It holds no pipeline. The only way to one is
    :meth:`opened`, whose block is one batch: the pipeline is built on entry
    and released (with a cycle collection, since Stanza pipelines are
    reference cycles) on exit, so no worker holds a reference to it between
    batches. Measured with ``scripts/measure_model_residency.py``, a built
    pipeline adds about 0.5 GB and resolving a document raises that to 2.5 GB
    or more (its weights are read on first use, and the peak grows with the
    document); a release after use returns about 1.6 GB. The Rust side leases
    each batch (``worker/residency.rs``).
    """

    engine: str

    @contextlib.contextmanager
    def opened(self) -> Iterator[Callable[[str], Document]]:
        """The pipeline, for one batch, built from the prepared local files."""
        pipeline = _build_pipeline()
        try:
            yield pipeline
        finally:
            del pipeline
            gc.collect()


def prepare_coref() -> PreparedCoref:
    """Prepare the coreference model: name it, then materialize its files.

    Runs at the coref task load (``ensure_task``), inside the loading stall
    window, so a slow first download never runs inside a request's fixed
    budget. The pipeline is built once to fetch exactly what the library
    reads, then released: residency belongs to the batches
    (:meth:`PreparedCoref.opened`).
    """
    engine = coref_engine()
    with PreparedCoref(engine).opened():
        pass
    return PreparedCoref(engine)


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


def batch_infer_coref(
    req: BatchInferRequest, coref: PreparedCoref
) -> BatchInferResponse:
    """Batch Stanza coref inference: sentences -> structured chain annotations.

    Each item is one complete document (list of sentences, each a list of
    words), resolved on a pipeline built for this batch from the files the
    coref task load prepared (``coref``) and released when the batch ends.
    The pipeline is opened by the first document that reaches the model, so
    a batch of invalid or empty documents builds nothing.

    Each item becomes one of:

    - ``{"kind": "resolved", "annotations": [...], "engine": "stanza-<version>/<package>"}``
    - ``{"kind": "no_sentences"}`` for a document with no sentences, which
      never reached the model and so names no engine
    - an item ``error`` when the payload is invalid or Stanza raised. A raise
      used to become an EMPTY annotation list, which is indistinguishable on
      the wire from a document that genuinely has no coreference chains, so
      the file was written without its ``%xcoref`` tiers and reported as a
      success.
    """

    with contextlib.ExitStack() as batch:
        opened: list[Callable[[str], Document]] = []

        def nlp() -> Callable[[str], Document]:
            """This batch's pipeline, opened on first use, closed with the batch."""
            if not opened:
                opened.append(batch.enter_context(coref.opened()))
                # Registered after the pipeline, so it runs FIRST on exit
                # (last in, first out): this batch's own reference is gone
                # before `opened` releases the pipeline and collects it.
                batch.callback(opened.clear)
            return opened[0]

        def _resolve(item_idx: int, item: CorefBatchItem) -> ItemOutcome:
            """Resolve one item's coreference on this batch's pipeline."""
            try:
                text = "\n\n".join(" ".join(s) for s in item.sentences)
                document = nlp()(text)
                return _CompletedCorefAnalysis(item, document).resolved(coref.engine)
            except Exception as e:
                L.warning("Coref infer failed for item %d: %s", item_idx, e)
                return ItemFailed(error=f"Coref failed: {e}")

        def work_for(item_idx: int, item: CorefBatchItem) -> ItemWork:
            if not item.sentences:
                return Unexecuted(ItemProduced(result={"kind": "no_sentences"}))
            return partial(_resolve, item_idx, item)

        return answer_batch("coref", req.items, CorefBatchItem, work_for)
