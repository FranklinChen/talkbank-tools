"""Stanza constituency inference: words -> utterance boundary assignments.

Pure inference, no CHAT, no caching, no pipeline.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Annotated, assert_never, cast

from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

if TYPE_CHECKING:
    from batchalign.inference.types import ConstituencyTree, StanzaNLP
    from batchalign.models.utterance.infer import BertUtteranceModel
    from batchalign.worker._pipeline_cache import LoadedPipeline

from batchalign.inference._domain_types import LanguageCode
from batchalign.models.utterance.evidence import (
    ClassifiedBoundaryEvidence,
    ModelShortCircuit,
    NormalizationOmission,
    UtteranceBoundaryPrediction,
)
from batchalign.providers import (
    BatchInferResponse,
    InferResponse,
    ItemFailed,
    ItemOutcome,
    ItemProduced,
)
from batchalign.worker._model_errors import ModelNotLoadedError
from batchalign.worker._types import WorkerJSONValue
from batchalign.worker._types_v2 import (
    UtsegBoundaryModelEvidenceV2,
    UtsegBoundaryModelItemV2,
    UtsegClassifiedBoundaryEvidenceV2,
    UtsegConstituencyItemV2,
    UtsegModelShortCircuitV2,
    UtsegNormalizationOmissionV2,
    UtsegSegmenterV2,
    UtsegUnattributedItemV2,
    UtsegWordBoundaryEvidenceV2,
)

L = logging.getLogger("batchalign.worker")


def _serialize_boundary_prediction(
    prediction: UtteranceBoundaryPrediction,
) -> dict[str, WorkerJSONValue]:
    """Lower one typed model prediction into the canonical worker shape."""

    wire_evidence: list[UtsegWordBoundaryEvidenceV2] = []
    for item in prediction.word_evidence:
        if isinstance(item, ClassifiedBoundaryEvidence):
            wire_evidence.append(
                UtsegClassifiedBoundaryEvidenceV2(
                    raw_action=item.raw_action.value,
                    applied_action=item.applied_action.value,
                    boundary_probability_micros=item.boundary_probability.micros,
                )
            )
        elif isinstance(item, NormalizationOmission):
            wire_evidence.append(UtsegNormalizationOmissionV2())
        elif isinstance(item, ModelShortCircuit):
            wire_evidence.append(UtsegModelShortCircuitV2())
        else:
            assert_never(item)

    result = UtsegBoundaryModelItemV2(
        assignments=list(prediction.assignments),
        boundary_model_evidence=UtsegBoundaryModelEvidenceV2(
            model_id=prediction.model_id,
            model_revision=prediction.model_revision,
            normalization_revision=prediction.normalization_revision,
            adjacency_policy_revision=prediction.adjacency_policy_revision,
            word_evidence=wire_evidence,
        ),
    )
    return cast(dict[str, WorkerJSONValue], result.model_dump(mode="json"))


class UtsegModelNotFoundError(ModelNotLoadedError):
    """Raised when a request names the boundary-model segmenter and this
    worker has no utterance-boundary model LOADED.

    This is a worker-local load fact, not a statement about which languages
    have a model, and the worker boundary reports it as ``model_unavailable``
    (it is a :class:`ModelNotLoadedError`). Which segmenter a language takes
    is decided in Rust before anything is dispatched
    (``crates/batchalign/src/utseg_route.rs``) and named on the request, and
    the boundary model's pin is a function of the resolved language, injected
    when a worker for that language is spawned (``spawn_engine_overrides``).
    A worker started for ``auto`` or ``per-file`` has no language and so no
    pin, and loads no boundary model; Rust dispatches a utseg request on the
    route's resolved language, so a boundary-model request reaches a worker
    that loaded the pin. Reaching here means that load did not happen.

    It is kept as a backstop rather than as the decision: surface the gap
    rather than silently substitute one model for another.
    """


# Stage identifier for the opt-in fallback notice. Stage names form a
# closed protocol vocabulary consumed by the dashboard / CLI for
# filtering and dedupe; the language goes in user_message, not the stage.
_STAGE_UTSEG_FALLBACK_OPT_IN = "utseg_unsupported_language_fallback"

# Per-process dedupe: warn once per requested language. Worker processes
# don't outlive a deploy, so the set never needs eviction.
_FALLBACK_NOTICE_FIRED: set[str] = set()


def _emit_stanza_fallback_notice(requested_lang: str) -> None:
    """Surface the BERT-absent → Stanza substitution to the user.

    Only fires for a request naming the Stanza constituency segmenter,
    which Rust names only when the operator opted in via
    ``--utseg-fallback-stanza`` for a language with no boundary model.
    """
    # Avoid a circular import at module load time, the progress
    # protocol pulls in worker config that imports this module
    # transitively in some test setups.
    from batchalign.worker._progress import emit_download_event

    if requested_lang in _FALLBACK_NOTICE_FIRED:
        return
    _FALLBACK_NOTICE_FIRED.add(requested_lang)

    user_message = (
        f"No TalkBank utseg model for language '{requested_lang}'; "
        f"using Stanza constituency parsing "
        f"because --utseg-fallback-stanza was passed. Quality will vary."
    )

    L.warning("utseg opt-in fallback: lang=%r → Stanza", requested_lang)

    emit_download_event(
        stage=_STAGE_UTSEG_FALLBACK_OPT_IN,
        user_message=user_message,
    )


class UtsegBatchItem(BaseModel):
    """A single item in the batch utseg payload from Rust.

    Matches Rust ``UtsegBatchItem`` in ``batchalign/src/utseg.rs``.
    """

    words: list[str]
    # Required, matching the Rust schema: an empty default would let an item
    # that lost its text segment silently rather than reporting anything.
    text: str


# ---------------------------------------------------------------------------
# Per-item work, and the timing that belongs to it
# ---------------------------------------------------------------------------
#
# A duration is only ever produced by measuring one item's own call
# (`InferResponse.timed`), and it travels with the index that earned it, so a
# batch total has no route onto an item.


@dataclass(frozen=True, slots=True)
class _Answered:
    """One batch position's own response, at the position that earned it."""

    index: int
    response: InferResponse


@dataclass(frozen=True, slots=True)
class _RejectedItem:
    """One batch position whose payload never parsed into a request item.

    No work is attributable to it, and this variant is what says so: its
    elapsed time is absent (``null`` on the wire), never a zero that reads
    like a measurement.
    """

    index: int
    error: str

    @property
    def response(self) -> InferResponse:
        """Report the rejection at its own position."""
        return InferResponse.unexecuted(ItemFailed(error=self.error))


@dataclass(frozen=True, slots=True)
class _PendingItem:
    """One validated batch item bound to the position it arrived at.

    `_admit_batch` is the only construction site, and it is the only place in
    this module that reads a position off the request. Work downstream carries
    the index along with the item instead of restating it, so a result has no
    route to another item's position.
    """

    index: int
    item: UtsegBatchItem

    def measure(self, work: Callable[[UtsegBatchItem], ItemOutcome]) -> _Answered:
        """Run this item's own work and attribute exactly that item's time."""
        return _Answered(self.index, InferResponse.timed(lambda: work(self.item)))


def _admit_batch(
    raw_items: list[WorkerJSONValue],
) -> tuple[list[_PendingItem], list[_RejectedItem]]:
    """Parse every raw payload once, binding each to the position it came from."""
    pending: list[_PendingItem] = []
    rejected: list[_RejectedItem] = []
    for index, raw_item in enumerate(raw_items):
        try:
            admitted = UtsegBatchItem.model_validate(raw_item)
        except ValidationError:
            rejected.append(_RejectedItem(index=index, error="Invalid batch item"))
            continue
        pending.append(_PendingItem(index=index, item=admitted))
    return pending, rejected


def _assemble(
    total: int,
    timed: list[_Answered],
    rejected: list[_RejectedItem],
) -> BatchInferResponse:
    """Order every position's own response, refusing a batch with a hole.

    The previous shape pre-filled the result list with an empty-trees default
    and overwrote it by index, so a position that no branch reached would have
    been reported as a successful empty segmentation. Here an unreached or
    twice-written position raises instead of returning a plausible result.
    """
    by_index: dict[int, InferResponse] = {}
    for item in timed:
        if item.index in by_index:
            raise RuntimeError(f"utseg batch wrote position {item.index} twice")
        by_index[item.index] = item.response
    for reject in rejected:
        if reject.index in by_index:
            raise RuntimeError(f"utseg batch wrote position {reject.index} twice")
        by_index[reject.index] = reject.response

    missing = [index for index in range(total) if index not in by_index]
    if missing:
        raise RuntimeError(
            f"utseg batch produced no result for positions {missing} of {total}"
        )
    return BatchInferResponse(results=[by_index[index] for index in range(total)])


def _single_word_outcome(item: UtsegBatchItem) -> ItemOutcome:
    """Segment an item that needs no segmenter: one word is one utterance."""
    return ItemProduced(
        result=UtsegUnattributedItemV2(assignments=[0] * len(item.words)).model_dump(
            mode="json"
        )
    )


def _boundary_outcome(
    model: BertUtteranceModel, index: int, item: UtsegBatchItem
) -> ItemOutcome:
    """Predict one item's boundary evidence, reporting its own failure."""
    try:
        prediction = model.predict_boundary_evidence(item.words)
    except (IndexError, AttributeError, TypeError, ValueError) as error:
        L.warning("Utseg boundary-model infer failed for item %d: %s", index, error)
        return ItemFailed(error=f"Utseg boundary-model inference failed: {error}")
    return ItemProduced(result=_serialize_boundary_prediction(prediction))


def _constituency_outcome(
    nlp: StanzaNLP, index: int, item: UtsegBatchItem
) -> ItemOutcome:
    """Run Stanza for one item and return its raw constituency tree strings.

    Rust handles tree parsing and assignment computation.
    """
    try:
        doc = nlp(" ".join(item.words))
        trees: list[str] = []
        for sent in doc.sentences:
            if sent.constituency is not None:
                trees.append(str(sent.constituency))
    except (IndexError, AttributeError, TypeError) as error:
        # A failure, at this item's position: an empty parse would read as
        # a successful segmentation into one utterance.
        L.warning("Utseg infer failed for item %d: %s", index, error)
        return ItemFailed(error=f"Utseg constituency parse failed: {error}")
    if not trees:
        # No sentence carried a constituency tree: no parse to segment by.
        L.warning("Utseg infer produced no constituency tree for item %d", index)
        return ItemFailed(error="Utseg constituency parse produced no tree")
    return ItemProduced(
        result=UtsegConstituencyItemV2(trees=trees).model_dump(mode="json")
    )


class UtsegInferenceRequest(BaseModel):
    """One utseg batch as the Rust control plane admitted it.

    The segmenter is the request's own, never derived here: Rust resolves the
    language's route (its boundary model, or the operator-authorized Stanza
    fallback) and admits it against this worker's Stanza catalog before
    dispatch. The worker used to re-derive the route from whether it happened
    to have a boundary model loaded plus an ``allow_stanza_fallback`` flag,
    which was a second decision that could disagree with the first.
    """

    model_config = ConfigDict(frozen=True)
    # A real ISO 639-3 code: a request without one has no pipeline to name
    # (it used to be segmented with an English one).
    lang: Annotated[
        LanguageCode,
        StringConstraints(
            pattern=r"^[a-z]{3}$", min_length=3, max_length=3, strict=True
        ),
    ]
    segmenter: UtsegSegmenterV2
    items: tuple[WorkerJSONValue, ...]


def batch_infer_utseg(
    req: UtsegInferenceRequest,
    load_stanza_pipeline: Callable[[LanguageCode], LoadedPipeline],
    utterance_boundary_model: BertUtteranceModel | None = None,
) -> BatchInferResponse:
    """Batch utterance segmentation with the segmenter the request names: a
    boundary model's evidence, or Stanza constituency trees.

    Parameters
    ----------
    req : UtsegInferenceRequest
        Batch of UtsegBatchItem payloads, with the request's language and the
        one segmenter Rust admitted for it.
    load_stanza_pipeline : callable
        The worker's ``load_utseg_pipeline``: the language's constituency
        pipeline, built once and reused across batches, behind its own guard.
    utterance_boundary_model : BertUtteranceModel | None
        The loaded boundary model, when this worker has one.

    Returns tree bracket notation strings. Assignment computation is done in Rust.
    """
    batch_started_at = time.monotonic()

    total = len(req.items)
    pending, rejected = _admit_batch(list(req.items))

    match req.segmenter:
        case UtsegSegmenterV2.BOUNDARY_MODEL:
            if utterance_boundary_model is None:
                raise UtsegModelNotFoundError(
                    f"The request names the boundary-model segmenter for language "
                    f"'{req.lang}', and this worker has no utterance-boundary "
                    f"model loaded. Rust names that segmenter only for a language "
                    f"whose model it pinned for this worker, so check this "
                    f"worker's model download and load logs."
                )
            timed = [
                entry.measure(
                    partial(_boundary_outcome, utterance_boundary_model, entry.index)
                )
                for entry in pending
            ]
            L.info(
                "batch_infer utseg(boundary-model): %d items, %.3fs",
                total,
                time.monotonic() - batch_started_at,
            )
            return _assemble(total, timed, rejected)
        case UtsegSegmenterV2.STANZA_CONSTITUENCY:
            return _constituency_batch(
                req, load_stanza_pipeline, pending, rejected, batch_started_at
            )
        case _:
            assert_never(req.segmenter)


def _constituency_batch(
    req: UtsegInferenceRequest,
    load_stanza_pipeline: Callable[[LanguageCode], LoadedPipeline],
    pending: list[_PendingItem],
    rejected: list[_RejectedItem],
    batch_started_at: float,
) -> BatchInferResponse:
    """Segment a batch with the language's Stanza constituency pipeline."""
    total = len(req.items)

    # A single-word item needs no segmenter at all: one word is one utterance.
    short_circuit: list[_Answered] = []
    needs_parse: list[_PendingItem] = []
    for entry in pending:
        if len(entry.item.words) <= 1:
            short_circuit.append(entry.measure(_single_word_outcome))
        else:
            needs_parse.append(entry)

    if not needs_parse:
        return _assemble(total, short_circuit, rejected)

    pipeline = load_stanza_pipeline(req.lang)
    _emit_stanza_fallback_notice(req.lang)

    # One pipeline, loaded once per worker and language, called through its
    # own guard for the whole batch: Stanza does not promise that one
    # pipeline is safe to call from two threads.
    with pipeline.calling() as nlp:
        parsed = [
            entry.measure(partial(_constituency_outcome, nlp, entry.index))
            for entry in needs_parse
        ]

    # The batch total is a fact about the batch, so it is reported where a
    # batch fact belongs: this log line. It is deliberately NOT written onto
    # any item, because no item performed it.
    L.info(
        "batch_infer utseg: %d items, %.3fs",
        total,
        time.monotonic() - batch_started_at,
    )
    return _assemble(total, short_circuit + parsed, rejected)


# ---------------------------------------------------------------------------
# Constituency tree helpers (moved from pipelines/utterance/_utseg_callback.py)
# ---------------------------------------------------------------------------


def _leaf_count(tree: ConstituencyTree) -> int:
    """Count the number of leaf nodes under a constituency subtree."""
    try:
        children = tree.children
    except AttributeError:
        return 0
    count = 0
    for c in children:
        if c.is_leaf():
            count += 1
        else:
            count += _leaf_count(c)
    return count


def _parse_tree_indices(subtree: ConstituencyTree, offset: int) -> list[list[int]]:
    """Recursively extract S-level phrase leaf-index ranges from a constituency tree.

    Raises ``AttributeError`` (re-raised) if ``subtree`` is missing the
    ``children`` attribute. The previous behavior of swallowing the
    error and returning ``[]`` masked malformed Stanza constituency
    output as empty utseg assignments, a silent-failure pattern that
    the system-wide graceful-failure invariant rules out. Any caller
    that genuinely wants to tolerate a missing-children subtree must
    catch the error explicitly and decide what to do, rather than
    relying on this function to invent an empty result.
    """
    children = subtree.children

    result: list[list[int]] = []
    subtree_labels = [c.label.lower() if c.label else "" for c in children]
    has_coordination = any(lbl in ("cc", "conj") for lbl in subtree_labels)

    child_offset = offset
    for child in children:
        if child.is_leaf():
            child_offset += 1
            continue

        n_leaves = _leaf_count(child)
        child_start = child_offset

        if has_coordination and child.label == "S":
            result.append(list(range(child_start, child_start + n_leaves)))

        result += _parse_tree_indices(child, child_start)

        child_offset = child_start + n_leaves

    return result


def compute_assignments(words: list[str], nlp: StanzaNLP) -> list[int]:
    """Run constituency parsing + tree walking to compute word->utterance assignments.

    Returns a list parallel to *words* where each element is a 0-based group ID.
    """
    from itertools import groupby

    n = len(words)
    if n <= 1:
        return [0] * n

    parse = nlp(" ".join(words)).sentences
    pt = parse[0].constituency

    phrase_ranges = _parse_tree_indices(pt, 0)
    phrase_ranges = sorted(phrase_ranges, key=len)

    unique_ranges: list[list[int]] = []
    for rng in [*reversed(phrase_ranges), list(range(n))]:
        rng_set = set(rng)
        for existing in unique_ranges:
            rng_set -= set(existing)
        if rng_set and not any(rng_set.issubset(set(x)) for x in unique_ranges):
            unique_ranges.append(sorted(rng_set))
    unique_ranges = list(reversed(unique_ranges))

    unique_ranges = [r for r in unique_ranges if len(r) > 1]

    if not unique_ranges:
        return [0] * n

    word_to_phrase = [-1] * n
    for phrase_id, indices in enumerate(unique_ranges):
        for idx in indices:
            if 0 <= idx < n:
                word_to_phrase[idx] = phrase_id

    for i in range(n):
        if word_to_phrase[i] != -1:
            continue
        for j in range(i + 1, n):
            if word_to_phrase[j] != -1:
                word_to_phrase[i] = word_to_phrase[j]
                break
        else:
            for j in range(i - 1, -1, -1):
                if word_to_phrase[j] != -1:
                    word_to_phrase[i] = word_to_phrase[j]
                    break

    if any(x == -1 for x in word_to_phrase):
        return [0] * n

    groups: list[list[int]] = [
        list(word_indices)
        for _, word_indices in groupby(range(n), key=lambda i: word_to_phrase[i])
    ]

    merged: list[list[int]] = []
    pending: list[int] = []
    for grp in groups:
        if len(grp) < 3:
            pending += grp
        else:
            merged.append(pending + grp)
            pending = []
    if pending:
        if merged:
            merged[-1] += pending
        else:
            merged.append(pending)

    assignments = [0] * n
    for group_id, group_indices in enumerate(merged):
        for idx in group_indices:
            assignments[idx] = group_id

    return assignments
