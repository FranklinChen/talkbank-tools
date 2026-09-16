"""Stanza constituency inference: words -> utterance boundary assignments.

Pure inference, no CHAT, no caching, no pipeline.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, assert_never, cast

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from batchalign.inference.types import ConstituencyTree, StanzaNLP
    from batchalign.models.utterance.infer import BertUtteranceModel

from batchalign.models.utterance.evidence import (
    ClassifiedBoundaryEvidence,
    ModelShortCircuit,
    NormalizationOmission,
    UtteranceBoundaryPrediction,
)
from batchalign.providers import (
    BatchInferRequest,
    BatchInferResponse,
    InferResponse,
)
from batchalign.worker._types import WorkerJSONValue
from batchalign.worker._types_v2 import (
    UtsegBoundaryModelEvidenceV2,
    UtsegClassifiedBoundaryEvidenceV2,
    UtsegItemResultV2,
    UtsegModelShortCircuitV2,
    UtsegNormalizationOmissionV2,
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

    result = UtsegItemResultV2(
        assignments=list(prediction.assignments),
        boundary_model_evidence=UtsegBoundaryModelEvidenceV2(
            model_id=prediction.model_id,
            model_revision=prediction.model_revision,
            normalization_revision=prediction.normalization_revision,
            adjacency_policy_revision=prediction.adjacency_policy_revision,
            word_evidence=wire_evidence,
        ),
    )
    return cast(
        dict[str, WorkerJSONValue],
        result.model_dump(mode="json", exclude_none=True),
    )


class UtsegModelNotFoundError(RuntimeError):
    """Raised when this worker has no utterance-boundary model LOADED for
    the requested language and the Stanza constituency-parser fallback was
    not authorized for the request.

    This is a worker-local load fact, not a statement about which languages
    have a model. Whether a boundary model EXISTS for a language is decided
    in Rust before anything is dispatched
    (``crates/batchalign/src/utseg_route.rs``), so a job in a language with
    no model is refused at planning time and never reaches here. Reaching
    here therefore means the model for a supported language could not be
    loaded in this process.

    It is kept as a backstop rather than as the decision. It mirrors the
    ``WhisperHubModelNotFoundError`` pattern in
    ``batchalign/inference/whisper_hub.py``: surface the gap rather than
    silently substitute one model for another.
    """


# Stage identifier for the opt-in fallback notice. Stage names form a
# closed protocol vocabulary consumed by the dashboard / CLI for
# filtering and dedupe; the language goes in user_message, not the stage.
_STAGE_UTSEG_FALLBACK_OPT_IN = "utseg_unsupported_language_fallback"

# Per-process dedupe: warn once per (requested_lang, pack) pair. Worker
# processes don't outlive a deploy, so the set never needs eviction.
_FALLBACK_NOTICE_FIRED: set[tuple[str, str | None]] = set()


def _emit_stanza_fallback_notice(
    requested_lang: str,
    pack: str | None,
) -> None:
    """Surface the BERT-absent → Stanza substitution to the user.

    Only fires when the operator has opted in via
    ``--utseg-fallback-stanza`` (which sets
    ``BatchInferRequest.allow_stanza_fallback=True``). The
    default-refuse path raises ``UtsegModelNotFoundError`` instead and
    never reaches this helper.
    """
    # Avoid a circular import at module load time, the progress
    # protocol pulls in worker config that imports this module
    # transitively in some test setups.
    from batchalign.worker._progress import emit_download_event

    key = (requested_lang, pack)
    if key in _FALLBACK_NOTICE_FIRED:
        return
    _FALLBACK_NOTICE_FIRED.add(key)

    requested_display = requested_lang or "<unspecified>"
    pack_display = pack if pack is not None else "<none>"

    user_message = (
        f"No TalkBank utseg model for language '{requested_display}'; "
        f"using Stanza constituency parsing ({pack_display} pack) "
        f"because --utseg-fallback-stanza was passed. Quality will vary."
    )

    L.warning(
        "utseg opt-in fallback: lang=%r → Stanza pack %r",
        requested_display,
        pack_display,
    )

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
# Until 2026-09-16 this module measured the whole batch and wrote that single
# duration onto ``results[0]``, leaving every other item at 0.0. The first
# item's timing was therefore inflated by every other item's work, every other
# item's timing was simply wrong, and neither was distinguishable from an item
# that genuinely took no measurable time. That is a misattribution inside
# provenance-bearing evidence, so the types below exist to make it
# unrepresentable rather than merely discouraged: a duration is only ever
# produced by measuring one item's own call, and it travels with the index
# that earned it.


@dataclass(frozen=True, slots=True)
class _Produced:
    """One item's own successful result payload."""

    result: dict[str, WorkerJSONValue]


@dataclass(frozen=True, slots=True)
class _Failed:
    """One item's own failure, reported against that item alone."""

    error: str


_ItemOutcome = _Produced | _Failed
"""What one item's work produced, before anything has been timed."""


@dataclass(frozen=True, slots=True, init=False)
class _TimedItem:
    """One item's outcome beside the elapsed time of that item's own work.

    There is no constructor that accepts a duration: `_PendingItem.measure` is
    the only route to a value of this type, and it measures the call it wraps.
    A batch total has no signature to travel through, which is precisely what
    the old `results[0] = InferResponse(..., elapsed_s=batch_total)` line did.
    """

    index: int
    outcome: _ItemOutcome
    elapsed_s: float

    @classmethod
    def _measured(
        cls, index: int, outcome: _ItemOutcome, elapsed_s: float
    ) -> _TimedItem:
        """Build the only way a timed item is ever built; see `measure`."""
        instance = object.__new__(cls)
        object.__setattr__(instance, "index", index)
        object.__setattr__(instance, "outcome", outcome)
        object.__setattr__(instance, "elapsed_s", elapsed_s)
        return instance

    @property
    def response(self) -> InferResponse:
        """Lower this item's own outcome and its own timing onto the wire."""
        match self.outcome:
            case _Produced(result=result):
                return InferResponse(result=result, elapsed_s=self.elapsed_s)
            case _Failed(error=error):
                return InferResponse(error=error, elapsed_s=self.elapsed_s)
            case _ as unreachable:
                assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class _RejectedItem:
    """One batch position whose payload never parsed into a request item.

    No work is attributable to it, and this variant is what says so: the zero
    it reports is not a measurement that happened to round down.
    """

    index: int
    error: str

    @property
    def response(self) -> InferResponse:
        """Report the rejection at its own position."""
        return InferResponse(error=self.error, elapsed_s=0.0)


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

    def measure(self, work: Callable[[UtsegBatchItem], _ItemOutcome]) -> _TimedItem:
        """Run this item's own work and attribute exactly that item's time."""
        started_at = time.monotonic()
        outcome = work(self.item)
        return _TimedItem._measured(self.index, outcome, time.monotonic() - started_at)


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
    timed: list[_TimedItem],
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


def _single_word_outcome(item: UtsegBatchItem) -> _ItemOutcome:
    """Segment an item that needs no segmenter: one word is one utterance."""
    return _Produced(result={"assignments": [0] * len(item.words)})


def _no_pipeline_outcome(_item: UtsegBatchItem) -> _ItemOutcome:
    """Report no trees when no language pipeline could be built."""
    return _Produced(result={"trees": []})


def _boundary_outcome(
    model: BertUtteranceModel, index: int, item: UtsegBatchItem
) -> _ItemOutcome:
    """Predict one item's boundary evidence, reporting its own failure."""
    try:
        prediction = model.predict_boundary_evidence(item.words)
    except (IndexError, AttributeError, TypeError, ValueError) as error:
        L.warning("Utseg boundary-model infer failed for item %d: %s", index, error)
        return _Failed(error=f"Utseg boundary-model inference failed: {error}")
    return _Produced(result=_serialize_boundary_prediction(prediction))


def _constituency_outcome(
    nlp: StanzaNLP, index: int, item: UtsegBatchItem
) -> _ItemOutcome:
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
        L.warning("Utseg infer failed for item %d: %s", index, error)
        return _Produced(result={"trees": []})
    return _Produced(result={"trees": trees})


def batch_infer_utseg(
    req: BatchInferRequest,
    build_stanza_config: Callable[
        [list[str]], tuple[list[str], dict[str, dict[str, str | bool]]]
    ],
    utterance_boundary_model: BertUtteranceModel | None = None,
) -> BatchInferResponse:
    """Batch Stanza constituency inference: (words) -> tree strings.

    Parameters
    ----------
    req : BatchInferRequest
        Batch of UtsegBatchItem payloads.
    build_stanza_config : callable
        Function ``(langs) -> (lang_alpha2, configs)`` from the utseg engine.

    Returns tree bracket notation strings. Assignment computation is done in Rust.
    """
    batch_started_at = time.monotonic()

    total = len(req.items)
    pending, rejected = _admit_batch(req.items)

    if utterance_boundary_model is not None:
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

    # A single-word item needs no segmenter at all: one word is one utterance.
    short_circuit: list[_TimedItem] = []
    needs_parse: list[_PendingItem] = []
    for entry in pending:
        if len(entry.item.words) <= 1:
            short_circuit.append(entry.measure(_single_word_outcome))
        else:
            needs_parse.append(entry)

    if not needs_parse:
        return _assemble(total, short_circuit, rejected)

    # This worker has no boundary model LOADED for the request, which is not
    # the same question as whether one EXISTS for the language: a worker can
    # reach here for a language that has a model which this process did not
    # load. Rust now answers the existence question before dispatch
    # (`crates/batchalign/src/utseg_route.rs`), from the language alone and
    # independent of payload content, so this raise is a worker-local backstop
    # rather than the decision that fails a job after ASR has run.
    if not req.allow_stanza_fallback:
        raise UtsegModelNotFoundError(
            f"This worker has no utterance-boundary model loaded for language "
            f"'{req.lang or '<unspecified>'}', and the Stanza constituency "
            f"fallback was not authorized for this request. Whether a model "
            f"exists for a language is decided before dispatch, in "
            f"crates/batchalign/src/utseg_route.rs, so a language with no "
            f"model is refused at planning time and does not reach here: check "
            f"this worker's model download and load logs first. To segment "
            f"with the legacy Stanza constituency parser instead, pass "
            f"--utseg-fallback-stanza (quality will vary, and for a language "
            f"that does have a model it will mask the load failure)."
        )

    langs: list[str] = [req.lang] if req.lang else ["eng"]
    lang_alpha2, configs = build_stanza_config(langs)

    import stanza
    from stanza import DownloadMethod

    from batchalign.worker._progress import emit_download_event

    _emit_stanza_fallback_notice(
        req.lang,
        lang_alpha2[0] if lang_alpha2 else None,
    )

    nlp: StanzaNLP
    if len(lang_alpha2) > 1:
        # Multilingual pipeline pulls one language pack per ``lang_alpha2`` plus
        # the language-id model. First-run cost is a sum across packs; emit a
        # single event so the user sees the wait, even if intermediate library
        # progress prints reach only stderr.
        emit_download_event(
            stage="downloading_stanza_utseg_multilingual",
            user_message=(
                "Downloading Stanza utterance-segmentation pipeline for "
                f"{', '.join(lang_alpha2)} (one-time, one language pack per "
                "language; future runs will use the local cache)…"
            ),
        )
        nlp = stanza.MultilingualPipeline(
            lang_configs=configs,
            lang_id_config={"langid_lang_subset": lang_alpha2},
            download_method=DownloadMethod.REUSE_RESOURCES,
        )
    elif lang_alpha2:
        emit_download_event(
            stage=f"downloading_stanza_utseg_{lang_alpha2[0]}",
            user_message=(
                f"Downloading Stanza utterance-segmentation pipeline for "
                f"{lang_alpha2[0]} (one-time, ~250-500 MB; future runs will "
                "use the local cache)…"
            ),
        )
        nlp = stanza.Pipeline(
            lang=lang_alpha2[0],
            **configs[lang_alpha2[0]],
            download_method=DownloadMethod.REUSE_RESOURCES,
        )
    else:
        return _assemble(
            total,
            short_circuit
            + [entry.measure(_no_pipeline_outcome) for entry in needs_parse],
            rejected,
        )

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
