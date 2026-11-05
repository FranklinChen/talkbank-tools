"""Utterance-model bootstrap helpers for worker startup."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from batchalign.inference._domain_types import LanguageCode
from batchalign.worker._model_loading.pinned_hub import resolve_transformers
from batchalign.worker._types_v2 import RequestedModelV2

if TYPE_CHECKING:
    from batchalign.inference.utseg import BoundaryModel

L = logging.getLogger("batchalign.worker")

# The engine-override key the Rust control plane sends the pinned boundary
# model under. Must equal `model_manifest::PINNED_UTSEG_MODEL_KEY` on the Rust
# side.
PINNED_UTSEG_MODEL_KEY = "utseg_pinned_model"

_PINNED_UTSEG_MODEL: TypeAdapter[RequestedModelV2] = TypeAdapter(RequestedModelV2)


def pinned_utseg_model(
    engine_overrides: dict[str, str] | None,
) -> RequestedModelV2 | None:
    """Return the boundary model Rust pinned for this worker, or ``None``.

    ``None`` is a real state rather than a missing value. The control plane
    injects the pin only for a language it seeds a boundary model for, so its
    absence means this language has none and this worker must not load one.
    Deciding that here, from a local language table, is precisely the second
    copy that was deleted: the table that decided which model loaded lived on
    this side, while the table that decided whether a job could run lived in
    Rust, and nothing kept them in step.

    A malformed value is NOT tolerated: it means the control plane and this
    worker disagree about the wire, and loading some other revision because the
    pin could not be read is exactly the silent substitution this workstream
    removes.
    """
    raw = (engine_overrides or {}).get(PINNED_UTSEG_MODEL_KEY)
    if raw is None:
        return None
    return _PINNED_UTSEG_MODEL.validate_json(raw)


def load_utterance_model(
    lang: LanguageCode,
    engine_overrides: dict[str, str] | None = None,
) -> BoundaryModel:
    """Load the pinned utterance-boundary model for one language, when there is one.

    Returns ``PinnedBoundaryModel`` with the loaded model, or
    ``NoPinnedBoundaryModel`` when Rust pinned none for this language, which
    the boundary segmenter refuses by name. The utseg load hands it to the
    handler it installs; nothing is written to worker state.
    """
    from batchalign.inference.utseg import NoPinnedBoundaryModel, PinnedBoundaryModel

    # Bootstrap imports this loader for every profile, including model-free
    # echo workers. Only the loading operation owns the heavy model import.
    from batchalign.models.utterance.infer import BertUtteranceModel

    pinned = pinned_utseg_model(engine_overrides)
    if pinned is None:
        L.info("No utterance boundary model pinned for %s", lang)
        return NoPinnedBoundaryModel()

    # The resolve below materializes the PINNED revision, under download
    # liveness: a host holding only an earlier revision downloads the pinned
    # one, announced and watched, rather than in silence.
    # Resolve the snapshot BEFORE constructing the model, so the loader is
    # handed a local path and the revision is read off the directory that
    # actually exists. Loading by name and asking the library afterwards is
    # what made the revision optional.
    resolved = resolve_transformers(pinned, kind="utterance boundary detection")
    model = BertUtteranceModel(snapshot=resolved, lang=lang)
    L.info(
        "Loaded utterance boundary model %s at %s for %s",
        pinned.id,
        resolved.commit,
        lang,
    )
    return PinnedBoundaryModel(model)
