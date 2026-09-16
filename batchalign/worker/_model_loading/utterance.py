"""Utterance-model bootstrap helpers for worker startup."""

from __future__ import annotations

import logging

from pydantic import TypeAdapter

from batchalign.inference._domain_types import LanguageCode
from batchalign.worker._model_loading.pinned_hub import (
    hub_commit_of,
    resolve_pinned_snapshot,
)
from batchalign.worker._types import _state
from batchalign.worker._types_v2 import RequestedModelV2

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
) -> None:
    """Load the pinned utterance-boundary model for one language, when there is one."""
    # Bootstrap imports this loader for every profile, including model-free
    # echo workers. Only the loading operation owns the heavy model import.
    from batchalign.models.utterance.infer import BertUtteranceModel
    from batchalign.worker._progress import (
        HF_ARTIFACTS_BERT_TOKEN_CLASSIFICATION,
        emit_hf_download_if_missing,
    )

    _state.utterance_boundary_model = None
    _state.utterance_model_name = ""

    pinned = pinned_utseg_model(engine_overrides)
    if pinned is None:
        L.info("No utterance boundary model pinned for %s", lang)
        return

    # Probed AT THE PINNED REVISION, which is the revision the resolve below
    # materializes. Without it the probe asks whether the moving head is
    # cached, so a host holding any earlier revision of this model reads as
    # fully cached and the pinned commit downloads in silence: the multi-minute
    # wait this function exists to announce, reintroduced by the commit that
    # introduced the pin.
    commit = hub_commit_of(pinned)
    emit_hf_download_if_missing(
        pinned.id,
        kind="utterance boundary detection",
        artifacts=HF_ARTIFACTS_BERT_TOKEN_CLASSIFICATION,
        revision=commit,
    )
    # Resolve the snapshot BEFORE constructing the model, so the loader is
    # handed a local path and the revision is read off the directory that
    # actually exists. Loading by name and asking the library afterwards is
    # what made the revision optional.
    resolved = resolve_pinned_snapshot(pinned.id, commit)
    _state.utterance_boundary_model = BertUtteranceModel(
        model_id=pinned.id,
        model_path=resolved.path,
        model_revision=resolved.commit,
        lang=lang,
    )
    _state.utterance_model_name = pinned.id
    L.info(
        "Loaded utterance boundary model %s at %s for %s",
        pinned.id,
        resolved.commit,
        lang,
    )
