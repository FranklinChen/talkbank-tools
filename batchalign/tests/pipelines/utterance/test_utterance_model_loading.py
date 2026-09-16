"""Worker utterance-model loading is driven by the pin Rust sends.

The worker no longer decides WHICH boundary model a language loads: the Rust
manifest names it and sends it with the spawn, so these tests drive the loader
through that wire rather than through a local language table.
"""

# affects: crates/batchalign/src/model_manifest.rs
# affects: batchalign/worker/_model_loading/utterance.py

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from batchalign.worker._model_loading.utterance import (
    PINNED_UTSEG_MODEL_KEY,
    load_utterance_model,
)
from batchalign.worker._types import _state

CANTONESE_ID = "PolyU-AngelChanLab/Cantonese-Utterance-Segmentation"
CANTONESE_COMMIT = "9784aeb9e11c674f55a5e70468094736f8285bb6"
MANDARIN_ID = "talkbank/CHATUtterance-zh_CN"
MANDARIN_COMMIT = "d52d3578344d570d652e644e5da3869f25a073e4"


def _overrides(model_id: str, commit: str) -> dict[str, str]:
    """The spawn overrides Rust sends for a language it pins a model for."""
    return {
        PINNED_UTSEG_MODEL_KEY: json.dumps(
            {"id": model_id, "revision": {"kind": "commit", "commit": commit}}
        )
    }


@pytest.fixture(autouse=True)
def _restore_state():
    """Never leak a loaded model or name into the next test."""
    old_model = _state.utterance_boundary_model
    old_name = _state.utterance_model_name
    yield
    _state.utterance_boundary_model = old_model
    _state.utterance_model_name = old_name


@pytest.fixture
def captured(monkeypatch) -> list[dict[str, Any]]:
    """Record what the loader constructs, with the hub call stubbed out."""
    calls: list[dict[str, Any]] = []

    class FakeBoundaryModel:
        def __init__(
            self,
            *,
            model_id: str,
            model_path: str,
            model_revision: str,
            lang: str | None = None,
        ) -> None:
            calls.append(
                {
                    "model_id": model_id,
                    "model_path": model_path,
                    "model_revision": model_revision,
                    "lang": lang,
                }
            )
            self.model_id = model_id
            self.model_revision = model_revision
            self.lang = lang

    def fake_snapshot(model_id: str, commit: str | None):
        from batchalign.worker._model_loading.pinned_hub import ResolvedSnapshot

        assert commit is not None, "a pinned load must ask for an exact commit"
        return ResolvedSnapshot(
            path=f"/cache/{model_id}/snapshots/{commit}", commit=commit
        )

    monkeypatch.setattr(
        "batchalign.models.utterance.infer.BertUtteranceModel", FakeBoundaryModel
    )
    monkeypatch.setattr(
        "batchalign.worker._model_loading.utterance.resolve_pinned_snapshot",
        fake_snapshot,
    )
    return calls


def test_loads_the_pinned_model_from_a_resolved_snapshot(captured) -> None:
    """The model loads from a local path, at the exact commit Rust pinned."""
    load_utterance_model("yue", _overrides(CANTONESE_ID, CANTONESE_COMMIT))

    assert captured == [
        {
            "model_id": CANTONESE_ID,
            "model_path": f"/cache/{CANTONESE_ID}/snapshots/{CANTONESE_COMMIT}",
            "model_revision": CANTONESE_COMMIT,
            "lang": "yue",
        }
    ]
    assert _state.utterance_model_name == CANTONESE_ID


def test_the_reported_identity_is_the_id_never_the_local_path(captured) -> None:
    """A stamp must never carry this machine's directory layout."""
    load_utterance_model("cmn", _overrides(MANDARIN_ID, MANDARIN_COMMIT))

    assert _state.utterance_model_name == MANDARIN_ID
    assert "/cache/" not in _state.utterance_model_name


def test_a_language_with_no_pin_loads_no_model(captured) -> None:
    """Absence of a pin means this language has no boundary model.

    The worker does not consult a local table to second-guess that: deciding it
    here is the duplicate table this change removed.
    """
    load_utterance_model("spa", {})

    assert captured == []
    assert _state.utterance_boundary_model is None
    assert _state.utterance_model_name == ""


def test_a_malformed_pin_is_refused_rather_than_loading_something_else(
    captured,
) -> None:
    """Loading a different revision because the pin was unreadable is the
    silent substitution this workstream removes."""
    with pytest.raises(ValidationError):
        load_utterance_model("eng", {PINNED_UTSEG_MODEL_KEY: "{not json"})

    assert captured == []
    assert _state.utterance_boundary_model is None
