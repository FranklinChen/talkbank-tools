"""Stand-in snapshots for tests that replace the model loaders.

A snapshot's constructor is its admission: ``TransformersSnapshot`` and
``RepositorySnapshot`` refuse a path that is not a snapshot directory named by
a commit (and, for Transformers, one holding no weights), and
``ModelScopeSnapshot`` refuses one that is incomplete. A stand-in is therefore
a real directory of that shape, built here under one temporary root removed
at exit, holding empty files no test reads. One shape, one place to change.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path

from batchalign.worker._model_loading.modelscope_hub import ModelScopeSnapshot
from batchalign.worker._model_loading.pinned_hub import (
    RepositorySnapshot,
    TransformersSnapshot,
    hub_commit_of,
    unpinned,
)
from batchalign.worker._types_v2 import RequestedModelV2

STAND_IN_COMMIT = "a" * 40
"""A well-formed commit no hub holds."""

_ROOT = Path(tempfile.mkdtemp(prefix="batchalign-stand-in-snapshots-"))
atexit.register(shutil.rmtree, _ROOT, ignore_errors=True)


def _snapshot_dir(model_id: str, commit: str, files: tuple[str, ...]) -> Path:
    """``<root>/<owner>--<name>/snapshots/<commit>/``, holding ``files``."""
    directory = _ROOT / model_id.replace("/", "--") / "snapshots" / commit
    directory.mkdir(parents=True, exist_ok=True)
    for name in files:
        (directory / name).touch()
    return directory


def transformers_snapshot(
    model_id: str = "org/stand-in", commit: str = STAND_IN_COMMIT
) -> TransformersSnapshot:
    """A Transformers snapshot of ``model_id`` holding an empty weight file."""
    directory = _snapshot_dir(model_id, commit, ("model.safetensors",))
    return TransformersSnapshot(requested=unpinned(model_id), path=str(directory))


def repository_snapshot(
    model_id: str = "org/stand-in-repository", commit: str = STAND_IN_COMMIT
) -> RepositorySnapshot:
    """A whole-repository snapshot of ``model_id``."""
    directory = _snapshot_dir(model_id, commit, ("model.pt",))
    return RepositorySnapshot(requested=unpinned(model_id), path=str(directory))


def modelscope_snapshot(model_id: str) -> ModelScopeSnapshot:
    """A complete ModelScope snapshot of ``model_id`` with its checkpoint."""
    directory = _snapshot_dir(model_id, "v0.0.0", ("model.pt",))
    return ModelScopeSnapshot(path=directory, requires=("model.pt",))


def fake_resolve_transformers(
    requested: RequestedModelV2, *, kind: str
) -> TransformersSnapshot:
    """``resolve_transformers`` without a hub: the request's snapshot, at its
    pinned commit or the stand-in one, recording the request it was given."""
    directory = _snapshot_dir(
        requested.id,
        hub_commit_of(requested) or STAND_IN_COMMIT,
        ("model.safetensors",),
    )
    return TransformersSnapshot(requested=requested, path=str(directory))


def fake_resolve_repository(
    requested: RequestedModelV2, *, kind: str
) -> RepositorySnapshot:
    """``resolve_repository`` without a hub, as ``fake_resolve_transformers``."""
    directory = _snapshot_dir(
        requested.id, hub_commit_of(requested) or STAND_IN_COMMIT, ("model.pt",)
    )
    return RepositorySnapshot(requested=requested, path=str(directory))
