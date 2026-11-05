"""A snapshot's constructor is its admission.

``TransformersSnapshot``, ``RepositorySnapshot`` and ``ModelScopeSnapshot`` are
proofs that a loader will read finished files of the model it was asked for.
Python cannot hide a dataclass constructor, so the constructor itself refuses
anything else, and these tests hold each refusal. They replace an audit that
only checked WHERE snapshots were built.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from batchalign.worker._model_loading import pinned_hub
from batchalign.worker._model_loading.modelscope_hub import (
    ModelScopeSnapshot,
    SnapshotRefusal,
    SnapshotRefused,
)
from batchalign.worker._model_loading.pinned_hub import (
    PinnedSnapshotError,
    RepositorySnapshot,
    TransformersSnapshot,
    resolve_transformers,
    unpinned,
)
from batchalign.worker._types_v2 import (
    ObservedCommitV2,
    RequestedCommitV2,
    RequestedModelV2,
)

COMMIT = "c" * 40
OTHER = "d" * 40


def snapshot_dir(root: Path, leaf: str, *files: str) -> Path:
    directory = root / "models--org--model" / "snapshots" / leaf
    directory.mkdir(parents=True)
    for name in files:
        (directory / name).touch()
    return directory


def pinned_to(commit: str) -> RequestedModelV2:
    return RequestedModelV2(id="org/model", revision=RequestedCommitV2(commit=commit))


def test_a_transformers_snapshot_derives_its_commit_and_records_the_load(
    tmp_path: Path,
) -> None:
    directory = snapshot_dir(tmp_path, COMMIT, "config.json", "model.safetensors")
    snapshot = TransformersSnapshot(requested=pinned_to(COMMIT), path=str(directory))
    assert snapshot.commit == COMMIT
    assert snapshot.model_id == "org/model"
    assert snapshot.loaded.observed == ObservedCommitV2(commit=COMMIT)
    assert snapshot.loaded.requested == RequestedCommitV2(commit=COMMIT)


@pytest.mark.parametrize("snapshot_type", [TransformersSnapshot, RepositorySnapshot])
def test_a_missing_directory_is_refused(tmp_path: Path, snapshot_type) -> None:
    with pytest.raises(PinnedSnapshotError, match="not a directory"):
        snapshot_type(requested=unpinned("org/model"), path=str(tmp_path / COMMIT))


@pytest.mark.parametrize("snapshot_type", [TransformersSnapshot, RepositorySnapshot])
def test_a_leaf_that_is_not_a_commit_is_refused(tmp_path: Path, snapshot_type) -> None:
    directory = snapshot_dir(tmp_path, "main", "model.safetensors")
    with pytest.raises(PinnedSnapshotError, match="not a 40-character commit"):
        snapshot_type(requested=unpinned("org/model"), path=str(directory))


@pytest.mark.parametrize("snapshot_type", [TransformersSnapshot, RepositorySnapshot])
def test_a_commit_other_than_the_pinned_one_is_refused(
    tmp_path: Path, snapshot_type
) -> None:
    directory = snapshot_dir(tmp_path, OTHER, "model.safetensors")
    with pytest.raises(PinnedSnapshotError, match="did not choose"):
        snapshot_type(requested=pinned_to(COMMIT), path=str(directory))


def test_a_transformers_snapshot_without_weights_is_refused(tmp_path: Path) -> None:
    directory = snapshot_dir(tmp_path, COMMIT, "config.json", "flax_model.msgpack")
    with pytest.raises(PinnedSnapshotError, match="no default-variant weights"):
        TransformersSnapshot(requested=pinned_to(COMMIT), path=str(directory))


def test_a_repository_snapshot_needs_no_weight_layout(tmp_path: Path) -> None:
    directory = snapshot_dir(tmp_path, COMMIT, "config.yaml", "model.pt")
    assert (
        RepositorySnapshot(requested=pinned_to(COMMIT), path=str(directory)).commit
        == COMMIT
    )


@pytest.mark.parametrize(
    ("files", "refusal"),
    [
        (("model.pt.incomplete",), SnapshotRefusal.PARTIAL),
        (("config.yaml",), SnapshotRefusal.LACKS_REQUIRED),
    ],
)
def test_a_modelscope_snapshot_refuses_an_incomplete_directory(
    tmp_path: Path, files, refusal
) -> None:
    directory = snapshot_dir(tmp_path, "v2.0.4", *files)
    with pytest.raises(SnapshotRefused) as refused:
        ModelScopeSnapshot(path=directory, requires=("model.pt",))
    assert refused.value.refusal is refusal
    assert ModelScopeSnapshot.admit(directory, ("model.pt",)) is refusal


def test_a_missing_modelscope_directory_is_refused(tmp_path: Path) -> None:
    assert (
        ModelScopeSnapshot.admit(tmp_path / "absent", ("model.pt",))
        is SnapshotRefusal.MISSING
    )


def test_an_unpinned_resolve_names_its_first_commit_on_every_later_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After the first hub call, every later layout call names the commit it
    landed on: no further round trip to ask what the default branch is, and
    no chance of assembling one snapshot from two revisions."""
    revisions: list[str | None] = []
    directory = tmp_path / "models--org--model" / "snapshots" / COMMIT

    def fake_snapshot_download(model_id, *, revision, allow_patterns):
        revisions.append(revision)
        directory.mkdir(parents=True, exist_ok=True)
        if "pytorch_model.bin" in allow_patterns:
            (directory / "pytorch_model.bin").touch()
        return str(directory)

    monkeypatch.setattr(pinned_hub, "snapshot_download", fake_snapshot_download)
    snapshot = resolve_transformers(unpinned("org/model"), kind="test")
    assert snapshot.commit == COMMIT
    # Single safetensors, sharded safetensors, then the single PyTorch file.
    assert revisions == [None, COMMIT, COMMIT]
