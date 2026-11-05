"""Which of a repository's files the pinned resolver downloads.

The hub call is faked: each fake "repository" is a list of file names, and
the fake ``snapshot_download`` materializes only those its ``allow_patterns``
admit, as the real one does. The selection was checked against the real
repositories and loaders separately (``scripts/check_hub_file_selection.py``,
``scripts/verify_hub_selection_loads.py``).
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

import pytest

from batchalign.worker._model_loading import pinned_hub
from batchalign.worker._model_loading.pinned_hub import (
    PinnedSnapshotError,
    WeightLayout,
    default_weights,
    resolve_repository,
    resolve_transformers,
)
from batchalign.worker._types_v2 import RequestedCommitV2, RequestedModelV2

COMMIT = "0" * 40


def pinned(model_id: str) -> RequestedModelV2:
    """A request for ``model_id`` at the test commit."""
    return RequestedModelV2(id=model_id, revision=RequestedCommitV2(commit=COMMIT))


WHISPER_REPOSITORY = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors",
    "model.fp32-00001-of-00002.safetensors",
    "model.safetensors.index.fp32.json",
    "pytorch_model.bin",
    "pytorch_model.fp32-00001-of-00002.bin",
    "flax_model.msgpack",
    "README.md",
]


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A fake hub: ``hub[repo] = [names]``; returns the calls it saw."""
    repositories: dict[str, list[str]] = {}
    calls: list[list[str] | None] = []

    def fake_snapshot_download(model_id, *, revision, allow_patterns):
        calls.append(allow_patterns)
        snapshot = (
            tmp_path / model_id.replace("/", "--") / "snapshots" / (revision or COMMIT)
        )
        snapshot.mkdir(parents=True, exist_ok=True)
        for name in repositories[model_id]:
            if allow_patterns is None or any(
                fnmatch.fnmatch(name, pattern) for pattern in allow_patterns
            ):
                (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
                (snapshot / name).write_text("")
        return str(snapshot)

    monkeypatch.setattr(pinned_hub, "snapshot_download", fake_snapshot_download)
    return repositories, calls


def files_in(path: str) -> set[str]:
    return {entry.name for entry in Path(path).iterdir()}


def test_a_transformers_model_fetches_one_weights_format_not_the_repository(
    hub,
) -> None:
    """Whisper's repository holds five weight formats; the loader reads one."""
    repositories, calls = hub
    repositories["openai/whisper-large-v3"] = WHISPER_REPOSITORY
    resolved = resolve_transformers(pinned("openai/whisper-large-v3"), kind="test")
    fetched = files_in(resolved.path)
    assert "model.safetensors" in fetched
    assert not fetched & {
        "model.fp32-00001-of-00002.safetensors",
        "pytorch_model.bin",
        "pytorch_model.fp32-00001-of-00002.bin",
        "flax_model.msgpack",
        "README.md",
    }
    assert {"config.json", "tokenizer.json", "merges.txt"} <= fetched
    assert len(calls) == 1, "safetensors were present, so no second fetch"


def test_without_safetensors_the_pytorch_weights_are_fetched(hub) -> None:
    """The precedence `from_pretrained` applies: safetensors, else `.bin`."""
    repositories, _ = hub
    repositories["talkbank/CHATUtterance-en"] = [
        "config.json",
        "tokenizer.json",
        "vocab.txt",
        "pytorch_model.bin",
        "README.md",
    ]
    resolved = resolve_transformers(pinned("talkbank/CHATUtterance-en"), kind="test")
    assert files_in(resolved.path) == {
        "config.json",
        "tokenizer.json",
        "vocab.txt",
        "pytorch_model.bin",
    }


def test_a_revision_without_default_weights_is_refused(hub) -> None:
    repositories, _ = hub
    repositories["org/configs-only"] = ["config.json", "flax_model.msgpack"]
    with pytest.raises(PinnedSnapshotError, match="no default-variant"):
        resolve_transformers(pinned("org/configs-only"), kind="test")


def test_a_whole_repository_family_fetches_everything(hub) -> None:
    repositories, calls = hub
    repositories["funasr/fsmn-vad"] = ["am.mvn", "config.yaml", "model.pt"]
    resolved = resolve_repository(pinned("funasr/fsmn-vad"), kind="test")
    assert files_in(resolved.path) == {"am.mvn", "config.yaml", "model.pt"}
    assert calls == [None]


AUDIO_FLAMINGO_REPOSITORY = [
    "config.json",
    "tokenizer.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "think/config.json",
    "think/adapter_model.safetensors",
]
"""``nvidia/audio-flamingo-3-hf``'s shape (read from the hub, 2026-11-05):
the same weights as one file and as shards, 16.5 GB each."""


def test_weights_shipped_twice_are_fetched_once_in_from_pretrained_order(
    hub,
) -> None:
    """``from_pretrained`` reads a single ``model.safetensors`` before an
    index, so the shards beside it are never read and must not download."""
    repositories, calls = hub
    repositories["nvidia/audio-flamingo-3-hf"] = AUDIO_FLAMINGO_REPOSITORY
    resolved = resolve_transformers(pinned("nvidia/audio-flamingo-3-hf"), kind="test")
    fetched = files_in(resolved.path)
    assert "model.safetensors" in fetched
    assert not fetched & {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    assert len(calls) == 1


def test_sharded_safetensors_are_fetched_when_there_is_no_single_file(hub) -> None:
    repositories, _ = hub
    repositories["org/sharded"] = [
        name for name in AUDIO_FLAMINGO_REPOSITORY if name != "model.safetensors"
    ]
    resolved = resolve_transformers(pinned("org/sharded"), kind="test")
    fetched = files_in(resolved.path)
    assert {
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    } <= fetched


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (["model.safetensors", "pytorch_model.bin"], WeightLayout.SAFETENSORS),
        (
            ["model.safetensors.index.json", "model-00001-of-00002.safetensors"],
            WeightLayout.SAFETENSORS_SHARDED,
        ),
        # An index whose shards are absent is not a layout a loader can read.
        (["model.safetensors.index.json", "pytorch_model.bin"], WeightLayout.PYTORCH),
        (
            ["pytorch_model.bin.index.json", "pytorch_model-00001-of-00003.bin"],
            WeightLayout.PYTORCH_SHARDED,
        ),
        (["config.json", "flax_model.msgpack"], None),
    ],
)
def test_default_weights_follows_from_pretrained_precedence(names, expected) -> None:
    assert default_weights(names) is expected
