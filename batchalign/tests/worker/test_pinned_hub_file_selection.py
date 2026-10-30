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
    HubFiles,
    PinnedSnapshotError,
    resolve_pinned_snapshot,
)

COMMIT = "0" * 40

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
    resolved = resolve_pinned_snapshot(
        "openai/whisper-large-v3", COMMIT, HubFiles.TRANSFORMERS
    )
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
    resolved = resolve_pinned_snapshot(
        "talkbank/CHATUtterance-en", COMMIT, HubFiles.TRANSFORMERS
    )
    assert files_in(resolved.path) == {
        "config.json",
        "tokenizer.json",
        "vocab.txt",
        "pytorch_model.bin",
    }


def test_a_revision_without_default_weights_is_refused(hub) -> None:
    repositories, _ = hub
    repositories["org/configs-only"] = ["config.json", "flax_model.msgpack"]
    with pytest.raises(PinnedSnapshotError, match="neither safetensors nor PyTorch"):
        resolve_pinned_snapshot("org/configs-only", COMMIT, HubFiles.TRANSFORMERS)


def test_a_whole_repository_family_fetches_everything(hub) -> None:
    repositories, calls = hub
    repositories["funasr/fsmn-vad"] = ["am.mvn", "config.yaml", "model.pt"]
    resolved = resolve_pinned_snapshot(
        "funasr/fsmn-vad", COMMIT, HubFiles.WHOLE_REPOSITORY
    )
    assert files_in(resolved.path) == {"am.mvn", "config.yaml", "model.pt"}
    assert calls == [None]
