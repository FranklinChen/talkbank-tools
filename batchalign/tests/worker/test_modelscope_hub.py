"""ModelScope snapshots are materialized before loading, and only a real
download is announced.

``resolve_modelscope_snapshot`` probes the local cache first; a hit returns
the cached directory without a network call or an event, a miss downloads
under ``download_liveness``, and a failed download is a named
``PinnedSnapshotError`` rather than a loader quietly fetching on first use.
"""

from __future__ import annotations

from pathlib import Path
from typing import assert_never

import pytest
from modelscope_hub.errors import CacheNotFound

import batchalign.worker._download_liveness as liveness
from batchalign.worker._model_loading import modelscope_hub
from batchalign.worker._model_loading.pinned_hub import PinnedSnapshotError
from batchalign.worker._types_v2 import DownloadGrowthV2, LoadMeasureV2, PhaseV2

MODEL = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"


CHECKPOINT = ("model.pt",)


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, object]:
    """A fake ModelScope over real directories.

    ``cached`` is what the local lookup finds: ``None`` (nothing), or a list
    of file names already in the snapshot directory. A download writes
    ``downloads`` (default: a complete snapshot) and ``fail`` makes it raise.
    """
    state: dict[str, object] = {
        "cached": None,
        "downloads": ["config.yaml", "model.pt"],
        "fail": None,
        "calls": [],
    }

    def snapshot_dir(model_id: str, local_dir: str | None) -> Path:
        # A pinned member names its own directory; an unpinned one leaves it
        # to ModelScope.
        if local_dir is not None:
            return Path(local_dir)
        return tmp_path / "cache" / "unpinned" / model_id

    def fill(directory: Path, names: object) -> None:
        assert isinstance(names, list)
        directory.mkdir(parents=True, exist_ok=True)
        for name in names:
            # Real sizes, so a download is growth the liveness meter counts.
            (directory / name).write_bytes(b"x" * (2 << 20))
        if "model.pt" in names:
            (directory / "model.pt.incomplete").unlink(missing_ok=True)

    def fake_snapshot_download(
        model_id, revision=None, local_dir=None, local_files_only=False
    ):
        calls = state["calls"]
        assert isinstance(calls, list)
        calls.append((model_id, revision, local_files_only))
        state["local_dir"] = local_dir
        directory = snapshot_dir(model_id, local_dir)
        if local_files_only:
            if state["cached"] is None:
                raise CacheNotFound(f"no snapshot for {model_id} at {revision}")
            fill(directory, state["cached"])
            return str(directory)
        if state["fail"] is not None:
            raise state["fail"]  # type: ignore[misc]
        fill(directory, state["downloads"])
        return str(directory)

    monkeypatch.setattr("modelscope.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(modelscope_hub, "_cache_root", lambda: tmp_path / "cache")
    # Liveness watches this cache, so a download here is growth it reports.
    monkeypatch.setattr(liveness, "modelscope_roots", lambda: (tmp_path / "cache",))
    # The download's phases (announced, completed), as ``<stage>:<edge>``;
    # its growth reports are not recorded.
    events: list[str] = []

    def record(stage: str, measure: LoadMeasureV2) -> None:
        match measure:
            case PhaseV2(edge=edge):
                events.append(f"{stage}:{edge.value}")
            case DownloadGrowthV2():
                pass
            case _ as unreachable:
                assert_never(unreachable)

    monkeypatch.setattr(liveness, "write_load_progress", record)
    state["events"] = events
    return state


STAGE = f"downloading_modelscope_{MODEL.replace('/', '_')}"


def resolve(tag: str | None = "v2.0.4") -> modelscope_hub.ModelScopeSnapshot:
    return modelscope_hub.resolve_modelscope_snapshot(
        MODEL, tag, kind="vad", requires=CHECKPOINT
    )


def test_a_cached_snapshot_is_returned_without_a_download_or_an_event(hub) -> None:
    hub["cached"] = ["config.yaml", "model.pt"]
    snapshot = resolve()
    assert snapshot.path.name == "v2.0.4"
    assert hub["calls"] == [(MODEL, "v2.0.4", True)]
    assert hub["events"] == []


def test_a_pinned_member_resolves_in_its_own_versioned_directory(hub) -> None:
    """ModelScope's legacy cache answers for any tag, and downloads a new tag
    over an old one; a pinned member names its versioned directory instead,
    so the tag is the directory by construction."""
    snapshot = resolve()
    expected = (
        modelscope_hub._cache_root()
        / "models"
        / MODEL.replace("/", "--")
        / "snapshots"
        / "v2.0.4"
    )
    assert snapshot.path == expected
    assert hub["local_dir"] == str(expected)


def test_a_missing_snapshot_downloads_at_the_pinned_tag_under_liveness(hub) -> None:
    snapshot = resolve()
    assert (snapshot.path / "model.pt").is_file()
    assert hub["calls"] == [(MODEL, "v2.0.4", True), (MODEL, "v2.0.4", False)]
    assert hub["events"] == [f"{STAGE}:begin", f"{STAGE}:complete"]


def test_a_cached_snapshot_a_stall_left_partial_is_downloaded_not_loaded(
    hub,
) -> None:
    """THE 2026-11-05 CASE: a stalled download left ``model.pt.incomplete``,
    and the local lookup returned the directory anyway. It must be completed
    by a download, not loaded."""
    hub["cached"] = ["config.yaml", "model.pt.incomplete"]
    snapshot = resolve()
    assert (snapshot.path / "model.pt").is_file()
    assert not (snapshot.path / "model.pt.incomplete").exists()
    assert hub["calls"] == [(MODEL, "v2.0.4", True), (MODEL, "v2.0.4", False)]


def test_a_cached_snapshot_without_its_checkpoint_is_downloaded(hub) -> None:
    hub["cached"] = ["config.yaml"]
    snapshot = resolve()
    assert (snapshot.path / "model.pt").is_file()
    assert len(hub["calls"]) == 2


def test_a_download_that_returns_incomplete_is_refused(hub) -> None:
    hub["downloads"] = ["config.yaml", "model.pt.incomplete"]
    with pytest.raises(PinnedSnapshotError, match=r"partial download"):
        resolve()


def test_a_failed_download_is_a_named_refusal(hub) -> None:
    hub["fail"] = ConnectionError("connection reset")
    with pytest.raises(PinnedSnapshotError, match=r"at v2\.0\.4 from ModelScope"):
        resolve()
    assert f"{STAGE}:complete" not in hub["events"]


def test_a_probe_failure_other_than_a_miss_is_a_named_refusal(hub, monkeypatch) -> None:
    def broken_lookup(*_args, **_kwargs):
        raise PermissionError("cache unreadable")

    monkeypatch.setattr("modelscope.snapshot_download", broken_lookup)
    with pytest.raises(PinnedSnapshotError, match="in the ModelScope cache"):
        resolve()


def test_an_unpinned_member_asks_for_the_hub_default(hub) -> None:
    hub["fail"] = ConnectionError("offline")
    with pytest.raises(PinnedSnapshotError, match="at the hub default"):
        resolve(None)
