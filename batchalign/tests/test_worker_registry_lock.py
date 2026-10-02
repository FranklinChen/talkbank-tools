"""The worker registry's read-modify-write holds the lock the Rust writer takes,
and writes the file with the Rust writer's mode."""

from __future__ import annotations

import stat
import sys
from typing import TYPE_CHECKING

import pytest

from batchalign.worker._registry import (
    WorkerRegistryEntry,
    _lock_path,
    _registry_lock,
    list_workers,
    register_worker,
    remove_stale_entry,
    unregister_worker,
)

if TYPE_CHECKING:
    from pathlib import Path


def _entry(pid: int, port: int) -> WorkerRegistryEntry:
    return WorkerRegistryEntry(
        pid=pid,
        host="127.0.0.1",
        port=port,  # type: ignore[arg-type]
        profile="gpu",
        lang="eng",  # type: ignore[arg-type]
        build_identity=None,
    )


def test_the_lock_file_is_the_one_the_rust_writer_takes(tmp_path: Path) -> None:
    """``workers.json.lock`` beside the registry, as `file_lock::lock_file_for`
    names it, never the registry file each write replaces."""
    registry = tmp_path / "workers.json"
    assert _lock_path(registry) == tmp_path / "workers.json.lock"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_registry_writes_are_owner_only(tmp_path: Path) -> None:
    """Every write leaves the registry ``0600``, the mode the Rust writer gives
    it, whichever writer wrote last."""
    registry = tmp_path / "state" / "workers.json"
    register_worker(_entry(1, 9001), registry_path=registry)
    assert stat.S_IMODE(registry.stat().st_mode) == 0o600
    register_worker(_entry(2, 9002), registry_path=registry)
    assert unregister_worker(host="127.0.0.1", port=9001, registry_path=registry)  # type: ignore[arg-type]
    assert stat.S_IMODE(registry.stat().st_mode) == 0o600
    assert [e.pid for e in list_workers(registry_path=registry)] == [2]
    assert remove_stale_entry(pid=2, registry_path=registry)
    assert list_workers(registry_path=registry) == []


@pytest.mark.skipif(sys.platform == "win32", reason="flock")
def test_a_held_registry_lock_excludes_other_holders(tmp_path: Path) -> None:
    """While one writer holds the lock, a second open of the lock file cannot
    take it, which is how the Rust writer (another process) sees it."""
    import fcntl

    registry = tmp_path / "workers.json"
    with _registry_lock(registry):
        with open(_lock_path(registry), "a+") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    with open(_lock_path(registry), "a+") as other:
        fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_windows_unlock_releases_rather_than_locks(monkeypatch, tmp_path: Path) -> None:
    """On Windows the unlock is ``LK_UNLCK`` at byte 0, the bytes the lock
    took; it was ``LK_NBLCK``, a second lock, so the lock was never released.
    """
    import types

    from batchalign.worker import _registry

    calls: list[tuple[int, int]] = []
    fake_msvcrt = types.SimpleNamespace(
        LK_LOCK=1,
        LK_NBLCK=2,
        LK_UNLCK=0,
        locking=lambda fd, mode, nbytes: calls.append((mode, nbytes)),
    )
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    monkeypatch.setattr(_registry, "_IS_WINDOWS", True)
    with open(tmp_path / "workers.json.lock", "a+", encoding="utf-8") as f:
        f.write("x")
        _registry._lock_file(f)
        f.write("moved the position")
        _registry._unlock_file(f)
    assert calls == [(fake_msvcrt.LK_LOCK, 1), (fake_msvcrt.LK_UNLCK, 1)]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX directory fsync")
def test_a_registry_write_fsyncs_its_directory(monkeypatch, tmp_path: Path) -> None:
    """The replace is made durable by an fsync of the registry's directory, as
    the Rust writer does."""
    import os

    synced_directories: list[bool] = []
    real_fsync = os.fsync

    def recording_fsync(fd: int) -> None:
        synced_directories.append(stat.S_ISDIR(os.fstat(fd).st_mode))
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    register_worker(_entry(1, 9001), registry_path=tmp_path / "workers.json")
    assert True in synced_directories
