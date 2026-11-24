"""The worker registry's read-modify-write holds the lock the Rust writer takes,
and writes the file with the Rust writer's mode."""

from __future__ import annotations

import os
import stat
import sys
from typing import TYPE_CHECKING

import pytest

from batchalign.worker._registry import (
    WorkerRegistryEntry,
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
        port=port,
        profile="gpu",
        lang="eng",
        build_identity=None,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_registry_writes_are_owner_only(tmp_path: Path) -> None:
    """Guards: Register, unregister and stale cleanup mutate only the named
        rows and publish owner-only registry bytes.
    Kind: contract
    Not a type: Actual POSIX modes and file contents can differ from the
        producer's intended state after atomic replacement.
    State: A private filesystem registry exercises create, replace and removal.
    Concurrency: The fixture owns its directory and no ambient registry.
    """
    registry = tmp_path / "state" / "workers.json"
    register_worker(_entry(1, 9001), registry_path=registry)
    assert stat.S_IMODE(registry.stat().st_mode) == 0o600
    register_worker(_entry(2, 9002), registry_path=registry)
    assert unregister_worker(host="127.0.0.1", port=9001, registry_path=registry)
    assert stat.S_IMODE(registry.stat().st_mode) == 0o600
    assert [e.pid for e in list_workers(registry_path=registry)] == [2]
    assert remove_stale_entry(pid=2, registry_path=registry)
    assert list_workers(registry_path=registry) == []


@pytest.mark.skipif(sys.platform == "win32", reason="flock")
def test_a_held_registry_lock_excludes_other_holders(tmp_path: Path) -> None:
    """Guards: The live lock excludes a second holder on the lock path used by
        the Rust writer and releases that path on exit.
    Kind: contract
    Not a type: The path spelling crosses languages, and the kernel decides
        whether a separately opened file can obtain the lock.
    State: Two handles on an actual POSIX lock path exercise kernel locking.
    Concurrency: A nonblocking attempt observes exclusion under the held guard
        and succeeds after release. No shared directory or process is needed.
    """
    import fcntl

    registry = tmp_path / "workers.json"
    # This is the independent spelling from Rust file_lock::lock_file_for.
    lock = tmp_path / "workers.json.lock"
    with _registry_lock(registry), lock.open("a+") as observer:
        with pytest.raises(BlockingIOError):
            fcntl.flock(observer.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    with lock.open("a+") as observer:
        fcntl.flock(observer.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_windows_unlock_releases_rather_than_locks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Guards: The Windows adapter locks and unlocks one byte at byte zero.
    Kind: contract
    Not a type: The platform call consumes native operation values and the
        actual descriptor position. Buffering can move the cursor between calls.
    State: One local file and an msvcrt double capture mode, length and offset.
    Concurrency: The mock is scoped to this test and the descriptor is private.
    """
    import types

    from batchalign.worker import _registry

    calls: list[tuple[int, int, int]] = []

    def locking(fd: int, mode: int, nbytes: int) -> None:
        calls.append((mode, nbytes, os.lseek(fd, 0, os.SEEK_CUR)))

    fake_msvcrt = types.SimpleNamespace(LK_LOCK=1, LK_UNLCK=0, locking=locking)
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    monkeypatch.setattr(_registry, "_IS_WINDOWS", True)
    with open(tmp_path / "workers.json.lock", "a+", encoding="utf-8") as f:
        f.write("x")
        _registry._lock_file(f)
        f.write("moved the position")
        _registry._unlock_file(f)
    assert calls == [(fake_msvcrt.LK_LOCK, 1, 0), (fake_msvcrt.LK_UNLCK, 1, 0)]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX directory fsync")
def test_a_registry_write_fsyncs_its_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Guards: Publishing a registry update syncs the actual containing directory.
    Kind: contract
    Not a type: An admitted path cannot prove that its descriptor reached the
        fsync adapter or that the descriptor refers to the intended directory.
    State: A local registry and delegated fsync capture the actual directory inode.
    Concurrency: The test owns the directory and scopes its adapter replacement.
    """
    synced_directories: list[tuple[int, int]] = []
    real_fsync = os.fsync

    def recording_fsync(fd: int) -> None:
        info = os.fstat(fd)
        if stat.S_ISDIR(info.st_mode):
            synced_directories.append((info.st_dev, info.st_ino))
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    register_worker(_entry(1, 9001), registry_path=tmp_path / "workers.json")
    directory = tmp_path.stat()
    assert (directory.st_dev, directory.st_ino) in synced_directories
