"""Worker registry file I/O for persistent TCP workers.

The registry file (``workers.json``) is the discovery mechanism between
independently started worker daemons and the Rust server. Each worker writes
its own entry on startup and removes it on shutdown. The server reads the
registry to discover pre-started workers, health-checks each one, and removes
stale entries (workers that crashed without cleanup).

Every read-modify-write holds an exclusive lock on ``workers.json.lock``
beside the registry (``fcntl.flock`` on Unix, ``msvcrt.locking`` on Windows):
the same lock file the Rust server's registry writer takes
(``crates/batchalign/src/file_lock.rs``), never the registry file itself,
which each write replaces by rename. Writes go through a private temporary
file (mode ``0600``, as the Rust writer's) and ``os.replace``.

Registry path: ``~/.batchalign3/workers.json`` (configurable via
``BATCHALIGN_STATE_DIR``).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from batchalign.inference._domain_types import LanguageCode, TcpPort

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WorkerRegistryEntry:
    """One worker's entry in the registry file."""

    pid: int
    host: str
    port: TcpPort
    profile: str
    lang: LanguageCode
    # The build identity of the server that spawned this daemon, from
    # `BATCHALIGN_BUILD_IDENTITY`; None when the daemon was started without
    # one (by hand, or by a build before the field existed). Required, not
    # defaulted: every construction states it, and `_entry_from_json` is the
    # one place an absent key is read as None. The Rust server refuses to
    # adopt a daemon whose build identity is not its own.
    build_identity: str | None
    engine_overrides: str = ""
    ownership: str = "external"
    owner_server_instance_id: str = ""
    owner_server_pid: int | None = None
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


# ---------------------------------------------------------------------------
# Registry path resolution
# ---------------------------------------------------------------------------


def _default_registry_path() -> Path:
    """Resolve the default registry file path from environment."""
    state_dir = os.environ.get("BATCHALIGN_STATE_DIR", "")
    if state_dir.strip():
        return Path(state_dir) / "workers.json"
    home = Path.home()
    return home / ".batchalign3" / "workers.json"


# ---------------------------------------------------------------------------
# File-locked read/write helpers
# ---------------------------------------------------------------------------


_IS_WINDOWS = sys.platform == "win32"


def _lock_path(registry_path: Path) -> Path:
    """The lock file guarding ``registry_path``: ``<registry>.lock`` beside it."""
    return registry_path.with_name(registry_path.name + ".lock")


def _lock_file(f: IO[str]) -> None:
    """Acquire an exclusive lock on the file descriptor."""
    fd = f.fileno()
    if _IS_WINDOWS:
        import msvcrt

        # `msvcrt.locking` locks bytes from the current position, so lock and
        # unlock both start at byte 0.
        f.seek(0)
        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX)


def _unlock_file(f: IO[str]) -> None:
    """Release the lock on the file descriptor."""
    fd = f.fileno()
    if _IS_WINDOWS:
        import msvcrt

        f.seek(0)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


def _fsync_directory(directory: Path) -> None:
    """Make a rename in ``directory`` durable, as the Rust writer does.

    POSIX only: a directory cannot be opened for an fsync on Windows, where
    the replace is already durable once it returns.
    """
    if _IS_WINDOWS:
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def _registry_lock(registry_path: Path) -> Iterator[None]:
    """Hold the registry's lock for one read-modify-write."""
    lock_path = _lock_path(registry_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as f:
        _lock_file(f)
        try:
            yield
        finally:
            _unlock_file(f)


def _entry_from_json(item: dict[str, object]) -> WorkerRegistryEntry:
    """Rebuild one registry entry from its JSON object.

    `build_identity` is the one key an entry may lack: entries written by a
    build before the field existed. Their build is unknown, which is exactly
    what None records. Any other missing or unexpected key still raises
    `TypeError`, which every caller handles as an unreadable entry.

    The single construction route from JSON; the four read paths each used to
    spell `WorkerRegistryEntry(**item)` themselves.
    """
    fields = dict(item)
    fields.setdefault("build_identity", None)
    return WorkerRegistryEntry(**fields)  # type: ignore[arg-type]


def _read_entries(registry_path: Path) -> list[WorkerRegistryEntry]:
    """Read all entries from the registry file (no locking)."""
    if not registry_path.exists():
        return []
    try:
        raw = json.loads(registry_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read worker registry %s: %s", registry_path, exc)
        return []
    if not isinstance(raw, list):
        return []
    entries: list[WorkerRegistryEntry] = []
    for item in raw:
        if isinstance(item, dict):
            try:
                entries.append(_entry_from_json(item))
            except TypeError:
                continue
    return entries


def _write_entries(registry_path: Path, entries: list[WorkerRegistryEntry]) -> None:
    """Replace the registry file atomically (the caller holds the lock).

    Through a uniquely named temporary file beside it, created ``0600`` by
    ``mkstemp`` (the mode the Rust writer gives it), then ``os.replace`` and
    an fsync of the directory, so the replacement survives a crash.
    """
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps([asdict(e) for e in entries], indent=2) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        dir=registry_path.parent, prefix=registry_path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_name, registry_path)
        _fsync_directory(registry_path.parent)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise


def _remove_entries(
    registry_path: Path, stale: Callable[[WorkerRegistryEntry], bool]
) -> bool:
    """Remove every entry ``stale`` matches, under the registry's lock.

    Returns ``True`` when at least one entry was removed. A missing registry
    has nothing to remove.
    """
    if not registry_path.exists():
        return False
    with _registry_lock(registry_path):
        entries = _read_entries(registry_path)
        remaining = [e for e in entries if not stale(e)]
        if len(remaining) == len(entries):
            return False
        _write_entries(registry_path, remaining)
        return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_worker(
    entry: WorkerRegistryEntry,
    *,
    registry_path: Path | None = None,
) -> None:
    """Add a worker entry to the registry file.

    If an entry with the same ``(host, port)`` already exists, it is replaced.
    """
    path = registry_path or _default_registry_path()
    with _registry_lock(path):
        # Replace an existing entry for the same (host, port).
        entries = [
            e
            for e in _read_entries(path)
            if not (e.host == entry.host and e.port == entry.port)
        ]
        entries.append(entry)
        _write_entries(path, entries)

    logger.info(
        "Registered worker pid=%d at %s:%d in %s",
        entry.pid,
        entry.host,
        entry.port,
        path,
    )


def unregister_worker(
    *,
    host: str,
    port: TcpPort,
    registry_path: Path | None = None,
) -> bool:
    """Remove a worker entry from the registry file.

    Returns ``True`` if an entry was removed, ``False`` if not found.
    """
    path = registry_path or _default_registry_path()
    return _remove_entries(path, lambda e: e.host == host and e.port == port)


def list_workers(
    *,
    registry_path: Path | None = None,
) -> list[WorkerRegistryEntry]:
    """Read all worker entries from the registry file."""
    path = registry_path or _default_registry_path()
    return _read_entries(path)


def remove_stale_entry(
    *,
    pid: int,
    registry_path: Path | None = None,
) -> bool:
    """Remove a worker entry by PID (for crash cleanup).

    Returns ``True`` if an entry was removed.
    """
    path = registry_path or _default_registry_path()
    return _remove_entries(path, lambda e: e.pid == pid)
