"""Native resource contract for the frontend preparation owner."""

import sys
from pathlib import Path

import pytest

from scripts.frontend_workflow import WorkspaceLease

if sys.platform != "win32":
    import fcntl


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX flock adapter contract")
def test_workspace_lease_excludes_an_independent_file_description(
    tmp_path: Path,
) -> None:
    """Guards: The workspace lease excludes a competing open and releases it on exit.
    Kind: contract.
    Not a type: The OS must enforce flock exclusion and descriptor-close release;
    the compiler can restrict producers but cannot establish those kernel effects.
    State: One private workspace and two independently opened real lock descriptors
    exercise the production adapter and a nonblocking kernel observation.
    Concurrency: Independent opens compete in one process without threads or sleeps;
    pytest's private directory separates this lock from every other workflow.
    """
    with WorkspaceLease.acquire(tmp_path) as lease:
        with (lease.root / "target/frontend-workflow.lock").open("a+b") as competing:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competing.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (tmp_path / "target/frontend-workflow.lock").open("a+b") as successor:
        fcntl.flock(successor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
