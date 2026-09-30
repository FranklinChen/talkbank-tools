"""A worker process never outlives the server that launched it.

A stdio worker's only consumer is the server holding its pipes. Once that
server is gone no result the worker computes can be delivered, so continuing is
pure waste: on 2026-09-30 a server stopped by ``batchalign3 serve stop`` (which
escalated to SIGKILL) left a busy Whisper worker, reparented to PID 1, running
until it was killed by hand. The sequential serving loop reads stdin only
between requests, so the closed pipe went unnoticed for the whole of the
request in progress; and stdin EOF alone is not a death signal anyway, because
a caller may half-close stdin and still read the replies.

So the worker watches its supervisor directly. The Rust launcher passes its own
PID as ``--supervisor-pid``; the worker arms an OS exit notification on that
process (``kqueue`` ``NOTE_EXIT`` on macOS and the BSDs, a pidfd on Linux) and,
when it fires, kills its own process group, taking any child processes (ffmpeg,
decoders) with it. The wait is an event, not a poll.

PID reuse cannot fool the watch: the worker also requires that its parent IS
the supervisor after arming. A supervisor that died before the watch was armed
has already reparented the worker, so the check fails and the worker exits.
"""

from __future__ import annotations

import logging
import os
import select
import signal
import sys
import threading
from enum import Enum

L = logging.getLogger("batchalign.worker")


class SupervisorGone(Exception):
    """The supervisor exited before its exit could be watched."""


class _KqueueExitWatch:
    """``NOTE_EXIT`` on a process, through kqueue (macOS and the BSDs)."""

    def __init__(self, pid: int) -> None:
        if sys.platform == "linux":
            raise OSError("kqueue is not available on Linux")
        self._kqueue = select.kqueue()
        event = select.kevent(
            pid,
            filter=select.KQ_FILTER_PROC,
            flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
            fflags=select.KQ_NOTE_EXIT,
        )
        try:
            # Registration only; a process that no longer exists is refused.
            self._kqueue.control([event], 0, 0)
        except ProcessLookupError as error:
            raise SupervisorGone(str(error)) from error

    def wait(self) -> None:
        """Block until the watched process exits."""
        self._kqueue.control(None, 1, None)


class _PidfdExitWatch:
    """A pidfd, readable once the process exits (Linux)."""

    def __init__(self, pid: int) -> None:
        self._pidfd: int
        if sys.platform != "linux":
            raise OSError("pidfd is available only on Linux")
        try:
            self._pidfd = os.pidfd_open(pid)
        except ProcessLookupError as error:
            raise SupervisorGone(str(error)) from error

    def wait(self) -> None:
        """Block until the watched process exits."""
        poller = select.poll()
        poller.register(self._pidfd, select.POLLIN)
        poller.poll()


def _arm_exit_watch(pid: int) -> _KqueueExitWatch | _PidfdExitWatch:
    """Arm this platform's exit notification for ``pid``."""
    if sys.platform == "linux":
        return _PidfdExitWatch(pid)
    return _KqueueExitWatch(pid)


class _GroupRole(Enum):
    """Whether this worker leads its own process group.

    The Rust launcher makes every worker a group leader, so killing the group
    takes the worker's children too. A worker started some other way shares
    its launcher's group, and killing THAT group would kill the launcher.
    """

    LEADER = "leader"
    MEMBER = "member"

    @classmethod
    def current(cls) -> _GroupRole:
        return cls.LEADER if os.getpgrp() == os.getpid() else cls.MEMBER


def _terminate_self() -> None:
    """End this worker, and its children when it leads its group."""
    match _GroupRole.current():
        case _GroupRole.LEADER:
            os.killpg(os.getpgrp(), signal.SIGKILL)
        case _GroupRole.MEMBER:
            os.kill(os.getpid(), signal.SIGKILL)


def watch_supervisor(supervisor_pid: int) -> None:
    """Terminate this worker when ``supervisor_pid`` exits.

    Arms the watch before returning, so a supervisor that is already gone ends
    the worker here rather than leaving it running unwatched.
    """
    try:
        watch = _arm_exit_watch(supervisor_pid)
    except SupervisorGone:
        L.warning("supervisor %d already exited; worker exiting", supervisor_pid)
        _terminate_self()
        return
    if os.getppid() != supervisor_pid:
        L.warning(
            "supervisor %d is not this worker's parent (%d); worker exiting",
            supervisor_pid,
            os.getppid(),
        )
        _terminate_self()
        return

    def _await_supervisor_exit() -> None:
        watch.wait()
        # Nobody can read what this worker would produce from here on.
        _terminate_self()

    threading.Thread(
        target=_await_supervisor_exit, name="supervisor-watch", daemon=True
    ).start()
