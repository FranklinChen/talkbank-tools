"""Verify that supervisor death interrupts an actual in-flight worker request."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from enum import Enum
from queue import Queue
from typing import TYPE_CHECKING

import pytest

from batchalign.tests._asr_model_pins import request_models
from batchalign.worker._types_v2 import (
    AsrBackendV2,
    AsrRequestV2,
    ExecuteRequestV2,
    InferenceTaskV2,
    PreparedAudioInputV2,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

# Failure bound for the actual pipe effects, not a simulated request duration.
PIPE_TIMEOUT_S = 30

_WORKER = """
import runpy
import threading
from batchalign.worker import _types

def hold_request():
    from batchalign.worker._protocol import _write_json
    _write_json({"fixture": "request-entered"})
    threading.Event().wait()

_types.sleep_test_delay = hold_request
runpy.run_module("batchalign.worker", run_name="__main__")
"""

_SUPERVISOR = """
import json, os, subprocess, sys
worker = subprocess.Popen(
    [sys.executable, "-c", sys.argv[1], "--test-echo", "--profile", "gpu",
     "--serving", "sequential", "--supervisor-pid", str(os.getpid())],
    stdin=subprocess.PIPE, stdout=sys.stdout, stderr=subprocess.DEVNULL,
    text=True, start_new_session=True,
)
print(json.dumps({"fixture_worker_pid": worker.pid}), flush=True)
worker.stdin.write(sys.argv[2] + "\\n")
worker.stdin.flush()
sys.stdin.read()
"""


def _held_request() -> str:
    request = ExecuteRequestV2(
        request_id="held-request",
        task=InferenceTaskV2.ASR,
        payload=AsrRequestV2(
            lang="eng",
            backend=AsrBackendV2.LOCAL_WHISPER,
            input=PreparedAudioInputV2(audio_ref_id="audio-ref-1"),
            models=request_models(AsrBackendV2.LOCAL_WHISPER),
        ),
        attachments=[],
    )
    return json.dumps({"op": "execute_v2", "request": request.model_dump(mode="json")})


class PipeEnd(Enum):
    EOF = "eof"


@dataclass(frozen=True, slots=True)
class InFlightRequest:
    """The worker has reached the request handler and is holding its pipe open."""

    _supervisor: subprocess.Popen[str]
    _lines: Queue[str | PipeEnd]

    def kill_supervisor_and_observe_worker_exit(self) -> None:
        self._supervisor.kill()
        self._supervisor.wait(timeout=PIPE_TIMEOUT_S)
        # Once the supervisor exits, only the worker can close the last writer.
        assert self._lines.get(timeout=PIPE_TIMEOUT_S) is PipeEnd.EOF


@contextmanager
def _in_flight_request() -> Iterator[InFlightRequest]:
    lines: Queue[str | PipeEnd] = Queue()
    supervisor = subprocess.Popen(
        [sys.executable, "-c", _SUPERVISOR, _WORKER, _held_request()],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert supervisor.stdin is not None
    assert supervisor.stdout is not None
    stdout = supervisor.stdout

    def read_pipe() -> None:
        for line in stdout:
            lines.put(line)
        lines.put(PipeEnd.EOF)

    reader = threading.Thread(target=read_pipe, daemon=True)
    reader.start()
    worker_pid: int | None = None
    try:
        # Parent PID announcement and worker readiness share the pipe. Either
        # process can write first, but the handler marker follows both.
        first = lines.get(timeout=PIPE_TIMEOUT_S)
        assert isinstance(first, str)
        first_report = json.loads(first)
        if "fixture_worker_pid" in first_report:
            announced_pid = first_report["fixture_worker_pid"]
            assert type(announced_pid) is int and announced_pid > 0
            worker_pid = announced_pid
            ready = lines.get(timeout=PIPE_TIMEOUT_S)
            assert isinstance(ready, str)
            report = json.loads(ready)
        else:
            report = first_report
            ownership = lines.get(timeout=PIPE_TIMEOUT_S)
            assert isinstance(ownership, str)
            announced_pid = json.loads(ownership)["fixture_worker_pid"]
            assert type(announced_pid) is int and announced_pid > 0
            worker_pid = announced_pid
        assert report["ready"] is True and report["pid"] == worker_pid

        entered = lines.get(timeout=PIPE_TIMEOUT_S)
        assert isinstance(entered, str)
        assert json.loads(entered) == {"fixture": "request-entered"}
        yield InFlightRequest(supervisor, lines)
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait(timeout=PIPE_TIMEOUT_S)
        supervisor.stdin.close()
        reader.join(timeout=PIPE_TIMEOUT_S)
        if reader.is_alive() and worker_pid is not None:
            with suppress(ProcessLookupError):
                os.killpg(worker_pid, signal.SIGKILL)
            reader.join(timeout=PIPE_TIMEOUT_S)
        if not reader.is_alive():
            stdout.close()


@pytest.mark.skipif(os.name != "posix", reason="requires isolated POSIX process groups")
def test_a_worker_busy_in_a_request_exits_when_its_supervisor_is_killed() -> None:
    """Guards: The actual worker closes its pipe on supervisor SIGKILL after
        reaching the V2 request handler.
    Kind: contract
    Not a type: Static launch states cannot prove kernel process liveness,
        delivery of the supervisor death signal, or EOF on the live pipe.
    State: The test owns a supervisor and one worker, with the echo handler held
        by an event after real request admission. No model or socket is needed.
    Concurrency: A reader thread forwards wire lines and EOF. The busy marker
        precedes the kill, and teardown only targets the fixture's process group.
    """
    with _in_flight_request() as request:
        request.kill_supervisor_and_observe_worker_exit()
