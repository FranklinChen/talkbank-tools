"""A stdio worker busy in a request dies with the server that launched it.

Field failure (2026-09-30): ``batchalign3 serve stop`` escalated to SIGKILL on
a server whose shutdown had not finished; its Whisper worker, in the middle of
an inference, was reparented to PID 1 and kept computing until it was killed
by hand. Nothing it computed could ever be delivered.

The boundary here is the real one: an intermediate process plays the server,
launches a real worker exactly as the Rust launcher does (its own process
group, stdin and stdout pipes, ``--supervisor-pid``), hands it one request the
test-echo worker holds for ten minutes, and is then SIGKILLed. The worker's
stdout is this test's pipe and nobody else's once the supervisor is dead, so
end-of-file on it IS the worker exiting: the wait is an event, and the bound
only decides failure.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading

from batchalign.tests._asr_model_pins import request_models
from batchalign.worker._types_v2 import (
    AsrBackendV2,
    AsrRequestV2,
    ExecuteRequestV2,
    InferenceTaskV2,
    PreparedAudioInputV2,
)

# The worker holds the request this long; it can only exit by itself after.
HELD_REQUEST_MS = 600_000
# The worker must be gone this long after its supervisor dies.
EXIT_BOUND_S = 30.0

_SUPERVISOR = """
import os, subprocess, sys
worker = subprocess.Popen(
    [sys.executable, "-m", "batchalign.worker", "--test-echo", "--profile", "gpu",
     "--serving", "sequential", "--test-delay-ms", sys.argv[2],
     "--supervisor-pid", str(os.getpid())],
    stdin=subprocess.PIPE, stdout=sys.stdout, stderr=subprocess.DEVNULL,
    text=True, start_new_session=True,
)
print(f"worker-pid {worker.pid}", flush=True)
worker.stdin.write(sys.argv[1] + "\\n")
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


def test_a_worker_busy_in_a_request_exits_when_its_supervisor_is_killed() -> None:
    supervisor = subprocess.Popen(
        [sys.executable, "-c", _SUPERVISOR, _held_request(), str(HELD_REQUEST_MS)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert supervisor.stdout is not None
    stdout = supervisor.stdout
    worker_pid_line = stdout.readline()
    assert worker_pid_line.startswith("worker-pid "), worker_pid_line
    worker_pid = int(worker_pid_line.split()[1])
    ready_line = stdout.readline()
    assert '"ready": true' in ready_line, ready_line

    supervisor.send_signal(signal.SIGKILL)
    supervisor.wait(timeout=10)

    # EOF arrives when the last writer, the worker, is gone.
    drained = threading.Event()

    def _drain() -> None:
        stdout.read()
        drained.set()

    threading.Thread(target=_drain, daemon=True).start()
    try:
        assert drained.wait(timeout=EXIT_BOUND_S), (
            f"worker {worker_pid} outlived its SIGKILLed supervisor by "
            f"{EXIT_BOUND_S}s while holding a request"
        )
    finally:
        if not drained.is_set():
            os.killpg(worker_pid, signal.SIGKILL)
        assert supervisor.stdin is not None
        supervisor.stdin.close()
        stdout.close()
