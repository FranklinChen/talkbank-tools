"""The worker serves requests the way its launcher says, never its own way.

Whether a worker process serves requests concurrently is decided once, by the
Rust pool that launches it and routes requests to it
(``batchalign::worker::serving::WorkerServing``), and passed as ``--serving``.
Until 2026-09-30 the worker decided for itself from a CUDA probe while the pool
assumed every GPU-profile worker was concurrent; on a CPU-only host the pool
multiplexed every request for a key into one process that served them one at
a time. These tests pin the worker half of the contract at its real boundary,
the process command line: the flag is required, and the serving loop the
worker enters is the one the flag names.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from batchalign.worker._main import parse_worker_args
from batchalign.worker._types import WorkerServing

# The pool size the worker is launched with and must report back.
THREADS = 3


def test_a_launch_without_a_serving_mode_is_refused() -> None:
    """No silent default: a launcher that does not say how to serve is a bug."""
    with pytest.raises(SystemExit):
        parse_worker_args(["--profile", "gpu", "--lang", "eng"])


@pytest.mark.parametrize("serving", list(WorkerServing))
def test_the_worker_enters_the_serving_loop_its_launcher_names(
    serving: WorkerServing,
) -> None:
    """A real echo worker reports, at startup, the mode it was told to use.

    The GPU profile is the case that mattered: without CUDA the worker used to
    choose sequential serving whatever its launcher assumed.
    """
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "batchalign.worker",
            "--test-echo",
            "--profile",
            "gpu",
            "--serving",
            serving.value,
            "--gpu-thread-pool-size",
            str(THREADS),
            "--verbose",
            "1",
            "--supervisor-pid",
            str(os.getpid()),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Closing stdin ends either loop once it has started; `communicate` then
    # returns everything the worker wrote.
    stdout, stderr = proc.communicate(input="", timeout=60)
    assert proc.returncode == 0, stderr
    assert '"ready": true' in stdout, stdout
    assert f"serving={serving.value} threads={THREADS}" in stderr, stderr
