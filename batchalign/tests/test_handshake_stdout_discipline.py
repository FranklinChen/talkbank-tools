"""Regression: progress events use stdout consistently across the
pre-ready and post-ready windows.

The Rust supervisor's ``read_ready_line``
(``crates/batchalign/src/worker/handle/lifecycle.rs``) accepts
``{"op": "progress_v2", ...}`` lines as bootstrap-time preamble before
the ``{"ready": true, ...}`` envelope, emitting each as
``tracing::info!``. Stderr is buffered until process exit, so
bootstrap-time visibility requires the stdout path. Routing pre-ready
progress events to stderr would break that visibility contract.

(Earlier supervisor versions strictly read one JSON line as the ready
signal and rejected anything else, this test pinned a stderr-routing
contract for that older protocol. The 2026-05-06 supervisor change
relaxed the contract; this test was rewritten to track the new
behavior.)

Every protocol line, before and after the ready line, goes through
``_write_json``.
"""

from __future__ import annotations

import io
import json
import sys
from unittest import mock

import pytest

from batchalign.worker import _protocol
from batchalign.worker._runtime_identity import (
    Sha256Digest,
    _hash_package_tree,
    observe_worker_runtime,
)
from batchalign.worker._types_v2 import PhaseEdgeV2, PhaseV2, UnitProgressV2


def test_pre_ready_progress_event_goes_to_stdout_as_preamble():
    """Before ready, ``write_load_progress`` emits a JSON line on stdout.

    The supervisor's ``read_ready_line`` accepts ``progress_v2`` lines
    as bootstrap-time preamble (one or more lines before the ready
    envelope) and forwards each as ``tracing::info!``. Stdout is the
    visibility channel during bootstrap; stderr is buffered until exit.
    """
    fake_stdout = io.StringIO()
    fake_stderr = io.StringIO()
    with (
        mock.patch.object(sys, "stdout", fake_stdout),
        mock.patch.object(sys, "stderr", fake_stderr),
    ):
        _protocol.write_load_progress(
            "downloading_stanza_catalog", PhaseV2(edge=PhaseEdgeV2.BEGIN)
        )

    line = fake_stdout.getvalue().strip()
    envelope = json.loads(line)
    assert envelope["op"] == "progress_v2", (
        "Pre-ready progress events should be emitted as a single "
        f"progress_v2 JSON line on stdout. Got: {line!r}"
    )
    assert envelope["event"] == {
        "scope": "load",
        "stage": "downloading_stanza_catalog",
        "measure": {"kind": "phase", "edge": "begin"},
    }
    assert fake_stderr.getvalue() == "", (
        "Stderr is buffered until process exit; pre-ready events must "
        "use the stdout preamble path. Got stderr: "
        f"{fake_stderr.getvalue()!r}"
    )


def test_print_ready_writes_ready_envelope():
    """``_print_ready(runtime)`` writes the ready envelope to stdout."""
    fake_stdout = io.StringIO()
    fake_stderr = io.StringIO()
    with (
        mock.patch.object(sys, "stdout", fake_stdout),
        mock.patch.object(sys, "stderr", fake_stderr),
    ):
        _protocol._print_ready(observe_worker_runtime())

    line = fake_stdout.getvalue().strip()
    envelope = json.loads(line)
    assert envelope["ready"] is True
    assert envelope["transport"] == "stdio"
    assert "pid" in envelope
    runtime = envelope["runtime"]
    assert set(runtime) == {
        "schema_version",
        "python_version",
        "python_executable_sha256",
        "batchalign_package_tree_sha256",
        "batchalign_core_extension_sha256",
        "distribution_inventory_sha256",
    }
    assert runtime["schema_version"] == 1
    assert all(
        len(runtime[field]) == 64
        for field in [
            "python_executable_sha256",
            "batchalign_package_tree_sha256",
            "batchalign_core_extension_sha256",
            "distribution_inventory_sha256",
        ]
    )
    assert not any("path" in field for field in runtime)


def test_runtime_digest_rejects_unvalidated_constructor_text():
    """Even direct construction cannot create a false digest proof."""
    for invalid in ["short", "A" * 64, "z" * 64]:
        with pytest.raises(ValueError, match="lowercase SHA-256"):
            Sha256Digest(invalid)


def test_runtime_package_identity_excludes_test_transients_but_tracks_runtime_code(
    tmp_path,
):
    """Parallel-test scratch cannot perturb the executing package identity."""
    package = tmp_path / "batchalign"
    tests = package / "tests"
    tests.mkdir(parents=True)
    runtime = package / "worker.py"
    transient = tests / "_xdist_transient.py"
    runtime.write_text("RUNTIME = 1\n")
    transient.write_text("attempt one\n")

    baseline = _hash_package_tree(package)
    transient.write_text("attempt two\n")
    without_test_churn = _hash_package_tree(package)
    runtime.write_text("RUNTIME = 2\n")
    with_runtime_change = _hash_package_tree(package)

    assert without_test_churn == baseline
    assert with_runtime_change != baseline


def test_runtime_package_identity_delimits_file_contents(tmp_path):
    """A file body cannot impersonate another file's framed contribution."""
    one_file = tmp_path / "one_file"
    two_files = tmp_path / "two_files"
    one_file.mkdir()
    two_files.mkdir()

    # Without a content frame, both trees contribute the byte stream
    # len("a"), "a", len("b"), "b" to the outer digest.
    (one_file / "a").write_bytes((1).to_bytes(8, "big") + b"b")
    (two_files / "a").write_bytes(b"")
    (two_files / "b").write_bytes(b"")

    assert _hash_package_tree(one_file) != _hash_package_tree(two_files)


def test_post_ready_progress_event_goes_to_stdout():
    """After ``_print_ready()``, progress events use the same stdout path.

    This is the contract that lets the runner's ``spawn_progress_forwarder``
    multiplex progress events into the per-job status sink.
    """
    fake_stdout = io.StringIO()
    fake_stderr = io.StringIO()
    with (
        mock.patch.object(sys, "stdout", fake_stdout),
        mock.patch.object(sys, "stderr", fake_stderr),
    ):
        _protocol.write_request_progress(
            "req-7", "stanza_processing", UnitProgressV2(completed=3, total=10)
        )

    line = fake_stdout.getvalue().strip()
    envelope = json.loads(line)
    assert envelope["op"] == "progress_v2"
    assert envelope["event"] == {
        "scope": "request",
        "request_id": "req-7",
        "stage": "stanza_processing",
        "progress": {"completed": 3, "total": 10},
    }
    # The event should NOT also be logged to stderr; that would double-
    # report it.
    assert fake_stderr.getvalue() == ""


def test_a_tcp_daemon_reports_its_startup_on_the_protocol_stream_then_ends_it():
    """A TCP daemon's startup is a stdio worker's: progress, then the ready
    line, on the stdout it claimed, which its spawner reads with the same
    reader (``read_ready_line``) under the same stall window.

    The ready line names the transport, where the daemon listens, and the
    runtime identity a stdio worker's carries. After it the stream ends: the
    spawner stops reading and the daemon outlives it, so a later protocol line
    (a request's progress) must not reach the pipe. Run in a child process,
    since the claim rewires the process's descriptors.
    """
    import subprocess

    script = (
        "from batchalign.worker import _protocol\n"
        "from batchalign.worker._runtime_identity import observe_worker_runtime\n"
        "from batchalign.worker._types_v2 import PhaseEdgeV2, PhaseV2, UnitProgressV2\n"
        "_protocol.claim_protocol_stdout()\n"
        "_protocol.write_load_progress('loading_task_fa_eng', PhaseV2(edge=PhaseEdgeV2.BEGIN))\n"
        "_protocol._print_ready_tcp('127.0.0.1', 9100, observe_worker_runtime())\n"
        "_protocol.write_request_progress('req-1', 'aligning', UnitProgressV2(completed=1, total=2))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(lines) == 2, result.stdout
    progress, ready = lines
    assert progress["op"] == "progress_v2"
    assert progress["event"]["scope"] == "load"
    assert ready["ready"] is True
    assert ready["transport"] == "tcp"
    assert ready["host"] == "127.0.0.1"
    assert ready["port"] == 9100
    assert ready["runtime"]["schema_version"] == 1
    assert "op" not in ready, "the ready line is the one untagged line"


def test_a_real_tcp_daemon_prints_its_ready_line_on_stdout(tmp_path):
    """The real daemon entry point: an echo TCP daemon's first protocol line
    on stdout is its ready line, naming its port, and nothing of its
    handshake goes to stderr. It registers in a private state directory and
    is stopped once ready.
    """
    import os
    import subprocess

    env = dict(os.environ, BATCHALIGN_STATE_DIR=str(tmp_path))
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "batchalign.worker",
            "--transport",
            "tcp",
            "--test-echo",
            "--profile",
            "stanza",
            "--lang",
            "eng",
            "--host",
            "127.0.0.1",
            "--serving",
            "sequential",
            "--gpu-thread-pool-size",
            "1",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert proc.stdout is not None
        line = proc.stdout.readline()
        ready = json.loads(line)
        assert ready["ready"] is True, line
        assert ready["transport"] == "tcp"
        assert isinstance(ready["port"], int) and ready["port"] > 0
        assert ready["pid"] == proc.pid
        assert "runtime" in ready
    finally:
        proc.terminate()
        _, stderr = proc.communicate(timeout=30)
    assert '"ready"' not in stderr, stderr


def test_claimed_protocol_stdout_keeps_library_output_off_the_protocol_stream():
    """After ``claim_protocol_stdout()``, only protocol lines reach stdout.

    A library's ``print`` and a C-level write to descriptor 1 land on
    stderr; ``_write_json`` still reaches the pipe the Rust server reads.
    Run in a child process, since the claim rewires the process's
    descriptors.
    """
    import subprocess

    script = (
        "import os, sys\n"
        "from batchalign.worker import _protocol\n"
        "_protocol.claim_protocol_stdout()\n"
        "print('library banner')\n"
        "sys.stdout.flush()\n"
        "os.write(1, b'c-level write\\n')\n"
        "_protocol._write_json({'op': 'shutdown'})\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == ['{"op": "shutdown"}']
    assert "library banner" in result.stderr
    assert "c-level write" in result.stderr
