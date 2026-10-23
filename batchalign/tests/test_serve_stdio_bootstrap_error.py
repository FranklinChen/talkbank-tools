"""Tests that ``_serve_stdio`` catches dispatch exceptions and emits a
structured error response with the right ``kind`` discriminator.

Pre-fix behavior: an uncaught exception in the dispatch path killed the
Python worker (exit code 1). The Rust orchestrator saw ``ProcessExited`` →
``WorkerCrash`` → retryable, retried up to 3× with a full traceback per
attempt, and a single-host instance of this loop produced multi-GB
``server.log`` spam from a deterministic Stanza-catalog miss.

Current behavior, by the exception's class:

- A deterministic load failure (``UnsupportedLanguageError``) emits
  ``{"op": "error", "error": "...", "kind": "bootstrap"}``. The Rust side
  classifies it as terminal and surfaces the verbatim message.
- A download that failed in transit (``StanzaCatalogDownloadError``, a Hub or
  ``requests`` connection failure) emits ``kind: "download"``, which the Rust
  side retries.
- A ``torch.OutOfMemoryError`` emits ``kind: "out_of_memory"``, retried after
  a backoff.
- Anything else emits ``kind: "runtime"``.

The worker keeps serving after a ``runtime`` or ``out_of_memory`` report. It
stops after a ``bootstrap`` or ``download`` report (a model may be half
loaded), for exactly the kinds after which the Rust readers retire it
(``batchalign_core.error_kind_ends_worker``), so a retry never lands on the
exited process.
"""

from __future__ import annotations

import io
import json
import queue
import subprocess
import sys
import threading
from unittest import mock

import pytest

from batchalign.worker._protocol import (
    _classify_dispatch_exception,
    _serve_stdio,
)


@pytest.mark.parametrize(
    ("code", "message", "expected"),
    [
        (
            "from batchalign.worker._protocol import _serve_stdio_concurrent; "
            "_serve_stdio_concurrent(max_threads=2)",
            {"op": "shutdown"},
            {"op": "shutdown"},
        ),
        (
            "from batchalign.worker import _protocol\n"
            "from batchalign.worker._stanza_capabilities import StanzaCatalogDownloadError\n"
            "def fail(request):\n"
            "    raise StanzaCatalogDownloadError('offline test catalog')\n"
            "_protocol.dispatch_prepared_protocol_message = fail\n"
            "_protocol._serve_stdio_concurrent(max_threads=2)\n",
            {"op": "ensure_task", "request": {"task": "morphosyntax"}},
            {"op": "error", "error": "offline test catalog", "kind": "download"},
        ),
    ],
    ids=["shutdown", "download"],
)
def test_concurrent_stdio_exits_with_parent_stdin_still_open(
    code: str, message: dict[str, object], expected: dict[str, object]
) -> None:
    """A terminal reply must not leave the worker waiting for another line."""
    child = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    replies: queue.Queue[str] = queue.Queue()
    assert child.stdin is not None and child.stdout is not None
    stdout = child.stdout
    reader = threading.Thread(
        target=lambda: replies.put(stdout.readline()), daemon=True
    )
    reader.start()
    try:
        child.stdin.write(json.dumps(message) + "\n")
        child.stdin.flush()
        assert json.loads(replies.get(timeout=10)) == expected
        # Coarse hang check, not a performance benchmark. Keep stdin open.
        assert child.wait(timeout=5) == 0
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        reader.join(timeout=5)
        child.stdin.close()
        child.stdout.close()
        assert child.stderr is not None
        child.stderr.close()


def test_concurrent_stdio_eof_drains_queued_requests() -> None:
    """Input completion must not cancel requests already admitted to the pool."""
    count = 32
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "from batchalign.worker._protocol import _serve_stdio_concurrent; "
            "_serve_stdio_concurrent(max_threads=1)",
        ],
        input='{"op": "health"}\n' * count,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert child.returncode == 0, child.stderr
    replies = [json.loads(line) for line in child.stdout.splitlines()]
    assert len(replies) == count
    assert all(reply["op"] == "health" for reply in replies)


# ---------------------------------------------------------------------------
# Classifier: bootstrap-vs-runtime discriminator on raw exception types.
# ---------------------------------------------------------------------------


def test_classify_stanza_catalog_download_error_is_download():
    """RED FIRST: a catalog download that failed in transit was
    bootstrap-kind, the deterministic class, so the server never retried it
    and it withheld a same-engine UTR fallback that a later attempt could
    have served. It is its own kind now; the worker still exits after it."""
    from batchalign.worker._stanza_capabilities import (
        StanzaCatalogDownloadError,
    )

    exc = StanzaCatalogDownloadError("network unreachable")
    assert _classify_dispatch_exception(exc) == "download"


def test_classify_a_model_download_connection_failure_is_download():
    """A Hugging Face model download that loses its connection is the same
    in-transit failure."""
    from requests.exceptions import ConnectionError as RequestsConnectionError

    exc = RequestsConnectionError("Max retries exceeded with url")
    assert _classify_dispatch_exception(exc) == "download"


def test_classify_unsupported_language_error_is_bootstrap():
    """Genuinely-unsupported language is also a bootstrap-class failure.

    Reasoning: it's deterministic (the same language will be unsupported
    on every retry), and the user-facing remediation is the same actionable
    "this language isn't supported by your Stanza install" message.
    """
    from batchalign.worker._stanza_loading import UnsupportedLanguageError

    exc = UnsupportedLanguageError("language 'que' not supported")
    assert _classify_dispatch_exception(exc) == "bootstrap"


def test_classify_generic_runtime_error_is_runtime():
    """A bare exception (not a typed bootstrap error) classifies as runtime."""
    exc = RuntimeError("something went wrong mid-inference")
    assert _classify_dispatch_exception(exc) == "runtime"


def test_classify_value_error_is_runtime():
    """Validation failures (ValueError on a malformed input) are per-request."""
    exc = ValueError("invalid utterance payload")
    assert _classify_dispatch_exception(exc) == "runtime"


# ---------------------------------------------------------------------------
# _serve_stdio integration: dispatch exception must produce a structured wire
# response, not a process exit.
# ---------------------------------------------------------------------------


def _run_serve_stdio_with_one_message(message_json: str, dispatch_side_effect):
    """Drive ``_serve_stdio`` with a single input line and a mocked dispatch.

    Returns the captured stdout as a list of decoded JSON envelopes (one
    per emitted line).
    """
    fake_stdin = io.StringIO(message_json + "\n")
    fake_stdout = io.StringIO()
    fake_stderr = io.StringIO()

    with (
        mock.patch.object(sys, "stdin", fake_stdin),
        mock.patch.object(sys, "stdout", fake_stdout),
        mock.patch.object(sys, "stderr", fake_stderr),
        mock.patch(
            "batchalign.worker._protocol.dispatch_protocol_message",
            side_effect=dispatch_side_effect,
        ),
    ):
        _serve_stdio()

    lines = [line for line in fake_stdout.getvalue().splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def test_serve_stdio_emits_bootstrap_kind_on_typed_bootstrap_error():
    """A deterministic load failure produces a ``kind=bootstrap`` envelope.

    This is the canonical regression for the bootstrap-retry-classification
    fix. Pre-fix, this exception would propagate up and kill the worker;
    post-fix, the worker emits a structured error and the Rust classifier
    (separately tested in Rust) treats it as terminal.
    """
    from batchalign.worker._language_errors import UnsupportedLanguageError

    def fake_dispatch(_message):
        raise UnsupportedLanguageError("no Stanza pipeline for 'que'")

    envelopes = _run_serve_stdio_with_one_message(
        '{"op": "ensure_task", "request": {"task": "morphosyntax"}}',
        fake_dispatch,
    )

    assert len(envelopes) == 1, f"Expected exactly one error envelope; got: {envelopes}"
    env = envelopes[0]
    assert env["op"] == "error"
    assert env["kind"] == "bootstrap"
    assert "que" in env["error"]


def test_serve_stdio_emits_download_kind_for_a_catalog_download_in_transit():
    """A catalog download that failed in transit produces ``kind=download``,
    which the server retries, rather than the terminal bootstrap kind."""
    from batchalign.worker._stanza_capabilities import (
        StanzaCatalogDownloadError,
    )

    def fake_dispatch(_message):
        raise StanzaCatalogDownloadError(
            "Failed to download Stanza resource catalog from "
            "https://raw.githubusercontent.com/stanfordnlp/stanza-resources/main: "
            "connection refused"
        )

    envelopes = _run_serve_stdio_with_one_message(
        '{"op": "ensure_task", "request": {"task": "morphosyntax"}}',
        fake_dispatch,
    )

    assert len(envelopes) == 1, f"Expected exactly one error envelope; got: {envelopes}"
    env = envelopes[0]
    assert env["op"] == "error"
    assert env["kind"] == "download"
    assert "connection refused" in env["error"]


def test_serve_stdio_emits_runtime_kind_on_generic_error():
    """A non-bootstrap exception must default to ``kind=runtime``.

    Existing retry semantics for transient per-request failures
    (different inputs, transient resource state, external-API hiccups)
    rely on the runtime kind. We don't want to suddenly classify all
    handler exceptions as terminal.
    """

    def fake_dispatch(_message):
        raise RuntimeError("transient inference error")

    envelopes = _run_serve_stdio_with_one_message(
        '{"op": "infer", "request": {}}',
        fake_dispatch,
    )

    assert len(envelopes) == 1
    env = envelopes[0]
    assert env["op"] == "error"
    assert env["kind"] == "runtime"
    assert "transient inference error" in env["error"]


def test_serve_stdio_continues_after_runtime_error():
    """After a runtime-kind error, ``_serve_stdio`` must keep reading input.

    Pre-fix, an uncaught exception killed the loop (and the worker
    process). Post-fix, runtime errors should not abort the loop, only
    bootstrap errors do.
    """
    call_count = {"n": 0}

    def fake_dispatch(_message):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("first call fails")
        # Second call returns a valid envelope.
        from batchalign.worker._protocol_ops import ProtocolDispatchResult

        return ProtocolDispatchResult(payload={"op": "health", "ok": True})

    fake_stdin = io.StringIO('{"op": "infer", "request": {}}\n' + '{"op": "health"}\n')
    fake_stdout = io.StringIO()
    fake_stderr = io.StringIO()
    with (
        mock.patch.object(sys, "stdin", fake_stdin),
        mock.patch.object(sys, "stdout", fake_stdout),
        mock.patch.object(sys, "stderr", fake_stderr),
        mock.patch(
            "batchalign.worker._protocol.dispatch_protocol_message",
            side_effect=fake_dispatch,
        ),
    ):
        _serve_stdio()

    lines = [line for line in fake_stdout.getvalue().splitlines() if line.strip()]
    assert len(lines) == 2, (
        f"Both messages must produce envelopes; got {len(lines)}: {lines}"
    )
    env1 = json.loads(lines[0])
    env2 = json.loads(lines[1])
    assert env1["op"] == "error"
    assert env1["kind"] == "runtime"
    assert env2["op"] == "health"


def test_serve_stdio_exits_after_bootstrap_error():
    """A bootstrap-kind error must terminate the loop after emitting.

    A worker that hit a bootstrap failure is in a partially-initialized
    state: the safest move is to exit cleanly so the pool spawns a fresh
    worker. The orchestrator's terminal classification of bootstrap
    errors prevents retry-storm cascades from this exit.
    """
    call_count = {"n": 0}

    def fake_dispatch(_message):
        from batchalign.worker._stanza_loading import UnsupportedLanguageError

        call_count["n"] += 1
        raise UnsupportedLanguageError("que not supported")

    fake_stdin = io.StringIO(
        '{"op": "ensure_task", "request": {"task": "morphosyntax"}}\n'
        + '{"op": "infer", "request": {}}\n'  # Should never be reached.
    )
    fake_stdout = io.StringIO()
    fake_stderr = io.StringIO()
    with (
        mock.patch.object(sys, "stdin", fake_stdin),
        mock.patch.object(sys, "stdout", fake_stdout),
        mock.patch.object(sys, "stderr", fake_stderr),
        mock.patch(
            "batchalign.worker._protocol.dispatch_protocol_message",
            side_effect=fake_dispatch,
        ),
    ):
        _serve_stdio()

    assert call_count["n"] == 1, (
        f"Loop must exit after the first bootstrap error; got "
        f"{call_count['n']} dispatch calls"
    )
    lines = [line for line in fake_stdout.getvalue().splitlines() if line.strip()]
    assert len(lines) == 1
    env = json.loads(lines[0])
    assert env["op"] == "error"
    assert env["kind"] == "bootstrap"


# ---------------------------------------------------------------------------
# Correlation: an error raised while dispatching an execute_v2 message must
# name the request it belongs to.
# ---------------------------------------------------------------------------


def test_serve_stdio_error_names_the_execute_v2_request_it_belongs_to():
    """A dispatch exception must carry the failed request's ``request_id``.

    The Rust reader routes an ``op=error`` envelope by ``request_id``: tagged,
    it fails that dispatch's pending oneshot at once; untagged, it goes to the
    single-slot sequential control channel, where a V2 caller has no receiver
    registered and the envelope is absorbed. So an untagged error costs the
    caller its whole per-request timeout for a fault the worker diagnosed
    immediately, which is exactly what stalled ``speaker-identify`` on
    2026-09-02: the worker answered in two milliseconds and the caller waited
    out the audio-task ceiling.

    The dispatcher is handed the message, so the id is in its hand when the
    exception is caught; dropping it is the defect.
    """

    def fake_dispatch(_message):
        raise RuntimeError(
            "module 'batchalign_core' has no attribute "
            "'execute_speaker_embedding_request_v2'"
        )

    envelopes = _run_serve_stdio_with_one_message(
        json.dumps(
            {
                "op": "execute_v2",
                "request": {
                    "request_id": "speaker-embedding-v2-request-1",
                    "task": "speaker_embedding",
                },
            }
        ),
        fake_dispatch,
    )

    assert len(envelopes) == 1
    env = envelopes[0]
    assert env["op"] == "error"
    assert env["kind"] == "runtime"
    assert env["request_id"] == "speaker-embedding-v2-request-1"


def test_serve_stdio_error_for_a_message_with_no_request_id_stays_uncorrelated():
    """A sequential op carries no request id, and the envelope must not invent one.

    ``health`` / ``capabilities`` / ``ensure_task`` are answered through the
    control channel, which is where an untagged error belongs. Tagging one
    with a fabricated id would route it to a pending map that has no such
    entry, and the sequential caller would then wait forever.
    """

    def fake_dispatch(_message):
        raise RuntimeError("transient inference error")

    envelopes = _run_serve_stdio_with_one_message(
        '{"op": "health"}',
        fake_dispatch,
    )

    assert len(envelopes) == 1
    assert "request_id" not in envelopes[0]
