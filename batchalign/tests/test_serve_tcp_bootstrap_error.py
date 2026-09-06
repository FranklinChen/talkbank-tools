"""Tests that the TCP transport handles dispatch exceptions the same way
the stdio transport does.

The original bootstrap-retry-classification fix landed only on stdio
(`_serve_stdio`, `_serve_stdio_concurrent`). The TCP handlers, used by
Stanza/IO-profile workers, the same profiles that load Stanza catalogs
and were the original incident shape, still let exceptions propagate
up, killing the connection handler instead of emitting a structured
error.

Post-fix:

- ``_handle_tcp_connection_sequential``: catches exceptions, classifies,
  emits ``{"op": "error", "kind": ...}``, breaks the loop on bootstrap.
- ``_handle_tcp_connection_concurrent``: same, but sets
  ``shutdown_event`` on bootstrap so the read loop exits cleanly.
"""

from __future__ import annotations

import json
import socket
import threading
from contextlib import suppress

from batchalign.worker._protocol import (
    _handle_tcp_connection_concurrent,
    _handle_tcp_connection_sequential,
)

# ---------------------------------------------------------------------------
# Test scaffolding: a connected socket pair lets us drive the handler from
# one end and read its responses from the other, all in-process.
# ---------------------------------------------------------------------------


def _make_connected_pair():
    """Return ``(client, server)``: a connected loopback TCP socket pair.

    ``socket.socketpair()`` doesn't produce TCP sockets on every platform,
    so we set up a tiny listener instead. Both ends are real
    ``AF_INET/SOCK_STREAM`` sockets that the handler can use directly.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", port))
    server, _addr = listener.accept()
    listener.close()
    return client, server


def _drain_lines(sock: socket.socket, until_close: bool = True) -> list[str]:
    """Read all newline-terminated lines from ``sock`` until close.

    Used after the handler has shut down so we can collect every response
    line without racing the writer.
    """
    sock.settimeout(2.0)
    rfile = sock.makefile("r", encoding="utf-8")
    out = []
    try:
        for line in rfile:
            line = line.strip()
            if line:
                out.append(line)
    except TimeoutError:
        # Treat a timeout as end-of-stream for the test; the handler
        # should have closed by now.
        pass
    finally:
        rfile.close()
    return out


def test_tcp_concurrent_shutdown_closes_without_client_eof() -> None:
    client, server = _make_connected_pair()
    handler = threading.Thread(
        target=_handle_tcp_connection_concurrent,
        args=(server, ("127.0.0.1", 0), 2),
    )
    handler.start()
    client.settimeout(5)
    reader = client.makefile("r", encoding="utf-8")
    try:
        client.sendall(b'{"op":"shutdown"}\n')
        assert json.loads(reader.readline()) == {"op": "shutdown"}
        assert reader.readline() == "", (
            "server must close while client write side stays open"
        )
        handler.join(timeout=5)
        assert not handler.is_alive()
    finally:
        with suppress(OSError):
            client.shutdown(socket.SHUT_RDWR)
        reader.close()
        client.close()
        handler.join(timeout=5)


# ---------------------------------------------------------------------------
# Sequential transport.
# ---------------------------------------------------------------------------


def test_tcp_sequential_emits_bootstrap_kind(monkeypatch):
    """A typed bootstrap exception over TCP must emit ``kind=bootstrap``.

    Same contract as the stdio variant: the handler catches the exception,
    classifies it, writes a structured error envelope with the kind
    discriminator, and tears the connection down. The orchestrator on
    the other end gets a typed error rather than a closed socket.
    """
    from batchalign.worker._stanza_capabilities import (
        StanzaCatalogDownloadError,
    )

    def fake_dispatch(_message):
        raise StanzaCatalogDownloadError("network unreachable")

    monkeypatch.setattr(
        "batchalign.worker._protocol.dispatch_protocol_message",
        fake_dispatch,
    )

    client, server = _make_connected_pair()

    # Run the handler in a background thread so we can drive it from the
    # client side. The handler reads from ``server`` and writes back.
    t = threading.Thread(
        target=_handle_tcp_connection_sequential,
        args=(server, ("127.0.0.1", 0)),
    )
    t.start()

    client.sendall(b'{"op": "ensure_task", "request": {"task": "morphosyntax"}}\n')

    # Sequential mode breaks the loop on bootstrap; the connection should
    # close after one response without us sending a second message.
    lines = _drain_lines(client)
    client.close()
    t.join(timeout=3.0)

    assert lines, "Handler must emit at least one response, not a silent close"
    env = json.loads(lines[0])
    assert env["op"] == "error", f"Expected error op, got {env}"
    assert env["kind"] == "bootstrap", f"Expected kind=bootstrap, got {env}"
    assert "network unreachable" in env["error"]


def test_tcp_sequential_emits_runtime_kind_on_generic_exception(monkeypatch):
    """Non-bootstrap exceptions over TCP must emit ``kind=runtime``.

    And the loop must keep reading the next request, mirroring the stdio
    semantics: only bootstrap errors abort the connection.
    """
    call_count = {"n": 0}

    def fake_dispatch(_message):
        from batchalign.worker._protocol_ops import ProtocolDispatchResult

        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("transient inference error")
        return ProtocolDispatchResult(payload={"op": "health", "ok": True})

    monkeypatch.setattr(
        "batchalign.worker._protocol.dispatch_protocol_message",
        fake_dispatch,
    )

    client, server = _make_connected_pair()
    t = threading.Thread(
        target=_handle_tcp_connection_sequential,
        args=(server, ("127.0.0.1", 0)),
    )
    t.start()

    client.sendall(b'{"op": "infer", "request": {}}\n')
    client.sendall(b'{"op": "health"}\n')
    client.shutdown(socket.SHUT_WR)  # signal EOF to the handler

    lines = _drain_lines(client)
    client.close()
    t.join(timeout=3.0)

    assert len(lines) == 2, (
        f"Both messages must produce envelopes; got {len(lines)}: {lines}"
    )
    env1 = json.loads(lines[0])
    env2 = json.loads(lines[1])
    assert env1["op"] == "error"
    assert env1["kind"] == "runtime"
    assert env2["op"] == "health"


# ---------------------------------------------------------------------------
# Concurrent transport.
# ---------------------------------------------------------------------------


def test_tcp_concurrent_emits_bootstrap_kind(monkeypatch):
    """Concurrent TCP variant must produce the same bootstrap envelope.

    The shared-GPU profile uses the concurrent handler. Bootstrap errors
    here must surface identically: the orchestrator can't distinguish
    transports, so behaviour must match.
    """
    from batchalign.worker._stanza_capabilities import (
        StanzaCatalogDownloadError,
    )

    def fake_dispatch(_message):
        raise StanzaCatalogDownloadError("network unreachable")

    monkeypatch.setattr(
        "batchalign.worker._protocol.dispatch_prepared_protocol_message",
        fake_dispatch,
    )

    client, server = _make_connected_pair()
    t = threading.Thread(
        target=_handle_tcp_connection_concurrent,
        args=(server, ("127.0.0.1", 0), 2),
    )
    t.start()

    client.sendall(b'{"op": "ensure_task", "request": {"task": "morphosyntax"}}\n')
    lines = _drain_lines(client)
    t.join(timeout=1.0)
    closed_without_client_eof = not t.is_alive()
    client.close()
    t.join(timeout=3.0)

    assert closed_without_client_eof, (
        "bootstrap failure must wake the blocked TCP reader"
    )
    assert lines, "Handler must emit at least one response"
    # Concurrent mode dispatches via a thread pool, so order is loose; the
    # only invariant we assert is that AT LEAST ONE bootstrap envelope is
    # produced before the handler tears down.
    bootstrap_envs = [
        json.loads(line)
        for line in lines
        if '"kind": "bootstrap"' in line or '"kind":"bootstrap"' in line
    ]
    assert bootstrap_envs, f"No bootstrap envelope found among responses: {lines}"
    env = bootstrap_envs[0]
    assert env["op"] == "error"
    assert env["kind"] == "bootstrap"


def test_tcp_sequential_error_names_the_execute_v2_request_it_belongs_to(monkeypatch):
    """The TCP transport reaches the same reader loop, so it owes the same tag.

    A GPU worker discovered over TCP and one spawned over stdio share
    ``reader_loop_generic`` on the Rust side, which routes an ``op=error`` by
    ``request_id``. Fixing only the stdio emitter would have left the TCP
    worker stalling its callers in exactly the way the stdio one used to.
    """

    def fake_dispatch(_message):
        raise RuntimeError("speaker embedding host is not loaded")

    monkeypatch.setattr(
        "batchalign.worker._protocol.dispatch_protocol_message",
        fake_dispatch,
    )

    client, server = _make_connected_pair()
    t = threading.Thread(
        target=_handle_tcp_connection_sequential,
        args=(server, ("127.0.0.1", 0)),
    )
    t.start()

    client.sendall(
        json.dumps(
            {
                "op": "execute_v2",
                "request": {
                    "request_id": "speaker-embedding-v2-request-4",
                    "task": "speaker_embedding",
                },
            }
        ).encode()
        + b"\n"
    )
    client.shutdown(socket.SHUT_WR)

    lines = _drain_lines(client)
    client.close()
    t.join(timeout=3.0)

    assert len(lines) == 1, f"expected one error envelope, got: {lines}"
    env = json.loads(lines[0])
    assert env["op"] == "error"
    assert env["kind"] == "runtime"
    assert env["request_id"] == "speaker-embedding-v2-request-4"
