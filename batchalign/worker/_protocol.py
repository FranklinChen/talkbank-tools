"""JSON-lines IPC loops for the Python worker (stdio and TCP transports).

Stdio modes:

- ``_serve_stdio()`` — sequential request/response, one at a time. Used by
  Stanza and IO profile workers where requests are CPU-bound and GIL-limited.
- ``_serve_stdio_concurrent()`` — dispatches requests to a
  ``ThreadPoolExecutor``, enabling concurrent GPU inference. Used by GPU
  profile workers where PyTorch releases the GIL during computation, allowing
  real parallelism across threads sharing the same loaded models.

TCP modes (persistent daemon workers):

- ``_serve_tcp()`` — sequential request/response over a TCP socket.
- ``_serve_tcp_concurrent()`` — concurrent GPU dispatch over a TCP socket.

TCP workers listen on ``(host, port)``, accept one connection at a time (the
Rust server reconnects on drop), and use the same JSON-lines protocol as
stdio. The only difference is the transport — all dispatch logic is shared.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from batchalign.inference._domain_types import TcpPort

from batchalign.worker._protocol_ops import dispatch_protocol_message
from batchalign.worker._types import WorkerJSONValue

logger = logging.getLogger(__name__)

# Reentrant stdout lock shared between sequential and concurrent modes.
# In sequential mode it is never contended (single thread); in concurrent
# mode the main thread and worker threads both need it.
_stdout_lock = threading.Lock()


def _registry_ownership_from_env() -> tuple[str, str, int | None]:
    """Resolve how a TCP worker should describe its registry ownership."""
    server_instance_id = os.environ.get("BATCHALIGN_SERVER_INSTANCE_ID", "").strip()
    if not server_instance_id:
        return ("external", "", None)

    raw_server_pid = os.environ.get("BATCHALIGN_SERVER_PID", "").strip()
    try:
        server_pid = int(raw_server_pid) if raw_server_pid else None
    except ValueError:
        server_pid = None
    return ("server_owned", server_instance_id, server_pid)


def _write_json(payload: dict[str, WorkerJSONValue]) -> None:
    """Emit a single JSON message line to stdout."""
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def write_progress_event(
    request_id: str,
    completed: int,
    total: int,
    stage: str = "stanza_processing",
) -> None:
    """Emit a progress event line during a long-running V2 task.

    The Rust worker handle reads these intermediate JSON lines before the
    final response.  Progress events use the ``progress_v2`` op tag so
    the handle can distinguish them from the final ``execute_v2`` response.
    """
    _write_json({
        "op": "progress_v2",
        "event": {
            "request_id": request_id,
            "completed": completed,
            "total": total,
            "stage": stage,
        },
    })


def _write_error(message: str) -> None:
    """Emit protocol-level error response for malformed requests/ops."""
    _write_json({"op": "error", "error": message})


def _print_ready() -> None:
    """Print a JSON ready line to stdout so the Rust parent can discover us."""
    _write_json({"ready": True, "pid": os.getpid(), "transport": "stdio"})


def _serve_stdio() -> None:
    """Run the sequential stdio request loop until shutdown or EOF."""
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            _write_error(f"invalid JSON request: {exc}")
            continue

        dispatch = dispatch_protocol_message(message)
        _write_json(dispatch.payload)
        if dispatch.should_shutdown:
            break


def _serve_stdio_concurrent(max_threads: int = 4) -> None:
    """Run the concurrent stdio request loop for GPU profile workers.

    The main thread reads stdin sequentially and dispatches each request to a
    ``ThreadPoolExecutor``. GPU inference (PyTorch) releases the GIL during
    CUDA kernels, enabling real concurrent model execution across threads that
    share the same in-process model weights.

    Responses are written under ``_stdout_lock`` so JSON lines never interleave.
    """
    pool = ThreadPoolExecutor(max_workers=max_threads)
    shutdown_event = threading.Event()

    def _handle_and_respond(message: object) -> None:
        dispatch = dispatch_protocol_message(message)
        with _stdout_lock:
            _write_json(dispatch.payload)
        if dispatch.should_shutdown:
            shutdown_event.set()

    for raw_line in sys.stdin:
        if shutdown_event.is_set():
            break
        line = raw_line.strip()
        if not line:
            continue

        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            with _stdout_lock:
                _write_error(f"invalid JSON request: {exc}")
            continue

        pool.submit(_handle_and_respond, message)

    pool.shutdown(wait=True)


# ---------------------------------------------------------------------------
# TCP transport
# ---------------------------------------------------------------------------


def _print_ready_tcp(host: str, port: TcpPort) -> None:
    """Print a JSON ready line to stderr so the CLI launcher can detect startup.

    Unlike stdio mode where ready goes to stdout (consumed by the Rust parent),
    TCP mode prints to stderr since stdout is not connected to any parent pipe.
    """
    ready = json.dumps({
        "ready": True,
        "pid": os.getpid(),
        "transport": "tcp",
        "host": host,
        "port": port,
    })
    sys.stderr.write(ready + "\n")
    sys.stderr.flush()


def _handle_tcp_connection_sequential(
    conn: socket.socket,
    addr: tuple[str, int],
) -> None:
    """Handle one TCP connection with sequential request/response dispatch."""
    logger.info("TCP connection from %s:%d", addr[0], addr[1])
    rfile = conn.makefile("r", encoding="utf-8")
    wfile = conn.makefile("w", encoding="utf-8")

    try:
        for raw_line in rfile:
            line = raw_line.strip()
            if not line:
                continue

            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                error_payload = json.dumps({"op": "error", "error": f"invalid JSON request: {exc}"})
                wfile.write(error_payload + "\n")
                wfile.flush()
                continue

            dispatch = dispatch_protocol_message(message)
            wfile.write(json.dumps(dispatch.payload) + "\n")
            wfile.flush()
            if dispatch.should_shutdown:
                return
    except (BrokenPipeError, ConnectionResetError):
        logger.info("TCP connection closed by peer %s:%d", addr[0], addr[1])
    finally:
        rfile.close()
        wfile.close()
        conn.close()


def _handle_tcp_connection_concurrent(
    conn: socket.socket,
    addr: tuple[str, int],
    max_threads: int,
) -> None:
    """Handle one TCP connection with concurrent GPU dispatch."""
    logger.info("TCP connection from %s:%d (concurrent)", addr[0], addr[1])
    rfile = conn.makefile("r", encoding="utf-8")
    wfile = conn.makefile("w", encoding="utf-8")
    write_lock = threading.Lock()
    pool = ThreadPoolExecutor(max_workers=max_threads)
    shutdown_event = threading.Event()

    def _handle_and_respond(message: object) -> None:
        dispatch = dispatch_protocol_message(message)
        with write_lock:
            try:
                wfile.write(json.dumps(dispatch.payload) + "\n")
                wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                shutdown_event.set()
        if dispatch.should_shutdown:
            shutdown_event.set()

    try:
        for raw_line in rfile:
            if shutdown_event.is_set():
                break
            line = raw_line.strip()
            if not line:
                continue

            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                with write_lock:
                    error_payload = json.dumps(
                        {"op": "error", "error": f"invalid JSON request: {exc}"}
                    )
                    wfile.write(error_payload + "\n")
                    wfile.flush()
                continue

            pool.submit(_handle_and_respond, message)
    except (BrokenPipeError, ConnectionResetError):
        logger.info("TCP connection closed by peer %s:%d", addr[0], addr[1])
    finally:
        pool.shutdown(wait=True)
        rfile.close()
        wfile.close()
        conn.close()


def _serve_tcp(
    host: str,
    port: TcpPort,
    *,
    registry_path: Path | None = None,
) -> None:
    """Run the sequential TCP request loop for Stanza/IO profile workers.

    Listens on ``(host, port)``, accepts one connection at a time, and serves
    requests sequentially. When the connection closes (Rust server restarts or
    disconnects), the worker waits for a new connection — it persists across
    server restarts.

    Registers itself in ``workers.json`` on startup and removes itself on
    shutdown.
    """
    from batchalign.worker._registry import (
        WorkerRegistryEntry,
        register_worker,
        unregister_worker,
    )
    from batchalign.worker._types import _state

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(1)
    actual_port = server_sock.getsockname()[1]

    bootstrap = _state.bootstrap
    ownership, owner_server_instance_id, owner_server_pid = _registry_ownership_from_env()
    entry = WorkerRegistryEntry(
        pid=os.getpid(),
        host=host,
        port=actual_port,
        profile=bootstrap.profile.value if bootstrap and bootstrap.profile else "",
        lang=bootstrap.lang if bootstrap else "eng",
        engine_overrides=json.dumps(bootstrap.engine_overrides) if bootstrap and bootstrap.engine_overrides else "",
        ownership=ownership,
        owner_server_instance_id=owner_server_instance_id,
        owner_server_pid=owner_server_pid,
    )
    register_worker(entry, registry_path=registry_path)

    _print_ready_tcp(host, actual_port)
    logger.info("TCP worker listening on %s:%d (sequential)", host, actual_port)

    try:
        while True:
            conn, addr = server_sock.accept()
            _handle_tcp_connection_sequential(conn, addr)
            # After connection closes, loop back and accept next connection.
            # This is the key difference from stdio: worker survives server restart.
    except KeyboardInterrupt:
        logger.info("TCP worker shutting down (KeyboardInterrupt)")
    finally:
        server_sock.close()
        unregister_worker(host=host, port=actual_port, registry_path=registry_path)


def _serve_tcp_concurrent(
    host: str,
    port: TcpPort,
    max_threads: int = 4,
    *,
    registry_path: Path | None = None,
) -> None:
    """Run the concurrent TCP request loop for GPU profile workers.

    Same as ``_serve_tcp()`` but dispatches requests to a thread pool for
    concurrent GPU inference within each connection.
    """
    from batchalign.worker._registry import (
        WorkerRegistryEntry,
        register_worker,
        unregister_worker,
    )
    from batchalign.worker._types import _state

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(1)
    actual_port = server_sock.getsockname()[1]

    bootstrap = _state.bootstrap
    ownership, owner_server_instance_id, owner_server_pid = _registry_ownership_from_env()
    entry = WorkerRegistryEntry(
        pid=os.getpid(),
        host=host,
        port=actual_port,
        profile=bootstrap.profile.value if bootstrap and bootstrap.profile else "",
        lang=bootstrap.lang if bootstrap else "eng",
        engine_overrides=json.dumps(bootstrap.engine_overrides) if bootstrap and bootstrap.engine_overrides else "",
        ownership=ownership,
        owner_server_instance_id=owner_server_instance_id,
        owner_server_pid=owner_server_pid,
    )
    register_worker(entry, registry_path=registry_path)

    _print_ready_tcp(host, actual_port)
    logger.info("TCP worker listening on %s:%d (concurrent, %d threads)", host, actual_port, max_threads)

    try:
        while True:
            conn, addr = server_sock.accept()
            _handle_tcp_connection_concurrent(conn, addr, max_threads)
    except KeyboardInterrupt:
        logger.info("TCP worker shutting down (KeyboardInterrupt)")
    finally:
        server_sock.close()
        unregister_worker(host=host, port=actual_port, registry_path=registry_path)
