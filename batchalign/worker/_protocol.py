"""JSON-lines IPC loops for the Python worker (stdio and TCP transports).

Stdio modes:

- ``_serve_stdio()``: sequential request/response, one at a time. Used by
  Stanza and IO profile workers where requests are CPU-bound and GIL-limited.
- ``_serve_stdio_concurrent()``: dispatches requests to a
  ``ThreadPoolExecutor``, enabling concurrent GPU inference. Used by GPU
  profile workers where PyTorch releases the GIL during computation, allowing
  real parallelism across threads sharing the same loaded models.

TCP modes (persistent daemon workers):

- ``_serve_tcp()``: sequential request/response over a TCP socket.
- ``_serve_tcp_concurrent()``: concurrent GPU dispatch over a TCP socket.

TCP workers listen on ``(host, port)``, accept one connection at a time (the
Rust server reconnects on drop), and use the same JSON-lines protocol as
stdio. The only difference is the transport, all dispatch logic is shared.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TextIO, cast

import batchalign_core

if TYPE_CHECKING:
    # What an ``{"op": "error"}`` line reports, and so what a retry could
    # change: ``runtime`` (the admitted work failed; another attempt may
    # succeed), ``bootstrap`` (a deterministic model-load, catalog or import
    # failure) or ``invalid_request`` (the request was refused as sent).
    # Required on the wire. Spelled once, in the extension's stub, against
    # Rust's own list (``batchalign_core.WORKER_ERROR_KINDS``).
    from batchalign.inference._domain_types import TcpPort
    from batchalign.worker._registry import WorkerRegistryEntry
    from batchalign_core import WorkerErrorKind

from batchalign.worker._protocol_ops import (
    PendingProtocolRequest,
    ProtocolDispatchResult,
    dispatch_prepared_protocol_message,
    dispatch_protocol_message,
    prepare_protocol_message,
)
from batchalign.worker._runtime_identity import observe_worker_runtime
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


def _build_identity_from_env() -> str | None:
    """The spawning server's build identity, or ``None`` when none was given.

    The Rust server sets ``BATCHALIGN_BUILD_IDENTITY`` on every daemon it
    spawns and refuses to adopt a registry daemon whose identity differs from
    its own, since a daemon from another build may speak a different wire
    contract. The value is kept byte for byte; an unset or blank variable is
    ``None``, a daemon whose build nobody vouched for.
    """
    value = os.environ.get("BATCHALIGN_BUILD_IDENTITY")
    if value is None or not value.strip():
        return None
    return value


def _registry_entry(host: str, port: TcpPort) -> WorkerRegistryEntry:
    """The registry entry this TCP daemon advertises itself under.

    One constructor for both serve loops, which used to spell the same
    eleven-line construction twice.
    """
    from batchalign.worker._registry import WorkerRegistryEntry
    from batchalign.worker._types import _state

    bootstrap = _state.bootstrap
    ownership, owner_server_instance_id, owner_server_pid = (
        _registry_ownership_from_env()
    )
    return WorkerRegistryEntry(
        pid=os.getpid(),
        host=host,
        port=port,
        profile=bootstrap.profile.value if bootstrap and bootstrap.profile else "",
        lang=bootstrap.lang if bootstrap else "eng",
        build_identity=_build_identity_from_env(),
        engine_overrides=json.dumps(bootstrap.engine_overrides)
        if bootstrap and bootstrap.engine_overrides
        else "",
        ownership=ownership,
        owner_server_instance_id=owner_server_instance_id,
        owner_server_pid=owner_server_pid,
    )


# The protocol stream before the ready line. The Rust supervisor's
# ``read_ready_line`` accepts ``{"op": "progress_v2", ...}`` lines as
# bootstrap-time preamble before the ``{"ready": true, ...}`` envelope and
# logs each as ``tracing::info!``, so a progress event takes the same route
# (``_write_json``) before and after the ready line: bootstrap timings reach
# the daemon log live, where stderr is buffered until process exit.


# Where protocol lines go. ``None`` until :func:`claim_protocol_stdout` runs:
# the protocol is then ``sys.stdout`` itself, as in tests that capture it.
# A stdio worker claims it at startup, after which this is a private
# descriptor onto the pipe the Rust server reads, and nothing else in the
# process can write to that pipe.
_protocol_out: TextIO | None = None


def claim_protocol_stdout() -> None:
    """Give the protocol the process's stdout and give everyone else stderr.

    The Rust server reads protocol lines from this process's stdout. Any
    library that prints (a model download banner, a C extension writing to
    file descriptor 1) would otherwise put text into that stream, and enough
    of it makes the server stop trusting the stream and retire the worker.
    So the pipe is duplicated onto a private descriptor that only
    :func:`_write_json` writes to, and descriptor 1 and ``sys.stdout`` are
    pointed at stderr, where such output is only logged. Called once, at
    startup, before any model loads; a second call does nothing.
    """
    global _protocol_out
    if _protocol_out is not None:
        return
    sys.stdout.flush()
    stdout_fd = sys.stdout.fileno()
    private_fd = os.dup(stdout_fd)
    os.dup2(sys.stderr.fileno(), stdout_fd)
    _protocol_out = os.fdopen(private_fd, "w", encoding="utf-8")
    sys.stdout = sys.stderr


def _protocol_stream() -> TextIO:
    """The stream protocol lines are written to."""
    return sys.stdout if _protocol_out is None else _protocol_out


def _write_json(payload: dict[str, WorkerJSONValue]) -> None:
    """Emit a single JSON message line on the protocol stream."""
    out = _protocol_stream()
    out.write(json.dumps(payload) + "\n")
    out.flush()


def write_progress_event(
    request_id: str,
    completed: int,
    total: int,
    stage: str = "stanza_processing",
) -> None:
    """Emit a progress event line during a long-running V2 task.

    The Rust worker handle reads these intermediate JSON lines before the
    final response. Progress events use the ``progress_v2`` op tag so
    the handle can distinguish them from the final ``execute_v2`` response.
    Before the ready line they are bootstrap preamble, which the Rust
    supervisor's ``read_ready_line`` logs; the line is the same either way.
    """

    payload: dict[str, WorkerJSONValue] = {
        "op": "progress_v2",
        "event": {
            "request_id": request_id,
            "completed": completed,
            "total": total,
            "stage": stage,
        },
    }
    _write_json(payload)


@dataclass(frozen=True, slots=True)
class ErrorCorrelation:
    """Which request an error envelope belongs to.

    The Rust reader routes ``{"op": "error"}`` by ``request_id``. Tagged, it
    fails that dispatch's pending oneshot at once. Untagged, it goes to the
    single-slot sequential control channel, where a V2 caller has registered no
    receiver, so the envelope is absorbed and the caller waits out its whole
    per-request timeout (1800 s for an audio task by default) for a fault the
    worker diagnosed in milliseconds. That is the 2026-09-02 ``speaker-identify``
    stall, and the correlation was in the dispatcher's hand the whole time.

    So it is a REQUIRED argument of :func:`error_envelope` rather than an
    optional field of the dict: an emitter has to say which of the two cases it
    is in, and neither spelling is the shorter one.
    """

    request_id: str | None

    @classmethod
    def uncorrelated(cls) -> ErrorCorrelation:
        """No request owns this error, and that is a fact, not an omission.

        Two real cases: the line never parsed, so there is no message to read
        an id out of; and the failing op is one whose request carries no id
        (health). A ``capabilities`` or ``ensure_task`` request carries a
        control request id, and the Rust reader routes a failure tagged with
        it to the control op that sent it, so a late failure of an op that
        timed out cannot answer the next one.
        """
        return cls(request_id=None)

    @classmethod
    def of_message(cls, message: object) -> ErrorCorrelation:
        """Read the correlation out of the message the dispatcher was handed.

        The id lives on the request body (``{"op": ..., "request": {"request_id":
        ...}}``), which is the same place the Rust-owned dispatcher reads it
        from when it refuses a payload; see
        ``crates/batchalign-pyo3/src/worker_protocol.rs::extract_request_id``.
        A message that carries none is uncorrelated, which a health probe
        legitimately is.
        """
        if not isinstance(message, dict):
            return cls.uncorrelated()
        request = message.get("request")
        if not isinstance(request, dict):
            return cls.uncorrelated()
        request_id = request.get("request_id")
        if isinstance(request_id, str) and request_id:
            return cls(request_id=request_id)
        return cls.uncorrelated()


def error_envelope(
    message: str,
    kind: WorkerErrorKind,
    correlation: ErrorCorrelation,
) -> dict[str, WorkerJSONValue]:
    """Build an ``{"op": "error"}`` envelope through the one Rust builder.

    ``batchalign_core.error_envelope`` is the same function the Rust-owned
    request dispatcher uses for its own refusals, so the Python request loops
    and the dispatcher cannot write two shapes of this line.
    """
    return cast(
        dict[str, WorkerJSONValue],
        batchalign_core.error_envelope(message, kind, correlation.request_id),
    )


def _write_error(
    message: str,
    *,
    correlation: ErrorCorrelation,
    kind: WorkerErrorKind,
) -> None:
    """Emit a protocol-level error response.

    ``kind`` is required on the wire (see :data:`WorkerErrorKind`); the Rust
    readers refuse an envelope without it.
    """
    _write_json(error_envelope(message, kind, correlation))


def _classify_dispatch_exception(
    exc: BaseException,
) -> Literal["runtime", "bootstrap"]:
    """Return ``"bootstrap"`` for typed bootstrap-class exceptions, else ``"runtime"``.

    The set of bootstrap-class exception types is small and stable
    every typed error raised by a model-load / catalog-download path
    inherits from one of the named classes here. New bootstrap-class
    error types added to the worker must extend this match.

    Side-effect-free: takes the exception, returns the wire kind.
    Imports the type modules lazily so a missing optional dep can't crash
    the classifier itself.
    """
    bootstrap_types: list[type[BaseException]] = []
    try:
        from batchalign.worker._stanza_capabilities import (
            StanzaCatalogDownloadError,
        )

        bootstrap_types.append(StanzaCatalogDownloadError)
    except ImportError:
        pass
    try:
        from batchalign.worker._stanza_loading import UnsupportedLanguageError

        bootstrap_types.append(UnsupportedLanguageError)
    except ImportError:
        pass

    return "bootstrap" if isinstance(exc, tuple(bootstrap_types)) else "runtime"


def _print_ready() -> None:
    """Print a JSON ready line on the protocol stream so the Rust parent can
    discover us."""
    _write_json(
        {
            "ready": True,
            "pid": os.getpid(),
            "transport": "stdio",
            "runtime": observe_worker_runtime().json_value(),
        }
    )


def _serve_stdio() -> None:
    """Run the sequential stdio request loop until shutdown or EOF.

    Per-request exceptions are caught and converted to structured error
    responses on the wire (see ``_write_error``). The worker stays alive;
    the orchestrator decides retry policy from the wire ``kind`` field.

    Pre-2026-05-06 behavior: an uncaught exception in a handler killed the
    Python process (exit code 1). The Rust orchestrator saw
    ``ProcessExited`` → ``WorkerCrash`` → retryable, and retried 3× with a
    full traceback to ``server.log`` per attempt. For deterministic
    bootstrap failures (e.g., a missing Stanza catalog), this could
    generate hundreds of GB of log spam in a single day on a long-running
    daemon: the orchestrator never broke out of the retry loop.
    """
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            _write_error(
                f"invalid JSON request: {exc}",
                correlation=ErrorCorrelation.uncorrelated(),
                kind="invalid_request",
            )
            continue

        try:
            dispatch = dispatch_protocol_message(message)
        # A broad catch is the point here: no dispatch failure may kill the
        # worker loop.
        except BaseException as exc:
            kind = _classify_dispatch_exception(exc)
            # Log full traceback once for diagnostics; emit a single
            # structured error line back to the orchestrator. Do NOT let
            # the exception kill the worker.
            import traceback

            sys.stderr.write(
                f"--- worker dispatch exception ({kind}) ---\n" + traceback.format_exc()
            )
            sys.stderr.flush()
            _write_error(
                str(exc) or exc.__class__.__name__,
                correlation=ErrorCorrelation.of_message(message),
                kind=kind,
            )
            # Bootstrap-class failures may have left the worker in a
            # partially-initialized state; safest to exit so the pool
            # tears the worker down. The Rust side classifies the typed
            # error as non-retryable, so this exit does NOT produce a
            # retry storm: unlike the pre-fix path that retried every
            # worker exit-1 as a transient crash.
            if kind == "bootstrap":
                break
            continue

        _write_json(dispatch.payload)
        if dispatch.should_shutdown:
            break


def _serve_stdio_concurrent(max_threads: int = 4) -> None:
    """Run the concurrent stdio request loop for GPU profile workers.

    The main thread receives stdin lines from an interruptible native mailbox
    and dispatches each request to a ``ThreadPoolExecutor``. GPU inference (PyTorch) releases the GIL during
    CUDA kernels, enabling real concurrent model execution across threads that
    share the same in-process model weights.

    Responses are written under ``_stdout_lock`` so JSON lines never interleave.
    """
    from batchalign_core import open_protocol_stdin

    reader = open_protocol_stdin()

    def _respond(payload: dict[str, WorkerJSONValue]) -> None:
        try:
            with _stdout_lock:
                _write_json(payload)
        except (BrokenPipeError, ConnectionResetError):
            reader.stop()

    def _handle_and_respond(request: PendingProtocolRequest) -> None:
        # Terminal failures cancel queued work; EOF still drains admitted work.
        if reader.stopped:
            return
        try:
            dispatch = dispatch_prepared_protocol_message(request)
        except BaseException as exc:
            kind = _classify_dispatch_exception(exc)
            import traceback

            sys.stderr.write(
                f"--- worker dispatch exception ({kind}) ---\n" + traceback.format_exc()
            )
            sys.stderr.flush()
            _respond(
                error_envelope(
                    str(exc) or exc.__class__.__name__,
                    kind,
                    ErrorCorrelation.of_message(request.message),
                )
            )
            if kind == "bootstrap":
                reader.stop()
            return

        _respond(dispatch.payload)

    try:
        with ThreadPoolExecutor(max_workers=max_threads) as pool:
            for raw_line in iter(reader.read_line, None):
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    message = json.loads(line)
                except json.JSONDecodeError as exc:
                    _respond(
                        error_envelope(
                            f"invalid JSON request: {exc}",
                            "invalid_request",
                            ErrorCorrelation.uncorrelated(),
                        )
                    )
                    continue

                prepared = prepare_protocol_message(message)
                if isinstance(prepared, ProtocolDispatchResult):
                    _respond(prepared.payload)
                    if prepared.should_shutdown:
                        reader.stop()
                        break
                    continue
                pool.submit(_handle_and_respond, prepared)
    finally:
        reader.stop()


# ---------------------------------------------------------------------------
# TCP transport
# ---------------------------------------------------------------------------


def _print_ready_tcp(host: str, port: TcpPort) -> None:
    """Print a JSON ready line to stderr so the CLI launcher can detect startup.

    Unlike stdio mode where ready goes to stdout (consumed by the Rust parent),
    TCP mode prints to stderr since stdout is not connected to any parent pipe.
    """
    ready = json.dumps(
        {
            "ready": True,
            "pid": os.getpid(),
            "transport": "tcp",
            "host": host,
            "port": port,
        }
    )
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
                error_payload = json.dumps(
                    error_envelope(
                        f"invalid JSON request: {exc}",
                        "invalid_request",
                        ErrorCorrelation.uncorrelated(),
                    )
                )
                wfile.write(error_payload + "\n")
                wfile.flush()
                continue

            try:
                dispatch = dispatch_protocol_message(message)
            except BaseException as exc:
                # Mirrors the stdio handler's contract: catch every dispatch
                # exception, classify against typed bootstrap error types,
                # emit a structured ``{"op":"error", "kind":...}`` envelope.
                # Without this, a bootstrap-class failure on a TCP-mode
                # worker would propagate up, kill the connection handler,
                # and the orchestrator would see a closed socket rather
                # than a typed error, bypassing the entire
                # bootstrap-vs-runtime classification machinery.
                kind = _classify_dispatch_exception(exc)
                import sys
                import traceback

                sys.stderr.write(
                    f"--- worker dispatch exception ({kind}) ---\n"
                    + traceback.format_exc()
                )
                sys.stderr.flush()
                error_payload = json.dumps(
                    error_envelope(
                        str(exc) or exc.__class__.__name__,
                        kind,
                        ErrorCorrelation.of_message(message),
                    )
                )
                wfile.write(error_payload + "\n")
                wfile.flush()
                if kind == "bootstrap":
                    # Worker is in a partially-initialized state; tear the
                    # connection down so the pool spawns a replacement.
                    # The orchestrator's terminal classification of
                    # bootstrap errors prevents this from cascading.
                    return
                continue

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

    def _stop_reading() -> None:
        """Publish shutdown and wake the connection's blocked reader together."""
        shutdown_event.set()
        with suppress(OSError):
            conn.shutdown(socket.SHUT_RD)

    def _handle_and_respond(request: PendingProtocolRequest) -> None:
        # Short-circuit if a prior pooled task already signalled shutdown
        # (bootstrap error, broken pipe). See ``_serve_stdio_concurrent``
        # for the same gate's rationale.
        if shutdown_event.is_set():
            return
        try:
            dispatch = dispatch_prepared_protocol_message(request)
        except BaseException as exc:
            # Same exception-shielding contract as the sequential TCP
            # handler. Classification + structured emit + bootstrap-on-
            # shutdown_event.
            kind = _classify_dispatch_exception(exc)
            import sys
            import traceback

            sys.stderr.write(
                f"--- worker dispatch exception ({kind}) ---\n" + traceback.format_exc()
            )
            sys.stderr.flush()
            with write_lock:
                error_payload = json.dumps(
                    error_envelope(
                        str(exc) or exc.__class__.__name__,
                        kind,
                        ErrorCorrelation.of_message(request.message),
                    )
                )
                try:
                    wfile.write(error_payload + "\n")
                    wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    _stop_reading()
            if kind == "bootstrap":
                _stop_reading()
            return

        with write_lock:
            try:
                wfile.write(json.dumps(dispatch.payload) + "\n")
                wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                _stop_reading()

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
                        error_envelope(
                            f"invalid JSON request: {exc}",
                            "invalid_request",
                            ErrorCorrelation.uncorrelated(),
                        )
                    )
                    wfile.write(error_payload + "\n")
                    wfile.flush()
                continue

            prepared = prepare_protocol_message(message)
            if isinstance(prepared, ProtocolDispatchResult):
                with write_lock:
                    wfile.write(json.dumps(prepared.payload) + "\n")
                    wfile.flush()
                if prepared.should_shutdown:
                    _stop_reading()
                    break
                continue
            pool.submit(_handle_and_respond, prepared)
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
    disconnects), the worker waits for a new connection, it persists across
    server restarts.

    Registers itself in ``workers.json`` on startup and removes itself on
    shutdown.
    """
    from batchalign.worker._registry import register_worker, unregister_worker

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(1)
    actual_port = server_sock.getsockname()[1]

    register_worker(_registry_entry(host, actual_port), registry_path=registry_path)

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
    from batchalign.worker._registry import register_worker, unregister_worker

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(1)
    actual_port = server_sock.getsockname()[1]

    register_worker(_registry_entry(host, actual_port), registry_path=registry_path)

    _print_ready_tcp(host, actual_port)
    logger.info(
        "TCP worker listening on %s:%d (concurrent, %d threads)",
        host,
        actual_port,
        max_threads,
    )

    try:
        while True:
            conn, addr = server_sock.accept()
            _handle_tcp_connection_concurrent(conn, addr, max_threads)
    except KeyboardInterrupt:
        logger.info("TCP worker shutting down (KeyboardInterrupt)")
    finally:
        server_sock.close()
        unregister_worker(host=host, port=actual_port, registry_path=registry_path)
