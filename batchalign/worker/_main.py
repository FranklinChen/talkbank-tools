"""Start an admitted Python worker and enter its original serving loop."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from batchalign.worker._launch import (
    AdmittedLaunch,
    FixedPort,
    StdioLaunch,
    admit_worker_launch,
)
from batchalign.worker._model_loading import start_worker
from batchalign.worker._protocol import (
    _print_ready,
    _serve_stdio,
    _serve_stdio_concurrent,
    _serve_tcp,
    _serve_tcp_concurrent,
    claim_protocol_stdout,
)
from batchalign.worker._runtime_identity import observe_worker_runtime
from batchalign.worker._stanza_capabilities import refresh_resources_manifest_if_present
from batchalign.worker._supervisor import watch_supervisor
from batchalign.worker._types import ModelStartup, WorkerServing

if TYPE_CHECKING:
    from batchalign.inference._domain_types import TcpPort
    from batchalign.worker._runtime_identity import WorkerRuntimeIdentity

L = logging.getLogger("batchalign.worker")


@dataclass(frozen=True, slots=True)
class _StartedLaunch:
    launch: AdmittedLaunch
    runtime: WorkerRuntimeIdentity

    def serve(self) -> None:
        """Serve using the same transport and serving choice Rust routed to."""
        transport = self.launch.transport
        serving = self.launch.serving
        threads = self.launch.threads
        L.info("serving=%s threads=%d", serving.value, threads)
        if isinstance(transport, StdioLaunch):
            _print_ready(self.runtime)
            if serving is WorkerServing.CONCURRENT:
                _serve_stdio_concurrent(max_threads=threads)
            else:
                _serve_stdio()
            return

        host = transport.host
        port = (
            transport.port.value
            if isinstance(transport.port, FixedPort)
            else _auto_assign_port(host)
        )
        startup = self.launch.startup
        if serving is WorkerServing.CONCURRENT:
            _serve_tcp_concurrent(
                host, port, self.runtime, startup, max_threads=threads
            )
        else:
            _serve_tcp(host, port, self.runtime, startup)


def _start(launch: AdmittedLaunch) -> _StartedLaunch:
    """Claim stdout, supervise stdio, and publish the admitted worker."""
    claim_protocol_stdout()
    if isinstance(launch.transport, StdioLaunch):
        watch_supervisor(launch.transport.supervisor)

    # The runtime identity this worker reports, observed NOW, before any
    # dependency or model is imported. It describes the installed runtime, and
    # the server refuses a worker whose identity differs from the first one it
    # admitted. Imports change what it would read later: loading FunASR's
    # dependencies puts setuptools' vendored packages on `sys.path`, which
    # changes the distribution inventory, so a worker that loaded models before
    # observing reported a different identity than one that had not, and was
    # refused (2026-11-05, when the FunASR load moved to bootstrap).
    runtime = observe_worker_runtime()

    log_level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(
        launch.verbosity, logging.DEBUG
    )
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    startup = launch.startup
    if isinstance(startup, ModelStartup):
        _install_audio_backend()
        # Refresh a present but stale manifest before any model can be loaded.
        # Echo workers have no model resources to refresh.
        refresh_resources_manifest_if_present()
    start_worker(startup)
    return _StartedLaunch(launch, runtime)


def _install_audio_backend() -> None:
    # Route torchaudio's decoder through soundfile BEFORE anything imports a
    # dependency that might call it. `torchaudio.load` goes to torchcodec,
    # which binds a fixed set of FFmpeg majors through an unversioned rpath; a
    # Homebrew upgrade past that set takes out every decode at job time. There
    # is no `backend=` escape (torchaudio 2.11 ignores it), so the fix is to
    # not call it. Logged rather than silent: an operator reading worker
    # startup should be able to see which decoder is in play.
    try:
        from batchalign.inference.audio import (
            SoundfileBackendInstall,
            install_soundfile_backend,
        )

        if install_soundfile_backend() is SoundfileBackendInstall.INSTALLED:
            logging.getLogger(__name__).info(
                "audio: torchaudio.load routed through soundfile"
            )
    except Exception as error:  # pragma: no cover - never block startup
        logging.getLogger(__name__).warning(
            "audio: could not route torchaudio.load through soundfile: %s", error
        )


def main() -> None:
    """Run the worker launch admitted from this process's argv."""
    _start(admit_worker_launch()).serve()


def _auto_assign_port(host: str) -> TcpPort:
    """Find an available port in the 9100-9199 range.

    Checks the worker registry to avoid collisions with already-registered
    workers, then verifies the port is bindable.
    """
    import socket

    from batchalign.worker._registry import list_workers

    registered = list_workers()
    used_ports = {e.port for e in registered if e.host == host}

    for candidate in range(9100, 9200):
        if candidate in used_ports:
            continue
        # Verify the port is actually available by trying to bind.
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, candidate))
                return candidate
        except OSError:
            continue

    raise RuntimeError(
        f"No available ports in range 9100-9199 on {host}. "
        "Stop some workers or specify --port explicitly."
    )
