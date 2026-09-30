# affects: batchalign/worker/_main.py
# affects: batchalign/worker/_stanza_capabilities.py
"""Integration regression: the worker refreshes a stale Stanza manifest.

2026-06-10 incident. Stanford re-published several lemma models under the
SAME Stanza resources version (1.12.0), changing their md5s. Every worker
pipeline is built with ``DownloadMethod.REUSE_RESOURCES``, which reuses the
cached ``resources.json`` and never re-fetches it, then verifies any
freshly-downloaded model against that cached manifest. A host whose cached
manifest predates the re-publish keeps the OLD md5, so the next model
download fails Stanza's integrity check
(``md5 for ... is X, expected Y`` -> surfaced as ``ensure_task failed``),
killing morphotag on any file that needs the re-published model. The trigger is
a bilingual transcript (e.g. ``eng,spa``) where a single ``[- spa]`` utterance
is the only item needing the re-published Spanish model.

Fix: the worker bootstrap (``batchalign/worker/_main.py::main``) refreshes a
present-but-stale ``resources.json`` from upstream once per worker, before any
model is loaded, guarded so an offline worker falls back to the cached manifest
instead of failing. The refresh lives in the worker bootstrap, NOT in
``get_cached_capability_table`` (which tests and tooling also call and which
must not trigger a network fetch or mutate the real on-disk cache).

These tests drive the REAL worker subprocess
(``python -m batchalign.worker --lazy --profile stanza``: it runs the full
bootstrap but loads no models, signalling ready immediately) and serve the
"new" manifest from a local HTTP server, so they are fully deterministic and
never touch Stanford's servers.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

# Sentinel markers distinguishing the cached (stale) manifest from the upstream
# (fresh) one. The worker's capability builder skips this non-dict key, so it is
# a safe, observable refresh witness.
_STALE = "STALE"
_FRESH = "FRESH"


def _stanza_resources_version() -> str:
    """The resources catalog version Stanza fetches (e.g. ``1.12.0``).

    Read from the same module constant the worker uses so the local upstream
    serves the exact ``resources_<version>.json`` filename Stanza requests.
    """
    import stanza.resources.common as src

    return src.DEFAULT_RESOURCES_VERSION


def _manifest(marker: str) -> dict[str, object]:
    """A minimal catalog with an observable ``_manifest_marker`` refresh witness."""
    return {
        "_manifest_marker": marker,
        "es": {
            "tokenize": {},
            "pos": {},
            "lemma": {"combined_nocharlm": {"md5": marker}},
            "depparse": {},
        },
    }


def _write_stale_model_dir(tmp_path: Path) -> tuple[Path, Path]:
    """Create a worker model dir holding a present-but-STALE manifest.

    Returns ``(model_dir, manifest_path)``.
    """
    model_dir = tmp_path / "stanza"
    model_dir.mkdir()
    manifest = model_dir / "resources.json"
    manifest.write_text(json.dumps(_manifest(_STALE)))
    return model_dir, manifest


def _worker_env(model_dir: Path, resources_url: str) -> dict[str, str]:
    """Worker env pointing Stanza at ``model_dir`` and at ``resources_url``."""
    env = dict(os.environ)
    env["STANZA_RESOURCES_DIR"] = str(model_dir)
    env["STANZA_RESOURCES_URL"] = resources_url
    return env


def _run_lazy_stanza_worker(
    env: dict[str, str], timeout_s: float = 90.0
) -> subprocess.CompletedProcess[str]:
    """Run a real lazy STANZA-profile worker to completion.

    ``--lazy --profile stanza`` runs the full worker bootstrap, including the
    one-time manifest refresh in ``_main.main``, but loads NO models
    (``load_worker_profile_lazy`` signals ready immediately). Closing stdin
    (EOF) makes the stdio serve loop (``for raw_line in sys.stdin``) exit
    cleanly, so the bootstrap side effect (the refresh) is complete by the time
    the process returns.
    """
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "batchalign.worker",
            "--lazy",
            "--profile",
            "stanza",
            "--serving",
            "sequential",
            "--supervisor-pid",
            str(os.getpid()),
        ],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = proc.communicate(input="", timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
    return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)


@pytest.mark.integration
def test_worker_refreshes_stale_resources_manifest(tmp_path: Path) -> None:
    ver = _stanza_resources_version()

    # 1. Local "upstream" serving the FRESH manifest as resources_<ver>.json.
    web_root = tmp_path / "upstream"
    web_root.mkdir()
    (web_root / f"resources_{ver}.json").write_text(json.dumps(_manifest(_FRESH)))

    def _handler(*args: object, **kwargs: object) -> SimpleHTTPRequestHandler:
        return SimpleHTTPRequestHandler(*args, directory=str(web_root), **kwargs)  # type: ignore[arg-type]

    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler)  # type: ignore[arg-type]
    port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        # 2. A present-but-STALE cached manifest; run a real worker pointing
        #    Stanza at it and at our local upstream.
        model_dir, manifest = _write_stale_model_dir(tmp_path)
        result = _run_lazy_stanza_worker(
            _worker_env(model_dir, f"http://127.0.0.1:{port}")
        )

        assert result.returncode == 0, (
            f"worker exited {result.returncode}; stderr={result.stderr!r}"
        )
        # 4. The on-disk manifest must now be the FRESH upstream copy.
        on_disk = json.loads(manifest.read_text())
        assert on_disk["_manifest_marker"] == _FRESH, (
            "worker did not refresh the stale resources.json "
            f"(marker still {on_disk['_manifest_marker']!r}); stderr={result.stderr!r}"
        )
    finally:
        server.shutdown()
        server_thread.join(timeout=5)


@pytest.mark.integration
def test_worker_keeps_cached_manifest_when_upstream_unreachable(tmp_path: Path) -> None:
    """Offline safety: an unreachable upstream must not break the worker.

    Air-gapped / offline fleet machines must keep working exactly as before the
    fix (plain ``REUSE_RESOURCES``): the refresh fails quietly, the cached
    manifest is left intact, and the worker still bootstraps. This is the
    load-bearing guarantee that the guarded refresh does not regress hosts that
    cannot reach Stanford at boot.
    """
    # Port 1 is closed: the fetch fails fast with connection-refused rather than
    # waiting out the refresh timeout, so the offline path is exercised
    # deterministically without a real network outage.
    model_dir, manifest = _write_stale_model_dir(tmp_path)
    result = _run_lazy_stanza_worker(_worker_env(model_dir, "http://127.0.0.1:1"))

    # The worker must still bootstrap and exit cleanly from the cached manifest.
    assert result.returncode == 0, (
        f"offline worker must still bootstrap cleanly: stderr={result.stderr!r}"
    )
    # The cached manifest is left untouched: no torn/empty file written.
    on_disk = json.loads(manifest.read_text())
    assert on_disk["_manifest_marker"] == _STALE, (
        "offline refresh must leave the cached manifest intact "
        f"(marker is {on_disk['_manifest_marker']!r})"
    )
    # No leftover temp file from a failed/partial install.
    assert not (model_dir / "resources.json.tmp-refresh").exists()
