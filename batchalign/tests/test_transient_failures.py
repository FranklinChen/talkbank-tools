"""Download and out-of-memory failures are classified by class, alike everywhere.

``batchalign.inference._transient_failures`` owns two questions (is this a
download that failed in transit; is this an accelerator out of memory). The
Python request loops and the Rust worker boundary both ask it, so these tests
drive each consumer: the classifier itself, the Hub access translation that
used to swallow connection failures, the request loop's wire kind, the worker
startup report, and a utseg request whose Stanza pipeline download fails
inside the request.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import httpx2
import pytest
import requests
import torch
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

from batchalign.inference._model_access_errors import (
    ModelAccessDeniedError,
    classify_huggingface_access_error,
)
from batchalign.inference._transient_failures import (
    is_device_out_of_memory,
    is_model_download_failure,
)
from batchalign.worker._execute_v2 import WorkerExecutionHostV2, execute_request_v2
from batchalign.worker._protocol import _classify_dispatch_exception
from batchalign.worker._stanza_capabilities import StanzaCatalogDownloadError
from batchalign.worker._text_v2 import TextExecutionHostV2
from batchalign.worker._types_v2 import (
    ExecuteErrorV2,
    ExecuteRequestV2,
    InferenceTaskV2,
    PreparedTextEncodingV2,
    PreparedTextRefV2,
    ProtocolErrorCodeV2,
    UtsegRequestV2,
    UtsegSegmenterV2,
)


def _hub_response(status: int) -> httpx2.Response:
    return httpx2.Response(
        status,
        request=httpx2.Request(
            "HEAD", "https://huggingface.co/owner/model/resolve/main/config.json"
        ),
    )


def _raised_from(error: BaseException, cause: BaseException) -> BaseException:
    """``error`` as ``raise error from cause`` leaves it."""
    try:
        raise error from cause
    except BaseException as raised:
        return raised


def _no_cached_copy_after(cause: BaseException) -> BaseException:
    """What ``hf_hub_download`` raises when it cannot reach the Hub: a
    ``LocalEntryNotFoundError`` raised from the transport failure."""
    return _raised_from(
        LocalEntryNotFoundError("cannot find the requested files in the local cache"),
        cause,
    )


_CONNECT = httpx2.ConnectError("connection refused")

DOWNLOADS = [
    pytest.param(_no_cached_copy_after(_CONNECT), id="no-cached-copy-after-connect"),
    pytest.param(
        _no_cached_copy_after(httpx2.ConnectTimeout("timed out")),
        id="no-cached-copy-after-timeout",
    ),
    pytest.param(
        _no_cached_copy_after(HfHubHTTPError("503", response=_hub_response(503))),
        id="no-cached-copy-after-hub-outage",
    ),
    pytest.param(httpx2.ReadError("connection reset"), id="httpx2-read-error"),
    pytest.param(httpx2.ReadTimeout("read timed out"), id="httpx2-read-timeout"),
    pytest.param(requests.exceptions.ConnectionError("refused"), id="requests-connect"),
    pytest.param(requests.exceptions.ReadTimeout("slow"), id="requests-timeout"),
    pytest.param(HfHubHTTPError("502", response=_hub_response(502)), id="hub-502"),
    pytest.param(
        StanzaCatalogDownloadError("catalog fetch failed"), id="stanza-catalog"
    ),
]

NOT_DOWNLOADS = [
    pytest.param(
        LocalEntryNotFoundError("offline, and no cached copy"),
        id="no-cached-copy-offline",
    ),
    pytest.param(
        _no_cached_copy_after(HfHubHTTPError("403", response=_hub_response(403))),
        id="no-cached-copy-after-403",
    ),
    pytest.param(HfHubHTTPError("401", response=_hub_response(401)), id="hub-401"),
    pytest.param(httpx2.ProxyError("proxy refused"), id="proxy-misconfigured"),
    pytest.param(RuntimeError("connection refused"), id="message-text-is-not-a-class"),
]


@pytest.mark.parametrize("error", DOWNLOADS)
def test_a_download_that_failed_in_transit_is_a_download(error: BaseException) -> None:
    assert is_model_download_failure(error)
    assert _classify_dispatch_exception(error) == "download"


@pytest.mark.parametrize("error", NOT_DOWNLOADS)
def test_what_a_retry_cannot_fix_is_not_a_download(error: BaseException) -> None:
    assert not is_model_download_failure(error)
    assert _classify_dispatch_exception(error) == "runtime"


def test_a_connection_failure_is_not_an_access_denial() -> None:
    """RED FIRST: ``hf_hub_download`` raises ``LocalEntryNotFoundError`` from
    a ``ConnectError`` when the Hub cannot be reached, and the access
    translation made every such error ``ModelAccessDeniedError``: a network
    blip told the operator to fix their credentials, was never retried, and
    withheld the UTR fallback."""
    assert classify_huggingface_access_error(_no_cached_copy_after(_CONNECT)) is None


def test_an_access_refusal_stays_an_access_denial() -> None:
    offline = classify_huggingface_access_error(
        LocalEntryNotFoundError("offline, and no cached copy")
    )
    forbidden = classify_huggingface_access_error(
        HfHubHTTPError("401", response=_hub_response(401))
    )
    assert isinstance(offline, ModelAccessDeniedError)
    assert isinstance(forbidden, ModelAccessDeniedError)


def test_a_real_hub_download_against_a_closed_port_is_a_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real library, not a hand-built exception: ``hf_hub_download``
    against an endpoint nothing listens on raises what a network outage
    raises. The Hub's own retry backoff is skipped so the test takes no
    time."""
    import time

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import _http

    monkeypatch.setattr(
        _http,
        "time",
        types.SimpleNamespace(
            sleep=lambda _seconds: None, time=time.time, monotonic=time.monotonic
        ),
    )

    with pytest.raises(Exception) as raised:
        hf_hub_download(
            repo_id="owner/model",
            filename="config.json",
            endpoint="http://127.0.0.1:1",
            cache_dir=tmp_path,
        )

    assert is_model_download_failure(raised.value), repr(raised.value)
    assert classify_huggingface_access_error(raised.value) is None


def test_device_out_of_memory_is_recognised_by_class_only() -> None:
    """``torch.OutOfMemoryError`` is the class torch raises for a CUDA
    shortfall. A plain ``RuntimeError`` saying the same words (which is what
    torch raises for an MPS shortfall) is not classified by its text."""
    assert is_device_out_of_memory(torch.OutOfMemoryError("CUDA out of memory"))
    assert (
        _classify_dispatch_exception(torch.OutOfMemoryError("oom")) == "out_of_memory"
    )
    assert not is_device_out_of_memory(RuntimeError("MPS backend out of memory"))
    assert not is_device_out_of_memory(MemoryError())


def test_a_utseg_pipeline_download_failing_inside_the_request_is_a_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED FIRST: an ``auto`` or ``per-file`` utseg worker loads the request
    language's Stanza pipeline inside the request. A download failure there
    reached the Rust boundary as a runtime failure of a local model, which
    the server does not retry. It now carries ``model_download_failed``."""
    from batchalign.worker import _stanza_loading
    from batchalign.worker._infer_hosts import build_utseg_inference_handler

    def _download_fails(_lang: str) -> object:
        raise _no_cached_copy_after(httpx2.ConnectError("connection refused"))

    monkeypatch.setattr(_stanza_loading, "load_utseg_pipeline", _download_fails)

    payload_path = tmp_path / "utseg-batch.json"
    payload_path.write_text(
        json.dumps(
            {
                "items": [
                    {"words": ["hello", "world", "again"], "text": "hello world again"}
                ]
            }
        ),
        encoding="utf-8",
    )
    response = execute_request_v2(
        request=ExecuteRequestV2(
            request_id="req-utseg-download",
            task=InferenceTaskV2.UTSEG,
            payload=UtsegRequestV2(
                lang="eng",
                payload_ref_id="text-ref-utseg-download",
                item_count=1,
                segmenter=UtsegSegmenterV2.STANZA_CONSTITUENCY,
            ),
            attachments=[
                PreparedTextRefV2(
                    id="text-ref-utseg-download",
                    path=str(payload_path),
                    encoding=PreparedTextEncodingV2.UTF8_JSON,
                    byte_offset=0,
                    byte_len=payload_path.stat().st_size,
                )
            ],
        ),
        host=WorkerExecutionHostV2(
            text=TextExecutionHostV2(utseg_runner=build_utseg_inference_handler())
        ),
    )

    assert isinstance(response.outcome, ExecuteErrorV2)
    assert response.outcome.code is ProtocolErrorCodeV2.MODEL_DOWNLOAD_FAILED


def test_a_worker_whose_startup_download_fails_says_so_in_place_of_ready(
    tmp_path: Path,
) -> None:
    """RED FIRST: a stdio worker that could not download its models at
    startup exited before its ready line, and the server could only call
    that a system failure. It now writes one classified error line instead,
    which the Rust ready reader turns into a retried download failure."""
    code = (
        "import sys, httpx2\n"
        "from huggingface_hub.errors import LocalEntryNotFoundError\n"
        "from batchalign.worker import _main\n"
        "def fail(_bootstrap):\n"
        "    try:\n"
        "        raise httpx2.ConnectError('connection refused')\n"
        "    except httpx2.ConnectError as cause:\n"
        "        raise LocalEntryNotFoundError('no cached copy') from cause\n"
        "_main.load_worker_task = fail\n"
        "sys.argv = ['worker', '--task', 'morphosyntax', '--lang', 'eng',\n"
        "            '--transport', 'stdio', '--serving', 'sequential', '--supervisor-pid', str(__import__('os').getppid())]\n"
        "_main.refresh_resources_manifest_if_present = lambda: None\n"
        "_main.main()\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 1, completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert len(lines) == 1, completed.stdout
    envelope = json.loads(lines[0])
    assert envelope["op"] == "error"
    assert envelope["kind"] == "download"
    assert "no cached copy" in envelope["error"]
