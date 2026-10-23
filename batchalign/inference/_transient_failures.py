"""Which raised exceptions are failures a later attempt can get past, by class.

Two classifiers decide what a worker failure means: the Python request loops
(``batchalign.worker._protocol._classify_dispatch_exception``, for a failure
that escapes an op such as ``ensure_task``) and the Rust worker boundary
(``classify_runner_error`` in ``batchalign-pyo3``, for a failure raised by a
model host while it runs a request). Both read the two questions below from
this module, so a download raised while a worker loads and the same download
raised in the middle of a request are classified alike.

Classification is by exception CLASS, never by message text. A library's
exception classes are read from ``sys.modules`` rather than imported: an
exception can only be an instance of a class whose module is loaded, so a
module that is not loaded answers "no" without being imported, and the
classifier never imports a model library to classify the first failure.

The classes are the ones the locked dependencies actually raise, checked
against the installed packages:

- ``huggingface_hub`` downloads through ``httpx2``. A failed connection or a
  timeout while it resolves a file reaches the caller as
  ``LocalEntryNotFoundError`` raised *from* the ``httpx2`` error (no cached
  copy, and the Hub could not be reached), or as the ``httpx2`` error itself
  while the file body downloads.
- Stanza fetches its models through ``huggingface_hub`` and other files
  through ``requests``; batchalign's own catalog fetch raises
  ``StanzaCatalogDownloadError``.
- ``torch`` raises ``torch.OutOfMemoryError`` (the same class as
  ``torch.cuda.OutOfMemoryError``) when a CUDA allocation fails. On Apple MPS,
  torch 2.14 raises a plain ``RuntimeError`` for the same condition, which no
  class distinguishes, so an MPS shortfall stays a runtime failure.
"""

from __future__ import annotations

import sys


def _loaded_class(module: str, name: str) -> type[BaseException] | None:
    """The exception class ``module.name`` if that module is loaded.

    ``None`` when the module is not loaded, in which case nothing raised so
    far can be an instance of it.
    """
    loaded = sys.modules.get(module)
    if loaded is None:
        return None
    candidate = getattr(loaded, name, None)
    if isinstance(candidate, type) and issubclass(candidate, BaseException):
        return candidate
    return None


def _loaded_classes(module: str, *names: str) -> tuple[type[BaseException], ...]:
    """Every named exception class of ``module`` that is loaded."""
    return tuple(
        cls
        for cls in (_loaded_class(module, name) for name in names)
        if cls is not None
    )


# HTTP statuses that say the Hub itself is briefly unable to serve, as
# opposed to refusing this machine (401, 403) or lacking the file (404).
_HUB_TRANSIENT_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


def _is_transport_failure(error: BaseException) -> bool:
    """A connection, a timeout or a broken transfer, in either HTTP stack."""
    transport = (
        # The connection could not be made or broke (``ConnectError``,
        # ``ReadError``, ...), took too long, or the server broke the protocol
        # mid-transfer. Not ``ProxyError``, which is this machine's
        # configuration and fails the same way again.
        *_loaded_classes(
            "httpx2", "NetworkError", "TimeoutException", "RemoteProtocolError"
        ),
        *_loaded_classes(
            "requests.exceptions", "ConnectionError", "Timeout", "ChunkedEncodingError"
        ),
    )
    return isinstance(error, transport)


def _is_hub_outage(error: BaseException) -> bool:
    """A Hub HTTP error whose status says the Hub is briefly unable to serve."""
    http_error = _loaded_class("huggingface_hub.errors", "HfHubHTTPError")
    if http_error is None or not isinstance(error, http_error):
        return False
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    return status in _HUB_TRANSIENT_STATUSES


def is_model_download_failure(error: BaseException) -> bool:
    """Whether ``error`` is a model or catalog download that failed in transit.

    True for a transport failure (connection, timeout, broken transfer), a Hub
    outage status, batchalign's own Stanza catalog download failure, and a
    Hub ``LocalEntryNotFoundError`` raised from one of those. False for a
    ``LocalEntryNotFoundError`` with no such cause (offline mode, or
    ``local_files_only``), which a retry cannot fix, and for any access
    refusal (a gated or private repository, HTTP 401 or 403).
    """
    if _is_transport_failure(error) or _is_hub_outage(error):
        return True
    catalog_error = _loaded_class(
        "batchalign.worker._stanza_capabilities", "StanzaCatalogDownloadError"
    )
    if catalog_error is not None and isinstance(error, catalog_error):
        return True
    no_cached_copy = _loaded_class("huggingface_hub.errors", "LocalEntryNotFoundError")
    if no_cached_copy is not None and isinstance(error, no_cached_copy):
        cause = error.__cause__
        return cause is not None and (
            _is_transport_failure(cause) or _is_hub_outage(cause)
        )
    return False


def is_device_out_of_memory(error: BaseException) -> bool:
    """Whether ``error`` is an accelerator out-of-memory failure.

    ``torch.OutOfMemoryError`` only; see the module docstring for why an MPS
    shortfall is not recognised.
    """
    out_of_memory = _loaded_class("torch", "OutOfMemoryError")
    return out_of_memory is not None and isinstance(error, out_of_memory)


__all__ = ["is_device_out_of_memory", "is_model_download_failure"]
