"""The worker's typed report that a request names a model it has not loaded.

Its own module, holding nothing else, so the Rust worker boundary
(``pyo3::import_exception!`` in ``batchalign-pyo3/src/worker_execute.rs``)
can recognise the class by identity, as the protocol code
``model_unavailable``, without importing the model libraries that raise it.
"""

from __future__ import annotations


class ModelNotLoadedError(RuntimeError):
    """A request names a model this worker process has not loaded.

    A worker-local load fact, not a failure of the request's input: the same
    request on a worker that loaded the model succeeds, and the same request
    here fails the same way again. Reported as ``model_unavailable`` rather
    than as a local model's runtime failure.
    """


__all__ = ["ModelNotLoadedError"]
