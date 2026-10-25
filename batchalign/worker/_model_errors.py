"""Lightweight exception identities for model availability and output admission.

Its own module, holding nothing else, so the Rust worker boundary
(``pyo3::import_exception!`` in ``batchalign-pyo3/src/worker_execute.rs``)
can recognise each class without importing the model libraries that raise it.
"""

from __future__ import annotations


class ModelNotLoadedError(RuntimeError):
    """A request names a model this worker process has not loaded.

    A worker-local load fact, not a failure of the request's input: the same
    request on a worker that loaded the model succeeds, and the same request
    here fails the same way again. Reported as ``model_unavailable`` rather
    than as a local model's runtime failure.
    """


class ModelOutputInvalidError(ValueError):
    """Inference returned data that cannot become admitted pipeline evidence.

    The request was valid and the model ran. Retrying the identical request or
    editing input CHAT is not a repair for its malformed result. The worker
    protocol keeps this distinct from invalid requests and transient failures.
    """


__all__ = ["ModelNotLoadedError", "ModelOutputInvalidError"]
