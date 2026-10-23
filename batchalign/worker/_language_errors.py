"""The worker's typed refusal of a language it has no pipeline for.

Its own module, holding nothing else, because two boundaries recognise the
class by identity and must be able to import it cheaply: the request loop's
error-kind classifier (``_protocol.py``) and the Rust worker boundary
(``pyo3::import_exception!`` in ``batchalign-pyo3/src/worker_execute.rs``),
which reports it as the protocol code ``unsupported_language``. Importing it
must never pull in Stanza or the pipeline loaders that raise it.
"""

from __future__ import annotations


class UnsupportedLanguageError(ValueError):
    """Stanza has no usable pipeline for the requested language.

    Distinct from a configuration error: the request itself cannot be
    served by this worker, so callers should reject the job upstream
    rather than retry. Surfaced as a typed error so downstream code
    can branch on it cleanly instead of pattern-matching the deep
    ``KeyError`` Stanza would otherwise raise from
    ``maintain_processor_list``.

    Subclasses ``ValueError`` for callers that predate it; the Rust worker
    boundary checks this class before any broader one, so the subclassing
    never decides its protocol code.
    """
