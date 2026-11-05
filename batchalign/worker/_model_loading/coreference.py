"""Coreference-model load for ``ensure_task``.

Coref loads on first use: the pool sends ``ensure_task("coref")`` to the
English worker its requests reach (the language Rust's ``COREF_LANGUAGE``
keys coref by, and probes in), under a loading stall window that download
growth re-arms. That is why the model is prepared here (``prepare_coref``)
and not inside a request: its files download now, and each batch then builds
the pipeline from them and releases it.
"""

from __future__ import annotations

import logging

from batchalign.worker._loads import LoadedCoref

L = logging.getLogger("batchalign.worker")


def load_coref_task() -> LoadedCoref:
    """Prepare this worker's coreference model and return its host.

    The host's runner closes over the ``PreparedCoref``, which holds no
    pipeline, so the registry entry keeps nothing resident between batches.
    """
    from batchalign.inference.coref import prepare_coref
    from batchalign.worker._infer_hosts import build_coref_runner

    coref = prepare_coref()
    L.info("Prepared coreference model %s", coref.engine)
    return LoadedCoref(runner=build_coref_runner(coref))
