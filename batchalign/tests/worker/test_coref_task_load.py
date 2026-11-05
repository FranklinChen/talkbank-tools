"""Coref loads on first use, and its model is resident only during a batch.

A coref worker starts with no coreference model. ``ensure_task("coref")``,
which the pool sends before the worker's first coref request, prepares it: the
pipeline is built once so its files download inside the loading stall window,
then released. Each request builds the pipeline from those files and releases
it when the batch ends, so an idle worker holds no model. A request on a worker
whose coref task never loaded is the ``model_unavailable`` refusal, never a
load.

``stanza`` is replaced by a fake whose ``Pipeline`` counts constructions and
live instances, so each test reads exactly where the model was built and
whether it is still held.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from batchalign.inference.coref import PreparedCoref
from batchalign.worker._execute_v2 import execute_request_v2
from batchalign.worker._infer import _batch_infer
from batchalign.worker._loads import COREF_LOAD
from batchalign.worker._model_loading.bootstrap import (
    ensure_task_loaded,
    start_worker,
)
from batchalign.worker._types import (
    BatchInferRequest,
    InferTask,
    TaskLoad,
    WorkerBootstrapRuntime,
    WorkerProfile,
    _state,
)
from batchalign.worker._types_v2 import (
    CorefRequestV2,
    CorefResultV2,
    ExecuteErrorV2,
    ExecuteRequestV2,
    ExecuteSuccessV2,
    InferenceTaskV2,
    PreparedTextEncodingV2,
    PreparedTextRefV2,
    ProtocolErrorCodeV2,
)

ENGINE = "stanza-1.99.0/ontonotes-singletons_roberta-large-lora"


class _CountingPipeline:
    """A coreference pipeline double that records constructions and lives."""

    constructions: list[dict[str, object]] = []
    alive = 0

    def __init__(self, **kwargs: object) -> None:
        type(self).constructions.append(kwargs)
        type(self).alive += 1

    def __del__(self) -> None:
        type(self).alive -= 1

    def __call__(self, text: str) -> SimpleNamespace:
        # One sentence per blank-line-separated block, every word chainless.
        return SimpleNamespace(
            sentences=[
                SimpleNamespace(
                    words=[
                        SimpleNamespace(text=word, coref_chains=[])
                        for word in block.split(" ")
                    ]
                )
                for block in text.split("\n\n")
            ]
        )


@pytest.fixture(autouse=True)
def _fresh_worker(monkeypatch) -> Iterator[None]:
    """A fresh worker state and a fake ``stanza``. The conftest already keeps
    ``download_liveness`` off the real caches."""
    _state.__init__()  # type: ignore[misc]
    _CountingPipeline.constructions = []
    _CountingPipeline.alive = 0
    monkeypatch.setitem(
        sys.modules,
        "stanza",
        SimpleNamespace(Pipeline=_CountingPipeline, __version__="1.99.0"),
    )
    yield
    _state.__init__()  # type: ignore[misc]


def _coref_worker(lang: str) -> WorkerBootstrapRuntime:
    return WorkerBootstrapRuntime(task=InferTask.COREF, lang=lang, num_speakers=1)


def _coref_request(tmp_path: Path, name: str) -> ExecuteRequestV2:
    """One V2 coref request over a prepared one-document batch."""
    payload_path = tmp_path / f"{name}.json"
    payload_path.write_text(
        json.dumps({"items": [{"sentences": [["she", "left"]]}]}), encoding="utf-8"
    )
    return ExecuteRequestV2(
        request_id=f"req-{name}",
        task=InferenceTaskV2.COREF,
        payload=CorefRequestV2(lang="eng", payload_ref_id=name, item_count=1),
        attachments=[
            PreparedTextRefV2(
                id=name,
                path=str(payload_path),
                encoding=PreparedTextEncodingV2.UTF8_JSON,
                byte_offset=0,
                byte_len=payload_path.stat().st_size,
            )
        ],
    )


def _execute(request: ExecuteRequestV2):
    """Execute against this worker's load registry, as the live worker loop
    does per request."""
    return execute_request_v2(request)


EXPECTED_PIPELINE = {
    "lang": "en",
    "processors": "tokenize,coref",
    "package": {"coref": "ontonotes-singletons_roberta-large-lora"},
    "tokenize_pretokenized": True,
}


def test_a_coref_worker_starts_without_the_model() -> None:
    start_worker(_coref_worker("eng"))

    assert _CountingPipeline.constructions == []
    assert _state.loads.get(COREF_LOAD) is None
    assert _state.loads.completed() == frozenset()


def test_ensure_task_prepares_the_model_and_releases_it() -> None:
    start_worker(_coref_worker("eng"))

    loaded = ensure_task_loaded("coref")
    again = ensure_task_loaded("coref")

    assert loaded.status == "loaded"
    assert again.status == "already_loaded"
    # Built once to fetch its files, and not held afterwards.
    assert _CountingPipeline.constructions == [EXPECTED_PIPELINE]
    assert _CountingPipeline.alive == 0
    # The registry entry is the prepared handle's runner, which holds no
    # pipeline: recording it kept nothing resident.
    assert _state.loads.completed() == frozenset({TaskLoad.COREF})
    host = _state.loads.get(COREF_LOAD)
    assert host is not None
    closed_over = [cell.cell_contents for cell in host.runner.__closure__ or ()]
    assert any(isinstance(value, PreparedCoref) for value in closed_over)
    assert not any(isinstance(value, _CountingPipeline) for value in closed_over)


def test_ensure_task_prepares_the_model_on_a_profile_worker() -> None:
    start_worker(
        WorkerBootstrapRuntime(
            task=None, lang="eng", num_speakers=1, profile=WorkerProfile.STANZA
        )
    )
    assert _CountingPipeline.constructions == []

    ensure_task_loaded("coref")

    assert len(_CountingPipeline.constructions) == 1
    assert _state.loads.get(COREF_LOAD) is not None


def test_each_request_holds_the_model_only_for_its_batch(tmp_path) -> None:
    start_worker(_coref_worker("eng"))
    ensure_task_loaded("coref")

    first = _execute(_coref_request(tmp_path, "first"))
    assert _CountingPipeline.alive == 0
    second = _execute(_coref_request(tmp_path, "second"))
    assert _CountingPipeline.alive == 0

    # The preparation's build, then one per request; none outlives its batch.
    assert len(_CountingPipeline.constructions) == 3
    for response in (first, second):
        assert isinstance(response.outcome, ExecuteSuccessV2)
        assert isinstance(response.result, CorefResultV2)
        item = response.result.items[0]
        assert item.kind == "resolved"
        assert item.engine == ENGINE


def test_the_legacy_batch_route_uses_the_prepared_model() -> None:
    start_worker(_coref_worker("eng"))
    ensure_task_loaded("coref")

    response = _batch_infer(
        BatchInferRequest(
            task=InferTask.COREF, lang="eng", items=[{"sentences": [["she"]]}]
        )
    )

    assert _CountingPipeline.alive == 0
    assert response.results[0].result == {
        "kind": "resolved",
        "annotations": [],
        "engine": ENGINE,
    }


def test_a_request_on_a_worker_without_the_model_is_model_unavailable(
    tmp_path,
) -> None:
    start_worker(_coref_worker("eng"))

    response = _execute(_coref_request(tmp_path, "unloaded"))

    # Refused by name; the request did not build the model.
    assert _CountingPipeline.constructions == []
    assert isinstance(response.outcome, ExecuteErrorV2)
    assert response.outcome.code is ProtocolErrorCodeV2.MODEL_UNAVAILABLE
    assert "the coref load has not completed" in response.outcome.message
