"""Tests for the infer-era worker IPC contract between Rust and Python.

Cross-language contract: this is the Python half. The Rust half lives in
``crates/batchalign/tests/worker_protocol_v2_compat.rs``. Both sides
must independently verify that the wire format roundtrips correctly, a
change to an IPC type must update both Rust and Python models.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from batchalign.worker import (
    BatchInferRequest,
    BatchInferResponse,
    CapabilitiesResponse,
    HealthResponse,
    InferRequest,
    InferResponse,
    InferTask,
)
from batchalign.worker._types import ItemFailed, ItemProduced


def test_health_response() -> None:
    """HealthResponse serializes with expected fields."""
    resp = HealthResponse(
        status="ok",
        command="infer:morphosyntax",
        lang="eng",
        pid=12345,
        uptime_s=120.5,
    )
    data = json.loads(resp.model_dump_json())
    assert data["status"] == "ok"
    assert data["command"] == "infer:morphosyntax"
    assert data["pid"] == 12345


def test_capabilities_response() -> None:
    """CapabilitiesResponse serializes with expected fields."""
    resp = CapabilitiesResponse(
        commands=["align", "morphotag", "opensmile"],
        free_threaded=True,
        infer_tasks=[],
        engine_versions={},
    )
    data = json.loads(resp.model_dump_json())
    assert data["commands"] == ["align", "morphotag", "opensmile"]
    assert data["free_threaded"] is True


def test_capabilities_response_with_infer_fields() -> None:
    """CapabilitiesResponse includes infer_tasks and engine_versions."""
    resp = CapabilitiesResponse(
        commands=["morphotag"],
        free_threaded=False,
        infer_tasks=[InferTask.MORPHOSYNTAX, InferTask.FA],
        # Only forced alignment's entry names an engine; the rest are null.
        engine_versions={
            InferTask.MORPHOSYNTAX: None,
            InferTask.FA: "wave2vec-fa-v1",
        },
    )
    data = json.loads(resp.model_dump_json())
    assert data["infer_tasks"] == ["morphosyntax", "fa"]
    assert data["engine_versions"] == {"morphosyntax": None, "fa": "wave2vec-fa-v1"}


def test_capabilities_response_reports_an_unnamed_engine_as_null() -> None:
    """A supported task whose engine the worker cannot name travels as null."""
    resp = CapabilitiesResponse(
        commands=[],
        free_threaded=False,
        infer_tasks=[InferTask.ASR],
        engine_versions={InferTask.ASR: None},
    )
    data = json.loads(resp.model_dump_json())
    assert data["engine_versions"] == {"asr": None}


@pytest.mark.parametrize(
    "name",
    ["", "  ", " stanza-1.9.2", "stanza|1.9.2", "stanza;1.9.2", "stanza]1", "a\nb"],
)
def test_capabilities_response_refuses_an_unreportable_engine_name(name: str) -> None:
    """A name a provenance stamp cannot hold fails here, not at the Rust gate."""
    with pytest.raises(ValidationError):
        CapabilitiesResponse(
            commands=[],
            free_threaded=False,
            infer_tasks=[InferTask.FA],
            engine_versions={InferTask.FA: name},
        )


def test_capabilities_response_missing_infer_fields_is_rejected() -> None:
    """CapabilitiesResponse requires infer_tasks and engine_versions."""
    with pytest.raises(ValidationError):
        CapabilitiesResponse(commands=["morphotag"], free_threaded=False)


def test_infer_request_serialization() -> None:
    """InferRequest serializes with task, lang, and payload."""
    req = InferRequest(
        task=InferTask.MORPHOSYNTAX,
        lang="eng",
        payload={
            "words": ["the", "dog", "runs"],
            "terminator": ".",
            "special_forms": [],
            "lang": "eng",
        },
    )
    data = json.loads(req.model_dump_json())
    assert data["task"] == "morphosyntax"
    assert data["lang"] == "eng"
    assert data["payload"]["words"] == ["the", "dog", "runs"]


def test_infer_response_success() -> None:
    """A produced item's outcome carries its result and nothing else."""
    resp = InferResponse.unexecuted(
        ItemProduced(
            result={"mor": "det|the n|dog v|run-3S", "gra": "1|2|DET 2|3|SUBJ 3|0|ROOT"}
        )
    )
    data = json.loads(resp.model_dump_json())
    assert data["outcome"]["kind"] == "produced"
    assert data["outcome"]["result"]["mor"].startswith("det|the")
    assert "error" not in data["outcome"]
    assert resp.result is not None and resp.error is None


def test_infer_response_elapsed_has_no_default() -> None:
    """Every item says whether it was timed; there is no zero to fall back on."""
    with pytest.raises(ValidationError):
        InferResponse(outcome=ItemFailed(error="x"))  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        InferResponse(outcome=ItemFailed(error="x"), elapsed_s=-0.5)


def test_an_item_is_a_result_or_a_failure_never_both_or_neither() -> None:
    """The outcome is a union: both fields, or neither, do not parse."""
    for malformed in (
        {"outcome": {"kind": "produced", "result": 1, "error": "x"}, "elapsed_s": None},
        {"outcome": {"kind": "failed"}, "elapsed_s": None},
        {"result": 1, "error": "x", "elapsed_s": None},
        {"elapsed_s": None},
    ):
        with pytest.raises(ValidationError):
            InferResponse.model_validate(malformed)


def test_unexecuted_item_writes_null_elapsed() -> None:
    """An item no work ran for reports no time (null), never 0.0."""
    resp = InferResponse.unexecuted(ItemFailed(error="Invalid batch item"))
    data = json.loads(resp.model_dump_json())
    assert data == {
        "outcome": {"kind": "failed", "error": "Invalid batch item"},
        "elapsed_s": None,
    }


def test_timed_item_reports_its_own_work(monkeypatch) -> None:
    """``timed`` reads the clock around the work it runs, and nothing else."""
    readings = iter([10.0, 12.5])
    monkeypatch.setattr(
        "batchalign.worker._types.time.monotonic", lambda: next(readings)
    )
    resp = InferResponse.timed(lambda: ItemProduced(result={"kind": "ok"}))
    assert resp.result == {"kind": "ok"}
    assert resp.error is None
    assert resp.elapsed_s == 2.5


def test_infer_response_error() -> None:
    """A failed item's outcome carries its error and no result."""
    resp = InferResponse.unexecuted(
        ItemFailed(error="infer task 'morphosyntax' not yet implemented")
    )
    data = json.loads(resp.model_dump_json())
    assert data["outcome"]["kind"] == "failed"
    assert "not yet implemented" in data["outcome"]["error"]
    assert resp.result is None


def test_batch_infer_request_serialization() -> None:
    """BatchInferRequest serializes items as a list."""
    req = BatchInferRequest(
        task=InferTask.MORPHOSYNTAX,
        lang="eng",
        items=[
            {"words": ["hello"], "terminator": ".", "special_forms": [], "lang": "eng"},
            {"words": ["world"], "terminator": ".", "special_forms": [], "lang": "eng"},
        ],
    )
    data = json.loads(req.model_dump_json())
    assert data["task"] == "morphosyntax"
    assert len(data["items"]) == 2
    assert data["items"][0]["words"] == ["hello"]


def test_batch_infer_response_serialization() -> None:
    """BatchInferResponse contains a list of InferResponse."""
    resp = BatchInferResponse(
        results=[
            InferResponse.unexecuted(ItemProduced(result={"mor": "n|hello"})),
            InferResponse.unexecuted(ItemFailed(error="failed")),
        ],
    )
    data = json.loads(resp.model_dump_json())
    assert len(data["results"]) == 2
    assert data["results"][0]["outcome"]["result"]["mor"] == "n|hello"
    assert data["results"][1]["outcome"]["error"] == "failed"


def test_rust_can_parse_python_infer_request() -> None:
    """Verify the JSON shape Python produces matches what Rust expects."""
    rust_json = json.dumps(
        {
            "task": "morphosyntax",
            "lang": "eng",
            "payload": {
                "words": ["hello", "world"],
                "terminator": ".",
                "special_forms": [],
                "lang": "eng",
            },
        }
    )
    req = InferRequest.model_validate_json(rust_json)
    assert req.task == InferTask.MORPHOSYNTAX
    assert req.payload["words"] == ["hello", "world"]  # type: ignore[index]


def test_python_can_parse_rust_infer_response() -> None:
    """Verify Python can parse the JSON that Rust produces for InferResponse."""
    rust_json = json.dumps(
        {
            "outcome": {
                "kind": "produced",
                "result": {"mor": "n|hello n|world", "gra": "1|2|SUBJ 2|0|ROOT"},
            },
            "elapsed_s": 0.123,
        }
    )
    resp = InferResponse.model_validate_json(rust_json)
    assert resp.result is not None
    assert resp.error is None
    assert resp.elapsed_s == 0.123
