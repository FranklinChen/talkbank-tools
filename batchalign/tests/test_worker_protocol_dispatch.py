"""Unit tests for the worker op registry and stdio dispatch layer."""

from __future__ import annotations

import pytest

from batchalign.worker._protocol_ops import (
    PendingProtocolRequest,
    dispatch_prepared_protocol_message,
    dispatch_protocol_message,
    prepare_protocol_message,
)


def test_prepared_operation_cannot_be_mutated_into_shutdown() -> None:
    message = {"op": "health"}
    prepared = prepare_protocol_message(message)
    assert isinstance(prepared, PendingProtocolRequest)
    message["op"] = "shutdown"
    prepared.message["op"] = "shutdown"
    response = dispatch_prepared_protocol_message(prepared)
    assert response.payload["op"] == "health"
    assert response.should_shutdown is False


def test_pending_request_has_no_python_constructor() -> None:
    with pytest.raises(TypeError):
        PendingProtocolRequest()


def test_invalid_unicode_operation_is_a_reply_not_a_reader_exception() -> None:
    response = dispatch_protocol_message({"op": "\ud800"})
    assert response.should_shutdown is False
    assert response.payload["op"] == "error"


def test_dispatch_protocol_message_wraps_health_response() -> None:
    """Health requests should produce a typed response envelope."""
    dispatch = dispatch_protocol_message({"op": "health"})

    assert dispatch.should_shutdown is False
    assert dispatch.payload["op"] == "health"
    response = dispatch.payload["response"]
    assert isinstance(response, dict)
    assert response["status"] in {"loading", "ok"}


def test_dispatch_protocol_message_rejects_unknown_op() -> None:
    """Unknown operations should return the standard protocol error envelope."""
    dispatch = dispatch_protocol_message({"op": "not-a-real-op"})

    assert dispatch.should_shutdown is False
    assert dispatch.payload == {
        "op": "error",
        "error": "unknown op: 'not-a-real-op'",
    }


def test_dispatch_protocol_message_requires_object_request() -> None:
    """Top-level requests must be JSON objects."""
    dispatch = dispatch_protocol_message("bad")

    assert dispatch.should_shutdown is False
    assert dispatch.payload == {
        "op": "error",
        "error": "request must be a JSON object",
    }


def test_dispatch_protocol_message_requires_mapping_request() -> None:
    """Request-bearing operations must include a mapping under ``request``."""
    dispatch = dispatch_protocol_message({"op": "infer", "request": "bad"})

    assert dispatch.should_shutdown is False
    assert dispatch.payload == {
        "op": "error",
        "error": "infer request must include mapping field 'request'",
    }


def test_dispatch_protocol_message_wraps_validation_errors() -> None:
    """Validation failures should stay protocol-level error envelopes."""
    dispatch = dispatch_protocol_message(
        {
            "op": "infer",
            "request": {"task": "not-a-real-task", "lang": "eng", "payload": {}},
        }
    )

    assert dispatch.should_shutdown is False
    assert dispatch.payload["op"] == "error"
    assert str(dispatch.payload["error"]).startswith("invalid infer request:")


def test_dispatch_protocol_message_handles_shutdown() -> None:
    """Shutdown should short-circuit the loop without a nested response body."""
    dispatch = dispatch_protocol_message({"op": "shutdown"})

    assert dispatch.should_shutdown is True
    assert dispatch.payload == {"op": "shutdown"}
