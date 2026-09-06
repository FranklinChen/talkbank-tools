"""Thin Python wrapper over Rust-owned worker stdio op dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from pydantic import ValidationError

import batchalign_core
from batchalign.worker._execute_v2 import execute_request_v2
from batchalign.worker._handlers import _capabilities, _ensure_task, _health
from batchalign.worker._infer import _batch_infer, _infer
from batchalign.worker._types import (
    BatchInferRequest,
    InferRequest,
    WorkerJSONValue,
)
from batchalign.worker._types_v2 import ExecuteRequestV2


@dataclass(frozen=True, slots=True)
class ProtocolDispatchResult:
    """One decoded protocol result ready for JSON-line emission."""

    payload: dict[str, WorkerJSONValue]
    should_shutdown: bool = False


PendingProtocolRequest = batchalign_core.PendingProtocolRequest


def prepare_protocol_message(
    message: object,
) -> PendingProtocolRequest | ProtocolDispatchResult:
    """Admit executable work or return a reply for the reader to handle now."""
    prepared = batchalign_core.prepare_protocol_message(message)
    if isinstance(prepared, batchalign_core.ImmediateProtocolReply):
        return ProtocolDispatchResult(
            payload=cast(dict[str, WorkerJSONValue], prepared.payload),
            should_shutdown=prepared.should_shutdown,
        )
    return prepared


def dispatch_protocol_message(message: object) -> ProtocolDispatchResult:
    """Dispatch a raw message synchronously, including reader control replies."""
    prepared = prepare_protocol_message(message)
    if isinstance(prepared, ProtocolDispatchResult):
        return prepared
    return dispatch_prepared_protocol_message(prepared)


def dispatch_prepared_protocol_message(
    request: PendingProtocolRequest,
) -> ProtocolDispatchResult:
    """Execute an admitted request; shutdown cannot enter this function."""
    payload = batchalign_core.dispatch_protocol_message(
        request,
        health_fn=_health,
        capabilities_fn=_capabilities,
        infer_fn=_infer,
        batch_infer_fn=_batch_infer,
        execute_v2_fn=execute_request_v2,
        ensure_task_fn=_ensure_task,
        infer_request_model=InferRequest,
        batch_infer_request_model=BatchInferRequest,
        execute_v2_request_model=ExecuteRequestV2,
        validation_error_type=ValidationError,
    )
    return ProtocolDispatchResult(
        payload=cast(dict[str, WorkerJSONValue], payload),
    )


__all__ = [
    "PendingProtocolRequest",
    "ProtocolDispatchResult",
    "dispatch_prepared_protocol_message",
    "dispatch_protocol_message",
    "prepare_protocol_message",
]
