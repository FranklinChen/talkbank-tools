"""Live worker-protocol V2 ASR executor.

The ASR load builds the host (``batchalign.worker._loads``): the runner over
the loaded model, the backend that loaded and the identity it recorded. The
Rust executor reads all three from the host it is handed, refuses a request
for another backend, and admits the identity against the request's pin.

**See also:** `../../INTERFACE_MAP.md` section "2. ASR Execution V2" for:
- Rust FFI function: `crates/batchalign-pyo3/src/worker_asr_exec.rs::execute_asr_request_v2()`
- Request/response types: `crates/batchalign-types/src/worker_v2/requests.rs`
- Full Rust/Python responsibility split and input/output contracts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batchalign.worker._types_v2 import ExecuteRequestV2, ExecuteResponseV2

if TYPE_CHECKING:
    from batchalign.worker._loads import LoadedLocalAsr, LoadedProviderAsr


def execute_asr_request_v2(
    request: ExecuteRequestV2,
    host: LoadedLocalAsr | LoadedProviderAsr,
) -> ExecuteResponseV2:
    """Execute one live V2 ASR request on the loaded ASR host."""

    import batchalign_core

    return ExecuteResponseV2.model_validate_json(
        batchalign_core.execute_asr_request_v2(request, host)
    )


__all__ = ["execute_asr_request_v2"]
