"""Live worker-protocol V2 speaker diarization executor.

The speaker load builds the host (``batchalign.worker._loads``): local pyannote
over the pipeline it loaded, or pyannoteAI, which is hosted and loaded
nothing. The Rust executor reads the host's backend and runner and refuses a
request for another backend.

**See also:** `../../INTERFACE_MAP.md` section "6. Media Analysis V2: Speaker Diarization" for:
- Rust FFI function: `crates/batchalign-pyo3/src/worker_media_exec.rs::execute_speaker_request_v2()`
- Full Rust/Python responsibility split and input/output contracts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batchalign.worker._types_v2 import ExecuteRequestV2, ExecuteResponseV2

if TYPE_CHECKING:
    from batchalign.worker._loads import LoadedDiarization


def execute_speaker_request_v2(
    request: ExecuteRequestV2,
    host: LoadedDiarization,
) -> ExecuteResponseV2:
    """Execute one live V2 speaker request on the loaded diarization host."""

    import batchalign_core

    return ExecuteResponseV2.model_validate_json(
        batchalign_core.execute_speaker_request_v2(request, host)
    )


__all__ = ["execute_speaker_request_v2"]
