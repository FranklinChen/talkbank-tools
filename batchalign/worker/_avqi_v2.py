"""Live worker-protocol V2 AVQI executor.

AVQI has no model; its load records the voice-quality analysis as its host
(``batchalign.worker._loads.LoadedAvqi``), so it is resolved like every other
task.

**See also:** `../../INTERFACE_MAP.md` section "5. Media Analysis V2: AVQI" for:
- Rust FFI function: `crates/batchalign-pyo3/src/worker_media_exec.rs::execute_avqi_request_v2()`
- Full Rust/Python responsibility split and input/output contracts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batchalign.inference.avqi import AvqiResponse, infer_avqi_prepared_audio
from batchalign.worker._loads import LoadedAvqi
from batchalign.worker._types_v2 import (
    ExecuteRequestV2,
    ExecuteResponseV2,
)

if TYPE_CHECKING:
    import numpy as np


def avqi_task() -> LoadedAvqi:
    """The AVQI load's host: the voice-quality analysis, no model."""

    def _run(
        cs_audio: np.ndarray,
        cs_sample_rate_hz: int,
        sv_audio: np.ndarray,
        sv_sample_rate_hz: int,
        cs_label: str,
        sv_label: str,
    ) -> AvqiResponse:
        return infer_avqi_prepared_audio(
            cs_audio,
            cs_sample_rate_hz,
            sv_audio,
            sv_sample_rate_hz,
            cs_label=cs_label,
            sv_label=sv_label,
        )

    return LoadedAvqi(runner=_run)


def execute_avqi_request_v2(
    request: ExecuteRequestV2,
    host: LoadedAvqi,
) -> ExecuteResponseV2:
    """Execute one live V2 AVQI request through the Rust control plane."""

    import batchalign_core

    return ExecuteResponseV2.model_validate_json(
        batchalign_core.execute_avqi_request_v2(request, host.runner)
    )


__all__ = [
    "avqi_task",
    "execute_avqi_request_v2",
]
