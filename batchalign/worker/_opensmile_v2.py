"""Live worker-protocol V2 openSMILE executor.

openSMILE has no model; its load records the feature extractor as its host
(``batchalign.worker._loads.LoadedOpenSmile``), so it is resolved like every
other task.

**See also:** `../../INTERFACE_MAP.md` section "4. Media Analysis V2: OpenSMILE" for:
- Rust FFI function: `crates/batchalign-pyo3/src/worker_media_exec.rs::execute_opensmile_request_v2()`
- Full Rust/Python responsibility split and input/output contracts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batchalign.inference.opensmile import (
    OpenSmileResponse,
    infer_opensmile_prepared_audio,
)
from batchalign.worker._loads import LoadedOpenSmile
from batchalign.worker._types_v2 import (
    ExecuteRequestV2,
    ExecuteResponseV2,
)

if TYPE_CHECKING:
    import numpy as np


def opensmile_task() -> LoadedOpenSmile:
    """The openSMILE load's host: the feature extractor, no model."""

    def _run(
        audio: np.ndarray,
        sample_rate_hz: int,
        feature_set: str,
        feature_level: str,
        audio_label: str,
    ) -> OpenSmileResponse:
        return infer_opensmile_prepared_audio(
            audio,
            sample_rate_hz,
            feature_set=feature_set,
            feature_level=feature_level,
            audio_label=audio_label,
        )

    return LoadedOpenSmile(runner=_run)


def execute_opensmile_request_v2(
    request: ExecuteRequestV2,
    host: LoadedOpenSmile,
) -> ExecuteResponseV2:
    """Execute one live V2 openSMILE request through the Rust control plane."""

    import batchalign_core

    return ExecuteResponseV2.model_validate_json(
        batchalign_core.execute_opensmile_request_v2(request, host.runner)
    )


__all__ = [
    "execute_opensmile_request_v2",
    "opensmile_task",
]
