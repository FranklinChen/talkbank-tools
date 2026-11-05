"""The imported native FA boundary cannot turn candidates into execution."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from batchalign.tests._loaded_hosts import whisper_fa
from batchalign.tests.test_worker_fa_v2 import _make_request
from batchalign.worker._fa_v2 import execute_forced_alignment_request_v2
from batchalign.worker._types_v2 import (
    ExecuteErrorV2,
    ExecuteRequestV2,
    ExecuteSuccessV2,
    FaBackendV2,
    FaCandidateV2,
    FaExecuteV2,
    FaPreparationResultPayloadV2,
    FaPreparedV2,
    FaPrepareV2,
    FaTextModeV2,
    ProtocolErrorCodeV2,
)


def must_not_infer(*_args: object) -> object:
    raise AssertionError("this boundary must not read PCM or run an aligner")


def request_fixture(tmp_path: Path) -> ExecuteRequestV2:
    return _make_request(
        tmp_path, backend=FaBackendV2.WHISPER, text_mode=FaTextModeV2.SPACE_JOINED
    )


def test_candidate_does_not_execute_without_preparation(tmp_path: Path) -> None:
    request = request_fixture(tmp_path)
    request.payload.phase = FaCandidateV2()
    response = execute_forced_alignment_request_v2(request, whisper_fa(must_not_infer))
    assert isinstance(response.outcome, ExecuteErrorV2)
    assert response.outcome.code is ProtocolErrorCodeV2.INVALID_PAYLOAD
    assert "requires preparation" in response.outcome.message


def test_prepare_needs_audio_metadata_but_does_not_read_pcm(tmp_path: Path) -> None:
    request = request_fixture(tmp_path)
    request.payload.phase = FaPrepareV2(execution_request_id=request.request_id)
    (tmp_path / "audio.pcm").unlink()
    response = execute_forced_alignment_request_v2(request, whisper_fa(must_not_infer))
    assert isinstance(response.outcome, ExecuteSuccessV2)
    assert isinstance(response.result, FaPreparationResultPayloadV2)
    assert isinstance(response.result.outcome, FaPreparedV2)
    assert response.result.outcome.plan.binding.samples == 4


@pytest.mark.parametrize(
    "changed",
    [
        {"engine": "another-engine"},
        {"samples": 5},
        {"text_mode": FaTextModeV2.CHAR_JOINED},
    ],
)
def test_execution_cannot_reuse_another_inputs_plan(
    tmp_path: Path, changed: dict[str, object]
) -> None:
    request = request_fixture(tmp_path)
    assert isinstance(request.payload.phase, FaExecuteV2)
    plan = request.payload.phase.plan
    binding = type(plan.binding).model_validate(plan.binding.model_dump() | changed)
    request.payload.phase = FaExecuteV2(
        plan=plan.model_copy(update={"binding": binding})
    )
    (tmp_path / "audio.pcm").unlink()
    response = execute_forced_alignment_request_v2(request, whisper_fa(must_not_infer))
    assert isinstance(response.outcome, ExecuteErrorV2)
    assert response.outcome.code is ProtocolErrorCodeV2.INVALID_PAYLOAD
    assert "does not bind" in response.outcome.message


def test_missing_phase_is_not_defaulted_to_execution(tmp_path: Path) -> None:
    request = request_fixture(tmp_path).model_dump(mode="json")
    del request["payload"]["phase"]
    with pytest.raises(ValidationError):
        ExecuteRequestV2.model_validate(request)
