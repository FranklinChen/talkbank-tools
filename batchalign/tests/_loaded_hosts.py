"""Loaded hosts and load registries for worker tests.

A test hands the worker the hosts a load would have built, exactly as
production does: a ``TaskLoadRegistry`` holding them, passed to
``execute_on_loads`` or installed as ``_state.loads`` for the duration of a
test. Nothing is ever written to worker state field by field; there are no
such fields.

Deliberately a plain module rather than a ``conftest.py`` fixture, like
``_asr_model_pins``: ordinary helpers want these too. The leading underscore
keeps pytest from collecting it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from batchalign.tests._asr_model_pins import loaded_identity
from batchalign.worker._loads import (
    ASR_LOAD,
    FA_LOAD,
    LoadedDiarization,
    LoadedLocalAsr,
    LoadedMorphosyntax,
    LoadedProviderAsr,
    LoadedSpeakerEmbedding,
)
from batchalign.worker._types import LoadKey, TaskLoadRegistry
from batchalign.worker._types_v2 import (
    AsrBackendV2,
    ExecuteRequestV2,
    ExecuteSuccessV2,
    FaBackendV2,
    FaExecuteV2,
    FaMmsCpuBasisV2,
    FaPreparationBindingV2,
    FaPreparationOutcomeV2,
    FaPreparationPlanV2,
    FaPreparationResultPayloadV2,
    FaPreparedV2,
    FaPrepareV2,
    FaProjectedMemoryV2,
    ForcedAlignmentRequestV2,
    SpeakerBackendV2,
    SpeakerEmbeddingBackendV2,
)


def loads(*records: tuple[LoadKey[Any], object]) -> TaskLoadRegistry:
    """A registry holding exactly these ``(key, host)`` loads."""
    registry = TaskLoadRegistry()
    for key, host in records:
        registry.record(key, host)
    return registry


def local_asr(
    runner: Callable[..., Any],
    backend: AsrBackendV2 = AsrBackendV2.LOCAL_WHISPER,
) -> LoadedLocalAsr:
    """A Whisper ASR host over ``runner``, reporting what ``backend`` pins."""
    return LoadedLocalAsr(
        backend=backend, runner=runner, model_identity=loaded_identity(backend)
    )


def provider_asr(
    backend: AsrBackendV2, runner: Callable[..., Any]
) -> LoadedProviderAsr:
    """A provider ASR host over ``runner``, recording what ``backend`` pins."""
    return LoadedProviderAsr(
        backend=backend, runner=runner, model_identity=loaded_identity(backend)
    )


def asr_loads(host: object) -> TaskLoadRegistry:
    """A registry whose only load is ASR, built as ``host``."""
    return loads((ASR_LOAD, host))


WHISPER_FA_TEST_ENGINE = "whisper-fa-test"
"""The FA identity a test Whisper FA host reports."""


@dataclass(frozen=True, slots=True)
class DeclaredFaHost:
    """A declared foreign callback for protocol tests, never a model load.

    Its positive allocation report is synthetic contract data. It establishes
    no real model capacity, numerical inference or production admission.
    """

    backend: FaBackendV2
    runner: Callable[..., Any]
    engine: str

    def prepare(
        self, binding_json: str, words: list[str], text: str
    ) -> FaPreparationOutcomeV2:
        binding = FaPreparationBindingV2.model_validate_json(binding_json)
        return FaPreparedV2(
            plan=FaPreparationPlanV2(
                binding=binding,
                memory=FaProjectedMemoryV2(
                    workspace=1,
                    model=1,
                    basis=FaMmsCpuBasisV2(
                        torch_version="declared-test",
                        torchaudio_revision="declared-test",
                    ),
                ),
            )
        )


def whisper_fa(
    runner: Callable[..., Any], *, engine: str = WHISPER_FA_TEST_ENGINE
) -> DeclaredFaHost:
    """A Whisper FA host over ``runner``."""
    return DeclaredFaHost(backend=FaBackendV2.WHISPER, runner=runner, engine=engine)


def word_fa(
    runner: Callable[..., Any], backend: FaBackendV2 = FaBackendV2.WAVE2VEC
) -> DeclaredFaHost:
    """A word-list FA host (Wave2Vec by default) over ``runner``."""
    return DeclaredFaHost(
        backend=backend, runner=runner, engine=f"{backend.value}-fa-test"
    )


def prepare_fa_fixture_on(
    request: ExecuteRequestV2, host: DeclaredFaHost
) -> ExecuteRequestV2:
    """Consume real native preparation before a declared host's execute fixture.

    This boundary fixture does not grant a host lease. The Rust pool's
    lease tests exercise that independent transition.
    """
    from batchalign.worker._fa_v2 import execute_forced_alignment_request_v2

    assert isinstance(request.payload, ForcedAlignmentRequestV2)
    query = request.model_copy(
        update={
            "request_id": f"{request.request_id}:fixture-prepare",
            "payload": request.payload.model_copy(
                update={
                    "phase": FaPrepareV2(execution_request_id=request.request_id),
                }
            ),
        }
    )
    response = execute_forced_alignment_request_v2(query, host)
    assert isinstance(response.outcome, ExecuteSuccessV2), response
    assert isinstance(response.result, FaPreparationResultPayloadV2)
    assert isinstance(response.result.outcome, FaPreparedV2)
    return request.model_copy(
        update={
            "payload": request.payload.model_copy(
                update={
                    "phase": FaExecuteV2(plan=response.result.outcome.plan),
                }
            ),
        }
    )


def prepare_fa_fixture(request: ExecuteRequestV2) -> ExecuteRequestV2:
    """Prepare using the declared identity shared by protocol fixture hosts."""
    assert isinstance(request.payload, ForcedAlignmentRequestV2)

    def no_inference(*_args: object) -> object:
        raise AssertionError("preparation must not run inference")

    backend = request.payload.backend
    match backend:
        case FaBackendV2.WHISPER:
            host = whisper_fa(no_inference)
        case FaBackendV2.WAVE2VEC | FaBackendV2.WAV2VEC_CANTO | FaBackendV2.QWEN3:
            host = word_fa(no_inference, backend)
    return prepare_fa_fixture_on(request, host)


def fa_loads(host: object) -> TaskLoadRegistry:
    """A registry whose only load is FA, built as ``host``."""
    return loads((FA_LOAD, host))


def diarization(
    runner: Callable[..., Any],
    backend: SpeakerBackendV2 = SpeakerBackendV2.PYANNOTE,
) -> LoadedDiarization:
    """A diarization host (local pyannote by default) over ``runner``."""
    return LoadedDiarization(backend=backend, runner=runner)


def speaker_embedding(runner: Callable[..., Any]) -> LoadedSpeakerEmbedding:
    """A speaker-embedding host over ``runner``."""
    return LoadedSpeakerEmbedding(
        backend=SpeakerEmbeddingBackendV2.PYANNOTE, runner=runner
    )


def morphosyntax(runner: Callable[..., Any]) -> LoadedMorphosyntax:
    """A morphosyntax host over a one-argument ``runner(req)``.

    The real runner also takes the request's progress callback; a test fake
    that ignores progress is adapted here, once.
    """
    return LoadedMorphosyntax(runner=lambda req, _progress: runner(req))
