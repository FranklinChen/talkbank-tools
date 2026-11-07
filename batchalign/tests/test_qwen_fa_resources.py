"""Native Qwen processor/model shape planning over synthetic, public inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from batchalign.inference.qwen_forced_alignment import (
    LoadedQwenFaHost,
)
from batchalign.tests._snapshots import fake_resolve_transformers
from batchalign.worker._model_loading.forced_alignment import load_fa_engine
from batchalign.worker._types import InferTask, WorkerBootstrapRuntime
from batchalign.worker._types_v2 import (
    FaBackendV2,
    FaPreparationBindingV2,
    FaPreparedV2,
    FaProjectedMemoryV2,
    FaQwenBasisV2,
    FaTextModeV2,
)

if TYPE_CHECKING:
    from batchalign.worker._loads import LoadedWordFa


@pytest.fixture
def loaded_qwen(
    native_qwen: LoadedQwenFaHost, monkeypatch: pytest.MonkeyPatch
) -> LoadedWordFa:
    monkeypatch.setattr(
        "batchalign.inference.qwen_forced_alignment.load_qwen_fa",
        lambda *_args, **_kwargs: native_qwen,
    )
    monkeypatch.setattr(
        "batchalign.worker._model_loading.forced_alignment.resolve_transformers",
        fake_resolve_transformers,
    )
    from batchalign.worker._loads import LoadedWordFa

    host = load_fa_engine(
        WorkerBootstrapRuntime(
            task=InferTask.FA,
            lang="eng",
            num_speakers=1,
            engine_overrides={"fa": "qwen3_fa"},
        )
    )
    assert isinstance(host, LoadedWordFa)
    return host


def binding(host: LoadedWordFa, samples: int) -> FaPreparationBindingV2:
    return FaPreparationBindingV2(
        execution_request_id="qwen-resource-fixture",
        backend=FaBackendV2.QWEN3,
        engine=host.engine,
        payload_ref_id="words",
        audio_ref_id="audio",
        samples=samples,
        sample_rate_hz=16000,
        text_mode=FaTextModeV2.SPACE_JOINED,
        attention_capture=None,
    )


def test_loaded_qwen_projects_its_native_components_without_a_duration_cutoff(
    loaded_qwen: LoadedWordFa, native_qwen: LoadedQwenFaHost
) -> None:
    model = native_qwen.aligner.model
    implementations = tuple(
        module.config._attn_implementation
        for module in (model, model.model.audio_tower, model.model.language_model)
    )
    parameters = tuple(
        (parameter.data_ptr(), parameter._version) for parameter in model.parameters()
    )
    small = loaded_qwen.prepare(
        binding(loaded_qwen, 3200).model_dump_json(), ["hi", "there"], "hi there"
    )
    long = loaded_qwen.prepare(
        binding(loaded_qwen, 60 * 16000).model_dump_json(), ["hi", "there"], "hi there"
    )
    assert isinstance(small, FaPreparedV2)
    assert isinstance(long, FaPreparedV2)
    assert isinstance(small.plan.memory, FaProjectedMemoryV2)
    assert isinstance(long.plan.memory, FaProjectedMemoryV2)
    assert isinstance(long.plan.memory.basis, FaQwenBasisV2)
    assert long.plan.memory.workspace > small.plan.memory.workspace
    assert long.plan.memory.model == small.plan.memory.model
    assert implementations == tuple(
        module.config._attn_implementation
        for module in (model, model.model.audio_tower, model.model.language_model)
    )
    assert parameters == tuple(
        (parameter.data_ptr(), parameter._version) for parameter in model.parameters()
    )
    assert not model.training


@pytest.mark.parametrize(
    "changed", [{"backend": FaBackendV2.WHISPER}, {"engine": "another-model"}]
)
def test_prepared_qwen_cannot_claim_another_model(
    native_qwen: LoadedQwenFaHost, loaded_qwen: LoadedWordFa, changed: dict[str, object]
) -> None:
    from batchalign.inference.fa_qwen_resources import QwenPreparedInputs
    from batchalign.inference.fa_virtual_resources import FaResourceProjectionError

    request = FaPreparationBindingV2.model_validate(
        binding(loaded_qwen, 3200).model_dump() | changed
    )
    with pytest.raises(FaResourceProjectionError, match="different loaded model"):
        QwenPreparedInputs.admit(native_qwen, request, ["hi", "there"])


def test_prepared_qwen_shapes_match_the_actual_processor(
    native_qwen: LoadedQwenFaHost, loaded_qwen: LoadedWordFa
) -> None:
    from batchalign.inference.fa_qwen_resources import QwenPreparedInputs

    prepared = QwenPreparedInputs.admit(
        native_qwen, binding(loaded_qwen, 3200), ["hi", "there"]
    )
    actual, word_lists = native_qwen.aligner.processor.prepare_forced_aligner_inputs(
        audio=np.zeros(3200, dtype=np.float32),
        transcript="hi there",
        language="English",
    )
    assert prepared.token_ids == tuple(actual["input_ids"][0].tolist())
    assert prepared.padded_frames == actual["input_features"].shape[-1]
    assert prepared.audio_tokens == int(
        (actual["input_ids"] == native_qwen.aligner.model.config.audio_token_id).sum()
    )
    assert prepared.units == tuple(word_lists[0])


def test_fixed_chunk_geometry_refusal_is_retained_for_the_loaded_epoch(
    native_qwen: LoadedQwenFaHost, loaded_qwen: LoadedWordFa
) -> None:
    from batchalign.inference.fa_qwen_resources import qwen_request_preparer
    from batchalign.worker._types_v2 import FaProjectionUnavailableV2, FaRefusedV2

    feature = native_qwen.aligner.processor.feature_extractor
    original = feature.n_window
    feature.n_window = original + 1
    prepare = qwen_request_preparer(native_qwen)
    feature.n_window = original
    result = prepare(binding(loaded_qwen, 3200).model_dump_json(), ["hi"], "hi")
    assert isinstance(result, FaRefusedV2)
    assert isinstance(result.cause, FaProjectionUnavailableV2)
    assert "chunk geometry" in result.cause.message
