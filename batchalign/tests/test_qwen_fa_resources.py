"""Native Qwen processor/model shape planning over synthetic, public inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from batchalign.inference.qwen_forced_alignment import (
    LoadedQwenFaHost,
    QwenForcedAligner,
    resolve_qwen_language,
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
def native_qwen() -> LoadedQwenFaHost:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        Qwen2TokenizerFast,
        Qwen3ASRConfig,
        Qwen3ASRFeatureExtractor,
        Qwen3ASRForTokenClassification,
        Qwen3ASRProcessor,
    )

    vocabulary = {
        "<pad>": 0,
        "<eos>": 1,
        "<audio>": 2,
        "<timestamp>": 3,
        "<a>": 4,
        "</a>": 5,
        "hi": 6,
        "there": 7,
        "<unk>": 8,
    }
    core = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = Qwen2TokenizerFast(
        tokenizer_object=core,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
        extra_special_tokens={
            "audio_token": "<audio>",
            "audio_bos_token": "<a>",
            "audio_eos_token": "</a>",
        },
        additional_special_tokens=["<timestamp>"],
    )
    template = "{% for m in messages %}{% for c in m['content'] %}{% if c['type'] == 'audio' %}<a><audio></a>{% elif c['type'] == 'text' %}<timestamp> {{ c['text'] }} <timestamp>{% endif %}{% endfor %}{% endfor %}"
    processor = Qwen3ASRProcessor(
        feature_extractor=Qwen3ASRFeatureExtractor(
            feature_size=4, n_window=50, min_length=0
        ),
        tokenizer=tokenizer,
        chat_template=template,
    )
    config = Qwen3ASRConfig(
        audio_config={
            "model_type": "qwen3_asr_encoder",
            "num_mel_bins": 4,
            "d_model": 8,
            "encoder_layers": 1,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 16,
            "n_window": 50,
            "n_window_infer": 100,
            "downsample_hidden_size": 1,
            "max_position_embeddings": 13,
            "output_dim": 8,
        },
        text_config={
            "model_type": "qwen3",
            "vocab_size": len(tokenizer),
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 128,
        },
        audio_token_id=tokenizer.audio_token_id,
        timestamp_token_id=3,
        pad_token_id=0,
        eos_token_id=1,
        num_labels=16,
    )
    model = Qwen3ASRForTokenClassification(config).eval()
    model.set_attn_implementation("sdpa")
    return LoadedQwenFaHost(
        aligner=QwenForcedAligner(processor=processor, model=model),
        language=resolve_qwen_language("eng"),
    )


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
