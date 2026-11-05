"""The loaded Whisper host prepares actual model shapes without audio inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pytest

from batchalign.inference.types import WhisperFAHandle, WhisperModelIdentity
from batchalign.tests._snapshots import fake_resolve_transformers
from batchalign.worker._model_loading.forced_alignment import load_fa_engine
from batchalign.worker._types import InferTask, WorkerBootstrapRuntime
from batchalign.worker._types_v2 import (
    FaBackendV2,
    FaConstraintUnitV2,
    FaModelConstraintV2,
    FaPreparationBindingV2,
    FaPreparedV2,
    FaProjectedMemoryV2,
    FaRefusedV2,
    FaTextModeV2,
    FaWhisperBasisV2,
)

if TYPE_CHECKING:
    from transformers import WhisperProcessor

    from batchalign.worker._loads import LoadedWhisperFa


@dataclass(frozen=True)
class DeclaredFeatureGeometry:
    feature_size: int = 4
    nb_max_frames: int = 32
    n_samples: int = 5120
    hop_length: int = 160
    n_fft: int = 400


class MetadataTokenizer:
    prefix_tokens = [1]
    eos_token_id = 2

    def __call__(self, text: str) -> dict[str, list[int]]:
        return {"input_ids": [1, *[3 for _ in text.split()], 2]}


@dataclass(frozen=True)
class MetadataProcessor:
    feature_extractor: DeclaredFeatureGeometry
    tokenizer: MetadataTokenizer


def tiny_whisper_handle() -> WhisperFAHandle:
    """An actual tiny model with declared text-only processor metadata."""
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    config = WhisperConfig(
        vocab_size=32,
        num_mel_bins=4,
        d_model=8,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_ffn_dim=16,
        max_source_positions=16,
        max_target_positions=12,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
        median_filter_width=3,
        attn_implementation="eager",
    )
    model = WhisperForConditionalGeneration(config).eval()
    model.generation_config.alignment_heads = [[0, 0]]
    # This double supplies only processor metadata and text tokenization.
    # The actual transformer architecture is projected with virtual tensors.
    processor = cast(
        "WhisperProcessor",
        MetadataProcessor(DeclaredFeatureGeometry(), MetadataTokenizer()),
    )
    handle = WhisperFAHandle(
        model,
        processor,
        16000,
        model_identity=WhisperModelIdentity("declared-tiny-whisper", "a" * 40),
    )
    return handle


@pytest.fixture
def loaded_whisper(monkeypatch: pytest.MonkeyPatch) -> LoadedWhisperFa:
    handle = tiny_whisper_handle()
    monkeypatch.setattr(
        "batchalign.inference.fa.load_whisper_fa", lambda *_args, **_kwargs: handle
    )
    monkeypatch.setattr(
        "batchalign.worker._model_loading.forced_alignment.resolve_transformers",
        fake_resolve_transformers,
    )
    host = load_fa_engine(
        WorkerBootstrapRuntime(
            task=InferTask.FA,
            lang="eng",
            num_speakers=1,
            engine_overrides={"fa": "whisper"},
        )
    )
    from batchalign.worker._loads import LoadedWhisperFa

    assert isinstance(host, LoadedWhisperFa)
    return host


def binding(host: LoadedWhisperFa, samples: int) -> str:
    return FaPreparationBindingV2(
        execution_request_id="whisper-resources",
        backend=FaBackendV2.WHISPER,
        engine=host.engine,
        payload_ref_id="words",
        audio_ref_id="audio",
        samples=samples,
        sample_rate_hz=16000,
        text_mode=FaTextModeV2.SPACE_JOINED,
        attention_capture=None,
    ).model_dump_json()


def test_loaded_whisper_prepares_virtual_encoder_decoder_and_alignment_workspace(
    loaded_whisper: LoadedWhisperFa,
) -> None:
    outcome = loaded_whisper.prepare(binding(loaded_whisper, 5120), ["hi"], "hi")
    assert isinstance(outcome, FaPreparedV2)
    assert isinstance(outcome.plan.memory, FaProjectedMemoryV2)
    assert isinstance(outcome.plan.memory.basis, FaWhisperBasisV2)
    assert outcome.plan.memory.workspace > 0
    assert outcome.plan.memory.model > 0


@pytest.mark.parametrize(
    "samples,text,unit,provided,maximum",
    [
        (5121, "hi", FaConstraintUnitV2.AUDIO_SAMPLES, 5121, 5120),
        (5120, " ".join(["hi"] * 12), FaConstraintUnitV2.TEXT_LABELS, 14, 12),
    ],
)
def test_loaded_model_context_constraints_precede_any_virtual_forward(
    loaded_whisper: LoadedWhisperFa,
    samples: int,
    text: str,
    unit: FaConstraintUnitV2,
    provided: int,
    maximum: int,
) -> None:
    outcome = loaded_whisper.prepare(
        binding(loaded_whisper, samples), text.split(), text
    )
    assert isinstance(outcome, FaRefusedV2)
    assert isinstance(outcome.cause, FaModelConstraintV2)
    assert (outcome.cause.unit, outcome.cause.provided, outcome.cause.maximum) == (
        unit,
        provided,
        maximum,
    )


def test_fixed_metadata_refusal_does_not_revalidate_a_mutated_producer() -> None:
    """A failed loaded epoch cannot become ready through a later request."""
    from batchalign.worker._loads import LoadedWhisperFa
    from batchalign.worker._types_v2 import FaProjectionUnavailableV2

    handle = tiny_whisper_handle()
    handle.model.generation_config.update(alignment_heads=[])
    host = LoadedWhisperFa.from_model(handle)
    handle.model.generation_config.update(alignment_heads=[[0, 0]])
    result = host.prepare(binding(host, 5120), ["hi"], "hi")
    assert isinstance(result, FaRefusedV2)
    assert isinstance(result.cause, FaProjectionUnavailableV2)
    assert "no alignment heads" in result.cause.message
