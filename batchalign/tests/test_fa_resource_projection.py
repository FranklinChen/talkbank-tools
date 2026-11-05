"""Request-shaped allocation planning uses virtual tensors, never recordings."""

from __future__ import annotations

from copy import copy

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensor

from batchalign.inference._admission import _FactoryAdmission
from batchalign.inference.fa_ctc_resources import (
    AdmittedCtcShape,
    MmsAlignedWords,
    MmsAlphabet,
    MmsTargetSequence,
)
from batchalign.inference.fa_native_resources import NativeCpuConvolutionWorkspace
from batchalign.inference.fa_resources import (
    AudioShape,
    ProjectedEncoderWorkspace,
    mms_request_preparer,
    project_encoder_workspace,
)
from batchalign.inference.fa_virtual_resources import (
    FaResourceProjectionError,
    ProjectedModelWorkspace,
    project_model_workspace,
)
from batchalign.inference.types import Wave2VecFAHandle


def mms_handle(model: nn.Module) -> Wave2VecFAHandle:
    """Pair an explicit synthetic encoder with the installed bundle alphabet."""
    from torchaudio.pipelines import MMS_FA

    return Wave2VecFAHandle(model, 16000, MmsAlphabet.admit(MMS_FA.get_dict()))


class QuadraticEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, None]:
        assert isinstance(audio, FakeTensor), "planning must allocate no real audio"
        scores = audio.unsqueeze(2) * audio.unsqueeze(1)
        return scores.sum(dim=2, keepdim=True) * self.scale, None


@pytest.mark.parametrize(
    "construct",
    [
        NativeCpuConvolutionWorkspace,
        MmsAlphabet,
        MmsTargetSequence,
        AdmittedCtcShape,
        MmsAlignedWords,
        ProjectedEncoderWorkspace,
        ProjectedModelWorkspace,
    ],
)
def test_admission_evidence_cannot_be_default_constructed(
    construct: type[_FactoryAdmission],
) -> None:
    with pytest.raises(TypeError, match="admission"):
        construct()  # type: ignore[call-arg]  # Intentional invalid-constructor boundary.


def test_workspace_follows_the_actual_input_shape_and_preserves_the_loaded_model() -> (
    None
):
    model = QuadraticEncoder().train()
    original = model.scale.detach().clone()
    small = project_encoder_workspace(
        model, AudioShape(samples=32), torch.device("cpu")
    )
    large = project_encoder_workspace(
        model, AudioShape(samples=128), torch.device("cpu")
    )
    assert small.encoded_frames == 32
    assert large.encoded_frames == 128
    assert large.peak_workspace_bytes > small.peak_workspace_bytes * 8
    assert large.peak_workspace_bytes >= 128 * 128 * torch.float32.itemsize
    assert model.training, "planning must not alter the live model's mode"
    assert model.scale.requires_grad
    assert torch.equal(model.scale.detach(), original)

    class ReusesServingModel(QuadraticEncoder):
        def __deepcopy__(self, memo: dict[int, object]) -> ReusesServingModel:
            return self

    reused = ReusesServingModel().train()
    with pytest.raises(FaResourceProjectionError):
        project_encoder_workspace(reused, AudioShape(samples=32), torch.device("cpu"))
    assert reused.training, "copy rejection must precede any serving-mode mutation"
    assert reused.scale.requires_grad

    class ReusesServingChild(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dropout = nn.Dropout()

        def __deepcopy__(self, memo: dict[int, object]) -> ReusesServingChild:
            return copy(self)

        def forward(self, audio: torch.Tensor) -> torch.Tensor:
            return self.dropout(audio)

    shared_child = ReusesServingChild().train()
    with pytest.raises(FaResourceProjectionError):
        project_model_workspace(
            shared_child, lambda virtual: virtual(torch.empty(1, 2))
        )
    assert shared_child.dropout.training, "a copied root cannot hide a shared child"


def test_persistent_weights_are_not_charged_again_as_request_workspace() -> None:
    class LargeWeights(QuadraticEncoder):
        def __init__(self) -> None:
            super().__init__()
            self.large_weight = nn.Parameter(torch.zeros(256, 256))

    projected = project_encoder_workspace(
        LargeWeights(), AudioShape(samples=32), torch.device("cpu")
    )
    assert projected.peak_workspace_bytes < 256 * 256 * torch.float32.itemsize


def test_projection_supports_the_installed_mms_encoder_operations() -> None:
    from torchaudio.models import wav2vec2_model
    from torchaudio.pipelines import MMS_FA

    # Exercise the installed encoder's real convolution/attention operations
    # with small random weights, not a downloaded acoustic model. Its
    # convolution geometry is the bundle's; only the transformer width/depth
    # are reduced so this control remains a cheap, deterministic unit test.
    parameters = dict(MMS_FA._params)
    parameters.update(
        encoder_embed_dim=16,
        encoder_ff_interm_features=32,
        encoder_num_heads=2,
        encoder_num_layers=1,
    )
    model = wav2vec2_model(**parameters)
    result = project_encoder_workspace(
        model, AudioShape(samples=20 * MMS_FA.sample_rate), torch.device("cpu")
    )
    assert result.encoded_frames > 0
    assert result.emission_classes == parameters["aux_num_out"]
    assert result.peak_workspace_bytes > 0


def test_unprojectable_model_is_a_named_failure_not_a_fabricated_estimate() -> None:
    class DataDependentEncoder(QuadraticEncoder):
        def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, None]:
            count = int(audio.sum().item())
            return audio[:, :count].unsqueeze(2), None

    with pytest.raises(FaResourceProjectionError, match="DataDependent"):
        project_encoder_workspace(
            DataDependentEncoder(), AudioShape(samples=32), torch.device("cpu")
        )


@pytest.mark.parametrize("samples", [0, -1])
def test_audio_shape_refuses_nonrecordings(samples: int) -> None:
    with pytest.raises(ValueError):
        AudioShape(samples=samples)


class MetadataEncoder(nn.Module):
    """A declared tiny encoder; its preparation must never see real samples."""

    def __init__(self) -> None:
        super().__init__()
        from torchaudio.models.wav2vec2.components import (
            ConvLayerBlock,
            FeatureExtractor,
        )

        self.feature_extractor = FeatureExtractor(
            nn.ModuleList(
                [
                    ConvLayerBlock(
                        1, 1, kernel_size=2, stride=2, bias=False, layer_norm=None
                    ),
                ]
            )
        )
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, None]:
        assert isinstance(audio, FakeTensor)
        features, _ = self.feature_extractor(audio, None)
        return features.expand(-1, -1, 29) * self.scale, None


def test_complete_mms_bundle_architecture_projects_without_checkpoint_or_waveform() -> (
    None
):
    """Exercise the installed bundle's actual encoder and probability wrapper.

    Meta construction creates tensor metadata only. There is no checkpoint
    download, weight allocation or acoustic inference in this qualification.
    """
    import torchaudio
    from torchaudio.pipelines._wav2vec2 import utils

    from batchalign.worker._types_v2 import FaPreparedV2, FaProjectedMemoryV2

    bundle = torchaudio.pipelines.MMS_FA
    # Exactly the installed get_model construction, omitting state-dict
    # acquisition/loading and using meta storage instead of real weights.
    with torch.device("meta"):
        model = utils._get_model(bundle._model_type, bundle._params)
        model = utils._extend_model(
            model,
            normalize_waveform=bundle._normalize_waveform,
            apply_log_softmax=True,
            append_star=True,
        ).eval()
    assert all(parameter.device.type == "meta" for parameter in model.parameters())
    prepare = mms_request_preparer(mms_handle(model), lambda words: words)
    short = prepare(_binding(20 * 16000), ["hi", "there"], "hi there")
    long = prepare(_binding(180 * 16000), ["hi", "there"], "hi there")
    assert isinstance(short, FaPreparedV2), short
    assert isinstance(long, FaPreparedV2), long
    assert isinstance(short.plan.memory, FaProjectedMemoryV2)
    assert isinstance(long.plan.memory, FaProjectedMemoryV2)
    assert long.plan.memory.workspace > short.plan.memory.workspace
    assert long.plan.memory.model == short.plan.memory.model


def _binding(samples: int, sample_rate: int = 16000) -> str:
    from batchalign.worker._types_v2 import (
        FaBackendV2,
        FaPreparationBindingV2,
        FaTextModeV2,
    )

    return FaPreparationBindingV2(
        execution_request_id="metadata-fixture",
        backend=FaBackendV2.WAVE2VEC,
        engine="metadata-encoder",
        payload_ref_id="words-fixture",
        audio_ref_id="audio-fixture",
        samples=samples,
        sample_rate_hz=sample_rate,
        text_mode=FaTextModeV2.SPACE_JOINED,
        attention_capture=None,
    ).model_dump_json()


def test_loaded_preparer_binds_real_shapes_labels_and_native_workspace() -> None:
    from batchalign.worker._types_v2 import FaPreparedV2, FaProjectedMemoryV2

    model = MetadataEncoder()
    prepare = mms_request_preparer(mms_handle(model), lambda words: words)
    small = prepare(_binding(32), ["hi"], "hi")
    large = prepare(_binding(128), ["hi"], "hi")
    assert isinstance(small, FaPreparedV2)
    assert isinstance(large, FaPreparedV2)
    assert isinstance(small.plan.memory, FaProjectedMemoryV2)
    assert isinstance(large.plan.memory, FaProjectedMemoryV2)
    assert small.plan.binding.samples == 32
    assert large.plan.binding.samples == 128
    assert large.plan.memory.workspace > small.plan.memory.workspace
    assert small.plan.memory.model == sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    assert large.plan.memory.model == small.plan.memory.model


def test_preparation_reports_a_real_ctc_label_constraint_before_execution() -> None:
    from batchalign.worker._types_v2 import FaModelConstraintV2, FaRefusedV2

    prepare = mms_request_preparer(mms_handle(MetadataEncoder()), lambda words: words)
    result = prepare(_binding(32), ["a" * 20], "a" * 20)
    assert isinstance(result, FaRefusedV2)
    assert isinstance(result.cause, FaModelConstraintV2)
    assert result.cause.provided == 39
    assert result.cause.maximum == 16


def test_unqualified_native_layout_cannot_supply_an_execution_plan(monkeypatch) -> None:
    import torchaudio.version

    from batchalign.worker._types_v2 import FaProjectionUnavailableV2, FaRefusedV2

    monkeypatch.setattr(torchaudio.version, "git_version", "unqualified-layout")
    prepare = mms_request_preparer(mms_handle(MetadataEncoder()), lambda words: words)
    result = prepare(_binding(32), ["hi"], "hi")
    assert isinstance(result, FaRefusedV2)
    assert isinstance(result.cause, FaProjectionUnavailableV2)
    assert "unqualified-layout" in result.cause.message


def test_loaded_mms_minimum_input_extent_is_a_typed_constraint_before_forward() -> None:
    from batchalign.worker._types_v2 import FaInsufficientInputV2, FaRefusedV2

    prepare = mms_request_preparer(mms_handle(MetadataEncoder()), lambda words: words)
    result = prepare(_binding(1), ["hi"], "hi")
    assert isinstance(result, FaRefusedV2)
    assert isinstance(result.cause, FaInsufficientInputV2)
    assert result.cause.provided == 1
    assert result.cause.minimum == 2


def test_cpu_convolution_projection_includes_native_columns() -> None:
    """The foreign CPU kernel owns scratch hidden from FakeTensor outputs."""

    class TemporalEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv1d(1, 32, 3, bias=False),
                nn.Conv1d(32, 8, 3, bias=False),
            )

        def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, None]:
            assert isinstance(audio, FakeTensor)
            return self.layers(audio.unsqueeze(1)).transpose(1, 2), None

    # Select the audited native MM path even on builds that also offer MKLDNN.
    with torch.backends.mkldnn.flags(enabled=False):
        projected = project_encoder_workspace(
            TemporalEncoder(), AudioShape(samples=128), torch.device("cpu")
        )
    # At the second convolution, its input, native im2col columns and output
    # coexist. Internal columns are not an output of aten.convolution.
    minimum = (32 * 126 + 32 * 3 * 124 + 8 * 124) * torch.float32.itemsize
    assert projected.peak_workspace_bytes >= minimum


def test_math_attention_projection_covers_hidden_score_and_softmax_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Math selection does not decompose this operator under FakeTensor."""

    class AttentionModel(nn.Module):
        def forward(self) -> torch.Tensor:
            query = torch.empty(1, 2, 16, 8, dtype=torch.float32)
            key = torch.empty_like(query)
            value = torch.empty_like(query)
            assert isinstance(query, FakeTensor)
            return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    projected = project_model_workspace(AttentionModel(), lambda model: model())
    # The pinned native math implementation keeps Q/K/V while its scores and
    # softmax result coexist; safe_softmax also retains an elementwise mask.
    # None of those internal score/mask tensors is the top-level SDPA output.
    inputs = 3 * 2 * 16 * 8 * torch.float32.itemsize
    score_elements = 2 * 16 * 16
    minimum = inputs + 2 * score_elements * torch.float32.itemsize + score_elements
    assert projected.peak_workspace_bytes >= minimum

    from batchalign.inference import fa_attention_resources as attention

    # An unavailable foreign composite cannot silently fall back to the opaque
    # result whose allocation evidence this control proved insufficient.
    monkeypatch.setattr(
        attention,
        "_COMPOSITES",
        {
            **attention._COMPOSITES,
            torch.ops.aten.scaled_dot_product_attention.default: (
                attention._UnavailableComposite(
                    "declared attention composite unavailable"
                )
            ),
        },
    )
    with pytest.raises(FaResourceProjectionError, match="composite unavailable"):
        project_model_workspace(AttentionModel(), lambda model: model())


def test_projection_refuses_an_unsupported_operator_before_numerical_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real CPU kernel is not an admissible way to discover virtual shapes."""
    from torch._subclasses import fake_tensor

    class NativeOnlyModel(nn.Module):
        def forward(self) -> torch.Tensor:
            values = torch.empty(2, dtype=torch.int64)
            assert isinstance(values, FakeTensor)
            # This installed core operator has a CPU kernel and no Meta kernel.
            return torch.ops.aten._test_optional_intlist.default(values, [1, 2])

    def forbidden_fallback(
        fake_mode: object,
        func: object,
        flat_args: object,
        args_spec: object,
        orig_not_implemented_exception: object,
    ) -> torch.Tensor:
        raise AssertionError("projection reached the real numerical fallback port")

    monkeypatch.setattr(fake_tensor, "run_fallback_kernel", forbidden_fallback)
    with pytest.raises(FaResourceProjectionError, match="UnsupportedOperatorException"):
        project_model_workspace(NativeOnlyModel(), lambda model: model())
