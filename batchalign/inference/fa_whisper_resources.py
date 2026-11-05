"""Loaded Whisper context admission and virtual request allocation planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    ValidationError,
)
from torch import nn

from batchalign.inference._admission import _FactoryAdmission
from batchalign.inference.fa_virtual_resources import (
    FaResourceProjectionError,
    UnavailableFaPreparation,
    model_storage_bytes,
    project_model_workspace,
)
from batchalign.worker._types_v2 import (
    FaConstraintUnitV2,
    FaInsufficientInputV2,
    FaModelConstraintV2,
    FaPreparationBindingV2,
    FaPreparationOutcomeV2,
    FaPreparationPlanV2,
    FaPreparedV2,
    FaProjectedMemoryV2,
    FaProjectionUnavailableV2,
    FaRefusedV2,
    FaWhisperBasisV2,
)

if TYPE_CHECKING:
    from batchalign.inference.types import WhisperFAHandle
    from batchalign.worker._loads import FaPreparer


class _WhisperGeometry(BaseModel):
    """A loaded model's frame and tokenizer geometry, never a duration policy."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    sample_rate: PositiveInt
    hop: PositiveInt
    fft_size: PositiveInt
    source_samples: PositiveInt
    mel_bins: PositiveInt
    mel_frames: PositiveInt
    encoder_stride: PositiveInt
    encoder_frames: PositiveInt
    decoder_labels: PositiveInt
    median_width: PositiveInt

    @classmethod
    def read(cls, handle: WhisperFAHandle) -> _WhisperGeometry:
        feature = handle.processor.feature_extractor
        encoder = handle.model.model.encoder
        geometry = cls(
            sample_rate=handle.sample_rate,
            hop=feature.hop_length,
            fft_size=feature.n_fft,
            source_samples=feature.n_samples,
            mel_bins=feature.feature_size,
            mel_frames=feature.nb_max_frames,
            encoder_stride=encoder.conv1.stride[0] * encoder.conv2.stride[0],
            encoder_frames=handle.model.config.max_source_positions,
            decoder_labels=handle.model.config.max_target_positions,
            median_width=handle.model.config.median_filter_width,
        )
        frames = geometry.mel_frames
        for convolution in (encoder.conv1, encoder.conv2):
            frames = (
                frames
                + 2 * convolution.padding[0]
                - convolution.dilation[0] * (convolution.kernel_size[0] - 1)
                - 1
            ) // convolution.stride[0] + 1
        if (
            frames != geometry.encoder_frames
            or geometry.mel_bins != handle.model.config.num_mel_bins
        ):
            raise FaResourceProjectionError(
                "Whisper feature geometry disagrees with the loaded encoder"
            )
        return geometry

    @property
    def maximum_samples(self) -> int:
        return min(
            self.source_samples, self.encoder_frames * self.encoder_stride * self.hop
        )


class _WhisperHead(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    layer: NonNegativeInt = Field(strict=True)
    head: NonNegativeInt = Field(strict=True)


def _loaded_heads(handle: WhisperFAHandle) -> tuple[_WhisperHead, ...]:
    # Alignment heads are checkpoint metadata added to GenerationConfig at
    # runtime, not a declared attribute of Transformers' generic config type.
    try:
        raw: object = handle.model.generation_config.alignment_heads  # type: ignore[attr-defined]
    except AttributeError as error:
        raise FaResourceProjectionError(
            "loaded Whisper exposes no alignment-head metadata"
        ) from error
    if not isinstance(raw, (list, tuple)) or not raw:
        raise FaResourceProjectionError("loaded Whisper has no alignment heads")
    selected: list[_WhisperHead] = []
    for pair in raw:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise FaResourceProjectionError(
                "Whisper alignment head is not a layer/head pair"
            )
        choice = _WhisperHead(layer=pair[0], head=pair[1])
        if (
            choice.layer >= len(handle.model.model.decoder.layers)
            or choice.head
            >= handle.model.model.decoder.layers[choice.layer].encoder_attn.num_heads
        ):
            raise FaResourceProjectionError(
                "Whisper alignment head lies outside the loaded decoder"
            )
        selected.append(choice)
    return tuple(selected)


@dataclass(frozen=True, slots=True, init=False)
class _LoadedWhisperPreparation(_FactoryAdmission):
    """Fixed facts and tokenizer authority bound to one loaded model epoch."""

    handle: WhisperFAHandle
    geometry: _WhisperGeometry
    heads: tuple[_WhisperHead, ...]
    prefix: tuple[int, ...]
    terminal: int
    device: torch.device
    dtype: torch.dtype
    model_bytes: int
    basis: FaWhisperBasisV2

    @classmethod
    def admit(cls, handle: WhisperFAHandle) -> _LoadedWhisperPreparation:
        import transformers

        geometry = _WhisperGeometry.read(handle)
        heads = _loaded_heads(handle)
        prefix = tuple(handle.processor.tokenizer.prefix_tokens)
        terminal = handle.processor.tokenizer.eos_token_id
        if (
            not prefix
            or any(type(value) is not int or value < 0 for value in prefix)
            or type(terminal) is not int
            or terminal < 0
        ):
            raise FaResourceProjectionError(
                "loaded Whisper has invalid tokenizer prefix or terminal metadata"
            )
        parameter = next(handle.model.parameters(), None)
        if parameter is None:
            raise FaResourceProjectionError("loaded Whisper has no parameter device")
        value = object.__new__(cls)
        for name, fact in (
            ("handle", handle),
            ("geometry", geometry),
            ("heads", heads),
            ("prefix", prefix),
            ("terminal", terminal),
            ("device", parameter.device),
            ("dtype", parameter.dtype),
            ("model_bytes", model_storage_bytes(handle.model)),
            (
                "basis",
                FaWhisperBasisV2(
                    torch_version=torch.__version__,
                    transformers_version=transformers.__version__,
                ),
            ),
        ):
            object.__setattr__(value, name, fact)
        return value

    def labels(self, text: str) -> tuple[int, ...]:
        observed = self.handle.processor.tokenizer(text)["input_ids"]
        if not isinstance(observed, list) or any(
            type(value) is not int or value < 0 for value in observed
        ):
            raise FaResourceProjectionError(
                "Whisper tokenizer did not return integer decoder labels"
            )
        labels = tuple(observed)
        if (
            labels[: len(self.prefix)] != self.prefix
            or len(labels) <= len(self.prefix) + 1
            or labels[-1] != self.terminal
        ):
            raise FaResourceProjectionError(
                "Whisper tokenizer omitted its prefix, lexical labels or terminal label"
            )
        return labels


def whisper_request_preparer(handle: WhisperFAHandle) -> FaPreparer:
    """One loaded host owns context admission, projection and the bound report."""

    from transformers.models.whisper.generation_whisper import _median_filter

    try:
        loaded = _LoadedWhisperPreparation.admit(handle)
    except (FaResourceProjectionError, ValidationError) as error:
        return UnavailableFaPreparation(str(error))

    def prepare(
        binding_json: str, words: list[str], text: str
    ) -> FaPreparationOutcomeV2:

        binding = FaPreparationBindingV2.model_validate_json(binding_json)
        try:
            geometry = loaded.geometry
            if binding.sample_rate_hz != geometry.sample_rate:
                raise FaResourceProjectionError(
                    "prepared waveform sample rate disagrees with loaded Whisper"
                )
            if binding.samples > geometry.maximum_samples:
                return FaRefusedV2(
                    cause=FaModelConstraintV2(
                        unit=FaConstraintUnitV2.AUDIO_SAMPLES,
                        provided=binding.samples,
                        maximum=geometry.maximum_samples,
                        engine=binding.engine,
                    )
                )
            minimum_samples = geometry.hop * geometry.encoder_stride
            if binding.samples < minimum_samples:
                return FaRefusedV2(
                    cause=FaInsufficientInputV2(
                        unit=FaConstraintUnitV2.AUDIO_SAMPLES,
                        provided=binding.samples,
                        minimum=minimum_samples,
                        engine=binding.engine,
                    )
                )
            labels = loaded.labels(text)
            if len(labels) > geometry.decoder_labels:
                return FaRefusedV2(
                    cause=FaModelConstraintV2(
                        unit=FaConstraintUnitV2.TEXT_LABELS,
                        provided=len(labels),
                        maximum=geometry.decoder_labels,
                        engine=binding.engine,
                    )
                )
            prefix_count = len(loaded.prefix)
            acoustic_frames = binding.samples // geometry.hop // geometry.encoder_stride
            heads = loaded.heads
            device, dtype = loaded.device, loaded.dtype

            def align(virtual_model: nn.Module) -> torch.Tensor:
                features = torch.empty(
                    (1, geometry.mel_bins, geometry.mel_frames),
                    dtype=dtype,
                    device=device,
                )
                tokens = torch.tensor([labels], dtype=torch.int64, device=device)
                output = virtual_model(
                    input_features=features, labels=tokens, output_attentions=True
                )
                attention = output.cross_attentions
                if not isinstance(attention, tuple) or any(
                    not isinstance(item, torch.Tensor) for item in attention
                ):
                    raise FaResourceProjectionError(
                        "loaded Whisper did not expose cross-attention tensors"
                    )
                all_heads = torch.cat(attention).cpu()
                weights = torch.stack(
                    [all_heads[selection.layer][selection.head] for selection in heads]
                )
                weights = weights[:, prefix_count:, :acoustic_frames]
                std, mean = torch.std_mean(
                    weights, dim=-2, keepdim=True, unbiased=False
                )
                filtered = _median_filter((weights - mean) / std, geometry.median_width)
                if not isinstance(filtered, torch.Tensor):
                    raise FaResourceProjectionError(
                        "Whisper median filter did not return a tensor"
                    )
                return -filtered.mean(dim=0).float()

            projected = project_model_workspace(loaded.handle.model, align)
            rows, columns = projected.output.shape
            if rows != len(labels) - prefix_count or columns != acoustic_frames:
                raise FaResourceProjectionError(
                    "Whisper alignment projection changed decoder/audio cardinality"
                )
            # The native DTW owns a copied f32 cost matrix, an f64 cumulative
            # grid, a byte trace grid, a two-usize path and the row-index output.
            # Keep this tied to batchalign-fa-core::warp and whisper_alignment.rs.
            pointer = np.dtype(np.uintp).itemsize
            dtw = (
                rows * columns * torch.float32.itemsize
                + (rows + 1)
                * (columns + 1)
                * (torch.float64.itemsize + torch.uint8.itemsize)
                + (rows + columns) * 2 * pointer
                + 2 * rows * pointer
            )
            # Whisper pads the feature input to its loaded extractor extent.
            # Charge waveform copies, STFT/magnitude arrays and mel features
            # in addition to the projected model/attention lifetime peak.
            spectral_cells = (geometry.mel_frames + 1) * (geometry.fft_size // 2 + 1)
            frontend = (
                2 * (binding.samples + geometry.source_samples) * torch.float32.itemsize
                + spectral_cells
                * (torch.complex64.itemsize + 3 * torch.float32.itemsize)
                + 3 * geometry.mel_bins * geometry.mel_frames * torch.float32.itemsize
            )
            return FaPreparedV2(
                plan=FaPreparationPlanV2(
                    binding=binding,
                    memory=FaProjectedMemoryV2(
                        workspace=projected.peak_workspace_bytes + int(dtw) + frontend,
                        model=loaded.model_bytes,
                        basis=loaded.basis,
                    ),
                )
            )
        except (FaResourceProjectionError, ValidationError) as error:
            return FaRefusedV2(cause=FaProjectionUnavailableV2(message=str(error)))

    return prepare
