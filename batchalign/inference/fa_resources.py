"""MMS constraints and request-shaped forced-alignment preparation.

This is allocation evidence, not a memory lease or acoustic inference. The
executor must separately account for native alignment workspace and acquire
the shared host's request lease before allocating real request tensors.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from pydantic import BaseModel, ConfigDict, PositiveInt
from torch import nn

from batchalign.inference._admission import _FactoryAdmission
from batchalign.inference.fa_virtual_resources import (
    FaResourceProjectionError,
    UnavailableFaPreparation,
    model_storage_bytes,
    project_model_workspace,
)

if TYPE_CHECKING:
    from batchalign.inference.types import Wave2VecFAHandle
    from batchalign.worker._loads import FaPreparer
    from batchalign.worker._types_v2 import FaMmsCpuBasisV2, FaPreparationOutcomeV2


class AudioShape(BaseModel):
    """The positive sample count read from the prepared audio's metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    samples: PositiveInt


@dataclass(frozen=True, slots=True, init=False)
class ProjectedEncoderWorkspace(_FactoryAdmission):
    """Virtual allocation evidence, constructible only by the projection."""

    peak_workspace_bytes: int
    encoded_frames: int
    emission_classes: int

    @classmethod
    def _from_projection(
        cls, peak: int, frames: int, classes: int
    ) -> ProjectedEncoderWorkspace:
        if peak <= 0 or frames <= 0 or classes <= 0:
            raise FaResourceProjectionError(
                f"nonpositive encoder projection: {peak=}, {frames=}, {classes=}"
            )
        value = object.__new__(cls)
        object.__setattr__(value, "peak_workspace_bytes", peak)
        object.__setattr__(value, "encoded_frames", frames)
        object.__setattr__(value, "emission_classes", classes)
        return value


@dataclass(frozen=True, slots=True, init=False)
class MmsFeatureGeometry(_FactoryAdmission):
    """Minimum waveform extent derived from the loaded convolution chain."""

    minimum_samples: int

    @classmethod
    def read(cls, model: nn.Module) -> MmsFeatureGeometry:
        from torchaudio.models.wav2vec2.components import FeatureExtractor

        extractors = [
            module for module in model.modules() if isinstance(module, FeatureExtractor)
        ]
        if len(extractors) != 1 or not extractors[0].conv_layers:
            raise FaResourceProjectionError(
                "MMS must expose one nonempty loaded feature-extractor chain"
            )
        # A nonempty transcript requires at least one encoded frame. Invert
        # the actual loaded convolution geometry, from last stage to first.
        minimum = 1
        for stage in reversed(extractors[0].conv_layers):
            convolution = stage.conv
            if not isinstance(convolution, nn.Conv1d) or isinstance(
                convolution.padding, str
            ):
                raise FaResourceProjectionError(
                    "MMS feature geometry is not an explicit Conv1d chain"
                )
            receptive = convolution.dilation[0] * (convolution.kernel_size[0] - 1) + 1
            minimum = max(
                1,
                (minimum - 1) * convolution.stride[0]
                + receptive
                - 2 * convolution.padding[0],
            )
        value = object.__new__(cls)
        object.__setattr__(value, "minimum_samples", minimum)
        return value


def project_encoder_workspace(
    model: nn.Module, audio: AudioShape, device: torch.device
) -> ProjectedEncoderWorkspace:
    """Project the waveform encoder while retaining its actual output geometry."""

    def encode(virtual_model: nn.Module) -> torch.Tensor:
        waveform = torch.empty((1, audio.samples), dtype=torch.float32, device=device)
        emission, _ = virtual_model(waveform)
        if (
            not isinstance(emission, torch.Tensor)
            or emission.ndim != 3
            or emission.shape[0] != 1
        ):
            raise FaResourceProjectionError(
                "encoder must return one batch/frame/class tensor"
            )
        return emission

    projected = project_model_workspace(model, encode)
    return ProjectedEncoderWorkspace._from_projection(
        projected.peak_workspace_bytes,
        int(projected.output.shape[1]),
        int(projected.output.shape[2]),
    )


@dataclass(frozen=True, slots=True, init=False)
class _LoadedMmsMetadata(_FactoryAdmission):
    """Fixed facts of one loaded MMS model, not a measured RSS envelope."""

    handle: Wave2VecFAHandle
    geometry: MmsFeatureGeometry
    device: torch.device
    model_bytes: int
    basis: FaMmsCpuBasisV2

    @classmethod
    def admit(cls, handle: Wave2VecFAHandle) -> _LoadedMmsMetadata:
        import torchaudio.version

        from batchalign.worker._types_v2 import FaMmsCpuBasisV2

        revision = torchaudio.version.git_version
        qualified = "455331b1b3bb4aabfe3228f6f911fc3f11ae93b1"
        if revision != qualified:
            raise FaResourceProjectionError(
                f"native CPU CTC allocation layout is unqualified: {revision}"
            )
        parameter = next(handle.model.parameters(), None)
        if parameter is None:
            raise FaResourceProjectionError("loaded MMS model has no parameter device")
        value = object.__new__(cls)
        object.__setattr__(value, "handle", handle)
        object.__setattr__(value, "geometry", MmsFeatureGeometry.read(handle.model))
        object.__setattr__(value, "device", parameter.device)
        object.__setattr__(value, "model_bytes", model_storage_bytes(handle.model))
        object.__setattr__(
            value,
            "basis",
            FaMmsCpuBasisV2(
                torch_version=torch.__version__,
                torchaudio_revision=revision,
            ),
        )
        return value


def mms_request_preparer(
    handle: Wave2VecFAHandle, shape_words: Callable[[list[str]], list[str]]
) -> FaPreparer:
    """Prepare labels and allocation evidence over this loaded MMS model.

    The source-shaping verb is shared with execution (identity for MMS,
    source-complete romanization for Cantonese). There is no waveform argument
    and no path from this callback to real encoder execution.
    """
    from batchalign.inference.fa_ctc_resources import (
        AdmittedCtcShape,
        CtcFrameCapacityError,
        MmsTargetSequence,
    )
    from batchalign.worker._types_v2 import (
        FaConstraintUnitV2,
        FaInsufficientInputV2,
        FaModelConstraintV2,
        FaPreparationBindingV2,
        FaPreparationPlanV2,
        FaPreparedV2,
        FaProjectedMemoryV2,
        FaProjectionUnavailableV2,
        FaRefusedV2,
    )

    try:
        loaded = _LoadedMmsMetadata.admit(handle)
    except FaResourceProjectionError as error:
        # A failed load-bound qualification is retained, never repeated for
        # every candidate. Cache-ready callers still need no preparation.
        return UnavailableFaPreparation(str(error))

    def prepare(
        binding_json: str, words: list[str], text: str
    ) -> FaPreparationOutcomeV2:
        binding = FaPreparationBindingV2.model_validate_json(binding_json)
        targets = MmsTargetSequence.from_words(
            words=shape_words(words),
            alphabet=loaded.handle.alphabet,
        )
        try:
            if binding.sample_rate_hz != loaded.handle.sample_rate:
                raise FaResourceProjectionError(
                    f"prepared audio rate {binding.sample_rate_hz} differs from model rate {loaded.handle.sample_rate}"
                )
            if binding.samples < loaded.geometry.minimum_samples:
                return FaRefusedV2(
                    cause=FaInsufficientInputV2(
                        unit=FaConstraintUnitV2.AUDIO_SAMPLES,
                        provided=binding.samples,
                        minimum=loaded.geometry.minimum_samples,
                        engine=binding.engine,
                    )
                )
            projected = project_encoder_workspace(
                loaded.handle.model,
                AudioShape(samples=binding.samples),
                loaded.device,
            )
            ctc = AdmittedCtcShape.admit(
                targets,
                frames=projected.encoded_frames,
                classes=projected.emission_classes,
                scalar_bytes=torch.float32.itemsize,
            )
            # Sum the tensor-lifetime peak and all alignment-stage allocations
            # conservatively, rather than assuming the two peaks are disjoint.
            # Copies onto CPU, path/scores and score exponentiation/gathering
            # remain charged even when a particular CPU execution aliases them.
            emissions = (
                projected.encoded_frames
                * projected.emission_classes
                * torch.float32.itemsize
            )
            waveform = binding.samples * torch.float32.itemsize
            labels = len(targets.ids) * torch.int64.itemsize
            frame_vectors = projected.encoded_frames * (
                3 * torch.int64.itemsize + 3 * torch.float32.itemsize
            )
            workspace = (
                projected.peak_workspace_bytes
                + ctc.native_workspace_bytes
                + 3 * emissions
                + waveform
                + labels
                + frame_vectors
            )
            return FaPreparedV2(
                plan=FaPreparationPlanV2(
                    binding=binding,
                    memory=FaProjectedMemoryV2(
                        workspace=workspace,
                        model=loaded.model_bytes,
                        basis=loaded.basis,
                    ),
                )
            )
        except CtcFrameCapacityError as error:
            return FaRefusedV2(
                cause=FaModelConstraintV2(
                    unit=FaConstraintUnitV2.ENCODED_FRAMES,
                    provided=error.required_frames,
                    maximum=error.available_frames,
                    engine=binding.engine,
                )
            )
        except FaResourceProjectionError as error:
            return FaRefusedV2(cause=FaProjectionUnavailableV2(message=str(error)))

    return prepare
