"""Request-bound Qwen shapes and virtual projection of its native components.

The native outer forward performs data-dependent metadata indexing. Preparation
owns those dimensions, so projection uses their admitted shapes while calling
the loaded CNN, attention, projector, language model and classifier modules.
The source qualification below refuses an unreviewed component graph.
"""

from __future__ import annotations

import hashlib
import inspect
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt, ValidationError
from torch import nn

from batchalign.inference._admission import _FactoryAdmission
from batchalign.inference.fa_virtual_resources import (
    FaResourceProjectionError,
    UnavailableFaPreparation,
    model_storage_bytes,
    project_model_workspace,
)
from batchalign.worker._types_v2 import (
    FaBackendV2,
    FaConstraintUnitV2,
    FaInsufficientInputV2,
    FaPreparationBindingV2,
    FaPreparationOutcomeV2,
    FaPreparationPlanV2,
    FaPreparedV2,
    FaProjectedMemoryV2,
    FaProjectionUnavailableV2,
    FaQwenBasisV2,
    FaRefusedV2,
)

if TYPE_CHECKING:
    from transformers import Qwen3ASRForTokenClassification, Qwen3ASRProcessor

    from batchalign.inference.qwen_forced_alignment import (
        LoadedQwenFaHost,
        WordSegmentation,
    )
    from batchalign.worker._loads import FaPreparer


_QUALIFIED_GRAPH = "98620120cdcf1ab25e1bf3dbe81eb86c319c0176a2f0f82a03816972496294f6"


@dataclass(frozen=True, slots=True, init=False)
class _NativeQwen(_FactoryAdmission):
    host: LoadedQwenFaHost
    model: Qwen3ASRForTokenClassification
    processor: Qwen3ASRProcessor
    graph: str

    @classmethod
    def admit(cls, host: LoadedQwenFaHost) -> _NativeQwen:
        from transformers import Qwen3ASRForTokenClassification, Qwen3ASRProcessor
        from transformers.modeling_layers import GenericForTokenClassification
        from transformers.models.qwen3_asr import (
            feature_extraction_qwen3_asr,
            modeling_qwen3_asr,
            processing_qwen3_asr,
        )

        model, processor = host.aligner.model, host.aligner.processor
        if not isinstance(model, Qwen3ASRForTokenClassification) or not isinstance(
            processor, Qwen3ASRProcessor
        ):
            raise FaResourceProjectionError(
                "Qwen preparation requires the loaded native model and processor"
            )
        material = "\n".join(
            inspect.getsource(item)
            for item in (
                modeling_qwen3_asr,
                processing_qwen3_asr,
                feature_extraction_qwen3_asr,
                GenericForTokenClassification,
            )
        )
        graph = hashlib.sha256(material.encode()).hexdigest()
        if graph != _QUALIFIED_GRAPH:
            raise FaResourceProjectionError(
                f"Qwen allocation graph is unqualified: {graph}"
            )
        value = object.__new__(cls)
        for name, fact in (
            ("host", host),
            ("model", model),
            ("processor", processor),
            ("graph", graph),
        ):
            object.__setattr__(value, name, fact)
        return value


class _QwenConvolution(BaseModel):
    """One admitted time-axis transform from the loaded encoder."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    padding: NonNegativeInt
    dilation: PositiveInt
    kernel: PositiveInt
    stride: PositiveInt

    @classmethod
    def read(cls, step: nn.Conv2d) -> _QwenConvolution:
        if isinstance(step.padding, str):
            raise FaResourceProjectionError(
                "Qwen projection requires explicit convolution padding"
            )
        return cls(
            padding=step.padding[1],
            dilation=step.dilation[1],
            kernel=step.kernel_size[1],
            stride=step.stride[1],
        )

    def output_frames(self, frames: int) -> int:
        return (
            frames + 2 * self.padding - self.dilation * (self.kernel - 1) - 1
        ) // self.stride + 1


class _QwenGeometry(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    sample_rate: PositiveInt
    hop: PositiveInt
    fft_size: PositiveInt
    minimum_samples: NonNegativeInt
    mel_bins: PositiveInt
    chunk_frames: PositiveInt
    inference_frames: PositiveInt
    convolutions: tuple[_QwenConvolution, _QwenConvolution, _QwenConvolution]

    @classmethod
    def read(cls, native: _NativeQwen) -> _QwenGeometry:
        feature = native.processor.feature_extractor
        encoder = native.model.model.audio_tower
        # Transformers attaches these native processor defaults at runtime.
        defaults = native.processor.valid_processor_kwargs._defaults["audio_kwargs"]  # type: ignore[attr-defined]
        if (
            defaults["n_window"] != encoder.n_window
            or feature.n_window != encoder.n_window
        ):
            raise FaResourceProjectionError(
                "Qwen processor chunk geometry disagrees with the loaded encoder"
            )
        geometry = cls(
            sample_rate=feature.sampling_rate,
            hop=feature.hop_length,
            fft_size=feature.n_fft,
            minimum_samples=feature.min_length,
            mel_bins=feature.feature_size,
            chunk_frames=2 * encoder.n_window,
            inference_frames=encoder.n_window_infer,
            convolutions=(
                _QwenConvolution.read(encoder.conv2d1),
                _QwenConvolution.read(encoder.conv2d2),
                _QwenConvolution.read(encoder.conv2d3),
            ),
        )
        if (
            geometry.inference_frames % geometry.chunk_frames
            or geometry.inference_frames < geometry.chunk_frames
            or geometry.mel_bins != encoder.config.num_mel_bins
        ):
            raise FaResourceProjectionError(
                "Qwen encoder window geometry is inconsistent"
            )
        return geometry

    def after_convolutions(self, frames: int) -> int:
        if frames == 0:
            return 0
        for step in self.convolutions:
            frames = step.output_frames(frames)
        return frames


@dataclass(frozen=True, slots=True, init=False)
class _LoadedQwenPreparation(_FactoryAdmission):
    """The qualified native model and all fixed allocation facts stay paired."""

    native: _NativeQwen
    geometry: _QwenGeometry
    chunk_tokens: int
    model_bytes: int
    basis: FaQwenBasisV2
    engine: str

    @classmethod
    def admit(cls, native: _NativeQwen) -> _LoadedQwenPreparation:
        import transformers

        geometry = _QwenGeometry.read(native)
        chunk_tokens = geometry.after_convolutions(geometry.chunk_frames)
        if (
            chunk_tokens
            > native.model.model.audio_tower.positional_embedding.positional_embedding.shape[
                0
            ]
        ):
            raise FaResourceProjectionError(
                "Qwen convolution output exceeds the loaded positional table"
            )
        value = object.__new__(cls)
        for name, fact in (
            ("native", native),
            ("geometry", geometry),
            ("chunk_tokens", chunk_tokens),
            ("model_bytes", model_storage_bytes(native.model)),
            (
                "basis",
                FaQwenBasisV2(
                    torch_version=torch.__version__,
                    transformers_version=transformers.__version__,
                    allocation_graph=native.graph,
                ),
            ),
            ("engine", native.host.cache_identity.wire_name),
        ):
            object.__setattr__(value, name, fact)
        return value


@dataclass(frozen=True, slots=True)
class _WindowStarts:
    windows: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _WindowLengths:
    windows: tuple[int, ...]

    def tolist(self) -> list[int]:
        return list(self.windows)


@dataclass(frozen=True, slots=True)
class _WindowEnds:
    windows: tuple[int, ...]

    def __sub__(self, other: _WindowStarts) -> _WindowLengths:
        if not isinstance(other, _WindowStarts) or other.windows is not self.windows:
            raise FaResourceProjectionError(
                "Qwen window subtraction crossed preparation owners"
            )
        return _WindowLengths(self.windows)


@dataclass(frozen=True, slots=True)
class _WindowBoundaries:
    """Metadata-only adapter for native attention's prefix-difference operation.

    These are admitted integer lengths, never neural tensors. New native uses
    of the boundary vector require qualification rather than guessed values.
    """

    windows: tuple[int, ...]

    def __getitem__(self, key: slice) -> _WindowStarts | _WindowEnds:
        if key.step is None and key.start == 1 and key.stop is None:
            return _WindowEnds(self.windows)
        if key.step is None and key.start is None and key.stop == -1:
            return _WindowStarts(self.windows)
        raise FaResourceProjectionError("unqualified Qwen attention boundary operation")


class _InputConstraint(ValueError):
    def __init__(self, cause: FaInsufficientInputV2) -> None:
        self.cause = cause
        super().__init__(str(cause))


@dataclass(frozen=True, slots=True, init=False)
class QwenPreparedInputs(_FactoryAdmission):
    _loaded: _LoadedQwenPreparation
    _segmentation: WordSegmentation
    binding: FaPreparationBindingV2
    mel_frames: int
    padded_frames: int
    token_ids: tuple[int, ...]
    _windows: tuple[int, ...]

    @property
    def audio_tokens(self) -> int:
        return sum(self._windows)

    @property
    def units(self) -> tuple[str, ...]:
        return self._segmentation.units

    @classmethod
    def admit(
        cls, host: LoadedQwenFaHost, binding: FaPreparationBindingV2, words: list[str]
    ) -> QwenPreparedInputs:
        return cls._from_loaded(
            _LoadedQwenPreparation.admit(_NativeQwen.admit(host)), binding, words
        )

    @classmethod
    def _from_loaded(
        cls,
        loaded: _LoadedQwenPreparation,
        binding: FaPreparationBindingV2,
        words: list[str],
    ) -> QwenPreparedInputs:
        native, geometry = loaded.native, loaded.geometry
        if binding.backend != FaBackendV2.QWEN3 or binding.engine != loaded.engine:
            raise FaResourceProjectionError(
                "Qwen preparation binding names a different loaded model"
            )
        if binding.sample_rate_hz != geometry.sample_rate:
            raise FaResourceProjectionError(
                "prepared waveform rate disagrees with loaded Qwen"
            )
        effective_samples = max(binding.samples, geometry.minimum_samples)
        minimum_fft = geometry.fft_size // 2 + 1
        if effective_samples < max(minimum_fft, geometry.hop):
            raise _InputConstraint(
                FaInsufficientInputV2(
                    unit=FaConstraintUnitV2.AUDIO_SAMPLES,
                    provided=binding.samples,
                    minimum=max(minimum_fft, geometry.hop),
                    engine=binding.engine,
                )
            )
        frames = effective_samples // geometry.hop
        padded = (
            (frames + geometry.chunk_frames - 1) // geometry.chunk_frames
        ) * geometry.chunk_frames
        full_chunks, last_frames = divmod(frames, geometry.chunk_frames)
        chunk_tokens = loaded.chunk_tokens
        tokens = full_chunks * chunk_tokens + geometry.after_convolutions(last_frames)
        # Compare to the processor's own metadata computation, using one real
        # integer only. No waveform or model activation is allocated here.
        observed = int(
            native.processor._get_audio_token_length(  # type: ignore[no-untyped-call]
                torch.tensor([frames]), n_window=geometry.chunk_frames // 2
            )[0]
        )
        if observed != tokens:
            raise FaResourceProjectionError(
                "Qwen processor token counts disagree with loaded convolutions"
            )
        maximum_chunk = geometry.after_convolutions(min(frames, geometry.chunk_frames))
        attention_window = maximum_chunk * (
            geometry.inference_frames // geometry.chunk_frames
        )
        complete, remainder = divmod(tokens, attention_window)
        windows = (attention_window,) * complete + ((remainder,) if remainder else ())
        segmentation = native.host.segment(words)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio"},
                    *({"type": "text", "text": unit} for unit in segmentation.units),
                ],
            }
        ]
        # ProcessorMixin's annotation is text-only; native Qwen itself uses
        # this exact multimodal conversation structure in its preparation API.
        rendered = native.processor.apply_chat_template(conversation, tokenize=False)  # type: ignore[arg-type]
        if (
            not isinstance(rendered, str)
            or rendered.count(native.processor.audio_token) != 1
        ):
            raise FaResourceProjectionError(
                "Qwen template did not produce one audio placeholder"
            )
        rendered = rendered.replace(
            native.processor.audio_token, native.processor.audio_token * tokens, 1
        )
        encoded = native.processor.tokenizer(rendered)
        ids = encoded["input_ids"]
        mask = encoded["attention_mask"]
        if (
            not isinstance(ids, list)
            or not ids
            or any(type(index) is not int or index < 0 for index in ids)
            or ids.count(native.model.config.audio_token_id) != tokens
            or mask != [1] * len(ids)
        ):
            raise FaResourceProjectionError(
                "Qwen tokenization does not bind the admitted audio population"
            )
        value = object.__new__(cls)
        for name, fact in (
            ("_loaded", loaded),
            ("_segmentation", segmentation),
            ("binding", binding),
            ("mel_frames", frames),
            ("padded_frames", padded),
            ("token_ids", tuple(ids)),
            ("_windows", windows),
        ):
            object.__setattr__(value, name, fact)
        return value

    def _project(self, virtual: nn.Module) -> torch.Tensor:
        from transformers import Qwen3ASRForTokenClassification

        if not isinstance(virtual, Qwen3ASRForTokenClassification):
            raise FaResourceProjectionError("Qwen virtual model lost its native type")
        virtual.set_attn_implementation("eager")
        parameter = next(virtual.parameters())
        geometry = self._loaded.geometry
        encoder = virtual.model.audio_tower
        features = torch.empty(
            (1, geometry.mel_bins, self.padded_frames),
            dtype=parameter.dtype,
            device=parameter.device,
        )
        feature_mask = torch.empty(
            (1, self.padded_frames), dtype=torch.int64, device=parameter.device
        )
        chunks = self.padded_frames // geometry.chunk_frames
        chunked = (
            features.view(1, geometry.mel_bins, chunks, geometry.chunk_frames)
            .permute(0, 2, 1, 3)
            .reshape(chunks, 1, geometry.mel_bins, geometry.chunk_frames)
        )
        convolution = torch.nn.functional.gelu(encoder.conv2d1(chunked))
        convolution = torch.nn.functional.gelu(encoder.conv2d2(convolution))
        convolution = torch.nn.functional.gelu(encoder.conv2d3(convolution))
        count, channels, frequencies, steps = convolution.shape
        convolution = encoder.conv_out(
            convolution.permute(0, 3, 1, 2)
            .contiguous()
            .view(count, steps, channels * frequencies)
        )
        convolution += encoder.positional_embedding.positional_embedding[:steps].to(
            convolution.dtype
        )
        # The source-qualified index selection's dimensions were admitted
        # above; charge its selector and gathered output without fake values.
        selector = torch.empty(
            (chunks, steps), dtype=torch.bool, device=parameter.device
        )
        indices = torch.empty(
            (self.audio_tokens,), dtype=torch.int64, device=parameter.device
        )
        packed = torch.empty(
            (self.audio_tokens, encoder.config.d_model),
            dtype=parameter.dtype,
            device=parameter.device,
        )
        boundaries = _WindowBoundaries(self._windows)
        for layer in encoder.layers:
            packed = layer(packed, boundaries)[0]
        packed = encoder.ln_post(packed)
        audio = virtual.model.multi_modal_projector(packed)
        ids = torch.empty(
            (1, len(self.token_ids)), dtype=torch.int64, device=parameter.device
        )
        embeddings = virtual.model.get_input_embeddings()(ids)
        merged = torch.empty_like(embeddings)
        output = virtual.model.language_model(
            inputs_embeds=merged, use_cache=virtual.config.get_text_config().use_cache
        )
        logits = virtual.score(virtual.dropout(output.last_hidden_state))
        if not isinstance(logits, torch.Tensor):
            raise FaResourceProjectionError("Qwen classifier did not return a tensor")
        decoded = logits.argmax(dim=-1).cpu()
        trace = logits.argmax(dim=-1)
        markers = torch.empty_like(ids, dtype=torch.bool)
        # Retain native-live populations together through the peak. Their
        # values are irrelevant to allocation planning and never leave it.
        _ = (feature_mask, selector, indices, audio, decoded, trace, markers)
        return logits

    def report(self) -> FaPreparedV2:
        projected = project_model_workspace(self._loaded.native.model, self._project)
        if tuple(projected.output.shape[:2]) != (1, len(self.token_ids)):
            raise FaResourceProjectionError(
                "Qwen projection changed its prepared token population"
            )
        geometry = self._loaded.geometry
        effective = max(self.binding.samples, geometry.minimum_samples)
        spectral = (self.mel_frames + 1) * (geometry.fft_size // 2 + 1)
        frontend = (
            2 * (self.binding.samples + effective) * torch.float32.itemsize
            + spectral * (torch.complex64.itemsize + 3 * torch.float32.itemsize)
            + 3 * geometry.mel_bins * self.padded_frames * torch.float32.itemsize
        )
        # Object templates account for the native decoder's Python records;
        # these are sizes only, never fabricated timing content.
        records = sum(sys.getsizeof(unit) for unit in self.units) + len(self.units) * (
            sys.getsizeof({"text": "", "start_time": 0.0, "end_time": 0.0})
            + 2 * sys.getsizeof(0.0)
        )
        return FaPreparedV2(
            plan=FaPreparationPlanV2(
                binding=self.binding,
                memory=FaProjectedMemoryV2(
                    workspace=projected.peak_workspace_bytes + frontend + records,
                    model=self._loaded.model_bytes,
                    basis=self._loaded.basis,
                ),
            )
        )


def qwen_request_preparer(host: LoadedQwenFaHost) -> FaPreparer:
    native = _NativeQwen.admit(host)
    try:
        loaded = _LoadedQwenPreparation.admit(native)
    except (FaResourceProjectionError, ValidationError) as error:
        return UnavailableFaPreparation(str(error))

    def prepare(
        binding_json: str, words: list[str], text: str
    ) -> FaPreparationOutcomeV2:
        binding = FaPreparationBindingV2.model_validate_json(binding_json)
        try:
            return QwenPreparedInputs._from_loaded(loaded, binding, words).report()
        except _InputConstraint as refused:
            return FaRefusedV2(cause=refused.cause)
        except (FaResourceProjectionError, ValidationError) as error:
            return FaRefusedV2(cause=FaProjectionUnavailableV2(message=str(error)))

    return prepare
