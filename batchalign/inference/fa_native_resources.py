"""Input-bound scratch of audited native CPU convolution kernels.

FakeTensor reports operator outputs, not temporary allocations inside a C++
operator. A native receipt accounts for those allocations alongside the live
virtual storage. It never runs a real kernel or authorizes model execution.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from math import prod

import torch
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensor

from batchalign.inference._admission import _FactoryAdmission


class _CpuKernel(Enum):
    SLOW_MM = auto()
    NNPACK = auto()


class NativeWorkspaceUnavailable(ValueError):
    """The selected foreign kernel has no qualified internal allocation layout."""


@dataclass(frozen=True, slots=True)
class _UnqualifiedCpuLayout:
    reason: str


@dataclass(frozen=True, slots=True, init=False)
class _QualifiedCpuLayout(_FactoryAdmission):
    """The source revision whose allocation/lifetime rules were inspected."""

    @classmethod
    def read(cls) -> _QualifiedCpuLayout | _UnqualifiedCpuLayout:
        revision = torch.version.git_version
        if revision != "d0f2af8b66b20da4baa50644c0cd3f887a2eae2e":
            return _UnqualifiedCpuLayout(
                f"native CPU convolution allocation layout is unqualified: {revision}"
            )
        return object.__new__(cls)


# Import ownership retains this fixed runtime fact, including its refusal.
# Backend selection is input-shaped and must still happen for each operation.
_CPU_LAYOUT = _QualifiedCpuLayout.read()


def _axes(value: object) -> list[int]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise NativeWorkspaceUnavailable("invalid native convolution axis descriptor")
    return [int(item) for item in value]


@dataclass(frozen=True, slots=True)
class _RequiredArgument:
    name: str


@dataclass(frozen=True, slots=True)
class _DefaultedArgument:
    name: str
    value: object


@dataclass(frozen=True, slots=True)
class _ConvolutionSchema:
    arguments: tuple[_RequiredArgument | _DefaultedArgument, ...]

    @classmethod
    def read(cls, operation: OpOverload) -> _ConvolutionSchema:
        return cls(
            tuple(
                _DefaultedArgument(argument.name, argument.default_value)
                if argument.has_default_value()
                else _RequiredArgument(argument.name)
                for argument in operation._schema.arguments
            )
        )

    def bind(
        self, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> dict[str, object]:
        bound: dict[str, object] = {}
        for index, argument in enumerate(self.arguments):
            if index < len(args):
                bound[argument.name] = args[index]
            elif argument.name in kwargs:
                bound[argument.name] = kwargs[argument.name]
            else:
                match argument:
                    case _DefaultedArgument(name, value):
                        bound[name] = value
                    case _RequiredArgument(name):
                        raise NativeWorkspaceUnavailable(
                            f"missing convolution argument {name}"
                        )
        return bound


# Retain the actual loaded operator schemas. Their defaults define the foreign
# operation; these are not guessed replacements for absent model metadata.
_CONVOLUTIONS = {
    operation: _ConvolutionSchema.read(operation)
    for operation in (
        torch.ops.aten.conv1d.default,
        torch.ops.aten.conv2d.default,
        torch.ops.aten.convolution.default,
    )
}


@dataclass(frozen=True, slots=True, init=False)
class NativeCpuConvolutionWorkspace(_FactoryAdmission):
    """A completed virtual operation's hidden peak beyond its tracked output."""

    additional_bytes: int

    @classmethod
    def observe(
        cls,
        operation: Callable[..., object],
        args: tuple[object, ...],
        kwargs: dict[str, object],
        output: object,
    ) -> NativeCpuConvolutionWorkspace | None:
        if not isinstance(operation, OpOverload) or operation not in _CONVOLUTIONS:
            return None
        if not isinstance(output, FakeTensor):
            raise NativeWorkspaceUnavailable(
                "native scratch projection requires virtual output"
            )
        if output.device.type != "cpu":
            # This receipt covers CPU scratch; it makes no GPU-workspace claim.
            return None
        bound = _CONVOLUTIONS[operation].bind(args, kwargs)
        source, weight, bias, groups = (
            bound["input"],
            bound["weight"],
            bound["bias"],
            bound["groups"],
        )
        if (
            not isinstance(source, FakeTensor)
            or not isinstance(weight, FakeTensor)
            or (bias is not None and not isinstance(bias, FakeTensor))
            or not isinstance(groups, int)
            or isinstance(groups, bool)
        ):
            raise NativeWorkspaceUnavailable("invalid virtual convolution descriptor")
        stride, padding, dilation = (
            _axes(bound["stride"]),
            _axes(bound["padding"]),
            _axes(bound["dilation"]),
        )
        if operation == torch.ops.aten.convolution.default:
            transposed, output_padding = (
                bound["transposed"],
                _axes(bound["output_padding"]),
            )
            if not isinstance(transposed, bool):
                raise NativeWorkspaceUnavailable("invalid native transpose descriptor")
        else:
            # The conv1d/conv2d operations themselves specify ordinary forward
            # convolution. Their semantic identity supplies these quantities.
            transposed, output_padding = False, [0] * len(stride)
        layout = _CPU_LAYOUT
        if isinstance(layout, _UnqualifiedCpuLayout):
            raise NativeWorkspaceUnavailable(layout.reason)
        return cls._from_cpu(
            layout,
            source,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            output,
        )

    @classmethod
    def _from_cpu(
        cls,
        layout: _QualifiedCpuLayout,
        source: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        stride: list[int],
        padding: list[int],
        dilation: list[int],
        transposed: bool,
        output_padding: list[int],
        groups: int,
        output: torch.Tensor,
    ) -> NativeCpuConvolutionWorkspace:
        # Use the installed shape-based selector, not an inferred backend from
        # device names or build flags. No samples or parameter data are read.
        backend = torch._C._select_conv_backend(
            source,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        )
        match backend.name:
            case "Slow2d":
                kernel = _CpuKernel.SLOW_MM
            case "NnpackSpatial":
                kernel = _CpuKernel.NNPACK
            case _:
                raise NativeWorkspaceUnavailable(
                    f"native CPU convolution workspace is unqualified for {backend}"
                )
        memory_format = torch._C._conv_determine_backend_memory_format(
            source, weight, backend
        )
        # Convolution.cpp lifts 1D to 2D after making its source contiguous.
        # Observe that geometry without allocating even a real input copy.
        if source.ndim == 3:
            source_format = torch.contiguous_format
        else:
            source_format = memory_format
        source_copy = (
            0
            if source.is_contiguous(memory_format=source_format)
            else source.numel() * source.element_size()
        )
        weight_copy = (
            0
            if weight.is_contiguous(memory_format=source_format)
            else weight.numel() * weight.element_size()
        )
        copies = source_copy + weight_copy
        match kernel:
            case _CpuKernel.SLOW_MM:
                # The native MM path aliases only the unpadded, stride-one
                # pointwise case; all other cases materialize columns.
                pointwise = (
                    all(k == 1 for k in weight.shape[2:])
                    and all(s == 1 for s in stride)
                    and all(p == 0 for p in padding)
                )
                scratch = (
                    0
                    if pointwise
                    else (
                        int(source.shape[0])
                        * int(weight.shape[1])
                        * prod(int(k) for k in weight.shape[2:])
                        * prod(int(n) for n in output.shape[2:])
                        * source.element_size()
                    )
                )
            case _CpuKernel.NNPACK:
                from batchalign.inference.fa_nnpack_resources import nnpack_workspace

                scratch = nnpack_workspace(
                    layout, source, weight, stride, padding, groups
                )
                # Native NNPack owns an explicit zero bias when absent.
                if bias is None:
                    scratch += int(weight.shape[0]) // groups * source.element_size()
                elif groups == 1 and not bias.is_contiguous():
                    copies += bias.numel() * bias.element_size()
        if groups == 1:
            additional = copies + scratch
        else:
            # Native groups run in order, retaining each output until cat.
            # Batch/channel slices may need their own contiguous copies.
            # One group's copies/columns coexist; outputs of all groups can
            # coexist with the final concatenation. Do not sum scratch over
            # every group as though the kernels ran concurrently.
            grouped_copies = 0
            for tensor, axis in ((source, 1), (weight, 0)):
                # The outer native path has made each base contiguous.
                base = torch.empty_like(tensor, memory_format=source_format)
                group = base.narrow(axis, 0, int(base.shape[axis]) // groups)
                if not group.is_contiguous(memory_format=source_format):
                    grouped_copies += group.numel() * group.element_size()
            if bias is not None and not bias.is_contiguous():
                grouped_copies += bias.numel() * bias.element_size() // groups
            output_bytes = output.numel() * output.element_size()
            additional = copies + max(scratch + grouped_copies, output_bytes)
        value = object.__new__(cls)
        object.__setattr__(value, "additional_bytes", additional)
        return value
