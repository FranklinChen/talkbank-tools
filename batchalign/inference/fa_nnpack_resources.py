"""NNPACK's documented sizing-only C API, without tensor-data pointers.

ABI and query-return paths are pinned by the qualified Torch/NNPACK source in
fa_native_resources. A query supplies a null workspace and a size output; the
native library returns before allocating workspace or reading model data.
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from batchalign.inference._admission import _FactoryAdmission

if TYPE_CHECKING:
    from batchalign.inference.fa_native_resources import _QualifiedCpuLayout


class NnpackSizingUnavailable(ValueError):
    """No supported native sizing operation can supply this request's demand."""


class _Size(ctypes.Structure):
    _fields_ = [("width", ctypes.c_size_t), ("height", ctypes.c_size_t)]


class _Padding(ctypes.Structure):
    _fields_ = [
        ("top", ctypes.c_size_t),
        ("right", ctypes.c_size_t),
        ("bottom", ctypes.c_size_t),
        ("left", ctypes.c_size_t),
    ]


type _CArgument = (
    type[ctypes.c_int]
    | type[ctypes.c_size_t]
    | type[ctypes.c_void_p]
    | type[_Size]
    | type[_Padding]
    | type[ctypes._Pointer[ctypes.c_size_t]]
)


@dataclass(frozen=True, slots=True)
class _Unavailable:
    reason: str


@dataclass(frozen=True, slots=True, init=False)
class _SizingApi(_FactoryAdmission):
    # Keeping the library owns the exported functions' lifetime. No callable
    # accepting model-data pointers is exposed by this owner.
    _library: ctypes.CDLL

    @classmethod
    def load(cls) -> _SizingApi | _Unavailable:
        names = {"darwin": "libtorch_cpu.dylib", "linux": "libtorch_cpu.so"}
        if sys.platform not in names:
            return _Unavailable(f"NNPACK sizing is unavailable on {sys.platform}")
        try:
            library = ctypes.CDLL(
                str(Path(torch.__file__).parent / "lib" / names[sys.platform])
            )
            common: list[_CArgument] = [
                ctypes.c_int,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                _Size,
                _Padding,
                _Size,
            ]
            tail: list[_CArgument] = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            library.nnp_convolution_output.argtypes = [*common, *tail]
            library.nnp_convolution_output.restype = ctypes.c_int
            library.nnp_convolution_inference.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_size_t,
                ctypes.c_size_t,
                _Size,
                _Padding,
                _Size,
                _Size,
                *tail,
            ]
            library.nnp_convolution_inference.restype = ctypes.c_int
        except (OSError, AttributeError) as error:
            return _Unavailable(f"NNPACK sizing exports are unavailable: {error}")
        value = object.__new__(cls)
        object.__setattr__(value, "_library", library)
        return value

    def workspace(
        self,
        batch: int,
        input_channels: int,
        output_channels: int,
        input_size: _Size,
        padding: _Padding,
        kernel: _Size,
        stride: _Size,
    ) -> int:
        size = ctypes.c_size_t()
        # Torch uses per-image inference for subsampling, even with a batch.
        # Query once: its per-image scratch is reused, never multiplied by N.
        if batch == 1 or stride.width != 1 or stride.height != 1:
            status = self._library.nnp_convolution_inference(
                0,
                1,
                input_channels,
                output_channels,
                input_size,
                padding,
                kernel,
                stride,
                None,
                None,
                None,
                None,
                None,
                ctypes.byref(size),
                0,
                None,
                None,
                None,
            )
        else:
            status = self._library.nnp_convolution_output(
                0,
                batch,
                input_channels,
                output_channels,
                input_size,
                padding,
                kernel,
                None,
                None,
                None,
                None,
                None,
                ctypes.byref(size),
                0,
                None,
                None,
                None,
            )
        if status != 0:
            raise NnpackSizingUnavailable(f"NNPACK sizing refused with status {status}")
        return size.value


# Fixed ABI/export discovery is retained once, including failure. Input-shaped
# queries still use the actual installed library and its CPU capabilities.
_API = _SizingApi.load()


def nnpack_workspace(
    layout: _QualifiedCpuLayout,
    source: torch.Tensor,
    weight: torch.Tensor,
    stride: list[int],
    padding: list[int],
    groups: int,
) -> int:
    api = _API
    if isinstance(api, _Unavailable):
        raise NnpackSizingUnavailable(api.reason)
    if source.ndim == 3:
        input_size = _Size(int(source.shape[2]), 1)
        kernel = _Size(int(weight.shape[2]), 1)
        native_padding = _Padding(0, padding[0], 0, padding[0])
        native_stride = _Size(stride[0], 1)
    elif source.ndim == 4:
        input_size = _Size(int(source.shape[3]), int(source.shape[2]))
        kernel = _Size(int(weight.shape[3]), int(weight.shape[2]))
        native_padding = _Padding(padding[0], padding[1], padding[0], padding[1])
        native_stride = _Size(stride[1], stride[0])
    else:
        raise NnpackSizingUnavailable("NNPACK requires 1D or 2D convolution geometry")
    return api.workspace(
        int(source.shape[0]),
        int(source.shape[1]) // groups,
        int(weight.shape[0]) // groups,
        input_size,
        native_padding,
        kernel,
        native_stride,
    )
