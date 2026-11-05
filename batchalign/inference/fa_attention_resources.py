"""Admit virtual execution of the installed attention composite kernels.

FakeTensor's top-level attention result hides internal score/mask storage.
The actual installed composite implementation can expose its primitive virtual
operations to the owning allocation tracker; no real numerical kernel runs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import torch
from torch._ops import OpOverload
from torch.utils._pytree import tree_flatten

from batchalign.inference._admission import _FactoryAdmission
from batchalign.inference.fa_native_resources import NativeWorkspaceUnavailable

if TYPE_CHECKING:
    from batchalign.inference.fa_virtual_resources import _VirtualAllocations


class _CompositeKernel(Enum):
    IMPLICIT = torch._C.DispatchKey.CompositeImplicitAutograd
    EXPLICIT = torch._C.DispatchKey.CompositeExplicitAutograd


@dataclass(frozen=True, slots=True)
class _UnavailableComposite:
    reason: str


@dataclass(frozen=True, slots=True, init=False)
class _AdmittedComposite(_FactoryAdmission):
    _invoke: Callable[..., object]

    @classmethod
    def read(
        cls, operation: OpOverload, kernel: _CompositeKernel
    ) -> _AdmittedComposite | _UnavailableComposite:
        if not torch._C._dispatch_has_kernel_for_dispatch_key(
            operation._schema.name, kernel.value
        ):
            return _UnavailableComposite(
                f"attention operator lacks its declared composite kernel: {operation}"
            )
        admitted = object.__new__(cls)
        # This is the same registered-kernel port used by OpOverload.decompose.
        # The explicit safe-softmax composite is needed as well as the implicit
        # attention composites; its masks are not visible in its return value.
        invoke = partial(cast(Callable[..., object], operation._op_dk), kernel.value)
        object.__setattr__(admitted, "_invoke", invoke)
        return admitted


_COMPOSITES: Mapping[OpOverload, _AdmittedComposite | _UnavailableComposite] = (
    MappingProxyType(
        {
            operation: _AdmittedComposite.read(operation, kernel)
            for operation, kernel in (
                (
                    torch.ops.aten.scaled_dot_product_attention.default,
                    _CompositeKernel.IMPLICIT,
                ),
                (
                    torch.ops.aten._scaled_dot_product_attention_math.default,
                    _CompositeKernel.IMPLICIT,
                ),
                (torch.ops.aten._safe_softmax.default, _CompositeKernel.EXPLICIT),
            )
        }
    )
)


@dataclass(frozen=True, slots=True, init=False)
class VirtualAttentionCall(_FactoryAdmission):
    """One bound virtual call, with no substitute operator or input parameters."""

    _composite: _AdmittedComposite
    _owner: _VirtualAllocations
    _args: tuple[object, ...]
    _kwargs: Mapping[str, object]

    @classmethod
    def prepare(
        cls,
        operation: Callable[..., object],
        args: tuple[object, ...],
        kwargs: dict[str, object],
        owner: _VirtualAllocations,
    ) -> VirtualAttentionCall | None:
        if not isinstance(operation, OpOverload):
            return None
        composite = _COMPOSITES.get(operation)
        if composite is None:
            return None
        if isinstance(composite, _UnavailableComposite):
            raise NativeWorkspaceUnavailable(composite.reason)
        values, _ = tree_flatten((args, kwargs))
        tensors = [value for value in values if isinstance(value, torch.Tensor)]
        if not tensors:
            raise NativeWorkspaceUnavailable(
                "attention projection requires tensor inputs"
            )
        if any(not owner.owns(value) for value in tensors):
            raise NativeWorkspaceUnavailable(
                "attention projection requires virtual tensors from its owning epoch"
            )
        admitted = object.__new__(cls)
        object.__setattr__(admitted, "_composite", composite)
        object.__setattr__(admitted, "_owner", owner)
        object.__setattr__(admitted, "_args", args)
        object.__setattr__(admitted, "_kwargs", MappingProxyType(dict(kwargs)))
        return admitted

    def project(self) -> object:
        # The owning epoch remains active even for scalar/mask constructors
        # with no tensor input. Re-enter this same ledger above that epoch.
        with self._owner.reenter_epoch():
            output = self._composite._invoke(*self._args, **self._kwargs)
        values, _ = tree_flatten(output)
        if any(
            isinstance(value, torch.Tensor) and not self._owner.owns(value)
            for value in values
        ):
            raise NativeWorkspaceUnavailable(
                "attention composite escaped virtual allocation projection"
            )
        return output
