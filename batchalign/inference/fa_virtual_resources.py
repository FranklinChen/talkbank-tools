"""Owned, metadata-only model projection and live virtual-storage accounting."""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, NewType

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeCopyMode, FakeTensor, FakeTensorMode
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.overrides import enable_reentrant_dispatch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from batchalign.inference._admission import _FactoryAdmission
from batchalign.inference.fa_attention_resources import VirtualAttentionCall
from batchalign.inference.fa_native_resources import NativeCpuConvolutionWorkspace

if TYPE_CHECKING:
    from batchalign.worker._types_v2 import FaPreparationOutcomeV2

_TensorId = NewType("_TensorId", int)
_StorageId = NewType("_StorageId", int)


class FaResourceProjectionError(RuntimeError):
    """A model could not supply request-shaped allocation evidence."""


@dataclass(frozen=True, slots=True)
class UnavailableFaPreparation:
    """A backend without qualified allocation evidence cannot grant execution."""

    reason: str

    def __call__(
        self, binding_json: str, words: list[str], text: str
    ) -> FaPreparationOutcomeV2:
        from batchalign.worker._types_v2 import FaProjectionUnavailableV2, FaRefusedV2

        return FaRefusedV2(cause=FaProjectionUnavailableV2(message=self.reason))


@dataclass(frozen=True, slots=True, init=False)
class _PreparedVirtualModel(_FactoryAdmission):
    """One copied model, numerical-fallback-free epoch and storage census."""

    mode: FakeTensorMode
    model: nn.Module
    persistent: frozenset[_StorageId]

    @classmethod
    def admit(cls, model: nn.Module) -> _PreparedVirtualModel:
        # Real parameter metadata must enter FakeCopyMode at the copy boundary.
        # Missing metadata kernels must never run numerical fallback on zeros.
        mode = FakeTensorMode(allow_non_fake_inputs=True, allow_fallback_kernels=False)
        serving_modules = {id(module) for module in model.modules()}
        with mode, FakeCopyMode(mode):
            virtual_model = deepcopy(model)
            persistent = set()
            for module in virtual_model.modules():
                if id(module) in serving_modules:
                    raise FaResourceProjectionError(
                        "model copy reused a serving module"
                    )
                for tensor in chain(
                    module.parameters(recurse=False), module.buffers(recurse=False)
                ):
                    if (
                        not isinstance(tensor, FakeTensor)
                        or tensor.fake_mode is not mode
                    ):
                        raise FaResourceProjectionError(
                            "model copy escaped its virtual epoch"
                        )
                    persistent.add(_StorageId(tensor.untyped_storage()._cdata))
            # Change modes only after the copy and its tensor ownership qualify.
            virtual_model.eval().requires_grad_(False)
        admitted = object.__new__(cls)
        object.__setattr__(admitted, "mode", mode)
        object.__setattr__(admitted, "model", virtual_model)
        object.__setattr__(admitted, "persistent", frozenset(persistent))
        return admitted


@dataclass(slots=True)
class _LiveStorage:
    size: int
    references: int


class _VirtualAllocations(TorchDispatchMode):  # type: ignore[no-untyped-call]
    """Own the live virtual tensor/storage relation without module hooks.

    Views share storage and are charged once. Each tracked tensor owns one
    reference; retiring its weak reference releases that share. Model weights
    are already leased and do not become request workspace through a view.
    No real tensor allocator or serving-model hook is modified.
    """

    def __init__(self, prepared: _PreparedVirtualModel) -> None:
        # PyTorch's mode registration and initializer are unannotated; the
        # dispatcher override and the storage ledger below are fully typed.
        super().__init__()  # type: ignore[no-untyped-call]
        self._prepared = prepared
        self._persistent = prepared.persistent
        self._tensors: dict[_TensorId, weakref.ReferenceType[torch.Tensor]] = {}
        self._storage: dict[_StorageId, _LiveStorage] = {}
        self._current = 0
        self.peak = 0

    @contextmanager
    def observe(self) -> Iterator[nn.Module]:
        with (
            self._prepared.mode,
            torch.inference_mode(),
            sdpa_kernel(SDPBackend.MATH),
            self,
        ):
            yield self._prepared.model

    @contextmanager
    def reenter_epoch(self) -> Iterator[None]:
        with self._prepared.mode, self, enable_reentrant_dispatch():
            yield

    def owns(self, tensor: torch.Tensor) -> bool:
        return (
            isinstance(tensor, FakeTensor) and tensor.fake_mode is self._prepared.mode
        )

    def validate_output(self, output: object) -> None:
        values, _ = tree_flatten(output)
        if any(
            isinstance(value, torch.Tensor) and not self.owns(value) for value in values
        ):
            raise FaResourceProjectionError(
                "operation returned a tensor outside its virtual epoch"
            )

    def _retire(self, tensor_id: _TensorId, storage_id: _StorageId) -> None:
        del self._tensors[tensor_id]
        storage = self._storage[storage_id]
        storage.references -= 1
        if storage.references == 0:
            self._current -= storage.size
            del self._storage[storage_id]

    def _track(self, tensor: torch.Tensor) -> None:
        tensor_id = _TensorId(id(tensor))
        storage = tensor.untyped_storage()
        storage_id = _StorageId(storage._cdata)
        if tensor_id in self._tensors or storage_id in self._persistent:
            return
        if storage_id in self._storage:
            self._storage[storage_id].references += 1
        else:
            size = storage.nbytes()
            self._storage[storage_id] = _LiveStorage(size=size, references=1)
            self._current += size
            self.peak = max(self.peak, self._current)
        self._tensors[tensor_id] = weakref.ref(
            tensor, lambda _: self._retire(tensor_id, storage_id)
        )

    def __torch_dispatch__(
        self,
        func: Callable[..., object],
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        arguments = {} if kwargs is None else kwargs
        attention = VirtualAttentionCall.prepare(func, args, arguments, self)
        if attention is None:
            output = func(*args, **arguments)
        else:
            # Re-enter this ledger while the actual composite emits primitive
            # virtual operations. Merely selecting math attention still lets
            # FakeTensor return opaque metadata for its score/softmax stages.
            output = attention.project()
        values, _ = tree_flatten(output)
        for value in values:
            if isinstance(value, torch.Tensor):
                self._track(value)
        native = NativeCpuConvolutionWorkspace.observe(func, args, arguments, output)
        if native is not None:
            self.peak = max(self.peak, self._current + native.additional_bytes)
        return output


@dataclass(frozen=True, slots=True, init=False)
class ProjectedModelWorkspace[Output](_FactoryAdmission):
    """A completed virtual operation, with its storage-lifetime peak."""

    peak_workspace_bytes: int
    output: Output


def project_model_workspace[Output](
    model: nn.Module, operation: Callable[[nn.Module], Output]
) -> ProjectedModelWorkspace[Output]:
    """Observe one metadata-only model epoch; numerical fallback is forbidden.

    Conversion admits the model, virtual mode and persistent-storage census
    together. The resulting ledger owns them; attention calls cannot substitute
    a mode or another ledger. Unknown metadata operators refuse instead of
    executing real tensors to discover a shape.
    """
    try:
        prepared = _PreparedVirtualModel.admit(model)
        tracker = _VirtualAllocations(prepared)
        with tracker.observe() as virtual_model:
            output = operation(virtual_model)
        tracker.validate_output(output)
        if tracker.peak <= 0:
            raise FaResourceProjectionError(
                "model operation produced no allocation evidence"
            )
        value: ProjectedModelWorkspace[Output] = object.__new__(ProjectedModelWorkspace)
        object.__setattr__(value, "peak_workspace_bytes", tracker.peak)
        object.__setattr__(value, "output", output)
        return value
    except FaResourceProjectionError:
        raise
    except Exception as error:
        raise FaResourceProjectionError(f"{type(error).__name__}: {error}") from error


def model_storage_bytes(model: nn.Module) -> int:
    """Persistent tensor storage, with shared parameter/buffer views counted once."""
    storages = {
        tensor.untyped_storage()._cdata: tensor.untyped_storage().nbytes()
        for tensor in (*model.parameters(), *model.buffers())
    }
    total = sum(storages.values())
    if total <= 0:
        raise FaResourceProjectionError("loaded model has no positive tensor storage")
    return total
