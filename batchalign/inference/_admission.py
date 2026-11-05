"""Construction boundary for evidence made only by a validating transition."""

from __future__ import annotations

from typing import Never, Self


class _FactoryAdmission:
    """Normal construction cannot manufacture an empty admission proof.

    ``dataclass(init=False)`` alone inherits ``object()`` and still permits
    an uninitialized instance. The owning admission transition initializes
    its validated fields through ``object.__new__`` explicitly.
    """

    __slots__ = ()

    def __new__(cls, admission: Never) -> Self:
        raise TypeError(f"{cls.__name__} requires its validating admission transition")
