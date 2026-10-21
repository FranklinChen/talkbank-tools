"""Type stubs for the batchalign_core package.

The package re-exports its compiled extension module, exactly as the runtime
`batchalign_core/__init__.py` does, so the stub has the runtime's shape and
`stubtest` checks every definition in `batchalign_core.pyi` against the built
extension (`make batchalign-stubtest`).
"""

from .batchalign_core import *  # noqa: F403
