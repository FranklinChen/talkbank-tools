# Re-export everything from the native Rust extension module.
# maturin places the .so as batchalign_core/batchalign_core.cpython-*.so
from batchalign_core.batchalign_core import *  # noqa: F401,F403
