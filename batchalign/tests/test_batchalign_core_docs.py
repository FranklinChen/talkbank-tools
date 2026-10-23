"""The compiled extension module and its functions carry their own docs.

A Rust doc comment above a `#[pyfunction]` becomes that function's
`__doc__`, and one above the `#[pymodule]` becomes the module's. A function
added between the module's doc comment and the module gave the module no
docstring and the function the module's first line.
"""

from __future__ import annotations

import batchalign_core.batchalign_core as extension


def test_the_extension_module_is_documented() -> None:
    assert extension.__doc__ == "batchalign_core: Rust worker runtime for batchalign3."


def test_a_function_docstring_is_its_own() -> None:
    doc = extension.repair_feats.__doc__
    assert doc is not None
    assert doc.startswith("The worker's FEATS repair")
