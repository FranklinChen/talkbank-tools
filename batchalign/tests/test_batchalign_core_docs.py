"""The compiled extension module and its functions carry their own docs.

A Rust doc comment above a `#[pyfunction]` becomes that function's
`__doc__`, and one above the `#[pymodule]` becomes the module's. A function
added between the module's doc comment and the module gave the module no
docstring and the function the module's first line.
"""

from __future__ import annotations

import batchalign_core.batchalign_core as extension


def test_module_and_function_docs_bind_to_their_exported_objects() -> None:
    """Guards: The imported module owns the runtime description and the repair function owns its own documentation.
    Kind: wire
    Not a type: PyO3 attaches docstrings when constructing CPython objects from Rust annotations.
    State: The installed extension supplies both actual exported objects.
    Concurrency: The assertions read immutable extension metadata in this interpreter.
    """
    assert "Rust worker runtime" in (extension.__doc__ or "")
    doc = extension.repair_feats.__doc__
    assert doc is not None
    assert doc.startswith("The worker's FEATS repair")
