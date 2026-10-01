"""pyannote's torchcodec notice is silenced at Batchalign's import sites only.

torchcodec is excluded from Batchalign deliberately (see `pyproject.toml`), so
pyannote warns at import that audio decoding "will fail". Batchalign never asks
pyannote to decode, so the warning is false for us; the import guard in
`batchalign.inference.audio` drops exactly that message.

Each case runs in a fresh interpreter, because the warning fires only on the
first import of pyannote in a process. The unguarded control proves the test
can see the notice at all: if torchcodec is ever installed, the control fails,
which says the premise changed rather than passing silently.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

_COUNT_TORCHCODEC_WARNINGS = textwrap.dedent(
    """
    import warnings

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        {import_statement}
    print(sum("torchcodec" in str(w.message) for w in seen))
    """
)


def _torchcodec_warnings_when(import_statement: str) -> int:
    script = _COUNT_TORCHCODEC_WARNINGS.format(
        import_statement=textwrap.indent(import_statement, " " * 4).lstrip()
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    return int(completed.stdout.strip().splitlines()[-1])


def test_an_unguarded_pyannote_import_shows_the_notice() -> None:
    assert _torchcodec_warnings_when("import pyannote.audio") > 0


def test_the_guarded_import_drops_only_the_notice() -> None:
    guarded = (
        "from batchalign.inference.audio import "
        "pyannote_import_without_torchcodec_notice\n"
        "with pyannote_import_without_torchcodec_notice():\n"
        "    import pyannote.audio"
    )
    assert _torchcodec_warnings_when(guarded) == 0
