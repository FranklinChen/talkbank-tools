"""Spanish probe cases.

Baseline coverage for a language with NO per-language BA2 override
rules. Covers the two canonical preposition+article contractions
(``al = a + el``, ``del = de + el``).

2026-04-23 parity audit: locked at observed counts as Stanza-drift
sentinels. Context forms expand; the alone forms are postprocessor
1-to-1 sentinels. Downstream Rust-side Range reassembly collapses
these back to 1-to-1 for final CHAT output, so production parity
with BA2 is preserved.

2026-06-19 (Stanza 1.13.0 re-evaluation): the ``del_alone`` drift
sentinel fired. Isolated ``del`` no longer MWT-expands to 2 UD
words; it now stays 1-to-1 (observed ``[('del', 'ADP', 'del')]``),
matching ``al``. Verified identical on the deployed Stanza 1.12.2
and on 1.13.0, so the change predates 1.13.0 (it landed with the
1.12.1 Spanish updates). Zero production impact: final CHAT output
is 1-to-1 either way. Re-locked ``del_alone`` at the new observed
count (1).

2026-10-01 (Stanza 1.15.0): isolated ``del`` MWT-expands again
(``de``+``el``), reversing the 1.12.1 change re-locked above. Re-locked
at 2. Final CHAT output is one ``%mor`` item either way.
"""

from __future__ import annotations

from .._probe_types import Phenomenon, ProbeCase

CASES: tuple[ProbeCase, ...] = (
    ProbeCase("al_alone", ("al",), Phenomenon.NATIVE_MWT, 1),
    ProbeCase("del_alone", ("del",), Phenomenon.NATIVE_MWT, 2),
    ProbeCase(
        "al_in_context",
        ("voy", "al", "cine"),
        Phenomenon.NATIVE_MWT,
        4,
    ),
    ProbeCase(
        "del_in_context",
        ("el", "libro", "del", "niño"),
        Phenomenon.NATIVE_MWT,
        5,
    ),
)
