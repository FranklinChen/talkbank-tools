#!/usr/bin/env python3
"""Generate the ISO 639-3 to ISO 639-1 table used by batchalign-types.

Why this table exists in batchalign3
------------------------------------
Converting an ISO 639-3 code to its ISO 639-1 two-letter form is a standards
fact, and BA3 needs it to choose provider language parameters (Tencent's
``EngineModelType``). Chatter is the CHAT and language authority and owns the
ISO 639-3 REGISTRY (``talkbank-model``'s ``is_valid_iso639_3``), but it answers
only "is this a real code": its vendored table keeps two columns of identifiers
plus a retirement status and deliberately drops SIL's ``Part1`` column, and the
module is ``pub(crate)``. So there is nothing to consume yet, and this table
lives here until chatter grows the conversion. See the W3 report.

Where the data comes from
-------------------------
``pycountry``, whose language database is itself derived from the ISO 639-3
code tables published by iso639-3.sil.org, the ISO registration authority.
pycountry is used deliberately rather than a fresh download: it is the exact
source the Python worker consulted before this table existed, so generating
from it makes the Rust conversion agree with the previous behaviour code for
code, and the only intended behaviour change is the separate Mandarin fix.

The upstream authority is SIL. If this table is ever regenerated from the SIL
tables directly, take the ``Part1`` column of ``iso-639-3.tab``.

Why the output is committed
---------------------------
Build-time data must be local and offline-reproducible, and this workspace's
dependency manifests are owned elsewhere, so there is no build script to
generate into ``OUT_DIR``. The generated file is committed and this script is
run deliberately when pycountry is upgraded, the same shape as chatter's own
``scripts/update_iso639_3.py``.

Usage:
    uv run scripts/generate_iso639_1_table.py
"""

from __future__ import annotations

from pathlib import Path

import pycountry

TABLE_PATH = (
    Path(__file__).resolve().parent.parent
    / "crates"
    / "batchalign-types"
    / "src"
    / "iso639_part1"
    / "table.rs"
)


def collect_pairs() -> list[tuple[str, str]]:
    """Return every (ISO 639-3, ISO 639-1) pair the registry assigns, sorted.

    Only languages that HAVE a two-letter code appear. Most ISO 639-3
    languages have none, which is a fact about the standard rather than a gap
    in this table, and the Rust side says so with its own typed absence.
    """
    pairs: list[tuple[str, str]] = []
    for language in pycountry.languages:
        alpha_3 = getattr(language, "alpha_3", None)
        alpha_2 = getattr(language, "alpha_2", None)
        if alpha_3 and alpha_2:
            pairs.append((alpha_3, alpha_2))
    pairs.sort()
    return pairs


def render(pairs: list[tuple[str, str]]) -> str:
    """Render the generated Rust module for `pairs`."""
    version = pycountry.__version__
    lines = [
        "//! ISO 639-3 to ISO 639-1 pairs. GENERATED FILE, DO NOT EDIT.",
        "//!",
        "//! Regenerate with `uv run scripts/generate_iso639_1_table.py`, which",
        "//! records where the data comes from and why it is committed.",
        "",
        "/// Where this table came from, carried in the binary so a support",
        "/// question can be answered without reading the generator.",
        f'pub const SOURCE: &str = "pycountry {version} '
        '(ISO 639-3 tables from iso639-3.sil.org)";',
        "",
        "/// Every (ISO 639-3, ISO 639-1) pair, sorted by the three-letter code",
        "/// so the lookup can binary search it.",
        "pub(super) const ISO_639_3_TO_PART1: &[(&str, &str)] = &[",
    ]
    lines.extend(f'    ("{alpha_3}", "{alpha_2}"),' for alpha_3, alpha_2 in pairs)
    lines.append("];")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    pairs = collect_pairs()
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(render(pairs), encoding="utf-8")
    print(f"wrote {len(pairs)} pairs to {TABLE_PATH}")


if __name__ == "__main__":
    main()
