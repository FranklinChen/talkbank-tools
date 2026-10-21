#!/usr/bin/env python3
"""Generate the ISO 639-3 to Whisper language-token table used by batchalign-types.

Why this table exists
---------------------
Stock Whisper is told which language to decode by a token from its own
vocabulary: ``en``, ``el``, ``yue``, ``haw``, and ``jw`` for Javanese. That
set is a fact about the model, not about ISO 639: most two-letter codes have no
Whisper token, and one Whisper token (``jw``) is not an ISO code at all. Before
this table, the control plane guessed membership from ISO 639-1 and the worker
handed Whisper a pycountry language NAME, so the two disagreed in both
directions: languages with no token were admitted and failed on every file,
and Greek, Malay, Nepali, Nynorsk, Occitan and Swahili, which Whisper supports,
failed because their pycountry names ("modern greek (1453-)") are not Whisper
names. The control plane now owns the mapping, sends the worker the token
itself, and admits exactly the languages that have one.

Where the data comes from
-------------------------
``transformers.models.whisper.tokenization_whisper.LANGUAGES``: the token table
the Whisper decoder in the worker actually checks. Each token is paired with
its ISO 639-3 code through pycountry, the same registry behind
``generate_iso639_1_table.py``: a two-letter token by its ISO 639-1 pairing, a
three-letter token as itself, and the one token that is neither (``jw``) by
the language name Whisper gives it. A token that cannot be paired stops the
generator; it is never dropped silently.

Input aliases
-------------
``ALIASES`` adds ISO 639-3 codes that are not the registry's code for a token's
language but that BA3 users pass for it. Each entry names its reason.

Why the output is committed
---------------------------
As for the ISO 639-1 table: build-time data must be local and reproducible, so
the generated file is committed and this script is run deliberately when
transformers is upgraded.

Usage:
    uv run scripts/generate_whisper_language_table.py
"""

from __future__ import annotations

from pathlib import Path

import pycountry
import transformers
from transformers.models.whisper.tokenization_whisper import LANGUAGES

TABLE_PATH = (
    Path(__file__).resolve().parent.parent
    / "crates"
    / "batchalign-types"
    / "src"
    / "whisper_language"
    / "table.rs"
)

# ISO 639-3 code -> (Whisper token, reason). Only codes the registry does not
# already pair with the token's language.
ALIASES: dict[str, tuple[str, str]] = {
    "cmn": (
        "zh",
        "Mandarin, the individual language inside the zho macrolanguage; BA3 "
        "accepts both codes, and the worker already decoded cmn with Whisper's "
        "Chinese token",
    ),
}


def iso3_for_token(token: str, name: str) -> str:
    """The ISO 639-3 code of the language a Whisper token names."""
    if len(token) == 2:
        language = pycountry.languages.get(alpha_2=token)
        if language is not None:
            return str(language.alpha_3)
    if len(token) == 3:
        language = pycountry.languages.get(alpha_3=token)
        if language is not None:
            return str(language.alpha_3)
    language = pycountry.languages.get(name=name)
    if language is not None:
        return str(language.alpha_3)
    raise SystemExit(
        f"Whisper token {token!r} ({name}) has no ISO 639-3 pairing; "
        "add it to ALIASES with a reason, never drop it"
    )


def collect_pairs() -> list[tuple[str, str]]:
    """Every (ISO 639-3, Whisper token) pair, sorted by the ISO code."""
    pairs: dict[str, str] = {}
    for token, name in LANGUAGES.items():
        iso3 = iso3_for_token(token, name)
        if iso3 in pairs:
            raise SystemExit(f"{iso3} pairs with both {pairs[iso3]!r} and {token!r}")
        pairs[iso3] = token
    for iso3, (token, _) in ALIASES.items():
        if token not in LANGUAGES:
            raise SystemExit(
                f"alias {iso3} names {token!r}, which is not a Whisper token"
            )
        if iso3 in pairs:
            raise SystemExit(
                f"alias {iso3} is already the registry code for {pairs[iso3]!r}"
            )
        pairs[iso3] = token
    return sorted(pairs.items())


def render_row(iso3: str, token: str) -> str:
    """One table row; an alias row carries its reason as a comment."""
    row = f'    ("{iso3}", "{token}"),'
    if iso3 in ALIASES:
        return f"    // Alias: {ALIASES[iso3][1]}.\n{row}"
    return row


def render(pairs: list[tuple[str, str]]) -> str:
    """Render the generated Rust module for `pairs`."""
    lines = [
        "//! ISO 639-3 to stock Whisper language tokens. GENERATED FILE, DO NOT EDIT.",
        "//!",
        "//! Regenerate with `uv run scripts/generate_whisper_language_table.py`,",
        "//! which records where the data comes from and why it is committed.",
        "",
        "/// Where this table came from, carried in the binary so a support",
        "/// question can be answered without reading the generator.",
        f'pub const SOURCE: &str = "transformers {transformers.__version__} '
        f"Whisper LANGUAGES ({len(LANGUAGES)} tokens), paired by pycountry "
        f'{pycountry.__version__}";',
        "",
        "/// Every (ISO 639-3, Whisper token) pair, sorted by the three-letter code",
        "/// so the lookup can binary search it.",
        "pub(super) const ISO_639_3_TO_WHISPER: &[(&str, &str)] = &[",
    ]
    lines.extend(render_row(iso3, token) for iso3, token in pairs)
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
