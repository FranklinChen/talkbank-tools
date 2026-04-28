"""Stanza capability table builder — single source of truth.

Reads Stanza's ``resources.json`` to discover per-language processor
availability.  Replaces 7 scattered hardcoded tables across Python and
Rust with one authoritative data structure built from the installed
Stanza version.

The table is built once (lazily) and cached for the process lifetime.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field

L = logging.getLogger(__name__)

# ISO-639-3 → Stanza alpha-2 overrides for codes that pycountry
# doesn't map correctly or that Stanza uses non-standard identifiers for.
_ISO3_OVERRIDES: dict[str, str] = {
    "nor": "nb",       # Norwegian → Bokmål (Stanza uses "nb")
    "yue": "zh-hans",  # Cantonese → Chinese (Stanza's zh-hans model)
    "cmn": "zh-hans",  # Mandarin → Chinese
    "zho": "zh-hans",  # Chinese (generic) → zh-hans
    "msa": "ms",       # Malay (ISO-639-3 msa) → Stanza ms
}


@dataclass(frozen=True)
class StanzaLanguageCapability:
    """Per-language processor availability from Stanza resources.json."""

    alpha2: str
    has_tokenize: bool = False
    has_pos: bool = False
    has_lemma: bool = False
    has_depparse: bool = False
    has_mwt: bool = False
    has_constituency: bool = False
    has_coref: bool = False


@dataclass(frozen=True)
class StanzaCapabilityTable:
    """Complete capability registry derived from resources.json.

    ``languages`` is keyed by ISO-639-3 code (e.g. ``"eng"``, ``"nld"``).
    ``iso3_to_alpha2`` maps ISO-639-3 → Stanza alpha-2 for all supported
    languages (derived from pycountry + overrides).
    """

    languages: dict[str, StanzaLanguageCapability] = field(default_factory=dict)
    iso3_to_alpha2: dict[str, str] = field(default_factory=dict)
    stanza_version: str = ""


def build_stanza_capability_table_from_resources(
    resources: dict[str, object],
    *,
    stanza_version: str = "unknown",
) -> StanzaCapabilityTable:
    """Build the capability table from an already-loaded resources mapping."""
    # Build alpha2 → capability mapping from resources.json.
    # Skip non-language keys (like "default") and alias entries.
    alpha2_caps: dict[str, StanzaLanguageCapability] = {}

    # Stanza resources keys are alpha-2 codes (or variants like "zh-hans").
    # We check which processors are listed as top-level keys in the resource entry.
    _SKIP_KEYS = {"default"}
    for alpha2, lang_data in resources.items():
        if alpha2 in _SKIP_KEYS:
            continue
        if not isinstance(lang_data, dict):
            continue
        # A real language entry has processor keys like "tokenize", "pos", etc.
        # Alias entries are strings pointing to another language.
        if "tokenize" not in lang_data:
            continue

        alpha2_caps[alpha2] = StanzaLanguageCapability(
            alpha2=alpha2,
            has_tokenize="tokenize" in lang_data,
            has_pos="pos" in lang_data,
            has_lemma="lemma" in lang_data,
            has_depparse="depparse" in lang_data,
            has_mwt="mwt" in lang_data,
            has_constituency="constituency" in lang_data,
            has_coref="coref" in lang_data,
        )

    # Build ISO-639-3 → alpha-2 mapping using pycountry.
    iso3_map: dict[str, str] = {}

    # First: apply explicit overrides (these take priority).
    for iso3, alpha2 in _ISO3_OVERRIDES.items():
        if alpha2 in alpha2_caps:
            iso3_map[iso3] = alpha2

    # Second: use pycountry for standard mappings.
    try:
        import pycountry

        for lang in pycountry.languages:
            alpha3 = getattr(lang, "alpha_3", None)
            alpha2_code: str | None = getattr(lang, "alpha_2", None)
            if not alpha3 or not alpha2_code:
                continue
            # Don't override explicit overrides.
            if alpha3 in iso3_map:
                continue
            # Check if Stanza has this alpha-2 code.
            if alpha2_code in alpha2_caps:
                iso3_map[alpha3] = alpha2_code
    except ImportError:
        L.warning(
            "pycountry not installed — iso3_to_alpha2 mapping will only "
            "include explicit overrides"
        )

    # Build the final table keyed by ISO-639-3.
    languages: dict[str, StanzaLanguageCapability] = {}
    for iso3, alpha2 in iso3_map.items():
        if alpha2 in alpha2_caps:
            languages[iso3] = alpha2_caps[alpha2]

    L.info(
        "Built Stanza capability table: %d languages, %d with constituency, "
        "%d with mwt (stanza %s)",
        len(languages),
        sum(1 for c in languages.values() if c.has_constituency),
        sum(1 for c in languages.values() if c.has_mwt),
        stanza_version,
    )

    return StanzaCapabilityTable(
        languages=languages,
        iso3_to_alpha2=iso3_map,
        stanza_version=stanza_version,
    )


def build_stanza_capability_table() -> StanzaCapabilityTable:
    """Build the capability table from Stanza's installed resources.json.

    This is the single source of truth for what Stanza can process per
    language.  Called once at worker startup; the result is cached.
    """
    import stanza
    import stanza.resources.common as src

    return build_stanza_capability_table_from_resources(
        src.load_resources_json(),
        stanza_version=getattr(stanza, "__version__", "unknown"),
    )


@functools.lru_cache(maxsize=1)
def get_cached_capability_table() -> StanzaCapabilityTable | None:
    """Return the cached capability table, building it on first call.

    Returns ``None`` if Stanza is not installed.
    """
    try:
        return build_stanza_capability_table()
    except ImportError:
        L.warning("Stanza not installed — capability table unavailable")
        return None
    except Exception as e:
        L.warning("Failed to build Stanza capability table: %s", e)
        return None
