"""Focused tests for worker-side Stanza language mapping."""

from __future__ import annotations

import logging

from batchalign.worker._stanza_capabilities import (
    StanzaCapabilityTable,
    StanzaLanguageCapability,
)
from batchalign.worker._stanza_loading import (
    iso3_to_alpha2,
    should_request_mwt,
)


def test_iso3_to_alpha2_maps_known_languages() -> None:
    assert iso3_to_alpha2("eng") == "en"
    assert iso3_to_alpha2("yue") == "zh"


def test_iso3_to_alpha2_preserves_existing_alpha2_codes() -> None:
    assert iso3_to_alpha2("en") == "en"
    assert iso3_to_alpha2("ja") == "ja"


def test_iso3_to_alpha2_leaves_unknown_iso3_unchanged(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="batchalign.worker"):
        assert iso3_to_alpha2("zzz") == "zzz"

    assert "Unknown ISO-639-3 code 'zzz' - passing through unchanged for Stanza" in caplog.text


# ---------------------------------------------------------------------------
# should_request_mwt — capability-driven processor selection
#
# Why this matters: ``load_stanza_models`` previously consulted a hardcoded
# ``MWT_LANGS`` set. That list said Swedish (``sv``) had MWT, but the actual
# Stanza catalog does not ship a Swedish MWT model — every Swedish worker
# spawn raised ``UnsupportedProcessorError`` and the language group failed.
# The 2026-04-15 overnight morphotag run lost an entire 500-file chunk to
# this. The principled fix is to query the capability table at runtime
# rather than maintaining a hand-edited mirror of Stanza's catalog.
# CLAUDE.md: "Per-language processor availability is determined by reading
# Stanza's resources.json at worker startup, NOT by hardcoded tables."
# ---------------------------------------------------------------------------


def _table_with(alpha2: str, *, has_mwt: bool) -> StanzaCapabilityTable:
    """Build a single-entry capability table for one alpha-2 language.

    Pure construction — no Stanza, no resources.json read. Lets us pin the
    helper's behavior without coupling tests to whatever upstream catalog
    happens to ship today.
    """
    cap = StanzaLanguageCapability(alpha2=alpha2, has_mwt=has_mwt)
    return StanzaCapabilityTable(languages={"xxx": cap}, iso3_to_alpha2={"xxx": alpha2})


class TestShouldRequestMwt:
    def test_returns_false_when_capability_table_lacks_mwt(self) -> None:
        # Swedish: Stanza ships tokenize/pos/lemma/depparse but NOT mwt.
        # Asking Stanza for mwt would raise UnsupportedProcessorError.
        table = _table_with("sv", has_mwt=False)
        assert should_request_mwt("sv", table) is False

    def test_returns_true_when_capability_table_has_mwt(self) -> None:
        # English: Stanza ships mwt; we should request it.
        table = _table_with("en", has_mwt=True)
        assert should_request_mwt("en", table) is True

    def test_returns_false_when_alpha2_not_in_table(self) -> None:
        # Conservative fallback: an unknown language is safer without mwt.
        # Stanza will at minimum tokenize/pos/lemma/depparse if those exist;
        # a missing mwt processor is the failure mode we're guarding against.
        table = _table_with("en", has_mwt=True)
        assert should_request_mwt("zz", table) is False

    def test_returns_false_when_table_is_none(self) -> None:
        # ``get_cached_capability_table()`` returns None when Stanza is not
        # importable. We must not request MWT in that case — there is no
        # way to confirm support, and a wrong guess crashes the worker.
        assert should_request_mwt("en", None) is False
