"""Tests for the Stanza capability table builder.

These tests use a deterministic resources fixture instead of depending on a
user-level Stanza cache on the CI runner.
"""

from __future__ import annotations

import sys
from unittest import mock

import pytest

from batchalign.worker._stanza_capabilities import (
    StanzaCapabilityTable,
    StanzaCatalogDownloadError,
    StanzaNotInstalled,
    build_stanza_capability_table_from_resources,
    get_cached_capability_table,
)
from batchalign_core import ISO_639_1_PAIRS

_FIXTURE_RESOURCES = {
    "default": {},
    "en": {
        "tokenize": {},
        "pos": {},
        "lemma": {},
        "depparse": {},
        "mwt": {},
        "constituency": {},
    },
    "es": {
        "tokenize": {},
        "pos": {},
        "lemma": {},
        "depparse": {},
    },
    "fr": {
        "tokenize": {},
        "pos": {},
        "lemma": {},
        "depparse": {},
        "mwt": {},
    },
    "ja": {
        "tokenize": {},
        "pos": {},
        "lemma": {},
        "depparse": {},
        "constituency": {},
    },
    "nl": {
        "tokenize": {},
        "pos": {},
        "lemma": {},
        "depparse": {},
        "mwt": {},
    },
    "zh-hans": {
        "tokenize": {},
        "pos": {},
        "lemma": {},
        "depparse": {},
    },
    "nb": {
        "tokenize": {},
        "pos": {},
        "lemma": {},
        "depparse": {},
    },
    "grc": {
        "tokenize": {},
        "pos": {},
        "lemma": {},
        "depparse": {},
    },
    # Aliases as the catalog writes them.
    "zh": {"alias": "zh-hans"},
    "no": {"alias": "nb"},
    "spa": {"alias": "es"},
    # Not a language entry: a top-level string.
    "url": "https://example.invalid/",
}


def build_fixture_table():
    return build_stanza_capability_table_from_resources(
        _FIXTURE_RESOURCES, ISO_639_1_PAIRS
    )


def test_table_is_non_empty():
    """Fixture resources should produce a non-empty capability table."""
    table = build_fixture_table()
    assert set(table.languages) >= {"eng", "fra", "jpn", "nld", "spa"}


def test_english_has_constituency():
    """English is one of the fixture languages with constituency parsing."""
    table = build_fixture_table()
    assert "eng" in table.languages
    assert table.languages["eng"].has("constituency")


def test_dutch_has_no_constituency():
    """Dutch does NOT have constituency, this caused an operator's crash."""
    table = build_fixture_table()
    assert "nld" in table.languages
    assert not table.languages["nld"].has("constituency")


def test_dutch_has_core_processors():
    """Dutch has tokenize, pos, lemma, depparse, morphotag should work."""
    table = build_fixture_table()
    nl = table.languages["nld"]
    assert nl.has("tokenize")
    assert nl.has("pos")
    assert nl.has("lemma")
    assert nl.has("depparse")


@pytest.mark.parametrize(
    ("iso3", "stanza_code"),
    [
        # Through the ISO 639-1 table.
        ("eng", "en"),
        ("fra", "fr"),
        ("jpn", "ja"),
        ("nld", "nl"),
        ("spa", "es"),
        # ISO 639-1 resolved through a catalog alias.
        ("zho", "zh-hans"),
        ("nor", "nb"),
        # ISO 639-1 directly.
        ("nob", "nb"),
        # The catalog's own three-letter key.
        ("grc", "grc"),
        # Policy: no ISO 639-1 code, no catalog entry of their own.
        ("cmn", "zh-hans"),
        ("yue", "zh-hans"),
    ],
)
def test_iso3_codes_map_to_the_entry_that_serves_them(iso3, stanza_code):
    """One mapping: the ISO table, then the catalog and its aliases."""
    assert build_fixture_table().languages[iso3].stanza_code == stanza_code


def test_every_key_is_iso_639_3_and_no_non_language_key_leaks():
    """Only three-letter codes reach the report; `url` and aliases are not
    languages, and two-letter keys are not ISO 639-3."""
    table = build_fixture_table()
    for code in table.languages:
        assert len(code) == 3 and code.isalpha() and code.islower(), code


def test_mwt_matches_resources():
    """MWT availability should come from resources, not a hardcoded list."""
    table = build_fixture_table()
    assert table.languages["fra"].has("mwt")
    assert table.languages["eng"].has("mwt")
    assert not table.languages["jpn"].has("mwt")


def test_japanese_has_constituency():
    """Japanese has constituency parsing in the fixture resources."""
    table = build_fixture_table()
    ja = table.languages["jpn"]
    assert ja.has("constituency")


def test_unsupported_language_not_in_table():
    """Languages absent from the fixture should not appear."""
    table = build_fixture_table()
    assert "que" not in table.languages
    assert "jam" not in table.languages


# ---------------------------------------------------------------------------
# Bootstrap-on-missing tests (the on-demand download contract).
#
# The contract: a fresh install with no Stanza cache must still produce a
# usable capability table. We test the get_cached_capability_table() wrapper
# under three states:
#
#   1. resources.json missing → download succeeds → table populated.
#   2. resources.json missing → download fails    → typed
#      StanzaCatalogDownloadError raised, NOT silent return None.
#   3. Stanza package itself not installed         → StanzaNotInstalled.
#
# These exist because BA3 used to silently return None on any
# ResourcesFileNotFoundError, which masked the fact that no language
# pack had ever been seeded, and the worker exit-1 that resulted was
# classified as a transient crash, retried 3×, and dumped a full Python
# traceback per attempt to the daemon log.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_capability_cache():
    """Reset the lru_cache on get_cached_capability_table between tests."""
    get_cached_capability_table.cache_clear()
    yield
    get_cached_capability_table.cache_clear()


def test_bootstrap_downloads_catalog_when_missing():
    """resources.json absent on first call → download + return populated table.

    Pre-fix: returns None silently (the bug). Post-fix: triggers
    download_resources_json, retries, returns a populated table.
    """
    from stanza.resources.common import ResourcesFileNotFoundError

    # First call to load_resources_json raises (catalog missing); after the
    # fake download_resources_json runs, the second call returns our fixture.
    fake_load = mock.Mock(
        side_effect=[
            ResourcesFileNotFoundError("/fake/path/resources.json"),
            _FIXTURE_RESOURCES,
        ]
    )
    fake_download = mock.Mock(return_value=None)

    with (
        mock.patch("stanza.resources.common.load_resources_json", fake_load),
        mock.patch("stanza.resources.common.download_resources_json", fake_download),
    ):
        table = get_cached_capability_table()

    assert isinstance(table, StanzaCapabilityTable), (
        "Bootstrap must download the catalog and return a populated table"
    )
    assert "eng" in table.languages
    fake_download.assert_called_once()
    assert fake_load.call_count == 2  # one miss, one success after download


def test_bootstrap_raises_typed_error_on_download_failure():
    """resources.json missing + download fails → typed error, not silent None.

    A real network failure must be visible. The orchestrator can later
    classify this as non-retryable; the user can read it as an actionable
    error. Silent return-None is the bug.
    """
    from stanza.resources.common import ResourcesFileNotFoundError

    fake_load = mock.Mock(
        side_effect=ResourcesFileNotFoundError("/fake/path/resources.json")
    )
    fake_download = mock.Mock(side_effect=ConnectionError("network unreachable"))

    with (
        mock.patch("stanza.resources.common.load_resources_json", fake_load),
        mock.patch("stanza.resources.common.download_resources_json", fake_download),
    ):
        with pytest.raises(StanzaCatalogDownloadError) as excinfo:
            get_cached_capability_table()

    assert "network unreachable" in str(excinfo.value) or isinstance(
        excinfo.value.__cause__, ConnectionError
    )


def test_stanza_not_installed_is_its_own_outcome():
    """The ``stanza`` import failing is "not installed", reported as itself."""
    with mock.patch.dict(sys.modules, {"stanza.resources.common": None}):
        result = get_cached_capability_table()

    assert isinstance(result, StanzaNotInstalled)


def test_a_failure_after_stanza_imports_is_not_not_installed():
    """RED FIRST: the catch-all ``except ImportError`` around the whole build
    turned a stale ``batchalign_core`` extension (no ``ISO_639_1_PAIRS``) into
    "Stanza not installed", and the report into an empty catalog. Only the
    ``stanza`` import is "not installed"; any other failure propagates with
    its own type and message, and the capability report names it."""
    from batchalign.worker._handlers import _stanza_catalog_report
    from batchalign.worker._types import StanzaCatalogFailed

    stale = ImportError("cannot import name 'ISO_639_1_PAIRS' from 'batchalign_core'")
    with mock.patch(
        "batchalign.worker._stanza_capabilities.build_stanza_capability_table",
        mock.Mock(side_effect=stale),
    ):
        with pytest.raises(ImportError, match="ISO_639_1_PAIRS"):
            get_cached_capability_table()
        report = _stanza_catalog_report()

    assert report == StanzaCatalogFailed(
        reason="ImportError: cannot import name 'ISO_639_1_PAIRS' from 'batchalign_core'"
    )


def test_each_command_requirement_comes_from_rust():
    """The worker's requirement sets are Rust's ``StanzaRequirement``, read
    through ``batchalign_core``, not copies."""
    from batchalign.worker._stanza_capabilities import (
        processors_argument,
        requirement_processors,
    )

    assert requirement_processors("morphosyntax") == {
        "tokenize",
        "pos",
        "lemma",
        "depparse",
    }
    assert requirement_processors("utterance_segmentation") == {
        "tokenize",
        "pos",
        "constituency",
    }
    assert requirement_processors("coreference") == {"tokenize", "coref"}
    assert (
        processors_argument({"mwt", "depparse", "tokenize"}) == "tokenize,mwt,depparse"
    )
