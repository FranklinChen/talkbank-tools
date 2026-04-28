"""Tests for the Stanza capability table builder.

These tests use a deterministic resources fixture instead of depending on a
user-level Stanza cache on the CI runner.
"""

from batchalign.worker._stanza_capabilities import (
    build_stanza_capability_table_from_resources,
)


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
    "alias-en": "en",
}


def build_fixture_table():
    return build_stanza_capability_table_from_resources(
        _FIXTURE_RESOURCES,
        stanza_version="test-stanza",
    )


def test_table_is_non_empty():
    """Fixture resources should produce a non-empty capability table."""
    table = build_fixture_table()
    assert set(table.languages) >= {"eng", "fra", "jpn", "nld", "spa"}
    assert table.stanza_version == "test-stanza"


def test_english_has_constituency():
    """English is one of the fixture languages with constituency parsing."""
    table = build_fixture_table()
    assert "eng" in table.languages
    assert table.languages["eng"].has_constituency


def test_dutch_has_no_constituency():
    """Dutch does NOT have constituency — this caused an operator's crash."""
    table = build_fixture_table()
    assert "nld" in table.languages
    assert not table.languages["nld"].has_constituency


def test_dutch_has_core_processors():
    """Dutch has tokenize, pos, lemma, depparse — morphotag should work."""
    table = build_fixture_table()
    nl = table.languages["nld"]
    assert nl.has_tokenize
    assert nl.has_pos
    assert nl.has_lemma
    assert nl.has_depparse


def test_iso3_mapping_covers_fixture_languages():
    """The derived iso3 mapping should cover every fixture language entry."""
    table = build_fixture_table()
    expected = {"eng", "fra", "jpn", "nld", "spa"}
    missing = expected - set(table.iso3_to_alpha2.keys())
    assert not missing, f"Fixture languages should map cleanly: {missing}"


def test_mwt_matches_resources():
    """MWT availability should come from resources, not a hardcoded list."""
    table = build_fixture_table()
    assert table.languages["fra"].has_mwt
    assert table.languages["eng"].has_mwt
    assert not table.languages["jpn"].has_mwt


def test_japanese_has_constituency():
    """Japanese has constituency parsing in the fixture resources."""
    table = build_fixture_table()
    ja = table.languages["jpn"]
    assert ja.has_constituency


def test_unsupported_language_not_in_table():
    """Languages absent from the fixture should not appear."""
    table = build_fixture_table()
    assert "que" not in table.languages
    assert "jam" not in table.languages
