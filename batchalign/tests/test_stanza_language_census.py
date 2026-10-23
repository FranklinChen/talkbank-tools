"""Which languages morphotag can analyze, measured on a real Stanza catalog.

The server once admitted a primary language only if a hand-maintained list of
53 codes named it, while the worker's capability table, built from the
installed catalog, was the authority everywhere else. The list refused Kazakh,
Serbian, Belarusian and dozens more that the catalog serves in full. The list
is gone; the capability table (Rust's ISO 639-1 table plus the catalog) is the
one mapping, and the server's registry admits from what it reports.

This census pins the migration against a committed snapshot of the Stanza
1.15.0 catalog's shape (``fixtures/stanza_catalog_shape.json``, written by
``scripts/snapshot_stanza_catalog.py``): every language the list admitted is
still admitted, except four whose catalog entries lack a lemmatizer, which the
list admitted at the file level although no morphotag pipeline could be built
for them (the per-utterance admission refused them). Regenerate the fixture
after a Stanza upgrade and revisit these expectations deliberately.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from batchalign.worker._stanza_capabilities import (
    StanzaCapabilityTable,
    build_stanza_capability_table_from_resources,
)
from batchalign_core import ISO_639_1_PAIRS

_FIXTURE = Path(__file__).parent / "fixtures" / "stanza_catalog_shape.json"

# The deleted static list (`SUPPORTED_STANZA_CODES`), frozen here as the
# census baseline. It is history, not an authority: nothing reads it but this
# test.
_FORMER_STATIC_LIST = frozenset(
    """eng spa fra deu ita por nld cat glg dan swe nor fin est lav lit isl pol
    ces ron hun bul hrv slk slv ukr rus ell cym gle gla eus mlt ara heb fas hin
    urd tur tam tel tha vie ind zho cmn yue jpn kor kat hye afr lat""".split()
)

# Admitted by the list, refused now: each catalog entry lacks `lemma`, so no
# morphotag pipeline exists for it (Maltese, Telugu, Thai, Vietnamese).
_LACK_LEMMA = {"mlt": "mt", "tel": "te", "tha": "th", "vie": "vi"}


@pytest.fixture(scope="module")
def table() -> StanzaCapabilityTable:
    snapshot = json.loads(_FIXTURE.read_text())
    assert snapshot["stanza_version"] == "1.15.0"
    return build_stanza_capability_table_from_resources(
        snapshot["resources"], ISO_639_1_PAIRS
    )


def test_every_code_the_static_list_admitted_is_still_admitted(
    table: StanzaCapabilityTable,
) -> None:
    assert len(_FORMER_STATIC_LIST) == 53
    lost = sorted(
        code for code in _FORMER_STATIC_LIST if not table.supports_morphosyntax(code)
    )
    assert lost == sorted(_LACK_LEMMA)


@pytest.mark.parametrize(("iso3", "stanza_code"), sorted(_LACK_LEMMA.items()))
def test_each_exception_is_an_entry_without_a_lemmatizer(
    table: StanzaCapabilityTable, iso3: str, stanza_code: str
) -> None:
    cap = table.languages[iso3]
    assert cap.stanza_code == stanza_code
    assert cap.has("tokenize") and cap.has("pos") and cap.has("depparse")
    assert not cap.has("lemma")


@pytest.mark.parametrize(
    ("iso3", "stanza_code"),
    [
        # Refused by the static list, served in full by the catalog.
        ("kaz", "kk"),
        ("srp", "sr"),
        ("sqi", "sq"),
        ("bel", "be"),
        ("mar", "mr"),
        ("pan", "pa"),
        ("yid", "yi"),
        ("uzb", "uz"),
        ("kir", "ky"),
        ("hat", "ht"),
        ("wol", "wo"),
        ("oci", "oc"),
        ("uig", "ug"),
        ("san", "sa"),
        # Catalog entries keyed by their own three-letter code.
        ("ckb", "ckb"),
        ("grc", "grc"),
        ("kmr", "kmr"),
        # North Sami: ISO 639-1 `se` is a catalog alias of `sme`.
        ("sme", "sme"),
        # Norwegian: the macrolanguage resolves through the catalog's `no`
        # alias to Bokmål, as it did through the old override; Bokmål and
        # Nynorsk resolve to their own entries.
        ("nor", "nb"),
        ("nob", "nb"),
        ("nno", "nn"),
    ],
)
def test_catalog_languages_are_admitted_with_their_entry(
    table: StanzaCapabilityTable, iso3: str, stanza_code: str
) -> None:
    assert table.supports_morphosyntax(iso3)
    assert table.languages[iso3].stanza_code == stanza_code


@pytest.mark.parametrize(
    ("iso3", "stanza_code"),
    [
        # Chinese: `zho` through ISO `zh`, a catalog alias; Mandarin and
        # Cantonese by policy (Cantonese POS is then PyCantonese's).
        ("zho", "zh-hans"),
        ("cmn", "zh-hans"),
        ("yue", "zh-hans"),
        # Japanese loads its combined packages; Italian carries the MWT
        # policy; Greek the lemma pretrain repair. Each keys off this code.
        ("jpn", "ja"),
        ("ita", "it"),
        ("ell", "el"),
    ],
)
def test_special_routes_keep_their_stanza_code(
    table: StanzaCapabilityTable, iso3: str, stanza_code: str
) -> None:
    assert table.supports_morphosyntax(iso3)
    assert table.languages[iso3].stanza_code == stanza_code


@pytest.mark.parametrize("iso3", ["que", "fao", "mya"])
def test_languages_without_a_morphotag_pipeline_are_not_admitted(
    table: StanzaCapabilityTable, iso3: str
) -> None:
    """Quechua has no entry; Faroese and Burmese have entries without one."""
    assert not table.supports_morphosyntax(iso3)


def test_every_key_is_an_iso_639_3_code(table: StanzaCapabilityTable) -> None:
    """The server's registry refuses a report with any other key."""
    for code in table.languages:
        assert len(code) == 3 and code.isascii() and code.isalpha(), code
        assert code.islower(), code


def test_measurement_of_the_admitted_set(table: StanzaCapabilityTable) -> None:
    """MEASUREMENT on the Stanza 1.15.0 snapshot, with a floor rather than an
    exact count, so a fixture regenerated for a new Stanza fails only if the
    set collapses."""
    admitted = {code for code in table.languages if table.supports_morphosyntax(code)}
    assert len(admitted) >= 100, len(admitted)
    assert len(admitted - _FORMER_STATIC_LIST) >= 50
