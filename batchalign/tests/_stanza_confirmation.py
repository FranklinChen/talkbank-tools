"""Which Stanza versions our documented behaviour has actually been confirmed on.

The defect-mitigation map carries a "Stanza version confirmed" column, and
several modules repeat a version in prose ("verified against stanza 1.13.0 on
...", "13k verb surface forms on stanza 1.13.0"). Every one of those is a claim
about a version that nothing compares against the version actually installed, so
bumping Stanza silently inherits every confirmation ever recorded.

That cost real time on 2026-07-31. Bumping to 1.14.0 produced two golden
failures, and there was no way to tell from the run which were caused by the
bump and which predated it, because no baseline was attached to anything. One
turned out to be a genuine 1.14.0 regression and the other had been failing
under 1.13.0 already. Distinguishing them needed a manual downgrade-and-rerun
that the version set below now makes unnecessary.

The rule was already written down. Writing it down is what failed. A frozen set
plus one test means an unconfirmed version announces itself instead of being
assumed fine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_VERSION = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)")


@dataclass(frozen=True, order=True)
class StanzaVersion:
    """A Stanza release, ordered numerically.

    A dataclass rather than the raw string Stanza hands out, for two reasons.
    The question asked of it is membership in a confirmed set, and string
    equality against "1.13" would answer that wrong for "1.13.0". And the
    ordering is load-bearing wherever these are sorted: as strings, "1.10.1"
    sorts BEFORE "1.9.0", so a version list would read as nonsense the first
    time a minor reached double digits.
    """

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, raw: str) -> StanzaVersion:
        matched = _VERSION.match(raw)
        if matched is None:
            raise ValueError(f"unparseable Stanza version {raw!r}")
        return cls(
            major=int(matched["major"]),
            minor=int(matched["minor"]),
            patch=int(matched["patch"]),
        )

    @classmethod
    def installed(cls) -> StanzaVersion:
        import stanza

        return cls.parse(stanza.__version__)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


def _v(raw: str) -> StanzaVersion:
    return StanzaVersion.parse(raw)


BUMP_PROCEDURE = """Finishing a Stanza bump:
  1. Run the Python golden suite: pytest batchalign/tests -m golden
  2. Run the Rust ML golden morphotag suite (its snapshots, L2 included):
     cargo test -p batchalign --features ml-golden --test ml_golden -- --test-threads=1 ml_golden::morphotag::golden
  3. Account for EVERY failure, every XPASS and every changed snapshot line. A
     failure may predate the bump, so establish that by running it on a
     confirmed version before blaming the new one. Accept a snapshot only for
     lines whose change is explained.
  4. Update book/src/batchalign/architecture/stanza-defect-mitigation-map.md
  5. Add the version to CONFIRMED_STANZA_VERSIONS."""

# Versions the golden suite has been run against with its results adjudicated.
#
# ADDING ONE IS THE LAST STEP OF A BUMP, NOT THE FIRST: see `BUMP_PROCEDURE`,
# which the failing assertion also prints, so the steps are stated once. An
# entry means "someone looked", and it is the only place that claim is
# machine-readable.
CONFIRMED_STANZA_VERSIONS: frozenset[StanzaVersion] = frozenset(
    {
        _v("1.10.1"),
        _v("1.11.1"),
        _v("1.12.0"),
        _v("1.12.1"),
        _v("1.13.0"),
        _v("1.14.0"),
        # 2026-10-01: golden suite attributed on 1.14.0 vs 1.15.0 with the same
        # BA3 code; see stanza-limitations.md, "Stanza 1.15.0 upgrade".
        _v("1.15.0"),
    }
)
