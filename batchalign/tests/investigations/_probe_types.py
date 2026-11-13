"""Typed data model for the Stanza MWT probe matrix.

Every investigation probe case is a :class:`ProbeCase`. The 6-language
registry (``_cases/__init__.py``) aggregates per-language tuples into a
single :data:`LANGUAGE_MATRIX`. The matrix-driven test module
(``test_stanza_mwt_probe_matrix.py``) reads the registry, runs each case
through the paired (free-tokenize vs postprocessor) pipelines, and
enforces the per-case invariant.

Why typed data
--------------
Before the 2026-04-22 consolidation, probe cases were hand-built
``list[tuple[str, list[str]]]`` tables scattered across 5 test modules
(~1080 lines total). Each module reinvented xfail handling, invariant
selection, and pipeline-fixture plumbing. Types here capture the
intent so every new case declares what it means to pass, fail, or
xfail: rather than encoding that in function names and module-level
comments.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import assert_never, overload

from batchalign.inference._admission import _FactoryAdmission


class Phenomenon(Enum):
    """The linguistic construction a probe case exercises.

    Used to group the case in per-language book tables and to help a
    contributor classify a new probe before writing it. Not referenced
    by test assertions: purely descriptive metadata.
    """

    NATIVE_MWT = "native_mwt"
    """Preposition+article or article+preposition contractions that
    Stanza's MWT processor expands natively (``au``, ``del``, ``am``)."""

    CLITIC_ELISION = "clitic_elision"
    """Apostrophe-separated clitic+host forms (``l'ami``, ``c'est``,
    ``qu'il``) where Stanza emits an MWT Range over the apostrophe."""

    CONTRACTION = "contraction"
    """English-style contractions (``don't``, ``isn't``) where Stanza
    emits an MWT hint tuple for its MWT processor to expand."""

    POSSESSIVE = "possessive"
    """Proper-name possessives or Dutch possessive idioms (``Claus's``,
    ``Maria's``, ``'s-avonds``)."""

    IDIOMATIC = "idiomatic"
    """Fixed lexical items with internal apostrophes that function as
    one word (``aujourd'hui``, ``d'água``)."""

    ELISION_PREFIX = "elision_prefix"
    """French elision-prefix tokens (``jusqu'à``, ``puisqu'il``,
    ``quelqu'un``): the Wave 4 invariant-break family."""

    MULTI_CLITIC = "multi_clitic"
    """Stacked apostrophe-internal clitics (``d'l'attraper``)."""

    CONTROL = "control"
    """Plain words used as a baseline to confirm the pipeline is
    otherwise healthy (``maison``, ``huis``)."""


@dataclass(frozen=True)
class XfailMark:
    """Explicit xfail annotation.

    ``defect_slug`` must match an anchor in
    ``book/src/reference/stanza-limitations.md`` so a successor
    reading a surprise xfail can trace it to the registered Stanza
    defect in one step.
    """

    defect_slug: str
    reason: str


@dataclass(frozen=True)
class ObserveOnly:
    """Record external model behavior without asserting a word count."""


@dataclass(frozen=True)
class ExpectedWords:
    count: int


@dataclass(frozen=True)
class KnownDefect:
    """An expected failure always owns the count predicate it must exercise."""

    expected: ExpectedWords
    defect: XfailMark


ProbeExpectation = ObserveOnly | ExpectedWords | KnownDefect


@dataclass(frozen=True, init=False)
class ProbeCase:
    """Source words and one closed observation/check/known-defect intent.

    The existing positional and keyword case notation is admitted here once.
    Its overloads exclude a defect without an expected count. Consumers receive
    the closed payload and never reconstruct it from independent nullable fields.
    """

    label: str
    words: tuple[str, ...]
    phenomenon: Phenomenon
    expectation: ProbeExpectation

    @overload
    def __init__(
        self,
        label: str,
        words: tuple[str, ...],
        phenomenon: Phenomenon,
        expected_post_mwt_count: int,
        xfail: XfailMark | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        label: str,
        words: tuple[str, ...],
        phenomenon: Phenomenon,
        expected_post_mwt_count: None = None,
        xfail: None = None,
    ) -> None: ...

    def __init__(
        self,
        label: str,
        words: tuple[str, ...],
        phenomenon: Phenomenon,
        expected_post_mwt_count: int | None = None,
        xfail: XfailMark | None = None,
    ) -> None:
        expectation: ProbeExpectation
        match expected_post_mwt_count, xfail:
            case None, None:
                expectation = ObserveOnly()
            case int(count), None:
                expectation = ExpectedWords(count)
            case int(count), XfailMark() as defect:
                expectation = KnownDefect(ExpectedWords(count), defect)
            case None, XfailMark():
                raise ValueError("a known defect requires an expected word count")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "words", words)
        object.__setattr__(self, "phenomenon", phenomenon)
        object.__setattr__(self, "expectation", expectation)

    @classmethod
    def observation_alone(
        cls,
        word: str,
        phenomenon: Phenomenon = Phenomenon.CLITIC_ELISION,
    ) -> ProbeCase:
        """Build a single-word observation-only probe with label
        ``f"{word}_alone"`` and words ``(word,)``. Standard shape
        for surface pins where the probe's only job is to record
        what Stanza emits for a single token in isolation.

        Use for allowlist-candidate surfaces where you want to
        confirm Stanza's raw output before committing a reconciler
        entry.
        """
        return cls(
            label=f"{word}_alone",
            words=(word,),
            phenomenon=phenomenon,
        )

    @classmethod
    def strict_alone(
        cls,
        word: str,
        phenomenon: Phenomenon = Phenomenon.CONTROL,
    ) -> ProbeCase:
        """Build a single-word probe with ``expected_post_mwt_count=1``
        - the standard shape for baseline/control tokens that must
        stay 1-to-1 under the postprocessor pipeline. Asserts that
        Stanza doesn't spuriously expand the token into an MWT.

        Use for CONTROL tokens (`hus`, `kot`, `dom`, `vin` etc.)
        and any other surface that should strictly preserve its
        single-token identity.
        """
        return cls(
            label=f"{word}_alone",
            words=(word,),
            phenomenon=phenomenon,
            expected_post_mwt_count=1,
        )


@dataclass(frozen=True, init=False)
class CheckedProbe(_FactoryAdmission):
    """A source case whose immutable expectation contains a real predicate."""

    source: ProbeCase
    expectation: ExpectedWords | KnownDefect

    @classmethod
    def admit(cls, source: ProbeCase) -> CheckedProbe | ObserveOnly:
        match source.expectation:
            case ObserveOnly() as observation:
                return observation
            case ExpectedWords() | KnownDefect() as expectation:
                checked = object.__new__(cls)
                object.__setattr__(checked, "source", source)
                object.__setattr__(checked, "expectation", expectation)
                return checked
            case _ as unreachable:
                assert_never(unreachable)

    @property
    def expected(self) -> ExpectedWords:
        match self.expectation:
            case ExpectedWords() as expected:
                return expected
            case KnownDefect(expected=expected):
                return expected
            case _ as unreachable:
                assert_never(unreachable)
