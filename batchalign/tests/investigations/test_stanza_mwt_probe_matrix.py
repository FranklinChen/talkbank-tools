"""Matrix-driven Stanza MWT probe harness.

One test module for all languages. Reads the typed
:data:`_cases.LANGUAGE_MATRIX` registry and parametrizes every
:class:`_probe_types.ProbeCase` through the paired pipelines:

* ``observe_stanza_mwt_free_tokenize`` records the native tokenizer and
  MWT output for explicit investigations. It is a measurement function.

* ``test_stanza_mwt_probe_with_postprocessor``: BA3's tokenizer
  postprocessor (char-DP realigner, no per-language override rules
  as of 2026-04-22) active. Enforces the case's
  declared count predicate, observes otherwise. Known defects
  marked via :class:`_probe_types.XfailMark` surface the pinned
  Stanza limitation without failing the suite.

Only admitted count checks enter pytest. The free-tokenize measurements
and observation-only postprocessor cases remain available for explicit reports.
Both measurement paths require real Stanza models.

History
-------
Before 2026-04-22 these probes lived in 5 hand-written modules under
``batchalign/tests/investigations/test_stanza_*.py``. The typed-matrix
consolidation replaced them with this single module + the data-only
``_cases/<lang>.py`` fixtures. See
``book/src/reference/languages/<lang>.md`` for the per-language audit
records the probes support.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, assert_never

import pytest

from batchalign.inference._admission import _FactoryAdmission

from ._cases import LanguageKey, all_cases
from ._probe_types import (
    CheckedProbe,
    ExpectedWords,
    KnownDefect,
    ObserveOnly,
    ProbeCase,
)

TokenSummary = tuple[str, str | None, str | None]


def _token_summary(doc: Any) -> tuple[TokenSummary, ...]:
    """Retain the model's words and actual missing fields without sentinels."""
    return tuple(
        (word.text, word.upos, word.lemma)
        for sentence in doc.sentences
        for word in sentence.words
    )


@dataclass(frozen=True, init=False)
class ProbeObservation(_FactoryAdmission):
    """The source and completed token projection of one real model call."""

    language: LanguageKey
    source: ProbeCase
    tokens: tuple[TokenSummary, ...]

    @property
    def word_count(self) -> int:
        return len(self.tokens)


def _record(language: LanguageKey, source: ProbeCase, doc: Any) -> ProbeObservation:
    observation = object.__new__(ProbeObservation)
    object.__setattr__(observation, "language", language)
    object.__setattr__(observation, "source", source)
    object.__setattr__(observation, "tokens", _token_summary(doc))
    return observation


def _case_id(pair: tuple[LanguageKey, ProbeCase]) -> str:
    lang, case = pair
    return f"{lang.alpha3}__{case.label}"


class StanzaCountMismatch(AssertionError):
    """Only the declared count comparison may produce an expected failure."""


_POST_PARAMETERS = []
marks: tuple[pytest.MarkDecorator, ...]
for language, source in all_cases():
    match CheckedProbe.admit(source):
        case ObserveOnly():
            # Observation-only measurements have no assertion to qualify.
            continue
        case CheckedProbe() as checked:
            match checked.expectation:
                case KnownDefect(defect=defect):
                    marks = (
                        pytest.mark.xfail(
                            strict=True,
                            raises=StanzaCountMismatch,
                            reason=f"{defect.defect_slug}: {defect.reason}",
                        ),
                    )
                case ExpectedWords():
                    marks = ()
                case _ as remaining_expectation:
                    assert_never(remaining_expectation)
            _POST_PARAMETERS.append(
                pytest.param(
                    (language, checked),
                    marks=marks,
                    id=_case_id((language, source)),
                )
            )
        case _ as remaining_admission:
            assert_never(remaining_admission)


def observe_stanza_mwt_free_tokenize(
    pair: tuple[LanguageKey, ProbeCase],
    pipeline: Any,
) -> ProbeObservation:
    """Explicit diagnostic measurement for the matching free-tokenize pipeline."""
    language, source = pair
    observation = _record(language, source, pipeline(" ".join(source.words)))
    print(
        f"{source.label} {language.alpha2} free "
        f"(chat={len(source.words)}) stanza_words={observation.word_count} "
        f"observed={observation.tokens}"
    )
    return observation


def observe_stanza_mwt_with_postprocessor(
    pair: tuple[LanguageKey, ProbeCase],
    pipeline_for: Callable[[LanguageKey], tuple[Any, Any]],
) -> ProbeObservation:
    """Measure once and retain the exact source and completed token projection."""
    language, source = pair
    pipeline, context = pipeline_for(language)
    context.original_words = [list(source.words)]
    try:
        document = pipeline(" ".join(source.words))
    finally:
        context.original_words = []
    observation = _record(language, source, document)
    print(
        f"{source.label} {language.alpha2} post "
        f"(chat={len(source.words)}) stanza_words={observation.word_count} "
        f"observed={observation.tokens}"
    )
    return observation


@pytest.mark.golden
@pytest.mark.mwt_probe
@pytest.mark.parametrize("pair", _POST_PARAMETERS)
def test_stanza_mwt_probe_with_postprocessor(
    pair: tuple[LanguageKey, CheckedProbe],
    post_pipeline_for: Callable[[LanguageKey], tuple[Any, Any]],
) -> None:
    """Guards: Bound CHAT source words produce the declared expanded UD word count.
    Kind: measurement
    Not a type: CheckedProbe excludes observation-only cases and requires a count predicate, but external neural weights still determine the actual tokenizer and MWT components.
    State: The real Stanza pipeline and trained weights supply the UD words, projected once into a source-bound ProbeObservation.
    Concurrency: Module-owned pipelines execute serially; each call binds its own source words and clears the context on exit.
    """
    language, checked = pair
    observation = observe_stanza_mwt_with_postprocessor(
        (language, checked.source),
        post_pipeline_for,
    )
    expected = checked.expected
    if observation.word_count != expected.count:
        raise StanzaCountMismatch(
            f"{language.alpha3} {observation.source.label}: "
            f"CHAT={len(observation.source.words)} "
            f"expected_stanza={expected.count} "
            f"got_stanza={observation.word_count} observed={observation.tokens}"
        )
