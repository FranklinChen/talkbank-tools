"""Matrix-driven Stanza decision-probe runner (v2, token-centric).

For every :class:`DecisionProbeCase` in the decision matrix this
runs the language's with-postprocessor Stanza pipeline on both
``pre_words`` and ``post_words``, builds token-centric
:class:`StanzaTokenOutput` tuples from ``doc.sentences[*].tokens``,
calls :func:`compare_stanza_outputs`, prints observed vs. expected,
and asserts (or xfails, or short-circuits on OBSERVE_ONLY).

Token-centric addressing
------------------------
Earlier v1 flattened ``doc.sentences[*].words`` and indexed by the
case author's pre-tokenized input position. That silently broke on
MWT expansions (``i'll`` → ``i`` + ``will`` shifts every later
index). v2 addresses **pre-MWT tokens** via
``doc.sentences[*].tokens`` (which are 1-to-1 with the input word
list when ``tokenize_no_ssplit`` is True and the tokenizer does not
split further), and carries each token's post-MWT UD word tuple as
``token.words``.

Marked ``@pytest.mark.golden`` so it runs only on machines with
real Stanza models.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, assert_never

import pytest

from batchalign.inference._admission import _FactoryAdmission

from ._cases import LanguageKey
from ._decision_cases import all_decision_cases
from ._decision_probe_types import (
    DecisionComparison,
    DecisionOutcome,
    DecisionProbeCase,
    StanzaTokenOutput,
    StanzaWordResult,
    compare_stanza_outputs,
)
from ._probe_types import ObserveOnly


def _model_field(word: Any, name: Literal["text", "upos", "lemma", "deprel"]) -> str:
    """Admit the observed model field without inventing an empty replacement."""
    value = getattr(word, name)
    if not isinstance(value, str):
        raise ValueError(f"Stanza word {word!r} has no textual {name}: {value!r}")
    return value


def _stanza_doc_to_tokens(doc: Any) -> tuple[StanzaTokenOutput, ...]:
    """Flatten a Stanza Document into our token-centric projection.

    Walks ``doc.sentences[*].tokens``; these are pre-MWT tokens,
    1-to-1 with the input word list under ``tokenize_no_ssplit`` +
    our realignment postprocessor. For each token, collects its
    ``.words`` (post-MWT UD words) into a :class:`StanzaTokenOutput`.

    Note: the returned sequence may not be 1-to-1 with the input
    word list if Stanza's tokenizer further splits an input word
    (rare with our postprocessor active). The caller's mapping
    indices are trusted to reflect Stanza's actual token count;
    out-of-range indices raise at comparator time with a clear
    message.
    """
    out: list[StanzaTokenOutput] = []
    for sent in doc.sentences:
        for token in sent.tokens:
            words = tuple(
                StanzaWordResult(
                    text=_model_field(w, "text"),
                    upos=_model_field(w, "upos"),
                    lemma=_model_field(w, "lemma"),
                    deprel=_model_field(w, "deprel"),
                )
                for w in token.words
            )
            out.append(StanzaTokenOutput(text=token.text, words=words))
    return tuple(out)


def _case_id(pair: tuple[LanguageKey, DecisionProbeCase]) -> str:
    lang, case = pair
    return f"{lang.alpha3}__{case.candidate_class.value}__{case.label}"


type AssertedDecisionOutcome = Literal[
    DecisionOutcome.POST_STRICTLY_BETTER,
    DecisionOutcome.POST_NEUTRAL,
    DecisionOutcome.POST_STRICTLY_WORSE,
    DecisionOutcome.MIXED,
]


@dataclass(frozen=True, init=False)
class CheckedDecisionProbe(_FactoryAdmission):
    """Only an adjudicated expectation grants assertion-suite admission."""

    source: DecisionProbeCase
    expected: AssertedDecisionOutcome

    @classmethod
    def admit(cls, source: DecisionProbeCase) -> CheckedDecisionProbe | ObserveOnly:
        match source.expected_outcome:
            case DecisionOutcome.OBSERVE_ONLY:
                return ObserveOnly()
            case (
                DecisionOutcome.POST_STRICTLY_BETTER
                | DecisionOutcome.POST_NEUTRAL
                | DecisionOutcome.POST_STRICTLY_WORSE
                | DecisionOutcome.MIXED
            ) as expected:
                checked = object.__new__(cls)
                object.__setattr__(checked, "source", source)
                object.__setattr__(checked, "expected", expected)
                return checked
            case _ as remaining_outcome:
                assert_never(remaining_outcome)


@dataclass(frozen=True, init=False)
class DecisionObservation(_FactoryAdmission):
    """The exact case and completed pre/post projections of real inference."""

    language: LanguageKey
    source: DecisionProbeCase
    pre_tokens: tuple[StanzaTokenOutput, ...]
    post_tokens: tuple[StanzaTokenOutput, ...]
    comparison: DecisionComparison


_PARAMETERS = []
for language, source in all_decision_cases():
    match CheckedDecisionProbe.admit(source):
        case ObserveOnly():
            continue
        case CheckedDecisionProbe() as checked:
            _PARAMETERS.append(
                pytest.param(
                    (language, checked),
                    id=_case_id((language, source)),
                )
            )


def observe_stanza_decision(
    pair: tuple[LanguageKey, DecisionProbeCase],
    post_pipeline_for: Callable[[LanguageKey], tuple[Any, Any]],
) -> DecisionObservation:
    """Measure an adjudicated or observation-only case without claiming a test pass."""
    lang, case = pair
    nlp, ctx = post_pipeline_for(lang)

    ctx.original_words = [list(case.pre_words)]
    try:
        pre_tokens = _stanza_doc_to_tokens(nlp(" ".join(case.pre_words)))
    finally:
        ctx.original_words = []
    ctx.original_words = [list(case.post_words)]
    try:
        post_tokens = _stanza_doc_to_tokens(nlp(" ".join(case.post_words)))
    finally:
        ctx.original_words = []

    comparison = compare_stanza_outputs(
        pre_tokens=pre_tokens,
        post_tokens=post_tokens,
        mappings=case.affected_mappings,
    )

    print(
        f"\n{lang.alpha3} {case.candidate_class.value} {case.label}: "
        f"observed={comparison.observed_outcome.value} "
        f"expected={case.expected_outcome.value} "
        f"notes={comparison.notes}"
    )

    observation = object.__new__(DecisionObservation)
    object.__setattr__(observation, "language", lang)
    object.__setattr__(observation, "source", case)
    object.__setattr__(observation, "pre_tokens", pre_tokens)
    object.__setattr__(observation, "post_tokens", post_tokens)
    object.__setattr__(observation, "comparison", comparison)
    return observation


@pytest.mark.golden
@pytest.mark.decision_probe
@pytest.mark.parametrize("pair", _PARAMETERS)
def test_stanza_decision_probe(
    pair: tuple[LanguageKey, CheckedDecisionProbe],
    post_pipeline_for: Callable[[LanguageKey], tuple[Any, Any]],
) -> None:
    """Guards: Real pre/post token projections produce the adjudicated normalization verdict.
    Kind: measurement
    Not a type: CheckedDecisionProbe excludes undecided expectations, but the external neural weights determine the actual tags, lemmas and token text used by the semantic comparator.
    State: Real Stanza pipelines produce both projections, retained with their source in DecisionObservation before comparison is asserted.
    Concurrency: Module-owned pipelines execute serially and bind/clear the context separately for each source projection.
    """
    language, checked = pair
    observation = observe_stanza_decision((language, checked.source), post_pipeline_for)
    assert observation.comparison.observed_outcome is checked.expected, (
        f"{language.alpha3} {checked.source.label}: "
        f"expected {checked.expected.value}, "
        f"got {observation.comparison.observed_outcome.value}; "
        f"notes={observation.comparison.notes}"
    )
