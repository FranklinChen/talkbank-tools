"""End-to-end CLI test: BA3 morphotag produces CHAT-correct %mor for
the four copula-contraction constructions that originally prompted the
Rust-side ``nlp::invariants::finite_verb_main_clause`` rewrite.

This is the single place where we verify the rewrite lands correctly
through the full pipeline (parse CHAT → extract → Stanza → Rust rewrite
→ inject %mor → serialize). The Python-layer observation tests in
``test_stanza_mwt_copula_observations.py`` document Stanza's
intermediate output (which is wrong for sink/lady); this file asserts
BA3's final output regardless of what Stanza did along the way.

Both assertions share a single CLI invocation via the
``morphotag_output`` fixture to keep test runtime bounded, one
managed analysis supplies both assertions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pytest

from ._managed_morphotag import ManagedMorphotag


class PredicateRelation(Enum):
    VERBAL = "AUX"
    NOMINAL = "COP"


@dataclass(frozen=True)
class CopulaExample:
    surface: str
    utterance: str
    predicate: PredicateRelation


# The sink analysis retains the model's nominal overflow predicate. Both
# 1.15.0 and 1.15.1 emit COP there; manufacturing an AUX relation would detach
# it from that predicate analysis. The other three predicates are verbal.
COPULA_EXAMPLES = (
    CopulaExample("stool's", "the stool's going over .", PredicateRelation.VERBAL),
    CopulaExample("he's", "and he's falling over .", PredicateRelation.VERBAL),
    CopulaExample("sink's", "and the sink's overflowing .", PredicateRelation.NOMINAL),
    CopulaExample("lady's", "the lady's washing dishes .", PredicateRelation.VERBAL),
)

CHAT_FIXTURE = (
    "@UTF8\n"
    "@Begin\n"
    "@Languages:\teng\n"
    "@Participants:\tPAR Adult\n"
    "@ID:\teng|test|PAR|||||Adult|||\n"
    + "".join(f"*PAR:\t{case.utterance}\n" for case in COPULA_EXAMPLES)
    + "@End\n"
)


# CHAT-correct %mor substring that every contracted 's case must produce.
# "aux|be-Fin-Ind-Pres-S3" is the contracted copula "is" in the CHAT
# morphological conventions BA3 emits.
REQUIRED_CLITIC_MOR = "~aux|be-Fin-Ind-Pres-S3"


@pytest.fixture(scope="module")
def morphotag_output() -> str:
    """Run ``batchalign3 morphotag`` once per module; return the output CHAT text.

    One completed managed analysis is shared by the MOR and GRA assertions.
    """
    return ManagedMorphotag.from_environment().run(CHAT_FIXTURE)


@pytest.mark.golden
@pytest.mark.integration
def test_all_four_copula_contractions_produce_correct_mor(
    morphotag_output: str,
) -> None:
    """Every ``<subject>'s <present-participle>`` utterance must emit
    tilde-joined ``~aux|be-Fin-Ind-Pres-S3``, no exceptions, no
    possessive readings allowed through.

    Guards: all four copula contractions produce correct mor for the supplied linguistic source.
    Kind: journey
    Not a type: ManagedMorphotag excludes direct-mode composition, but cannot prove subprocess success or these model-derived serialized MOR and GRA bytes. The residual observation is all four copula contractions produce correct mor for the supplied linguistic source.
    State: The shared managed CLI fixture stages one CHAT source and reads its real output file.
    Concurrency: Each submission has private temporary paths while the server owns shared model scheduling.
    """
    mor_lines = [
        line for line in morphotag_output.splitlines() if line.startswith("%mor:")
    ]
    assert len(mor_lines) == len(COPULA_EXAMPLES), (
        f"Expected {len(COPULA_EXAMPLES)} %mor lines, got {len(mor_lines)}. Output:\n{morphotag_output}"
    )

    for example, mor_line in zip(COPULA_EXAMPLES, mor_lines, strict=True):
        assert REQUIRED_CLITIC_MOR in mor_line, (
            f"[{example.surface}] BA3 did not produce {REQUIRED_CLITIC_MOR!r} for "
            f"the contracted 's. Full %mor line:\n  {mor_line}\n"
            f"Full output:\n{morphotag_output}"
        )
        for bad in (
            "noun|stool-Plur",
            "noun|he-Plur",
            "noun|lady-Plur",
            "~part|s",  # possessive reading leaking through
        ):
            assert bad not in mor_line, (
                f"[{example.surface}] Degraded analysis {bad!r} present in "
                f"%mor. BA3 emitted:\n  {mor_line}\n"
            )


@pytest.mark.golden
@pytest.mark.integration
def test_gra_tier_consistent_with_rewritten_mor(
    morphotag_output: str,
) -> None:
    """Every utterance's %gra must attach its finite be component according
    to the model's verbal or nominal predicate, with NSUBJ and ROOT. Catches the
    class of bug where %mor and %gra diverge after a rewrite.

    Guards: gra tier consistent with rewritten mor for the supplied linguistic source.
    Kind: journey
    Not a type: ManagedMorphotag excludes direct-mode composition, but cannot prove subprocess success or these model-derived serialized MOR and GRA bytes. The residual observation is gra tier consistent with rewritten mor for the supplied linguistic source.
    State: The shared managed CLI fixture stages one CHAT source and reads its real output file.
    Concurrency: Each submission has private temporary paths while the server owns shared model scheduling.
    """
    gra_lines = [
        line for line in morphotag_output.splitlines() if line.startswith("%gra:")
    ]
    assert len(gra_lines) == len(COPULA_EXAMPLES), (
        f"Expected {len(COPULA_EXAMPLES)} %gra lines, got {len(gra_lines)}"
    )

    for example, gra_line in zip(COPULA_EXAMPLES, gra_lines, strict=True):
        assert example.predicate.value in gra_line, (
            f"{example.surface}: expected {example.predicate.value} for "
            f"{example.predicate.name.lower()} predicate:\n  {gra_line}"
        )
        assert "NSUBJ" in gra_line, (
            f"%gra line missing NSUBJ (subject) relation:\n  {gra_line}"
        )
        assert "ROOT" in gra_line, f"%gra line missing ROOT relation:\n  {gra_line}"
