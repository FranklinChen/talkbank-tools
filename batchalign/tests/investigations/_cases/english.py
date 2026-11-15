"""English MWT probe cases.

Covers the phenomena BA2-jan9 explicitly gated in its English
apostrophe-contraction rule (``ud.py:694-697``):

* Standard contractions (``don't``, ``I'm``, ``can't``,
  ``you're``, ``it's``, ``they've``, ``we'd``, ``she'll``,
  ``won't``, ``isn't``). Stanza's English MWT processor expands
  these to 2 UD words; we assert the count so a regression
  would surface.
* Possessives (``John's``, ``dog's``). Stanza treats possessive
  ``'s`` as a separate UD word via MWT expansion.
* ``o'clock``: BA2's explicit control. The apostrophe rule did
  NOT fire on ``o'`` + word, so ``o'clock`` stayed unsplit.
  Probe asserts Stanza produces 1 UD word.
* ``gonna``: Stanza's English MWT expands this to ``gon + na``
  (or similar) even without an apostrophe. Assert 2 UD words.
* Native MWT baselines: short controls to confirm the English
  pipeline is otherwise healthy.

BA2 rule origin
---------------
BA2 ``tokenizer_processor()`` line 694-697::

    elif (("en" in lang) and matches_in(i, "'") and
          not (len(conform(i).split("'")) > 1 and
               conform(i).split("'")[0].strip() == "o")):
        res.append((conform(i), True))

The ``True`` flag told BA2's downstream morphology pass to join
the contraction back together after Stanza split it. BA3 has no
equivalent; Stanza's native MWT processor is expected to handle
these cases. These probes confirm whether Stanza does.

The prepared seed owns a nonempty contraction population and both o'clock
controls. The registry consumes that seed. Model count predicates remain in
the shared golden runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, final

from batchalign.inference._admission import _FactoryAdmission

from .._probe_types import Phenomenon, ProbeCase


@final
@dataclass(frozen=True, init=False)
class ContractionProbe(_FactoryAdmission):
    """A declared contraction always owns a count predicate."""

    source: ProbeCase

    @classmethod
    def declare(
        cls,
        *,
        label: str,
        words: tuple[str, *tuple[str, ...]],
        expected_post_mwt_count: int,
    ) -> ContractionProbe:
        source = ProbeCase(
            label=label,
            words=words,
            phenomenon=Phenomenon.CONTRACTION,
            expected_post_mwt_count=expected_post_mwt_count,
        )
        declaration = object.__new__(cls)
        object.__setattr__(declaration, "source", source)
        return declaration


@final
@dataclass(frozen=True, init=False)
class EnglishMwtSeed(_FactoryAdmission):
    """Completed English fixture preparation, with required contraction coverage."""

    cases: tuple[ProbeCase, ...]

    @classmethod
    def prepare(
        cls,
        *,
        contractions: tuple[ContractionProbe, *tuple[ContractionProbe, ...]],
        possessives: tuple[ProbeCase, ...],
        compounds: tuple[ContractionProbe, ...],
        controls: tuple[ProbeCase, ...],
    ) -> EnglishMwtSeed:
        seed = object.__new__(cls)
        object.__setattr__(
            seed,
            "cases",
            (
                *(probe.source for probe in contractions),
                *possessives,
                *(probe.source for probe in compounds),
                ProbeCase(
                    label="oclock_alone",
                    words=("o'clock",),
                    phenomenon=Phenomenon.CONTROL,
                    expected_post_mwt_count=1,
                ),
                ProbeCase(
                    label="oclock_in_context",
                    words=("at", "six", "o'clock"),
                    phenomenon=Phenomenon.CONTROL,
                    expected_post_mwt_count=3,
                ),
                *controls,
            ),
        )
        return seed


SEED: Final = EnglishMwtSeed.prepare(
    contractions=(
        # ── Standard contractions (MWT should expand to 2 UD words) ─────
        ContractionProbe.declare(
            label="dont_alone",
            words=("don't",),
            expected_post_mwt_count=2,
        ),
        ContractionProbe.declare(
            label="im_alone",
            words=("I'm",),
            expected_post_mwt_count=2,
        ),
        ContractionProbe.declare(
            label="cant_alone",
            words=("can't",),
            expected_post_mwt_count=2,
        ),
        ContractionProbe.declare(
            label="youre_alone",
            words=("you're",),
            expected_post_mwt_count=2,
        ),
        ContractionProbe.declare(
            label="its_alone",
            words=("it's",),
            expected_post_mwt_count=2,
        ),
        ContractionProbe.declare(
            label="theyve_alone",
            words=("they've",),
            expected_post_mwt_count=2,
        ),
        ContractionProbe.declare(
            label="wed_alone",
            words=("we'd",),
            expected_post_mwt_count=2,
        ),
        ContractionProbe.declare(
            label="shell_alone",
            words=("she'll",),
            expected_post_mwt_count=2,
        ),
        ContractionProbe.declare(
            label="wont_alone",
            words=("won't",),
            expected_post_mwt_count=2,
        ),
        ContractionProbe.declare(
            label="isnt_alone",
            words=("isn't",),
            expected_post_mwt_count=2,
        ),
        # ── Contraction in context ─────────────────────────────────────
        ContractionProbe.declare(
            label="dont_in_context",
            # 4 CHAT words → expect 5 UD words (don't expands to 2)
            words=("I", "don't", "know", "that"),
            expected_post_mwt_count=5,
        ),
    ),
    possessives=(
        # ── Possessives (MWT-expansion family, 2 UD words) ─────────────
        ProbeCase(
            label="johns_possessive",
            words=("John's", "book"),
            phenomenon=Phenomenon.POSSESSIVE,
            expected_post_mwt_count=3,
        ),
        ProbeCase(
            label="dogs_possessive",
            words=("the", "dog's", "tail"),
            phenomenon=Phenomenon.POSSESSIVE,
            expected_post_mwt_count=4,
        ),
    ),
    compounds=(
        # ── Compound MWT (Stanza expands even without apostrophe) ──────
        ContractionProbe.declare(
            label="gonna_alone",
            words=("gonna",),
            expected_post_mwt_count=2,
        ),
        ContractionProbe.declare(
            label="gonna_in_context",
            words=("I'm", "gonna", "go"),
            # I'm → 2, gonna → 2, go → 1: total 5 UD
            expected_post_mwt_count=5,
        ),
    ),
    controls=(
        # ── Native MWT / plain-word controls ───────────────────────────
        ProbeCase(
            label="hello_alone",
            words=("hello",),
            phenomenon=Phenomenon.CONTROL,
            expected_post_mwt_count=1,
        ),
        ProbeCase(
            label="the_dog_plain",
            words=("the", "dog", "ran"),
            phenomenon=Phenomenon.CONTROL,
            expected_post_mwt_count=3,
        ),
    ),
)
