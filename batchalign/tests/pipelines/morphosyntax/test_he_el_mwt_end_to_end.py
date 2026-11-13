"""End-to-end CHAT pipeline test: Hebrew and Greek MWT splits land
correctly in ``%mor`` after the 2026-04-15 capability-driven loader
fix. Companion to ``test_stanza_he_el_et_mwt_splits.py`` (pure
Stanza observation tests): this file verifies that what Stanza
splits, BA3 actually writes as tilde-joined ``%mor``.

This is the regression gate. If a future BA3 change suppresses MWT
for he/el, by reintroducing a hardcoded include/exclude set, by
breaking the capability table, or by mis-handling MWT in
``morphosyntax/inject.rs``: these tests fail loudly.

Discovery date: 2026-04-15. Defect 5 in
``book/src/reference/stanza-limitations.md`` carries the full
write-up (Ignas's BA2 ``fec893e1`` history, the BA3 regression,
the capability-driven fix, and per-language behavior).

Pattern source: ``test_preserve_mwt_end_to_end.py`` (one CLI
submission per fixture, with completed output shared across assertions).
"""

from __future__ import annotations

import pytest

from ._managed_morphotag import ManagedMorphotag

# Minimal valid CHAT requires @UTF8, @Begin, @Languages, @Participants,
# @ID, utterances, @End. The constructions are the same ones whose
# Stanza splits are pinned in test_stanza_he_el_et_mwt_splits.py.

_HEBREW_FIXTURE = (
    "@UTF8\n"
    "@Begin\n"
    "@Languages:\theb\n"
    "@Participants:\tPAR Adult\n"
    "@ID:\theb|test|PAR|||||Adult|||\n"
    "*PAR:\tבבית גדול .\n"
    "*PAR:\tמהילד הזה .\n"
    "@End\n"
)

# Surface tokens that MUST appear as tilde-joined splits in their
# corresponding %mor lines (one entry per *PAR utterance, in order).
_HEBREW_MWT_SURFACES_PER_LINE: list[list[str]] = [
    ["בבית"],  # line 1: בבית splits as ב+בית
    ["מהילד", "הזה"],  # line 2: מהילד and הזה both split
]
# `לאישה היפה` was a third line until Stanza 1.15.0 stopped splitting
# `לאישה` (2026-10-01). That is Stanza's model, not this wiring, so it is
# pinned as a strict xfail in test_stanza_he_el_et_mwt_splits.py; the two
# lines above still prove BA3 keeps Hebrew MWT on.


_GREEK_FIXTURE = (
    "@UTF8\n"
    "@Begin\n"
    "@Languages:\tell\n"
    "@Participants:\tPAR Adult\n"
    "@ID:\tell|test|PAR|||||Adult|||\n"
    "*PAR:\tστο σπίτι μου .\n"
    "*PAR:\tστον δρόμο .\n"
    "*PAR:\tστις πέντε .\n"
    "@End\n"
)

_GREEK_MWT_SURFACES_PER_LINE: list[list[str]] = [
    ["στο"],
    ["στον"],
    ["στις"],
]


def _mor_lines(chat_text: str) -> list[str]:
    """Return the ``%mor:`` tier lines from a CHAT document, in order."""
    return [line for line in chat_text.splitlines() if line.startswith("%mor:")]


# ---------------------------------------------------------------------------
# Hebrew
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hebrew_morphotag_output() -> str:
    return ManagedMorphotag.from_environment().run(_HEBREW_FIXTURE)


@pytest.mark.golden
@pytest.mark.integration
def test_hebrew_mwt_emits_tilde_joined_mor(
    hebrew_morphotag_output: str,
) -> None:
    """Each Hebrew utterance with a contraction must have a tilde
    ``~`` in its ``%mor`` line; that is the CHAT marker for an
    MWT-joined morph sequence. If a future regression suppresses
    MWT for Hebrew (e.g., a hardcoded list reappears, or the
    capability table breaks), the contracted forms would surface
    as single morph items with no ``~`` and this test fails.

    Guards: hebrew mwt emits tilde joined mor for the supplied linguistic source.
    Kind: journey
    Not a type: ManagedMorphotag excludes direct-mode composition, but cannot prove subprocess success or these model-derived serialized MOR and GRA bytes. The residual observation is hebrew mwt emits tilde joined mor for the supplied linguistic source.
    State: The shared managed CLI fixture stages one CHAT source and reads its real output file.
    Concurrency: Each submission has private temporary paths while the server owns shared model scheduling.
    """
    mor_lines = _mor_lines(hebrew_morphotag_output)
    assert len(mor_lines) == len(_HEBREW_MWT_SURFACES_PER_LINE), (
        f"Expected {len(_HEBREW_MWT_SURFACES_PER_LINE)} %mor lines, "
        f"got {len(mor_lines)}.\nFull output:\n{hebrew_morphotag_output}"
    )
    for utterance_idx, (mor_line, surfaces) in enumerate(
        zip(mor_lines, _HEBREW_MWT_SURFACES_PER_LINE, strict=True)
    ):
        assert "~" in mor_line, (
            f"Hebrew %mor line {utterance_idx + 1} is missing the "
            f"tilde-joined MWT marker ~. The surface "
            f"contraction(s) {surfaces} should have produced "
            f"tilde-joined splits.\nLine: {mor_line}\n"
            f"Full output:\n{hebrew_morphotag_output}"
        )


@pytest.mark.golden
@pytest.mark.integration
def test_hebrew_mwt_does_not_emit_unsplit_surface_as_lemma(
    hebrew_morphotag_output: str,
) -> None:
    """The contracted surface form must not appear as a lemma of a
    single morph item; that would mean MWT was silently suppressed
    and the surface was tagged whole. Stanza's split components
    (e.g., ב, בית) are the correct lemmas after MWT fires.

    Guards: hebrew mwt does not emit unsplit surface as lemma for the supplied linguistic source.
    Kind: journey
    Not a type: ManagedMorphotag excludes direct-mode composition, but cannot prove subprocess success or these model-derived serialized MOR and GRA bytes. The residual observation is hebrew mwt does not emit unsplit surface as lemma for the supplied linguistic source.
    State: The shared managed CLI fixture stages one CHAT source and reads its real output file.
    Concurrency: Each submission has private temporary paths while the server owns shared model scheduling.
    """
    mor_lines = _mor_lines(hebrew_morphotag_output)
    for utterance_idx, (mor_line, surfaces) in enumerate(
        zip(mor_lines, _HEBREW_MWT_SURFACES_PER_LINE, strict=True)
    ):
        for surface in surfaces:
            # A single un-split morph item with this surface as lemma
            # would look like ``<pos>|<surface>`` somewhere in the
            # tier without being tilde-adjacent. Easiest check: the
            # surface must not appear as a bare lemma followed by a
            # space or end-of-line (i.e., not part of a tilde join).
            bad_substring = f"|{surface} "
            assert bad_substring not in mor_line + " ", (
                f"Hebrew %mor line {utterance_idx + 1} contains "
                f"the contracted surface {surface!r} as an unsplit "
                f"lemma (substring {bad_substring!r}). MWT was "
                f"suppressed.\nLine: {mor_line}\n"
                f"Full output:\n{hebrew_morphotag_output}"
            )


# ---------------------------------------------------------------------------
# Greek
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def greek_morphotag_output() -> str:
    return ManagedMorphotag.from_environment().run(_GREEK_FIXTURE)


@pytest.mark.golden
@pytest.mark.integration
def test_greek_mwt_emits_tilde_joined_mor(
    greek_morphotag_output: str,
) -> None:
    """Same shape as the Hebrew test: every Greek utterance with a
    σε+article contraction must have a ``~`` in its ``%mor`` line.

    Guards: greek mwt emits tilde joined mor for the supplied linguistic source.
    Kind: journey
    Not a type: ManagedMorphotag excludes direct-mode composition, but cannot prove subprocess success or these model-derived serialized MOR and GRA bytes. The residual observation is greek mwt emits tilde joined mor for the supplied linguistic source.
    State: The shared managed CLI fixture stages one CHAT source and reads its real output file.
    Concurrency: Each submission has private temporary paths while the server owns shared model scheduling.
    """
    mor_lines = _mor_lines(greek_morphotag_output)
    assert len(mor_lines) == len(_GREEK_MWT_SURFACES_PER_LINE), (
        f"Expected {len(_GREEK_MWT_SURFACES_PER_LINE)} %mor lines, "
        f"got {len(mor_lines)}.\nFull output:\n{greek_morphotag_output}"
    )
    for utterance_idx, (mor_line, surfaces) in enumerate(
        zip(mor_lines, _GREEK_MWT_SURFACES_PER_LINE, strict=True)
    ):
        assert "~" in mor_line, (
            f"Greek %mor line {utterance_idx + 1} is missing the "
            f"tilde-joined MWT marker ~. The surface "
            f"contraction(s) {surfaces} should have produced "
            f"tilde-joined splits.\nLine: {mor_line}\n"
            f"Full output:\n{greek_morphotag_output}"
        )


@pytest.mark.golden
@pytest.mark.integration
def test_greek_mwt_does_not_emit_unsplit_surface_as_lemma(
    greek_morphotag_output: str,
) -> None:
    """Negative assertion mirroring the Hebrew counterpart.

    Guards: greek mwt does not emit unsplit surface as lemma for the supplied linguistic source.
    Kind: journey
    Not a type: ManagedMorphotag excludes direct-mode composition, but cannot prove subprocess success or these model-derived serialized MOR and GRA bytes. The residual observation is greek mwt does not emit unsplit surface as lemma for the supplied linguistic source.
    State: The shared managed CLI fixture stages one CHAT source and reads its real output file.
    Concurrency: Each submission has private temporary paths while the server owns shared model scheduling.
    """
    mor_lines = _mor_lines(greek_morphotag_output)
    for utterance_idx, (mor_line, surfaces) in enumerate(
        zip(mor_lines, _GREEK_MWT_SURFACES_PER_LINE, strict=True)
    ):
        for surface in surfaces:
            bad_substring = f"|{surface} "
            assert bad_substring not in mor_line + " ", (
                f"Greek %mor line {utterance_idx + 1} contains "
                f"the contracted surface {surface!r} as an unsplit "
                f"lemma (substring {bad_substring!r}). MWT was "
                f"suppressed.\nLine: {mor_line}\n"
                f"Full output:\n{greek_morphotag_output}"
            )
